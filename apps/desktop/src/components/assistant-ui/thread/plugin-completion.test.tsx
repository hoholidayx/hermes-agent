import { cleanup, render, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { handlePluginCompletionEvent } from '@/app/session/hooks/use-message-stream/gateway-event/plugin-completion'
import type { GatewayEventContext } from '@/app/session/hooks/use-message-stream/gateway-event/types'
import { liveTurnSettledDuringResume } from '@/app/session/hooks/use-session-actions/resume-reconcile'
import type { ClientSessionState } from '@/app/types'
import { toChatMessages } from '@/lib/chat-messages'
import { toRuntimeMessage } from '@/lib/chat-runtime'
import { $connection } from '@/store/session'
import type { SessionMessage } from '@/types/hermes'

import { stubThreadEnvironment, ThreadRuntime } from '../test-utils'

import { Thread } from '.'

const imagePath = '/synthetic/plugin-results/geometric.png'
const imageData = 'data:image/png;base64,c3ludGhldGlj'
const caption = 'Geometric image ready'

const note: SessionMessage = {
  role: 'user',
  content: `${caption}\nMEDIA:${imagePath}`,
  display_kind: 'plugin_completion',
  display_metadata: { display_text: caption },
  row_id: 42,
  timestamp: 2
}

const initial: SessionMessage[] = [{ role: 'user', content: 'Make a geometric image.', row_id: 40, timestamp: 1 }]
const api = vi.fn(async (_request: { path: string; profile?: string }) => ({ dataUrl: imageData }))
let previousDesktop: typeof window.hermesDesktop

beforeEach(() => {
  stubThreadEnvironment()
  api.mockClear()
  previousDesktop = window.hermesDesktop
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { api } })
  $connection.set({ mode: 'remote', profile: 'plugin-test' } as never)
})

afterEach(() => {
  cleanup()
  $connection.set(null)
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: previousDesktop })
  vi.unstubAllGlobals()
})

it('paints a settled plugin note immediately and restores the same image and caption from history', async () => {
  let state = { messages: toChatMessages(initial), busy: false, awaitingResponse: false } as ClientSessionState
  const completeAssistantMessage = vi.fn()

  const ctx = {
    event: { type: 'message.complete', session_id: 'origin-runtime' },
    payload: { text: '', completion_note: { ...note, text: note.content, content: undefined } },
    sessionId: 'origin-runtime',
    deps: {
      activeSessionIdRef: { current: 'origin-runtime' },
      completeAssistantMessage,
      scheduleSessionsRefresh: vi.fn(),
      updateSessionState: vi.fn((sessionId: string, update: (state: ClientSessionState) => ClientSessionState) => {
        expect(sessionId).toBe('origin-runtime')
        state = update(state)

        return state
      })
    }
  } as unknown as GatewayEventContext

  const view = render(
    <ThreadRuntime messages={state.messages.map(toRuntimeMessage)}>
      <Thread />
    </ThreadRuntime>
  )

  expect(handlePluginCompletionEvent(ctx)).toBe(true)
  expect(handlePluginCompletionEvent(ctx)).toBe(true)
  expect(state.messages).toHaveLength(2)
  expect(state.messages[1].role).toBe('system')
  expect(state.messages[1].asyncResultKind).toBeUndefined()
  expect(completeAssistantMessage).not.toHaveBeenCalled()
  expect(state.busy).toBe(false)
  expect(liveTurnSettledDuringResume({ session_id: 'origin-runtime' }, Date.now() / 1000 - 1, true)).toBe(true)
  view.rerender(
    <ThreadRuntime messages={state.messages.map(toRuntimeMessage)}>
      <Thread />
    </ThreadRuntime>
  )

  await waitFor(() => expect(view.container.querySelector('img')?.getAttribute('src')).toBe(imageData))
  expect(view.container.querySelector('[data-slot="aui_plugin-completion"]')?.textContent).toContain(caption)
  expect(view.container.querySelector('[data-role="assistant"]')).toBeNull()
  expect(view.container.querySelectorAll('[data-role="user"]')).toHaveLength(1)
  expect(api).toHaveBeenCalledWith({
    path: `/api/fs/read-data-url?path=${encodeURIComponent(imagePath)}`,
    profile: 'plugin-test'
  })
  const liveId = state.messages[1].id
  view.unmount()

  const restored = toChatMessages([...initial, { ...note, display_metadata: JSON.stringify(note.display_metadata) }])
  expect(restored[1].id).toBe(liveId)

  const history = render(
    <ThreadRuntime messages={restored.map(toRuntimeMessage)}>
      <Thread />
    </ThreadRuntime>
  )

  await waitFor(() => expect(history.container.querySelector('img')?.getAttribute('src')).toBe(imageData))
  expect(history.container.querySelector('[data-slot="aui_plugin-completion"]')?.textContent).toContain(caption)
})

it('keeps untyped user and system MEDIA text out of the plugin media renderer', () => {
  const messages = toChatMessages([
    { role: 'user', content: `literal MEDIA:${imagePath}`, timestamp: 1 },
    { role: 'system', content: `literal MEDIA:${imagePath}`, timestamp: 2 }
  ]).map(toRuntimeMessage)

  const { container } = render(
    <ThreadRuntime messages={messages}>
      <Thread />
    </ThreadRuntime>
  )

  expect(container.querySelector('[data-slot="aui_plugin-completion"]')).toBeNull()
  expect(container.querySelector('img')).toBeNull()
  expect(api).not.toHaveBeenCalled()
})

it('updates the originating session while another conversation is foreground', () => {
  const foreground = { messages: toChatMessages(initial) } as ClientSessionState
  let origin = { messages: [] } as unknown as ClientSessionState
  const activeSessionIdRef = { current: 'foreground-runtime' }

  const ctx = {
    event: { type: 'message.complete', session_id: 'origin-runtime' },
    payload: { completion_note: note },
    sessionId: 'origin-runtime',
    deps: {
      activeSessionIdRef,
      scheduleSessionsRefresh: vi.fn(),
      updateSessionState: vi.fn((sessionId: string, update: (state: ClientSessionState) => ClientSessionState) => {
        expect(sessionId).toBe('origin-runtime')
        origin = update(origin)

        return origin
      })
    }
  } as unknown as GatewayEventContext

  expect(handlePluginCompletionEvent(ctx)).toBe(true)
  expect(origin.messages[0].rowId).toBe(note.row_id)
  expect(foreground.messages).toHaveLength(1)
  expect(activeSessionIdRef.current).toBe('foreground-runtime')
  expect(liveTurnSettledDuringResume({ session_id: 'origin-runtime' }, Date.now() / 1000 - 1, true)).toBe(false)
})

it('refuses an unscoped completion instead of adopting the foreground conversation', () => {
  const updateSessionState = vi.fn()
  const ctx = {
    event: { type: 'message.complete' },
    payload: { completion_note: note },
    sessionId: 'foreground-runtime',
    deps: { updateSessionState }
  } as unknown as GatewayEventContext

  expect(handlePluginCompletionEvent(ctx)).toBe(true)
  expect(updateSessionState).not.toHaveBeenCalled()
})

it('shows only the approved artifact when a plain caption contains escaped Markdown, HTML, and MEDIA text', async () => {
  const escapedCaption =
    '&#33;&#91;private&#93;&#40;&#47;synthetic&#47;unapproved&#46;png&#41; ' +
    '&#60;img src&#61;&#34;&#47;synthetic&#47;html&#46;png&#34;&#62; ' +
    '&#33;&#91;remote&#93;&#40;https&#58;&#47;&#47;example&#46;invalid&#47;unapproved&#46;png&#41;' +
    '\nMEDIA&#58;&#47;synthetic&#47;injected&#46;png'

  const messages = toChatMessages([{ ...note, content: `${escapedCaption}\n\nMEDIA:${imagePath}` }]).map(
    toRuntimeMessage
  )

  const { container } = render(
    <ThreadRuntime messages={messages}>
      <Thread />
    </ThreadRuntime>
  )

  await waitFor(() => expect(container.querySelector('img')?.getAttribute('src')).toBe(imageData))
  expect(container.querySelectorAll('img')).toHaveLength(1)
  expect(container.textContent).toContain('![private](/synthetic/unapproved.png)')
  expect(container.textContent).toContain('<img src="/synthetic/html.png">')

  for (const [request] of api.mock.calls) {
    expect(request.path).toBe(`/api/fs/read-data-url?path=${encodeURIComponent(imagePath)}`)
  }
})
