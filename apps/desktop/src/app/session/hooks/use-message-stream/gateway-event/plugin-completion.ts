import { noteTurnSettled } from '@/app/session/hooks/use-session-actions/settled-turns'
import { toChatMessages } from '@/lib/chat-messages'

import type { GatewayEventContext } from './types'

/** A persisted machine result is a transcript row, never an assistant turn end. */
export function handlePluginCompletionEvent(ctx: GatewayEventContext): boolean {
  if (ctx.event.type !== 'message.complete' || ctx.payload?.completion_note?.display_kind !== 'plugin_completion') {
    return false
  }

  const note = ctx.payload.completion_note

  if (
    !ctx.sessionId ||
    ctx.event.session_id !== ctx.sessionId ||
    note.role !== 'user' ||
    typeof note.row_id !== 'number' ||
    !Number.isInteger(note.row_id) ||
    note.row_id <= 0
  ) {
    return true
  }

  const [message] = toChatMessages([{ ...note, content: note.content ?? note.text }])

  if (message) {
    // A resume snapshot taken before this persisted note must not erase it.
    noteTurnSettled(ctx.sessionId, ctx.deps.activeSessionIdRef.current === ctx.sessionId)
    ctx.deps.updateSessionState(ctx.sessionId, state =>
      state.messages.some(item => item.rowId === note.row_id)
        ? state
        : { ...state, messages: [...state.messages, message] }
    )
  }

  ctx.deps.scheduleSessionsRefresh()

  return true
}
