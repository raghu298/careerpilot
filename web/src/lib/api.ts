import type { ChatResult, Persona, TraceEvent } from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") || "http://localhost:8000";

export async function getPersonas(): Promise<Persona[]> {
  const res = await fetch(`${API_BASE}/demo/personas`);
  const data = await res.json();
  return data.personas as Persona[];
}

export async function createSession(personaId: string): Promise<string> {
  const res = await fetch(`${API_BASE}/demo/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ persona_id: personaId }),
  });
  const data = await res.json();
  return data.session as string;
}

export async function getOutbox(session: string) {
  const res = await fetch(`${API_BASE}/demo/outbox`, { headers: { "x-session": session } });
  const data = await res.json();
  return data.items as { channel: string; destination: string; subject: string; body: string }[];
}

/**
 * POST to an SSE endpoint and dispatch each event. Resolves with the final result.
 * Uses fetch + a streaming reader so we can POST a body (EventSource is GET-only).
 */
export async function streamPost(
  path: string,
  body: unknown,
  onEvent: (e: TraceEvent) => void,
): Promise<ChatResult | null> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.body) return null;

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let result: ChatResult | null = null;

  const handleFrame = (frame: string) => {
    // A frame may have several lines; collect all "data:" lines (SSE allows multi-line).
    const data = frame
      .split("\n")
      .filter((l) => l.startsWith("data:"))
      .map((l) => l.slice(5).trim())
      .join("");
    if (!data) return;
    const payload = JSON.parse(data);
    if (payload.type === "result") result = payload as ChatResult;
    else onEvent(payload as TraceEvent);
  };

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    // Normalize CRLF → LF so the "\n\n" event delimiter matches regardless of how
    // the server (or a proxy like Render) terminates SSE lines.
    buffer += decoder.decode(value, { stream: true }).replace(/\r/g, "");
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";
    for (const frame of frames) handleFrame(frame);
  }
  // Process any trailing frame that lacked a final blank line.
  if (buffer.trim()) handleFrame(buffer);
  return result;
}
