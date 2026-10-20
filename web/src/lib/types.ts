export type Persona = { id: string; name: string; sql_role: string };

export type TraceEvent = {
  type: string;
  agent?: string;
  tool?: string;
  agents?: string[];
  reasoning?: string;
  kind?: string;
  message?: string;
};

export type Reply = { agent: string; content: string; metadata: Record<string, unknown> };

export type Pending = {
  agent: string;
  kind: string;
  destination: string;
  subject: string;
  body: string;
  preview: string;
};

export type ChatResult = {
  type: "result";
  routing_reasoning: string;
  replies: Reply[];
  pending: Pending | null;
};

export type Message =
  | { role: "user"; content: string }
  | { role: "assistant"; replies: Reply[]; routing_reasoning: string };
