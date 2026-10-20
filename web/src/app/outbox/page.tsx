"use client";

import { useEffect, useState } from "react";
import { getOutbox } from "@/lib/api";

type Item = { channel: string; destination: string; subject: string; body: string };

export default function OutboxPage() {
  const [items, setItems] = useState<Item[]>([]);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    const tok = localStorage.getItem("cp_session");
    if (!tok) {
      window.location.href = "/";
      return;
    }
    getOutbox(tok)
      .then(setItems)
      .finally(() => setLoaded(true));
  }, []);

  return (
    <div className="center">
      <div className="card" style={{ maxWidth: 640 }}>
        <h1>Demo Outbox</h1>
        <p>
          Confirmed outreach drafts are captured here instead of being sent. <a href="/chat">← Back to chat</a>
        </p>
        {loaded && items.length === 0 && <p style={{ color: "var(--muted)" }}>Nothing captured yet.</p>}
        {items.map((it, i) => (
          <div className="outbox-item" key={i}>
            <div className="to">
              [{it.channel}] to {it.destination}
            </div>
            <div style={{ fontWeight: 600, margin: "4px 0" }}>{it.subject}</div>
            <div style={{ whiteSpace: "pre-wrap" }}>{it.body}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
