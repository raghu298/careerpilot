const STARTERS: { agent: string; icon: string; question: string }[] = [
  { agent: "Knowledge", icon: "RAG", question: "Does my resume fit the Senior GenAI JD? What are my gaps?" },
  { agent: "Jobs", icon: "MCP", question: "Find senior GenAI roles in Bangalore" },
  { agent: "Analytics", icon: "SQL", question: "How many of my applications got a response, and which are stale?" },
  { agent: "Interview", icon: "PREP", question: "Prep me for the interview — what questions will they ask?" },
  { agent: "Notify", icon: "HITL", question: "Find a role I haven't applied to yet and draft a cover letter for it" },
];

export default function StarterPrompts({ onPick }: { onPick: (q: string) => void }) {
  return (
    <div className="starter-prompts">
      <div className="starter-prompts-heading">
        <p className="starter-kicker">Try one</p>
        <h2>Ask CareerPilot anything about your job hunt</h2>
        <p>One question routes to the right specialist agent(s) — and can fan out to several at once.</p>
      </div>
      <div className="starter-prompt-grid">
        {STARTERS.map((s) => (
          <button key={s.question} className="starter-prompt-card" onClick={() => onPick(s.question)}>
            <span className="starter-prompt-icon">{s.icon}</span>
            <span className="starter-prompt-copy">
              <span className="starter-prompt-agent">{s.agent}</span>
              <span className="starter-prompt-question">{s.question}</span>
            </span>
            <span className="starter-prompt-arrow">→</span>
          </button>
        ))}
      </div>
    </div>
  );
}
