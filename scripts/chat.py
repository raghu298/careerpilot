"""CLI chat client — the fastest way to try every agent from one terminal.

    python scripts/chat.py            # default persona (riya)
    python scripts/chat.py arjun      # senior-switcher persona

Runs fully offline with the deterministic stand-ins unless OPENAI_API_KEY is set.
When an agent stages a draft, a 3-option menu (Send / Edit / Cancel) drives the
human-in-the-loop confirmation — deliberately not free-text matching.
"""

from __future__ import annotations

import asyncio
import sys

from careerpilot.access_policy import DEFAULT_PERSONA
from careerpilot.bootstrap import build_session_orchestrator, build_shared_resources


def _print_event(e: dict) -> None:
    t = e.get("type")
    if t == "routing_decided":
        print(f"  · routed → {', '.join(e.get('agents', []))}")
    elif t == "tool_called":
        print(f"  · {e.get('agent')}: {e.get('tool')}()")


async def main() -> None:
    persona = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PERSONA
    resources = await build_shared_resources()
    orch = build_session_orchestrator(resources, persona_id=persona)
    mode = "offline" if resources.offline else "OpenAI"
    print(f"CareerPilot CLI — persona={persona}, mode={mode}. Type 'quit' to exit.\n")

    while True:
        try:
            user = input("you › ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user.lower() in {"quit", "exit"}:
            break
        if not user:
            continue

        out = await orch.handle(user, on_event=_print_event)
        for name, reply in out.results.items():
            print(f"\n[{name}] {reply.content}\n")

        while out.pending is not None:
            print(f"--- staged {out.pending.kind} ---\n{out.pending.preview}\n")
            choice = input("Send it / Edit / Cancel? ").strip().lower()
            if choice.startswith("s"):
                out = await orch.resume_pending({"action": "confirm"})
            elif choice.startswith("e"):
                edit = input("How should I revise it? ").strip()
                out = await orch.resume_pending({"action": "revise", "edit": edit})
            else:
                out = await orch.resume_pending({"action": "cancel"})
            for name, reply in out.results.items():
                if name == "notify":
                    print(f"\n[{name}] {reply.content}\n")


if __name__ == "__main__":
    asyncio.run(main())
