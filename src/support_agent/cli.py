"""Interactive terminal chat for manual exploratory testing."""

from __future__ import annotations

import logging

from support_agent.agent import SupportAgent


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    agent = SupportAgent.from_settings()
    print(f"ShopCo Support Agent [{agent.backend.name}]. Commands: /reset, /exit")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input == "/exit":
            print("Goodbye!")
            break
        if user_input == "/reset":
            agent.reset()
            print("Memory cleared.")
            continue

        response = agent.chat(user_input)
        print(f"Agent: {response.answer}")
        print(f"  [context passages: {len(response.retrieval_context)} | turn: {response.turn_index}]")


if __name__ == "__main__":
    main()