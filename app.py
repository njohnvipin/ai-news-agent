"""
app.py — Agent mode: research AI news, draft a briefing, and email it
ONLY after a human approves it.

Run with:
    python app.py             research, review, then send / revise / cancel
    python app.py --dry-run   research and show the draft; never sends
"""
import argparse
import sys

MAX_REVISIONS = 3


def main():
    parser = argparse.ArgumentParser(description="AI news digest agent with human approval")
    parser.add_argument("--dry-run", action="store_true", help="show the draft but never send it")
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("   Valence Analytics — AI News Agent")
    print("=" * 50 + "\n")

    try:
        from config import missing_email_settings
        if not args.dry_run:
            missing = missing_email_settings()
            if missing:
                raise EnvironmentError(
                    f"Sending email needs these in .env: {', '.join(missing)} "
                    "(or run with --dry-run to only see the draft)"
                )

        from agents.memory import save_digest
        from agents.news_agent import AINewsAgent
        from services.email_service import send_email

        agent = AINewsAgent()
        print("Agent steps (Reason → Act → Observe):\n")
        draft = agent.draft()
        revisions = 0

        while True:
            if not draft:
                print("\n The agent did not produce a draft. Nothing was sent.")
                sys.exit(1)

            print("\n" + "-" * 50 + "\n DRAFT FOR REVIEW\n" + "-" * 50)
            print(draft["body"])
            print("-" * 50)

            if args.dry_run:
                print("\n Dry run: nothing was sent.")
                return

            choice = input("\n[s]end, [r]evise or [c]ancel? ").strip().lower()

            if choice == "s":
                if send_email(draft["body"]):
                    save_digest(draft["insights"], draft["headlines"])
                    print("\n Sent, and saved to the agent's long-term memory.")
                else:
                    print("\n Sending failed — see the log above. Nothing was saved.")
                    sys.exit(1)
                return

            if choice == "r":
                if revisions >= MAX_REVISIONS:
                    print(f" Revision limit ({MAX_REVISIONS}) reached. Send or cancel.")
                    continue
                feedback = input("What should change? ").strip()
                if not feedback:
                    continue
                revisions += 1
                print(f"\n Revising (the agent remembers {agent.memory_size()} earlier messages)...\n")
                revised = agent.revise(feedback)
                if revised is None:
                    print("\n The agent didn't save a new version; showing the previous draft.")
                else:
                    draft = revised
                continue

            print("\n Cancelled. Nothing was sent.")
            return

    except EnvironmentError as e:
        print(f"\n Configuration Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n Cancelled. Nothing was sent.")
        sys.exit(1)
    except Exception as e:
        print(f"\n Agent Error: {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
