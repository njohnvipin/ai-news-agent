"""
app.py — Agent mode: fetch AI news, write a short analysis, email it.

Run with:
    python app.py
"""
import sys


def main():
    print("\n" + "=" * 50)
    print("   Valence Analytics — AI News Agent")
    print("=" * 50 + "\n")

    try:
        from config import missing_email_settings
        missing = missing_email_settings()
        if missing:
            raise EnvironmentError(
                f"Agent mode sends email, but these are missing in .env: {', '.join(missing)}"
            )

        from agents.news_agent import AINewsAgent
        AINewsAgent().run()

    except EnvironmentError as e:
        print(f"\n Configuration Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n Agent Error: {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
