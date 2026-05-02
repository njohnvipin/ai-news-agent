import sys
from agents.news_agent import AINewsAgent


def main():
    print("\n" + "="*50)
    print("   Valence Analytics — AI News Agent")
    print("="*50 + "\n")

    try:
        agent = AINewsAgent()
        agent.run()
    except EnvironmentError as e:
        print(f"\n Configuration Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n Agent Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()