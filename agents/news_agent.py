from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from agents.tools import fetch_ai_news, send_email_tool
from config import MODEL_NAME


class AINewsAgent:

    def __init__(self):
        self.llm = ChatOllama(model=MODEL_NAME, keep_alive=-1)
        self.tools = [fetch_ai_news, send_email_tool]

        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            checkpointer=None,

            system_prompt="""You are an AI news agent. You have two tools.

STEP 1: Call fetch_ai_news tool to get the news.

STEP 2: Build an email body with TWO sections:

SECTION 1 — Article List:
List every article exactly as received with:
- Title
- Link
- Published date

SECTION 2 — Overall Insights:
Write exactly 5 sentences of analysis.
No company names. No specific examples.
Focus on trends, ethics, business impact.

STEP 3: Call send_email_tool with the full email body containing both sections.
Always call send_email_tool. No exceptions.
Do not ask questions. Do not wait for confirmation.
Just execute all 3 steps immediately."""
        )

    def run(self):
        print("\n Agent started\n")

     
        result = self.agent.invoke(
        {"messages": [("human", "Execute all 3 steps now.")]},
        config={"callbacks": [], "verbose": True} 
    )


        final_message = result["messages"][-1].content
        print("\nAgent finished\n")
        print(final_message)
        return result