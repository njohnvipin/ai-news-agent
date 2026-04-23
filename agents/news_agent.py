from langchain.agents import create_agent
from langchain_groq import ChatGroq
import os
from agents.tools import fetch_ai_news, send_email_tool

class AINewsAgent:

    def __init__(self):
        self.llm = ChatGroq(
            model=os.getenv("MODEL_NAME"),
            api_key=os.getenv("GROQ_API_KEY")
        )
        self.tools = [fetch_ai_news, send_email_tool]

        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt="""You are an AI news agent. You have two tools.

STEP 1: Call fetch_ai_news tool to get the news titles.

STEP 2: Write EXACTLY 5 sentences of analysis.
No company names. Focus on trends and ethics.

STEP 3: Call send_email_tool with ONLY your 5 sentence analysis as insights.
Do not include article titles or links in the insights parameter.
Just pass your 5 sentence paragraph."""
        )

    def run(self):
        print("\n Agent started\n")

        result = self.agent.invoke({
            "messages": [("human", "Execute all 3 steps now.")]
        })

        final_message = result["messages"][-1].content
        print("\nAgent finished\n")
        print(final_message)
        return result