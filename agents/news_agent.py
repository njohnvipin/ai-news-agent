"""
news_agent.py — LangGraph ReAct agent that fetches AI news and sends an email.

Bug fix: the original code imported `create_agent` which does not exist in
LangChain.  The correct function is `create_react_agent` from langgraph.
"""
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from agents.tools import fetch_ai_news, send_email_tool
from config import MODEL_NAME, GROQ_API_KEY

SYSTEM_PROMPT = """You are an AI news analyst agent. Follow these steps ONCE and stop.

STEP 1: Call fetch_ai_news ONCE to get today's AI news headlines.

STEP 2: Write EXACTLY 5 sentences of analysis:
        - Focus on industry trends, business impact, ethics, long-term implications
        - Professional tone, no company/product names, no specific article examples

STEP 3: Call send_email_tool EXACTLY ONCE with your analysis.

After send_email_tool returns, you are DONE. Do not call any tool again. Stop immediately."""

class AINewsAgent:

    def __init__(self):
        self.llm = ChatGroq(
            model=MODEL_NAME,
            api_key=GROQ_API_KEY,
            temperature=0.3,
        )
        self.tools = [fetch_ai_news, send_email_tool]
        # create_react_agent is the correct LangGraph entry-point
        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=SYSTEM_PROMPT,
        )

    def run(self):
        print(" Agent thinking...\n")

        result = self.agent.invoke(
            {"messages": [("human", "Execute all 3 steps now. Stop after sending the email.")]},
            config={"recursion_limit": 10}  # hard stop after 10 steps
        )

        final_message = result["messages"][-1].content
        if final_message:
            print(f"\n Agent: {final_message}")

        return result
