"""
news_agent.py — LangGraph ReAct agent that fetches AI news and emails a digest.

Note: this uses langgraph.prebuilt.create_react_agent, which works on both
LangGraph 0.x and 1.x. LangChain 1.0 introduced langchain.agents.create_agent
as its successor; migrate to that when you move to LangChain 1.x.
"""
import logging

from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from agents.tools import fetch_ai_news, send_email_tool
from config import GROQ_API_KEY, MODEL_NAME

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an AI news analyst agent. Follow these steps ONCE and stop.

STEP 1: Call fetch_ai_news ONCE to get today's AI news headlines.

STEP 2: Write EXACTLY 5 sentences of analysis:
        - Focus on industry trends, business impact, ethics, long-term implications
        - Professional tone, no company/product names, no specific article examples

STEP 3: Call send_email_tool EXACTLY ONCE with your analysis.

After send_email_tool returns, you are DONE, whether it succeeded or failed.
Do not call any tool again. Reply with one short line saying whether the email was sent."""

RECURSION_LIMIT = 10  # hard stop so the agent can never loop forever


class AINewsAgent:

    def __init__(self):
        self.llm = ChatGroq(model=MODEL_NAME, api_key=GROQ_API_KEY, temperature=0.3)
        self.agent = create_react_agent(
            model=self.llm,
            tools=[fetch_ai_news, send_email_tool],
            prompt=SYSTEM_PROMPT,
        )

    def run(self) -> dict:
        logger.info("Agent running with model %s", MODEL_NAME)

        result = self.agent.invoke(
            {"messages": [("human", "Execute all 3 steps now. Stop after sending the email.")]},
            config={"recursion_limit": RECURSION_LIMIT},
        )

        final_message = result["messages"][-1].content
        if final_message:
            print(f"\n Agent: {final_message}")
        return result
