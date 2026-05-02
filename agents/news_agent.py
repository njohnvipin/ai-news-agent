from langchain.agents import create_agent
from langchain_groq import ChatGroq
from agents.tools import fetch_ai_news, send_email_tool
from config import MODEL_NAME, GROQ_API_KEY

SYSTEM_PROMPT = """You are an AI news analyst agent. You have exactly two tools.

STEP 1: Call fetch_ai_news to get today's AI news headlines.

STEP 2: Analyse the headlines and write EXACTLY 5 sentences.
        Rules for analysis:
        - No company names or product names
        - No specific examples from the articles
        - Focus on: industry trends, business impact, ethics, long-term implications
        - Write as one continuous paragraph
        - Professional industry report tone

STEP 3: Call send_email_tool ONCE with your 5 sentence analysis as the insights argument.
        - Pass ONLY the analysis paragraph
        - Do NOT include article titles or links
        - Do NOT call send_email_tool more than once
        - Do NOT add any closing remarks after calling the tool"""


class AINewsAgent:

    def __init__(self):
        self.llm = ChatGroq(
            model=MODEL_NAME,
            api_key=GROQ_API_KEY,
            temperature=0.3,       # lower = more consistent output
        )
        self.tools = [fetch_ai_news, send_email_tool]
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=SYSTEM_PROMPT
        )

    def run(self):
        print(" Agent thinking...\n")

        result = self.agent.invoke({
            "messages": [("human", "Execute all 3 steps now.")]
        })

        final_message = result["messages"][-1].content
        if final_message:
            print(f"\n Agent: {final_message}")

        return result