from langchain_ollama import OllamaLLM
from services.news_service import fetch_ai_articles
from services.email_service import send_email
from config import MODEL_NAME


class AINewsAgent:

    def __init__(self):
        self.llm = OllamaLLM(model=MODEL_NAME, num_predict=300)

    def run(self):
        print("\nAgent started\n")

        # Step 1: Fetch
        print("Fetching news...")
        articles = fetch_ai_articles()

        for i, a in enumerate(articles, 1):
            print(f"{i}. {a['title']}")

        # Prepare context
        rich_articles = "\n\n".join([
            f"{i+1}. {a['title']}\nSummary: {a['description']}"
            for i, a in enumerate(articles)
        ])

        print("\nThinking...")

        prompt = f"""
You are a senior AI industry analyst.

Here are AI news articles with context:
{rich_articles}

Task:
- Analyze ALL articles together
- Write ONE single continuous paragraph (no line breaks)
- Use exactly 5 sentences

STRICT RULES:
- Do NOT mention any company names, product names, locations, article titles, or specific examples
- Do NOT describe individual articles
- Do NOT use phrases like "for example", "such as", or similar
- If any specific reference appears, rewrite it into a generalized statement

Focus on:
- AI adoption trends across industries
- Business and workforce transformation
- Ethical risks and responsible usage
- Long-term impact of AI

Guidelines:
- Keep the analysis abstract and high-level
- Ensure smooth flow between sentences
- Make it sound like a professional industry report
- Output must be exactly ONE paragraph only

"""

        try:
            result = self.llm.invoke(prompt)
            # Remove line breaks (force single paragraph)
            result = result.replace("\n", " ").strip()
           
        except:
            result = "AI analysis unavailable.\n\nBest regards,\nValence Analytics"

        final_output = "Daily AI Insights\n\n"

        for i, a in enumerate(articles, 1):
            final_output += f"{i}. {a['title']}\n"
            final_output += f"{a['link']}\n"
            final_output += f"Published: {a['published']}\n\n"

        final_output += "Overall Insights:\n\n"
        final_output += result + "\n\n"
        final_output += "Best regards,\nValence Analytics"

        print("\nFinal Output:\n")
        print(final_output)

        print("\nSending email...")
        send_email(final_output)

        return final_output