#  AI News Agent

An autonomous AI agent that fetches the latest AI news, analyses industry trends using a large language model, and emails a formatted daily digest to multiple recipients — automatically.

Built with **Python**, **LangChain**, **Groq**, and **Gmail SMTP**.

---

##  What It Does

1. **Fetches** the latest AI news from Google News RSS (last 1 hour)
2. **Analyses** trends using `llama-3.3-70b` via Groq cloud
3. **Writes** a 5-sentence professional industry analysis
4. **Emails** a formatted digest with article links + analysis to all recipients

---

##  Project Structure

```
ai-news-agent/
│
├── app.py                      # Entry point — run this to start the agent
├── config.py                   # Loads environment variables from .env
├── requirements.txt            # All Python dependencies
│
├── agents/
│   ├── news_agent.py           # The brain — LLM + agentic loop
│   └── tools.py                # Agent tools: fetch_ai_news, send_email_tool
│
├── services/
│   ├── news_service.py         # Fetches articles from Google News RSS
│   └── email_service.py        # Sends email via Gmail SMTP
│
├── utils/
│   └── text_utils.py           # Cleans HTML from RSS feed text
│
├── .env                        # Your secrets (never commit this)
└── .env.example                # Template — safe to commit
```

---

##  How It Works

```
python app.py
      │
      ▼
AINewsAgent created
      │
      ▼
Groq LLM connected (llama-3.3-70b)
      │
      ▼
Agent decides → calls fetch_ai_news tool
      │
      ▼
news_service.py hits Google News RSS
Returns 5 articles (title, link, date)
      │
      ▼
LLM writes 5-sentence analysis
      │
      ▼
Agent decides → calls send_email_tool
      │
      ▼
email_service.py sends via Gmail SMTP
      │
      ▼
 Email delivered to all recipients
```

---

##  Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/njohnvipin/ai-news-agent.git
cd ai-news-agent
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv agentenv
agentenv\Scripts\activate


# Mac / Linux
python -m venv agentenv
source agentenv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

### 4. Set Up Environment Variables

Copy the example file and fill in your values:

```bash
cp .env.example .env
```

Edit `.env`:

```dotenv
EMAIL=your_gmail@gmail.com
APP_PASSWORD=your_16_char_app_password
MODEL_NAME=llama-3.3-70b-versatile
GROQ_API_KEY=gsk_your_groq_api_key
RECIPIENTS=email1@gmail.com,email2@gmail.com
```

### 5. Run the Agent

```bash
python app.py
```

---

##  Getting Your Credentials

### Gmail App Password
1. Go to [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords)
2. Create a new app password
3. Copy the 16-character key into `.env` as `APP_PASSWORD`

>  This is NOT your Gmail login password. It is a separate app-specific key.

### Groq API Key (Free)
1. Go to [console.groq.com](https://console.groq.com)
2. Sign up free — no credit card needed
3. Go to **API Keys** → **Create New Key**
4. Copy the key (starts with `gsk_`) into `.env` as `GROQ_API_KEY`

---

##  Dependencies

```
langchain
langchain-core
langchain-community
langchain-groq
langchain-ollama
langgraph
python-dotenv
requests
feedparser
```

Install all at once:

```bash
pip install langchain langchain-core langchain-community langchain-groq langchain-ollama langgraph python-dotenv requests feedparser
```

---

##  Agent Architecture

This project uses a **single-agent, 2-tool architecture**:

```
┌─────────────────────────────────────┐
│           AINewsAgent               │
│                                     │
│  LLM: llama-3.3-70b (Groq)         │
│                                     │
│  Tools:                             │
│    🔧 fetch_ai_news()               │
│       → fetches 5 AI headlines      │
│                                     │
│    🔧 send_email_tool(insights)     │
│       → builds + sends email        │
└─────────────────────────────────────┘
```

The LLM autonomously decides:
- When to call each tool
- What content to pass to `send_email_tool`
- When the task is complete

---

##  Sample Email Output

```
Subject: Valence Analytics - Daily AI Insights

Daily AI Insights
==================================================

ARTICLES
==================================================

1. AI regulation debate intensifies in Washington
   Link      : https://news.google.com/...
   Published : Wed, 23 Apr 2026 08:00:00 GMT

2. New breakthroughs in multimodal AI models
   Link      : https://news.google.com/...
   Published : Wed, 23 Apr 2026 07:45:00 GMT

...

OVERALL INSIGHTS
==================================================

Artificial intelligence continues to reshape industries
at an unprecedented pace, with regulatory frameworks
struggling to keep up with rapid technological advancement.
The growing adoption of AI across sectors raises important
questions about workforce transformation and ethical governance.
Businesses are increasingly investing in responsible AI
deployment to balance innovation with accountability.
Long-term societal implications of autonomous systems
are driving new conversations among policymakers worldwide.
The convergence of AI capabilities and ethical oversight
will define the next phase of technological evolution.

==================================================
Best regards,
Valence Analytics
```

---

##  Security

| File | Committed to Git | Contains |
|---|---|---|
| `.env` |  Never | Real passwords and API keys |
| `.env.example` |  Yes | Placeholder values only |
| `config.py` |  Yes | Only `os.getenv()` calls |

Make sure `.env` is in your `.gitignore`:

```gitignore
.env
agentenv/
__pycache__/
*.pyc
```

---

##  Automating Daily Runs

### Windows Task Scheduler
```
Trigger : Daily at 8:00 AM
Action  : python app.py
Start in: C:\path\to\ai-news-agent
```

### Linux / Mac Cron Job
```bash
crontab -e
# Add this line:
0 8 * * * cd /path/to/ai-news-agent && python app.py
```

### GitHub Actions (Free, No Server Needed)

Create `.github/workflows/daily_news.yml`:

```yaml
name: Daily AI News Agent

on:
  schedule:
    - cron: '0 8 * * *'
  workflow_dispatch:

jobs:
  run-agent:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - name: Run Agent
        env:
          EMAIL: ${{ secrets.EMAIL }}
          APP_PASSWORD: ${{ secrets.APP_PASSWORD }}
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
          MODEL_NAME: ${{ secrets.MODEL_NAME }}
          RECIPIENTS: ${{ secrets.RECIPIENTS }}
        run: python app.py
```

Add secrets in GitHub: **Repo → Settings → Secrets and variables → Actions**

---

##  Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `Missing in .env` | `.env` file missing or incomplete | Check all 5 variables are set |
| `SMTPAuthenticationError` | Wrong App Password | Regenerate at myaccount.google.com/apppasswords |
| `tool_use_failed` | Tool payload too large | URLs are already shortened in fetch_ai_news |
| `AttributeError: bind_tools` | Using `OllamaLLM` instead of `ChatOllama` | Change to `ChatOllama` |
| Agent takes too long | Using local Ollama on CPU | Switch to Groq API |
| No email received | Check spam folder | Also verify DEBUG lines show ` Email sent` |

---

##  What to Build Next

- [ ] Add more tools — sentiment analysis, topic filtering, priority scoring
- [ ] Build a multi-agent system — separate fetch, analyse, and email agents
- [ ] Add a web dashboard — trigger agent from browser
- [ ] Replace `@tool` with MCP servers for portability
- [ ] Schedule with GitHub Actions for daily automated runs
- [ ] Add Slack or WhatsApp notification tool

---

##  Author

**Nimmy J Vipin** — built as part of Agentic AI learning journey  

---

##  License

MIT License — free to use, modify, and distribute.
