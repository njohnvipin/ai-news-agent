import os
from dotenv import load_dotenv

load_dotenv()

EMAIL        = os.getenv("EMAIL")
APP_PASSWORD = os.getenv("APP_PASSWORD")
MODEL_NAME   = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
RECIPIENTS   = [r.strip() for r in os.getenv("RECIPIENTS", "").split(",") if r.strip()]

# catch missing values early
missing = [k for k, v in {
    "EMAIL":        EMAIL,
    "APP_PASSWORD": APP_PASSWORD,
    "GROQ_API_KEY": GROQ_API_KEY,
    "RECIPIENTS":   RECIPIENTS[0] if RECIPIENTS else None,
}.items() if not v]

if missing:
    raise EnvironmentError(
        f"Missing required values in .env: {', '.join(missing)}"
    )