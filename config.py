import os
from dotenv import load_dotenv

load_dotenv()

EMAIL        = os.getenv("EMAIL")
APP_PASSWORD = os.getenv("APP_PASSWORD")
MODEL_NAME   = os.getenv("MODEL_NAME", "llama3.2:3b")
RECIPIENTS   = os.getenv("RECIPIENTS", "").split(",")

# catch missing values early
missing = [k for k, v in {
    "EMAIL": EMAIL,
    "APP_PASSWORD": APP_PASSWORD,
    "RECIPIENTS": RECIPIENTS[0] if RECIPIENTS else None
}.items() if not v]

if missing:
    raise EnvironmentError(
        f"❌ Missing in .env: {', '.join(missing)}"
    )
