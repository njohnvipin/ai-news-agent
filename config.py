import os
from dotenv import load_dotenv

load_dotenv()

EMAIL = os.getenv("EMAIL")
APP_PASSWORD = os.getenv("APP_PASSWORD")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.1")
RECIPIENTS = os.getenv("RECIPIENTS").split(",")