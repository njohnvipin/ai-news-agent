"""Shared test setup: tests never need a real Groq key."""
import os

os.environ.setdefault("GROQ_API_KEY", "test-key-not-used")
