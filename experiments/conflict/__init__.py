"""Contradiction-aware RAG experiments (see README.md in this folder)."""
from pathlib import Path

try:  # load OPENAI_API_KEY etc. from the repo's .env, like core/config.py does
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass
