import os
from langchain.serpapi import SerpAPIWrapper
from dotenv import load_dotenv

load_dotenv()
serpapi_api_key = os.getenv("SERPAPI_API_KEY")


def get_profile_url(text: str) -> str:
    """Searches for LinkedIn profile page for a given name."""
    search = SerpAPIWrapper()
    res = search.run(f"{text}")
    return res
