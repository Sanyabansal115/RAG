import os
from dotenv import load_dotenv

def load_env(env_path: str = ".env"):
    """
    Load environment variables from a .env file (project root).
    Returns a dict with FIRST_NAME, MISTRAL_API_KEY, MISTRAL_MODEL.
    """
    load_dotenv(env_path)
    config = {
        "FIRST_NAME": os.getenv("FIRST_NAME", "Danish"),
        "MISTRAL_API_KEY": os.getenv("MISTRAL_API_KEY", ""),
        "MISTRAL_MODEL": os.getenv("MISTRAL_MODEL", "mistral-large-latest"),
    }
    if not config["MISTRAL_API_KEY"]:
        print("[WARN] MISTRAL_API_KEY is empty. Set it in the .env file.")
    return config
