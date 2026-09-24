import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """
    Application configuration settings
    Centralizes all environment variables for secure access throughout the system.
    """
    Project_name: str = "AI Document Analyzer"
    Version: str = "1.0.0"
    GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")

settings = Settings()

    