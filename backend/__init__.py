from pathlib import Path

try:
    from dotenv import load_dotenv
    # Load environment from backend/.env to keep backend self-contained
    load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)
except Exception:
    # If dotenv is unavailable, continue without failing
    pass


