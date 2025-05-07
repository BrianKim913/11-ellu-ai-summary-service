# config.py
import os
from dotenv import load_dotenv

load_dotenv()  # .env 파일 로드

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))