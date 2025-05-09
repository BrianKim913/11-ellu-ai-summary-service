from fastapi import FastAPI
from pydantic import BaseModel
import logging
import chromadb
import httpx
from llm.wiki_chain import WikiSummarizer
from llm.meeting_chain import MeetingTaskParser
from config import CHROMA_HOST, CHROMA_PORT
from dotenv import load_dotenv
load_dotenv()
import logging
import os

BE_URL = os.getenv("BE_URL")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

wiki_chain = WikiSummarizer()
Task_Parser=MeetingTaskParser()

app = FastAPI()

try:
    chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    print(chroma_client.heartbeat())
    
except Exception as e:
    chroma_client = None
    logging.error(f"ChromaDB 연결 실패: {e}")

class WikiInput(BaseModel):
    project_id: int
    content: str
    updated_at : str

class MeetingNote(BaseModel):
    project_id: int
    content: str
    position: str
    nickname: str
    
@app.get("/")
def read_root():
    if chroma_client:
        try:
            status = chroma_client.heartbeat()
            return {"status": "ok", "chroma": status}
        except Exception as e:
            return {"status": "partial", "error": str(e)}
    else:
        return {"status": "fail", "message": "ChromaDB not connected"}

@app.post("/ai/wiki")
def summarize_wiki(input: WikiInput):
    wiki_chain.summarize_wiki(input)

    return {
        "message": "Wiki_saved"
    }

# BE → AI 회의록 전달
@app.post("/projects/{id}/notes")
async def receive_meeting_note(id: int, input: MeetingNote):

    try:
        result = Task_Parser.summarize_and_generate_tasks(
            project_id=input.project_id,
            meeting_note=input.content,
            nickname=input.nickname
        )

        # AI → BE 콜백 전달
        await send_result_to_backend(id, result)

        return {
            "message": "processing_complete",
            "detail": "Result sent to backend"
        }
    
    except Exception as e:
        logging.error(f"Error processing meeting note: {str(e)}")
        return {
            "message": "processing_failed",
            "detail": str(e)
        }

# 콜백 함수
async def send_result_to_backend(project_id: int, result: dict):
    backend_callback_url = f"{BE_URL}/ai-callback/projects/{project_id}/preview"
    async with httpx.AsyncClient() as client:
        try:
            await client.post(
                backend_callback_url,
                json=result
            )
        except Exception as e:
            logging.error(f"콜백 전송 실패: {e}")