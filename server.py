from fastapi import FastAPI
from pydantic import BaseModel
import logging
import chromadb
import httpx
from llm.wiki_chain import WikiSummarizer
from vectordb.chroma_db import add_document_to_chroma
from llm.meeting_chain import MeetingTaskParser
from config import CHROMA_HOST, CHROMA_PORT
from dotenv import load_dotenv
load_dotenv()

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

class MeetingInput(BaseModel):
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
    result = wiki_chain.summarize_wiki(input)

# 벡터 DB에 저장
    add_document_to_chroma(
        text=result['summary'],
        project_id=input.project_id
    )

    return {
        "message": "Wiki_saved"
    }

# BE → AI 회의록 전달
@app.post("/projects/{id}/notes")
async def receive_meeting_note(id: int, input: MeetingInput):
    result = Task_Parser.summarize_and_generate_tasks(
        meeting_note=input.meeting_note,
        nickname=input.nickname,
        project_id=id
    )

    # AI → BE 콜백 전달
    await send_result_to_backend(id, result)

    return {
        "message": "processing_complete",
        "detail": "Result sent to backend"
    }

# 콜백 함수
async def send_result_to_backend(project_id: int, result: dict):
    backend_callback_url = f"http://[BE_URL]/ai-callback/projects/{project_id}/preview"
    async with httpx.AsyncClient() as client:
        try:
            await client.post(
                backend_callback_url,
                json=result
            )
        except Exception as e:
            logging.error(f"콜백 전송 실패: {e}")