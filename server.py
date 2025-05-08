from fastapi import FastAPI
from pydantic import BaseModel
import logging
import chromadb
# from llm.meeting_chain import summarize_and_generate_tasks
from llm.wiki_chain import WikiSummarizer
from vectordb.chroma_db import add_document_to_chroma
from llm.meeting_chain import MeetingTaskParser
from config import CHROMA_HOST, CHROMA_PORT
from huggingface_hub import login
import os
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
        "message": "Wiki 업데이트 및 요약 완료"
    }


@app.post("/ai/notes")
def process_meeting(input: MeetingInput):
    result = Task_Parser.summarize_and_generate_tasks(
        meeting_note=input.content,
        nickname=input.nickname,
        project_id=input.project_id
    )

    return {
        "message": "keywords_created",
        "detail": result
    }