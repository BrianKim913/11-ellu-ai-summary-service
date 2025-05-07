from fastapi import FastAPI
from pydantic import BaseModel
import logging
import chromadb
# from llm.meeting_chain import summarize_and_generate_tasks
# from llm.wiki_chain import wiki_chain
# from vectordb.chroma_db import add_document_to_chroma
from config import CHROMA_HOST, CHROMA_PORT
import chromadb



app = FastAPI()

try:
    chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    print(chroma_client.heartbeat())
    
except Exception as e:
    chroma_client = None
    logging.error(f"ChromaDB 연결 실패: {e}")



class WikiInput(BaseModel):
    project_id: int
    wiki: str

class MeetingInput(BaseModel):
    project_id: int
    meeting_note: str
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

# @app.post("/ai/projects/wiki")
# def summarize_wiki(input: WikiInput):
#     result = wiki_chain.invoke({"text": input.wiki})

# # 벡터 DB에 저장
#     add_document_to_chroma(
#         text=result['content'],
#         project_id=input.project_id
#     )

#     return {
#         "message": "Wiki 업데이트 및 요약 완료"
#     }


# @app.post("/ai/notes")
# def process_meeting(input: MeetingInput):
#     result = summarize_and_generate_tasks(
#         meeting_note=input.meeting_note,
#         nickname=input.nickname,
#         project_id=input.project_id
#     )

#     return {
#         "message": "keywords_created",
#         "detail": result
#     }