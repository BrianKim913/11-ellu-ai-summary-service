from langchain_community.vectorstores import Chroma
from .embedding import KoSimCSEEmbedding
import os

# import chromadb
# chroma_client = chromadb.HttpClient(host='localhost', port=8000)


# vectorstore 인스턴스를 반환하는 함수
def get_vectorstore():
    embedding_fn = KoSimCSEEmbedding()
    
    # Chroma DB 디렉토리 생성
    persist_directory = "./chroma"
    os.makedirs(persist_directory, exist_ok=True)
    
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_fn,
        collection_name="wiki_docs"  # 컬렉션 이름 지정
    )
    return vectorstore

def add_document_to_chroma(text: str, project_id: int):
    vectorstore = get_vectorstore()
    vectorstore.add_texts(
        texts=[text],        metadatas=[{"project_id": project_id}]
    )
    vectorstore.persist() 
    print(f"저장 완료: {text[:30]}... (project_id={project_id})")