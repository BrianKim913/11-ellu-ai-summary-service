from langchain_community.vectorstores import Chroma
from .embed_model import CustomEmbeddingFunction  # KoSimCSEEmbedding 대신 CustomEmbeddingFunction 사용
import os

# vectorstore 인스턴스를 반환하는 함수
def get_vectorstore():
    embedding_fn = CustomEmbeddingFunction()  # 여기서 변경
    
    # Chroma DB a디렉토리 생성
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
        texts=[text],
        metadatas=[{"project_id": project_id}]
    )
    vectorstore.persist() 
    print(f"저장 완료: {text[:30]}... (project_id={project_id})")