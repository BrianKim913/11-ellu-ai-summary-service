import chromadb
from .embed_model import CustomEmbeddingFunction

from config import CHROMA_HOST, CHROMA_PORT
import chromadb

chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

# chroma_client = chromadb.HttpClient(host='localhost', port=8000)

COLLECTION_NAME = "wiki_summaries"
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

# ------------------ 임베딩 및 저장 함수 ------------------
def embed_and_store(summary: str, metadata: dict):
    doc_id = f"{metadata['project_id']}_{metadata.get('updated_at', 'unknown')}"
    
    # # 동일 project_id 기존 문서 삭제
    # collection.delete(where={"project_id": metadata["project_id"]})

    # # 새 문서 추가
    # embedding = CustomEmbeddingFunction()([summary])[0]   # 단건이므로 첫 번째만
    # collection.add(ids=[doc_id], documents=[summary], embeddings=[embedding], metadatas=[metadata])
    
    # print(f"✅ DB 갱신 완료: {doc_id}, embedding: {embedding[:5]}...")

    # 저장 전에 존재 여부 체크
    search_result = collection.get(where={"project_id": metadata["project_id"]})
    already_exists = len(search_result['ids']) > 0

    # 기존 project_id 문서 삭제
    if already_exists:
        collection.delete(where={"project_id": metadata["project_id"]})

    # 새 문서 추가
    embedding = CustomEmbeddingFunction()([summary])[0]
    collection.add(ids=[doc_id], documents=[summary], embeddings=[embedding], metadatas=[metadata])
    
    print(f"✅ DB 갱신 완료: {doc_id}, embedding: {embedding[:5]}...")

    # 저장 상태 반환
    return "wiki_update" if already_exists else "wiki_saved"
