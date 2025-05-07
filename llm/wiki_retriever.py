# llm/wiki_retriever.py
from vectordb.chroma_db import get_vectorstore

def retrieve_wiki_context(keyword: str, project_id: int, k: int = 3) -> str:
    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(
        query=keyword,
        k=k,
        filter={"project_id": project_id}  # 프로젝트별로 필터링
    )
    return "\n".join([doc.page_content for doc in docs])
