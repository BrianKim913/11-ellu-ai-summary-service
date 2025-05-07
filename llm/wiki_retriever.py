from vectordb.chroma_store import collection
from vectordb.embed_model import CustomEmbeddingFunction

def retrieve_wiki_context(keyword: str, project_id: int, k: int = 3) -> str:
    embedding_fn = CustomEmbeddingFunction()
    query_embedding = embedding_fn.embed_query(keyword)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where={"project_id": project_id}
    )

    documents = results.get("documents", [[]])[0]
    return "\n".join(documents)
