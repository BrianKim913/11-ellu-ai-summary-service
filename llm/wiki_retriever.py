from vectordb.chroma_store import collection
from vectordb.embed_model import CustomEmbeddingFunction

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

embedding_fn = CustomEmbeddingFunction()

def retrieve_wiki_context(task: str, project_id: int, k: int = 3) -> str:
    query_embedding = embedding_fn.embed_query(task)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where={"project_id": project_id}
    )

    # documents = results.get("documents", [[]])[0]
    # return "\n".join(documents)

    documents = results.get("documents", [[]])
    # if not documents or not documents[0]:
    #     return ""

    # return "\n".join(documents[0])

    if not documents or not documents[0]:
        logger.warning(f"[wiki_retriever] No documents found for task: {task} (project {project_id})")
        return task  # fallback