import chromadb
from .embed_model import CustomEmbeddingFunction
from config import CHROMA_HOST, CHROMA_PORT

class ChromaDBManager:
    def __init__(self, collection_name="wiki_summaries", host=CHROMA_HOST, port=CHROMA_PORT):
        """Initialize ChromaDB connection and collection."""
        self.client = chromadb.HttpClient(host=host, port=port)
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.embedding_function = CustomEmbeddingFunction()
    
    def embed_and_store(self, summary: str, metadata: dict):
        """Embed and store document in ChromaDB, replacing existing documents with same project_id."""
        doc_id = f"{metadata['project_id']}_{metadata.get('updated_at', 'unknown')}"
        
        # Delete existing documents with the same project_id
        self.collection.delete(where={"project_id": metadata["project_id"]})

        # Generate embedding and add new document
        embedding = self.embedding_function([summary])[0]  # Get first item since it's a single document
        self.collection.add(
            ids=[doc_id], 
            documents=[summary], 
            embeddings=[embedding], 
            metadatas=[metadata]
        )
        
        print(f"DB 갱신 완료: {doc_id}, embedding: {embedding[:5]}...")
        return doc_id
    
    def search(self, query_text, n_results=5, where_filter=None):
        """Search for similar documents."""
        query_embedding = self.embedding_function([query_text])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter
        )
        return results
    
    def get_by_project_id(self, project_id):
        """Retrieve documents by project ID."""
        return self.collection.get(where={"project_id": project_id})
    
    def delete_by_project_id(self, project_id):
        """Delete documents by project ID."""
        return self.collection.delete(where={"project_id": project_id})


default_db_manager = ChromaDBManager()
