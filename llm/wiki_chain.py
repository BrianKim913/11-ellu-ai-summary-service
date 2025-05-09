import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from vectordb.chroma_store import ChromaDBManager
import torch 
import logging
import time

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

class WikiSummarizer:
    def __init__(
        self,
        model_name: str = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B",
        embed_func=ChromaDBManager()
    ):
        logger.info("Initializing WikiSummarizer...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.token = os.getenv("HUGGINGFACE_API_KEY")  # 토큰 직접 불러오기
        if not self.token:
          logger.warning("HUGGINGFACE_API_KEY not found in environment variables!")

        logger.info(f"Loading model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=self.token  # 여기 전달
        )
        logger.info("Model loaded successfully")
        
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=self.token  # 여기도 전달
        )
        logger.info("Tokenizer loaded successfully")

        logger.info("Setting up pipeline...")
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=1024,
            temperature=0,
            device=0 if self.device == "cuda" else -1
        )
        logger.info("Pipeline created")

        self.llm = HuggingFacePipeline(pipeline=self.pipeline)
        self.prompt = PromptTemplate.from_template("요약해줘: {text}")
        self.chain = self.prompt | self.llm
        self.embed_func = embed_func.embed_and_store
        logger.info("WikiSummarizer initialization complete")

    def summarize_wiki(self, state: dict) -> dict:
        logger.info(f"Starting wiki summarization for project_id: {state.dict().get('project_id')}")
        state = state.dict()
        content = state.get("content")

        if not content:
            logger.error("'content' key missing in input state")
            raise KeyError("'content' 키가 없습니다.")

        logger.info("Generating summary...")
        start_time = time.time()
        summary = self.chain.invoke({"text": content})
        end_time = time.time()
        logger.info(f"Summary generated in {end_time - start_time:.2f} seconds")
        logger.info(f"Summary length: {len(summary)} characters")

        metadata = {"project_id": state["project_id"]}
        if state.get("updated_at"):
            metadata["updated_at"] = state["updated_at"]
    
        logger.info(f"Storing summary in vector database with metadata: {metadata}")
        self.embed_func(summary, metadata)
        logger.info("Wiki summarization completed successfully")

        return {
            "message": "wiki_saved"
        }
