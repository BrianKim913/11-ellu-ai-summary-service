import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from vectordb.chroma_store import ChromaDBManager
import torch 

class WikiSummarizer:
    def __init__(
        self,
        model_name: str = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B",
        embed_func=ChromaDBManager()
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.token = os.getenv("HUGGINGFACE_API_KEY")  # ✅ 토큰 직접 불러오기

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=self.token  # ✅ 여기 전달
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=self.token  # ✅ 여기도 전달
        )
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=1024,
            temperature=0,
            device=0 if self.device == "cuda" else -1
        )
        self.llm = HuggingFacePipeline(pipeline=self.pipeline)
        self.prompt = PromptTemplate.from_template("요약해줘: {text}")
        self.chain = self.prompt | self.llm
        self.embed_func = embed_func.embed_and_store

    def summarize_wiki(self, state: dict) -> dict:
        state = state.dict()
        content = state.get("content")
        if not content:
            raise KeyError("'content' 키가 없습니다.")

        summary = self.chain.invoke({"text": content})

        metadata = {"project_id": state["project_id"]}
        if state.get("updated_at"):
            metadata["updated_at"] = state["updated_at"]

        self.embed_func(summary, metadata)

        return {
            "summary": summary,
            "message": "wiki_saved"
        }
