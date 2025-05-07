from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
import os
load_dotenv()

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # pipeline 추가 필요
from langchain_community.llms import HuggingFacePipeline

# HyperCLOVAX 모델 초기화 추가
model = AutoModelForCausalLM.from_pretrained("naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B")
tokenizer = AutoTokenizer.from_pretrained("naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B")

# LangChain 파이프라인 생성
llm = HuggingFacePipeline(
    pipeline=pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=1024,
        temperature=0
    )
)

prompt = PromptTemplate.from_template("요약해줘: {text}")
wiki_chain = prompt | llm

# wiki 요약 및 임베딩 저장
from vectordb.chroma_store import embed_and_store

def summarize_wiki(state: dict):
    content = state.get("content")
    if not content:
        raise KeyError("'content' 키가 없습니다.")

    summary = wiki_chain.invoke({"text": content})

    metadata = {"project_id": state["project_id"]}
    if state.get("updated_at") is not None:
        metadata["updated_at"] = state["updated_at"]
    
    embed_and_store(summary, metadata)

    return {"summary": summary, "message": "wiki_saved", "project_id": state["project_id"], "updated_at": state.get("updated_at", None)}