from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
import os

load_dotenv()
# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

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
# wiki_chain = chain.invoke({"text": "이건 테스트입니다."})

from vectordb.chroma_db import add_document_to_chroma

# 2. 요약 후 Chroma에 저장
def summarize_wiki(text: str, project_id: int):
    result = wiki_chain.invoke({"text": text})
    summary = result.content  

    # 벡터 저장
    add_document_to_chroma(summary, project_id)
    return {"project_id": project_id, "summary": summary}