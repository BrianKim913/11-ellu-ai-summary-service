import os
import json
import re
import logging
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from llm.wiki_retriever import retrieve_wiki_context
from llm.tavily_search import retrieve_web_context

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
login(token=os.getenv("HUGGINGFACE_API_KEY"))

model = AutoModelForCausalLM.from_pretrained("naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B")
tokenizer = AutoTokenizer.from_pretrained("naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B")


def generate_response(chat: list) -> str:
    inputs = tokenizer.apply_chat_template(
        chat,
        return_tensors="pt", 
        return_dict=True, 
        add_generation_prompt=True)

    outputs = model.generate(
        **inputs,
        # max_new_tokens=128,
        max_new_tokens=512,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id
    )
    # 3) 입력(prompt) 길이 이후 토큰만 잘라서 디코딩
    # 어시스턴트 응답부분만 나오도록
    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = outputs[0][prompt_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def clean_json_codeblock(text: str) -> str:
    return re.sub(r"```json|```", "", text).strip()

# 1) 요약 단계
def summarize_and_generate_tasks(meeting_note: str, nickname: str, project_id: int):
    system_prompt = {
        "role": "system",
        "content": (
            "너는 회의록에서 특정 사용자의 '오늘 할 일 및 우선순위' 항목만을 "
            "추출해주는 전문가야. "
            "출력은 오직 핵심 항목들만 **콤마(,)로 구분된 한 줄**로 작성해. "
            "**절대 설명을 붙이지 마.**"
        )
    }

    user_prompt = {
        "role": "user",
        "content": f"""
        입력된 회의록에서 '{nickname}' 사용자의 '오늘 할 일 및 우선순위' 내용을 요약해줘.  
        다른 사람 내용은 무시해도 돼.  

        **출력은 절대 설명 없이, 아래와 같이 콤마로 구분된 한 줄 요약으로만 해줘.**

        예시:
        설계 과제 4번 완료하기, 모의 면접, 카카오투어

        회의록:
        {meeting_note}
    """
    }

    summary = generate_response([system_prompt, user_prompt])

    task_candidates = summary.split(',')
