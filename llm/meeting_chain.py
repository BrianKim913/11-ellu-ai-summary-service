import os
import json
import re
import logging
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from llm.wiki_retriever import retrieve_wiki_context
# from llm.tavily_search import retrieve_web_context
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MeetingTaskParser:
    def __init__(self):
        logger.info("Initializing MeetingTaskParser...")
        load_dotenv()
        token = os.getenv("HUGGINGFACE_API_KEY")
        if not token:
            logger.warning("HUGGINGFACE_API_KEY not found in environment variables!")

        model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
        logger.info(f"Using model: {model_name}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        logger.info("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=token
        ).to(self.device)
        logger.info("Model loaded successfully")

        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=token
        )
        logger.info("Tokenizer loaded successfully")


    def generate_response(self, chat: list) -> str:
        inputs = self.tokenizer.apply_chat_template(
            chat,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id
        )
        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = outputs[0][prompt_len:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    def clean_json_codeblock(self, text: str) -> str:
        cleaned = re.sub(r"```json|```", "", text).strip()
        if cleaned.count("{") > cleaned.count("}"):
            cleaned += "}"
        return cleaned

    def clean_keyword_output(self, text: str, nickname: str) -> str:
        pattern = rf"\b{re.escape(nickname)}(의| )?"
        return re.sub(pattern, "", text).strip()

    def summarize_and_generate_tasks(self, meeting_note: str, nickname: str, project_id: int, position: str):
        logger.info(f"Processing meeting note for nickname: {nickname}, project_id: {project_id}")
        
        system_prompt = {
            "role": "system",
            "content": (
                f"너는 회의록에서 '{nickname}' 사용자의 오늘 할 일과 우선순위를 정확히 추출하는 전문가야. "
                "입력된 회의록은 표 형식이 아닌 자연어 문장으로 구성되어 있어. "
                "출력은 절대로 설명 없이, 콤마(,)로 구분된 핵심 업무 목록 한 줄로만 작성해. "
                "사용자의 이름이나 '누구의 할 일' 같은 표현은 생략해."
            )
        }

        user_prompt = {
            "role": "user",
            "content": f"""
회의록:
{meeting_note}

'{nickname}' 사용자의 오늘 할 일과 우선순위를 요약해줘.  
다른 사람 내용은 무시해도 돼. 만약 '{nickname}' 이 할 일이 무엇인지 모르면 내용 기반으로 오늘 할 일과 우선순위를 요약해줘  

**출력은 오직 콤마(,)로 구분된 한 줄 요약으로만 해줘.**
**사용자 이름이나 '누구의 할 일' 같은 표현은 절대 포함하지 마.**
"""
        }

        summary = self.generate_response([system_prompt, user_prompt])
        task_candidates = summary.split(',')
        logger.info(f"Extracted {len(task_candidates)} task candidates")

        parsed_results = []
        for task in task_candidates:
            task = task.strip()
            if not task:
                continue

            logger.info(f"Processing task: {task}")
            wiki_context = retrieve_wiki_context(task, project_id)
            logger.info(f"Retrieved wiki for Project ID {project_id}")

            langchain_definition = """
LangChain은 LLM을 기반으로 LLM, Chain 등의 모듈을 연결하여
사용자 질의에 대해 복합적인 처리를 수행할 수 있도록 돕는 프레임워크입니다.
주로 체인 구성, 프롬프트 설정, 응답 처리, Tool 연동 등의 작업이 필요합니다.
"""

            task_chat = [
                {
                    "role": "system",
                    "content": f"""
Wiki Context: {wiki_context}

{wiki_context}를 바탕으로 {nickname} 사용자의 {task}를 의미 있는 작업 단위로 나눠줘.
각 작업은 반드시 2개 ~ 4개의 세부 작업(subtasks)을 포함해야 해. 

- 이 사용자의 역할은 '{position}'이야. {position} 역할에서 일반적으로 수행하는 구현 작업 범위를 벗어나지 마.
참고로, LangChain은 다음과 같은 시스템입니다: {langchain_definition}

- subtasks는 절대 빈 배열([])이면 안 돼.  
- "task" 항목은 반드시 입력으로 주어진 "{task}"를 그대로 사용해야 해.
- 절대로 task 이름을 변경하거나 유사한 표현으로 바꾸면 안 돼.  
- 출력은 절대 설명 없이, 아래와 같이 콤마로 구분된 한 줄 요약으로만 해줘.
- subtasks는 **핵심 동사 + 명사 위주로만 간단하게 작성하고, 목적이나 설명은 빼.**
- 출력은 반드시 하나의 작업에 대해서만 아래 JSON 형식으로 응답해야 하며, 복수 작업은 포함하지 마.

출력 예시:
{{
"task": "{task}",
"subtasks": [
    "세부 작업 1",
    "세부 작업 2",
    "세부 작업 3"
]
}}
"""
                }
            ]

            response = self.generate_response(task_chat)
            logger.info(f"Raw model response: {response}")

            try:
                parsed = json.loads(self.clean_json_codeblock(response))

                keyword = self.clean_keyword_output(parsed["task"], nickname)
                subtasks = [self.clean_keyword_output(st, nickname) for st in parsed.get("subtasks", [])]

                subtasks_count = len(subtasks)
                logger.info(f"Successfully parsed {subtasks_count} subtasks for task: {keyword}")

                parsed_results.append({
                    "keyword": keyword,
                    "subtasks": subtasks
                })

            except Exception as e:
                logger.error(f"Failed to parse response for task '{task}': {e}")
                logger.error(f"Response was: {response}")
                continue

        return {"message": "keywords_created", "detail": parsed_results}
