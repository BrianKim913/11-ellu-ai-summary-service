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
            token=token  # 로그인 없이 토큰 전달
        ).to(self.device) # 44issue
        logger.info("Model loaded successfully")

        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=token  # 동일하게 토큰 전달
        )
        logger.info("Tokenizer loaded successfully")

    def extract_row_for_nickname(self, note: str, nickname: str) -> str:
        lines = note.strip().split("\n")
        content_lines = [line for line in lines if "|" in line]
        filtered_lines = [line for line in content_lines if nickname in line]
        header_lines = content_lines[:2]
        return "\n".join(header_lines + filtered_lines)

    def extract_priority_cell(self, raw_note: str, nickname: str) -> str:
        filtered = self.extract_row_for_nickname(raw_note, nickname)
        lines = filtered.split("\n")
        if len(lines) < 3:
            return ""
        header_cells = [c.strip() for c in lines[0].split("|")]
        data_cells = [c.strip() for c in lines[2].split("|")]
        try:
            idx = header_cells.index("오늘 할 일 및 우선순위")
        except ValueError:
            idx = 2
        return data_cells[idx] if idx < len(data_cells) else ""

    
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
        return re.sub(r"```json|```", "", text).strip()

    def summarize_and_generate_tasks(self, meeting_note: str, nickname: str, project_id: int):
        logger.info(f"Processing meeting note for nickname: {nickname}, project_id: {project_id}")
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
            # web_context = retrieve_web_context(task)  # 주석 해제 시 외부에서 정의 필요
            logger.info(f"Retrieved wiki for Project ID {project_id}")
        
            task_chat = [
                {
                    "role": "system",
                    "content": f"""
                    {wiki_context}를 바탕으로 {nickname} 사용자의 {task}를 의미 있는 작업 단위로 나눠줘.
                    각 작업은 반드시 2개 이상의 세부 작업(subtasks)을 포함해야 해. 

                    - subtasks는 절대 빈 배열([])이면 안 돼.  
                    - 출력은 **반드시** 아래 JSON 형식으로만, 다른 설명은 포함하지 마:
                    - task 항목은 가능하면 \"{task}\"를 그대로 사용해줘. 
                    - 출력은 절대 설명 없이, 아래와 같이 콤마로 구분된 한 줄 요약으로만 해줘.
                    - 세부 작업들은 간단 명료하게 써줘.

                    Wiki Context: {wiki_context}

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
            try:
                parsed = json.loads(self.clean_json_codeblock(response))
                subtasks_count = len(parsed.get("subtasks", []))
                logger.info(f"Successfully parsed {subtasks_count} subtasks for task: {parsed.get('task')}")
                parsed_results.append({
                    "keyword": parsed["task"],
                    "subtasks": parsed["subtasks"]
                })
                logger.info(f"Completed processing with {len(parsed_results)} tasks generated")
            except Exception as e:
                logger.error(f"Failed to parse response for task '{task}': {e}")
                logger.error(f"Response was: {response}")
                continue
        
        return { "message" : "keywords_created", "detail": parsed_results}
