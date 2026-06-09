"""
完整合规审核流水线：VLM读图 + Agent推理 + RAG检索 → 审核报告

一个模型（Qwen2.5-VL-7B）同时承担视觉理解和推理两个角色
Agent 可以自主决定：何时看图、何时查规则、何时输出结论

用法: conda run -n 2024_VLM python practice_vlm/pipeline.py
      conda run -n 2024_VLM python practice_vlm/pipeline.py images/p0_violation.png
"""

import json
import re
import sys
import torch
import chromadb
from pathlib import Path
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from FlagEmbedding import BGEM3FlagModel

MODEL_PATH = "/home/leon/Model/ModelScope/models/Qwen/Qwen2.5-VL-7B-Instruct"
BGE_MODEL_PATH = "/home/leon/Model/ModelScope/models/BAAI/bge-m3"
CHROMA_DIR = "/home/leon/Output/2020_RAG/chroma_db"
IMAGES_DIR = Path(__file__).parent / "images"
MAX_STEPS = 12

SYSTEM_PROMPT = """你是一名专业的证券合规审核智能体，负责审核上市公司信息披露公告是否符合深交所规则。

你有两个工具：

1. ask_image(question: str)
   对公告图片提问，获取图片中的具体信息
   例：ask_image({"question": "交易对方是自然人还是法人？"})

2. search_rules(query: str)
   在合规规则数据库中检索相关规定条款
   例：search_rules({"query": "自然人交易对方必须披露哪些信息"})

审核必须覆盖以下五个维度，缺一不可：

① 交易概述完整性
   用 ask_image 确认：交易类型、交易对方、标的、价格、金额、支付方式、是否重大资产重组、是否关联交易、审批进展

② 交易对方信息完整性
   自然人：姓名、国籍、证件号码（后六位）、住所、职业及社会关系、关联关系、犯罪记录、资金来源
   法人：名称、注册地、法定代表人、注册资本及实缴、主营业务、关联关系、财务数据

③ 隐性关联关系核查（重点，最常见的P0来源）
   必须用 ask_image 分别提取：
   - 上市公司的控股股东、实际控制人是谁
   - 交易对方的法定代表人/实际控制人是谁
   然后判断：两者是否为同一人或存在控制关系
   若是，则本次交易构成关联交易，公告未认定即为P0违规

④ 审批程序核查
   判断是否应提交股东大会审议（构成重大资产重组或重大关联交易时必须）

⑤ 违规定级汇总
   对每项发现分别给出 P0/P1/P2 级别和理由

输出 Final Answer 时必须按五个维度逐项输出，不得遗漏。

输出格式（严格遵守）：
Thought: <当前分析和下一步计划>
Action: ask_image
Action Input: {"question": "..."}

或：
Thought: <分析>
Action: search_rules
Action Input: {"query": "..."}

完成时：
Thought: <总结所有发现>
Final Answer: <完整审核报告，包含违规项和P0/P1/P2定级>"""


class ComplianceAuditPipeline:
    def __init__(self):
        print("加载 Qwen2.5-VL-7B-Instruct...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float16, device_map="cuda"
        )
        self.processor = AutoProcessor.from_pretrained(MODEL_PATH)

        print("加载 BGE-M3 embedding 模型...")
        self.embed_model = BGEM3FlagModel(BGE_MODEL_PATH, use_fp16=True)

        print("连接 ChromaDB...")
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = client.get_collection("compliance_rules")
        print("流水线就绪\n")

    # ── 工具 1：VLM 看图 ──────────────────────────────────────────────
    def ask_image(self, image_path: str, question: str) -> str:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
            ],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=300, temperature=0.2, do_sample=True)
        generated = out[0][len(inputs.input_ids[0]):]
        return self.processor.decode(generated, skip_special_tokens=True)

    # ── 工具 2：RAG 检索规则 ──────────────────────────────────────────
    def search_rules(self, query: str) -> str:
        emb = self.embed_model.encode([query])["dense_vecs"]
        results = self.collection.query(query_embeddings=emb.tolist(), n_results=3)
        return "\n\n---\n\n".join(results["documents"][0])

    # ── Agent 推理（纯文本循环，不带图片）────────────────────────────
    def _generate_agent_step(self, messages: list) -> str:
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], padding=True, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=400, temperature=0.1, do_sample=True,
                stop_strings=["Observation:", "\nObservation"],
                tokenizer=self.processor.tokenizer,
            )
        generated = out[0][len(inputs.input_ids[0]):]
        return self.processor.decode(generated, skip_special_tokens=True)

    def _parse(self, text: str):
        if m := re.search(r'Final Answer:\s*(.*)', text, re.DOTALL):
            return "final", m.group(1).strip()
        action = re.search(r'Action:\s*(\w+)', text)
        arg = re.search(r'Action Input:\s*(\{.*?\})', text, re.DOTALL)
        if action and arg:
            try:
                return "action", (action.group(1).strip(), json.loads(arg.group(1)))
            except json.JSONDecodeError:
                pass
        return "unknown", text

    # ── 主入口 ────────────────────────────────────────────────────────
    def run(self, image_path: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"请对以下公告图片进行完整的合规审核：{image_path}"},
        ]

        recent_actions: list[tuple] = []   # 用于检测循环

        for step in range(MAX_STEPS):
            print(f"\n── Step {step + 1} ──")

            # 步骤过半仍未结束，提醒模型汇总输出
            if step == MAX_STEPS // 2:
                messages.append({
                    "role": "user",
                    "content": "你已完成一半步骤。请根据目前收集到的所有信息，"
                               "直接输出 Final Answer（五个维度逐项列出，不要再调用工具）。"
                })

            response = self._generate_agent_step(messages)
            print(response.strip())

            kind, parsed = self._parse(response)

            if kind == "final":
                return parsed

            elif kind == "action":
                action_name, args = parsed
                action_key = (action_name, json.dumps(args, ensure_ascii=False))

                # 检测到重复动作，打断循环
                if action_key in recent_actions:
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": "你刚才重复了已经问过的问题。"
                                   "请不要再调用工具，直接根据已有信息输出 Final Answer。"
                    })
                    continue

                recent_actions.append(action_key)
                if len(recent_actions) > 4:
                    recent_actions.pop(0)

                if action_name == "ask_image":
                    obs = self.ask_image(image_path, args.get("question", ""))
                elif action_name == "search_rules":
                    obs = self.search_rules(args.get("query", ""))
                else:
                    obs = f"未知工具: {action_name}"

                preview = obs[:200] + "..." if len(obs) > 200 else obs
                print(f"\nObservation: {preview}")

                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"Observation: {obs}"})

            else:
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": "请继续按格式输出 Action 或 Final Answer。"})

        return "审核未完成（达到最大步骤数）"


def main():
    pipeline = ComplianceAuditPipeline()

    # 命令行指定单张图片，否则跑全部
    if len(sys.argv) > 1:
        targets = [Path(sys.argv[1])]
    else:
        targets = sorted(IMAGES_DIR.glob("*.png"))

    for image_file in targets:
        print(f"\n{'='*65}")
        print(f"审核文件: {image_file.name}")
        print(f"{'='*65}")
        report = pipeline.run(str(image_file))
        print(f"\n{'='*65}")
        print("最终审核报告")
        print(f"{'='*65}")
        print(report)


if __name__ == "__main__":
    main()
