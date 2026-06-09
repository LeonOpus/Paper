"""
ReAct 合规审核 Agent
循环：Thought → Action → Observation → Thought → ... → Final Answer
"""

import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tools import TOOL_REGISTRY, TOOLS_DESCRIPTION

LLM_PATH = "/home/leon/Model/ModelScope/models/Qwen/Qwen2.5-7B-Instruct"
MAX_STEPS = 12

SYSTEM_PROMPT = f"""你是一名专业的证券合规审核智能体，负责审核上市公司信息披露公告是否符合深交所规则要求。

{TOOLS_DESCRIPTION}

审核时请依次检查：
1. 交易概述章节是否包含全部必要要素（交易类型/对方/标的/价格/金额/支付方式/是否重大资产重组/是否关联交易/审批进展）
2. 交易对方信息是否完整（自然人和法人的披露要求不同）
3. 公告中对关联交易的认定是否正确（核对交易对方与上市公司控股股东/实控人的关系）
4. 其他重要披露项
"""


def _parse_response(text: str):
    """
    解析模型输出，返回 ("final", report) 或 ("action", (name, args)) 或 ("error", text)
    """
    # 优先检查 Final Answer
    final_match = re.search(r'Final Answer\s*[:：](.*)', text, re.DOTALL | re.IGNORECASE)
    if final_match:
        return "final", final_match.group(1).strip()

    # 检查 Action + Action Input
    action_match = re.search(r'Action\s*[:：]\s*(\w+)', text)
    input_match = re.search(r'Action Input\s*[:：]\s*(\{.*?\})', text, re.DOTALL)
    if action_match and input_match:
        name = action_match.group(1).strip()
        try:
            args = json.loads(input_match.group(1))
        except json.JSONDecodeError:
            args = {}
        return "action", (name, args)

    return "error", text


def _generate(model, tokenizer, messages: list) -> str:
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=600,
            temperature=0.1,
            do_sample=True,
            stop_strings=["Observation:"],
            tokenizer=tokenizer,
        )
    return tokenizer.decode(
        output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()


class ComplianceAuditAgent:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def audit(self, announcement: str, label: str = "") -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"请审核以下上市公司公告，识别所有信息披露违规项：\n\n{announcement}",
            },
        ]

        header = f"【{label}】" if label else ""
        print(f"\n{'='*65}")
        print(f"开始审核 {header}")
        print(f"{'='*65}")

        for step in range(MAX_STEPS):
            response = _generate(self.model, self.tokenizer, messages)
            kind, parsed = _parse_response(response)

            # 打印模型的 Thought 部分（去掉已知的 Observation: 前缀）
            thought_text = response.split("Action:")[0].strip() if "Action:" in response else response
            if thought_text:
                print(f"\n[Step {step+1}] {thought_text[:200]}{'...' if len(thought_text)>200 else ''}")

            if kind == "final":
                print(f"\n{'='*65}")
                print("审核完成")
                return parsed

            elif kind == "action":
                name, args = parsed
                print(f"  → 调用工具: {name}({args})")
                if name in TOOL_REGISTRY:
                    observation = TOOL_REGISTRY[name](**args)
                else:
                    observation = f"未知工具: {name}"
                short_obs = observation[:200] + "..." if len(observation) > 200 else observation
                print(f"  ← 检索结果: {short_obs}")

                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"Observation: {observation}"})

            else:
                # 格式不对，给提示继续
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": "请按格式继续输出 Action 或 Final Answer。",
                })

        return "（达到最大步骤数，审核未完成）"
