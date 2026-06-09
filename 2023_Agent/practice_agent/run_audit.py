"""
对三份测试公告分别跑合规审核 Agent
用法: conda run -n 2020_RAG python practice_agent/run_audit.py
"""

import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from agent import ComplianceAuditAgent

LLM_PATH = "/home/leon/Model/ModelScope/models/Qwen/Qwen2.5-7B-Instruct"
ANNOUNCEMENTS_DIR = Path(__file__).parent / "announcements"
OUTPUT_DIR = Path("/home/leon/Output/2023_Agent")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CASES = [
    ("p0_violation.txt",  "P0违规：关联交易未认定"),
    ("p1_violation.txt",  "P1违规：自然人信息不完整"),
    ("compliant.txt",     "合规公告"),
]


def main():
    print("加载 Qwen2.5-7B-Instruct...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_PATH, torch_dtype=torch.float16, device_map="cuda"
    )
    agent = ComplianceAuditAgent(model, tokenizer)

    results = []
    for filename, label in CASES:
        announcement = (ANNOUNCEMENTS_DIR / filename).read_text(encoding="utf-8")
        report = agent.audit(announcement, label=label)
        results.append((label, report))

        # 保存单份报告
        out_file = OUTPUT_DIR / filename.replace(".txt", "_report.txt")
        out_file.write_text(f"【{label}】\n\n{report}\n", encoding="utf-8")
        print(f"\n  已保存: {out_file}")

    # 打印汇总
    print(f"\n\n{'#'*65}")
    print("审核汇总")
    print(f"{'#'*65}")
    for label, report in results:
        print(f"\n{'='*65}")
        print(f"【{label}】")
        print("-" * 65)
        print(report)


if __name__ == "__main__":
    main()
