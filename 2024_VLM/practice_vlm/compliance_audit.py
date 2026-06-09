"""
VLM 合规审核：直接输入公告图片，输出结构化审核报告
对比纯文本 RAG 和 VLM 的能力差异
用法: conda run -n 2024_VLM python practice_vlm/compliance_audit.py
"""

import torch
from pathlib import Path
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_PATH = "/home/leon/Model/ModelScope/models/Qwen/Qwen2.5-VL-7B-Instruct"
IMAGES_DIR = Path(__file__).parent / "images"

AUDIT_PROMPT = """你是一名专业的证券合规审核员。请仔细阅读这份上市公司公告图片，完成以下审核：

1. 【交易概述核查】列出公告中已披露和缺失的交易概述要素
2. 【交易对方核查】检查交易对方信息是否完整（区分自然人/法人要求不同）
3. 【关联交易核查】判断是否存在关联交易未认定的情况
4. 【违规定级】对每项发现的问题给出 P0/P1/P2 级别，并说明理由
5. 【审核结论】总结本次审核结果

请逐项输出，格式清晰。"""


def audit_announcement(model, processor, image_path: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": AUDIT_PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=600, temperature=0.2, do_sample=True)

    generated = [output_ids[i][len(inputs.input_ids[i]):] for i in range(len(output_ids))]
    return processor.batch_decode(generated, skip_special_tokens=True)[0]


print("加载 Qwen2.5-VL-7B-Instruct...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="cuda",
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
print("模型就绪\n")

for image_file in sorted(IMAGES_DIR.glob("*.png")):
    print("=" * 65)
    print(f"审核文件: {image_file.name}")
    print("=" * 65)
    result = audit_announcement(model, processor, str(image_file))
    print(result)
    print()

print("审核完成")
