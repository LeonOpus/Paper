"""
Qwen2.5-VL 基础推理：看图回答问题
用法: conda run -n 2024_VLM python practice_vlm/inference.py
"""

import torch
from pathlib import Path
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_PATH = "/home/leon/Model/ModelScope/models/Qwen/Qwen2.5-VL-7B-Instruct"
IMAGES_DIR = Path(__file__).parent / "images"

# 每张图片对应的问题
TASKS = [
    {
        "image": "p0_violation.png",
        "questions": [
            "这份公告中，交易对方是谁？法定代表人是谁？",
            "公告中的控股股东是谁？交易对方法定代表人和控股股东是否为同一人？",
            "根据你的判断，这份公告是否存在关联交易未认定的问题？",
        ],
    },
    {
        "image": "p1_violation.png",
        "questions": [
            "交易对方是自然人还是法人？叫什么名字？",
            "公告中披露了交易对方的哪些个人信息？缺少哪些必要信息？",
        ],
    },
    {
        "image": "compliant.png",
        "questions": [
            "这份公告的交易概述包含哪些要素？",
            "独立董事发表了什么意见？",
        ],
    },
]


def ask(model, processor, image_path: str, question: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
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
        output_ids = model.generate(**inputs, max_new_tokens=300, temperature=0.3, do_sample=True)

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

for task in TASKS:
    image_path = str(IMAGES_DIR / task["image"])
    print("=" * 65)
    print(f"图片: {task['image']}")
    print("=" * 65)
    for q in task["questions"]:
        print(f"\n问: {q}")
        print(f"答: {ask(model, processor, image_path, q)}")

print("\n推理完成")
