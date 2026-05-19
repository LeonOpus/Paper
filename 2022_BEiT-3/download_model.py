"""
从 GitHub Releases 下载 BEiT-3 模型权重和分词器。
同时克隆微软官方 BEiT-3 源代码（torchscale 依赖所需）。
用法：python download_model.py
"""
import os
import subprocess

MODEL_DIR = os.path.expanduser("~/Model/BEiT-3")

FILES = {
    "beit3.spm":                                "https://github.com/addf400/files/releases/download/beit3/beit3.spm",
    "beit3_base_patch16_384_coco_retrieval.pth": "https://github.com/addf400/files/releases/download/beit3/beit3_base_patch16_384_coco_retrieval.pth",
    "beit3_base_patch16_480_vqa.pth":           "https://github.com/addf400/files/releases/download/beit3/beit3_base_patch16_480_vqa.pth",
}

BEIT3_CODE_DIR = os.path.expanduser("~/Model/BEiT-3/unilm_beit3")


def wget(url: str, dest: str):
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        print(f"[skip] {os.path.basename(dest)}")
        return
    print(f"Downloading {os.path.basename(dest)} ...")
    subprocess.run(["wget", "-c", "--show-progress", "-O", dest, url], check=True)


if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)

    for fname, url in FILES.items():
        wget(url, os.path.join(MODEL_DIR, fname))

    # 克隆微软 BEiT-3 源码（基于 torchscale 的实现）
    if not os.path.isdir(BEIT3_CODE_DIR):
        print("Cloning microsoft/unilm beit3 source ...")
        subprocess.run([
            "git", "clone", "--depth=1", "--filter=blob:none", "--sparse",
            "https://github.com/microsoft/unilm.git", BEIT3_CODE_DIR,
        ], check=True)
        subprocess.run(["git", "-C", BEIT3_CODE_DIR, "sparse-checkout", "set", "beit3"],
                       check=True)
    else:
        print("[skip] beit3 source already cloned")

    print("Done.")
