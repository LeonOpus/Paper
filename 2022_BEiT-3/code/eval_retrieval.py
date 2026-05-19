"""
在 COCO Karpathy 测试集上评估 BEiT-3 图文检索性能。
使用微软官方基于 torchscale 的实现。
报告文本检索（TR）和图像检索（IR）的 R@1/R@5/R@10 指标。
用法：python eval_retrieval.py --config ../configs/retrieval_coco.yaml
"""
import argparse
import json
import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import XLMRobertaTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg

BEIT3_SRC = os.path.expanduser("~/Model/BEiT-3/unilm_beit3/beit3")
MODEL_DIR  = os.path.expanduser("~/Model/BEiT-3")


def load_beit3_retrieval(ckpt_path: str, device):
    sys.path.insert(0, BEIT3_SRC)
    import modeling_finetune
    import utils as beit3_utils

    model = modeling_finetune.beit3_base_patch16_384_retrieval()
    beit3_utils.load_model_and_may_interpolate(ckpt_path, model, "model|module", "")
    return model.to(device).eval()


def build_image_transform(img_size: int = 384):
    mean = (0.5, 0.5, 0.5)
    std  = (0.5, 0.5, 0.5)
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


class CocoImageDataset(Dataset):
    def __init__(self, images, image_root, transform):
        self.images = images
        self.image_root = image_root
        self.transform = transform

    def __len__(self): return len(self.images)

    def __getitem__(self, i):
        from PIL import Image
        path = os.path.join(self.image_root, self.images[i]["image"])
        return self.transform(Image.open(path).convert("RGB"))


def recall_at_k(scores, gt_map, ks=(1, 5, 10)):
    results = {}
    for k in ks:
        topk = scores.topk(k, dim=1).indices
        hits = sum(
            any(j.item() in (set(gt_map[i]) if isinstance(gt_map[i], list) else {gt_map[i]})
                for j in topk[i])
            for i in range(scores.size(0))
        )
        results[f"R@{k}"] = hits / scores.size(0) * 100
    return results


@torch.no_grad()
def extract_features(model, tokenizer, dataset_info, image_root, device,
                     batch_size, num_workers):
    transform = build_image_transform(384)
    images    = dataset_info["images"]
    texts     = dataset_info["texts"]

    # 提取图像特征
    img_ds = CocoImageDataset(images, image_root, transform)
    img_loader = DataLoader(img_ds, batch_size=batch_size,
                            num_workers=num_workers, pin_memory=True)
    img_feats = []
    for batch in tqdm(img_loader, desc="Image features"):
        batch = batch.to(device)
        with torch.cuda.amp.autocast():
            vision_cls, _ = model(image=batch, only_infer=True)
            img_feats.append(F.normalize(vision_cls, dim=-1).cpu())
    img_feats = torch.cat(img_feats)

    # 提取文本特征
    txt_feats = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Text features"):
        batch_texts = texts[i:i + batch_size]
        enc = tokenizer(batch_texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=64).to(device)
        with torch.cuda.amp.autocast():
            _, language_cls = model(
                text_description=enc["input_ids"],
                padding_mask=(enc["attention_mask"] == 0),
                only_infer=True,
            )
            txt_feats.append(F.normalize(language_cls, dim=-1).cpu())
    txt_feats = torch.cat(txt_feats)

    return img_feats, txt_feats


def evaluate(args):
    import yaml
    conf = yaml.safe_load(open(args.config))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = os.path.join(MODEL_DIR, "beit3_base_patch16_384_coco_retrieval.pth")
    print(f"Loading model: {ckpt}")
    model = load_beit3_retrieval(ckpt, device)

    spm_path = os.path.join(MODEL_DIR, "beit3.spm")
    tokenizer = XLMRobertaTokenizer(spm_path)

    # 构建数据集信息
    anns = json.load(open(cfg.resolve_data(conf["test_ann"])))
    images, texts = [], []
    img2txt, txt2img = {}, {}
    seen = {}
    for item in anns:
        iid = item["image_id"]
        if iid not in seen:
            seen[iid] = len(images)
            images.append({"image_id": iid, "image": item["image"]})
        img_idx = seen[iid]
        caps = item["caption"] if isinstance(item["caption"], list) else [item["caption"]]
        for cap in caps:
            cap_idx = len(texts)
            texts.append(cap)
            img2txt.setdefault(img_idx, []).append(cap_idx)
            txt2img[cap_idx] = img_idx

    dataset_info = {"images": images, "texts": texts}
    image_root   = cfg.resolve_data(conf.get("image_root", ""))
    print(f"Images: {len(images)}, Texts: {len(texts)}")

    img_feats, txt_feats = extract_features(
        model, tokenizer, dataset_info, image_root, device,
        conf.get("eval_batch_size", 32), conf.get("num_workers", 4),
    )

    sims  = img_feats @ txt_feats.T
    # TR = 文本检索 = 以图像为查询检索标注（I2T），对所有图像取平均
    # IR = 图像检索 = 以文本为查询检索图像（T2I），对所有文本取平均
    tr_r  = recall_at_k(sims,   img2txt)
    ir_r  = recall_at_k(sims.T, txt2img)

    print("\n=== BEiT-3 COCO Retrieval Results ===")
    print(f"Text  Retrieval  R@1={tr_r['R@1']:.2f}  R@5={tr_r['R@5']:.2f}  R@10={tr_r['R@10']:.2f}")
    print(f"Image Retrieval  R@1={ir_r['R@1']:.2f}  R@5={ir_r['R@5']:.2f}  R@10={ir_r['R@10']:.2f}")
    print(f"(Paper base: TR R@1=84.8 / IR R@1=67.2)")

    out_dir = os.path.join(cfg.OUTPUT_DIR, conf.get("output_dir", ""))
    os.makedirs(out_dir, exist_ok=True)
    json.dump({"TR": tr_r, "IR": ir_r},
              open(os.path.join(out_dir, "retrieval_results.json"), "w"), indent=2)
    print(f"Saved to {out_dir}/retrieval_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    evaluate(parser.parse_args())
