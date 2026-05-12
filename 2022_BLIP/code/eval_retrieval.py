"""
Evaluate image-text retrieval on COCO Karpathy test split.
Reports R@1, R@5, R@10 for TR (text retrieval) and IR (image retrieval).
Usage: python eval_retrieval.py --config ../configs/retrieval_coco.yaml
"""
import argparse
import json
import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import BlipProcessor, BlipForImageTextRetrieval
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from dataset import CocoRetrievalDataset


@torch.no_grad()
def extract_image_features(model, dataset, device, batch_size=64, num_workers=4):
    """Extract image embeddings for all images in the dataset."""
    from torch.utils.data import DataLoader as DL

    class ImgDS(torch.utils.data.Dataset):
        def __init__(self, ds): self.ds = ds
        def __len__(self): return len(self.ds.images)
        def __getitem__(self, i): return self.ds.get_image(i)

    loader = DL(ImgDS(dataset), batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    feats = []
    for batch in tqdm(loader, desc="Extracting image features"):
        batch = batch.to(device)
        with torch.cuda.amp.autocast():
            vision_out = model.vision_model(pixel_values=batch)
            img_feat = model.vision_proj(vision_out.last_hidden_state[:, 0, :])
            img_feat = F.normalize(img_feat, dim=-1)
        feats.append(img_feat.cpu())
    return torch.cat(feats, dim=0)


@torch.no_grad()
def extract_text_features(model, dataset, device, batch_size=128, num_workers=4):
    """Extract text embeddings for all captions in the dataset."""
    from torch.utils.data import DataLoader as DL

    class TxtDS(torch.utils.data.Dataset):
        def __init__(self, ds): self.ds = ds
        def __len__(self): return len(self.ds.texts)
        def __getitem__(self, i):
            enc = self.ds.get_text(i)
            return {k: v.squeeze(0) for k, v in enc.items()}

    def collate(batch):
        return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}

    loader = DL(TxtDS(dataset), batch_size=batch_size, num_workers=num_workers,
                pin_memory=True, collate_fn=collate)
    feats = []
    for batch in tqdm(loader, desc="Extracting text features"):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.cuda.amp.autocast():
            text_out = model.text_encoder(**batch)
            txt_feat = model.text_proj(text_out.last_hidden_state[:, 0, :])
            txt_feat = F.normalize(txt_feat, dim=-1)
        feats.append(txt_feat.cpu())
    return torch.cat(feats, dim=0)


def recall_at_k(scores, gt_map, ks=(1, 5, 10)):
    """
    scores: (N_query, N_target) similarity matrix
    gt_map: list of length N_query, each entry is a list/int of correct target indices
    """
    results = {}
    n = scores.shape[0]
    for k in ks:
        topk = scores.topk(k, dim=1).indices
        hits = 0
        for i, gt in enumerate(gt_map):
            gt_set = set(gt) if isinstance(gt, list) else {gt}
            if any(j.item() in gt_set for j in topk[i]):
                hits += 1
        results[f"R@{k}"] = hits / n * 100
    return results


def evaluate(args):
    import yaml
    with open(args.config) as f:
        conf = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = cfg.resolve_model(conf["model_name"])
    print(f"Loading model: {model_name}")
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForImageTextRetrieval.from_pretrained(model_name).to(device)
    model.eval()

    dataset = CocoRetrievalDataset(
        ann_file=cfg.resolve_data(conf["test_ann"]),
        image_root=cfg.resolve_data(conf.get("image_root", "")),
        processor=processor,
    )
    print(f"Images: {len(dataset.images)}, Texts: {len(dataset.texts)}")

    img_feats = extract_image_features(model, dataset, device,
                                       batch_size=conf.get("eval_batch_size", 64))
    txt_feats = extract_text_features(model, dataset, device,
                                      batch_size=conf.get("text_batch_size", 128))

    # Similarity matrix (n_img x n_txt)
    sims = img_feats @ txt_feats.T  # (N_img, N_txt)

    # TR: for each text, find its image (text -> image retrieval)
    # IR: for each image, find its texts (image -> text retrieval)

    # Text Retrieval (TR): query=text, gallery=images
    # sims.T is (N_txt, N_img)
    tr_scores = sims.T
    tr_r = recall_at_k(tr_scores, dataset.txt2img)

    # Image Retrieval (IR): query=image, gallery=texts
    ir_r = recall_at_k(sims, dataset.img2txt)

    print("\n=== COCO Retrieval Results ===")
    print(f"Text  Retrieval  R@1={tr_r['R@1']:.2f}  R@5={tr_r['R@5']:.2f}  R@10={tr_r['R@10']:.2f}")
    print(f"Image Retrieval  R@1={ir_r['R@1']:.2f}  R@5={ir_r['R@5']:.2f}  R@10={ir_r['R@10']:.2f}")

    mean_r1 = (tr_r["R@1"] + ir_r["R@1"]) / 2
    print(f"Mean R@1: {mean_r1:.2f}")

    out_rel = conf.get("output_dir", "")
    output_dir = os.path.join(cfg.OUTPUT_DIR, out_rel) if out_rel else cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    result = {"TR": tr_r, "IR": ir_r, "mean_R@1": mean_r1}
    import json
    json.dump(result, open(os.path.join(output_dir, "retrieval_results.json"), "w"), indent=2)
    print(f"Saved to {output_dir}/retrieval_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    evaluate(args)
