"""
Evaluate captioning on COCO Karpathy test split.
Reports CIDEr, BLEU@4, METEOR, ROUGE-L, SPICE.
Usage: python eval_caption.py --config ../configs/caption_coco.yaml [--checkpoint path/to/ckpt]
"""
import argparse
import json
import os
import sys
import torch
from torch.utils.data import DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg

from dataset import CocoCaptionDataset

try:
    from pycocoevalcap.eval import COCOEvalCap
    from pycocotools.coco import COCO
except ImportError:
    print("WARNING: pycocoevalcap not installed. Run: pip install pycocoevalcap pycocotools")
    COCOEvalCap = None


def evaluate(args):
    import yaml
    with open(args.config) as f:
        conf = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_name = args.checkpoint if args.checkpoint else conf["model_name"]
    model_name = cfg.resolve_model(raw_name) if not os.path.isdir(raw_name) else raw_name
    print(f"Loading model from: {model_name}")
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()

    test_dataset = CocoCaptionDataset(
        ann_file=cfg.resolve_data(conf["test_ann"]),
        image_root=cfg.resolve_data(conf.get("image_root", "")),
        processor=processor,
        split="test",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=conf.get("eval_batch_size", 32),
        shuffle=False,
        num_workers=conf.get("num_workers", 4),
        pin_memory=True,
    )

    results = []
    print(f"Generating captions for {len(test_dataset)} images...")

    with torch.no_grad():
        for batch, image_ids in tqdm(test_loader):
            pixel_values = batch["pixel_values"].to(device)
            with torch.cuda.amp.autocast():
                out_ids = model.generate(
                    pixel_values=pixel_values,
                    max_length=conf.get("max_gen_length", 30),
                    num_beams=conf.get("num_beams", 3),
                )
            captions = processor.batch_decode(out_ids, skip_special_tokens=True)
            for iid, cap in zip(image_ids, captions):
                iid_val = iid.item() if hasattr(iid, "item") else iid
                results.append({"image_id": iid_val, "caption": cap.strip()})

    out_rel = conf.get("output_dir", "")
    output_dir = os.path.join(cfg.OUTPUT_DIR, out_rel) if out_rel else cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    result_file = os.path.join(output_dir, "caption_results.json")
    json.dump(results, open(result_file, "w"))
    print(f"Saved {len(results)} predictions to {result_file}")

    if COCOEvalCap is None:
        print("Install pycocoevalcap to compute metrics.")
        return

    # Build COCO-format annotation file from our json
    ann_file = cfg.resolve_data(conf["test_ann"])
    anns_raw = json.load(open(ann_file))
    # Build coco annotation dict
    coco_anns = {"images": [], "annotations": [], "type": "captions", "info": {}, "licenses": []}
    seen_imgs = {}
    ann_id = 0
    for item in anns_raw:
        iid = item.get("image_id")
        if iid is None:
            continue
        if iid not in seen_imgs:
            seen_imgs[iid] = True
            coco_anns["images"].append({"id": iid})
        cap = item["caption"]
        if isinstance(cap, list):
            for c in cap:
                coco_anns["annotations"].append({"image_id": iid, "id": ann_id, "caption": c})
                ann_id += 1
        else:
            coco_anns["annotations"].append({"image_id": iid, "id": ann_id, "caption": cap})
            ann_id += 1

    coco_ann_file = os.path.join(output_dir, "coco_gt_captions.json")
    json.dump(coco_anns, open(coco_ann_file, "w"))

    from pycocotools.coco import COCO as _COCO
    coco_gt = _COCO(coco_ann_file)
    coco_res = coco_gt.loadRes(result_file)
    coco_eval = COCOEvalCap(coco_gt, coco_res)
    coco_eval.evaluate()

    print("\n=== Captioning Results ===")
    for metric, score in coco_eval.eval.items():
        print(f"  {metric}: {score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None, help="Path to fine-tuned checkpoint")
    args = parser.parse_args()
    evaluate(args)
