import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100

import config
import open_clip


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# A compact set of prompt templates inspired by CLIP zero-shot evaluation.
CLIP_PROMPT_TEMPLATES = [
    "a photo of a {}",
    "a blurry photo of a {}",
    "a black and white photo of a {}",
    "a low contrast photo of a {}",
    "a high contrast photo of a {}",
    "a bad photo of a {}",
    "a good photo of a {}",
    "a photo of a small {}",
    "a photo of a big {}",
    "a photo of the {}",
    "a blurry photo of the {}",
    "a black and white photo of the {}",
    "a low contrast photo of the {}",
    "a high contrast photo of the {}",
    "a bad photo of the {}",
    "a good photo of the {}",
    "a photo of the small {}",
    "a photo of the big {}",
]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce CLIP zero-shot on CIFAR")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--model-name", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-mode", type=str, default="clip_ensemble", choices=["single", "clip_ensemble"])
    parser.add_argument("--prompt-template", type=str, default="a photo of a {}")
    return parser.parse_args()


def load_dataset_and_classes(dataset_name: str):
    if dataset_name == "cifar10":
        dataset = CIFAR10(root=config.CIFAR10_DIR, train=False, download=True)
        class_names = CIFAR10_CLASSES
        dataset_root = config.CIFAR10_DIR
    else:
        dataset = CIFAR100(root=config.CIFAR100_DIR, train=False, download=True)
        class_to_idx = dataset.class_to_idx
        class_names = [name for name, idx in sorted(class_to_idx.items(), key=lambda item: item[1])]
        dataset_root = config.CIFAR100_DIR
    return dataset, class_names, dataset_root


def build_text_features(model, tokenizer, class_names, templates, device):
    template_features = []
    with torch.no_grad():
        for template in templates:
            prompts = [template.format(name) for name in class_names]
            tokens = tokenizer(prompts).to(device)
            text_features = model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            template_features.append(text_features)

    text_features = torch.stack(template_features, dim=0).mean(dim=0)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features


def evaluate(model, preprocess, text_features, dataset, args, device):
    def transform_item(item):
        image, label = item
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        return preprocess(image), label

    class WrappedDataset(torch.utils.data.Dataset):
        def __init__(self, base, max_samples):
            self.base = base
            if max_samples and max_samples > 0:
                self.length = min(len(base), max_samples)
            else:
                self.length = len(base)

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            return transform_item(self.base[idx])

    wrapped = WrappedDataset(dataset, args.max_samples)
    loader = DataLoader(
        wrapped,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    total = 0
    top1_correct = 0
    top5_correct = 0

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = 100.0 * image_features @ text_features.T

            top1_preds = logits.argmax(dim=-1)
            top1_correct += (top1_preds == labels).sum().item()

            k = min(5, logits.shape[1])
            topk_preds = logits.topk(k=k, dim=-1).indices
            top5_correct += (topk_preds == labels.unsqueeze(1)).any(dim=1).sum().item()

            total += labels.numel()

    top1_acc = top1_correct / total if total else 0.0
    top5_acc = top5_correct / total if total else 0.0

    return {
        "num_samples": total,
        "top1_correct": top1_correct,
        "top1_accuracy": top1_acc,
        "top5_correct": top5_correct,
        "top5_accuracy": top5_acc,
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name,
        pretrained=args.pretrained,
        cache_dir=config.HF_HUB_CACHE,
    )
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer(args.model_name)

    dataset, class_names, dataset_root = load_dataset_and_classes(args.dataset)

    if args.prompt_mode == "single":
        templates = [args.prompt_template]
    else:
        templates = CLIP_PROMPT_TEMPLATES

    text_features = build_text_features(
        model=model,
        tokenizer=tokenizer,
        class_names=class_names,
        templates=templates,
        device=device,
    )

    metrics = evaluate(
        model=model,
        preprocess=preprocess,
        text_features=text_features,
        dataset=dataset,
        args=args,
        device=device,
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "run_id": run_id,
        "paper": "2021_CLIP",
        "device": device,
        "dataset": args.dataset,
        "model_name": args.model_name,
        "pretrained": args.pretrained,
        "prompt_mode": args.prompt_mode,
        "prompt_templates": templates,
        "num_classes": len(class_names),
        "batch_size": args.batch_size,
        "max_samples": args.max_samples,
        "paths": {
            "paper_data": config.PAPER_DATA,
            "hf_home": config.HF_HOME,
            "hf_hub_cache": config.HF_HUB_CACHE,
            "dataset_root": dataset_root,
            "output_dir": config.OUTPUT_DIR,
        },
        "metrics": metrics,
    }

    result_name = f"clip_zeroshot_{args.dataset}_{args.prompt_mode}_{run_id}.json"
    result_path = os.path.join(config.OUTPUT_DIR, result_name)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"Saved result: {result_path}")


if __name__ == "__main__":
    main()
