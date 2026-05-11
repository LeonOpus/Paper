# VLMO Reproduction

Paper: [VLMo: Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts](https://arxiv.org/abs/2111.02358)  
Venue: NeurIPS 2022

## Directory layout

```
2022_VLMO/
├── config.py                  # path config (auto-detects PAPER_NAME)
├── env.sh                     # shell environment variables
├── requirements.txt
├── run_retrieval_coco.sh
├── run_retrieval_flickr.sh
└── code/
    ├── configs/               # task YAML configs
    ├── configs_local/         # local path overrides (not committed)
    ├── models/vlmo.py         # MoME transformer + task heads
    ├── datasets/
    ├── Retrieval.py
    └── VQA.py
```

## Path conventions

| Purpose | Path |
|---------|------|
| Task data | `/home/leon/DataSet/HuggingFace/2022_VLMO/data/` |
| Model weights (HF hub) | `/home/leon/Model/HuggingFace/hub/` |
| Training output | `/home/leon/Output/2022_VLMO/` |

## Environment setup

```bash
conda create -n 2022_VLMO python=3.9
conda activate 2022_VLMO
pip install -r requirements.txt
```

## Data preparation

### COCO Retrieval
```bash
# Download COCO 2014 images → data/coco/images/
# Download annotation JSONs → data/
```

### Flickr30K
```bash
# Download Flickr30K images → data/flickr30k/images/
# Download annotation JSONs → data/
```

## Fine-tuning

```bash
# COCO Retrieval
bash run_retrieval_coco.sh

# Flickr30K Retrieval
bash run_retrieval_flickr.sh --config configs/Retrieval_flickr.yaml
```

## Expected results (VLMO-Base)

| Task | R@1 (i2t) | R@1 (t2i) |
|------|-----------|-----------|
| COCO Retrieval | 78.2 | 60.6 |
| Flickr30K Retrieval | 97.7 | 86.7 |
