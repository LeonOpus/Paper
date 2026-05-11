# 2021 ALBEF Reproduction Plan (Single 3090, 24G)

## 1) Directory Convention

- Code root: `/home/lnick/GitHub/Paper/2021_ALBEF`
- Dataset root: `/home/lnick/DataSet/2021_ALBEF`
- Output root: `/home/lnick/Output/2021_ALBEF`

Current prepared folders:

- `/home/lnick/DataSet/2021_ALBEF/annotations`
- `/home/lnick/DataSet/2021_ALBEF/images`
- `/home/lnick/DataSet/2021_ALBEF/cache`
- `/home/lnick/DataSet/2021_ALBEF/checkpoints`
- `/home/lnick/Output/2021_ALBEF/logs`
- `/home/lnick/Output/2021_ALBEF/checkpoints`
- `/home/lnick/Output/2021_ALBEF/runs`

## 2) Environment Baseline

Validated on this server:

- GPU: `NVIDIA GeForce RTX 3090 (24G)`
- Driver/CUDA (system): `570.211.01 / 12.8`
- Disk free: about `509G`
- Conda env: `2021_ALBEF`
- Python/Torch in env: `Python 3.10.x`, `torch 2.10.0+cu128`

## 3) Recommended Repro Flow

### Step A: Clone ALBEF code

Choose one source (official or high-quality reproduction), then:

```bash
cd /home/lnick/GitHub/Paper/2021_ALBEF
# Example:
# git clone <ALBEF_REPO_URL> code
```

### Step B: Install project dependencies

```bash
PYTHONNOUSERSITE=1 conda run -n 2021_ALBEF python -m pip install -r /home/lnick/GitHub/Paper/2021_ALBEF/requirements_albef.txt
```

### Step C: Data placement

Place task data under:

- Image data: `/home/lnick/DataSet/2021_ALBEF/images`
- Annotation json/txt: `/home/lnick/DataSet/2021_ALBEF/annotations`
- Pretrained checkpoints (if needed): `/home/lnick/DataSet/2021_ALBEF/checkpoints`

### Step D: Path config unification

In ALBEF config/training scripts, set:

- `DATA_ROOT=/home/lnick/DataSet/2021_ALBEF`
- `OUTPUT_ROOT=/home/lnick/Output/2021_ALBEF`
- `HF_HOME=/home/lnick/DataSet/Hugging-Face` (if transformers/datasets used)

### Step E: Smoke test before full run

Run a tiny debug launch first (small batch, few steps, single GPU), verify:

- data can be read
- checkpoints/logs are written to `/home/lnick/Output/2021_ALBEF`
- no OOM on 24G VRAM

### Step F: Full reproduction run

After smoke test passes, run full settings with:

- gradient accumulation enabled
- mixed precision enabled
- periodic checkpoint save to `/home/lnick/Output/2021_ALBEF/checkpoints`

## 4) Practical 24G VRAM Defaults

Start conservative if config is unknown:

- global batch: `64` target via grad accumulation
- per-device batch: `8` or `16` (depends on image resolution)
- amp/mixed precision: `on`
- num_workers: `4~8`

## 5) Deliverables After First Successful Run

- A fixed `run.sh` for one-command launch
- A fixed `config.yaml` with absolute data/output roots
- A `README_repro.md` logging:
  - exact commit id
  - exact env versions
  - final command used
  - key metrics
