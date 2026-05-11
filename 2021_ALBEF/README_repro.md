# ALBEF Repro Quickstart

## Paths

- Code: `/home/lnick/GitHub/Paper/2021_ALBEF/code`
- Data: `/home/lnick/DataSet/2021_ALBEF`
- Output: `/home/lnick/Output/2021_ALBEF`

## 1) Initialize project paths

```bash
bash /home/lnick/GitHub/Paper/2021_ALBEF/setup_paths.sh
```

This creates the `code/data -> /home/lnick/DataSet/2021_ALBEF/annotations` symlink.

## 2) Install ALBEF dependencies (2021_ALBEF env)

```bash
PYTHONNOUSERSITE=1 conda run -n 2021_ALBEF python -m pip install -r /home/lnick/GitHub/Paper/2021_ALBEF/requirements_albef.txt
PYTHONNOUSERSITE=1 conda run -n 2021_ALBEF python -m pip install "ruamel.yaml<0.18" "torchvision==0.25.0" --force-reinstall
```

## 3) Download ALBEF official json/checkpoint assets

```bash
bash /home/lnick/GitHub/Paper/2021_ALBEF/download_albef_assets.sh
```

If network is unstable, re-run the script. It uses `wget -c` for resume.

## 4) Prepare raw datasets from original sources

Place images under:

- `/home/lnick/DataSet/2021_ALBEF/images/coco`
- `/home/lnick/DataSet/2021_ALBEF/images/flickr30k`
- `/home/lnick/DataSet/2021_ALBEF/images/visual-genome`
- `/home/lnick/DataSet/2021_ALBEF/images/nlvr2`
- `/home/lnick/DataSet/2021_ALBEF/images/snli-ve`

## 5) Run retrieval finetune

Flickr30k:

```bash
bash /home/lnick/GitHub/Paper/2021_ALBEF/run_retrieval_flickr.sh
```

MSCOCO:

```bash
bash /home/lnick/GitHub/Paper/2021_ALBEF/run_retrieval_coco.sh
```

## 6) Validation check done in this setup

`Retrieval.py` now starts correctly and reaches dataset loading stage, confirming code/env compatibility.
