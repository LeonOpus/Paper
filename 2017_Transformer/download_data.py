"""
Download WMT14 EN-DE from statmt.org and train a shared SentencePiece BPE tokenizer.
Training sources: Europarl v7 + News Commentary v9 (≈2.1M pairs, lighter than full WMT14)
Test set: newstest2014 (3003 pairs, official WMT14 benchmark)
Usage: python download_data.py
"""
import os
import sys
import json
import subprocess
import tarfile
import gzip

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg

SOURCES = {
    "europarl": (
        "http://www.statmt.org/europarl/v7/de-en.tgz",
        "europarl-v7.de-en",
    ),
    "commoncrawl": (
        "http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz",
        "commoncrawl.de-en",
    ),
    "news_commentary": (
        "http://www.statmt.org/wmt14/training-parallel-nc-v9.tgz",
        "news-commentary-v9.de-en",
    ),
}
NEWSTEST2014_URL = "http://www.statmt.org/wmt14/test-full.tgz"
NEWSTEST2013_URL = "http://www.statmt.org/wmt14/dev.tgz"


def wget(url: str, dest: str):
    print(f"Downloading (or resuming) {os.path.basename(dest)} ...")
    # -c lets wget skip complete files and resume partials automatically
    subprocess.run(["wget", "-c", "--show-progress", "-O", dest, url], check=True)


def extract_parallel(tgz_path: str, stem: str, tmp_dir: str):
    """Extract .en and .de files from a tgz archive."""
    en_out = os.path.join(tmp_dir, stem + ".en")
    de_out = os.path.join(tmp_dir, stem + ".de")
    if os.path.exists(en_out) and os.path.exists(de_out):
        return en_out, de_out
    print(f"Extracting {os.path.basename(tgz_path)} ...")
    with tarfile.open(tgz_path) as tar:
        for member in tar.getmembers():
            name = os.path.basename(member.name)
            if name == stem + ".en" or name == stem + ".de":
                member.name = name
                tar.extract(member, tmp_dir)
    return en_out, de_out


def read_lines(path: str) -> list:
    with open(path, encoding="utf-8", errors="ignore") as f:
        return [l.strip() for l in f if l.strip()]


def build_pairs(en_lines, de_lines, max_len_chars=500):
    pairs = []
    for en, de in zip(en_lines, de_lines):
        if len(en) < max_len_chars and len(de) < max_len_chars:
            pairs.append({"src": en, "tgt": de})
    return pairs


def parse_sgm(content: str) -> list:
    """Extract text from WMT SGM format."""
    import re
    return re.findall(r'<seg[^>]*>(.*?)</seg>', content, re.DOTALL)


def extract_newstest(tgz_path: str, tmp_dir: str, year: str):
    """Extract newstest{year} EN-DE src/ref from WMT SGM tgz."""
    en_lines, de_lines = [], []
    with tarfile.open(tgz_path) as tar:
        for member in tar.getmembers():
            n = member.name
            # EN->DE: src=en, ref=de  (e.g. newstest2014-deen-src.en.sgm)
            if f"newstest{year}-deen-src.en.sgm" in n:
                f = tar.extractfile(member)
                en_lines = parse_sgm(f.read().decode("utf-8", errors="ignore"))
            if f"newstest{year}-deen-ref.de.sgm" in n:
                f = tar.extractfile(member)
                de_lines = parse_sgm(f.read().decode("utf-8", errors="ignore"))
    return en_lines, de_lines


if __name__ == "__main__":
    os.makedirs(cfg.DATA_DIR, exist_ok=True)
    tmp = os.path.join(cfg.DATA_DIR, "tmp")
    os.makedirs(tmp, exist_ok=True)

    # --- Download archives ---
    tgz_paths = {}
    for name, (url, _) in SOURCES.items():
        dest = os.path.join(tmp, os.path.basename(url))
        wget(url, dest)
        tgz_paths[name] = dest

    test_tgz = os.path.join(tmp, "test-full.tgz")
    dev_tgz  = os.path.join(tmp, "dev.tgz")
    wget(NEWSTEST2014_URL, test_tgz)
    wget(NEWSTEST2013_URL, dev_tgz)

    # --- Extract parallel files ---
    all_train_pairs = []
    for name, (_, stem) in SOURCES.items():
        en_f, de_f = extract_parallel(tgz_paths[name], stem, tmp)
        pairs = build_pairs(read_lines(en_f), read_lines(de_f))
        print(f"  {name}: {len(pairs)} pairs")
        all_train_pairs.extend(pairs)
    print(f"Total train pairs: {len(all_train_pairs)}")

    # --- Train tokenizer ---
    tok_model = os.path.join(cfg.DATA_DIR, "tokenizer.model")
    if not os.path.exists(tok_model):
        import sentencepiece as spm
        corpus_file = os.path.join(tmp, "corpus.txt")
        print("Writing tokenizer corpus ...")
        with open(corpus_file, "w") as f:
            for p in all_train_pairs:
                f.write(p["src"] + "\n")
                f.write(p["tgt"] + "\n")
        print("Training SentencePiece BPE tokenizer (32K vocab) ...")
        spm.SentencePieceTrainer.train(
            input=corpus_file,
            model_prefix=os.path.join(cfg.DATA_DIR, "tokenizer"),
            vocab_size=32000,
            character_coverage=0.9995,
            model_type="bpe",
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
            pad_piece="<pad>", unk_piece="<unk>",
            bos_piece="<s>", eos_piece="</s>",
        )
        print(f"Tokenizer saved: {tok_model}")
    else:
        print(f"[skip] tokenizer exists")

    # --- Save splits ---
    train_out = os.path.join(cfg.DATA_DIR, "train.json")
    if not os.path.exists(train_out):
        json.dump(all_train_pairs, open(train_out, "w"), ensure_ascii=False)
        print(f"[train] {len(all_train_pairs)} pairs -> {train_out}")
    else:
        print(f"[skip] {train_out}")

    val_out = os.path.join(cfg.DATA_DIR, "val.json")
    if not os.path.exists(val_out):
        en_lines, de_lines = extract_newstest(dev_tgz, tmp, "2013")
        val_pairs = [{"src": en, "tgt": de} for en, de in zip(en_lines, de_lines)]
        json.dump(val_pairs, open(val_out, "w"), ensure_ascii=False)
        print(f"[val] {len(val_pairs)} pairs -> {val_out}")
    else:
        print(f"[skip] {val_out}")

    test_out = os.path.join(cfg.DATA_DIR, "test.json")
    if not os.path.exists(test_out):
        en_lines, de_lines = extract_newstest(test_tgz, tmp, "2014")
        test_pairs = [{"src": en, "tgt": de} for en, de in zip(en_lines, de_lines)]
        json.dump(test_pairs, open(test_out, "w"), ensure_ascii=False)
        print(f"[test] {len(test_pairs)} pairs -> {test_out}")
    else:
        print(f"[skip] {test_out}")

    print("Done.")
