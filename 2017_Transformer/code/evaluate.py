"""
在 WMT14 newstest 上评估 BLEU。
用法：python evaluate.py --config ../configs/translation_wmt14.yaml --ckpt <checkpoint.pt>
"""
import argparse
import json
import os
import sys
import torch
import sacrebleu

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from model import Transformer


def beam_search(model, src, src_mask, bos_id, eos_id, max_len, beam_size, length_penalty, device):
    """单条句子的 beam search，返回最优 token id 列表（不含 bos/eos）"""
    enc_out, src_mask = model.encode(src)  # src: (1, S)

    # 初始化：beam 条目为 (log_prob, token_ids)
    beams = [(0.0, [bos_id])]
    completed = []

    for _ in range(max_len):
        candidates = []
        for log_prob, tokens in beams:
            if tokens[-1] == eos_id:
                completed.append((log_prob, tokens))
                continue
            tgt = torch.tensor([tokens], dtype=torch.long, device=device)
            logits = model.decode_step(tgt, enc_out, src_mask)  # (1, T, V)
            log_probs = torch.log_softmax(logits[0, -1], dim=-1)  # (V,)
            topk_lp, topk_ids = log_probs.topk(beam_size)
            for lp, idx in zip(topk_lp.tolist(), topk_ids.tolist()):
                candidates.append((log_prob + lp, tokens + [idx]))

        # 保留 beam_size 条最优路径
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = []
        for score, tokens in candidates:
            if tokens[-1] == eos_id:
                completed.append((score, tokens))
            else:
                beams.append((score, tokens))
            if len(beams) >= beam_size:
                break

        if len(beams) == 0:
            break

    completed += beams  # 未遇到 eos 的也加进来

    # length penalty: score / len^alpha
    def penalized(item):
        score, tokens = item
        l = max(len(tokens) - 1, 1)  # 减去 bos
        return score / (l ** length_penalty)

    best = max(completed, key=penalized)
    result = best[1][1:]  # 去掉 bos
    if eos_id in result:
        result = result[:result.index(eos_id)]
    return result


def evaluate(args):
    import yaml
    with open(args.config) as f:
        conf = yaml.safe_load(f)

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=cfg.resolve_data("tokenizer.model"))
    bos_id, eos_id, pad_id = sp.bos_id(), sp.eos_id(), sp.pad_id()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = Transformer(
        vocab_size=conf["vocab_size"],
        d_model=conf["d_model"],
        n_heads=conf["n_heads"],
        d_ff=conf["d_ff"],
        n_layers=conf["n_layers"],
        dropout=0.0,
        max_len=conf["max_len"] + 2,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"加载 checkpoint: {args.ckpt}  (epoch={ckpt.get('epoch')}, global_step={ckpt.get('global_step')})")

    test_data = json.load(open(cfg.resolve_data("test.json")))
    beam_size     = conf.get("beam_size", 4)
    length_penalty = conf.get("length_penalty", 0.6)
    max_len       = conf["max_len"]
    limit         = args.limit  # 只跑前 N 条，None = 全部

    hypotheses, references = [], []

    with torch.no_grad():
        for i, item in enumerate(test_data[:limit]):
            src_ids = sp.encode(item["src"])[:max_len]
            src = torch.tensor([src_ids], dtype=torch.long, device=device)
            src_mask = model.make_src_mask(src)

            pred_ids = beam_search(model, src, src_mask, bos_id, eos_id, max_len, beam_size, length_penalty, device)
            hyp = sp.decode(pred_ids)
            ref = item["tgt"]
            hypotheses.append(hyp)
            references.append(ref)

            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(test_data[:limit] if limit else test_data)}", flush=True)

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    print(f"\n{'='*50}")
    print(f"Checkpoint : {os.path.basename(args.ckpt)}")
    print(f"样本数      : {len(hypotheses)}")
    print(f"BLEU       : {bleu.score:.2f}")
    print(f"{'='*50}")
    print(f"\n论文基准 (Transformer base, WMT14 EN-DE): BLEU 27.3")

    # 打印几条样例
    print("\n--- 样例 ---")
    for i in range(min(3, len(hypotheses))):
        print(f"[src] {test_data[i]['src']}")
        print(f"[ref] {references[i]}")
        print(f"[hyp] {hypotheses[i]}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt",   required=True)
    parser.add_argument("--limit",  type=int, default=None, help="只评估前 N 条（调试用）")
    evaluate(parser.parse_args())
