"""
Evaluate Transformer on WMT14 EN-DE newstest2014.
Beam search decoding + sacrebleu BLEU.
Usage: python eval.py --config ../configs/translation_wmt14.yaml --checkpoint path/to/ckpt.pt
"""
import argparse
import json
import os
import sys
import yaml
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from code.model import Transformer


def beam_search(model, src, src_mask, bos_id, eos_id, pad_id,
                beam_size: int, max_len: int, length_penalty: float, device):
    B = src.size(0)
    assert B == 1, "beam search runs one sentence at a time"

    enc_out = model.encoder(src, src_mask)

    beams  = [[bos_id]]       # list of token sequences
    scores = [0.0]
    done   = []

    for _ in range(max_len):
        candidates = []
        for seq, score in zip(beams, scores):
            tgt = torch.tensor([seq], device=device)
            with torch.no_grad():
                logits = model.decode_step(tgt, enc_out, src_mask)
            log_probs = torch.log_softmax(logits[0, -1], dim=-1)
            topk = log_probs.topk(beam_size)
            for prob, idx in zip(topk.values, topk.indices):
                new_seq   = seq + [idx.item()]
                new_score = score + prob.item()
                candidates.append((new_seq, new_score))

        candidates.sort(key=lambda x: x[1] / (len(x[0]) ** length_penalty), reverse=True)
        beams, scores = [], []
        for seq, score in candidates[:beam_size]:
            if seq[-1] == eos_id:
                done.append((seq[1:-1], score / (len(seq) ** length_penalty)))
            else:
                beams.append(seq)
                scores.append(score)
        if not beams:
            break

    if not done:
        done = [(beams[0][1:], scores[0])]
    done.sort(key=lambda x: x[1], reverse=True)
    return done[0][0]


def evaluate(args):
    with open(args.config) as f:
        conf = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=cfg.resolve_data("tokenizer.model"))
    bos_id, eos_id, pad_id = sp.bos_id(), sp.eos_id(), sp.pad_id()

    model = Transformer(
        vocab_size=conf["vocab_size"],
        d_model=conf["d_model"],
        n_heads=conf["n_heads"],
        d_ff=conf["d_ff"],
        n_layers=conf["n_layers"],
        dropout=0.0,
        max_len=conf["max_len"] + 2,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    test_data = json.load(open(cfg.resolve_data("test.json")))
    hypotheses, references = [], []

    print(f"Translating {len(test_data)} sentences ...")
    for i, item in enumerate(test_data):
        src_ids = sp.encode(item["src"])[:conf["max_len"]]
        src = torch.tensor([src_ids], device=device)
        src_mask = Transformer.make_src_mask(src)

        hyp_ids = beam_search(
            model, src, src_mask, bos_id, eos_id, pad_id,
            beam_size=conf["beam_size"],
            max_len=conf["max_len"],
            length_penalty=conf["length_penalty"],
            device=device,
        )
        hypotheses.append(sp.decode(hyp_ids))
        references.append(item["tgt"])

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(test_data)}")

    import sacrebleu
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    print(f"\n=== WMT14 EN-DE Results ===")
    print(f"  BLEU: {bleu.score:.2f}")
    print(f"  (Paper Transformer-base: 27.3)")

    out_rel = conf.get("output_dir", "")
    output_dir = os.path.join(cfg.OUTPUT_DIR, out_rel) if out_rel else cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    result = {"BLEU": bleu.score, "signature": str(bleu)}
    import json as _json
    _json.dump(result, open(os.path.join(output_dir, "eval_results.json"), "w"), indent=2)
    print(f"Saved to {output_dir}/eval_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True)
    parser.add_argument("--checkpoint", required=True)
    evaluate(parser.parse_args())
