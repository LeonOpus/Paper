"""
无需 Java 即可计算图像描述评估指标。
将 pycocoevalcap 的 PTB 分词器替换为纯 Python 实现。
用法：python compute_metrics.py --config ../configs/caption_coco.yaml
"""
import re
import json
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg


def _python_tokenize(caption: str) -> str:
    return re.sub(r'[^a-z0-9 ]', ' ', caption.lower()).split().__len__() and \
           ' '.join(re.sub(r'[^a-z0-9 ]', ' ', caption.lower()).split()) or ''


def _patch_pycocoeval():
    """
    将 pycocoevalcap 中依赖 Java 的组件替换为纯 Python 实现。
    保留：BLEU、ROUGE-L、CIDEr（均为纯 Python）。
    移除：METEOR、SPICE（均需要 Java）。
    """
    import pycocoevalcap.tokenizer.ptbtokenizer as ptb
    import pycocoevalcap.eval as evalmod

    class PythonTokenizer:
        def tokenize(self, captions: dict) -> dict:
            result = {}
            for img_id, cap_list in captions.items():
                tokens = []
                for entry in cap_list:
                    text = entry.get('caption', entry) if isinstance(entry, dict) else entry
                    tokens.append(' '.join(re.sub(r'[^a-z0-9 ]', ' ', text.lower()).split()))
                result[img_id] = tokens
            return result

    ptb.PTBTokenizer = PythonTokenizer

    # 从评估循环中移除依赖 Java 的评分器（Meteor、SPICE）
    _orig_evaluate = evalmod.COCOEvalCap.evaluate

    def _patched_evaluate(self):
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.cider.cider import Cider

        tokenizer = PythonTokenizer()
        gts = tokenizer.tokenize(self.coco.imgToAnns)
        res = tokenizer.tokenize(self.cocoRes.imgToAnns)

        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(),  "ROUGE_L"),
            (Cider(),  "CIDEr"),
        ]
        self.eval = {}
        self.imgToEval = {}
        for scorer, method in scorers:
            print(f'computing {method if isinstance(method, str) else method[0]}...')
            score, scores = scorer.compute_score(gts, res)
            if isinstance(method, list):
                for m, s, ss in zip(method, score, scores):
                    self.setEval(s, m)
                    self.setImgToEvalImgs(ss, list(gts.keys()), m)
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, list(gts.keys()), method)
        self.setEvalImgs()

    evalmod.COCOEvalCap.evaluate = _patched_evaluate


def compute(result_file: str, ann_file: str, output_dir: str):
    _patch_pycocoeval()

    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    anns_raw = json.load(open(ann_file))
    coco_anns = {'images': [], 'annotations': [], 'type': 'captions', 'info': {}, 'licenses': []}
    seen, ann_id = {}, 0
    for item in anns_raw:
        iid = item.get('image_id')
        if iid is None:
            continue
        if iid not in seen:
            seen[iid] = True
            coco_anns['images'].append({'id': iid})
        cap = item['caption']
        if isinstance(cap, list):
            for c in cap:
                coco_anns['annotations'].append({'image_id': iid, 'id': ann_id, 'caption': c})
                ann_id += 1
        else:
            coco_anns['annotations'].append({'image_id': iid, 'id': ann_id, 'caption': cap})
            ann_id += 1

    os.makedirs(output_dir, exist_ok=True)
    gt_file = os.path.join(output_dir, 'coco_gt_captions.json')
    json.dump(coco_anns, open(gt_file, 'w'))

    coco_gt = COCO(gt_file)
    coco_res = coco_gt.loadRes(result_file)
    coco_eval = COCOEvalCap(coco_gt, coco_res)
    coco_eval.evaluate()

    print('\n=== Captioning Results ===')
    for metric, score in coco_eval.eval.items():
        print(f'  {metric}: {score:.4f}')

    json.dump(dict(coco_eval.eval),
              open(os.path.join(output_dir, 'caption_metrics.json'), 'w'), indent=2)
    return coco_eval.eval


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--result', default=None, help='Path to caption_results.json')
    args = parser.parse_args()

    import yaml
    conf = yaml.safe_load(open(args.config))
    out_rel = conf.get('output_dir', '')
    output_dir = os.path.join(cfg.OUTPUT_DIR, out_rel) if out_rel else cfg.OUTPUT_DIR
    result_file = args.result or os.path.join(output_dir, 'caption_results.json')
    ann_file = cfg.resolve_data(conf['test_ann'])

    compute(result_file, ann_file, output_dir)
