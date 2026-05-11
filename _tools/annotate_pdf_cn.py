#!/usr/bin/env python3
import argparse
import os
import re
import textwrap
import time
from pathlib import Path

import fitz  # PyMuPDF
import requests

TRANSLATE_URL = "https://translate.googleapis.com/translate_a/single"
GCLOUD_TRANSLATE_URL = "https://translation.googleapis.com/language/translate/v2"


def looks_like_reference_page(page_text: str) -> bool:
    return False


def is_good_paragraph(text: str) -> bool:
    s = " ".join(text.split())
    if len(s) < 25:
        return False
    if "@" in s:
        return False
    if len(re.findall(r"[A-Za-z]", s)) < 10:
        return False
    # Skip tiny headings like "Introduction" / "Conclusion"
    word_count = len(s.split())
    if word_count <= 6 and not re.search(r"[.!?;:]", s):
        return False
    if re.fullmatch(r"[\d\s\W_]+", s):
        return False
    return True


def extract_paragraphs(page: fitz.Page):
    blocks = page.get_text("blocks")
    blocks.sort(key=lambda b: (b[1], b[0]))
    paras = []
    for b in blocks:
        txt = b[4].strip()
        if not txt:
            continue
        parts = re.split(r"\n\s*\n", txt)
        for p in parts:
            p = " ".join(p.split())
            if is_good_paragraph(p):
                paras.append(p)
    return paras


def translate_en_to_zh(
    text: str,
    session: requests.Session,
    cache: dict,
    translator: str,
    api_key: str,
    sleep_sec: float = 0.1,
) -> str:
    if text in cache:
        return cache[text]

    try:
        if translator == "google_cloud":
            if not api_key:
                raise ValueError("GOOGLE_TRANSLATE_API_KEY not set")
            payload = {
                "q": text,
                "source": "en",
                "target": "zh-CN",
                "format": "text",
                "model": "nmt",
                "key": api_key,
            }
            r = session.post(GCLOUD_TRANSLATE_URL, data=payload, timeout=25)
            r.raise_for_status()
            data = r.json()
            translated = data["data"]["translations"][0]["translatedText"]
            translated = re.sub(r"<[^>]+>", "", translated)
        else:
            params = {
                "client": "gtx",
                "sl": "en",
                "tl": "zh-CN",
                "dt": "t",
                "q": text,
            }
            r = session.get(TRANSLATE_URL, params=params, timeout=25)
            r.raise_for_status()
            data = r.json()
            translated = "".join(seg[0] for seg in data[0] if seg and seg[0])
        translated = " ".join(translated.split())
    except Exception:
        translated = "[翻译失败，请稍后重试该段]"
    cache[text] = translated
    time.sleep(sleep_sec)
    return translated


def short_cn(text: str, max_chars: int = 120) -> str:
    if max_chars <= 0:
        return text.strip()
    s = text.strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 1] + "…"


def to_brief_cn(text: str) -> str:
    s = text.strip()
    if not s:
        return "（该段解析为空，可能由公式或扫描文本导致）"
    # Clean mixed CJK / Latin spacing for better reading.
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", s)
    s = re.sub(r"(?<=[\u4e00-\u9fff])(?=[A-Za-z0-9])", " ", s)
    s = re.sub(r"(?<=[A-Za-z0-9])(?=[\u4e00-\u9fff])", " ", s)
    if not s.endswith(("。", "！", "？")):
        s += "。"
    return s


def _is_ascii_word(tok: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+\-/%]*", tok))


def wrap_zh_line(text: str, width: int = 48):
    lines = []
    for raw in text.splitlines():
        if not raw:
            lines.append("")
            continue

        tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9._+\-/%]*|[\u4e00-\u9fff]|[^\s]", raw)
        cur = ""
        cur_len = 0
        prev_tok = ""

        for tok in tokens:
            need_space = False
            if cur:
                if (_is_ascii_word(prev_tok) and _is_ascii_word(tok)) or (
                    _is_ascii_word(prev_tok) and re.fullmatch(r"[\u4e00-\u9fff]", tok)
                ) or (re.fullmatch(r"[\u4e00-\u9fff]", prev_tok) and _is_ascii_word(tok)):
                    need_space = True

            add = (" " if need_space else "") + tok
            add_len = len(add)
            if cur and cur_len + add_len > width:
                lines.append(cur)
                cur = tok
                cur_len = len(tok)
            else:
                cur += add
                cur_len += add_len
            prev_tok = tok

        if cur:
            lines.append(cur)
    return lines


def _textbox_fits(text: str, width: float, height: float, fontfile: str, fontsize: float) -> bool:
    tdoc = fitz.open()
    tp = tdoc.new_page(width=width, height=height)
    tp.insert_font(fontname="cjk", fontfile=fontfile)
    rect = fitz.Rect(42, 58, width - 42, height - 42)
    rc = tp.insert_textbox(rect, text, fontsize=fontsize, fontname="cjk", align=fitz.TEXT_ALIGN_LEFT)
    tdoc.close()
    return rc >= 0


def _best_fit_prefix(text: str, width: float, height: float, fontfile: str, fontsize: float) -> int:
    lo, hi = 1, len(text)
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if _textbox_fits(text[:mid], width, height, fontfile, fontsize):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    if best <= 0:
        return 1
    cut = best
    floor = max(1, best - 200)
    while cut > floor and text[cut - 1] not in "\n。！？；;.!?":
        cut -= 1
    return cut if cut > floor else best


def draw_note_pages(dst: fitz.Document, base_w: float, base_h: float, page_index: int, entries, fontfile: str):
    margin_x = 42
    margin_y = 42
    content_rect = fitz.Rect(margin_x, margin_y + 16, base_w - margin_x, base_h - margin_y)
    body = "\n\n".join(entries).strip()
    if not body:
        return

    fontsize = 11
    chunks = []
    rest = body
    while rest:
        if _textbox_fits(rest, base_w, base_h, fontfile, fontsize):
            chunks.append(rest)
            break
        n = _best_fit_prefix(rest, base_w, base_h, fontfile, fontsize)
        chunks.append(rest[:n].rstrip())
        rest = rest[n:].lstrip()

    for i, chunk in enumerate(chunks, start=1):
        p = dst.new_page(width=base_w, height=base_h)
        p.insert_font(fontname="cjk", fontfile=fontfile)
        title = f"第 {page_index + 1} 页中文解析（{i}/{len(chunks)}）"
        p.insert_text((margin_x, margin_y - 10), title, fontsize=14, fontname="cjk")
        p.insert_textbox(content_rect, chunk, fontsize=fontsize, fontname="cjk", align=fitz.TEXT_ALIGN_LEFT)


def process(input_pdf: Path, output_pdf: Path, fontfile: str, max_cn_chars: int, translator: str, api_key: str):
    src = fitz.open(input_pdf)
    dst = fitz.open()
    session = requests.Session()
    cache = {}

    total_pages = len(src)
    for idx in range(total_pages):
        sp = src[idx]

        # Copy original page
        op = dst.new_page(width=sp.rect.width, height=sp.rect.height)
        op.show_pdf_page(op.rect, src, idx)

        paras = extract_paragraphs(sp)
        if not paras:
            continue

        entries = []
        for j, para in enumerate(paras, start=1):
            zh = translate_en_to_zh(para, session, cache, translator=translator, api_key=api_key)
            zh = to_brief_cn(short_cn(zh, max_chars=max_cn_chars))
            entries.append(f"[{j}] {zh}")

        draw_note_pages(dst, sp.rect.width, sp.rect.height, idx, entries, fontfile)
        print(f"processed page {idx + 1}/{total_pages}, paragraphs={len(paras)}")

    dst.save(output_pdf)
    dst.close()
    src.close()


def main():
    parser = argparse.ArgumentParser(description="Add Chinese notes per paragraph to a PDF as note pages.")
    parser.add_argument("input_pdf", type=Path)
    parser.add_argument("output_pdf", type=Path)
    parser.add_argument(
        "--fontfile",
        default="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        help="Path to a CJK-capable font file",
    )
    parser.add_argument(
        "--translator",
        choices=["auto", "google_cloud", "google_web"],
        default="auto",
        help="Translator backend. auto prefers official Google Cloud API when GOOGLE_TRANSLATE_API_KEY is set.",
    )
    parser.add_argument("--max-cn-chars", type=int, default=0, help="Max chars per Chinese note; <=0 means no truncation")
    args = parser.parse_args()

    api_key = os.getenv("GOOGLE_TRANSLATE_API_KEY", "").strip()
    if args.translator == "auto":
        translator = "google_cloud" if api_key else "google_web"
    elif args.translator == "google_cloud":
        translator = "google_cloud"
    else:
        translator = "google_web"

    if translator == "google_cloud" and not api_key:
        raise SystemExit("translator=google_cloud requires env GOOGLE_TRANSLATE_API_KEY")

    print(f"translator backend: {translator}")
    process(args.input_pdf, args.output_pdf, args.fontfile, args.max_cn_chars, translator=translator, api_key=api_key)
    print(f"done: {args.output_pdf}")


if __name__ == "__main__":
    main()
