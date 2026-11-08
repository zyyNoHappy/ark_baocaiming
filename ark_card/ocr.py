#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""本地 OCR 认干员名（PaddleOCR，可选）

设计成「有就用、没有就退」：
  · 没装 paddleocr        → available() 返回 False，调用方走视觉模型
  · 装了但识别不出来       → ocr_names() 返回 None，调用方走视觉模型
  · 识别出来了但跟排序表对不上 → 交给 match_order() 做模糊纠正，仍对不上的落空

顺带处理了三个环境坑（只在检测到问题时才动手，正常环境完全不受影响）：
  1. 某些受限环境 tempfile.mkdtemp() 建出来的目录写不进文件 → 换成 os.makedirs
  2. paddle 3.x 在部分 CPU 上 oneDNN 会崩 → 关掉 mkldnn
  3. 模型缓存目录不可写 → 用 PADDLE_PDX_CACHE_HOME 指到别处
"""
from __future__ import annotations

import os
import re
import sys
import tempfile
import uuid
from pathlib import Path

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore

_engine = None
_engine_failed = False
_booted = False


# ------------------------------------------------------------------ 环境
def _candidate_pylibs() -> list[Path]:
    here = Path(__file__).resolve()
    cands = [
        here.parent.parent / "pylibs",         # 插件目录/pylibs
        here.parent.parent.parent / "pylibs",   # 插件上级/pylibs
        here.parent.parent.parent.parent / "pylibs",
    ]
    env = os.environ.get("ARK_PYLIBS")
    if env:
        cands.insert(0, Path(env))
    return cands


def _patch_mkdtemp_if_broken() -> None:
    """mkdtemp 建的目录如果写不进文件，就换成普通 makedirs。"""
    try:
        probe = tempfile.mkdtemp(prefix="ark_probe_")
    except Exception:
        probe = None
    if probe:
        ok = False
        try:
            p = os.path.join(probe, "x.txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write("1")
            os.remove(p)
            ok = True
        except Exception:
            ok = False
        finally:
            try:
                os.rmdir(probe)
            except Exception:
                pass
        if ok:
            return

    def _mkdtemp(suffix: str = "", prefix: str = "tmp", dir: str | None = None) -> str:
        base = dir or tempfile.gettempdir()
        d = os.path.join(base, f"{prefix}{uuid.uuid4().hex[:10]}{suffix}")
        os.makedirs(d, exist_ok=True)
        return d

    tempfile.mkdtemp = _mkdtemp  # type: ignore[assignment]


def _bootstrap(cache_dir: str | None = None) -> None:
    global _booted
    if _booted:
        return
    _booted = True

    for lib in _candidate_pylibs():
        if lib.is_dir() and str(lib) not in sys.path:
            sys.path.insert(0, str(lib))

    _patch_mkdtemp_if_broken()

    # 模型缓存：默认 ~/.paddlex，不可写就退到临时目录
    if not os.environ.get("PADDLE_PDX_CACHE_HOME"):
        base = Path(cache_dir) if cache_dir else Path.home() / ".paddlex"
        try:
            base.mkdir(parents=True, exist_ok=True)
            probe = base / ".write_test"
            probe.write_text("1", encoding="utf-8")
            probe.unlink()
        except Exception:
            base = Path(tempfile.gettempdir()) / "paddlex_cache"
            base.mkdir(parents=True, exist_ok=True)
        os.environ["PADDLE_PDX_CACHE_HOME"] = str(base)

    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "modelscope")
    os.environ.setdefault("FLAGS_use_mkldnn", "0")


def available(cache_dir: str | None = None) -> bool:
    """paddleocr 能不能用（会真去 import 一次，有缓存）。"""
    global _engine_failed
    if _engine_failed:
        return False
    _bootstrap(cache_dir)
    try:
        import paddleocr  # noqa: F401
        return True
    except Exception:
        _engine_failed = True
        return False


def _get_engine(cache_dir: str | None = None):
    global _engine, _engine_failed
    if _engine is not None:
        return _engine
    if _engine_failed:
        raise RuntimeError("paddleocr 不可用")
    _bootstrap(cache_dir)
    try:
        from paddleocr import PaddleOCR

        _engine = PaddleOCR(
            lang="ch",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
        )
        return _engine
    except Exception:
        _engine_failed = True
        raise


# ------------------------------------------------------------------ 识别
def clean_name(texts: list[str]) -> str:
    """从 OCR 结果里挑出干员名：去掉职业代号等噪声，取最长的中文段。"""
    cands: list[str] = []
    for t in texts or []:
        t = re.sub(r"\s+", "", str(t))
        t = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9·\-]", "", t)
        if t:
            cands.append(t)
    if not cands:
        return ""
    cn = [c for c in cands if re.search(r"[\u4e00-\u9fa5]", c)]
    return max(cn or cands, key=len)


def _name_strip(card: "Image.Image"):
    """取卡片底部名条。

    以前放大 3 倍再识别，实测（141 张已确认卡里抽 40 张）3x 与 1x
    都是 40/40 命中，但 1x 快 1.6 倍，所以不放大。
    """
    w, h = card.size
    return card.crop((0, int(h * 0.87), w, h))


def _sim(a: str, b: str) -> float:
    """名字相似度 0~1。

    子串包含不能一律给高分：五星名常常包含六星名
    （拉普兰德⊂荒芜拉普兰德、火龙S黑角⊃黑、阿米娅⊃阿），
    一律 0.9 会让五星卡抢走六星的位置。
    所以按「短的是长的多少」来给分：4/6=0.67 不达标，
    而真正的错别字（司霆惊蛰/司霆惊垫）走 SequenceMatcher 仍然够分。
    """
    import difflib
    if a == b:
        return 1.0
    if a and b and (a in b or b in a):
        short, long_ = (a, b) if len(a) <= len(b) else (b, a)
        return len(short) / len(long_)
    return difflib.SequenceMatcher(None, a, b).ratio()


_CODE_RE = None


def _looks_like_code(s: str) -> bool:
    """像不像职业代号：2-6 位字母数字（ISW / ART / PLX / 7WW …）。

    干员名里像 Mon3tr 这种也符合形状，所以调用方还要再用排序表排除一次。
    """
    global _CODE_RE
    if _CODE_RE is None:
        import re as _re
        _CODE_RE = _re.compile(r'^[A-Za-z0-9]{2,6}$')
    return bool(_CODE_RE.match(s))


def pick_name(texts: list[str], valid: list[str] | None = None,
              scores: list | None = None) -> str:
    """从名条里识别出的一堆文本中挑干员名。

    名条上除了名字还有职业代号（PLX X / ART X / CSR X…）和等级数字，单看长度
    会把代号当成名字。给了排序表就按「像不像某个合法干员名」来挑。

    同分时再看 OCR 置信度、再看长度：否则短名字（W / 年 / 黑 / 阿 / 夕）会先到
    先得，把同一个名条里真正的名字挤掉 —— 实测「塔露拉」被识别成 W 就是这么来的。
    """
    cands = []
    valid_set = set(valid or [])
    for i, t in enumerate(texts or []):
        c = clean_name([t])
        if not c:
            continue
        # 名条上还有职业代号（ISW / ARTX / PLXX / HOKX…）。它们不是干员名，
        # 但 ISW 这类"含 W"的代号会以子串方式命中真正的干员 W，必须排掉。
        if _looks_like_code(c) and c not in valid_set:
            continue
        sc = float(scores[i]) if scores and i < len(scores) else 1.0
        cands.append((c, sc))
    if not cands:
        return ''
    if valid:
        best, best_key = '', (0.0, 0.0, 0)
        for c, sc in cands:
            sim = max((_sim(c, v) for v in valid), default=0.0)
            key = (sim, sc, len(c))
            if key > best_key:
                best, best_key = c, key
        # 短名字（1-2 字）容易"抢占"：繁简差异会让长名字掉到 0.8，
        # 而 W/年/黑 这类短名精确命中就是 1.0。这时改选更像的长名字。
        if 0 < len(best) <= 2:
            longer = [(c, max((_sim(c, v) for v in valid), default=0.0))
                      for c, _ in cands if len(c) >= 3]
            longer = [x for x in longer if x[1] >= 0.75]
            if longer:
                best = max(longer, key=lambda x: x[1])[0]
                best_key = (max(x[1] for x in longer), best_key[1], len(best))
        if best and best_key[0] >= 0.80:
            return best
    return clean_name(texts)


def ocr_names(cards: list, cache_dir: str | None = None,
              valid: list[str] | None = None) -> list[str] | None:
    """一批卡片 → 名字列表；不可用或全空时返回 None。

    valid 给了排序表的干员名，会用来在名条的多个文本块里挑对的那个。
    """
    if not cards or Image is None:
        return None
    try:
        engine = _get_engine(cache_dir)
    except Exception:
        return None

    out: list[str] = []
    for card in cards:
        try:
            strip = _name_strip(card)
            tmp = Path(tempfile.gettempdir()) / f"ark_name_{uuid.uuid4().hex[:8]}.png"
            strip.save(tmp)
            try:
                res = engine.predict(str(tmp))
                texts: list[str] = []
                scores: list = []
                for r in res:
                    texts.extend(r.get("rec_texts") or [])
                    scores.extend(r.get("rec_scores") or [])
            finally:
                tmp.unlink(missing_ok=True)
            out.append(pick_name(texts, valid, scores))
        except Exception:
            out.append("")
    if not any(out):
        return None
    return out


# ------------------------------------------------------------------ 纠错
def match_order(names: list[str], valid: list[str], threshold: float = 0.70):
    """把 OCR 结果对齐到排序表里的正式名。

    返回 (纠正后的列表, 命中率)。对不上的位置是 None。

    相似度用 _sim（子串按长度占比给分），阈值 0.70：
    五星名包含六星名的情况（拉普兰德/荒芜拉普兰德 = 0.67）会被挡住，
    而真正的错别字（司霆惊垫/司霆惊蛰 = 0.87）仍然能纠正。
    """
    valid_set = set(valid)
    fixed: list[str | None] = []
    hit = 0
    for n in names:
        n = (n or "").strip()
        if not n:
            fixed.append(None)
            continue
        if n in valid_set:
            fixed.append(n)
            hit += 1
            continue
        best, score = None, 0.0
        for v in valid:
            r = _sim(n, v)
            if r > score:
                best, score = v, r
        if best and score >= threshold:
            fixed.append(best)
            hit += 1
        else:
            fixed.append(None)
    return fixed, (hit / len(names) if names else 0.0)
