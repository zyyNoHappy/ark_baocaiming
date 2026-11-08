#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把每张卡底部的干员名条拼成一张"名字表", 交给视觉模型读"""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    'C:/Windows/Fonts/msyh.ttc',
    'C:/Windows/Fonts/simhei.ttf',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
]


def _font(size: int):
    for p in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def name_strip(card: Image.Image) -> Image.Image:
    """取卡片底部的一条(名字所在区域)"""
    w, h = card.size
    return card.crop((int(w * 0.16), h - int(h * 0.115), w, h))


def build_sheet(cards: list[Image.Image], cols: int = 4, scale: float = 1.6,
                tag: str = '') -> Image.Image:
    strips = [name_strip(c) for c in cards]
    if not strips:
        raise ValueError('没有卡片')
    sw = max(s.width for s in strips)
    sh = max(s.height for s in strips)
    tw, th = int(sw * scale), int(sh * scale)
    rows = (len(strips) + cols - 1) // cols
    pad = 6
    label_h = 22
    sheet = Image.new('RGB', (cols * (tw + pad) + pad, rows * (th + label_h + pad) + pad),
                      (18, 18, 20))
    d = ImageDraw.Draw(sheet)
    f = _font(16)
    for i, s in enumerate(strips):
        x = pad + (i % cols) * (tw + pad)
        y = pad + (i // cols) * (th + label_h + pad)
        sheet.paste(s.resize((tw, th), Image.LANCZOS), (x, y))
        d.text((x + 2, y + th + 2), f'#{i + 1}', font=f, fill=(150, 150, 160))
    if tag:
        d.text((sheet.width - 90, 2), tag, font=f, fill=(90, 90, 100))
    return sheet


def parse_names(text: str, expected: int) -> list[str] | None:
    """从模型回复里抠出干员名; 数量对不上就返回 None"""
    import re
    # 优先取 JSON 数组
    m = re.search(r'\[[^\]]*\]', text, re.S)
    if m:
        import json
        try:
            arr = json.loads(m.group(0))
            names = [str(x).strip() for x in arr]
            if len(names) == expected:
                return names
        except Exception:
            pass
    # 退化为逐行
    lines = [ln.strip(' \t-*·、,，。."\'') for ln in text.splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith('```')]
    names = []
    for ln in lines:
        ln = re.sub(r'^\d+[.、)]\s*', '', ln)
        if 1 <= len(ln) <= 8 and re.fullmatch(r'[\u4e00-\u9fa5A-Za-z0-9·\-]+', ln):
            names.append(ln)
    if len(names) == expected:
        return names
    return None
