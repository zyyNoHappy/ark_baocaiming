#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把同一干员、各玩家的卡面合成一帧"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

W, H = 1920, 1080
ASPECT = 393 / 192.0
MARGIN = 56
VGAP = 50                  # 纵向统一间距：顶部 / 排间 / 排-日期 / 日期-底部
GAP_X = 40
GAP_Y = 44

FONT_CANDIDATES = ['C:/Windows/Fonts/msyhbd.ttc', 'C:/Windows/Fonts/msyh.ttc',
                   '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc']
ACCENTS = [(84, 146, 232), (232, 160, 84), (128, 200, 128), (200, 120, 200),
           (232, 120, 120), (120, 200, 210)]


def font(size: int):
    for p in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def default_background() -> Image.Image:
    """没有背景图时的深色渐变兜底"""
    y, x = np.mgrid[0:H, 0:W]
    d = np.sqrt(((x - W / 2) / (W * 0.75)) ** 2 + ((y - H * 0.42) / (H * 0.95)) ** 2)
    v = np.clip(1.15 - d * 0.95, 0, 1) ** 1.5
    base = np.zeros((H, W, 3))
    for i, (lo, hi) in enumerate([(10, 34), (11, 36), (14, 42)]):
        base[..., i] = lo + (hi - lo) * v
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def load_background(path: str | None = None) -> Image.Image:
    if path and Path(path).exists():
        bg = Image.open(path).convert('RGB')
        if bg.size != (W, H):
            bg = bg.resize((W, H), Image.LANCZOS)
        return bg
    return default_background()


def layout(n: int, bg: Image.Image, fit: float = 1.0, gap_ratio: float = 0.06):
    """返回 (cols, rows, block_w, header_h, card_h, start_x, start_y, date_y)

    纵向节奏统一：顶部 = 排间 = 排到日期 = 日期到底部 = VGAP。
    fit<1 时整体缩小（露出更多背景），gap_ratio 控制组间距。
    """
    avail_w = W - 2 * MARGIN
    date_size = int(H * 0.062)
    date_h = int(date_size * 1.15)
    unit = 1 / 3.2 + ASPECT                    # 每块高 / 块宽

    best = None
    for cols in range(1, n + 1):
        rows = (n + cols - 1) // cols
        bw_h = (H - (rows + 2) * VGAP - date_h) / (rows * unit)
        bw_w = avail_w / (cols + gap_ratio * (cols - 1))
        bw = int(min(bw_h, bw_w))
        if bw < 40:
            continue
        if best is None or bw > best[0]:
            best = (bw, cols, rows)
    if best is None:
        best = (60, max(1, n), 1)

    bw, cols, rows = best
    bw = max(40, int(bw * fit))
    gap = int(bw * gap_ratio)
    hdr = int(round(bw / 3.2))
    card_h = int(round(bw * ASPECT))
    block_h = hdr + card_h
    total_w = cols * bw + (cols - 1) * gap
    total_h = rows * block_h + (rows + 1) * VGAP + date_h
    rest = H - total_h
    y0 = VGAP + max(0, rest // 2)
    date_y = y0 + rows * block_h + rows * VGAP
    return cols, rows, bw, hdr, card_h, (W - total_w) // 2, y0, date_y


def fill_slot(img: Image.Image, size: tuple[int, int], mode: str = 'contain') -> Image.Image:
    """把任意尺寸的表情包放进固定卡位。

    contain（默认）: 等比完整放入；横向撑满卡位宽，纵向居中，
                     上下空出来的地方保持透明（露出背景），不裁内容、不糊底。
    cover         : 等比铺满卡位，居中裁剪。
    """
    tw, th = size
    img = img.convert('RGB')

    if mode == 'cover':
        s = max(tw / img.width, th / img.height)
        nw, nh = max(1, int(img.width * s + 0.5)), max(1, int(img.height * s + 0.5))
        im = img.resize((nw, nh), Image.LANCZOS)
        left, top = (nw - tw) // 2, (nh - th) // 2
        return im.crop((left, top, left + tw, top + th))

    s = min(tw / img.width, th / img.height)
    nw, nh = max(1, int(img.width * s + 0.5)), max(1, int(img.height * s + 0.5))
    fg = img.resize((nw, nh), Image.LANCZOS).convert('RGBA')
    canvas = Image.new('RGBA', (tw, th), (0, 0, 0, 0))
    canvas.alpha_composite(fg, ((tw - nw) // 2, (th - nh) // 2))
    return canvas


def placeholder_card(size: tuple[int, int], text: str = '未持有') -> Image.Image:
    """没设占位图时的兜底卡位"""
    tw, th = size
    img = Image.new('RGB', (tw, th), (34, 36, 44))
    d = ImageDraw.Draw(img)
    f = font(max(16, int(tw * 0.11)))
    tw_, th_ = d.textlength(text, font=f), f.size
    d.text(((tw - tw_) / 2, (th - th_) / 2), text, font=f, fill=(120, 124, 136))
    return img


def _rounded(img: Image.Image, r: int) -> Image.Image:
    mask = Image.new('L', img.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, img.size[0] - 1, img.size[1] - 1], r, fill=255)
    out = img.convert('RGBA')
    if img.mode in ('RGBA', 'LA'):          # 已有透明区域(占位图上下留空) -> 与圆角相乘
        out.putalpha(ImageChops.multiply(out.getchannel('A'), mask))
    else:
        out.putalpha(mask)
    return out


def _accent(nick: str) -> tuple[int, int, int]:
    h = hashlib.md5(nick.encode('utf-8')).digest()[0]
    return ACCENTS[h % len(ACCENTS)]


def build_block(nick: str, card_path: str | None, bw: int, hdr_h: int, card_h: int,
                gap: int, level: str | None = None,
                header_path: str | None = None,
                placeholder: str | None = None,
                placeholder_mode: str = 'contain') -> Image.Image:
    rounded_edges = True
    if card_path:
        card = Image.open(card_path).convert('RGB').resize((bw, card_h), Image.LANCZOS)
    elif placeholder:
        card = fill_slot(Image.open(placeholder), (bw, card_h), placeholder_mode)
        rounded_edges = False          # 占位卡保持直角（黑角是它的一部分）
    else:
        card = placeholder_card((bw, card_h))
    block = Image.new('RGBA', (bw, hdr_h + gap + card_h), (0, 0, 0, 0))

    if header_path and Path(header_path).exists():
        hp = Image.open(header_path).convert('RGB').resize((bw, hdr_h), Image.LANCZOS)
        block.alpha_composite(_rounded(hp, int(bw * 0.035)), (0, 0))
        layer = _rounded(card, max(3, int(bw * 0.012))) if rounded_edges else card
        block.alpha_composite(layer, (0, hdr_h + gap))
        return block

    # 名片条: 圆角深色底 + 左侧色条 + 昵称
    hdr = Image.new('RGBA', (bw, hdr_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(hdr)
    d.rounded_rectangle([0, 0, bw - 1, hdr_h - 1], int(bw * 0.035), fill=(22, 24, 32, 232))
    acc = _accent(nick)
    d.rounded_rectangle([0, 0, int(bw * 0.022), hdr_h - 1], int(bw * 0.011), fill=acc + (255,))
    fs = int(hdr_h * 0.42)
    f = font(fs)
    # 昵称过长时缩字号
    while fs > 10 and d.textlength(nick, font=f) > bw * 0.86:
        fs -= 1
        f = font(fs)
    tw = d.textlength(nick, font=f)
    d.text(((bw - tw) / 2, (hdr_h - fs) / 2 - fs * 0.08), nick, font=f, fill=(245, 246, 250, 255))
    block.alpha_composite(hdr, (0, 0))
    layer = _rounded(card, max(3, int(bw * 0.012))) if rounded_edges else card
    block.alpha_composite(layer, (0, hdr_h + gap))
    return block


def draw_title(img: Image.Image, index, name: str):
    """顶部标题：排序数字(金色) + 干员名(白色)"""
    if not name and index is None:
        return
    size = int(H * 0.050)
    f = font(size)
    d = ImageDraw.Draw(img)
    num = str(index) if index is not None else ''
    wn = d.textlength(num, font=f) if num else 0
    wm = d.textlength(name, font=f) if name else 0
    gap = size * 0.45 if num and name else 0
    x = (W - (wn + gap + wm)) / 2
    y = int(H * 0.022)
    # 阴影，保证浅色背景上也看得清
    for dx, dy in ((2, 2),):
        if num:
            d.text((x + dx, y + dy), num, font=f, fill=(0, 0, 0, 140))
        if name:
            d.text((x + wn + gap + dx, y + dy), name, font=f, fill=(0, 0, 0, 140))
    if num:
        d.text((x, y), num, font=f, fill=(255, 208, 92))
        x += wn + gap
    if name:
        d.text((x, y), name, font=f, fill=(255, 255, 255))


def draw_date(img: Image.Image, date: str, y: int):
    """右下角：国服上线时间（20% 不透明度，无刻度线）"""
    if not date or len(date) < 8:
        return
    txt = f'{date[:4]}-{date[4:6]}-{date[6:8]}'
    size = int(H * 0.062)
    f = font(size)
    d = ImageDraw.Draw(img)
    tw = d.textlength(txt, font=f)
    x = W - tw - int(W * 0.030)
    # 画在独立透明层上再合成 —— 直接往 RGBA 图层画会覆盖 alpha，
    # 最后 convert('RGB') 时 alpha 被丢掉，透明度就失效了
    layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
    ImageDraw.Draw(layer).text((x, y), txt, font=f, fill=(255, 255, 255, 204))
    img.alpha_composite(layer)


def compose_frame(entries: list[dict], out_path: str,
                  background: str | None = None, title: str | None = None,
                  index=None, date: str = '',
                  fit: float = 1.0, gap_ratio: float = 0.06) -> str:
    """entries: [{'nick': str, 'card': path, 'level': str?}, ...]"""
    bg = load_background(background).convert('RGBA')
    if not entries:
        raise ValueError('这一帧没有可用的卡')
    cols, rows, bw, hdr_h, card_h, x0, y0, date_y = layout(len(entries), bg, fit, gap_ratio)
    gap = 0
    block_h = hdr_h + gap + card_h

    for i, e in enumerate(entries):
        r, c = divmod(i, cols)
        x = x0 + c * (bw + int(bw * gap_ratio))
        y = y0 + r * (block_h + VGAP)
        blk = build_block(e['nick'], e.get('card'), bw, hdr_h, card_h, gap,
                          e.get('level'), e.get('header'),
                          e.get('placeholder'), e.get('placeholder_mode', 'contain'))
        sh = Image.new('RGBA', blk.size, (0, 0, 0, 0))
        sh.paste((0, 0, 0, 120), (0, 0), blk.split()[3].point(lambda v: int(v * 0.6)))
        bg.alpha_composite(sh.filter(ImageFilter.GaussianBlur(max(6, bw // 30))), (x + 6, y + 12))
        bg.alpha_composite(blk, (x, y))

    if title:
        draw_title(bg, index, title)
    draw_date(bg, date, date_y)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    bg.convert('RGB').save(out_path, quality=96)
    return out_path


# ---------------- 三段式：模板 / 套模板 ----------------

def render_template(players: list[dict], background: str | None = None,
                    fit: float = 1.0, gap_ratio: float = 0.06,
                    default_placeholder: str | None = None):
    """第一步：背景 + 所有参与者的名片 + 空卡位（占位卡），产出模板。

    players: [{'nick': str, 'header': 名片路径或None}, ...]（顺序即最终排布顺序）
    返回 (模板图, 几何信息 dict)
    """
    bg = load_background(background).convert('RGBA')
    n = len(players)
    cols, rows, bw, hdr_h, card_h, x0, y0, date_y = layout(n, bg, fit, gap_ratio)
    gap = int(bw * gap_ratio)
    block_h = hdr_h + card_h
    for i, p in enumerate(players):
        r, c = divmod(i, cols)
        x = x0 + c * (bw + gap)
        y = y0 + r * (block_h + VGAP)
        blk = build_block(p['nick'], None, bw, hdr_h, card_h, 0, None,
                          p.get('header'), default_placeholder, 'contain')
        bg.alpha_composite(blk, (x, y))
    geom = {'cols': cols, 'rows': rows, 'bw': bw, 'hdr': hdr_h, 'card_h': card_h,
            'x0': x0, 'y0': y0, 'gap': gap, 'vgap': VGAP, 'date_y': date_y,
            'players': [p['nick'] for p in players]}
    return bg, geom


def compose_on_template(tpl: Image.Image, geom: dict, items: list[dict],
                        date: str = '', out_path: str | None = None):
    """第三步：把每个参与者的卡面/表情包贴进模板的对应格子，再画日期。

    items 与 geom['players'] 顺序一致，元素: {'card': path|None, 'placeholder': path|None}
    没有卡也没有表情包 -> 保留模板里自带的占位卡。
    """
    img = tpl.copy().convert('RGBA')
    bw, card_h = geom['bw'], geom['card_h']
    for i, it in enumerate(items):
        if i >= geom['cols'] * geom['rows']:
            break
        if not it.get('card') and not it.get('placeholder'):
            continue
        r, c = divmod(i, geom['cols'])
        x = geom['x0'] + c * (bw + geom['gap'])
        y = geom['y0'] + r * (geom['hdr'] + card_h + geom['vgap']) + geom['hdr']
        if it.get('card'):
            layer = _rounded(Image.open(it['card']).convert('RGB').resize((bw, card_h), Image.LANCZOS),
                             max(3, int(bw * 0.012)))
        else:
            layer = fill_slot(Image.open(it['placeholder']), (bw, card_h),
                              it.get('mode', 'contain'))
        img.alpha_composite(layer, (x, y))
    draw_date(img, date, geom['date_y'])
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        img.convert('RGB').save(out_path, quality=96)
    return img
