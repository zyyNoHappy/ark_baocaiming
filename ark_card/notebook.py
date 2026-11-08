#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""「笔记本」版式（坐标全部来自 报菜名_输出.psd 实测）

画布 2138×1316；20 个卡位 160×304，每行 8 格（左页 4 + 右页 4），
行 y = 116 / 534 / 956；顺序是**跨双页逐行**（第 3 行只有左页 4 格）。
每个卡位上方贴一张名片，右下角盖干员上线日期。
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from .compose import fill_slot

CANVAS = (2138, 1316)

# 20 个卡位（x0, y0, x1, y1），行优先
_ROW_X = [60, 309, 556, 805, 1149, 1398, 1645, 1894]
SLOTS = [(x, 116, x + 159, 419) for x in _ROW_X] \
      + [(x, 534, x + 159, 837) for x in _ROW_X] \
      + [(x, 956, x + 159, 1259) for x in _ROW_X[:4]]

PAGE_L = (60, 964)      # 左页卡位可用范围
PAGE_R = (1149, 2053)   # 右页卡位可用范围
ROW_Y = [116, 534, 956]
CARD_W, CARD_H = 160, 304
COL_GAP = 89

HDR_H = 58            # 名片高度
HDR_PAD = 17          # 名片比卡位左右各宽出多少
HDR_GAP = 16          # 名片与卡位之间的空隙

# ---- 日期印章（坐标取自 PSD 图层边界）----
TOWER_POS = (1670, 941)          # 图层69 塔形
FRAME_POS = (1282, 1165)         # 矩形 898 拷贝：外框
DATE_OUTER = (1282, 1165, 1844, 1266)
SPINE_X, PAPER_RIGHT = 1073, 2054   # 分页中线 / 纸面右缘
BOX_POS = (1561, 1173)           # 矩形 904：月日灰块
YEAR_BOX = (1301, 1180, 247, 74)  # 年份文字区
MMDD_BOX = (1579, 1180, 246, 74)  # 月日文字区（挖空）

FONT_FALLBACK = ['C:/Windows/Fonts/msyhbd.ttc', 'C:/Windows/Fonts/Dengb.ttf']


def _longest_run(flags, min_len: int):
    """flags: 布尔序列 -> 最长连续 True 段 (start, end)；不够长返回 None。"""
    best, s = (0, 0), None
    for i, v in enumerate(list(flags) + [False]):
        if v and s is None:
            s = i
        elif not v and s is not None:
            if i - s > best[1] - best[0]:
                best = (s, i)
            s = None
    return best if best[1] - best[0] >= min_len else None


def page_rect(img: Image.Image) -> tuple[int, int, int, int]:
    """在背景图里找「纸面」矩形 (x, y, w, h)。

    · 背景本身就是纸面（bg_blank.png）→ 返回整张图，页面坐标 1:1
    · 背景是带封面/桌面的整幅插画（bg_scene.png）→ 返回中间那张纸的范围，
      页面坐标按这个矩形缩放对齐，卡位/日期印自动落到纸上。
    判定失败时退回整张图，不会把画面搞乱。
    """
    g = img.convert('L')
    W, H = g.size
    try:
        rows = list(g.resize((1, H), Image.BOX).getdata())
        cols = list(g.resize((W, 1), Image.BOX).getdata())
    except Exception:
        return (0, 0, W, H)

    ry = _longest_run([v > 170 for v in rows], int(H * 0.25))
    rx = _longest_run([v > 170 for v in cols], int(W * 0.25))
    if not ry or not rx:
        return (0, 0, W, H)
    y0, y1 = ry
    x0, x1 = rx
    if (x1 - x0) * (y1 - y0) < W * H * 0.5:    # 找到的“纸面”太小，多半不对
        return (0, 0, W, H)
    return (int(x0), int(y0), int(x1 - x0), int(y1 - y0))


def _blit(base: Image.Image, layer: Image.Image, rect) -> Image.Image:
    """把页面坐标系(chrome.py 的 CANVAS)画好的层，贴到纸面矩形上。"""
    x, y, w, h = rect
    if (w, h) != layer.size:
        layer = layer.resize((max(1, w), max(1, h)), Image.LANCZOS)
    base.alpha_composite(layer, (int(x), int(y)))
    return base


def _font(size: int, path: str | None):
    for p in ([path] if path else []) + FONT_FALLBACK:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def hdr_rect(slot) -> tuple[int, int, int, int]:
    """返回 (x, y, w, h)"""
    x0, y0, x1, _ = slot
    return (x0 - HDR_PAD, y0 - HDR_GAP - HDR_H,
            (x1 - x0 + 1) + 2 * HDR_PAD, HDR_H)


def _rounded(img: Image.Image, r: int) -> Image.Image:
    mask = Image.new('L', img.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, img.size[0] - 1, img.size[1] - 1], r, fill=255)
    out = img.convert('RGBA')
    if img.mode in ('RGBA', 'LA'):
        out.putalpha(ImageChops.multiply(out.getchannel('A'), mask))
    else:
        out.putalpha(mask)
    return out


def _fit_text(draw, txt, font_path, box_w, box_h):
    """字号从高度往下试，返回 (font, bbox) 让整串正好塞进给定区域。"""
    size = box_h
    while size > 10:
        f = _font(size, font_path)
        bb = draw.textbbox((0, 0), txt, font=f)
        if (bb[2] - bb[0]) <= box_w and (bb[3] - bb[1]) <= box_h:
            return f, bb
        size -= 2
    f = _font(size, font_path)
    return f, draw.textbbox((0, 0), txt, font=f)


def draw_date(img: Image.Image, date: str, font_path: str | None = None,
              assets: str | Path | None = None, scale: float = 1.16,
              chrome: bool = True, align: str = 'page'):
    """右下角日期印，复刻 PSD 层次：塔 → 外框 → 年份 → 灰块挖空月日。

    整组（年份 + 灰块）在【矩形898】外框里水平居中；字号按 scale 放大。
    """
    if not date or len(date) < 8:
        return
    a = Path(assets) if assets else Path(__file__).parent.parent / 'assets'

    tower, frame, box = a / 'tower.png', a / 'date_frame.png', a / 'date_box.png'
    if chrome:
        if tower.exists():
            img.alpha_composite(Image.open(tower).convert('RGBA'), TOWER_POS)
        if frame.exists():
            img.alpha_composite(Image.open(frame).convert('RGBA'), FRAME_POS)

    # 水平位置：page = 右页居中；canvas = 整张画布居中
    span = DATE_OUTER[2] - DATE_OUTER[0]
    dx = 0.0
    if align == 'canvas':
        dx = (CANVAS[0] - span) / 2 - DATE_OUTER[0]
    else:
        page_c = (SPINE_X + PAPER_RIGHT) / 2
        dx = (page_c - span / 2) - DATE_OUTER[0]

    year, mmdd = date[:4], date[4:8]
    d0 = ImageDraw.Draw(img)

    box_img = Image.open(box).convert('RGBA') if box.exists() else None
    box_w = box_img.width if box_img else MMDD_BOX[2]
    box_h = box_img.height if box_img else MMDD_BOX[3]

    outer_w = DATE_OUTER[2] - DATE_OUTER[0] - 16
    max_h = DATE_OUTER[3] - DATE_OUTER[1] - 10
    gap = 12
    year_max_w = outer_w - box_w - gap

    # 字号：年份要能塞进左侧、月日要能塞进灰块，取两者都满足的最大值，再放大一档
    best = 10
    for size in range(10, 140):
        fy = _font(size, font_path)
        by = d0.textbbox((0, 0), year, font=fy)
        fm = _font(size, font_path)
        bm = d0.textbbox((0, 0), mmdd, font=fm)
        if (by[2] - by[0]) <= year_max_w and (bm[2] - bm[0]) <= box_w - 16 \
           and (by[3] - by[1]) <= max_h and (bm[3] - bm[1]) <= max_h:
            best = size
    best = int(best * scale)
    while best > 10:
        fy = _font(best, font_path); by = d0.textbbox((0, 0), year, font=fy)
        bm = d0.textbbox((0, 0), mmdd, font=fy)
        if (by[2] - by[0]) <= year_max_w and (bm[2] - bm[0]) <= box_w - 16 \
           and max(by[3] - by[1], bm[3] - bm[1]) <= max_h:
            break
        best -= 2
    f_year = _font(best, font_path)
    f_mmdd = f_year
    bb_year = d0.textbbox((0, 0), year, font=f_year)
    bb_mmdd = d0.textbbox((0, 0), mmdd, font=f_mmdd)

    w_year = bb_year[2] - bb_year[0]
    w_mmdd = bb_mmdd[2] - bb_mmdd[0]
    total = w_year + gap + box_w
    left = DATE_OUTER[0] + dx + ((DATE_OUTER[2] - DATE_OUTER[0]) - total) / 2
    cy = DATE_OUTER[1] + (DATE_OUTER[3] - DATE_OUTER[1]) / 2

    # 年份：#C2C1B8，背景透明
    layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.text((left - bb_year[0], cy - (bb_year[1] + bb_year[3]) / 2),
           year, font=f_year, fill=(0xC2, 0xC1, 0xB8, 255))
    img.alpha_composite(layer)

    if box_img is None:
        return
    # 灰块 + 挖空月日
    bx = int(round(left + w_year + gap))
    by = int(round(cy - box_h / 2))
    hole = Image.new('L', box_img.size, 0)
    hd = ImageDraw.Draw(hole)
    hd.text(((box_w - w_mmdd) / 2 - bb_mmdd[0],
             (box_h - (bb_mmdd[3] - bb_mmdd[1])) / 2 - bb_mmdd[1]),
            mmdd, font=f_mmdd, fill=255)
    box_img.putalpha(ImageChops.subtract(box_img.getchannel('A'), hole))
    img.alpha_composite(box_img, (bx, by))


def date_group(date: str, font_path: str | None, assets, chrome: bool,
               tower: bool) -> Image.Image:
    """把整组日期印（外框 / 塔 / 年份 / 灰块月日）画在页面坐标系里的透明层上。

    之后由 _place_date_group 整体缩放放到右下角，字号自然跟着缩小，
    不用再调 font size。
    """
    a = Path(assets) if assets else Path(__file__).parent.parent / 'assets'
    layer = Image.new('RGBA', CANVAS, (0, 0, 0, 0))
    if chrome:
        items = [('date_frame.png', FRAME_POS)] + ([('tower.png', TOWER_POS)] if tower else [])
        for name, pos in items:
            f = a / name
            if f.exists():
                layer.alpha_composite(Image.open(f).convert('RGBA'), pos)
    if date and len(date) >= 8:
        draw_date(layer, date, font_path, assets=a, chrome=False, align='page')
    return layer


def _place_date_group(img: Image.Image, layer: Image.Image, rect, ref=None) -> None:
    """把日期印整组等比缩放到 rect 里并贴上。

    缩放用固定的参考框 ref（外框原始范围）算，而不是图层内容的外接框 ——
    这样「只有外框」和「只有文字」两张图层会落在完全一致的位置上，
    模板和帧才能对齐。
    """
    rx0, ry0, rx1, ry1 = ref or DATE_REF
    x0, y0, x1, y1 = rect
    k = min((x1 - x0) / (rx1 - rx0), (y1 - y0) / (ry1 - ry0))
    w, h = max(1, int((rx1 - rx0) * k)), max(1, int((ry1 - ry0) * k))
    g = layer.crop((int(rx0), int(ry0), int(rx1), int(ry1)))
    img.alpha_composite(g.resize((w, h), Image.LANCZOS),
                        (int(x0 + ((x1 - x0) - w) / 2), int(y0 + ((y1 - y0) - h) / 2)))


def compose_notebook(entries: list[dict], out_path: str, background: str,
                     date: str = '', font_path: str | None = None,
                     placeholder_mode: str = 'cover', chrome: bool = True,
                     align: str = 'page') -> str:
    """entries 与 SLOTS 顺序一一对应；多出的忽略。"""
    if background and Path(background).exists():
        base = Image.open(background).convert('RGBA')
    else:
        base = Image.new('RGBA', CANVAS, (28, 30, 34, 255))
    rect = page_rect(base)

    # 先按页面坐标系(2138×1316)画在透明层上，再整体贴到背景的纸面矩形
    img = Image.new('RGBA', CANVAS, (0, 0, 0, 0))

    for e, g in zip(entries, layout_for(len(entries))):
        hx, hy, hw, hh = g['hdr']
        sx, sy, cw, ch = g['card']

        # 名片
        nick = e.get('nick') or ''
        hdr = None
        if e.get('header') and Path(e['header']).exists():
            src = Image.open(e['header']).convert('RGB')
            # 整张名片缩放到刚好装进名片条，不裁任何内容
            k = min(hw / src.width, hh / src.height)
            nw, nh = max(1, int(round(src.width * k))), max(1, int(round(src.height * k)))
            hdr = Image.new('RGBA', (hw, hh), (0, 0, 0, 0))
            hdr.alpha_composite(src.resize((nw, nh), Image.LANCZOS).convert('RGBA'),
                                ((hw - nw) // 2, (hh - nh) // 2))
            hdr = hdr.filter(ImageFilter.UnsharpMask(radius=1.2, percent=140,
                                                     threshold=2)).convert('RGBA')
        elif nick:
            hdr = Image.new('RGBA', (hw, hh), (233, 231, 226, 255))
            d = ImageDraw.Draw(hdr)
            f = _font(max(12, int(hh * 0.52)), None)
            tw = d.textlength(nick, font=f)
            d.text(((hw - tw) / 2, (hh - f.size) / 2 - hh * 0.04), nick,
                   font=f, fill=(64, 62, 58, 255))
        # 空位：没有名片也没有昵称 -> 什么都不画，留白
        if hdr is not None:
            img.alpha_composite(_rounded(hdr, 6), (hx, hy))

        # 卡位：等比缩放塞进格子，绝不拉伸
        layer = None
        if e.get('card') and Path(e['card']).exists():
            src = Image.open(e['card']).convert('RGB')
            k = ch / src.height                      # 以高度为准，保持原比例
            nw = max(1, int(round(src.width * k)))
            src = src.resize((nw, ch), Image.LANCZOS)
            if nw > cw:                              # 比格子宽：居中裁掉两边
                off = (nw - cw) // 2
                layer = _rounded(src.crop((off, 0, off + cw, ch)), 6)
            else:                                    # 比格子窄：居中留一点边
                layer = Image.new('RGBA', (cw, ch), (0, 0, 0, 0))
                layer.alpha_composite(_rounded(src, 6), ((cw - nw) // 2, 0))
        elif e.get('placeholder') and Path(e['placeholder']).exists():
            layer = fill_slot(Image.open(e['placeholder']), (cw, ch),
                              e.get('mode', placeholder_mode))
        if layer is not None:
            img.alpha_composite(layer.convert('RGBA'), (sx, sy))

    if len(entries) < 20:
        # 人少：整组日期印缩小，压到右下角
        layer = date_group(date, font_path, None, chrome, tower=False)
        _place_date_group(img, layer, DATE_TARGET)
    else:
        draw_date(img, date, font_path, chrome=chrome, align=align)
    _blit(base, img, rect)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    base.convert('RGB').save(out_path, quality=96)
    return out_path


def build_notebook_template(background: str, out_path: str,
                            assets: str | Path | None = None,
                            tower: bool = True) -> str:
    """出模板：底图 + 塔（已避让外框）+ 日期外框，不含名片/卡面/日期数字。

    之后每帧都在这张模板上只贴 名片 + 卡面 + 日期数字，保证版式完全一致。
    """
    a = Path(assets) if assets else Path(__file__).parent.parent / 'assets'
    if background and Path(background).exists():
        base = Image.open(background).convert('RGBA')
    else:
        # 没放背景图也能跑：给一张素纸，版式照常
        base = Image.new('RGBA', CANVAS, (247, 246, 243, 255))
    rect = page_rect(base)

    # 人少时日期印要压在右下角，模板里就按目标区域画好，保证每帧一致
    if DATE_TARGET is not None:
        layer = date_group('', None, a, chrome=True, tower=tower)
        tmp = Image.new('RGBA', CANVAS, (0, 0, 0, 0))
        _place_date_group(tmp, layer, DATE_TARGET)
        layer = tmp
    else:
        layer = Image.new('RGBA', CANVAS, (0, 0, 0, 0))
        items = [('date_frame.png', FRAME_POS)] + ([('tower.png', TOWER_POS)] if tower else [])
        for name, pos in items:
            f = a / name
            if f.exists():
                layer.alpha_composite(Image.open(f).convert('RGBA'), pos)
    _blit(base, layer, rect)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    base.convert('RGB').save(out_path, quality=96)
    return out_path


# ---- 按人缩放用的比例（全部相对卡宽 160）----
R_HDR = 58 / 160        # 名片高
R_PAD = 17 / 160        # 名片左右各宽出
R_GAP = 16 / 160        # 名片与卡位间距
R_CARD = 304 / 160      # 卡位高/宽
R_COL = 72 / 160        # 列间距（收紧，卡片可以更大）
R_ROW = 24 / 160        # 行间距（收紧）
R_CELL = R_HDR + R_GAP + R_CARD

DATE_MARGIN = 30        # 卡位和日期框之间至少留这么多，贴着会显得挤
PAGE_TOP = 72           # 纸面上沿到第一排卡片的留白（原来 116，太厚）
PAGE_BOT = 1240         # 日期印已经挪到右下角，最后一排可以贴到底
PAGE_H = PAGE_BOT - PAGE_TOP

EDGE_MARGIN = 103       # 左右外边距（最左/最右那列离纸边的距离）
CARD_ASPECT = 3.205     # 名片原图 宽/高，用来自适应名片条高度
HDR_PAD_RATIO = 0.20    # 名片条每边比卡片宽出这么多（同时就是贴边时的内缩量）

# 日期印整组要缩到右下角这块区域里（页面坐标系）
DATE_TARGET = (1666, 1250, 2128, 1314)
DATE_REF = (DATE_OUTER[0], DATE_OUTER[1],
            FRAME_POS[0] + 562, FRAME_POS[1] + 101)   # 外框原始范围


def _hdr_ratios(n: int):
    """名片条高 / 左右外扩 / 与卡位的间距（相对卡宽）。

    20 人时沿用 PSD 实测比例；人少（<=12）时空间富余，名片条做得更大，
    并且让整张名片刚好铺满名片条 —— 不裁图，只是把它放大。
    名片原图约 3.2:1，所以条宽 = 1+2*pad 倍卡宽时，条高 = 条宽/3.205。
    """
    if n == 20:
        return 58 / 160, 17 / 160, 16 / 160
    pad = 0.245
    return (1 + 2 * pad) / 3.205, pad, 0.10


def _cell(cw: float, x: float, y: float, hr: float, pr: float, gr: float) -> dict:
    """给定卡宽 cw 与单元格左上角，算出名片与卡位的矩形。"""
    hdr_h, pad, gap = cw * hr, cw * pr, cw * gr
    card_h = cw * R_CARD
    cy = y + gap + hdr_h
    return {'card': (int(round(x)), int(round(cy)), int(round(cw)), int(round(card_h))),
            'hdr': (int(round(x - pad)), int(round(y)),
                    int(round(cw + 2 * pad)), int(round(hdr_h)))}


def _row_cells(k: int, x0: float, x1: float, y: float, cw: float,
               hr: float, pr: float, gr: float, margin: float) -> list:
    """一整行：左右两边固定留 margin，卡片之间的空档等宽。

    最左列贴左、最右列贴右 —— 只剩边距那么宽；中间 (k-1) 条空档均分剩余空间。
    """
    if k <= 0:
        return []
    gap = (x1 - x0 + 1 - 2 * margin - k * cw) / (k - 1) if k > 1 else 0
    return [_cell(cw, x0 + margin + i * (cw + gap), y, hr, pr, gr) for i in range(k)]


def _fit_cols(R: int, C: int, margin: float):
    """把卡片撑到最大：名片条外沿贴齐纸边，中间空档等宽。

    名片条比卡片宽出 pad*cw，所以「贴边」贴的是名片条的外沿 ——
    卡片自身内缩 pad 那么多，正好和参考图一致。
    卡片越宽，需要的外边距（= pad*cw）和空档也越大，所以逐个卡宽试算取最大。
    """
    cols = 2 * C
    span = PAGE_R[1] - PAGE_L[0] + 1
    pad = HDR_PAD_RATIO
    best = None
    cw = 60.0
    while cw < 420:
        edge = pad * cw                       # 名片条越出卡片的部分 = 外边距
        if cols > 1:
            gap = (span - cols * cw - 2 * edge) / (cols - 1)
        else:
            gap = 0.0
        ok_gap = gap >= 6.0
        ok_hdr = cols <= 1 or gap >= (2 * edge - 8)    # 相邻名片条不重叠
        if ok_gap and ok_hdr:
            hr = (1 + 2 * pad) / CARD_ASPECT           # 名片条刚好装下整张名片
            cell = cw * (hr + R_GAP + R_CARD)
            total = R * cell + (R - 1) * cw * R_ROW
            if total <= PAGE_H:
                best = (cw, gap, hr, pad, edge)
        cw += 0.5
    return best


def _pick_grid(n: int):
    """挑一个让卡最大的排法：空位最少优先，其次卡片最大。"""
    margin = EDGE_MARGIN
    best = None
    for R in range(1, 4):
        for C in range(1, 5):
            cap = 2 * R * C
            if cap < n:
                continue
            fit = _fit_cols(R, C, margin)
            if not fit:
                continue
            cw, gap, hr, pad, edge = fit
            key = (cap - n, -round(cw, 2))
            if best is None or key < best[0]:
                best = (key, R, C, cw, gap, hr, pad, edge)
    if best is None:
        raise ValueError('排不下：人数或版面参数不合理')
    _, R, C, cw, gap, hr, pad, edge = best
    return R, C, cw, gap, hr, pad, edge


def layout_for(n: int) -> list:
    """按人数排格子，人越少卡片越大。

    20 人 → 沿用 PSD 实测坐标（4 列 × 3 行，8/8/4）
    其余 → 自动挑「空位最少、卡片最大」的行列数，卡片等比放大
      12 人 → 2 行 × 每页 3
       8 人 → 2 行 × 每页 2
       6 人 → 1 行 × 每页 3
    """
    if n == 20:
        geom = []
        for s in SLOTS:
            x0, y0, x1, y1 = s
            hx, hy, hw, hh = hdr_rect(s)
            geom.append({'card': (x0, y0, x1 - x0 + 1, y1 - y0 + 1),
                         'hdr': (hx, hy, hw, hh)})
        return geom

    R, C, cw, gap, hr, pad, edge = _pick_grid(n)
    cell_h = cw * (hr + R_GAP + R_CARD)
    # 第一排贴上、最后一排贴下，中间的行均分
    step = (PAGE_H - cell_h) / (R - 1) if R > 1 else 0

    base, extra = divmod(n, R)
    out = []
    for r in range(R):
        k = base + (1 if r < extra else 0)
        if k <= 0:
            continue
        y = PAGE_TOP + r * step
        out += _row_cells(k, PAGE_L[0], PAGE_R[1], y, cw, hr, pad, R_GAP, edge)
    return out


def tower_needed(n: int) -> bool:
    """塔形放右下角；如果最后一行的卡片会压到它，就别画（被挡住反而碍眼）。"""
    if n == 20:
        return True
    tx0, ty0, tx1, ty1 = TOWER_POS[0], TOWER_POS[1], TOWER_POS[0] + 413, TOWER_POS[1] + 341
    for g in layout_for(n):
        x, y, w, h = g['card']
        if x < tx1 and x + w > tx0 and y < ty1 and y + h > ty0:
            return False
    return True
