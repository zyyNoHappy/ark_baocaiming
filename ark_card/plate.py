#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""干员列表截图 -> 逐张卡片（不写死分辨率 / 版式）

四类老毛病对应的处理：
  【左侧/顶部白边】 卡位坐标从"卡与卡之间的缝"推出来，最后再 trim_page 削一遍页面底色
  【名字显示不全】  卡的上下边界按"底部深色名条"定位，名条整条包含在内
  【底部多出一条】  卡底取名条下沿，再往下那圈是投影，不要

流程：
  1. 页面底色 = 量化众数色；与底色差 < TOL 的像素算"空白"
  2. 列间隙：某列存在很长的连续空白段（穿行而过），这些列成段出现即卡缝
  3. 卡宽 = 相邻缝隙的"缝到缝"距离（取中位数）；卡位左沿 = 上一条缝的右沿
  4. 行：纵向剖面上 < 82 的连续段就是卡片底部的深色名条，其下沿即卡底
  5. 卡高 = 卡底 - 卡顶（卡顶取名条上方最强的"由亮转暗"边）
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

BG_TOL = 26            # 与页面底色的最大通道差
BANNER_DARK = 82       # 名条的判定亮度
COL_EMPTY_RATIO = 0.55  # 某列连续空白段超过全高的这个比例 -> 是卡缝
TRIM_LIMIT = 26        # trim_page 最多削多少像素
EDGE_SLACK = 12        # 贴边那一列允许超出画面多少像素（截图常在卡边切断）


@dataclass
class Grid:
    cols: int
    rows: int
    card_w: int
    card_h: int
    boxes: list = field(default_factory=list)      # (x, y, w, h)
    edge_gaps: list = field(default_factory=list)  # 每张卡离画面边缘的距离


# ------------------------------------------------------------------ 基础
def _page_color(arr: np.ndarray) -> np.ndarray:
    """量化到 16 级后的众数颜色，当作页面底色。"""
    q = (arr // 16 * 16).astype(np.int16).reshape(-1, 3)
    keys = (q[:, 0].astype(np.int32) << 16) | (q[:, 1].astype(np.int32) << 8) | q[:, 2]
    vals, counts = np.unique(keys, return_counts=True)
    k = int(vals[counts.argmax()])
    return np.array([(k >> 16) & 255, (k >> 8) & 255, k & 255], dtype=np.float32) + 8.0


def _runs(mask, min_len: int = 1):
    out, s = [], None
    for i, v in enumerate(mask):
        if v and s is None:
            s = i
        elif not v and s is not None:
            if i - s >= min_len:
                out.append((s, i))
            s = None
    if s is not None and len(mask) - s >= min_len:
        out.append((s, len(mask)))
    return out


def _longest_col_run(empty: np.ndarray) -> np.ndarray:
    """每一列「最长连续空白」的长度。"""
    res = np.zeros(empty.shape[1], dtype=np.int32)
    cur = np.zeros(empty.shape[1], dtype=np.int32)
    for y in range(empty.shape[0]):
        cur = np.where(empty[y], cur + 1, 0)
        np.maximum(res, cur, out=res)
    return res


# ------------------------------------------------------------------ 边界微调
def _is_page(line: np.ndarray, mean_min: float = 195.0, std_max: float = 30.0) -> bool:
    """一行/一列像素是不是"页面底色"（含卡片边缘的浅色投影）。"""
    return float(line.mean()) > mean_min and float(line.std()) < std_max


def trim_page(card: Image.Image, limit: int = TRIM_LIMIT) -> Image.Image:
    """削掉卡片四周残留的页面底色。

    左右放宽（卡边常带几像素浅色投影，实测右侧最多 8 列），
    上下收紧（卡顶是硬边、卡底是名条，几乎不会有多余的页面色）。
    """
    a = np.asarray(card.convert('RGB')).astype(np.float32)
    h, w, _ = a.shape
    top, bot, left, right = 0, h, 0, w

    lim_v = min(limit, 14)
    while left < lim_v and left < right - 4 and _is_page(a[top:bot, left]):
        left += 1
    while right > w - lim_v and right - 1 > left + 3 and _is_page(a[top:bot, right - 1]):
        right -= 1

    lim_h = min(limit, 8)
    while bot > h - lim_h and bot - 1 > top + 3 and _is_page(a[bot - 1, left:right], 232.0, 45.0):
        bot -= 1

    # 顶部可能残留若干行页面底色，分两档处理：
    # ① 纯页面行（很白且完全均匀）—— 绝不可能是卡面，有多少削多少
    cap_top = min(24, bot - 4)
    while top < cap_top:
        line = a[top, left:right]
        if float(line.mean()) > 240 and float(line.std()) < 8:
            top += 1
        else:
            break
    # ② 浅色台阶（不够白，但和下一行落差明显），最多再削 lim_h 行
    stepped = 0
    while stepped < lim_h and top + 1 < bot - 4:
        m0 = float(a[top, left:right].mean())
        m1 = float(a[top + 1, left:right].mean())
        if m0 > 215 and (m0 - m1) > 18:
            top += 1
            stepped += 1
        else:
            break

    if (top, bot, left, right) == (0, h, 0, w):
        return card
    return card.crop((left, top, right, bot))


# ------------------------------------------------------------------ 主流程
def _column_gaps(empty: np.ndarray) -> list[tuple[int, int]]:
    h = empty.shape[0]
    lr = _longest_col_run(empty)
    return _runs(lr > COL_EMPTY_RATIO * h, min_len=4)


def _insets(empty: np.ndarray, x: int, y: int, cw: int, ch: int) -> tuple[int, int]:
    """卡片左右两侧夹着的页面底色条有多宽。

    只看卡面部分（名条那一段是黑底，会把整列的均值拉下来，所以不能整列统计）。
    """
    x0, x1 = max(0, x), min(empty.shape[1], x + cw)
    a = empty[y + 6: y + max(12, int(ch * 0.72)), x0:x1]
    if a.size == 0:
        return 0, 0
    n = a.shape[1]
    frac = a.mean(axis=0)
    cap = n // 5
    left = 0
    while left < cap and frac[left] > 0.5:
        left += 1
    right = 0
    while right < cap and frac[n - 1 - right] > 0.5:
        right += 1
    if left + right >= n - 40:      # 整张都像底色 -> 判定不可靠，不缩
        return 0, 0
    # 卡片被画面切掉的部分算作内缩，免得越界
    return left + max(0, -x), right + max(0, x + cw - empty.shape[1])


def _pitch(gaps: list[tuple[int, int]]) -> float:
    """相邻卡缝的间距（取出现最多的那个差值，抗个别缝漏检）。"""
    deltas = [gaps[i + 1][0] - gaps[i][0] for i in range(len(gaps) - 1)]
    if not deltas:
        return 0.0
    best, best_cnt = None, 0
    for d in deltas:
        cnt = sum(1 for e in deltas if abs(e - d) <= 3)
        if cnt > best_cnt:
            best, best_cnt = d, cnt
    near = [e for e in deltas if abs(e - best) <= 3]
    return sum(near) / len(near)


def _tops(prof: np.ndarray) -> list[int]:
    """卡顶：由页面(很亮)掉到卡面。上面还得确实是连续一截页面（排除工具栏下沿）。"""
    out = []
    for y in range(12, len(prof) - 1):
        if prof[y] > 232 and prof[y + 1] < 215:
            if float(np.median(prof[y - 12:y - 1])) > 225:
                out.append(y)
    return out


def _bottoms(prof: np.ndarray) -> list[int]:
    """卡底：深色名条之后突然变亮，且下面持续是亮的（投影/页面）。

    名条里的白色干员名会让剖面中途变亮，所以必须要求「持续」，否则会把
    名字那一行误当成卡底，切出来的卡片名字就被削掉半截。
    """
    out = []
    n = len(prof)
    for y in range(20, n - 26):
        if prof[y] > 100 or prof[y + 1] < 105:
            continue
        if prof[y + 1] - prof[y] < 25:
            continue
        if float(np.median(prof[y - 20:y - 2])) > 95:          # 上面是名条(暗)
            continue
        if float(np.percentile(prof[y + 1:y + 26], 20)) < 105:  # 下面持续亮
            continue
        # 名条下沿常是斜的，剖面会在真正下沿前先翘起来一点；
        # 在候选点附近取最暗的那一行，它才是名条的最低处
        lo, hi = y, min(n - 1, y + 13)
        y = lo + int(np.argmin(prof[lo:hi + 1]))
        out.append(y + 1)
    return out


def _bands(prof: np.ndarray, thr: float = 232.0) -> list[tuple[int, int]]:
    """纵向剖面上的「亮带」（行与行之间那条页面），亮带结束处就是下一行卡片的顶。"""
    out, s = [], None
    for y, v in enumerate(prof):
        if v > thr and s is None:
            s = y
        elif v <= thr and s is not None:
            if y - s >= 6:
                out.append((s, y))
            s = None
    if s is not None and len(prof) - s >= 6:
        out.append((s, len(prof)))
    return out


def _card_height(tops, bots, cw: int) -> int:
    """卡高：优先用「卡顶-卡底」的实测比例，取出现最多的一档。"""
    ratios = [(b - t) / cw for t in tops for b in bots if 1.9 * cw <= b - t <= 2.1 * cw]
    if not ratios:
        return int(round(cw * 1.99))
    ratios.sort()
    best, cnt = ratios[0], 0
    for r in ratios:
        c = sum(1 for e in ratios if abs(e - r) <= 0.02)
        if c > cnt:
            best, cnt = r, c
    return int(round(cw * best))


def _row_starts(tops, bots, card_h: int, h: int, prof: np.ndarray) -> list[int]:
    """把卡顶/卡底两种线索合成行首，并要求该行确实带着底部名条。

    实测卡顶（页面→卡面的硬边）比「卡底反推」可靠得多：反推依赖卡底判得准，
    偏几像素就会把整行抬上去、把名字切掉。所以同一处冲突时优先实测卡顶。
    """
    cand = [(y, 0) for y in tops] + [(b - card_h, 1) for b in bots]
    cand = [(y, p) for y, p in cand if 0 <= y and y + card_h <= h]
    cand.sort()
    merged: list[list[int]] = []
    for y, p in cand:
        if merged and y - merged[-1][0] < max(8, card_h // 6):
            if p < merged[-1][1]:          # 更可信的线索，替换掉
                merged[-1] = [y, p]
            continue
        merged.append([y, p])

    rows = []
    for y, _ in merged:
        lo, hi = y + card_h - 34, y + card_h - 4
        # 名条是深色，卡底那一截应该偏暗；阈值放宽到 125 是因为
        # 有些界面的名条底色偏灰（实测 111），卡得过死会整行被丢掉
        if lo >= 0 and hi > lo and float(prof[lo:hi].mean()) < 125:
            rows.append(y)

    # 去掉「和下一行重叠」的候选：窗口标题栏/工具栏有时会被判成一行，
    # 它的位置和真正第一行只差一两百像素（小于卡高），真行不会这样。
    rows.sort()
    clean = []
    for i, y in enumerate(rows):
        nxt = rows[i + 1] if i + 1 < len(rows) else None
        if nxt is not None and nxt - y < card_h * 0.8:
            continue
        clean.append(y)
    return clean


def _panel_mask(arr: np.ndarray) -> np.ndarray:
    """每一列是不是「侧栏」（整列又暗又均匀）。

    端游窗口截图两侧常挂深色侧栏（收起/职业筛选），侧栏里露出的半张卡不能切。
    """
    gray = arr.mean(axis=2)
    return (gray < 62).mean(axis=0) > 0.72


def crop_cards(img: Image.Image, debug: bool = False):
    """返回 (cards, Grid)。cards 顺序：逐行、从左到右。"""
    rgb = img.convert('RGB')
    arr = np.asarray(rgb).astype(np.float32)
    h, w, _ = arr.shape

    bg = _page_color(arr)
    empty = np.abs(arr - bg).max(axis=2) < BG_TOL
    panel = _panel_mask(arr)

    gaps = _column_gaps(empty)
    if len(gaps) < 2:
        raise ValueError('没找到列间隙：确认这是干员列表截图，且卡与卡之间有背景缝')
    gap_w = int(np.median([b - a for a, b in gaps]))
    cw = int(np.median([gaps[i + 1][0] - gaps[i][1] for i in range(len(gaps) - 1)]))
    pt = _pitch(gaps)
    if cw <= 40 or pt <= cw:
        raise ValueError(f'卡宽推算异常(cw={cw}, pitch={pt:.0f})')

    # 卡位左沿：从第一条缝的右沿按间距向两边铺开（漏检的缝也能补齐）
    base = gaps[0][1]
    xs = []
    k = 0
    while True:
        x = int(round(base - k * pt))
        if x + cw <= 0 or k > 40:
            break
        xs.append(x)
        k += 1
    k = 1
    while True:
        x = int(round(base + k * pt))
        if x >= w or k > 40:
            break
        xs.append(x)
        k += 1
    # 贴边那一列允许超出画面 EDGE_SLACK 像素：横屏截图常在卡边切断，
    # 超出的那几像素正好被右侧内缩削掉，削完仍是完整一张卡。
    # 但这一列不能有太多落在侧栏里 —— 侧栏里那半张卡不进。
    def _usable(x: int) -> bool:
        if max(-x, x + cw - w) > EDGE_SLACK:
            return False
        x0, x1 = max(0, x), min(w, x + cw)
        return x1 > x0 and float(panel[x0:x1].mean()) <= 0.35

    xs = sorted({x for x in xs if _usable(x)})
    if not xs:
        raise ValueError('卡位没落在画面内')

    # 纵向剖面：取中间几列平均，抗单张卡的个性化内容
    mid = xs[max(0, len(xs) // 2 - 1): len(xs) // 2 + 1] or xs
    prof = np.mean([arr[:, max(0, x):min(w, x + cw)].mean(axis=(1, 2)) for x in mid], axis=0)

    tops = _tops(prof)
    bots = _bottoms(prof)
    card_h = _card_height(tops, bots, cw)
    # 卡顶的另一种线索：行间那条亮带（页面）结束的位置。渐变过渡时
    # 单步下降检测不到，但亮带边缘很稳。
    tops2 = tops + [b for _, b in _bands(prof)]
    rows = _row_starts(tops2, bots, card_h, h, prof)
    if not rows:
        raise ValueError('没定位到卡片行：截图可能不是列表页，或卡片被裁掉了')

    # 每行单独定高度：两行的卡高偶尔差几像素，共用同一个值会把某行的名字削掉。
    # 但只有和整图卡高接近（±6px）的测量才可信，差太多说明卡底判歪了，退回整图值。
    heights = []
    for y in rows:
        cand = [b - y for b in bots if 1.8 * cw <= b - y <= 2.2 * cw]
        if cand:
            hh = min(cand, key=lambda x: abs(x - card_h))
            heights.append(hh if abs(hh - card_h) <= 6 else card_h)
        else:
            heights.append(card_h)

    # 内缩：同一张截图里所有卡的几何是一样的，所以按整图取中位数，
    # 避免个别卡片内容偏亮导致它自己被削窄
    ls, rs = [], []
    for y, ch in zip(rows, heights):
        for x in xs:
            l, r = _insets(empty, x, y, cw, ch)
            ls.append(l)
            rs.append(r)
    il = int(np.median(ls)) if ls else 0
    ir = int(np.median(rs)) if rs else 0
    if il + ir >= cw - 40:
        il = ir = 0

    boxes, edge_gaps = [], []
    for y, ch in zip(rows, heights):
        for x in xs:
            bx0 = max(0, x + il)
            bx1 = min(w, x + cw - ir)
            boxes.append((bx0, y, bx1 - bx0, ch))
            edge_gaps.append(min(bx0, w - 1 - bx1))

    grid = Grid(cols=len(xs), rows=len(rows), card_w=cw,
                card_h=int(np.median(heights)) if heights else card_h,
                boxes=boxes, edge_gaps=edge_gaps)
    cards = [trim_page(rgb.crop((x, y, x + bw, y + bh))) for x, y, bw, bh in boxes]
    if debug:
        print(f'    gap_w={gap_w} cw={cw} pitch={pt:.0f} tops={tops} bots={bots} '
              f'card_h={card_h} rows={rows} heights={heights} -> {len(cards)} 张')
    return cards, grid


def crop_file(path: str, debug: bool = False):
    return crop_cards(Image.open(path), debug=debug)
