#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
明日方舟「干员列表」截图 -> 逐张卡片切图

不写死分辨率 / 版式:
  1. 卡与卡之间是纯背景缝隙, 竖直边缘能量沿 x 周期重复 -> 自相关求列距(取基频)
  2. 卡片上下是背景, 横向边缘能量同样求行距
  3. 卡面素材比例固定(高/宽 = 393/192), 量出卡高即可反推卡宽
  4. 用"顶边框那一行的跃变起点"定位第一张卡的左边缘
  5. 只保留完整落在画面内的卡片(被 UI 挡住的残卡自动丢弃)
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from PIL import Image

ASPECT = 393 / 192.0          # 卡高 / 卡宽


@dataclass
class Grid:
    pitch_x: float
    pitch_y: float
    x0: int
    y0: int
    card_w: int
    card_h: int
    cols: int
    rows: int

    def boxes(self, img_w: int, img_h: int) -> list[tuple[int, int, int, int]]:
        out = []
        for r in range(self.rows):
            for c in range(self.cols):
                x = self.x0 + c * self.pitch_x
                y = self.y0 + r * self.pitch_y
                if x < 0 or y < 0:
                    continue
                if x + self.card_w > img_w or y + self.card_h > img_h:
                    continue
                out.append((int(round(x)), int(round(y)), self.card_w, self.card_h))
        return out


def _gray(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert('RGB')).astype(float).mean(axis=2)


def _step_profile(g: np.ndarray, axis: int, thr: int = 18) -> np.ndarray:
    d = np.abs(np.diff(g, axis=axis)) > thr
    prof = d.sum(axis=0 if axis == 1 else 1).astype(float)
    return np.concatenate([[0], prof])


def _fundamental(sig: np.ndarray, lo: int, hi: int) -> tuple[float, float]:
    """自相关取基频: 不取全局最大(可能命中 2 倍周期), 取 >=90% 最大值的最小滞后"""
    s = np.convolve(sig, np.ones(2) / 2, mode='same')
    s = s - s.mean()
    n = len(s)
    corr = {}
    for lag in range(max(2, lo), min(hi, n - 10)):
        a, b = s[:n - lag], s[lag:]
        corr[lag] = float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))
    if not corr:
        raise ValueError('无法估计卡片间距: 图里可能没有干员列表')
    mx = max(corr.values())
    lag = min(l for l, c in corr.items() if c >= 0.90 * mx)
    for d in (-2, -1, 1, 2):
        if lag + d in corr and corr[lag + d] > corr[lag]:
            lag += d
    return float(lag), corr[lag]


def detect(img: Image.Image) -> Grid:
    g = _gray(img)
    h, w = g.shape

    ex = _step_profile(g, 1)
    pitch_x, _ = _fundamental(ex, 60, 520)

    ey = _step_profile(g, 0)
    pitch_y, _ = _fundamental(ey, 80, 800)
    pitch_y = max(pitch_y, pitch_x * 1.4)

    # 卡片顶边框所在行
    rowhits = np.concatenate([[0], (np.abs(np.diff(g, axis=0)) > 18).sum(axis=1).astype(float)])
    top_zone = range(int(h * 0.03), int(h * 0.30))
    y_top = max(top_zone, key=lambda y: rowhits[y])
    y_top = max(range(max(1, y_top - 2), min(h - 1, y_top + 3)), key=lambda y: rowhits[y])

    # 沿这条顶边框, 最左边出现跃变的位置 = 卡框左上角
    drow = np.abs(g[y_top, :] - g[y_top - 1, :]) > 18
    xs = [x for x in range(1, int(pitch_x * 0.9)) if drow[x]]
    x_edge = xs[0] if xs else 0

    # 卡底边所在行
    est = pitch_x * 0.893 * ASPECT
    lo, hi = int(y_top + est * 0.90), int(y_top + est * 1.03)
    y_bot = lo + int(np.argmax(rowhits[lo:hi]))

    card_h = y_bot - y_top
    card_w = card_h / ASPECT

    off = max(2, int(round(pitch_x * 0.023)))     # 左边框内缩, 与示例切法一致
    x0 = x_edge - off
    y0 = y_top

    cols = int((w - x0) // pitch_x)
    rows = 0
    while y0 + rows * pitch_y + card_h <= h:
        rows += 1
    if cols <= 0 or rows <= 0:
        raise ValueError('没找到完整的卡片, 请确认截图是干员列表页')
    return Grid(pitch_x, pitch_y, x0, y0, int(round(card_w)), int(round(card_h)), cols, rows)


def crop_image(img: Image.Image) -> tuple[list[Image.Image], Grid]:
    grid = detect(img)
    boxes = grid.boxes(img.width, img.height)
    return [img.crop((x, y, x + w, y + h)) for (x, y, w, h) in boxes], grid


def crop_file(path: str) -> tuple[list[Image.Image], Grid]:
    return crop_image(Image.open(path))
