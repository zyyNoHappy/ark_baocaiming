#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从游戏「个人信息」整屏截图里裁出名片块。

同一个游戏在不同设备/模拟器上，整屏截图的**宽高比不同**，名片块在画面里的
相对位置也跟着变（手机 19.5:9 与 16:9 就不一样）。所以按宽高比选预设：

  · 宽高比 ≥ 2.0（如 2292×1080、2400×1080）→ BOX_WIDE
  · 其它（如 1920×1080）                    → BOX_169

两套坐标都是「左上＝等级徽章外缘、右下＝名片板圆角、右侧黑边不要」量出来的。
遇到没量过的设备，把整屏图发我，补一行预设即可。
"""
from __future__ import annotations

from PIL import Image

# 相对整屏的 (x0, y0, x1, y1)
BOX_WIDE = (965 / 2292, 60 / 1080, 1888 / 2292, 348 / 1080)   # 2292×1080 实测
BOX_169 = (1013 / 1920, 115 / 1080, 1788 / 1920, 357 / 1080)  # 1920×1080 实测

CARD_ASPECT = 3.20          # 名片 宽/高，用来判断是不是已经裁过了


def looks_like_card(img: Image.Image, tol: float = 0.35) -> bool:
    return abs(img.width / img.height - CARD_ASPECT) < tol


def pick_box(img: Image.Image) -> tuple[float, float, float, float]:
    return BOX_WIDE if (img.width / img.height) >= 2.0 else BOX_169


def crop_profile_card(img: Image.Image, box=None, force: bool = False) -> Image.Image:
    """整屏个人信息截图 -> 名片块。已经是名片比例的图（并且没要求 force）原样返回。"""
    if not force and looks_like_card(img):
        return img
    x0, y0, x1, y1 = box or pick_box(img)
    w, h = img.size
    return img.crop((int(round(w * x0)), int(round(h * y0)),
                     int(round(w * x1)), int(round(h * y1))))
