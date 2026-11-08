#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ark_card —— 明日方舟「报菜名」卡片切图 / 合成核心库

    from ark_card import crop_file, build_sheet, compose_frame
"""
from .grid import Grid, detect, crop_file, crop_image
from .sheet import build_sheet, name_strip, parse_names
from .compose import compose_frame, load_background, default_background
from .profile import crop_profile_card, pick_box, BOX_WIDE, BOX_169

__all__ = [
    'Grid', 'detect', 'crop_file', 'crop_image',
    'build_sheet', 'name_strip', 'parse_names',
    'compose_frame', 'load_background', 'default_background',
    'crop_profile_card', 'pick_box', 'BOX_WIDE', 'BOX_169',
]
__version__ = '1.0.0'
