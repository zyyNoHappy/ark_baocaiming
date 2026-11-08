#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""命令行入口

    python -m ark_card crop    <截图或目录> ... --out cards/
    python -m ark_card compose <卡片目录>      --order order.txt --out frames/ [--bg bg.png]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

from . import compose as C
from . import grid as G
from . import sheet as S

IMG_EXT = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}


def _iter_images(paths: list[str]):
    for p in paths:
        p = Path(p)
        if p.is_dir():
            for f in sorted(p.rglob('*')):
                if f.suffix.lower() in IMG_EXT:
                    yield f
        elif p.suffix.lower() in IMG_EXT:
            yield p


def cmd_crop(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    all_cards, total = [], 0
    for shot in _iter_images(args.shots):
        cards, g = G.crop_file(str(shot))
        stem = shot.stem[:10]
        for i, card in enumerate(cards):
            name = f'{stem}_r{i // g.cols + 1}c{i % g.cols + 1}.png'
            card.save(out / name)
            all_cards.append(card)
        total += len(cards)
        print(f'{shot.name}: {g.cols}列 × {g.rows}行 = {len(cards)} 张 '
              f'(卡面 {g.card_w}×{g.card_h})')
    if all_cards:
        S.build_sheet(all_cards).save(out / '_names.png')
    print(f'合计 {total} 张 -> {out}')


def cmd_compose(args):
    root = Path(args.cards)
    players = [d for d in sorted(root.iterdir()) if d.is_dir()] if root.is_dir() else []
    if not players:
        sys.exit(f'{root} 下没有玩家目录')
    order = [ln.strip() for ln in Path(args.order).read_text(encoding='utf-8').splitlines()
             if ln.strip()] if args.order else None

    names = set()
    for p in players:
        names |= {f.stem for f in p.glob('*.png')}
    ops = [o for o in order if o in names] if order else sorted(names)
    missing = [o for o in order if o not in names] if order else []

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    made = 0
    for op in ops:
        entries = []
        for p in players:
            card = p / f'{op}.png'
            if card.exists():
                entries.append({'nick': p.name, 'card': str(card)})
        if not entries:
            continue
        C.compose_frame(entries, str(out / f'{op}.png'), args.bg)
        made += 1
        print(f'{op:<8} {len(entries)} 人')
    print(f'合成 {made} 帧 -> {out}')
    if missing:
        print(f'注意: 顺序表里有 {len(missing)} 个干员谁都没有: {"、".join(missing[:10])}…')


def main(argv=None):
    ap = argparse.ArgumentParser(prog='ark_card')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p1 = sub.add_parser('crop', help='从干员列表截图切出卡片')
    p1.add_argument('shots', nargs='+')
    p1.add_argument('--out', required=True)
    p1.set_defaults(func=cmd_crop)

    p2 = sub.add_parser('compose', help='按干员合成帧')
    p2.add_argument('cards', help='卡片根目录, 每个玩家一个子目录')
    p2.add_argument('--order', help='干员顺序 txt, 一行一个')
    p2.add_argument('--out', required=True)
    p2.add_argument('--bg', help='背景图路径')
    p2.set_defaults(func=cmd_compose)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
