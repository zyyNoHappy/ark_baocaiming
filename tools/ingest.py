# -*- coding: utf-8 -*-
"""报菜名入库工具：切图 + 认名 + 按 order.csv 命名，增量执行。

设计要点（避免重复劳动）：
  · 卡片的身份 = 参与者 + 截图文件名 + 行 + 列，与像素坐标无关。
    所以以后改了切图算法、卡框位移几像素，也不会让已识别的结果失效。
  · 每张卡的 OCR 原始文本块存在 state.json 里，名字随时可由 pick_name 现算。
  · 已经处理过的参与者默认整个跳过；只有 --only/--force 才会重做。

用法：
    python tools/ingest.py                  # 全部人，增量（已有的跳过）
    python tools/ingest.py --only 例A        # 只处理某个人
    python tools/ingest.py --recrop 例A      # 重切某人（仍复用识别缓存）
    python tools/ingest.py --force           # 忽略缓存全部重做
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # 工程根
sys.path.insert(0, str(ROOT / 'pkg' / 'astrbot_plugin_ark_baocaiming'))

from ark_card import plate as P          # noqa: E402
from ark_card import ocr as OCR          # noqa: E402
from pipeline import read_order          # noqa: E402

try:
    from PIL import Image                # noqa: E402
except ImportError:
    raise SystemExit('需要 Pillow')

DATA = ROOT / 'data'
SHOTS = DATA / 'shots'                   # data/shots/<参与者>/*.png
CROP = DATA / 'crops'                    # 切图结果
NAMED = DATA / 'crops_named'             # 按序号命名
STATE = DATA / 'state.json'              # {"<参与者>": {"<文件#行#列>": {"texts":[], "scores":[]}}}
IMG = ('.png', '.jpg', '.jpeg', '.webp')


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding='utf-8'))
    return {}


def load_overrides() -> dict:
    """人工确认过的补名，形如 {"例A": {"6.png#1#0": "黑"}}。

    1x 识别偶尔会漏掉单字名，确认过之后写在这里，重跑不会再丢。
    """
    p = DATA / 'overrides.json'
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}


def save_state(st: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, ensure_ascii=False), encoding='utf-8')


def duplicate_report(named_dir: Path) -> list[list[str]]:
    """找出同一人目录里内容完全相同的卡，返回分组后的文件名列表。

    正常情况下一张卡只有一个文件；如果两个不同干员的文件 MD5 相同，
    说明命名环节把同一张图存成了两个名字，成片里那格会是错的干员。
    """
    by_hash: dict[str, list[str]] = {}
    for p in sorted(named_dir.glob('*.png')):
        try:
            h = hashlib.md5(p.read_bytes()).hexdigest()
        except OSError:
            continue
        by_hash.setdefault(h, []).append(p.name)
    return [v for v in by_hash.values() if len(v) > 1]


def crop_person(person: str):
    """返回 [(key, card, row, col, shot_name)]，顺序逐行从左到右。

    键里带截图内容指纹：「同名但内容不同」的新素材不会被旧缓存误命中，
    内容一致的重跑仍然复用。

    单张截图切不出来（不是列表页、被裁过）只跳过它，不影响同一个人其他截图。
    """
    d = SHOTS / person
    out = []
    for p in sorted(q for q in d.rglob('*') if q.suffix.lower() in IMG):
        try:
            cards, g = P.crop_cards(Image.open(p))
        except Exception as e:
            print(f'   跳过 {p.name}: {e}')
            continue
        h = hashlib.md5(p.read_bytes()).hexdigest()[:8]
        for k, c in enumerate(cards):
            row, col = divmod(k, max(1, g.cols))
            out.append((f'{p.name}#{row}#{col}@{h}', c, row, col, p.name))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', action='append', default=[])
    ap.add_argument('--recrop', action='append', default=[])
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()

    order = read_order(ROOT / 'pkg' / 'astrbot_plugin_ark_baocaiming' / 'assets' / 'order.csv')
    valid = [r['name'] for r in order]
    idx = {r['name']: r['index'] for r in order}
    st = load_state()

    people = sorted(d.name for d in SHOTS.iterdir() if d.is_dir())
    if a.only:
        people = [p for p in people if p in a.only]

    engine = None
    total_ocr = 0
    for person in people:
        named_dir = NAMED / person
        cached = st.get(person) or {}
        if named_dir.is_dir() and cached and not a.force and person not in a.recrop:
            n = len(list(named_dir.glob('*.png')))
            print(f'{person}: 已有结果（{n} 个干员），跳过')
            continue

        items = crop_person(person)
        crop_dir = CROP / person
        if crop_dir.exists():
            shutil.rmtree(crop_dir)
        crop_dir.mkdir(parents=True)
        for i, (key, card, *_rest) in enumerate(items, 1):
            card.save(crop_dir / f'{i:03d}.png')

        todo = [it for it in items if it[0] not in cached]
        print(f'{person}: {len(items)} 张卡，需识别 {len(todo)} 张', flush=True)
        if todo:
            if engine is None:
                engine = OCR._get_engine(str(DATA / 'paddlex'))
            tmp = DATA / '_tmp'; tmp.mkdir(parents=True, exist_ok=True)
            for j, (key, card, *_rest) in enumerate(todo, 1):
                q = tmp / 'one.png'
                OCR._name_strip(card).save(q)
                texts, scores = [], []
                for r in engine.predict(str(q)):
                    texts.extend(r.get('rec_texts') or [])
                    scores.extend(float(s) for s in (r.get('rec_scores') or []))
                cached[key] = {'texts': texts, 'scores': scores}
                if j % 30 == 0:
                    print(f'   {j}/{len(todo)}', flush=True)
            total_ocr += len(todo)
            st[person] = cached
            save_state(st)

        # 命名落盘
        if named_dir.exists():
            shutil.rmtree(named_dir)
        named_dir.mkdir(parents=True)
        ov = load_overrides().get(person) or {}
        raw = []
        for key, _card, _r, _c, _s in items:
            if key in ov:
                raw.append(ov[key])
                continue
            b = cached.get(key) or {}
            raw.append(OCR.pick_name(b.get('texts') or [], valid, b.get('scores')))
        fixed, _hit = OCR.match_order(raw, valid)      # 模糊对齐到排序表
        used, miss, dup = {}, [], []
        for (key, card, row, col, shot), nm in zip(items, fixed):
            if not nm:
                miss.append(f'{shot}#{row}#{col}')
                continue
            if nm in used:
                dup.append(nm)
                continue
            used[nm] = key
            card.save(named_dir / f'{idx[nm]:03d}-{nm}.png')
        st[person] = cached
        print(f'   命名 {len(used)}，未识别 {len(miss)}，重复 {len(dup)}，'
              f'缺 {len(valid) - len(used)}', flush=True)

        # 自检：同一人目录内不应有两张内容完全相同的卡。
        # 出现就说明命名把同一张卡写成了两个名字（历史 bug：塔露拉被存成 W），
        # 成片里会表现为「某个干员那格是别的干员」。
        clash = duplicate_report(named_dir)
        if clash:
            print('   ⚠ 发现内容重复的卡（很可能是命名错误）：', flush=True)
            for names in clash:
                print('     ' + ' = '.join(names), flush=True)

    save_state(st)
    print(f'\n本次实际 OCR {total_ocr} 张（其余全部复用缓存）')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
