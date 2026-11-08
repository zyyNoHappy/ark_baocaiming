#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
报菜名流水线（纯逻辑，不依赖 astrbot，方便单独测试）

    store = Store(root)
    store.add_screenshot('某个玩家', 'shot.png', namer=fn)   # 切图 + 命名入库
    store.compose('锏', out_path)
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image

from ark_card import grid as G
from ark_card import plate as P
from ark_card import sheet as S
from ark_card.compose import compose_frame, render_template, compose_on_template
from ark_card import notebook as NB

IMG_EXT = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
DEFAULT_PLACEHOLDER = Path(__file__).parent / 'assets' / 'placeholder.png'
DEFAULT_BG = Path(__file__).parent / 'assets' / 'bg_scene.png'   # 整幅插画（卡位自动落到中间纸面）
FALLBACK_BG = Path(__file__).parent / 'assets' / 'bg_blank.png'  # 只有纸面，圆出去用这张


def read_order(path: str | Path) -> list[dict]:
    """读干员顺序表。
    - CSV（表头含"干员"）: 排序,干员,...  -> [{'index':1,'name':'能天使'}, ...]
    - 纯文本: 一行一个，顺序即排序（从 1 起）
    """
    p = Path(path)
    raw = p.read_bytes()
    text = None
    for enc in ('utf-8-sig', 'utf-8', 'gb18030'):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError(f'{p} 编码无法识别')

    lines = [ln for ln in text.splitlines() if ln.strip()]
    if lines and '干员' in lines[0]:
        import csv, io
        rows = list(csv.reader(io.StringIO(text)))
        header = rows[0]
        try:
            i_idx = header.index('排序')
        except ValueError:
            i_idx = 0
        try:
            i_name = header.index('干员')
        except ValueError:
            i_name = 1
        i_date = header.index('国服上线时间') if '国服上线时间' in header else None
        out = []
        for r in rows[1:]:
            if len(r) <= i_name or not r[i_name].strip():
                continue
            try:
                idx = int(r[i_idx])
            except Exception:
                idx = len(out) + 1
            date = ''
            if i_date is not None and len(r) > i_date:
                date = r[i_date].strip()
            out.append({'index': idx, 'name': r[i_name].strip(), 'date': date})
        return out
    return [{'index': i + 1, 'name': ln.strip()} for i, ln in enumerate(lines)]


class Store:
    layout_mode = 'notebook'

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.players = self.root / 'players'
        self.players.mkdir(parents=True, exist_ok=True)

    # ---------- 基础 ----------
    def _pdir(self, nick: str) -> Path:
        safe = ''.join(c for c in nick if c not in r'\/:*?"<>|').strip() or 'unknown'
        d = self.players / safe
        (d / 'cards').mkdir(parents=True, exist_ok=True)
        return d

    def _meta(self, nick: str) -> dict:
        f = self._pdir(nick) / 'meta.json'
        if f.exists():
            try:
                return json.loads(f.read_text(encoding='utf-8'))
            except Exception:
                pass
        return {'nick': nick, 'shots': [], 'cards': {}}

    def _save_meta(self, nick: str, meta: dict):
        (self._pdir(nick) / 'meta.json').write_text(
            json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')

    # ---------- 入库 ----------
    def prepare(self, image_path: str):
        """切图（纯 CPU），命名另行处理。返回 (cards, grid)"""
        return P.crop_file(image_path)

    def save(self, nick: str, image_path: str, cards, g, names=None,
             uid: str | None = None) -> dict:
        meta = self._meta(nick)
        if uid:
            meta['uid'] = uid
        card_dir = self._pdir(nick) / 'cards'
        saved, dup, unnamed = 0, [], []
        for i, card in enumerate(cards):
            nm = names[i] if names and i < len(names) else None
            if not nm:
                nm = f"未识别_{len(meta['cards']) + i + 1:02d}"
                unnamed.append(nm)
            gap = g.edge_gaps[i] if i < len(g.edge_gaps) else 0
            old = meta['cards'].get(nm)
            if old is not None:
                # 同一个干员已经有一张了：留离屏幕边缘更远的那张，
                # 贴边的那张常被侧栏裁掉一截
                dup.append(nm)
                if gap <= old.get('edge_gap', -1):
                    continue
            p = card_dir / f'{nm}.png'
            card.save(p)
            meta['cards'][nm] = {'file': p.name, 'shot': Path(image_path).name,
                                 'edge_gap': gap}
            saved += 1
        meta['shots'].append({'file': Path(image_path).name, 'count': len(cards),
                              'grid': [g.cols, g.rows, g.card_w, g.card_h]})
        self._save_meta(nick, meta)
        return {'count': len(cards), 'saved': saved, 'grid': [g.cols, g.rows],
                'card_size': [g.card_w, g.card_h], 'names': names,
                'unnamed': unnamed, 'dup': dup}

    def add_screenshot(self, nick: str, image_path: str, namer=None) -> dict:
        """一步到位（命名函数需同步），给命令行 / 测试用"""
        cards, g = self.prepare(image_path)
        names = None
        if namer:
            try:
                names = namer(image_path, cards)
            except Exception:
                names = None
        return self.save(nick, image_path, cards, g, names)

    def add_from_zip(self, nick: str, zip_path: str, namer=None) -> dict:
        """一个参与者交的压缩包 -> 逐张截图切图入库。返回汇总（含 token 用量）。"""
        import uuid, zipfile
        zip_path = Path(zip_path)
        tmp = self.root / '_tmp' / f'zip_{uuid.uuid4().hex[:8]}'
        tmp.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmp)
        imgs = sorted(f for f in tmp.rglob('*') if f.suffix.lower() in IMG_EXT)
        if not imgs:
            raise ValueError('压缩包里没有图片')

        tokens = 0
        detail = []
        for img in imgs:
            cards, g = self.prepare(str(img))
            names, tk = (None, 0)
            if namer:
                try:
                    names, tk = namer(str(img), cards)
                except Exception:
                    names, tk = None, 0
            tokens += int(tk or 0)
            r = self.save(nick, str(img), cards, g, names)
            r['file'] = img.name
            detail.append(r)
        return {'shots': len(detail), 'cards': sum(d['count'] for d in detail),
                'tokens': tokens, 'detail': detail}

    def set_placeholder(self, nick: str, image_path: str, operator: str | None = None):
        """占位表情包。给了 operator 就存成该干员专用，否则作为通用兜底。"""
        img = Image.open(image_path)
        if getattr(img, 'n_frames', 1) > 1:      # 动图取第一帧
            img.seek(0)
        dst = (self._pdir(nick) / 'placeholders' / f'{operator}.png') if operator             else (self._pdir(nick) / 'placeholder.png')
        dst.parent.mkdir(parents=True, exist_ok=True)
        img.convert('RGB').save(dst)
        return dst

    def merge_memes(self, nick: str, zip_path: str) -> int:
        """把参与者按干员名命名的表情包压缩包并进他的数据。返回收录张数。"""
        import uuid, zipfile
        tmp = self.root / '_tmp' / f'meme_{uuid.uuid4().hex[:8]}'
        tmp.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmp)
        dst_dir = self._pdir(nick) / 'placeholders'
        dst_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in sorted(tmp.rglob('*')):
            if f.suffix.lower() not in IMG_EXT:
                continue
            name = f.stem.strip()
            if not name:
                continue
            img = Image.open(f)
            if getattr(img, 'n_frames', 1) > 1:
                img.seek(0)
            img.convert('RGB').save(dst_dir / f'{name}.png')
            n += 1
        return n

    def set_header(self, nick: str, image_path: str):
        """可选: 玩家名片截图 -> 合成时用作顶部名片条"""
        img = Image.open(image_path).convert('RGB')
        img.save(self._pdir(nick) / 'header.png')

    def remove(self, nick: str):
        d = self._pdir(nick)
        if d.exists():
            shutil.rmtree(d)

    # ---------- 查询 ----------
    def list_players(self) -> dict[str, dict]:
        out = {}
        for d in sorted(self.players.iterdir()):
            if not d.is_dir():
                continue
            meta = json.loads((d / 'meta.json').read_text(encoding='utf-8')) \
                if (d / 'meta.json').exists() else {'cards': {}}
            out[d.name] = {'cards': len(meta.get('cards', {})),
                           'shots': len(meta.get('shots', [])),
                           'header': (d / 'header.png').exists(),
                           'placeholder': (d / 'placeholder.png').exists(),
                           'bgs': len(list((d / 'placeholders').glob('*.png')))
                                  if (d / 'placeholders').is_dir() else 0}
        return out

    def operator_names(self) -> set[str]:
        names = set()
        for d in self.players.iterdir():
            c = d / 'cards'
            if c.is_dir():
                names |= {f.stem for f in c.glob('*.png')}
        return names

    def default_bg(self):
        for p in (DEFAULT_BG, FALLBACK_BG):
            if p.exists():
                return p
        return None

    def date_font(self):
        f = Path(__file__).parent / 'assets' / 'date_font.ttf'
        return str(f) if f.exists() else None

    # ---------- 模板 ----------
    def template_paths(self):
        return self.root / 'template.png', self.root / 'template.json'

    def build_template(self, bg: str | None = None, fit: float = 1.0,
                       gap_ratio: float = 0.06, layout_mode: str = 'notebook') -> dict:
        """第一步：背景 + 所有参与者的名片 + 占位卡 -> template.png / template.json"""
        players = []
        for d in sorted(self.players.iterdir()):
            if not d.is_dir():
                continue
            hdr = d / 'header.png'
            players.append({'nick': d.name, 'header': str(hdr) if hdr.exists() else None})
        if not players:
            raise ValueError('还没有任何参与者')

        if layout_mode == 'notebook':
            png, js = self.template_paths()
            NB.build_notebook_template(str(bg or self.default_bg()), str(png), tower=NB.tower_needed(len(players)))
            import json
            geom = {'layout': 'notebook', 'players': [p['nick'] for p in players],
                    'slots': len(players)}
            js.write_text(json.dumps(geom, ensure_ascii=False, indent=1), encoding='utf-8')
            return geom

        img, geom = render_template(players, bg, fit, gap_ratio,
                                    str(DEFAULT_PLACEHOLDER) if DEFAULT_PLACEHOLDER.exists() else None)
        png, js = self.template_paths()
        img.convert('RGB').save(png, quality=96)
        import json
        js.write_text(json.dumps(geom, ensure_ascii=False, indent=1), encoding='utf-8')
        return geom

    def load_template(self):
        import json
        png, js = self.template_paths()
        if not (png.exists() and js.exists()):
            return None, None
        return Image.open(png).convert('RGBA'), json.loads(js.read_text(encoding='utf-8'))

    def _entry_for(self, d: Path, operator: str, placeholder_mode: str) -> dict:
        card = d / 'cards' / f'{operator}.png'
        ph = d / 'placeholders' / f'{operator}.png'
        if not ph.exists():
            ph = d / 'placeholder.png'
        e = {'nick': d.name, 'mode': placeholder_mode,
             'header': str(d / 'header.png') if (d / 'header.png').exists() else None}
        if card.exists():
            e['card'] = str(card)
        elif ph.exists():
            e['placeholder'] = str(ph)
        elif DEFAULT_PLACEHOLDER.exists():
            e['placeholder'] = str(DEFAULT_PLACEHOLDER)
        return e

    # ---------- 合成 ----------
    def compose(self, operator: str, out_path: str, bg: str | None = None,
                index=None, title: bool = False,
                placeholder_mode: str = 'contain', date: str = '',
                fit: float = 1.0, gap_ratio: float = 0.06) -> str | None:
        """第三步：出帧。notebook 模式直接按模板里的参与者顺序贴格子。"""
        if self.layout_mode == 'notebook':
            _, geom = self.load_template()
            order = (geom or {}).get('players') or [d.name for d in sorted(self.players.iterdir()) if d.is_dir()]
            entries, owners = [], 0
            for nick in order:
                d = self.players / nick
                e = self._entry_for(d, operator, placeholder_mode)
                if e.get('card'):
                    owners += 1
                entries.append(e)
            if owners == 0:
                return None
            tpl_png, _ = self.template_paths()
            base = str(tpl_png) if tpl_png.exists() else str(bg or self.default_bg())
            NB.compose_notebook(entries, out_path, base, date, self.date_font(),
                                placeholder_mode, chrome=not tpl_png.exists())
            return out_path

        tpl, geom = self.load_template()
        if tpl is not None:
            order = geom.get('players') or []
            items, owners = [], 0
            for nick in order:
                d = self.players / nick
                e = self._entry_for(d, operator, placeholder_mode)
                if e.get('card'):
                    owners += 1
                items.append(e)
            if owners == 0:
                return None
            compose_on_template(tpl, geom, items, date, out_path)
            return out_path

        entries, owners = [], 0
        for d in sorted(self.players.iterdir()):
            if not d.is_dir():
                continue
            e = self._entry_for(d, operator, placeholder_mode)
            if e.get('card'):
                owners += 1
            entries.append(e)
        if not entries or owners == 0:
            return None
        return compose_frame(entries, out_path, bg, operator if title else None, index, date,
                             fit, gap_ratio)

    def compose_all(self, order_rows, out_dir: str, bg: str | None = None,
                    placeholder_mode: str = 'contain',
                    fit: float = 1.0, gap_ratio: float = 0.06) -> list[str]:
        """order_rows: [{'index':1,'name':'能天使'}, ...] 或 ['能天使', ...]"""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        made = []
        for i, row in enumerate(order_rows, 1):
            if isinstance(row, dict):
                name, idx, date = row.get('name'), row.get('index', i), row.get('date', '')
            else:
                name, idx, date = str(row), i, ''
            fname = f'{idx:03d}-{name}.png' if isinstance(idx, int) else f'{name}.png'
            p = self.compose(name, str(out / fname), bg, idx,
                             placeholder_mode=placeholder_mode, date=date,
                             fit=fit, gap_ratio=gap_ratio)
            if p:
                made.append(p)
        return made

    def make_zip(self, nick: str, out_zip: str, order=None) -> str | None:
        """切图 + 表情包 合并成一个包。

        干员卡按排序表命名成 `序号-干员名.png`（排序表里没有的排在后面），
        表情包放在 表情包/ 下。
        """
        import zipfile
        base = self._pdir(nick)
        cards = sorted((base / 'cards').glob('*.png'))
        memes = sorted((base / 'placeholders').glob('*.png')) \
            if (base / 'placeholders').is_dir() else []
        if not cards and not memes:
            return None

        idx: dict = {}
        for i, row in enumerate(order or [], 1):
            if isinstance(row, dict) and row.get('name'):
                idx[row['name']] = row.get('index', i)
            elif isinstance(row, str):
                idx[row] = i

        def _key(f: Path):
            return (idx.get(f.stem, 10 ** 6), f.stem)

        Path(out_zip).parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as z:
            for f in sorted(cards, key=_key):
                i = idx.get(f.stem)
                name = f'{i:03d}-{f.stem}.png' if isinstance(i, int) else f.name
                z.write(f, name)
            for f in memes:
                z.write(f, f'表情包/{f.name}')
        return out_zip
