# -*- coding: utf-8 -*-
"""쇼케이스 첫 장면(신경망 그림)에 쓸 실제 데이터를 작게 뽑는다. 새로 재는 것은 없다.

  python VIZ/src_v3/mk_net_data.py     ->  VIZ/src_v3/net_data.js

VIZ/data_roster/viz.json의 시험 사진 12장 가운데, v1 쇼케이스와 같은 8장(종류마다 첫 사진)을 고른다.
사진마다 층 4개의 중심점 번호(j), 중심점 수(M), 켜는 개수(K), 뉴런 4,096개의 켜짐 여부(on, 뉴런 번호 순 비트 묶음)를 그대로 옮긴다.
"""
import base64
import io
import json
import os

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data_roster')
KO = ['티셔츠', '바지', '풀오버', '원피스', '코트', '샌들', '셔츠', '운동화', '가방', '앵클부츠']
EN = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
WANT = [7, 4, 8, 9, 1, 0, 2, 3]   # v1 쇼케이스와 같은 순서

viz = json.load(io.open(os.path.join(DATA, 'viz.json'), encoding='utf-8'))
first = {}
for ex in viz['examples']:
    first.setdefault(int(ex['label']), ex)

out = []
for lbl in WANT:
    ex = first[lbl]
    raw = np.frombuffer(base64.b64decode(ex['img']), dtype=np.uint8).reshape(28, 28)
    buf = io.BytesIO()
    Image.fromarray(raw, mode='L').save(buf, format='PNG', optimize=True)
    layers = []
    for st in ex['steps']:
        on = np.unpackbits(np.frombuffer(base64.b64decode(st['on']), dtype=np.uint8))[:4096]
        assert int(on.sum()) == int(st['K']), (lbl, int(on.sum()), st['K'])   # 켜짐 비트 수 = K
        layers.append(dict(M=int(st['M']), j=int(st['j']), K=int(st['K']), on=st['on']))
    out.append(dict(ko=KO[lbl], en=EN[lbl], img='data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii'), layers=layers))

js = ("'use strict';\n/* 쇼케이스 첫 장면용 실제 데이터 (mk_net_data.py가 VIZ/data_roster/viz.json에서 뽑음). "
      "on = 뉴런 4,096개의 켜짐 여부를 뉴런 번호 순으로 묶은 비트(base64) */\n"
      "const NET = " + json.dumps(dict(ex=out), ensure_ascii=False, separators=(',', ':')) + ";\n")
io.open(os.path.join(HERE, 'net_data.js'), 'w', encoding='utf-8', newline='\n').write(js)
print('net_data.js %.1f KB' % (len(js.encode('utf-8')) / 1024))
for e in out:
    print(' ', e['ko'], [(l['j'], l['M'], l['K'], round(100 * l['K'] / 4096, 1)) for l in e['layers']])
