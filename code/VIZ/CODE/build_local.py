"""neuron_recipe.html(게시용: data/를 옆에서 불러옴) -> 프로젝트 맨 위 RECIPE_VIZ.html(데이터와 그림을 파일 안에 넣어, 폴더에서 눌러 혼자 열림).
python build_local.py
"""
import base64
import json
import os

VIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
page = open(os.path.join(VIZ, 'neuron_recipe.html'), encoding='utf-8').read()
data = open(os.path.join(VIZ, 'data', 'viz.json'), encoding='utf-8').read()
d = json.loads(data)
files = [s['file'] for s in d['sprites']['rep']] + [d['sprites']['feat']['file'], d['sprites']['inf']['file']]
spr = {f: 'data:image/png;base64,' + base64.b64encode(open(os.path.join(VIZ, 'data', f), 'rb').read()).decode() for f in files}
inject = '<script>window.__VIZ__=%s;window.__SPR__=%s;</script>\n' % (data, json.dumps(spr))
mark = '<script>\n(() => {'
assert page.count(mark) == 1
out = ('<!doctype html>\n<html lang="ko">\n<head>\n<meta charset="utf-8">\n'
       '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
       '<style>:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}'
       'body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>\n</head>\n<body>\n'
       + page.replace(mark, inject + mark) + '\n</body>\n</html>\n')
path = os.path.join(os.path.dirname(VIZ), 'RECIPE_VIZ.html')      # 프로젝트 맨 위: 폴더에서 눌러 바로 연다
open(path, 'w', encoding='utf-8').write(out)
print('%s %.1f KB' % (os.path.basename(path), os.path.getsize(path) / 1024))
