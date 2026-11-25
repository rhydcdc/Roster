# -*- coding: utf-8 -*-
"""Roster v3 HTML 빌더.

GUIDE/01 ~ 05의 표와 접이식을 가져와 탭 HTML(tab01 ~ tab05.html)에 끼우고,
자립형 HTML 두 개(ROSTER_VIZ.html, ROSTER_SHOWCASE.html)를 만든다.

  python VIZ/src_v3/build.py            # 둘 다 만들고 검사
탭 HTML 안의 자리 표시:
  <gt d="03" h="머리줄에 든 글" n="1"/>   가이드 03에서 머리줄에 그 글이 든 n번째 표
  <gt d="04" h="..." drop="행에 든 글"/>   그 표에서 그 글이 든 행만 빼고 가져온다 (가이드는 그대로 둔다)
  <gt d="04" h="..." dropcol="머리줄에 든 글"/>   그 표에서 그 글이 든 열만 빼고 가져온다
  <gf d="01" s="요약에 든 글" t="새 제목"/>  가이드 01에서 요약에 그 글이 든 접이식
"""
import io
import json
import os
import re
import sys

import markdown

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
GUIDE = os.path.join(ROOT, 'GUIDE')
DOCS = {'01': '01_TRAINING_FLOW.md', '02': '02_SINGLE_INFERENCE_FLOW.md', '03': '03_BATCH_TILING_FLOW.md',
        '04': '04_KERNEL_DISPATCH_FLOW.md', '05': '05_NOVELTY_FLOW.md'}
TABS = [('01', '학습'), ('02', '단일 요청'), ('03', '배치 타일링'), ('04', '실행'), ('05', '새로운 점')]


def rd(p):
    return io.open(p, encoding='utf-8').read()


def wr(p, t):
    io.open(p, 'w', encoding='utf-8', newline='\n').write(t)


# 가이드 01 ~ 04의 내부 이름을 05의 공개 이름으로 바꾼다 (사용자 결정: HTML은 05의 이름으로 통일)
PUB = [
    (r' \(`CH2_M32_B1B2_r9`\)', ''),
    (r'\(`CH2_M32_B1B2_r9`, ', '('),
    (r'`CH2_M32_B1B2_r9`', '`Roster` 최종 모델'),
    (r'`CH2 \+ Roster-Tile \+ v4 \+ A3V4`', '`Roster` + `Roster-Tile`'),
    (r'`CH2 \+ v4 \+ A3V4`', '`Roster` + `Roster-Tile`'),
    (r'`CH2 \+ Roster-Tile` \(`A3V4`\)', '`Roster-Tile` 실시간 타일링'),
    (r'`s1_seed2`, ', ''),
    (r'`A3V4` 디스패처', '실시간 타일 계획기'),
    (r'`A3V4` 방식', '우리 방식'),
    (r'`A3V4`(는|가) ', r'실시간 타일 계획기\1 '),
    (r'`A3V4` (4층 합|최악 1층)', r'실시간 계획 \1'),
    (r'CPU `A3V4` 계획', 'CPU 실시간 계획'),
    (r'`A3V4`', '실시간 타일링'),
    (r'`v4` 커널 하드웨어 튜닝', '커널 하드웨어 튜닝'),
    (r'`v4` 커널', '우리 커널'),
    (r'기존 `v3` 대비', '이전 구현 대비'),
    (r'`pack_h_v4`, ', ''),
    (r'`Strip512`', '512열 구역'),
    (r'CPU 디스패치 & GPU 커널 실행', 'CPU 실시간 계획 & GPU 커널 실행'),
    (r'런타임 디스패치', '실시간 타일 계획'),
    (r'디스패치가 빨라', '계획이 빨라'),
    (r'디스패치', '타일 계획'),
    (r'가족\(`B1`\)', '가족'),
    (r'블록\(`B2`\)', '블록'),
    (r'`CH2`', '`Roster`'),
    (r'<code>TM = (\d+)</code>', r'\1명 묶음'),
    (r'`TM = (\d+)`', r'`\1`명 묶음'),
    (r'\(`TM`\)', '(묶음 크기)'),
    (r'`TM`', '묶음 크기'),
    (r' · 심사 방어선 & 한계', ''),
    (r'\(`0\d_[A-Z_]+\.md`, 이 문서\)', '(이 탭)'),
    (r'\$2\^\{?(\d+)\}?( = \d+)?\$', r'2<sup>\1</sup>\2'),
]
FORBID = ['직렬', 'CH2', 'A3V4', '디스패', '합승', '트랙 A', '트랙 B', 'Lc6', '색인', 'pack_h', 'Strip512', 's1_seed2',
          '`v4`', 'v4 커널', '돌파', 'B1B2']


# 사용자가 HTML에서 빼기로 한 '직렬 합산'(어텐션 중첩 없이 다 더한 E2E)의 언급. 가이드는 그대로 두고 가져올 때만 잘라 낸다
CUT = [
    (r'<br>직렬 E2E \*\*`[^`]*`\*\*', ''),
    (r' · 직렬 \*\*`[^`]*`\*\*', ''),
    (r' · `1\.66 ~ 3\.10×` 직렬', ''),
    (r' \(직렬 \*\*`[^`]*`\*\*\)', ''),
    (r' \(직렬도 `[^`]*`\)', ''),
    (r' · 직렬 `[^`]*`', ''),
]
# 그 수치들은 화면에 없는 것이 맞으므로 수치 대조에서 뺀다
CUT_NUMS = {'1.66', '2.58', '3.10', '5.779', '7.393', '12.347', '5.78', '7.39', '12.35', '0.72', '1.24', '1.58', '3.451', '3.892', '6.261'}


# 사용자 결정: 정확도는 '90.81%'만 쓰지 않고 '베이스 대비 99.77% (90.81%)' 꼴로 쓴다. 앞의 세 줄은 가이드에 자주 나오는 꼴, 넷째 줄은 나머지 전부
ACC = [
    (r'\*\*`?90\.81%`?\*\* \(베이스의 \*{0,2}`?99\.77%`?\*{0,2}\)', '베이스 대비 **99.77%** (90§81%)'),
    (r'\*\*`90\.81%`\*\* \(\*\*`99\.77%`\*\*, ', '베이스 대비 **`99.77%`** (`90§81%`, '),
    (r'\*\*`90\.81%`\*\* \(베이스 `91\.02%`의 `99\.77%`\)', '베이스 대비 **`99.77%`** (`90§81%`, 베이스 `91.02%`)'),
    (r'(?:\*\*)?`?90\.81%`?(?:\*\*)?', '베이스 대비 `99.77%`(`90§81%`)'),
    (r'90§81', '90.81'),
]


def pub(t):
    for a, b in PUB + CUT + ACC:
        t = re.sub(a, b, t)
    return t


_guide = {}


def guide(doc):
    if doc in _guide:
        return _guide[doc]
    txt = rd(os.path.join(GUIDE, DOCS[doc]))
    if doc == '05':  # 6단계(방어선)는 넣지 않는다 (사용자 결정). 맨 끝의 5부작 접이식은 살린다
        a, b = txt.index('## 6단계'), txt.rindex('<details>')
        txt = txt[:a] + txt[b:]
    lines, tables, i = txt.split('\n'), [], 0
    while i < len(lines):
        if lines[i].lstrip().startswith('|') and i + 1 < len(lines) and re.match(r'^\s*\|[\s:\-|]+\|\s*$', lines[i + 1]):
            j = i
            while j < len(lines) and lines[j].lstrip().startswith('|'):
                j += 1
            tables.append((lines[i], '\n'.join(l.strip() for l in lines[i:j])))
            i = j
        else:
            i += 1
    folds = re.findall(r'<details>\s*<summary>(.*?)</summary>(.*?)</details>', txt, re.S)
    _guide[doc] = (txt, tables, folds)
    return _guide[doc]


MARK = {'○': 'mk-ok', '△': 'mk-mid', '×': 'mk-no', '—': ''}


def to_html(md):
    out = markdown.markdown(pub(md), extensions=['tables'])
    out = out.replace('<table>', '<div class="tw"><table>').replace('</table>', '</table></div>')

    def row(m):
        tr = m.group(0)
        first = re.search(r'<td[^>]*>(.*?)</td>', tr, re.S)
        if first and re.search(r'우리|Roster', re.sub(r'<[^>]+>', '', first.group(1))):
            return tr.replace('<tr>', '<tr class="me">', 1)
        return tr
    out = re.sub(r'<tr>.*?</tr>', row, out, flags=re.S)
    out = re.sub(r'<a href="0(\d)_[A-Z_]+\.md">.*?</a>', lambda m: '<a href="#0%s">0%s 탭</a>' % (m.group(1), m.group(1)), out)
    out = re.sub(r'<td([^>]*)>(<strong>)?<code>([○△×—])</code>',
                 lambda m: '<td%s class="%s">%s%s' % (m.group(1), MARK[m.group(3)], m.group(2) or '', m.group(3)), out)
    return out


used = set()
dropped = {}  # 문서별로 HTML에서 뺀 표의 행 (수치 대조에서도 뺀다)


def sub_gt(m):
    a = dict(re.findall(r'(\w+)="([^"]*)"', m.group(0)))
    _, tables, _ = guide(a['d'])
    hits = [t for t in tables if a['h'] in t[0]]
    n = int(a.get('n', 1))
    if len(hits) < n:
        raise SystemExit('표를 못 찾음: %s' % a)
    used.add((a['d'], tables.index(hits[n - 1])))
    md = hits[n - 1][1]
    if a.get('drop'):  # 사용자가 빼기로 한 행: 가이드는 그대로 두고 HTML에서만 뺀다
        rows = md.split('\n')
        gone = [r for r in rows if a['drop'] in r]
        if not gone:
            raise SystemExit('뺄 행을 못 찾음: %s' % a)
        dropped.setdefault(a['d'], []).extend(gone)
        md = '\n'.join(r for r in rows if a['drop'] not in r)
    if a.get('dropcol'):  # 머리줄에 그 글이 든 열을 뺀다
        rows = [re.split(r'(?<!\\)\|', r.strip().strip('|')) for r in md.split('\n')]
        cut = [i for i, c in enumerate(rows[0]) if a['dropcol'] in c]
        if not cut:
            raise SystemExit('뺄 열을 못 찾음: %s' % a)
        md = '\n'.join('|' + '|'.join(c for i, c in enumerate(r) if i not in cut) + '|' for r in rows)
    return to_html(md)


def sub_gf(m):
    a = dict(re.findall(r'(\w+)="([^"]*)"', m.group(0)))
    _, _, folds = guide(a['d'])
    hits = [f for f in folds if a['s'] in f[0]]
    if not hits:
        raise SystemExit('접이식을 못 찾음: %s' % a)
    summ = a.get('t') or pub(re.sub(r'</?b>', '', hits[0][0])).strip()
    summ = re.sub(r'^(참고|상세 근거|상세 대조|다시 보기)\s*·\s*', '', summ)
    return '<details class="fold"><summary>%s</summary><div class="fold-b">%s</div></details>' % (summ, to_html(hits[0][1]))


def fill(t):
    t = re.sub(r'<gt [^>]*/>', sub_gt, t)
    t = re.sub(r'<gf [^>]*/>', sub_gf, t)
    return t


FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
         '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&'
         'family=IBM+Plex+Sans+KR:wght@400;500;600;700&display=swap">')


def page(title, desc, css, body, js, lang='ko'):
    return ('<!doctype html>\n<html lang="%s" data-theme="dark">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            '<title>%s</title>\n<meta name="description" content="%s">\n%s\n<style>\n%s\n</style>\n</head>\n<body>\n%s\n'
            '<script>\n%s\n</script>\n</body>\n</html>\n') % (lang, title, desc, FONTS, css, body, js)


def build_viz(showcase_href='ROSTER_SHOWCASE.html'):
    tabs = ''.join('<button type="button" class="tab-b" role="tab" data-tab="%s"><span class="n">%s</span>%s</button>' % (k, k, n)
                   for k, n in TABS)
    body = ['<header class="bar"><div class="wrap bar-in"><div class="brand">Roster<small>FFN 가속 · 직관 가이드</small></div>'
            '<nav class="tabs" role="tablist" aria-label="가이드 탭">%s</nav>'
            '<div class="tools"><a class="tool" href="%s">쇼케이스</a>'
            '<button type="button" class="tool" data-theme-btn>밝게</button></div></div></header>' % (tabs, showcase_href),
            '<main class="wrap">', rd(os.path.join(HERE, 'kpis.html'))]
    for k, n in TABS:
        body.append('<div class="tab" id="tab-%s" role="tabpanel">%s</div>' % (k, fill(rd(os.path.join(HERE, 'tab%s.html' % k)))))
    body.append('<p class="foot">Roster · 실험 모델은 Fashion-MNIST 4층 SwiGLU FFN (d = 1,024, 뉴런 4,096개) · '
                '실측은 RTX 4060 Laptop, 순수 bf16 (양자화 없음)</p></main>')
    js = '\n'.join(rd(os.path.join(HERE, f)) for f in ('core.js', 'km.js', 'ex.js', 'viz.js'))
    return page('Roster · FFN 가속 직관 가이드', '뉴런 출석부와 비격자 타일링으로 FFN을 가속하는 구조를 작은 예시로 따라가는 가이드',
                rd(os.path.join(HERE, 'base.css')), '\n'.join(body), js)


def build_showcase(viz_href='ROSTER_VIZ.html'):
    body = rd(os.path.join(HERE, 'showcase.html')).replace('{{VIZ}}', viz_href)
    js = '\n'.join(rd(os.path.join(HERE, f)) for f in ('core.js', 'km.js', 'net_data.js', 'ex.js', 'showcase.js'))
    css = rd(os.path.join(HERE, 'base.css')) + '\n' + rd(os.path.join(HERE, 'showcase.css'))
    return page('Roster · 쇼케이스', 'Roster FFN 가속 구조를 장면으로 넘겨 보는 쇼케이스', css, body, js)


def text_of(htmltxt):
    t = re.sub(r'<script>.*?</script>', ' ', htmltxt, flags=re.S)
    t = re.sub(r'<style>.*?</style>', ' ', t, flags=re.S)
    return re.sub(r'<[^>]+>', ' ', t)


def check(name, out):
    bad = 0
    for w in FORBID:
        for m in re.finditer(re.escape(w), out):
            bad += 1
            print('  [금지어] %s: …%s…' % (w, out[max(0, m.start() - 30):m.end() + 30].replace('\n', ' ')))
    txt = text_of(out)
    for pat, why in ((r'\*\*', '굵게 표시가 안 풀림'), (r'\$[^$\n]{1,40}\$', '수식이 안 풀림'), (r'<g[tf] ', '자리 표시가 남음')):
        for m in re.finditer(pat, txt if why != '자리 표시가 남음' else out):
            bad += 1
            src = txt if why != '자리 표시가 남음' else out
            print('  [%s] …%s…' % (why, src[max(0, m.start() - 30):m.end() + 30].replace('\n', ' ')))
    ids = set(re.findall(r'id="([^"]+)"', out))
    for hsh in set(re.findall(r'href="#(s\d\d-[\w]+)"', out)):
        if hsh not in ids:
            bad += 1
            print('  [끊긴 링크] #%s' % hsh)
    print('%s: %.1f KB, 문제 %d건' % (name, len(out.encode('utf-8')) / 1024, bad))
    return bad


def guide_numbers():
    """가이드에 나오는 눈에 띄는 수치(소수이거나 세 자리 이상)를 문서별로 모은다. 실행 검사에서 화면의 글과 대조한다."""
    res = {}
    for d in DOCS:
        txt = guide(d)[0]
        for row in dropped.get(d, []):
            txt = txt.replace(row, '')
        toks = set()
        for m in re.finditer(r'\d[\d,]*(?:\.\d+)?', txt):
            tok = m.group(0).rstrip(',')
            if ('.' in tok or len(tok.replace(',', '')) >= 3) and tok not in CUT_NUMS:
                toks.add(tok)
        res[d] = sorted(toks)
    return res


def main():
    bad = 0
    viz = build_viz()
    bad += check('ROSTER_VIZ.html', viz)
    wr(os.path.join(ROOT, 'ROSTER_VIZ.html'), viz)
    wr(os.path.join(ROOT, 'VIZ', 'roster_viz.html'), build_viz('roster_showcase.html'))  # 미러는 옆 파일 이름으로 잇는다
    for d in DOCS:  # 가져오지 않은 가이드 표를 알려 준다 (위젯으로 대신한 것인지 사람이 확인)
        _, tables, _ = guide(d)
        rest = [t[0][:70] for i, t in enumerate(tables) if (d, i) not in used]
        print('  가이드 %s: 표 %d개 중 %d개 가져옴, 위젯·손글로 대신한 표 %d개' % (d, len(tables), len(tables) - len(rest), len(rest)))
        if '-v' in sys.argv:
            for r in rest:
                print('      · ' + r)
    if os.path.exists(os.path.join(HERE, 'showcase.html')):
        sc = build_showcase()
        bad += check('ROSTER_SHOWCASE.html', sc)
        wr(os.path.join(ROOT, 'ROSTER_SHOWCASE.html'), sc)
        wr(os.path.join(ROOT, 'VIZ', 'roster_showcase.html'), build_showcase('roster_viz.html'))
    os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
    wr(os.path.join(HERE, 'out', 'guide_numbers.json'), json.dumps(guide_numbers(), ensure_ascii=False))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
