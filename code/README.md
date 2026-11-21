# 코드

레시피 페이지(`../index.html`)의 결과를 낸 코드를 그대로 추렸다. 폴더 구조도 연구할 때 그대로라 실험 흔적이 섞여 있다. 연구 때와 다른 곳은 경로 두 곳뿐이다: `AUTO/EXP1/CODE/auto.py`가 이 `code/` 폴더를 맨 위로 알아보게 했고, `EXP1/CODE/runs/s1_pick.json`의 베이스 모델 경로를 상대 경로로 바꿨다.

필요한 것: Python 3.13, PyTorch 2.6 (CUDA), torchvision, numpy, scipy, Pillow. Fashion-MNIST는 처음 돌릴 때 torchvision이 받는다.

| 차례 | 폴더 | 명령 | 만드는 것 |
|---|---|---|---|
| 1. 사전 학습 (베이스 모델, 뉴런을 다 켬) | `EXP1/CODE` | `python s1_train.py all` | `runs/s1_seed2/best.pt`, `runs/s1_pick.json` |
| 2. 출발점: 층마다 중심점 256개와 뉴런 목록, K 50%로 가볍게 사후 학습 | `AUTO/EXP1/CODE` | `python auto.py queue --set F` | `runs/F_K2048_lam3/best.pt`, `runs/rules/ltable_M256.pt` |
| 3. ① 사후 학습 → ② K 줄이기를 되풀이 (1 ~ 4바퀴) | `AUTO/EXP2/CODE` | `python lchain.py run` | `runs/Lb1 ~ Lb4`, `runs/rules/lv_Lb*.pt` |
| 4. 되풀이 5 ~ 6바퀴 (최종 모델) | `AUTO/EXP2/CODE` | `python lc.py` | `runs/Lc6_lr9.49367e-07/best.pt`, `runs/rules/lv_Lc6.pt` |
| 5. 묶어서 계산: 뉴런 쪽 블록 + 블록 합치기의 연산량과 읽는 양 | `BATCHING/CODE` | `python batch_s7.py s7` | `runs/batch_s7.json` |
| 6. 레시피 페이지 | `VIZ/CODE`, `BATCHING/CODE` | `python extract.py` → `python build_local.py` → `python viz_batch_data.py` → `python viz_batch_inject.py` | `RECIPE_VIZ.html` (= 맨 위 `index.html`) |

- 사후 학습의 손실 = 교차 엔트로피 + 10 × 출력 맞추기 + 1 × 목록 앞쪽으로 몰기 (`auto.py`의 `make_loss`). 고른 뉴런만 계산한 채로 학습한다.
- 모든 숫자는 RTX 4060 Laptop, bf16에서 냈다.
