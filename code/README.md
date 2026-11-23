# 코드 (최종 통합판: `CH2_M32_B1B2_r9` + `v4` 커널 + `A3V4` 디스패처)

아키텍처 시각화 페이지(`../index.html`)의 모든 수치를 재현하는 **최신·필수 소스 코드만** 정리했다(이전 `Lc6`·초기 프로토타입 코드는 제거).
경로 보정 사항: `AUTO/EXP1/CODE/auto.py`가 이 `code/` 폴더를 루트로 인식하며, `EXP1/CODE/runs/s1_pick.json`은 상대 경로를 사용한다.

필수 환경: Python 3.13, PyTorch 2.6 (CUDA), torchvision, numpy, scipy, scikit-learn, Pillow, `g++` (AVX2 C++ 엔진 자동 컴파일). CUDA 커널은 PyTorch 내장 NVRTC로 런타임 컴파일되며, Fashion-MNIST 데이터셋은 첫 실행 시 자동 다운로드된다.

| 차례 | 폴더 | 명령 | 만드는 것 |
|---|---|---|---|
| 1. 베이스 사전 학습 (뉴런 100%) | `EXP1/CODE` | `python s1_train.py all` | `runs/s1_seed2/best.pt` (91.02%) |
| 2. 구조 스크리닝 & `CH2` 9바퀴 공유 사후 학습 (`M=[32,32,32,256]` + `B1/B2`) | `OPT/TRACK_B` | `python b_structural_screen.py` → `python b_champ_chains.py` | `runs/CH2_M32_B1B2_r9_lr1e-06_lb0/best.pt`, `lv_CH2_M32_B1B2_r9.pt`, `winner_orders.npz` |
| 3. 검증 전용 `1*` 선택 & 10,000장 연쇄 시험 검증 | `OPT/TRACK_B` | `python b3_opt_scan_val.py` → `python verify_b_10k.py` | `runs/b_10k_accuracy.json` (`1*=0%`, **90.81%**) |
| 4. RTX 4060 최종 E2E 실측 (서버 + 단일 `B=1` + 10회 `BSMR` 대조) | `OPT/TRACK_B` | `python bench_b_e2e.py` → `python bench_b_single.py` → `python compare_b_bsmr.py` | `runs/track_b_final.json`, `speed_b1_ch2.json`, `compare_bsmr_ch2.json` |
| 5. 자립형 인터랙티브 시각화 빌드 | `OPT/TRACK_B` | `python extract_roster_viz.py` → `python build_roster_viz.py` | `VIZ/roster_viz.html`, `ROSTER_VIZ.html` (= 루트 `index.html`) |

### 주요 모듈 구성
- **`OPT/TRACK_A/`:** `p1_v4.py` (`P1Unified v4` 4-CTA 텐서 코어 커널), `p2_v4.py` · `p2_kernels.py` (`P2Fast v4` + `pack_h_v4` + `autoTM`), `a3_dispatch_v4.cpp` · `a3_dispatch_v4.py` (16스레드 AVX2 제로-브랜치 오프라인-인덱스 디스패처).
- **`OPT/TRACK_B/`:** `track_b_lib.py` (`B1` 균형 가족 + `B2` 32열 블록 공유 사후 학습 엔진), 학습·검증·실측 스크립트 및 `extract_roster_viz.py` · `roster_viz_js.py` · `build_roster_viz.py`.
- **`EXP1/CODE/`, `AUTO/EXP1/CODE/`, `AUTO/EXP2/CODE/`, `BATCHING/CODE/`:** 위 최종 파이프라인이 직접 임포트하는 공통 모델·커널·`BSMR` 대조·`ms_engine.cpp` 라이브러리만 유지했다.
