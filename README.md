# Roster

*Layer-differentiated neuron rosters, block-friendly shared post-training, non-grid 5-tier tiling, and sub-3ms AVX2 dispatch for pure-`bf16` FFN acceleration.*

순수 `bf16` FFN 가속 아키텍처 (양자화 없음 · `1* = 0%` 배치 무관 100% 동일 출력 · 10,000장 시험 정확도 **90.81%**, 베이스 91.02%의 **99.77%** · 평균 활성 뉴런 **13.52%**).
RTX 4060 3라운드 교대 CUDA Graph 4층 합계 실측 기준:
- **단일 요청 (`B = 1`):** `d = 1,024` **6.20×** · `d = 4,096` **6.27×** (0-Mask 대조군 `0.99×`)
- **대형 서버 (`B = 512 / 1,024 / 2,048`, `d = 4,096`):** 어텐션 중첩 E2E **3.08× / 4.09× / 4.57×** (`A3V4+H2D` 노출 `0.00 ms`, 여유 `2.3~6.4×`) · 중첩 없는 직렬 E2E **1.66× / 2.58× / 3.10×** (0-Mask `1.02~1.03×`, `d = 1,024` GPU FFN **3.00× / 3.83× / 4.32×**)

- **인터랙티브 시각 쇼케이스 (`ROSTER_SHOWCASE` · KO/EN):** https://rhydcdc.github.io/Roster/showcase.html
- **상세 아키텍처 시각화 (`ROSTER_VIZ`):** https://rhydcdc.github.io/Roster/
- **E2E 실측 바로가기:** https://rhydcdc.github.io/Roster/#s04-4
- **핵심 기여 바로가기:** https://rhydcdc.github.io/Roster/#05
- **재현 코드 안내:** [`code/README.md`](code/README.md)

<details>
<summary><b>Key Contributions (English)</b></summary>

| # | Contribution | Core Mechanism | vs. Prior Work |
| :---: | :--- | :--- | :--- |
| **1** | **Layer-Differentiated Centroid Rosters (`M=[32,32,32,256]`)** | Single dot-product against finite layer centroids skips `gate · up · down` upfront with **0 learned routers** and overlapping variable-length rosters (`K_j`). Using `M=32` on L1–L3 naturally coalesces 16–64 tokens/centroid for dense `64×32` tiles, while `M=256` on L4 retains `0.80%` fine-grained sparsity. | *Deja Vu* / *ShadowLLM* (per-token MLP predictor, uncoalesced); *MoEfication* / *CMoE* (disjoint fixed experts) |
| **2** | **Tile & Block-Friendly Shared Post-Training (`B1 + B2`)** | Combines OLO-balanced family ranking (`B1`, `α=0.35`) and 32-column block ranking + Group-L2 (`B2`, `β=0.25`, `tot_on` norm) so the model learns 32-col block sparsity natively: `×0` drops to **12.6%**, `P2` union to **16.1%**, DRAM reads to **64.5%**, and **90.81% (99.77%)** accuracy with **`1* = 0%`** (batch-invariant exact outputs). | Standard per-token or per-centroid pruning suffers high tile padding (`×0`), `W_down` union blowup, and batch-dependent drops |
| **3** | **Non-Grid 5-Tier Tiling × Offline-Indexed `A3V4` AVX2 Dispatch** | Breaks fixed row-panel boundaries (*BSMR* / *Fused3S* / *TC-GNN*) via a measured-cost 5-tier ladder (`64×32 → 8×16`, `64×32` **76.7%**, fill **87.4%**). Offline indexing (`13.1 ms` once) cuts runtime 4-layer dispatch from 46s to **1.49–2.83 ms** (`0.00 ms` exposed behind Attention), matching offline exact `ms_solve` within `0.22%p`. | *MatFormer* (static global prefix); *BSMR* / *PIT* / *TileSpMM* (fixed row panels or single-scale kernels without sub-3ms dynamic dispatch) |
| **4** | **Unified Single & Large-Batch End-to-End Silicon Proof** | Verified end-to-end in pure `bf16` on RTX 4060 (CPU `A3V4` + pinned H2D + `v4` 4-CTA Tensor Core kernels + `autoTM` `P2Fast` + CUDA Graph + 10k cascading test): `B = 1` **6.20–6.27×**, `B = 512–2,048` (`d = 4,096`) **3.08–4.57×** overlapped E2E (**1.66–3.10×** zero-overlap serial E2E). | *PowerInfer* / *TurboSparse* / *CATS* (`B = 1` / small-batch focus); *Polar Sparsity* / *Park et al.* (large-batch SwiGLU FFN collapse) |

</details>

<details>
<summary><b>핵심 기여 4축 (한국어)</b></summary>

| # | 핵심 기여 | 핵심 원리 | 선행 연구 대비 차별점 |
| :---: | :--- | :--- | :--- |
| **1** | **학습 라우터 없는 층별 차등 중심점 명단 (`M=[32,32,32,256]`)** | 단순 중심점 내적 1회로 `gate · up · down` 3행렬을 선제 스킵하며 공용 뉴런을 자유롭게 공유. 앞 1~3층은 `M=32`로 대형 배치에서 중심점당 <b>16~64토큰</b>이 자연 결집해 `64×32` 뼈대를 만들고, 출력 직전 4층은 `M=256`으로 <b>0.80%</b> 정밀 분기를 유지. | *Deja Vu*·*ShadowLLM*(토큰별 예측 → 겹침 붕괴), *MoEfication*·*CMoE*(겹침 없는 고정 전문가)와 구별 |
| **2** | **타일·블록 친화적 공유 사후 학습 (`B1 + B2`)** | OLO 균형 가족 몰기(`B1`, `α=0.35`)와 32열 블록 몰기 + Group-L2(`B2`, `β=0.25`, `tot_on` 정규화)로 모델이 32열 블록 단위로 뭉쳐 희소해지도록 학습: `×0` <b>12.6%</b> · `P2` 합집합 <b>16.1%</b> · DRAM 읽기 <b>64.5%</b> · **`1* = 0%`** 배치 무관 동일 출력(<b>90.81%</b>, 베이스의 <b>99.77%</b>). | 개별 몰기(`M=256`) 대비 동일 커널 위 순수 <b>+47~70%</b> 추가 가속과 정확도 동시 향상 |
| **3** | **비격자 5단계 타일링 × 오프라인-인덱스 `A3V4` AVX2 디스패치** | 고정 행 패널 경계를 깬 비격자 5단계 비용 사다리(`64×32` 단독 <b>76.7%</b>, 채움 <b>87.4%</b>). 오프라인 인덱스 1회(`13.1 ms`)로 런타임 4층 계획을 46초 → <b>1.49~2.83 ms</b>(16,000배 단축, 어텐션 뒤 노출 `0.00 ms`)로 끝내며, 10회 `BSMR` 5단계 행 패널 상한 대비 `×0` <b>-54.7%</b> 우위. | *MatFormer*(전역 정적), *BSMR*·*Fused3S*·*TC-GNN*(고정 행 패널), *PIT*/*TileSpMM*(실시간 디스패치 미포함) 돌파 |
| **4** | **단일 요청 & 대형 서버 E2E 실측 검증 (순수 `bf16`)** | CPU `A3V4` · H2D · `v4` 4-CTA 텐서 코어 커널 · `autoTM` `P2Fast` · CUDA Graph · 10,000장 연쇄 시험까지 동일 RTX 4060에서 전수 검증: 단일 `B=1` <b>6.20~6.27×</b>, 대형 서버(`d=4,096`) 어텐션 중첩 E2E <b>3.08~4.57×</b> · 직렬 E2E <b>1.66~3.10×</b>. | *PowerInfer*·*TurboSparse*·*CATS*(`B=1` 중심), *Polar Sparsity*·*Park 등*(대형 배치 SwiGLU FFN 붕괴) 한계 돌파 |

</details>
