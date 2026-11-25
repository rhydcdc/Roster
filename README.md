# Roster

*Neuron rosters fixed before each FFN layer, sharing-aware post-training, and real-time non-grid 5-tier tiling, for pure-`bf16` FFN acceleration at batch size 1 and at large batches.*

순수 `bf16` FFN 가속 아키텍처다 (양자화 없음). 평균 <b>13.52%</b>의 뉴런만 켜고, 시험 정확도는 베이스 대비 **99.77%** (90.81%, 베이스 91.02%)다. 배치 크기와 무관하게 출력이 같다 (`1* = 0%`).

RTX 4060에서 잰 4층 합계다 (3라운드 교대 CUDA Graph).

- **단일 요청 (`B = 1`):** `d = 1,024` **6.20×** · `d = 4,096` **6.27×** (0-Mask 대조군 `0.99×`)
- **대형 서버 (`B = 512 / 1,024 / 2,048`, `d = 4,096`):** 어텐션 중첩 E2E **3.08× / 4.09× / 4.57×** (타일 계획이 어텐션 뒤에 드러나는 시간 `0.000 ms`, 0-Mask 대조군 `1.02 ~ 1.03×`)
- **`d = 1,024`, 같은 배치:** GPU의 FFN만 잰 값으로 **3.00× / 3.83× / 4.32×** (이 폭에서는 어텐션이 짧아 타일 계획 시간을 다 숨기지 못한다)

검증 범위: 실험 모델은 Fashion-MNIST 4층 SwiGLU FFN(`d = 1,024`, 뉴런 4,096개)이다. `d = 4,096` 수치는 같은 출석부와 활성 비율을 대형 LLM 층 폭에 적용해 커널 속도를 잰 것이다.

- **쇼케이스 (9장면 슬라이드 · KO/EN):** https://rhydcdc.github.io/Roster/showcase.html
- **직관 가이드 (탭 5개):** https://rhydcdc.github.io/Roster/
- **E2E 실측 바로가기:** https://rhydcdc.github.io/Roster/#s04-4
- **핵심 기여 바로가기:** https://rhydcdc.github.io/Roster/#05
- **가이드 문서 원문 (한국어):** [`code/GUIDE/`](code/GUIDE/) (01 학습 · 02 단일 요청 · 03 배치 타일링 · 04 실행 · 05 새로운 점)
- **재현 코드 안내:** [`code/README.md`](code/README.md)

<details>
<summary><b>Key Contributions (English)</b></summary>

Ordered from the most to the least novel.

| Rank | Contribution | Where prior work stops | What is different here |
| :---: | :--- | :--- | :--- |
| **1** | **Real-time non-grid 5-tier tiling** | Recent Tensor Core SDDMM kernels (*BSMR*, *Fused3S*, *TC-GNN*) are bound to fixed row panels: a token cannot change its row partners, and they use a single tile size. Park et al. (2026-05) name token/neuron block grouping as future work. | Every tile re-pairs rows and columns freely, on a measured-cost ladder of five tile sizes (`64×32` carries 76.7% of the work). A table prepared once at model load (13.1 ms) plus a 1.49–2.83 ms per-batch plan that hides behind attention (0.000 ms exposed). On the same table it removes 60.9% more wasted `×0` cells than an upper-bound 5-tier BSMR. |
| **2** | **Reversed paradigm: fix the rosters first, then train the model to fit** | *Deja Vu* and *ShadowLLM* predict, after the fact, which of 2<sup>4096</sup> patterns the model fires, and collapse in large batches. *MoEfication* and *CMoE* split neurons into disjoint experts, so partly shared neurons cannot be shared. *AWC* defines no selection rule and no post-training. | No learned router: one dot product against a finite set of centroids (`M = [32, 32, 32, 256]`) selects an overlapping, variable-length roster before the layer runs. Family- and 32-column-block-aware post-training makes the model fit those rosters while keeping 99.77% of the base accuracy (90.81%), with `1* = 0%`. |
| **3** | **Per-token sparsity kept in large-batch SwiGLU FFN, measured end to end** | *PowerInfer* and *TurboSparse* target `B = 1`. *Polar Sparsity* gives up on large-batch SwiGLU FFN because the batch union fills up to 100%. Park et al. fall back to dense at large batch. *CMoE* reports 1.17× with coarse experts. | Per-token neuron sets are kept, not merged into a batch union. In pure `bf16`, one model is measured at `B = 1` (6.20–6.27×) and at `B = 512–2,048` (3.08–4.57× with attention overlap), across the densities of all four layers (0.8–22%). |

</details>

<details>
<summary><b>핵심 기여 3축 (한국어)</b></summary>

새로움이 강한 순서다 (1 → 2 → 3).

| 순위 | 핵심 기여 축 | 기존 학계가 막힌 지점 | 우리 해법의 핵심 차별점 |
|:---:|---|---|---|
| **1순위** | **실시간 비격자 5단계 타일링** *(+ 필요 시 `1*` 버퍼 일반 메커니즘)* | 최신 SDDMM(`BSMR · Fused3S · TC-GNN`)은 **고정 행 패널 경계**에 묶여 행 짝을 못 바꾸고 단일 틀·고정 `1`에 머묾. Park 등(2026-05)도 "토큰·뉴런 블록 그룹핑"을 향후 과제로만 남김 | 행 패널 경계를 깨고 **타일마다 행·열을 자유 재조합** + 5단계 실측 비용 사다리(`64×32` `76.7%`) + `13.1 ms` 사전 준비 표 & `1.49 ~ 2.83 ms` 실시간 타일링(어텐션 노출 `0.000 ms`). 똑같은 표에서도 `BSMR` 5단계 상한 대비 **`×0`을 `-60.9%`** 추가 절감 |
| **2순위** | **패러다임 역전: “예측기 추종 → 선제 유한 차등 명단 × 하드웨어 친화 공유 사후학습”** | *Deja Vu · ShadowLLM*은 모델의 2<sup>4096</sup>가지 산발 패턴을 **예측기가 뒤따라 맞히다** 대형 배치에서 붕괴. *MoEfication · CMoE*는 고정 방 분할로 부분 공용 뉴런 공유 불가. *AWC*는 고르는 법·사후학습 부재 | **출발점을 역전:** 학습 라우터 없이 단순 내적 1회로 **고정 차등 중심점(`M=[32,32,32,256]`) · 중복 허용 가변 `K` 명단**을 먼저 확정하고, **가족·32열 블록 압력 사후학습**으로 모델이 규칙에 맞춰 정확도(베이스 대비 `99.77%`(`90.81%`), `1* = 0%`)를 유지. 1순위 타일링·사전 준비 표와 4중 맞물림 |
| **3순위** | **대형 배치 SwiGLU FFN 토큰별 희소성 유지 (단일 · 서버 E2E 동시 실측)** | *PowerInfer · TurboSparse*는 `B = 1` 중심, *Polar Sparsity*는 합집합이 `100%`로 꽉 차는 문제로 SwiGLU 대형 배치 FFN을 포기(어텐션 우회), *Park 등*은 큰 배치에서 밀집 복귀, *CMoE*는 굵은 전문가 `1.17×` | 합집합으로 뭉개지 않고 **토큰별 뉴런 집합을 유지한 채** 순수 `bf16`으로 단일(`6.20 ~ 6.27×`)과 대형 서버(`3.08 ~ 4.57×` 중첩)를 동시 실측 + 4개 층 전 밀도 구간(`0.8 ~ 22%`) 밀도 무관 가속 |

</details>
