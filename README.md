# Roster

*Router-free neuron rosters for FFNs: each input region gets a fixed roster of neurons, so only the on-duty neurons are computed.*

FFN 가속 방법 (순수 bf16). RTX 4060 커널 실측 기준 단일 요청 5.33 ~ 6.55배(정확도 99.63%), 실제 서버(배치 512 ~ 2,048 · 5단계 다중 크기 + L2 열-스트립) 최대 2.56 ~ 2.72배(정확도 99.59%, 이론 상한 3.6 ~ 5.2배).

- **설명 페이지 (인터랙티브 시각화):** https://rhydcdc.github.io/Roster/
- **핵심 기여 시각화 바로가기:** https://rhydcdc.github.io/Roster/#contrib

<details>
<summary><b>Key Contributions (English)</b></summary>

| Contribution | Core Mechanism | vs. Prior Work |
| :--- | :--- | :--- |
| **Per-Centroid Front-Loading (`1*`) × Non-Grid 5-Tier Tile Covering** | Creates region-specific tail `1*` (don't-care) bits via front-loading, then breaks 1D row-boundary locks by freely pairing `(row_set, col_set)` per tile across a measured-cost ladder (`64×32 → 8×16`, free ON / auto OFF → `8×16` **0.2%**, acc **99.59%**). | *MatFormer* (static global prefix); *PIT* / *TileSpMM* (rigid `1`s, no don't-care); 1D SpMM reordering (grid-locked) |
| **Router-Free Finite Centroid Rosters & Automatic Batch Coalescing** | Single dot-product against `M = 256` centroids selects overlapping variable-length rosters (`K_j`), skipping `gate · up · down` upfront and automatically coalescing same-centroid tokens with zero accuracy loss as batch grows. | *Deja Vu* / *ShadowLLM* (per-token MLP predictor, uncoalesced); *MoEfication* (disjoint experts + router) |
| **Breaking Large-Batch Contextual FFN Collapse (Single & Server)** | Co-designing `Roster` + `1*` + `Roster-Tile` + L2 strips achieves pure-`bf16` RTX 4060 speedups on both `B = 1` (**5.33–6.55×**, 99.63%) and server batches `B = 512–2,048` (up to **2.56–2.72×**, 99.59%). | *PowerInfer* / *TurboSparse* / *CATS* (`B = 1` focused); *Polar Sparsity* (drops batched FFN for attention) |

</details>

<details>
<summary><b>핵심 기여 (한국어)</b></summary>

| 핵심 기여 | 핵심 원리 | 선행 연구 대비 차별점 |
| :--- | :--- | :--- |
| **중심점별 앞쪽 몰기(`1*`) × 비격자 5단계 타일 커버링** | 중심점별 고유 순위 앞쪽 몰기로 끝쪽 10% `1*`(Don't-Care)을 만들고, 1D 가로 경계를 깨고 타일마다 행·열을 자유롭게 짝짓는 5단계 커버링(`64×32 → 8×16`, 공짜 ON · 자동 OFF → `8×16` <b>0.2%</b>, 정확도 <b>99.59%</b>). | *MatFormer*(전역 정적) · *PIT*/*TileSpMM*(고정 `1` 제약) · 1D SpMM 재배열(가로 격자 한계)과 차별화 |
| **라우터 없는 유한 중심점 가변 명단 + 배치 자동 합승** | 256개 중심점 내적 1회로 겹침 허용 가변 명단(`K_j`)을 선제 확정해 `gate · up · down`을 스킵하고, 배치가 커질수록 같은 중심점 토큰이 손실 0%로 자동 합승. | *Deja Vu*·*ShadowLLM*(토큰별 MLP 예측 → 합승 불가) · *MoEfication*(겹침 없는 전문가 + 라우터)과 구별 |
| **대형 배치 FFN 붕괴 돌파 (단일 요청 & 대형 서버 동시 실측)** | `Roster` + `1*` + `Roster-Tile` + L2 스트립 코디자인으로 순수 `bf16` RTX 4060 실측 단일 `B = 1` <b>5.33 ~ 6.55×</b>(99.63%) · 대형 서버 `B = 512 ~ 2,048` 최대 <b>2.56 ~ 2.72×</b>(99.59%) 증명. | *PowerInfer*·*TurboSparse*·*CATS*(`B = 1` 중심) · *Polar Sparsity*(대형 배치 FFN 포기 → 어텐션 우회) 한계 돌파 |

</details>
