# Roster

*Router-free neuron rosters for FFNs: each input region gets a fixed roster of neurons, so only the on-duty neurons are computed.*

FFN 가속 방법 (순수 bf16). RTX 4060 커널 실측 기준 단일 요청 5.33 ~ 6.55배(정확도 99.63%), 실제 서버(배치 512 ~ 2,048 · 5단계 다중 크기 + L2 열-스트립) 최대 2.56 ~ 2.72배(정확도 99.59%, 이론 상한 3.6 ~ 5.2배).

**설명 페이지: https://rhydcdc.github.io/Roster/**

<details>
<summary><b>Key Contributions (English)</b></summary>

| Contribution | Core Idea | Differentiation from Prior Work |
| :--- | :--- | :--- |
| **Algorithm–Hardware Co-Design for Large-Batch Contextual FFN Sparsity** | Contextual sparsity typically collapses at large batch sizes because the union of active neurons approaches 100%. By co-designing finite input-centroid rosters (`Roster`), don't-care tail sparsity (`1*`), and non-grid 5-tier tile covering (`Roster-Tile`), Roster achieves measured pure-`bf16` GPU speedups on RTX 4060 across both single-request (`B = 1`, **5.33–6.55×**) and large server batches (`B = 512–2,048`, up to **2.56–2.72×**). | Directly overcomes the large-batch FFN collapse where *Deja Vu*, *PowerInfer*, *TurboSparse*, and *CATS* focus on `B = 1` / low concurrency, and *Polar Sparsity* abandons batched FFN sparsity in favor of attention-head sparsity. |
| **Per-Centroid Front-Loading Training & Don't-Care (`1*`) Tile-Covering Co-Design** | Post-training applies per-centroid rank front-loading pressure to deliberately create an optional tail 10% (`1*`, don't-care) per region. The tile solver exploits this freedom directly: `1*` cells falling inside already-opened tiles ride for free (ON), while isolated `1*` cells that would open new tiles are automatically dropped (OFF), preserving **99.59%** of base accuracy. | Extends *MatFormer*'s static global prefix nesting to per-centroid dynamic rankings, and introduces don't-care flexibility into Tensor Core tile covering—unlike *PIT* and *TileSpMM*, which treat every `1` as a rigid constraint. |
| **Router-Free Finite Input-Centroid Variable Rosters & Automatic Batch Coalescing** | Maps input hidden states via a single nearest-centroid lookup (`M = 256`) to overlapping, variable-length per-centroid neuron rosters (`K_j`), eliminating separate predictor/router networks and skipping `gate`, `up`, and `down` ahead of execution. Because the codebook is finite, tokens sharing a centroid automatically coalesce with zero accuracy change as batch size grows, compressing the tiling row space and enlarging dense tile blocks. | Avoids per-token uncoalesced MLP prediction (*Deja Vu*, *ShadowLLM*) and disjoint neuron-expert partitioning with learned routers (*MoEfication*). |
| **Non-Grid Free Row–Column Multi-Scale Cost-Weighted Tile Covering (`Roster-Tile`)** | Breaks the 1D global row/column permutation bottleneck (RCM/BCSR) by exploiting column-independent SwiGLU stage-1 outputs (`atomicAdd = 0`) to pair arbitrary `(row_set, col_set)` subsets independently per tile. Tiles are selected by measured GPU execution cost across a 5-tier frame ladder (`64×32 → 32×32 → 32×16 → 16×16 → 8×16`), driving slow `8×16` tiles down to **0.2%** inside a single strip-ordered Tensor Core kernel. | Distinct from *PIT* (SOSP '23, dynamic micro-tile gathering) and *TileSpMM* (ICPP '26, static SpMM variable tiling) by targeting dynamic `1×1` neuron-level batched SwiGLU FFNs with don't-care `1*` bits and a silicon cost-weighted frame ladder. |

</details>

<details>
<summary><b>핵심 기여 (한국어)</b></summary>

| 핵심 기여 | 핵심 내용 | 선행 연구 대비 차별점 |
| :--- | :--- | :--- |
| **대형 배치 FFN 문맥적 희소성 붕괴를 뚫은 알고리즘–하드웨어 코디자인** | 기존 문맥적 희소성은 배치가 커지면 활성 뉴런 합집합이 100%에 육박해 붕괴한다. 유한 중심점 명단(`Roster`), 꺼도 되는 `1*` 학습, 비격자 5단계 타일 커버링(`Roster-Tile`)을 맞물려 단일 요청(`B = 1`, **5.33 ~ 6.55×**)뿐 아니라 대형 서버 배치(`B = 512 ~ 2,048`)에서도 순수 `bf16` RTX 4060 실측 **최대 2.56 ~ 2.72×**를 달성했다. | *Deja Vu*, *PowerInfer*, *TurboSparse*, *CATS* (`B = 1`·저동시성 중심), *Polar Sparsity* (대형 배치 FFN 포기 → 어텐션 우회)의 한계를 정면 돌파 |
| **중심점별 앞쪽 몰기 학습 → ‘꺼도 되는 1(`1*`, Don't-Care)’ 타일 커버링 연동** | 중심점마다 다른 뉴런 순위로 앞쪽 몰기 학습 압력을 걸어 끝쪽 10% `1*` 자유도를 의도적으로 생성하고, 타일 솔버에서 **열린 타일 안은 공짜 ON · 새 타일을 여는 외딴 `1*`은 자동 OFF**로 처리해 베이스 정확도의 **99.59%**를 유지했다. | *MatFormer* (전역 정적 앞쪽 몰기만 사용), *PIT*·*TileSpMM* (모든 `1`이 고정 제약, Don't-Care 개념 없음)과 뚜렷하게 차별화 |
| **라우터 없는 유한 입력 중심점 가변 명단(`Roster`) + 배치 자동 합승** | 별도 예측기·라우터 없이 입력 은닉 상태의 최근접 중심점(`M = 256`) 내적 1회만으로 겹침을 허용하는 가변 길이 명단(`K_j`)을 선제 확정해 `gate · up · down`을 모두 건너뛴다. 유한 코드북 구조 덕분에 배치가 커질수록 출력 손실 0%의 **같은 중심점 자동 합승**이 일어나 타일링 행 공간이 압축된다. | *Deja Vu*·*ShadowLLM* (토큰별 난잡한 MLP 예측 → 합승 불가), *MoEfication* (뉴런을 겹침 없는 전문가로 분할 + 학습 라우터 필요)과 구별 |
| **SwiGLU 전용 비격자 자유 행·열 조합 + 실측 비용 5단계 타일 커버링 (`Roster-Tile`)** | 전역 1D 행·열 재배열(RCM·BCSR)의 한계를 깨고 SwiGLU 1단계의 열 독립성(`atomicAdd = 0`)을 활용해 타일마다 독립 `(R_t, C_t)`를 자유롭게 짝짓는다. GPU 실측 비용 사다리(`64×32 → 32×32 → 32×16 → 16×16 → 8×16`)로 커버링해 저속 `8×16` 비중을 **0.2%**까지 줄이고 단일 스트립 통합 커널로 실행한다. | *PIT* (SOSP '23, 동적 마이크로타일 Gather), *TileSpMM* (ICPP '26, 정적 SpMM 가변 타일) 대비 **동적 `1×1` 뉴런 단위 배치 FFN + `1*` 자유도 + 실측 비용 사다리**로 차별화 |

</details>
