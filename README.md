# Roster

*Router-free neuron rosters for FFNs: each input region gets a fixed roster of neurons, so only the on-duty neurons are computed.*

FFN 가속 방법 (순수 bf16). RTX 4060 커널 실측 기준 단일 요청 5.33 ~ 6.55배(정확도 99.63%), 실제 서버(배치 512 ~ 2,048 · 5단계 다중 크기 + L2 열-스트립) 최대 2.56 ~ 2.72배(정확도 99.59%, 이론 상한 3.6 ~ 5.2배).

**설명 페이지: https://rhydcdc.github.io/Roster/**
