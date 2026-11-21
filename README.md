# Roster

*Router-free neuron rosters for FFNs: each input region gets a fixed roster of neurons, so only the on-duty neurons are computed.*

FFN의 층마다, 층 입력과 가장 가까운 중심점을 찾아 그 중심점의 뉴런 명단(당번표)에서 앞 K개만 계산한다. 고르는 데 신경망을 쓰지 않고, 고른 뉴런만으로 버티게 사후 학습한다. Fashion-MNIST FFN 분류기에서 입력 하나가 켜는 뉴런은 평균 13.73%, 시험 정확도는 90.68%다(뉴런을 다 켠 모델 91.02%).

**설명 페이지: https://rhydcdc.github.io/Roster/**

코드는 [`code/`](code/). 학습된 모델은 나중에 공개한다.
