# LSTM_Multivariate_Weatherforcasting

# LSTM 기반 기상 예측 (단변량 · 다변량 시계열)

TensorFlow/Keras의 **LSTM(Long Short-Term Memory)** 모델을 사용해 **기온을 예측**하는 시계열 프로젝트입니다.  
단일 변수(기온)만 사용하는 **단변량 예측**과, 여러 기상 변수(기압/기온/공기밀도)를 함께 사용하는 **다변량 예측**을 모두 실험했습니다.

---

## 프로젝트 목표

- **Jena Climate 데이터셋**으로부터 기상 시계열 패턴을 학습
- **Baseline(평균 예측)** 대비 LSTM의 예측 성능(정성적 시각화 기준) 확인
- 단변량/다변량 입력 구성에 따른 예측 특성 비교

---

## 데이터셋
<img width="1471" height="182" alt="image" src="https://github.com/user-attachments/assets/9f942ef8-4f2b-4ab5-9791-91161d33d7e0" />


- 데이터: **Jena Climate (2009-01-10 ~ 2016-12-31)**
- 관측 주기: **10분 간격**
- 특징(feature): 온도, 기압, 습도 등 **총 14개 변수** 포함
- 본 프로젝트에서 사용한 주요 변수
  - 단변량: `T (degC)` (기온)
  - 다변량: `p (mbar)`, `T (degC)`, `rho (g/m**3)`

> 노트북에서 `tf.keras.utils.get_file()`로 데이터(zip)를 자동 다운로드합니다.

---

## 실험 구성

### 1) Baseline (단순 평균 예측)
<img width="694" height="540" alt="image" src="https://github.com/user-attachments/assets/2434af91-e5ed-4622-a86e-c9faae1bbbde" />

- 과거 입력 구간(History)의 **평균값을 그대로 예측값**으로 사용
- 관찰 결과(시각화 기준):  
  - 안정 구간에서는 근사치가 가능하지만, **추세/변동을 반영하지 못해 한계**가 뚜렷함

### 2) 단변량 LSTM (Univariate)
<img width="680" height="536" alt="image" src="https://github.com/user-attachments/assets/6e91c275-d0d0-4f1d-bec7-45b3014d0649" />

- 입력: 과거 **20 step**의 기온(`T`)
- 타깃: **현재 시점 값**(offset 0)
- 모델(구성 요약)
  - `LSTM(8)` → `Dense(1)`
  - Optimizer: `Adam`, Loss: `MAE`
- 관찰 결과(시각화 기준)
  - Baseline 대비 **True Future에 더 근접한 예측**을 보이며, 국소 패턴을 더 잘 따라감

### 3) 다변량 LSTM (Multivariate, Single-step)
<img width="664" height="538" alt="image" src="https://github.com/user-attachments/assets/362badb4-0ca5-4cea-a65e-885be4d16293" />

- 입력 변수: `p`, `T`, `rho`
- 전처리: 학습 구간 기준 **표준화(mean/std)**
- 윈도우 설정(프로젝트 내 설정값)
  - `past_history = 720`
  - `future_target = 72`
  - `STEP = 6`
  - (Jena 데이터가 10분 단위이므로, 위 설정은 일반적으로 “과거 5일(시간 단위 샘플링)로부터 약 12시간 이후” 패턴을 다루는 구성으로 해석 가능)
- 모델(구성 요약)
  - `LSTM(32)` → `Dense(1)`
  - Optimizer: `RMSprop`, Loss: `MAE`
- 관찰 결과(시각화 기준)
  - 기온 단일 변수뿐 아니라 **기압/밀도와의 상관관계를 함께 활용**할 수 있어, 단변량보다 정보량이 풍부
  - 일부 구간에서 추세 반영이 더 자연스럽게 나타남

### 4) (준비) Multi-step용 데이터 형태 생성
- `x_train_multi`, `y_train_multi` 형태로 **미래 시퀀스 타깃**을 만들 수 있도록 데이터 생성까지 포함
- 향후 Seq2Seq/다중 스텝 예측 모델로 확장 가능

---

## 기술 스택

- Python
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib

---

## 실행 방법

### 1) 환경 준비

pip install tensorflow pandas numpy matplotlib


### 2) 노트북 실행

WeatherForcasting.ipynb 실행

데이터는 자동 다운로드되며, 전처리/학습/시각화까지 순서대로 진행됩니다.



### 주요 결과 요약 

Baseline(평균 예측): 변동성이 낮은 구간에서는 그럭저럭 맞지만, 추세/변화를 반영하지 못함

단변량 LSTM: 최근 과거 패턴을 학습해 baseline보다 예측값이 실제값에 더 근접

다변량 LSTM: 온도 외 변수까지 활용하여 추세/상호의존성 반영 가능성을 확인


