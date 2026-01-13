# XLSTM 기반 기상 예측 (단변량 · 다변량 시계열)

TensorFlow/Keras의 **Neural Network, LSTM, XLSTM(Long Short-Term Memory)** 모델을 사용해 **기온을 예측**하는 시계열 프로젝트입니다.  
단일 변수(기온)만 사용하는 **단변량 예측**과, 여러 기상 변수(기압/기온/공기밀도)를 함께 사용하는 **다변량 예측**을 모두 실험했습니다.

---

## What is xLSTM (eXtended LSTM)?
<img width="1085" height="500" alt="image" src="https://github.com/user-attachments/assets/07dbf2f4-e2fe-4991-a21b-12c56a73f4ca" />

xLSTM는 기존 LSTM을 **(1) exponential gating(지수 게이팅)** 과 **(2) 새로운 메모리 구조**로 확장한 모델 패밀리입니다. 큰 틀에서 다음 두 가지 변형을 포함합니다.

- **sLSTM (scalar LSTM)**  
  - 메모리는 **스칼라(cell state)** 기반이지만, input/forget gate에 **exponential gating**을 적용해 “언제/얼마나 저장할지” 결정을 더 유연하게 만들고,
  - 값 폭발(overflow)을 막기 위한 **stabilizer state**와 스케일을 안정화하는 **normalizer state**를 추가해 학습/추론을 안정화합니다.
- **mLSTM (matrix LSTM)**  
  - 메모리를 **행렬(matrix)** 로 확장해 저장 용량을 키우고, (특히 긴 문맥/희귀 패턴에 강점)
  - 특정 구조에서는 병렬화 가능한 형태로 설계될 수 있습니다.

> 이 프로젝트(WeatherForcasting.ipynb)는 xLSTM 중 **sLSTM 아이디어를 반영한 커스텀 RNN Cell(sLSTMCell)** 을 TensorFlow로 구현해, 기존 LSTM과 기상 시계열 예측 성능을 비교합니다.


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

### 4) XLSTM 기반 Weather Forecasting 
#### 4-1) 단변량 예측 (Univariate)
<img width="677" height="538" alt="image" src="https://github.com/user-attachments/assets/06e51508-a14f-4c74-b129-5b345a34187b" />

- 입력 윈도우: 과거 **20 step**
- 타깃: `target_size = 0` (현재 시점 값 예측 형태)
- 파이프라인: `tf.data.Dataset.cache().shuffle(10000).batch(256).repeat()`
- 모델
  - **Simple LSTM**: `LSTM(8) -> Dense(1)`
  - **xLSTM(sLSTM)**: `RNN(sLSTMCell(16)) -> Dense(1)`

#### 4-2) 다변량 단일 스텝 예측 (Multivariate Single-Step)
<img width="661" height="543" alt="image" src="https://github.com/user-attachments/assets/6ec293b6-94c3-437b-8fc2-0036ef0dd246" />

- 사용 피처: `p (mbar)`, `T (degC)`, `rho (g/m**3)`
- 입력 히스토리: `past_history = 720`, `STEP = 6`  
  - 즉, 과거 720 step을 6 step 간격으로 샘플링 → **120 step 입력**
- 예측 시점: `future_target = 72`  
  - 72 step = 12시간 뒤(10분 * 72)
- 모델
  - **LSTM**: `LSTM(32) -> Dense(1)` (optimizer=RMSprop, loss=MAE)
  - **xLSTM(sLSTM)**: `RNN(sLSTMCell(64)) -> Dense(1)` (optimizer=Adam, loss=MAE)

#### 4-3) 다변량 다중 스텝 예측 (Multivariate Multi-Step)
<img width="664" height="537" alt="image" src="https://github.com/user-attachments/assets/c4d97045-c2ad-40d9-abbd-2dd7d2afd0bd" />

- 입력: 위와 동일(과거 120 step, 피처 3개)
- 타깃: 미래 **72 step 시퀀스 전체**(multi-step)
- 모델
  - **xLSTM(sLSTM)**: `RNN(sLSTMCell(64)) -> Dense(72)` (optimizer=Adam, loss=MAE)


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

> 손실 함수는 모두 **MAE**(표준화된 값 기준)이며, Notebook 출력 일부는 `...`로 생략되어 **일부 실험은 최종 epoch 로그가 남아있지 않습니다.**

### 핵심 수치(로그에 남아있는 값 기준)

| 실험 | 모델 | 설정(요약) | Validation MAE(관측된 값) |
|---|---|---|---|
| Univariate | Simple LSTM | LSTM(8), Adam, 10 epochs | **~0.0183 (epoch 8까지 확인)** |
| Univariate | xLSTM(sLSTM) | sLSTMCell(16), Adam, 10 epochs | **0.0141 (epoch 10)** |
| Multivariate Single-Step | LSTM | LSTM(32), RMSprop, 10 epochs | **~0.2472 (epoch 2까지 확인)** *(이후 로그 출력 생략)* |
| Multivariate Single-Step | xLSTM(sLSTM) | sLSTMCell(64), Adam, 10 epochs | **0.2660 (epoch 10)** |
| Multivariate Multi-Step | xLSTM(sLSTM) | sLSTMCell(64), Adam, 10 epochs | **0.1810 (epoch 10)** |

Baseline(평균 예측): 변동성이 낮은 구간에서는 그럭저럭 맞지만, 추세/변화를 반영하지 못함

단변량 LSTM: 최근 과거 패턴을 학습해 baseline보다 예측값이 실제값에 더 근접

다변량 LSTM: 온도 외 변수까지 활용하여 추세/상호의존성 반영 가능성을 확인


