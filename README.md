# PV Power Forecasting & Reforecasting Pipeline  
**(with Feature Engineering, SHAP Explainability, and Model-Based Feature Selection)**

본 프로젝트는 태양광 발전소(PV Stations)의  
**단기 발전량 예측(Forecast)** 및  
**오차 기반 재예측(Reforecast)** 기능을 갖춘 전체 파이프라인입니다.  

또한 **SHAP Explainability 분석**,  
**Model-based Feature Selection 기반 재학습** 기능까지 포함합니다.

---

## 📂 프로젝트 구성

### **0) py  파일과 ipynb 파일**
해당 프로젝트에는 py 파일과 ipynb 파일이 혼재되어 있습니다.  
py 파일은 데이터 전처리 ~ 결과 데이터 셋 까지의 프레임워크입니다.  
해당 파일들의 함수들은 모두 main.py 에 파사드 형태로 묶여 있으며, main.py 를 실행함으로 데이터프레임워크을 실행할 수 있습니다.  

ipynb 파일들은 데이터프레임워크 외에 저희가 데이터를 분석한 예시들입니다.  
해당 파일은 실행자가 어떤 데이터셋을 어떻게 분석할지 모르기 때문에 정리가 불가했으므로, 본 연구원들이 진행한 분석 예시로 구성되어 있습니다.  
또한 데이터 분석에는 매번 칼럼을 찍어보거나, plot 을 분석할 일이 많기 때문에 이를 py 로 정리하는 것은 무의미하다 판단하여, ipynb 파일로 그 예시를 제공하고 있습니다.  

### **1) 전처리 (Preprocessing)**
모듈: `handle_data_with_preprocessing.py`

- Raw Excel 데이터를 읽고(DataFrame List)
- 결측치 처리(Interpolation)
- 날짜/시간 기반 Feature 추가
- 7일 윈도우 기반 통계 Feature 추가
- Lag Feature(1~8 step) & Diff Feature 추가  
- 전처리 결과를 `lag_added_dataset/*.csv` 로 저장

---

### **2) 기본 예측(Base Forecast)**  
모듈: `forecast_reforecast.py`

사용 모델:
- **MLR (Linear Regression)**
- **SVR**
- **LightGBM**
- **MLP**

작업 내용:
1. 피처 리스트 기반 모델 훈련  
2. 시계열 기반 Train/Test Split  
3. 모델별 예측 수행  
4. t+1 Shift된 예측값 저장  
5. 예측 CSV 저장  
6. 운전 시간(Operation Hours) 기반 필터링  
7. MAE / MSE / RMSE 성능 평가 출력

---

### **3) 재예측(Reforecast)**  
모듈: `forecast_reforecast.py`

Base Forecast의 오차를 다시 예측하여 미래 예측값을 보정하는 방식입니다.

- Target 예시:
error_mlr, error_svr, error_lgb, error_mlp

작업 내용:
1. Reforecast Feature 리스트 적용  
2. 재예측 모델(MLR/SVR/LGB/MLP) 학습  
3. 예측 오차 기반 재예측 PV 계산  
4. Reforecast CSV 저장  
5. 운전 시간 필터 후 성능 평가

---

### **4) SHAP 기반 Feature Importance 분석**
모듈: `SHAP.py`  
함수: `do_reforecast_train_shap`, `do_forecast_train_shap`

지원 기능:
- 각 모델(Mlr, Svr, Lgb, Mlp)에 대한 SHAP 중요도 계산  
- Summary Plot 출력  
- Mean |SHAP| 기반 Ranking  
- Elbow Point 자동 검출 (Distance-to-Line метода)  
- Feature Selection을 위한 상위 피처 자동 선택

---

### **5) Feature Selection 기반 재학습**
모듈:
- `forecast_after_feature_selection.py`  
- `reforecast_after_feature_selection.py`

작업 내용:
- SHAP으로 선정된 Best Features만 사용하여  
Forecast / Reforecast 재학습  
- 성능 비교  
- 결과 저장

---

### **6) 전체 파이프라인 실행 (main.py)**

전체 실행 순서 요약:

1. 원시 데이터 로드  
2. 결측치 제거 & 시간 Feature 추가  
3. Feature-added CSV 저장  
4. Base Forecast 수행 및 평가  
5. Reforecast 위한 Target/Shifted Target 생성  
6. Reforecast 수행 및 평가  
7. SHAP 분석 및 결과 저장  
8. Feature Selection 기반 Forecast/Reforecast 재학습

---

## 📁 디렉토리 구조 예시

project/  
│  
├── dataset/  
│ ├── dataset/solar_stations/.xlsx  
│ ├── lag_added_dataset/.csv  
│ └── result_of_paper/  
│ ├── forecast/  
│ ├── reforecast/  
│ ├── feature_selection/  
│ └── ann/  
│  
├── handle_data_with_preprocessing.py  
├── forecast_reforecast.py  
├── SHAP.py  
├── forecast_after_feature_selection.py  
├── reforecast_after_feature_selection.py  
├── feature_lists.py  
└── main.py  


---

## 🧪 실행 방법

### 1) 가상환경 생성
```bash
python -m venv venv
source venv/bin/activate        # Windows → venv\Scripts\activate
```

### 2) 필요 패키지 설치
```pip install -r requirements.txt```

### 3) main.py 실행
```
python main.py
초기 실행 시 전체 파이프라인이 자동 실행됩니다.
```

## 📝 주요 설정값
이 들은 원하는 main.py 코드 내에서 원하는 테스트 데이터 비율, forecast, reforecast 모델 설정에 따라 변경시키면 됩니다.

### 테스트 데이터 비율
test_size = 2 / 24
### 예측(Forecast) 모델 설정
forecast_models = ['mlr', 'svr', 'lgb', 'mlp']
### 재예측(Reforecast) 모델 설정
reforecast_models = ['mlr', 'svr', 'lgb', 'mlp']


# ExoTFT-CRC
