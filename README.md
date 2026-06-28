# CKD-prediction
🌐 **Live Demo**: https://ckd-prediction-bridy.streamlit.app/
# 🩺 만성신장질환(CKD) 예측 시스템

## 프로젝트 개요
UCI Machine Learning Repository의 만성신장질환 데이터셋을 활용하여
환자의 검사 수치를 입력하면 CKD 여부를 예측하는 머신러닝 모델과 웹 애플리케이션을 개발했습니다.

생명과학 석사 및 연구원 경력을 바탕으로 데이터 분석 결과에 의학적 해석을 더했습니다.

---

## 데이터셋
- **출처**: UCI Machine Learning Repository - Chronic Kidney Disease Dataset
- **데이터 크기**: 397명, 25개 변수
- **목표 변수**: CKD 여부 (ckd / notckd)

---

## 주요 분석 내용

### 1. 데이터 전처리
- 결측값 처리 (수치형: 중앙값, 범주형: 최빈값)
- 이상 문자열 제거 (탭 문자, 공백 등)
- 범주형 변수 인코딩

### 2. 탐색적 데이터 분석 (EDA)
- CKD vs Non-CKD 그룹별 주요 수치 분포 비교
- 변수 간 상관관계 히트맵 분석

### 3. 주요 발견
- **헤모글로빈(hemo), 적혈구용적률(pcv)**: CKD 환자에서 현저히 낮음 → 신장의 EPO 분비 저하로 인한 빈혈
- **크레아티닌(sc), 혈중요소(bu)**: CKD 환자에서 높음 → 신장의 노폐물 여과 기능 저하
- **혈당(bgr), 요당(su)**: 양의 상관관계 → 당뇨와 CKD의 연관성

### 4. 머신러닝 모델
| 항목 | 결과 |
|------|------|
| 모델 | Random Forest Classifier |
| 정확도 | 96.25% |
| CKD 재현율 | 100% |
| Non-CKD 정밀도 | 100% |

### 5. 변수 중요도 Top 5
1. hemo (헤모글로빈)
2. pcv (적혈구용적률)
3. sc (크레아티닌)
4. rbcc (적혈구 수)
5. sg (요비중)

---

## 웹 애플리케이션
Streamlit을 활용하여 환자 수치를 입력하면 CKD 여부와 확률을 실시간으로 예측하는 웹앱을 개발했습니다.

### 실행 방법
```bash
pip install streamlit
streamlit run CKD_app.py
```

---

## 기술 스택
- **Language**: Python
- **Library**: pandas, numpy, scikit-learn, matplotlib, seaborn, streamlit
- **Model**: Random Forest Classifier

---

## 파일 구조
```
CKD-prediction/
├── CKD1.ipynb                      # 데이터 전처리 및 EDA
├── CKD2.ipynb                      # 머신러닝 모델 및 예측
├── CKD_app.py                      # Streamlit 웹 애플리케이션
└── chronic_kidney_disease.arff     # 데이터셋
```
