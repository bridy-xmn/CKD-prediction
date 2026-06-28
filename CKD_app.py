import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 데이터 로드 및 모델 학습
@st.cache_resource
def load_model():
    with open("chronic_kidney_disease.arff", "r") as f:
        lines = f.readlines()

    data_start = [i for i, line in enumerate(lines) if line.strip().upper() == '@DATA'][0] + 1
    data_lines = lines[data_start:]
    columns = [line.split()[1] for line in lines if line.strip().upper().startswith('@ATTRIBUTE')]

    data_str = ''.join(data_lines)
    data_df = pd.read_csv(StringIO(data_str), header=None, names=columns, na_values='?', on_bad_lines='skip')
    data_df.columns = data_df.columns.str.replace("'", "").str.strip()

    obj_cols = data_df.select_dtypes(include='object').columns
    for col in obj_cols:
        data_df[col] = data_df[col].str.strip()

    for col in ['pcv', 'wbcc', 'rbcc']:
        data_df[col] = pd.to_numeric(data_df[col], errors='coerce')

    num_cols = data_df.select_dtypes(include='float64').columns
    data_df[num_cols] = data_df[num_cols].fillna(data_df[num_cols].median())

    obj_cols = data_df.select_dtypes(include='object').columns
    for col in obj_cols:
        data_df[col] = data_df[col].fillna(data_df[col].mode()[0])

    le = LabelEncoder()
    for col in obj_cols:
        data_df[col] = le.fit_transform(data_df[col])

    X = data_df.drop('class', axis=1)
    y = data_df['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    return model, X.columns

model, feature_cols = load_model()

# 앱 UI
st.title('🩺 만성신장질환(CKD) 예측 시스템')
st.write('환자의 검사 수치를 입력하면 CKD 여부를 예측합니다.')

st.subheader('기본 정보')
col1, col2 = st.columns(2)
with col1:
    age = st.number_input('나이', 0, 100, 50)
    bp = st.number_input('혈압 (bp)', 0, 200, 80)
    sg = st.number_input('요비중 (sg)', 1.000, 1.030, 1.020, step=0.001, format="%.3f")
    al = st.number_input('알부민 (al)', 0, 5, 0)
    su = st.number_input('요당 (su)', 0, 5, 0)
with col2:
    bgr = st.number_input('혈당 (bgr)', 0, 500, 100)
    bu = st.number_input('혈중요소 (bu)', 0, 200, 20)
    sc = st.number_input('크레아티닌 (sc)', 0.0, 20.0, 1.0, step=0.1)
    sod = st.number_input('나트륨 (sod)', 0, 200, 140)
    pot = st.number_input('칼륨 (pot)', 0.0, 10.0, 4.5, step=0.1)

st.subheader('혈액 검사')
col3, col4 = st.columns(2)
with col3:
    hemo = st.number_input('헤모글로빈 (hemo)', 0.0, 20.0, 15.0, step=0.1)
    pcv = st.number_input('적혈구용적률 (pcv)', 0, 60, 44)
    wbcc = st.number_input('백혈구 수 (wbcc)', 0, 20000, 7800)
    rbcc = st.number_input('적혈구 수 (rbcc)', 0.0, 10.0, 5.2, step=0.1)
with col4:
    rbc = st.selectbox('적혈구 이상 (rbc)', [('정상', 1), ('비정상', 0)], format_func=lambda x: x[0])
    pc = st.selectbox('농뇨 (pc)', [('정상', 1), ('비정상', 0)], format_func=lambda x: x[0])
    pcc = st.selectbox('농뇨세포 (pcc)', [('없음', 0), ('있음', 1)], format_func=lambda x: x[0])
    ba = st.selectbox('세균뇨 (ba)', [('없음', 0), ('있음', 1)], format_func=lambda x: x[0])

st.subheader('병력')
col5, col6 = st.columns(2)
with col5:
    htn = st.selectbox('고혈압 (htn)', [('없음', 0), ('있음', 1)], format_func=lambda x: x[0])
    dm = st.selectbox('당뇨 (dm)', [('없음', 0), ('있음', 1)], format_func=lambda x: x[0])
    cad = st.selectbox('관상동맥질환 (cad)', [('없음', 0), ('있음', 1)], format_func=lambda x: x[0])
with col6:
    appet = st.selectbox('식욕 (appet)', [('좋음', 1), ('나쁨', 0)], format_func=lambda x: x[0])
    pe = st.selectbox('부종 (pe)', [('없음', 0), ('있음', 1)], format_func=lambda x: x[0])
    ane = st.selectbox('빈혈 (ane)', [('없음', 0), ('있음', 1)], format_func=lambda x: x[0])

if st.button('예측하기'):
    patient = pd.DataFrame([[
        age, bp, sg, al, su, rbc[1], pc[1], pcc[1], ba[1],
        bgr, bu, sc, sod, pot, hemo, pcv, wbcc, rbcc,
        htn[1], dm[1], cad[1], appet[1], pe[1], ane[1]
    ]], columns=feature_cols)

    prediction = model.predict(patient)[0]
    probability = model.predict_proba(patient)[0]

    st.subheader('예측 결과')
    if prediction == 0:
        st.error(f'⚠️ CKD (만성신장질환) 의심\nCKD 확률: {probability[0]*100:.1f}%')
    else:
        st.success(f'✅ 정상 (Non-CKD)\n정상 확률: {probability[1]*100:.1f}%')