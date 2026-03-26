import streamlit as st
import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.impute import SimpleImputer

# --- [B의 긴급 수술 코드] ---
# 구버전 모델이 최신 sklearn 환경에서 돌아가도록 속성을 강제로 주입합니다.
if not hasattr(SimpleImputer, "_fill_dtype"):
    SimpleImputer._fill_dtype = lambda self, X: X.dtype
# -------------------------

# 그 다음에 모델을 불러옵니다.
model = joblib.load("model.pkl")

st.title("💰 보험료 예측 서비스")
st.info("📱 QR코드를 스캔해서 직접 입력해보세요!")

st.write("아래 정보를 입력하면 예상 보험료를 알려드립니다.")

# -------------------------
# 입력 UI (한국어)
# -------------------------

age = st.number_input(
    "나이",
    min_value=18,
    max_value=100,
    value=None,
    step=1
)

sex = st.radio("성별", ["여성","남성"])

height = st.number_input(
    "키 (cm)",
    min_value=100.0,
    max_value=220.0,
    value=None,
    step=1.0,
    format="%.1f"
)

weight = st.number_input(
    "몸무게 (kg)",
    min_value=30.0,
    max_value=200.0,
    value=None,
    step=1.0,
    format="%.1f"
)
children = st.slider("자녀 수", 0, 5, 0)

smoker = st.radio("흡연 여부", ["아니오","예"])

# -------------------------
# 퍼센타일 기준 (고정값)
# -------------------------
q50 = 9382.033
q80 = 20260.626406

# -------------------------
# 버튼
# -------------------------

if st.button("보험료 예측하기"):

    # BMI 계산
    height_m = height / 100
    bmi = weight / (height_m ** 2)

    # 모델 입력값 맞추기
    sex_val = "male" if sex == "남성" else "female"
    smoker_val = "yes" if smoker == "예" else "no"

    is_obese = int(bmi >= 30)
    is_smoker = int(smoker_val == "yes")
    obese_smoker = is_obese * is_smoker

    input_df = pd.DataFrame({
        "age": [age],
        "sex": [sex_val],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker_val],
        "is_obese": [is_obese],
        "is_smoker": [is_smoker],
        "obese_smoker": [obese_smoker]
    })

    # 예측 (log -> 원래값)
    pred_log = model.predict(input_df)[0]
    pred = np.expm1(pred_log)

    # -------------------------
    # 등급 (퍼센타일 기준)
    # -------------------------
    if pred < q50:
        grade = "🟢 낮음"
        desc = "의료비 지출이 적은 그룹입니다"
    elif pred < q80:
        grade = "🟡 보통"
        desc = "평균적인 의료비 지출 그룹입니다."
    else:
        grade = "🔴 높음"
        desc = "상대적으로 높은 의료비 지출 그룹입니다."

    # -------------------------
    # 퍼센타일 위치 계산 (근사)
    # -------------------------
    if pred < q50:
        percentile = (pred / q50) * 50
    elif pred < q80:
        percentile = 50 + (pred - q50) / (q80 - q50) * 30
    else:
        percentile = 80 + (pred - q80) / q80 * 20
        percentile = min(percentile, 99.9)

    krw = pred * 1500

    # -------------------------
    # 결과 출력
    # -------------------------
    st.subheader(f"💰 예상 연간 미국 의료비: ${pred:,.0f} (약 ₩{krw:,.0f})")
    st.subheader(f"🏆 의료비 수준 등급: {grade}")
    st.write(desc)
    st.write(f"📊 전체 사용자 중 약 상위 {100 - percentile:.1f}% 수준입니다.")
