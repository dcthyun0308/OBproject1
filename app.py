import streamlit as st
import numpy as np
import pandas as pd
import joblib
import sklearn

# --- [근본 해결: 구버전 모델 호환성 패치] ---
from sklearn.impute import SimpleImputer
# 최신 환경에서 구버전 모델의 missing attribute(_fill_dtype)를 채워주는 로직입니다.
if not hasattr(SimpleImputer, "_fill_dtype"):
    SimpleImputer._fill_dtype = property(lambda self: self.statistics_.dtype if hasattr(self, 'statistics_') else np.float64)
# ------------------------------------------

# 모델 불러오기
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

st.title("💰 보험료 예측 서비스")
st.info("📱 QR코드를 스캔해서 직접 입력해보세요!")

# -------------------------
# 입력 UI (가독성 개선)
# -------------------------
age = st.number_input("나이", min_value=18, max_value=100, value=25, step=1)
sex = st.radio("성별", ["여성", "남성"], horizontal=True)

col1, col2 = st.columns(2)
with col1:
    height = st.number_input("키 (cm)", min_value=100.0, max_value=220.0, value=170.0, step=0.1, format="%.1f")
with col2:
    weight = st.number_input("몸무게 (kg)", min_value=30.0, max_value=200.0, value=65.0, step=0.1, format="%.1f")

children = st.slider("자녀 수", 0, 5, 0)
smoker = st.radio("흡연 여부", ["아니오", "예"], horizontal=True)

# -------------------------
# 예측 로직
# -------------------------
if st.button("🚀 보험료 예측하기", use_container_width=True):
    # 1. 수치 계산
    height_m = height / 100
    bmi = weight / (height_m ** 2)

    # 2. 인코딩 (글자 -> 숫자) 
    # 모델이 학습될 때 사용된 기준(남성=1, 여성=0 / 예=1, 아니오=0)으로 변환합니다.
    sex_num = 1.0 if sex == "남성" else 0.0
    smoker_num = 1.0 if smoker == "예" else 0.0
    is_obese = 1.0 if bmi >= 30 else 0.0
    
    # 3. 데이터프레임 구성 (모두 수치형으로 변환하여 전달)
    input_df = pd.DataFrame({
        "age": [float(age)],
        "sex": [sex_num],
        "bmi": [float(bmi)],
        "children": [float(children)],
        "smoker": [smoker_num],
        "is_obese": [is_obese],
        "is_smoker": [smoker_num],
        "obese_smoker": [float(is_obese * smoker_num)]
    })

    try:
        # 4. 결과 도출
        pred_log = model.predict(input_df)[0]
        pred = np.expm1(pred_log)
        krw = pred * 1500

        # 등급 판정
        q50, q80 = 9382.033, 20260.626
        if pred < q50:
            grade, color = "🟢 낮음", "green"
        elif pred < q80:
            grade, color = "🟡 보통", "orange"
        else:
            grade, color = "🔴 높음", "red"

        # 5. 시각적 출력
        st.divider()
        st.subheader(f"📊 분석 결과")
        st.metric("예상 연간 의료비", f"${pred:,.0f}", f"약 ₩{krw:,.0f}")
        st.markdown(f"### 의료비 지출 등급: :{color}[{grade}]")
        st.balloons()
        
    except Exception as e:
        st.error(f"⚠️ 모델 계산 중 오류가 발생했습니다: {e}")
