import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time
import sklearn
from sklearn.impute import SimpleImputer

# ---------------------------------------------------------
# [Step 1] 오류 해결을 위한 응급처치 (Monkey Patch)
# ---------------------------------------------------------
# 최신 sklearn의 SimpleImputer가 요구하는 내부 속성을 강제로 생성합니다.
if not hasattr(SimpleImputer, "_fill_dtype"):
    SimpleImputer._fill_dtype = property(lambda self: self.statistics_.dtype if hasattr(self, 'statistics_') else np.float64)

# ---------------------------------------------------------
# [Step 2] 페이지 설정 및 디자인
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI 보험료 예측 매니저",
    page_icon="💰",
    layout="centered"
)

# 모델 불러오기 (캐싱 적용으로 속도 향상)
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"모델 파일을 불러오는 중 오류가 발생했습니다: {e}")

# ---------------------------------------------------------
# [Step 3] 사이드바 UI (UX 개선)
# ---------------------------------------------------------
with st.sidebar:
    st.header("📋 정보 입력")
    st.write("사용자의 기본 정보를 입력해주세요.")
    
    age = st.number_input("나이", min_value=18, max_value=100, value=25, step=1)
    sex = st.radio("성별", ["여성", "남성"], horizontal=True)
    
    st.divider()
    
    height = st.number_input("키 (cm)", min_value=100.0, max_value=220.0, value=170.0, step=0.1, format="%.1f")
    weight = st.number_input("몸무게 (kg)", min_value=30.0, max_value=200.0, value=65.0, step=0.1, format="%.1f")
    
    st.divider()
    
    children = st.select_slider("자녀 수", options=[0, 1, 2, 3, 4, 5], value=0)
    smoker = st.radio("흡연 여부", ["아니오", "예"], horizontal=True)
    
    st.info("💡 사이드바에 정보를 입력한 후 메인 화면의 버튼을 눌러주세요.")

# ---------------------------------------------------------
# [Step 4] 메인 화면 디자인
# ---------------------------------------------------------
st.title("💰 AI 보험료 예측 서비스")
st.markdown("""
스마트한 AI가 당신의 건강 지표를 분석하여 **예상 의료비**와 **의료비 수준 등급**을 알려드립니다.
---
""")

# 📱 QR 안내 (UX 포인트)
st.success("📱 QR코드를 스캔해서 모바일에서도 간편하게 입력해보세요!")

# -------------------------
# 퍼센타일 기준 (고정값)
# -------------------------
q50 = 9382.033
q80 = 20260.626406

# ---------------------------------------------------------
# [Step 5] 버튼 클릭 시 예측 로직 실행
# ---------------------------------------------------------
if st.button("🚀 보험료 분석 및 예측 시작", use_container_width=True):
    with st.spinner('AI가 데이터를 정밀 분석 중입니다...'):
        time.sleep(1) # 시각적 효과를 위한 잠깐의 대기

        # BMI 계산
        height_m = height / 100
        bmi = weight / (height_m ** 2)

        # 모델 입력값 맞추기
        sex_val = "male" if sex == "남성" else "female"
        smoker_val = "yes" if smoker == "예" else "no"

        is_obese = int(bmi >= 30)
        is_smoker = int(smoker_val == "yes")
        obese_smoker = is_obese * is_smoker

        # 데이터프레임 생성 (타입 충돌 방지를 위해 float 강제 지정)
        input_df = pd.DataFrame({
            "age": [float(age)],
            "sex": [sex_val],
            "bmi": [float(bmi)],
            "children": [float(children)],
            "smoker": [smoker_val],
            "is_obese": [float(is_obese)],
            "is_smoker": [float(is_smoker)],
            "obese_smoker": [float(obese_smoker)]
        })

        try:
            # 예측 로직 (log -> 원래값)
            pred_log = model.predict(input_df)[0]
            pred = np.expm1(pred_log)

            # 등급 분류 및 시각적 피드백
            if pred < q50:
                grade = "🟢 낮음"
                desc = "의료비 지출이 적은 그룹입니다. 건강 관리가 매우 우수하시네요!"
                st.balloons()
            elif pred < q80:
                grade = "🟡 보통"
                desc = "평균적인 의료비 지출 그룹입니다. 현재 건강을 잘 유지해주세요."
            else:
                grade = "🔴 높음"
                desc = "상대적으로 높은 의료비 지출 그룹입니다. 생활 습관 개선을 권장합니다."

            # 퍼센타일 위치 계산
            if pred < q50:
                percentile = (pred / q50) * 50
            elif pred < q80:
                percentile = 50 + (pred - q50) / (q80 - q50) * 30
            else:
                percentile = 80 + (pred - q80) / q80 * 20
                percentile = min(percentile, 99.9)

            krw = pred * 1500

            # ---------------------------------------------------------
            # [Step 6] 결과 시각화 (B의 핵심 기여)
            # ---------------------------------------------------------
            st.divider()
            st.subheader("🔍 분석 결과 리포트")
            
            # 메트릭 카드
            c1, c2, c3 = st.columns(3)
            c1.metric("예상 보험료(USD)", f"${pred:,.0f}")
            c2.metric("예상 보험료(KRW)", f"₩{krw:,.0f}")
            c3.metric("BMI 지수", f"{bmi:.1f}")

            # 결과 메시지 박스
            if pred < q50:
                st.success(f"### 의료비 등급: {grade}\n{desc}")
            elif pred < q80:
                st.warning(f"### 의료비 등급: {grade}\n{desc}")
            else:
                st.error(f"### 의료비 등급: {grade}\n{desc}")

            # 상세 통계
            st.info(f"📊 전체 사용자 중 약 **상위 {100 - percentile:.1f}%** 수준의 지출이 예상됩니다.")
            st.progress(int(percentile))
            
        except Exception as e:
            st.error(f"예측 과정에서 기술적 오류가 발생했습니다. (오류 내용: {e})")
            st.info("💡 팀원 A에게 'scikit-learn 버전'을 확인해달라고 요청해보세요.")

st.divider()
st.caption("© 2026 OB Project Team - Insurance Prediction Service")
