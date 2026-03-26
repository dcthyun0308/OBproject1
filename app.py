import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time

# 1. 페이지 설정 (웹 브라우저 탭 이름과 아이콘)
st.set_page_config(
    page_title="AI 보험료 예측 매니저",
    page_icon="💰",
    layout="centered"
)

# 모델 불러오기
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# -------------------------
# 사이드바: 입력 UI (UX 개선)
# -------------------------
with st.sidebar:
    st.header("📋 정보 입력")
    st.write("분석을 위해 아래 정보를 입력해주세요.")
    
    age = st.number_input("나이", min_value=18, max_value=100, value=25, step=1)
    sex = st.radio("성별", ["여성", "남성"], horizontal=True)
    
    st.divider()
    
    height = st.number_input("키 (cm)", min_value=100.0, max_value=220.0, value=170.0, step=0.1, format="%.1f")
    weight = st.number_input("몸무게 (kg)", min_value=30.0, max_value=200.0, value=65.0, step=0.1, format="%.1f")
    
    st.divider()
    
    children = st.select_slider("자녀 수", options=[0, 1, 2, 3, 4, 5], value=0)
    smoker = st.radio("흡연 여부", ["아니오", "예"], horizontal=True)
    
    st.info("💡 모든 정보를 입력한 후 메인 화면의 버튼을 눌러주세요.")

# -------------------------
# 메인 화면: 안내 및 디자인
# -------------------------
st.title("💰 AI 보험료 예측 서비스")
st.markdown("""
스마트한 AI가 당신의 건강 지표를 분석하여 **예상 의료비**와 **의료비 수준 등급**을 알려드립니다.
---
""")

# 발표용 가이드 (B의 기여 포인트)
col1, col2 = st.columns(2)
with col1:
    st.markdown("### 📱 How to use")
    st.write("1. 왼쪽 사이드바에 정보를 입력합니다.")
    st.write("2. '분석 및 예측 시작' 버튼을 누릅니다.")
with col2:
    st.markdown("### 🔍 분석 포인트")
    st.write("- BMI(비만도)와 흡연 여부의 상관관계")
    st.write("- 연령별 의료비 지출 패턴 분석")

st.divider()

# -------------------------
# 퍼센타일 기준 (고정값)
# -------------------------
q50 = 9382.033
q80 = 20260.626406

# -------------------------
# 버튼 및 예측 로직
# -------------------------

if st.button("🚀 분석 및 보험료 예측 시작", use_container_width=True):
    with st.spinner('AI가 데이터를 분석하고 예측 모델을 실행 중입니다...'):
        time.sleep(1.5) # 분석 애니메이션 효과

        # BMI 계산
        height_m = height / 100
        bmi = weight / (height_m ** 2)

        # 모델 입력값 맞추기
        sex_val = "male" if sex == "남성" else "female"
        smoker_val = "yes" if smoker == "예" else "no"

        is_obese = int(bmi >= 30)
        is_smoker = int(smoker_val == "yes")
        obese_smoker = is_obese * is_smoker

       # 기존 코드를 아래처럼 dtype을 명시하도록 수정
        input_df = pd.DataFrame({
            "age": [float(age)], # 숫자는 모두 실수형(float)으로 강제 변환
            "sex": [sex_val],
            "bmi": [float(bmi)],
            "children": [float(children)],
            "smoker": [smoker_val],
            "is_obese": [float(is_obese)],
            "is_smoker": [float(is_smoker)],
            "obese_smoker": [float(obese_smoker)]
        })

        # 예측 (log -> 원래값)
        pred_log = model.predict(input_df)[0]
        pred = np.expm1(pred_log)

        # -------------------------
        # 등급 (퍼센타일 기준) 및 시각적 피드백
        # -------------------------
        if pred < q50:
            status_color = "green"
            grade = "🟢 낮음 (Low Risk)"
            desc = "축하합니다! 의료비 지출이 매우 적은 우수 관리 그룹입니다."
            st.balloons() # 낮은 그룹일 때 축하 효과
        elif pred < q80:
            status_color = "orange"
            grade = "🟡 보통 (Average)"
            desc = "평균적인 의료비 지출 그룹입니다. 현재 건강을 유지하세요."
        else:
            status_color = "red"
            grade = "🔴 높음 (High Risk)"
            desc = "주의하세요! 상대적으로 높은 의료비 지출이 예상되는 고위험 그룹입니다."

        # 퍼센타일 위치 계산 (근사)
        if pred < q50:
            percentile = (pred / q50) * 50
        elif pred < q80:
            percentile = 50 + (pred - q50) / (q80 - q50) * 30
        else:
            percentile = 80 + (pred - q80) / q80 * 20
            percentile = min(percentile, 99.9)

        krw = pred * 1500

        # -------------------------
        # 결과 대시보드 출력 (UX 개선)
        # -------------------------
        st.subheader("🔍 분석 리포트")
        
        # 지표 카드 디자인
        m1, m2, m3 = st.columns(3)
        m1.metric("예상 보험료(USD)", f"${pred:,.0f}")
        m2.metric("예상 보험료(KRW)", f"₩{krw:,.0f}")
        m3.metric("BMI 지수", f"{bmi:.1f}")

        # 결과 안내 박스
        if status_color == "green":
            st.success(f"**의료비 등급: {grade}**\n\n{desc}")
        elif status_color == "orange":
            st.warning(f"**의료비 등급: {grade}**\n\n{desc}")
        else:
            st.error(f"**의료비 등급: {grade}**\n\n{desc}")

        # 분석 상세 설명
        with st.expander("📊 상세 분석 결과 보기"):
            st.write(f"본 예측은 사용자의 연령({age}세), BMI({bmi:.1f}), 흡연 여부({smoker}) 등을 종합적으로 고려한 결과입니다.")
            st.progress(int(percentile), text=f"전체 사용자 중 상위 {100 - percentile:.1f}% 수준")
            st.caption("※ 본 결과는 AI 모델에 의한 예측치이며 실제 보험사와 차이가 있을 수 있습니다.")

st.divider()
st.caption("© 2024 OB Project Team - Insurance Prediction Service")
