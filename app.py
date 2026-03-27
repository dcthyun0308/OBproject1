import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time
from sklearn.impute import SimpleImputer  # 이 줄을 추가하세요


if not hasattr(SimpleImputer, "_fill_dtype"):
    def get_fill_dtype(self):
        return self.statistics_.dtype if hasattr(self, 'statistics_') else np.float64
    SimpleImputer._fill_dtype = property(get_fill_dtype)
# ---------------------------------------------------------
# [1] 페이지 설정 및 테마 정의 (전문적인 분위기)
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI 건강 보험료 예측기",
    page_icon="🩺",
    layout="centered"
)

# 모델 불러오기 (캐싱 적용으로 속도 향상)
@st.cache_resource
def load_model():
    # LightGBM과 XGBoost가 모두 필요한 스태킹 모델이므로 호환성 이슈 해결된 상태여야 함
    return joblib.load("model.pkl")

model = load_model()

# ---------------------------------------------------------
# [2] 메인 화면 디자인 (깔끔한 헤더와 안내)
# ---------------------------------------------------------
st.markdown("<h1 style='text-align: center; color: #2c3e50;'>🩺 스마트 AI 건강 보험료 예측 서비스</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #7f8c8d; font-size: 1.2em;'>당신의 건강 지표를 분석하여 예상되는 미 보험료 수준을 알려드립니다.</p>", unsafe_allow_html=True)
st.divider()

# ---------------------------------------------------------
# [3] 입력 UI 디자인 (사이드바로 깔끔하게 정리)
# ---------------------------------------------------------
with st.sidebar:
    st.header("📋 정보 입력")
    st.write("당신의 신체 정보를 입력해 주세요.")
    
    age = st.number_input("나이 (세)", min_value=18, max_value=100, value=25, step=1)
    sex = st.radio("성별", ["여성", "남성"], horizontal=True)
    
    st.divider()
    
    height = st.number_input("키 (cm)", min_value=100.0, max_value=220.0, value=170.0, step=0.1, format="%.1f")
    weight = st.number_input("몸무게 (kg)", min_value=30.0, max_value=200.0, value=65.0, step=0.1, format="%.1f")
    
    st.divider()
    
    children = st.select_slider("자녀 수", options=[0, 1, 2, 3, 4, 5], value=0)
    smoker = st.radio("흡연 여부", ["아니오", "예"], horizontal=True)
    
    st.info("💡 모든 정보를 입력한 후 '예측하기' 버튼을 눌러주세요.")

# 📱 QR 코드 안내 (B님의 기여 포인트)
st.success("📱 발표 현장에서 이 QR코드를 스캔하여 모바일로 직접 체험해 보세요!")

# ---------------------------------------------------------
# [4] 예측 로직 및 결과 시각화
# ---------------------------------------------------------
# 퍼센타일 기준값 (발표용 고정값)
q50 = 9382.033
q80 = 20260.626

if st.button("🚀 당신의 건강 보험료 예측하기", use_container_width=True):
    with st.spinner('AI가 데이터를 정밀 분석 중입니다...'):
        time.sleep(1.5) # 분석 중인 듯한 애니메이션 효과

        # BMI 및 데이터 변환 (B님의 핵심 로직)
        height_m = height / 100
        bmi = weight / (height_m ** 2)
        sex_num = 1.0 if sex == "남성" else 0.0
        smoker_num = 1.0 if smoker == "예" else 0.0
        is_obese = 1.0 if bmi >= 30 else 0.0
        
        # 모델 입력 데이터프레임 구성 (모두 숫자형으로!)
        input_df = pd.DataFrame({
            "age": [float(age)], "sex": [sex_num], "bmi": [float(bmi)],
            "children": [float(children)], "smoker": [smoker_num],
            "is_obese": [is_obese], "is_smoker": [smoker_num],
            "obese_smoker": [float(is_obese * smoker_num)]
        })

        try:
            # 예측 실행 (log -> 원래값)
            pred_log = model.predict(input_df)[0]
            pred = np.expm1(pred_log)
            krw = pred * 1500 # 환율 적용 환산

            # 결과 발표 대시보드 디자인
            st.divider()
            st.subheader("🔍 분석 리포트")
            
            # 지표 카드 디자인
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("예상 보험료(USD)", f"${pred:,.0f}")
            with col2:
                st.metric("예상 보험료(KRW)", f"₩{krw:,.0f}")
            with col3:
                st.metric("BMI 지수", f"{bmi:.1f}")

            # 등급 분석 및 박스 디자인
            if pred < q50:
                grade, desc, color = "낮음", "의료비 지출이 적은 우수 관리 그룹입니다.", "#2ecc71" #🟢Green
                st.balloons()
            elif pred < q80:
                grade, desc, color = "보통", "평균적인 의료비 지출 그룹입니다.", "#f39c12" #🟡Orange
            else:
                grade, desc, color = "높음", "상대적으로 높은 의료비 지출 그룹입니다. 건강 관리에 주의가 필요합니다.", "#e74c3c" #🔴Red

            # 컬러 박스로 등급 표시
            st.markdown(f"""
            <div style="background-color: {color}; padding: 20px; border-radius: 10px; color: white; margin-top: 20px;">
                <h3 style="margin: 0; color: white;">🏆 의료비 수준 등급: {grade}</h3>
                <p style="margin: 5px 0 0 0; font-size: 1.1em; color: white;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

            # 퍼센타일 그래프 시각화 (Progress bar)
            if pred < q50:
                percentile = (pred / q50) * 50
            elif pred < q80:
                percentile = 50 + (pred - q50) / (q80 - q50) * 30
            else:
                percentile = 80 + (pred - q80) / q80 * 20
                percentile = min(percentile, 99.9)

            st.divider()
            st.write(f"📊 당신은 전체 사용자 중 상위 **{100 - percentile:.1f}%** 수준의 의료비를 지출할 것으로 예상됩니다.")
            st.progress(int(percentile))
            st.caption("※ 본 예측 결과는 입력하신 신체 지표를 기반으로 한 AI 통계 모델의 결과이며, 실제 보험사의 가입 심사 결과와는 다를 수 있습니다.")

        except Exception as e:
            st.error(f"⚠️ 모델 계산 중 오류가 발생했습니다: {e}")
            st.info("💡 로컬 환경의 라이브러리 버전을 확인해 보세요.")

st.divider()
st.caption("© 2026 Team Insurance Prediction Project. All rights reserved.")
