import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time
import sklearn
from sklearn.impute import SimpleImputer

# ---------------------------------------------------------
# [⭐ 필수] 깃허브 서버용 오류 방어 코드 (건드리지 마세요)
# ---------------------------------------------------------
if not hasattr(SimpleImputer, "_fill_dtype"):
    def get_fill_dtype(self):
        return self.statistics_.dtype if hasattr(self, 'statistics_') else np.float64
    SimpleImputer._fill_dtype = property(get_fill_dtype)

# ---------------------------------------------------------
# [1] 모바일 전용 페이지 설정 및 테마
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI 보험료 예측기",
    page_icon="🩺",
    layout="centered" # 모바일에서는 중앙 집중형이 좋습니다.
)

# 모바일 가독성을 위한 맞춤형 CSS (폰트 크기 및 간격 조정)
st.markdown("""
<style>
    /* 메인 타이틀 크기 조정 (모바일 배려) */
    .big-font { font-size: 1.8em !important; font-weight: 700; color: #2c3e50; text-align: center; margin-bottom: 5px; }
    .sub-font { font-size: 1.0em !important; color: #7f8c8d; text-align: center; margin-bottom: 25px; }
    
    /* 입력 위젯 간격 조정 */
    .stNumberInput, .stRadio, .stSlider { margin-bottom: 15px; }
    
    /* 모바일용 결과 카드 디자인 */
    .result-card {
        background-color: #f8f9fa; border-radius: 10px; padding: 15px;
        margin-bottom: 10px; border-left: 5px solid #3498db;
    }
    .result-label { font-size: 0.9em; color: #7f8c8d; margin-bottom: 2px; }
    .result-value { font-size: 1.6em; font-weight: 700; color: #2c3e50; }
    .result-value-krw { font-size: 1.4em; font-weight: 600; color: #7f8c8d; }

    /* 최종 등급 박스 (모바일) */
    .grade-box {
        padding: 15px; border-radius: 10px; color: white; margin-top: 15px; text-align: center;
    }
    .grade-header { font-size: 1.1em; font-weight: 600; margin: 0; color: white; }
    .grade-desc { font-size: 0.9em; margin: 5px 0 0 0; color: white; }
</style>
""", unsafe_allow_html=True)

# 모델 불러오기 (캐싱)
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# ---------------------------------------------------------
# [2] 메인 화면 디자인 (모바일용 헤더)
# ---------------------------------------------------------
st.markdown('<p class="big-font">🩺 AI 건강 보험료 예측기</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-font">당신의 지표로 연간 예상 의료비 지출을 분석합니다.</p>', unsafe_allow_html=True)

st.success("📱 모바일로 체험 중이시군요? 한 손으로 편하게 입력해 보세요!")
st.divider()

# ---------------------------------------------------------
# [3] 입력 UI - 메인 화면으로 이동 + Expander 그룹화
# ---------------------------------------------------------
st.subheader("📋 당신의 정보 입력")
st.write("아래 항목을 터치하여 입력해 주세요.")

# 그룹 1: 기본 인적 사항
with st.expander("👤 1. 기본 정보 (나이, 성별)", expanded=True):
    age = st.number_input("나이 (세)", min_value=18, max_value=100, value=25, step=1)
    sex = st.radio("성별", ["여성", "남성"], horizontal=True)

# 그룹 2: 신체 지표
with st.expander("📏 2. 신체 정보 (키, 몸무게)"):
    height = st.number_input("키 (cm)", min_value=100.0, max_value=220.0, value=170.0, step=0.1, format="%.1f")
    weight = st.number_input("몸무게 (kg)", min_value=30.0, max_value=200.0, value=65.0, step=0.1, format="%.1f")

# 그룹 3: 생활 습관
with st.expander("🚬 3. 건강 습관 (자녀, 흡연)"):
    children = st.select_slider("자녀 수", options=[0, 1, 2, 3, 4, 5], value=0)
    smoker = st.radio("흡연 여부", ["아니오", "예"], horizontal=True)

st.divider()

# 퍼센타일 기준
q50, q80 = 9382.033, 20260.626

# ---------------------------------------------------------
# [4] 예측 및 모바일 전용 결과 시각화
# ---------------------------------------------------------
if st.button("🚀 나의 예상 보험료 확인하기", use_container_width=True):
    with st.spinner('AI 분석 중... 잠시만 기다려주세요.'):
        time.sleep(1)

        # 1. 데이터 전처리 (B님의 로직 그대로)
        height_m = height / 100
        bmi = weight / (height_m ** 2)
        sex_num = 1.0 if sex == "남성" else 0.0
        smoker_num = 1.0 if smoker == "예" else 0.0
        is_obese = 1.0 if bmi >= 30 else 0.0
        
        input_df = pd.DataFrame({
            "age": [float(age)], "sex": [sex_num], "bmi": [float(bmi)],
            "children": [float(children)], "smoker": [smoker_num],
            "is_obese": [is_obese], "is_smoker": [smoker_num],
            "obese_smoker": [float(is_obese * smoker_num)]
        })

        try:
            # 2. 예측 실행
            pred_log = model.predict(input_df)[0]
            pred = np.expm1(pred_log)
            krw = pred * 1500

            # 3. 모바일용 결과 리포트 디자인 (사이드바 결과 카드를 모바일 세로형으로 변경)
            st.divider()
            st.subheader("🔍 당신의 분석 리포트")
            
            # 달러 카드
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">예상 연간 보험료 (USD)</div>
                <div class="result-value">${pred:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # 원화 카드
            st.markdown(f"""
            <div class="result-card" style="border-left-color: #f39c12;">
                <div class="result-label">원화 환산 (KRW)</div>
                <div class="result-value-krw">약 ₩{krw:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # BMI 카드
            st.markdown(f"""
            <div class="result-card" style="border-left-color: #2ecc71; border-left-width: 3px; padding: 10px;">
                <div class="result-label">BMI 지수</div>
                <div class="result-value" style="font-size: 1.2em;">{bmi:.1f}</div>
            </div>
            """, unsafe_allow_html=True)

            # 4. 등급 분석 및 모바일 박스 디자인
            if pred < q50:
                grade, desc, color = "🟢 낮음", "의료비 지출이 적은 우수 관리 그룹입니다.", "#2ecc71"
                st.balloons()
            elif pred < q80:
                grade, desc, color = "🟡 보통", "평균적인 의료비 지출 그룹입니다.", "#f39c12"
            else:
                grade, desc, color = "🔴 높음", "상대적으로 높은 의료비 지출 그룹입니다.", "#e74c3c"

            # 컬러 박스로 등급 표시 (모바일 크기 조정)
            st.markdown(f"""
            <div class="grade-box" style="background-color: {color};">
                <div class="grade-header">🏆 의료비 수준 등급: {grade}</div>
                <div class="grade-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

            # 퍼센타일 시각화 (모바일 배려: 간단히)
            if pred < q50:
                percentile = (pred / q50) * 50
            elif pred < q80:
                percentile = 50 + (pred - q50) / (q80 - q50) * 30
            else:
                percentile = 80 + (pred - q80) / q80 * 20
                percentile = min(percentile, 99.9)

            st.write(f"📊 당신은 전체 사용자 중 상위 **{100 - percentile:.1f}%** 수준의 의료비를 지출할 것으로 예상됩니다.")
            st.progress(int(percentile))
            st.caption("※ 본 결과는 AI 통계 모델의 예측이며, 실제 보험사와는 다를 수 있습니다.")

        except Exception as e:
            st.error(f"⚠️ 모델 계산 중 오류가 발생했습니다: {e}")

st.divider()
st.caption("© 2026 Team Insurance Prediction Project")
