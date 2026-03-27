import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time
import sklearn
from sklearn.impute import SimpleImputer

# ---------------------------------------------------------
# [⭐ 필수] 깃허브 서버용 오류 방어 코드
# ---------------------------------------------------------
if not hasattr(SimpleImputer, "_fill_dtype"):
    def get_fill_dtype(self):
        return self.statistics_.dtype if hasattr(self, 'statistics_') else np.float64
    SimpleImputer._fill_dtype = property(get_fill_dtype)

# ---------------------------------------------------------
# [1] 페이지 설정 및 모바일 커스텀 CSS
# ---------------------------------------------------------
st.set_page_config(page_title="AI 보험료 예측기", page_icon="🩺", layout="centered")

st.markdown("""
<style>
    .big-font { font-size: 1.8em !important; font-weight: 700; color: #2c3e50; text-align: center; margin-bottom: 5px; }
    .sub-font { font-size: 1.0em !important; color: #7f8c8d; text-align: center; margin-bottom: 25px; }
    .section-title { font-size: 1.1em; font-weight: 700; color: #34495e; margin-top: 20px; margin-bottom: 10px; }
    .result-card { background-color: #f8f9fa; border-radius: 10px; padding: 15px; margin-bottom: 10px; border-left: 5px solid #3498db; }
    .result-label { font-size: 0.9em; color: #7f8c8d; }
    .result-value { font-size: 1.6em; font-weight: 700; color: #2c3e50; }
    .grade-box { padding: 15px; border-radius: 10px; color: white; margin-top: 15px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# 모델 불러오기
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# ---------------------------------------------------------
# [2] 메인 헤더
# ---------------------------------------------------------
st.markdown('<p class="big-font">🩺 AI 건강 보험료 예측기</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-font">당신의 정보를 한눈에 입력하고 결과를 확인하세요.</p>', unsafe_allow_html=True)
st.divider()

# ---------------------------------------------------------
# [3] 입력 UI - 펼침형 (All-at-once)
# ---------------------------------------------------------
st.subheader("📋 당신의 정보 입력")

# 섹션 1: 기본 정보
st.markdown('<p class="section-title">👤 1. 기본 정보</p>', unsafe_allow_html=True)
age = st.number_input("나이 (세)", min_value=18, max_value=100, value=25, step=1)
sex = st.radio("성별", ["여성", "남성"], horizontal=True)

st.divider() # 섹션 구분선

# 섹션 2: 신체 지표
st.markdown('<p class="section-title">📏 2. 신체 정보</p>', unsafe_allow_html=True)
height = st.number_input("키 (cm)", min_value=100.0, max_value=220.0, value=170.0, step=0.1, format="%.1f")
weight = st.number_input("몸무게 (kg)", min_value=30.0, max_value=200.0, value=65.0, step=0.1, format="%.1f")

st.divider() # 섹션 구분선

# 섹션 3: 건강 습관
st.markdown('<p class="section-title">🚬 3. 건강 습관</p>', unsafe_allow_html=True)
children = st.select_slider("자녀 수", options=[0, 1, 2, 3, 4, 5], value=0)
smoker = st.radio("흡연 여부", ["아니오", "예"], horizontal=True)

st.divider()

# ---------------------------------------------------------
# [4] 예측 및 결과 출력
# ---------------------------------------------------------
if st.button("🚀 나의 예상 보험료 확인하기", use_container_width=True):
    with st.spinner('AI 분석 중...'):
        time.sleep(1)
        bmi = weight / ((height / 100) ** 2)
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
            pred_log = model.predict(input_df)[0]
            pred = np.expm1(pred_log)
            krw = pred * 1500

            st.subheader("🔍 분석 리포트")
            
            # 모바일용 결과 카드
            st.markdown(f'<div class="result-card"><div class="result-label">예상 보험료 (USD)</div><div class="result-value">${pred:,.0f}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-card" style="border-left-color: #f39c12;"><div class="result-label">원화 환산 (KRW)</div><div class="result-value">약 ₩{krw:,.0f}</div></div>', unsafe_allow_html=True)
            
            # 등급 판정
            if pred < 9382:
                grade, color = "🟢 낮음", "#2ecc71"
                st.balloons()
            elif pred < 20260:
                grade, color = "🟡 보통", "#f39c12"
            else:
                grade, color = "🔴 높음", "#e74c3c"

            st.markdown(f'<div class="grade-box" style="background-color: {color};">🏆 의료비 수준 등급: {grade}</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"⚠️ 오류 발생: {e}")

st.caption("© 2026 Team Insurance Prediction Project")
