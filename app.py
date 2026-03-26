import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time

# 1. 환경 패치 (구버전 모델 호환용)
@st.cache_resource
def load_model():
    try:
        import sklearn
        from sklearn.impute import SimpleImputer
        if not hasattr(SimpleImputer, "_fill_dtype"):
            SimpleImputer._fill_dtype = property(lambda self: np.float64)
        return joblib.load("model.pkl")
    except:
        return None

model = load_model()

# 2. 페이지 디자인
st.set_page_config(page_title="AI 보험료 예측 서비스", page_icon="💰")

with st.sidebar:
    st.header("📋 정보 입력")
    age = st.number_input("나이", 18, 100, 25)
    sex = st.radio("성별", ["여성", "남성"], horizontal=True)
    st.divider()
    height = st.number_input("키 (cm)", 100.0, 220.0, 170.0)
    weight = st.number_input("몸무게 (kg)", 30.0, 200.0, 65.0)
    st.divider()
    children = st.select_slider("자녀 수", options=[0, 1, 2, 3, 4, 5], value=0)
    smoker = st.radio("흡연 여부", ["아니오", "예"], horizontal=True)

st.title("💰 AI 보험료 예측 서비스")
st.info("📱 QR코드로 접속하신 것을 환영합니다!")

# 3. 예측 실행 (에러 근본 원인 해결)
if st.button("🚀 분석 및 결과 확인", use_container_width=True):
    if model:
        with st.spinner('AI 분석 중...'):
            time.sleep(1)
            
            # BMI 계산
            bmi = weight / ((height/100) ** 2)
            
            # [🔥 에러 해결 핵심] 
            # 모델이 'male'이라는 글자를 못 읽으므로, 무조건 숫자로 바꿔서 보냅니다.
            s_num = 1.0 if sex == "남성" else 0.0
            sm_num = 1.0 if smoker == "예" else 0.0
            is_obese = 1.0 if bmi >= 30 else 0.0
            
            # 데이터를 모델이 원하는 '숫자' 형태로만 구성
            input_df = pd.DataFrame({
                "age": [float(age)],
                "sex": [s_num],        # 여기에 절대 'male'이라는 글자가 들어가면 안 됩니다!
                "bmi": [float(bmi)],
                "children": [float(children)],
                "smoker": [sm_num],     # 여기에 절대 'yes'라는 글자가 들어가면 안 됩니다!
                "is_obese": [is_obese],
                "is_smoker": [sm_num],
                "obese_smoker": [float(is_obese * sm_num)]
            })

            try:
                # 예측 및 결과 출력
                pred = np.expm1(model.predict(input_df)[0])
                krw = pred * 1500
                
                st.divider()
                st.subheader(f"💵 예상 보험료: ${pred:,.0f} (약 {krw:,.0f}원)")
                
                if pred < 10000:
                    st.success("🟢 의료비 등급: 낮음")
                    st.balloons()
                elif pred < 20000:
                    st.warning("🟡 의료비 등급: 보통")
                else:
                    st.error("🔴 의료비 등급: 높음")
            except Exception as e:
                # 만약 또 에러가 나면, 팀원이 글자를 원할 수도 있으니 대비책 출력
                st.error(f"계산 중 오류 발생: {e}")
