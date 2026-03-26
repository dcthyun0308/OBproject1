import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time

# ---------------------------------------------------------
# [1] 에러 방지 패치 (가장 안전한 방식)
# ---------------------------------------------------------
try:
    import sklearn
    from sklearn.impute import SimpleImputer
    if not hasattr(SimpleImputer, "_fill_dtype"):
        SimpleImputer._fill_dtype = property(lambda self: np.float64)
except:
    pass # 라이브러리가 없어도 일단 화면은 뜨게 함

# ---------------------------------------------------------
# [2] 페이지 기본 설정
# ---------------------------------------------------------
st.set_page_config(page_title="보험료 예측 서비스", page_icon="💰")

# 모델 로드 함수 (오류 나도 앱은 안 죽게 보호)
@st.cache_resource
def load_model():
    try:
        return joblib.load("model.pkl")
    except Exception as e:
        return f"Error: {e}"

model_obj = load_model()

# ---------------------------------------------------------
# [3] 메인 디자인
# ---------------------------------------------------------
st.title("💰 AI 보험료 예측 서비스")
st.success("📱 QR코드를 스캔해서 접속 중이신가요? 환영합니다!")

# 사이드바 입력창
with st.sidebar:
    st.header("📋 정보 입력")
    age = st.number_input("나이", 18, 100, 25)
    sex = st.radio("성별", ["여성", "남성"], horizontal=True)
    height = st.number_input("키 (cm)", 100.0, 220.0, 170.0)
    weight = st.number_input("몸무게 (kg)", 30.0, 200.0, 65.0)
    children = st.slider("자녀 수", 0, 5, 0)
    smoker = st.radio("흡연 여부", ["아니오", "예"], horizontal=True)

# ---------------------------------------------------------
# [4] 예측 실행
# ---------------------------------------------------------
if st.button("🚀 분석 및 결과 확인", use_container_width=True):
    if isinstance(model_obj, str):
        st.error(f"모델을 불러오지 못했습니다. 파일명을 확인해 주세요. ({model_obj})")
    else:
        with st.spinner('AI 분석 중...'):
            time.sleep(0.5)
            
            # 입력 데이터 변환
            bmi = weight / ((height/100) ** 2)
            sex_val = "male" if sex == "남성" else "female"
            smoker_val = "yes" if smoker == "예" else "no"
            
            input_df = pd.DataFrame({
                "age": [float(age)], "sex": [sex_val], "bmi": [float(bmi)],
                "children": [float(children)], "smoker": [smoker_val],
                "is_obese": [float(int(bmi >= 30))],
                "is_smoker": [float(int(smoker_val == "yes"))],
                "obese_smoker": [float(int(bmi >= 30) * int(smoker_val == "yes"))]
            })

            try:
                pred = np.expm1(model_obj.predict(input_df)[0])
                st.divider()
                st.subheader(f"💰 예상 보험료: ${pred:,.0f} (약 ₩{pred*1500:,.0f})")
                
                if pred < 9382:
                    st.balloons()
                    st.success("🟢 의료비 지출 수준: 낮음")
                elif pred < 20260:
                    st.warning("🟡 의료비 지출 수준: 보통")
                else:
                    st.error("🔴 의료비 지출 수준: 높음")
            except Exception as e:
                st.error(f"계산 중 오류 발생: {e}")
