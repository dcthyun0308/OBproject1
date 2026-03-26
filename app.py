import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# 1. 페이지 설정
st.set_page_config(page_title="보험료 예측 서비스", page_icon="💰")

# 2. 모델 로드 (가장 안전한 방식)
@st.cache_resource
def load_model():
    model_path = "model.pkl"
    if not os.path.exists(model_path):
        return None, "파일 없음"
    try:
        # 최신 sklearn과의 호환성을 위한 임시 패치
        import sklearn
        from sklearn.impute import SimpleImputer
        if not hasattr(SimpleImputer, "_fill_dtype"):
            SimpleImputer._fill_dtype = property(lambda self: np.float64)
            
        model = joblib.load(model_path)
        return model, "성공"
    except Exception as e:
        return None, str(e)

model, status = load_model()

# 3. 화면 디자인
st.title("💰 AI 보험료 예측 서비스")

if status == "파일 없음":
    st.error("🚨 'model.pkl' 파일이 GitHub 저장소에 없습니다! 파일 업로드 여부를 확인해주세요.")
elif model is None:
    st.error(f"🚨 모델 로드 중 기술적 오류 발생: {status}")
    st.info("팀원 A에게 'scikit-learn 버전'을 물어보거나 모델을 다시 받아야 할 수도 있습니다.")
else:
    st.success("✅ 시스템이 정상적으로 준비되었습니다.")

# 4. 입력 UI (사이드바)
with st.sidebar:
    st.header("📋 정보 입력")
    age = st.number_input("나이", 18, 100, 25)
    sex = st.radio("성별", ["여성", "남성"], horizontal=True)
    height = st.number_input("키 (cm)", 100.0, 220.0, 170.0)
    weight = st.number_input("몸무게 (kg)", 30.0, 200.0, 65.0)
    children = st.slider("자녀 수", 0, 5, 0)
    smoker = st.radio("흡연 여부", ["아니오", "예"], horizontal=True)

# 5. 예측 실행
if st.button("🚀 분석 및 결과 확인", use_container_width=True):
    if model:
        try:
            bmi = weight / ((height/100) ** 2)
            sex_val = "male" if sex == "남성" else "female"
            smoker_val = "yes" if smoker == "예" else "no"
            
            # 입력 데이터 구성
            input_df = pd.DataFrame({
                "age": [float(age)], "sex": [sex_val], "bmi": [float(bmi)],
                "children": [float(children)], "smoker": [smoker_val],
                "is_obese": [float(int(bmi >= 30))],
                "is_smoker": [float(int(smoker_val == "yes"))],
                "obese_smoker": [float(int(bmi >= 30) * int(smoker_val == "yes"))]
            })

            pred = np.expm1(model.predict(input_df)[0])
            
            st.divider()
            st.subheader(f"💵 예상 연간 보험료: ${pred:,.0f}")
            st.write(f"(한화 약 {pred*1500:,.0f}원)")
            
            if pred < 10000: st.balloons()
        except Exception as e:
            st.error(f"계산 중 오류 발생: {e}")
