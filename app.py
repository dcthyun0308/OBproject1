import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time

# 1. 페이지 설정 및 디자인
st.set_page_config(page_title="AI 보험료 예측 서비스", page_icon="💰")

# 모델 로드 (에러 방지 처리)
@st.cache_resource
def load_model():
    try:
        # 최신 sklearn 호환성을 위한 패치
        import sklearn
        from sklearn.impute import SimpleImputer
        if not hasattr(SimpleImputer, "_fill_dtype"):
            SimpleImputer._fill_dtype = property(lambda self: np.float64)
        
        return joblib.load("model.pkl")
    except Exception as e:
        return None

model = load_model()

# 2. 사이드바 디자인
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

# 3. 메인 화면
st.title("💰 AI 보험료 예측 서비스")
st.info("📱 QR코드를 통해 접속하신 것을 환영합니다!")
st.divider()

if model is None:
    st.error("🚨 모델 파일을 찾을 수 없습니다. GitHub에 'model.pkl'이 있는지 확인해주세요.")

# 4. 예측 실행 (핵심 수정 부분!)
if st.button("🚀 분석 및 결과 확인", use_container_width=True):
    if model:
        with st.spinner('AI가 데이터를 분석 중입니다...'):
            time.sleep(1)
            
            # BMI 계산
            bmi = weight / ((height/100) ** 2)
            
            # [수정 포인트] 글자를 숫자로 변환 (모델이 이해할 수 있게!)
            # 보통 male=1, female=0 / yes=1, no=0 을 사용합니다.
            sex_num = 1 if sex == "남성" else 0
            smoker_num = 1 if smoker == "예" else 0
            
            is_obese = 1 if bmi >= 30 else 0
            is_smoker = smoker_num
            obese_smoker = is_obese * is_smoker

            # 입력 데이터프레임 구성 (모두 숫자로!)
            input_df = pd.DataFrame({
                "age": [float(age)],
                "sex": [float(sex_num)],      # 'male' 대신 숫자로!
                "bmi": [float(bmi)],
                "children": [float(children)],
                "smoker": [float(smoker_num)], # 'yes' 대신 숫자로!
                "is_obese": [float(is_obese)],
                "is_smoker": [float(is_smoker)],
                "obese_smoker": [float(obese_smoker)]
            })

            try:
                # 예측 및 결과 출력
                pred = np.expm1(model.predict(input_df)[0])
                krw = pred * 1500
                
                st.subheader("🔍 분석 결과 리포트")
                col1, col2 = st.columns(2)
                col1.metric("예상 보험료(USD)", f"${pred:,.0f}")
                col2.metric("예상 보험료(KRW)", f"₩{krw:,.0f}")

                if pred < 10000:
                    st.success("🟢 의료비 수준: 낮음 (관리 우수)")
                    st.balloons()
                elif pred < 20000:
                    st.warning("🟡 의료비 수준: 보통 (평균 지출)")
                else:
                    st.error("🔴 의료비 수준: 높음 (주의 필요)")
                    
            except Exception as e:
                st.error(f"계산 중 오류 발생: {e}")
                st.info("💡 만약 숫자로 바꿔도 안 된다면, 팀원 A에게 '모델이 글자를 직접 받는지' 물어봐야 합니다.")

st.divider()
st.caption("© 2026 OB Project Team")
