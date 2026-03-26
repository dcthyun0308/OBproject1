import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time
import sklearn

# ---------------------------------------------------------
# [Step 1] 오류 해결을 위한 세이프 모드 (실행 시점에 보정)
# ---------------------------------------------------------
@st.cache_resource
def load_model_safely():
    try:
        model = joblib.load("model.pkl")
        # 모델 내부의 '임퓨터' 오류를 실행 직전에 잡아주는 안전 장치
        from sklearn.impute import SimpleImputer
        if hasattr(model, "named_steps"):
            for name, step in model.named_steps.items():
                if isinstance(step, SimpleImputer) and not hasattr(step, "_fill_dtype"):
                    step._fill_dtype = np.float64
        return model
    except Exception as e:
        st.error(f"모델 로드 실패: {e}")
        return None

# ---------------------------------------------------------
# [Step 2] 디자인 및 레이아웃
# ---------------------------------------------------------
st.set_page_config(page_title="AI 보험료 예측 서비스", page_icon="💰")

model = load_model_safely()

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

# 메인 화면
st.title("💰 AI 보험료 예측 서비스")
st.info("📱 QR코드를 스캔해서 모바일에서도 체험해보세요!")
st.divider()

# 퍼센타일 기준
q50, q80 = 9382.033, 20260.626

if st.button("🚀 분석 및 예측 시작", use_container_width=True):
    if model is not None:
        with st.spinner('AI 분석 중...'):
            time.sleep(1)
            
            # BMI 및 입력값 전처리
            bmi = weight / ((height/100) ** 2)
            sex_val = "male" if sex == "남성" else "female"
            smoker_val = "yes" if smoker == "예" else "no"
            is_obese, is_smoker = int(bmi >= 30), int(smoker_val == "yes")
            
            input_df = pd.DataFrame({
                "age": [float(age)], "sex": [sex_val], "bmi": [float(bmi)],
                "children": [float(children)], "smoker": [smoker_val],
                "is_obese": [float(is_obese)], "is_smoker": [float(is_smoker)],
                "obese_smoker": [float(is_obese * is_smoker)]
            })

            # 예측 실행
            try:
                pred = np.expm1(model.predict(input_df)[0])
                krw = pred * 1500

                # 결과 출력
                st.subheader("🔍 분석 결과")
                c1, c2 = st.columns(2)
                c1.metric("예상 보험료(USD)", f"${pred:,.0f}")
                c2.metric("예상 보험료(KRW)", f"₩{krw:,.0f}")

                if pred < q50:
                    st.success(f"🟢 낮음: 의료비 지출이 적은 우수 그룹입니다.")
                    st.balloons()
                elif pred < q80:
                    st.warning(f"🟡 보통: 평균적인 의료비 지출 그룹입니다.")
                else:
                    st.error(f"🔴 높음: 고위험 지출 그룹입니다. 관리가 필요합니다.")
            except Exception as e:
                st.error(f"예측 오류 발생: {e}. 팀원 A에게 '데이터 컬럼 순서'를 확인해보세요.")
    else:
        st.error("모델이 로드되지 않았습니다. 파일명을 확인해주세요.")

st.divider()
st.caption("© 2026 OB Project Team")
