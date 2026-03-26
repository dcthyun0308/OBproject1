import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time
import sklearn
from sklearn.impute import SimpleImputer

# ---------------------------------------------------------
# [1] 오류 해결: 구버전 모델 호환성 패치 (Monkey Patch)
# ---------------------------------------------------------
# 최신 sklearn에서 발생하는 _fill_dtype 오류를 방지하기 위해 강제로 속성을 주입합니다.
if not hasattr(SimpleImputer, "_fill_dtype"):
    def get_fill_dtype(self):
        return self.statistics_.dtype if hasattr(self, 'statistics_') else np.float64
    SimpleImputer._fill_dtype = property(get_fill_dtype)

# ---------------------------------------------------------
# [2] 페이지 설정 및 디자인 레이아웃
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI 건강 보험료 예측기",
    page_icon="🩺",
    layout="centered"
)

# 모델 불러오기 (안전 모드)
@st.cache_resource
def load_model():
    try:
        return joblib.load("model.pkl")
    except:
        return None

model = load_model()

# ---------------------------------------------------------
# [3] 사이드바: 사용자 입력창 (UX 개선)
# ---------------------------------------------------------
with st.sidebar:
    st.header("📋 정보 입력")
    st.write("분석을 위해 정보를 입력해 주세요.")
    
    age = st.number_input("나이", min_value=18, max_value=100, value=25, step=1)
    sex = st.radio("성별", ["여성", "남성"], horizontal=True)
    
    st.divider()
    
    height = st.number_input("키 (cm)", min_value=100.0, max_value=220.0, value=170.0, step=0.1, format="%.1f")
    weight = st.number_input("몸무게 (kg)", min_value=30.0, max_value=200.0, value=65.0, step=0.1, format="%.1f")
    
    st.divider()
    
    children = st.select_slider("자녀 수", options=[0, 1, 2, 3, 4, 5], value=0)
    smoker = st.radio("흡연 여부", ["아니오", "예"], horizontal=True)
    
    st.info("💡 모든 정보를 입력한 후 메인 화면의 버튼을 눌러주세요.")

# ---------------------------------------------------------
# [4] 메인 화면 구성
# ---------------------------------------------------------
st.title("🩺 AI 기반 건강 보험료 예측 시스템")
st.markdown("""
스마트한 AI가 당신의 신체 지표를 분석하여 **예상 의료비 수준**을 알려드립니다.
---
""")

# 모델 로드 실패 시 경고
if model is None:
    st.error("⚠️ 모델 파일(model.pkl)을 찾을 수 없거나 로드에 실패했습니다. GitHub에 파일이 있는지 확인해 주세요.")

# 📱 QR 안내 (UX 가산점)
st.success("📱 QR코드를 스캔하여 모바일에서도 간편하게 사용해 보세요!")

# -------------------------
# 기준 데이터 (고정값)
# -------------------------
q50 = 9382.033
q80 = 20260.626

# ---------------------------------------------------------
# [5] 예측 실행 및 결과 출력
# ---------------------------------------------------------
if st.button("🚀 분석 및 결과 확인하기", use_container_width=True):
    if model is not None:
        with st.spinner('AI가 데이터를 정밀 분석 중입니다...'):
            time.sleep(1.5) # 분석 애니메이션 효과

            # BMI 계산
            height_m = height / 100
            bmi = weight / (height_m ** 2)

            # 모델 입력 데이터 생성 (타입 충돌 방지 위해 float 강제 지정)
            sex_val = "male" if sex == "남성" else "female"
            smoker_val = "yes" if smoker == "예" else "no"
            is_obese = int(bmi >= 30)
            is_smoker = int(smoker_val == "yes")

            input_df = pd.DataFrame({
                "age": [float(age)],
                "sex": [sex_val],
                "bmi": [float(bmi)],
                "children": [float(children)],
                "smoker": [smoker_val],
                "is_obese": [float(is_obese)],
                "is_smoker": [float(is_smoker)],
                "obese_smoker": [float(is_obese * is_smoker)]
            })

            try:
                # 예측 실행 (log -> 원래값)
                pred_log = model.predict(input_df)[0]
                pred = np.expm1(pred_log)
                krw = pred * 1500  # 환율 적용 환산

                # 결과 대시보드 출력
                st.divider()
                st.subheader("🔍 분석 리포트")
                
                # 지표 카드 디자인
                col1, col2, col3 = st.columns(3)
                col1.metric("예상 보험료(USD)", f"${pred:,.0f}")
                col2.metric("예상 보험료(KRW)", f"₩{krw:,.0f}")
                col3.metric("BMI 지수", f"{bmi:.1f}")

                # 등급 분류 및 박스 디자인
                if pred < q50:
                    st.success(f"### 🟢 등급: 낮음\n의료비 지출이 적은 우수 관리 그룹입니다.")
                    st.balloons()
                elif pred < q80:
                    st.warning(f"### 🟡 등급: 보통\n평균적인 의료비 지출 그룹입니다.")
                else:
                    st.error(f"### 🔴 등급: 높음\n고위험 지출 그룹입니다. 건강 관리에 주의가 필요합니다.")

                # 퍼센타일 위치 (Progress bar)
                percentile = 50 + (pred - q50) / (q80 - q50) * 30 if pred >= q50 else (pred / q50) * 50
                percentile = min(max(percentile, 1), 99.9)
                st.write(f"📊 전체 사용자 중 상위 **{100 - percentile:.1f}%** 수준입니다.")
                st.progress(int(percentile))

            except Exception as e:
                st.error(f"⚠️ 예측 도중 오류가 발생했습니다: {e}")
                st.info("팀원 A에게 모델의 입력 컬럼(Column) 순서가 맞는지 확인을 요청해 보세요.")

st.divider()
st.caption("© 2026 OB Project Team - Insurance Prediction System")
