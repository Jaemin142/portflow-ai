import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression


# =========================================================
# 0. 기본 설정
# =========================================================
st.set_page_config(
    page_title="PortFlow AI",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at top right, rgba(29, 78, 216, 0.08), transparent 30%),
        radial-gradient(circle at bottom left, rgba(14, 165, 233, 0.06), transparent 28%),
        linear-gradient(180deg, #F8FAFC 0%, #F1F5F9 100%);
}
.block-container { padding-top: 2rem; padding-bottom: 3rem; }

h1 { color: #0F172A; font-weight: 800; letter-spacing: -0.03em; }
h2, h3 { color: #1E293B; font-weight: 700; letter-spacing: -0.02em; }
p, li { color: #334155; line-height: 1.65; }

section[data-testid="stSidebar"] {
    background-color: #0F2A4A;
    border-right: none;
}
section[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] .stMarkdown p {
    color: #F1F5F9 !important;
}
section[data-testid="stSidebar"] label {
    color: #94A3B8 !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}
section[data-testid="stSidebar"] .stSelectbox > div > div {
    background-color: #1E3A5F !important;
    border: 1px solid #2D5A8E !important;
    color: #E2E8F0 !important;
    border-radius: 8px !important;
}

div[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 0.5px solid #E2E8F0;
    border-top: 3px solid #1D4ED8;
    padding: 1rem 1.1rem;
    border-radius: 14px;
    box-shadow: 0 8px 20px rgba(15, 23, 42, 0.04);
}
div[data-testid="stMetricLabel"] {
    color: #64748B !important;
    font-weight: 600;
    font-size: 0.78rem;
}
div[data-testid="stMetricValue"] {
    color: #0F172A !important;
    font-weight: 800;
    font-size: 1.6rem;
    white-space: nowrap;
}

button[data-baseweb="tab"] {
    font-weight: 600 !important;
    color: #64748B !important;
    font-size: 0.85rem !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #1D4ED8 !important;
    border-bottom-color: #1D4ED8 !important;
}
/* 선택된 탭 하단 라인 색상 */
div[data-baseweb="tab-highlight"] {
    background-color: #1D4ED8 !important;
}


.js-plotly-plot {
    border-radius: 12px;
    border: 0.5px solid #E2E8F0;
}
div[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 0.5px solid #E2E8F0;
}
div[data-testid="stAlert"] { border-radius: 12px; }
details {
    background-color: #FFFFFF;
    border: 0.5px solid #E2E8F0 !important;
    border-radius: 12px;
}
.stButton > button {
    background-color: #1D4ED8;
    color: #FFFFFF;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.5rem 1.25rem;
}
.stButton > button:hover { background-color: #1E40AF; }

</style>
""", unsafe_allow_html=True)

# =========================================================
# 0-1. 시각화 색상 설정
# =========================================================
COLOR_EXPORT = "#1D4ED8"      # 수출 / 실제값 / 기본 블루
COLOR_IMPORT = "#64748B"      # 수입 / 비교 기준
COLOR_FORECAST = "#F59E0B"    # 예측값 / 주의
COLOR_BALANCE = "#2563EB"     # 무역수지
COLOR_LOW = "#22C55E"         # 낮음
COLOR_MID = "#F59E0B"         # 보통
COLOR_HIGH = "#EF4444"        # 높음

PORTFLOW_PALETTE = [
    "#1D4ED8",
    "#64748B",
    "#0F766E",
    "#7C3AED",
    "#F59E0B",
    "#EF4444",
    "#0891B2",
    "#84CC16",
    "#DB2777",
    "#475569"
]

RISK_COLOR_MAP = {
    "낮음": COLOR_LOW,
    "보통": COLOR_MID,
    "높음": COLOR_HIGH
}

# =========================================================
# 1. 함수 정의
# =========================================================
# ============================================================
# 국가별 공급망 의존도 분석 함수
# ============================================================

def get_dependency_level(score):
    """
    공급망 의존도 점수에 따른 등급 분류
    """
    if score >= 30:
        return "높음"
    elif score >= 15:
        return "보통"
    else:
        return "낮음"


def get_dependency_message(level):
    """
    공급망 의존도 등급별 해석 문장
    """
    if level == "높음":
        return "특정 국가에 대한 수입 및 교역 비중이 높아 공급망 충격 발생 시 리스크 전이가 클 수 있습니다."
    elif level == "보통":
        return "일정 수준의 국가 집중도가 확인되며, 주요 거래국 변화에 대한 모니터링이 필요합니다."
    else:
        return "국가별 거래 비중이 비교적 분산되어 있어 특정 국가 의존도는 낮은 편입니다."


def compute_country_dependency(df):
    """
    관세청 수출입 데이터를 기반으로 국가별 공급망 의존도 분석 테이블 생성

    현재 app.py 기준 필요 컬럼:
    - query_country_name: 국가명
    - expDlr: 수출금액
    - impDlr: 수입금액

    반환 컬럼:
    - 국가
    - 수출금액
    - 수입금액
    - 총교역금액
    - 수입의존도(%)
    - 총교역비중(%)
    - 공급망의존도점수
    - 의존도등급
    - 해석
    """

    required_cols = ["query_country_name", "expDlr", "impDlr"]

    for col in required_cols:
        if col not in df.columns:
            return pd.DataFrame()

    temp = df.copy()

    temp["expDlr"] = pd.to_numeric(temp["expDlr"], errors="coerce").fillna(0)
    temp["impDlr"] = pd.to_numeric(temp["impDlr"], errors="coerce").fillna(0)

    country_dep = (
        temp.groupby("query_country_name", as_index=False)
        .agg({
            "expDlr": "sum",
            "impDlr": "sum"
        })
    )

    country_dep = country_dep.rename(
        columns={
            "query_country_name": "국가",
            "expDlr": "수출금액",
            "impDlr": "수입금액"
        }
    )

    country_dep["총교역금액"] = country_dep["수출금액"] + country_dep["수입금액"]

    total_import = country_dep["수입금액"].sum()
    total_trade = country_dep["총교역금액"].sum()

    if total_import == 0:
        country_dep["수입의존도(%)"] = 0
    else:
        country_dep["수입의존도(%)"] = country_dep["수입금액"] / total_import * 100

    if total_trade == 0:
        country_dep["총교역비중(%)"] = 0
    else:
        country_dep["총교역비중(%)"] = country_dep["총교역금액"] / total_trade * 100

    country_dep["공급망의존도점수"] = (
        country_dep["수입의존도(%)"] * 0.7
        + country_dep["총교역비중(%)"] * 0.3
    )

    country_dep["의존도등급"] = country_dep["공급망의존도점수"].apply(get_dependency_level)
    country_dep["해석"] = country_dep["의존도등급"].apply(get_dependency_message)

    country_dep = country_dep.sort_values(
        by="공급망의존도점수",
        ascending=False
    ).reset_index(drop=True)

    country_dep["수출금액"] = country_dep["수출금액"].round(0).astype(int)
    country_dep["수입금액"] = country_dep["수입금액"].round(0).astype(int)
    country_dep["총교역금액"] = country_dep["총교역금액"].round(0).astype(int)
    country_dep["수입의존도(%)"] = country_dep["수입의존도(%)"].round(1)
    country_dep["총교역비중(%)"] = country_dep["총교역비중(%)"].round(1)
    country_dep["공급망의존도점수"] = country_dep["공급망의존도점수"].round(1)

    return country_dep

def make_forecast(monthly_df, target_col, periods=3):
    """
    월별 시계열 데이터에 대해 LinearRegression 기반 향후 periods개월 예측을 수행.
    데이터가 부족하면 최근 3개월 평균으로 대체.
    """
    model_df = monthly_df[["month", target_col]].copy()
    model_df = model_df.dropna()
    model_df = model_df.sort_values("month")

    if model_df.empty:
        return pd.DataFrame(columns=["month", "forecast", "model_type"])

    last_month = model_df["month"].max()

    if len(model_df) < 6:
        recent_avg = model_df[target_col].tail(3).mean()
        future_months = pd.date_range(
            start=last_month + pd.DateOffset(months=1),
            periods=periods,
            freq="MS"
        )

        return pd.DataFrame({
            "month": future_months,
            "forecast": [recent_avg] * periods,
            "model_type": ["최근 3개월 평균"] * periods
        })

    model_df["t"] = range(len(model_df))

    X = model_df[["t"]]
    y = model_df[target_col]

    model = LinearRegression()
    model.fit(X, y)

    last_t = int(model_df["t"].max())
    future_t = pd.DataFrame({
        "t": [last_t + i for i in range(1, periods + 1)]
    })

    forecast_values = model.predict(future_t)

    # 중량·금액은 음수가 될 수 없지만, 무역수지는 음수가 가능함
    if target_col != "balPayments":
        forecast_values = [max(0, value) for value in forecast_values]

    future_months = pd.date_range(
        start=last_month + pd.DateOffset(months=1),
        periods=periods,
        freq="MS"
    )

    return pd.DataFrame({
        "month": future_months,
        "forecast": forecast_values,
        "model_type": ["LinearRegression"] * periods
    })


def make_forecast_plot_df(monthly_df, target_col, forecast_df):
    """
    실제값과 예측값을 하나의 그래프에 표시하기 위한 데이터프레임 생성.
    예측선이 마지막 실제값에서 자연스럽게 이어지도록 마지막 실제 관측치를 예측선에도 포함.
    """
    actual_df = monthly_df[["month", target_col]].copy()
    actual_df = actual_df.rename(columns={target_col: "value"})
    actual_df["type"] = "실제값"

    last_actual = monthly_df[["month", target_col]].tail(1).copy()
    last_actual = last_actual.rename(columns={target_col: "forecast"})
    last_actual["model_type"] = "연결 기준점"

    pred_df = pd.concat([last_actual, forecast_df], ignore_index=True)
    pred_df = pred_df[["month", "forecast"]].copy()
    pred_df = pred_df.rename(columns={"forecast": "value"})
    pred_df["type"] = "예측값"

    plot_df = pd.concat([actual_df, pred_df], ignore_index=True)
    plot_df = plot_df.sort_values("month")

    return plot_df


def get_risk_level(score):
    if score >= 70:
        return "높음"
    if score >= 40:
        return "보통"
    return "낮음"


def format_number(value):
    try:
        return f"{value:,.0f}"
    except Exception:
        return "-"

def format_usd_short(value):
    """
    큰 USD 금액을 대시보드 카드용으로 짧게 표시.
    예: 80,000,000,000 -> 800.0억 USD
    """
    try:
        value = float(value)

        if abs(value) >= 100_000_000:
            return f"{value / 100_000_000:,.1f}억 USD"
        elif abs(value) >= 10_000:
            return f"{value / 10_000:,.1f}만 USD"
        else:
            return f"{value:,.0f} USD"

    except Exception:
        return "-"

def get_forecast_caution_level(volatility, forecast_change, monthly_count, forecast_model_type):
    """
    예측 주의도 산정 함수

    기준:
    - 데이터 기간이 짧을수록 주의도 증가
    - 최근 변동성이 클수록 주의도 증가
    - 다음 달 예측 변화율이 클수록 주의도 증가
    - 최근 3개월 평균 대체 모델이면 주의도 증가
    """

    caution_score = 0
    reasons = []

    # 1. 데이터 기간 기준
    if monthly_count < 6:
        caution_score += 40
        reasons.append("월별 데이터가 6개월 미만으로 예측 안정성이 낮습니다.")
    elif monthly_count < 12:
        caution_score += 25
        reasons.append("월별 데이터가 12개월 미만으로 장기 추세 해석에 주의가 필요합니다.")
    elif monthly_count < 24:
        caution_score += 10
        reasons.append("월별 데이터가 24개월 미만이므로 계절성 판단에는 한계가 있습니다.")

    # 2. 변동성 기준
    if volatility >= 50:
        caution_score += 30
        reasons.append("최근 월별 변동성이 매우 높아 예측값 변동 가능성이 큽니다.")
    elif volatility >= 25:
        caution_score += 20
        reasons.append("최근 월별 변동성이 비교적 높아 예측값 해석에 주의가 필요합니다.")
    elif volatility >= 10:
        caution_score += 10
        reasons.append("일정 수준의 월별 변동성이 관찰됩니다.")

    # 3. 예측 변화율 기준
    if abs(forecast_change) >= 30:
        caution_score += 30
        reasons.append("다음 달 예측 변화율이 매우 크게 나타났습니다.")
    elif abs(forecast_change) >= 15:
        caution_score += 20
        reasons.append("다음 달 예측 변화율이 비교적 크게 나타났습니다.")
    elif abs(forecast_change) >= 10:
        caution_score += 10
        reasons.append("다음 달 예측값에 일정 수준의 변화가 예상됩니다.")

    # 4. 모델 대체 여부
    if forecast_model_type == "최근 3개월 평균":
        caution_score += 20
        reasons.append("데이터 부족으로 LinearRegression 대신 최근 3개월 평균값을 사용했습니다.")

    caution_score = min(caution_score, 100)

    if caution_score >= 60:
        level = "높음"
        message = (
            "예측값은 참고 가능하지만, 데이터 변동성 또는 기간 제약이 있어 "
            "단기 의사결정에는 추가 점검이 필요합니다."
        )
    elif caution_score >= 30:
        level = "보통"
        message = (
            "예측값은 현재 추세를 파악하는 데 활용할 수 있으나, "
            "최근 변동성과 데이터 기간을 함께 고려해야 합니다."
        )
    else:
        level = "낮음"
        message = (
            "현재 데이터 기준으로 예측값 해석에 큰 제약은 낮은 편입니다. "
            "다만 외부 변수 변화 가능성은 계속 모니터링해야 합니다."
        )

    if not reasons:
        reasons.append("데이터 기간, 변동성, 예측 변화율 기준에서 큰 주의 요인은 확인되지 않았습니다.")

    return {
        "예측 주의도 점수": round(caution_score, 1),
        "예측 주의도": level,
        "해석": message,
        "주의 요인": reasons
    }

def get_action_recommendation(score, forecast_change, selected_country, selected_hs):
    """
    리스크 점수와 예측 변화율을 바탕으로 대응방안 문장 생성.
    """
    if score >= 70:
        return (
            f"{selected_country} - {selected_hs} 수출입 물류 흐름의 예측 반영 리스크가 높은 수준입니다. "
            "단기적으로 선적 일정, 재고 확보, 대체 운송경로, 주요 거래처 납기 리스크를 사전에 점검해야 합니다. "
            "특히 다음 달 예측 변화율이 큰 경우 운송 여력, 통관 일정, 공급처 다변화 가능성을 함께 검토할 필요가 있습니다."
        )

    if score >= 40:
        return (
            f"{selected_country} - {selected_hs} 수출입 물류 흐름의 리스크는 보통 수준입니다. "
            "급격한 병목 가능성은 제한적이지만, 최근 수요 변화와 변동성을 지속적으로 모니터링해야 합니다. "
            "수출입 물량이 특정 월에 집중될 경우 선적·통관 일정 분산을 검토할 수 있습니다."
        )

    return (
        f"{selected_country} - {selected_hs} 수출입 물류 흐름의 리스크는 낮은 수준입니다. "
        "현재는 안정적 흐름으로 판단되지만, 특정 국가·품목 의존도가 높을 경우 정기적 모니터링을 유지하는 것이 바람직합니다."
    )


def calculate_risk_for_group(group_df, target_col):
    """
    국가-품목 조합별 리스크 점수 계산 함수.
    현재 앱의 리스크 산식과 동일한 구조를 사용한다.
    """
    monthly = (
        group_df
        .groupby("month", as_index=False)[["expWgt", "expDlr", "impWgt", "impDlr", "balPayments"]]
        .sum()
        .sort_values("month")
    )

    if len(monthly) < 2:
        return None

    latest = monthly.iloc[-1]
    previous = monthly.iloc[-2]

    latest_value = latest[target_col]
    previous_value = previous[target_col]

    if previous_value != 0:
        mom_change = ((latest_value - previous_value) / abs(previous_value)) * 100
    else:
        mom_change = 0

    volatility = (
        monthly[target_col]
        .pct_change()
        .replace([float("inf"), -float("inf")], pd.NA)
        .dropna()
        .std() * 100
    )

    if pd.isna(volatility):
        volatility = 0

    total_exp_wgt = monthly["expWgt"].sum()
    total_imp_wgt = monthly["impWgt"].sum()

    if total_imp_wgt != 0:
        imbalance_ratio = abs(total_exp_wgt - total_imp_wgt) / total_imp_wgt * 100
    else:
        imbalance_ratio = 0

    forecast_df = make_forecast(monthly, target_col, periods=1)

    if not forecast_df.empty:
        forecast_next = forecast_df["forecast"].iloc[0]
    else:
        forecast_next = monthly[target_col].tail(3).mean()

    if latest_value != 0:
        forecast_change = ((forecast_next - latest_value) / abs(latest_value)) * 100
    else:
        forecast_change = 0

    demand_risk = min(max(abs(mom_change) * 2, 0), 35)
    volatility_risk = min(max(volatility * 1.5, 0), 35)
    imbalance_risk = min(max(imbalance_ratio * 0.3, 0), 30)
    forecast_risk = min(max(abs(forecast_change) * 0.5, 0), 20)

    base_risk = min(demand_risk + volatility_risk + imbalance_risk, 100)
    forecast_based_risk = min(base_risk + forecast_risk, 100)

    return {
        "최근월 변화율": round(mom_change, 1),
        "변동성": round(volatility, 1),
        "수출입 불균형": round(imbalance_ratio, 1),
        "예측 변화율": round(forecast_change, 1),
        "리스크 점수": round(forecast_based_risk, 1),
        "리스크 등급": get_risk_level(forecast_based_risk)
    }

def get_dependency_adjustment(score):
    """
    공급망 의존도 점수에 따른 리스크 보정점수
    """
    if score >= 30:
        return 10
    elif score >= 15:
        return 5
    else:
        return 0


def make_top_risk_ranking(df, target_col, top_n=10):
    """
    전체 국가-품목 조합별 리스크 점수를 계산해 TOP N 랭킹 생성.
    기존 리스크 점수에 국가별 공급망 의존도 보정점수를 반영한다.
    """
    ranking_rows = []

    grouped = df.groupby(["query_country_name", "query_hs_name"])

    for (country, hs_name), group_df in grouped:
        result = calculate_risk_for_group(group_df, target_col)

        if result is None:
            continue

        dependency_base_df = df[df["query_hs_name"] == hs_name].copy()
        dependency_df = compute_country_dependency(dependency_base_df)

        if dependency_df.empty:
            import_share = 0
            trade_share = 0
            dependency_score = 0
            dependency_level = "낮음"
            dependency_adjustment = 0
        else:
            matched_dependency = dependency_df[dependency_df["국가"] == country]

            if matched_dependency.empty:
                import_share = 0
                trade_share = 0
                dependency_score = 0
                dependency_level = "낮음"
                dependency_adjustment = 0
            else:
                dep_row = matched_dependency.iloc[0]

                import_share = dep_row["수입의존도(%)"]
                trade_share = dep_row["총교역비중(%)"]
                dependency_score = dep_row["공급망의존도점수"]
                dependency_level = dep_row["의존도등급"]
                dependency_adjustment = get_dependency_adjustment(dependency_score)

        base_risk_score = result["리스크 점수"]
        final_risk_score = min(base_risk_score + dependency_adjustment, 100)
        final_risk_level = get_risk_level(final_risk_score)

        ranking_rows.append({
            "국가": country,
            "품목군": hs_name,
            "조합": f"{country} - {hs_name}",
            "기존 리스크 점수": round(base_risk_score, 1),
            "수입의존도(%)": round(import_share, 1),
            "총교역비중(%)": round(trade_share, 1),
            "공급망의존도점수": round(dependency_score, 1),
            "의존도등급": dependency_level,
            "의존도 보정점수": dependency_adjustment,
            "최종 리스크 점수": round(final_risk_score, 1),
            "최종 리스크 등급": final_risk_level,
            "최근월 변화율": result["최근월 변화율"],
            "변동성": result["변동성"],
            "수출입 불균형": result["수출입 불균형"],
            "예측 변화율": result["예측 변화율"]
        })

    ranking_df = pd.DataFrame(ranking_rows)

    if ranking_df.empty:
        return ranking_df

    ranking_df = ranking_df.sort_values("최종 리스크 점수", ascending=False).head(top_n)
    ranking_df = ranking_df.reset_index(drop=True)
    ranking_df.index = ranking_df.index + 1
    ranking_df = ranking_df.reset_index().rename(columns={"index": "순위"})

    return ranking_df

# =========================================================
# 2. 데이터 불러오기
# =========================================================
DATA_PATH = "data/raw/customs_trade.csv"

try:
    raw_df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    st.error("data/raw/customs_trade.csv 파일을 찾을 수 없습니다. 먼저 관세청 API 수집 코드를 실행하세요.")
    st.stop()


# =========================================================
# 3. 데이터 전처리
# =========================================================
df = raw_df.copy()

if "year" not in df.columns:
    st.error("CSV에 year 컬럼이 없습니다.")
    st.stop()

df["year"] = df["year"].astype(str).str.strip()
df = df[df["year"] != "총계"].copy()

if "month" in df.columns:
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
else:
    df["year_clean"] = (
        df["year"]
        .astype(str)
        .str.replace(".", "", regex=False)
        .str.replace("-", "", regex=False)
        .str.replace("/", "", regex=False)
        .str[:6]
    )
    df["month"] = pd.to_datetime(df["year_clean"], format="%Y%m", errors="coerce")

df = df.dropna(subset=["month"]).copy()

numeric_cols = ["expWgt", "expDlr", "impWgt", "impDlr", "balPayments"]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    else:
        df[col] = 0

required_cols = ["query_country_name", "query_hs_name", "month"]
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    st.error(f"필수 컬럼이 없습니다: {missing_cols}")
    st.stop()


# =========================================================
# 4. 사이드바 필터
# =========================================================
st.sidebar.header("분석 조건")
st.sidebar.caption("국가·품목·지표를 선택해 수요예측과 리스크 점수를 확인합니다.")

country_options = sorted(df["query_country_name"].dropna().unique().tolist())
hs_options = sorted(df["query_hs_name"].dropna().unique().tolist())

selected_country = st.sidebar.selectbox(
    "분석 국가 선택",
    country_options,
    index=country_options.index("중국") if "중국" in country_options else 0
)

selected_hs = st.sidebar.selectbox(
    "분석 품목군 선택",
    hs_options,
    index=hs_options.index("전기기기") if "전기기기" in hs_options else 0
)

metric_option = st.sidebar.selectbox(
    "분석 지표 선택",
    ["수출중량", "수입중량", "수출금액", "수입금액", "무역수지"]
)

metric_map = {
    "수출중량": "expWgt",
    "수입중량": "impWgt",
    "수출금액": "expDlr",
    "수입금액": "impDlr",
    "무역수지": "balPayments"
}

unit_map = {
    "수출중량": "kg",
    "수입중량": "kg",
    "수출금액": "USD",
    "수입금액": "USD",
    "무역수지": "USD"
}

target_col = metric_map[metric_option]
target_unit = unit_map[metric_option]


# =========================================================
# 5. 필터 적용 및 월별 집계
# =========================================================
filtered_df = df[
    (df["query_country_name"] == selected_country) &
    (df["query_hs_name"] == selected_hs)
].copy()

if filtered_df.empty:
    st.error("선택한 조건에 해당하는 데이터가 없습니다.")
    st.stop()

monthly_df = (
    filtered_df
    .groupby("month", as_index=False)[numeric_cols]
    .sum()
    .sort_values("month")
)

if monthly_df.empty:
    st.error("월별 집계 데이터가 비어 있습니다.")
    st.stop()


# =========================================================
# 6. KPI 계산
# =========================================================
latest = monthly_df.iloc[-1]
previous = monthly_df.iloc[-2] if len(monthly_df) >= 2 else latest

latest_value = latest[target_col]
previous_value = previous[target_col]

if previous_value != 0:
    mom_change = ((latest_value - previous_value) / abs(previous_value)) * 100
else:
    mom_change = 0

recent_3m = monthly_df.tail(3)[target_col].mean()

volatility = (
    monthly_df[target_col]
    .pct_change()
    .replace([float("inf"), -float("inf")], pd.NA)
    .dropna()
    .std() * 100
)

if pd.isna(volatility):
    volatility = 0

total_exp_wgt = monthly_df["expWgt"].sum()
total_imp_wgt = monthly_df["impWgt"].sum()

if total_imp_wgt != 0:
    imbalance_ratio = abs(total_exp_wgt - total_imp_wgt) / total_imp_wgt * 100
else:
    imbalance_ratio = 0


# =========================================================
# 7. AI 예측
# =========================================================
forecast_df = make_forecast(monthly_df, target_col, periods=3)
forecast_plot_df = make_forecast_plot_df(monthly_df, target_col, forecast_df)

if not forecast_df.empty:
    forecast_next = forecast_df["forecast"].iloc[0]
    forecast_model_type = forecast_df["model_type"].iloc[0]
else:
    forecast_next = recent_3m
    forecast_model_type = "예측 불가"

if latest_value != 0:
    forecast_change = ((forecast_next - latest_value) / abs(latest_value)) * 100
else:
    forecast_change = 0


# =========================================================
# 8. 리스크 점수 계산
# =========================================================
demand_risk = min(max(abs(mom_change) * 2, 0), 35)
volatility_risk = min(max(volatility * 1.5, 0), 35)
imbalance_risk = min(max(imbalance_ratio * 0.3, 0), 30)

risk_score = demand_risk + volatility_risk + imbalance_risk
risk_score = round(min(risk_score, 100), 1)

forecast_risk = min(max(abs(forecast_change) * 0.5, 0), 20)
forecast_based_risk_score = risk_score + forecast_risk
forecast_based_risk_score = round(min(forecast_based_risk_score, 100), 1)

risk_level = get_risk_level(risk_score)
forecast_based_risk_level = get_risk_level(forecast_based_risk_score)

# 사이드바 리스크 게이지 (8번 계산 완료 후)
risk_color = "#EF4444" if forecast_based_risk_score >= 70 else \
             "#F59E0B" if forecast_based_risk_score >= 40 else "#22C55E"
risk_width = int(forecast_based_risk_score)

st.sidebar.markdown("---")

st.sidebar.markdown(
    f"""
    <div style="background:#142F52; border:1px solid #2D5A8E; border-radius:14px; padding:14px; margin-top:8px;">
        <div style="color:#93C5FD; font-size:0.72rem; font-weight:800; letter-spacing:0.05em; margin-bottom:8px;">
            RISK SIGNAL
        </div>
        <div style="color:{risk_color}; font-size:1.65rem; font-weight:900; line-height:1.1; margin-bottom:6px;">
            {forecast_based_risk_score}점
        </div>
        <div style="color:#CBD5E1; font-size:0.74rem; line-height:1.45; margin-bottom:10px;">
            예측 변화율을 반영한 현재 선택 조건의 공급망 리스크 신호입니다.
        </div>
        <div style="background:#1E3A5F; border-radius:999px; height:8px; overflow:hidden; margin-bottom:6px;">
            <div style="background:{risk_color}; width:{risk_width}%; height:8px; border-radius:999px;"></div>
        </div>
        <div style="color:#CBD5E1; font-size:0.72rem; font-weight:700; text-align:right;">
            {forecast_based_risk_level}
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# 9. 상단 요약
# =========================================================

# 
start_period = monthly_df["month"].min().strftime("%y.%m")
end_period = monthly_df["month"].max().strftime("%y.%m")


# 헤더: 제목 + 한 줄 컨텍스트
st.title("PortFlow AI")
st.markdown(
    f"""
    <p style="font-size:0.95rem; color:#64748B; 
              margin-top:-0.5rem; margin-bottom:1rem;">
        {selected_country} &nbsp;·&nbsp; {selected_hs} &nbsp;·&nbsp; 
        {metric_option} &nbsp;·&nbsp; {start_period} ~ {end_period}
    </p>
    """,
    unsafe_allow_html=True
)

# 배너 유지
st.markdown("""
<div style="
    background: linear-gradient(90deg, #0F2A4A 0%, #1D4ED8 100%);
    padding: 1.25rem 1.5rem;
    border-radius: 16px;
    margin: 0 0 1.5rem 0;
    color: white;
">
    <div style="font-size:1.05rem; font-weight:700; margin-bottom:0.35rem;">
        데이터 기반 수출입 물류 리스크 조기경보
    </div>
    <div style="font-size:0.9rem; color:#DBEAFE; line-height:1.6;">
        관세청 수출입 데이터를 활용해 국가·품목별 물류 수요를 예측하고,
        우선 점검이 필요한 공급망 리스크 조합을 탐색합니다.
    </div>
</div>
""", unsafe_allow_html=True)

# KPI 카드 (리스크 카드 색상 분리)
risk_card_bg = "#FFF5F5" if forecast_based_risk_score >= 70 else \
               "#FFFBEB" if forecast_based_risk_score >= 40 else "#F0FDF4"
risk_card_border = "#FECACA" if forecast_based_risk_score >= 70 else \
                   "#FDE68A" if forecast_based_risk_score >= 40 else "#BBF7D0"
risk_card_top = "#DC2626" if forecast_based_risk_score >= 70 else \
                "#F59E0B" if forecast_based_risk_score >= 40 else "#16A34A"
risk_text_color = "#DC2626" if forecast_based_risk_score >= 70 else \
                  "#92400E" if forecast_based_risk_score >= 40 else "#166534"

st.write("## 핵심 지표")
k1, k2, k3, k4 = st.columns(4)

# 핵심 지표 표시값 축약
if target_unit == "USD":
    latest_value_display = format_usd_short(latest_value)
    forecast_next_display = format_usd_short(forecast_next)
else:
    latest_value_display = f"{format_number(latest_value)} {target_unit}"
    forecast_next_display = f"{format_number(forecast_next)} {target_unit}"

k1.metric(
    label=f"최근월 {metric_option}",
    value=latest_value_display,
    delta=f"{mom_change:.1f}%"
)

k2.metric(
    label="AI 다음 달 예측값",
    value=forecast_next_display,
    delta=f"{forecast_change:.1f}%"
)

# 리스크 카드는 HTML로 직접 렌더링
with k3:
    st.markdown(
        f"""
        <div style="background:{risk_card_bg}; border:0.5px solid {risk_card_border};
                    border-top:3px solid {risk_card_top}; border-radius:12px;
                    padding:1rem 1.1rem;">
            <p style="color:{risk_text_color}; font-size:0.78rem; 
                      font-weight:600; margin:0 0 4px;">현재 리스크 점수</p>
            <p style="color:{risk_card_top}; font-size:1.6rem; 
                      font-weight:800; margin:0 0 4px;">{risk_score}점</p>
            <p style="color:{risk_text_color}; font-size:0.78rem; margin:0;">
                ↑ {risk_level}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with k4:
    st.markdown(
        f"""
        <div style="background:{risk_card_bg}; border:0.5px solid {risk_card_border};
                    border-top:3px solid {risk_card_top}; border-radius:12px;
                    padding:1rem 1.1rem;">
            <p style="color:{risk_text_color}; font-size:0.78rem; 
                      font-weight:600; margin:0 0 4px;">예측 반영 리스크</p>
            <p style="color:{risk_card_top}; font-size:1.6rem; 
                      font-weight:800; margin:0 0 4px;">{forecast_based_risk_score}점</p>
            <p style="color:{risk_text_color}; font-size:0.78rem; margin:0;">
                ↑ {forecast_based_risk_level}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================================================
# 10. 탭 구성
# =========================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "데이터 개요",
    "수출입 추이",
    "AI 수요예측",
    "국가·품목 비교",
    "리스크 분석",
    "What-if 시뮬레이션",
    "TOP 리스크 랭킹",
    "확장 설계"
])


# =========================================================
# 탭 1. 데이터 개요
# =========================================================
with tab1:
    st.write("### 데이터 개요")

    st.markdown(
        """
        PortFlow AI는 **관세청 품목별·국가별 수출입실적 데이터**를 기반으로
        국가·품목별 수출입 물류 수요를 분석하고, AI 예측값과 공급망 리스크 점수를 통해
        우선 점검이 필요한 국가·품목 조합을 탐색하는 데이터 기반 웹서비스입니다.

        현재 구현 버전은 관세청 수출입 데이터를 중심으로 구성되어 있으며,
        항만운영 통계, 공컨테이너 수급 데이터, 선박 입출항 정보는
        직접 연동 데이터가 아니라 **향후 확장 데이터**로 설계되어 있습니다.
        """
    )

    st.divider()

    # -----------------------------------------------------
    # 전체 데이터 현황
    # -----------------------------------------------------
    st.write("### 전체 데이터 현황")

    total_rows = len(df)
    total_countries = df["query_country_name"].nunique()
    total_hs = df["query_hs_name"].nunique()
    full_start_period = df["month"].min().strftime("%y.%m")
    full_end_period = df["month"].max().strftime("%y.%m")

    total_export_value = df["expDlr"].sum()
    total_import_value = df["impDlr"].sum()
    total_balance_value = df["balPayments"].sum()

    d1, d2, d3, d4 = st.columns(4)

    d1.metric("전체 데이터 행 수", f"{total_rows:,}건")
    d2.metric("분석 국가 수", f"{total_countries:,}개")
    d3.metric("분석 품목군 수", f"{total_hs:,}개")
    d4.metric("분석 기간", f"{full_start_period} ~ {full_end_period}")

    d5, d6, d7 = st.columns(3)

    d5.metric("전체 수출금액", format_usd_short(total_export_value))
    d6.metric("전체 수입금액", format_usd_short(total_import_value))
    d7.metric("전체 무역수지", format_usd_short(total_balance_value))

    st.divider()

    # -----------------------------------------------------
    # 선택 조건 요약
    # -----------------------------------------------------
    st.write("### 선택 조건 요약")

    selected_total_export = monthly_df["expDlr"].sum()
    selected_total_import = monthly_df["impDlr"].sum()
    selected_total_balance = monthly_df["balPayments"].sum()
    selected_month_count = monthly_df["month"].nunique()
    selected_latest_month = monthly_df["month"].max().strftime("%y.%m")

    s1, s2, s3, s4 = st.columns(4)

    s1.metric("선택 국가", selected_country)
    s2.metric("선택 품목군", selected_hs)
    s3.metric("선택 지표", metric_option)
    s4.metric("최근월", selected_latest_month)

    s5, s6, s7, s8 = st.columns(4)

    s5.metric("선택 조건 행 수", f"{len(filtered_df):,}건")
    s6.metric("월별 집계 수", f"{selected_month_count:,}개월")
    s7.metric("선택 조건 수출금액", format_usd_short(selected_total_export))
    s8.metric("선택 조건 수입금액", format_usd_short(selected_total_import))

    # -----------------------------------------------------
    # 선택 조건 해석
    # -----------------------------------------------------
    st.write("### 선택 조건 거래 구조")

    if selected_total_import > selected_total_export:
        trade_status = "수입 중심"
        trade_comment = (
            "선택 조건에서는 수입금액이 수출금액보다 크게 나타납니다. "
            "따라서 해당 국가·품목 조합은 수입 의존 흐름이 상대적으로 강한 구조로 해석할 수 있습니다."
        )
        st.warning(f"**{trade_status}**: {trade_comment}")
    elif selected_total_export > selected_total_import:
        trade_status = "수출 중심"
        trade_comment = (
            "선택 조건에서는 수출금액이 수입금액보다 크게 나타납니다. "
            "따라서 해당 국가·품목 조합은 수출 물류 수요를 중심으로 해석할 수 있습니다."
        )
        st.info(f"**{trade_status}**: {trade_comment}")
    else:
        trade_status = "균형 흐름"
        trade_comment = (
            "선택 조건에서는 수출금액과 수입금액이 유사한 수준으로 나타납니다. "
            "수출입 양방향 흐름을 함께 모니터링할 필요가 있습니다."
        )
        st.success(f"**{trade_status}**: {trade_comment}")

    st.markdown(
        f"""
        선택 조건의 누적 무역수지는 **{format_usd_short(selected_total_balance)}**입니다. 
        이후 탭에서는 이 데이터를 기반으로 수출입 추이, AI 수요예측, 리스크 점수,
        공급망 의존도 분석, What-if 시뮬레이션을 수행합니다.
        """
    )

    st.divider()

    # -----------------------------------------------------
    # 사용 데이터 구조
    # -----------------------------------------------------
    st.write("### 사용 데이터 구조")

    data_schema_df = pd.DataFrame({
        "컬럼": [
            "query_country_name",
            "query_hs_name",
            "month",
            "expWgt",
            "impWgt",
            "expDlr",
            "impDlr",
            "balPayments"
        ],
        "의미": [
            "분석 국가명",
            "분석 품목군명",
            "월별 기준 시점",
            "수출중량",
            "수입중량",
            "수출금액",
            "수입금액",
            "무역수지"
        ],
        "활용 방식": [
            "국가별 필터 및 국가 간 비교",
            "품목군별 필터 및 품목 간 비교",
            "월별 추이 분석과 AI 예측 기준",
            "수출 물류 수요 흐름 분석",
            "수입 물류 수요 흐름 분석",
            "수출 규모 및 수요 변화 분석",
            "수입 규모 및 공급망 의존도 분석",
            "수출입 불균형 및 리스크 보조 지표"
        ]
    })

    st.dataframe(
        data_schema_df,
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------------------------------
    # 접기 영역
    # -----------------------------------------------------
    with st.expander("선택 조건 원본 데이터 미리보기"):
        st.dataframe(
            filtered_df.head(100),
            use_container_width=True,
            hide_index=True
        )

    with st.expander("선택 조건 월별 집계 데이터 보기"):
        monthly_display_df = monthly_df.copy()
        monthly_display_df["month"] = monthly_display_df["month"].dt.strftime("%Y-%m")

        st.dataframe(
            monthly_display_df,
            use_container_width=True,
            hide_index=True
        )

    with st.expander("현재 구현 범위와 확장 설계 구분 보기"):
        scope_df = pd.DataFrame({
            "구분": [
                "현재 구현 데이터",
                "현재 구현 기능",
                "확장 설계 데이터",
                "주의할 표현"
            ],
            "내용": [
                "관세청 품목별·국가별 수출입실적 데이터",
                "수출입 추이 분석, AI 3개월 수요예측, 리스크 점수, 공급망 의존도 분석, What-if 시뮬레이션, TOP 리스크 랭킹",
                "항만운영 통계, 공컨테이너 수급 데이터, 선박 입출항 정보",
                "인천항 데이터를 실제 사용했다거나 항만 실시간 혼잡을 예측한다고 표현하지 않음"
            ]
        })

        st.dataframe(
            scope_df,
            use_container_width=True,
            hide_index=True
        )
# =========================================================
# 탭 2. 수출입 추이
# =========================================================
with tab2:
    st.write("### 수출입 추이 분석")

    st.markdown(
        f"""
        선택한 **{selected_country} - {selected_hs}** 조합의 월별 수출입 흐름을 시각화합니다.

        수출입 중량, 수출입 금액, 무역수지 변화를 함께 확인하여
        특정 국가·품목군의 물류 수요 증가, 감소, 변동성 확대 여부를 파악할 수 있습니다.
        """
    )

    st.write(f"### {selected_country} - {selected_hs} 월별 수출입 중량 추이")

    fig_wgt = px.line(
        monthly_df,
        x="month",
        y=["expWgt", "impWgt"],
        markers=True,
        title=f"{selected_country} {selected_hs} 월별 수출입 중량 추이",
        color_discrete_sequence=[COLOR_EXPORT, COLOR_IMPORT]
    )
    fig_wgt.update_layout(
        xaxis_title="월",
        yaxis_title="중량(kg)",
        legend_title_text="구분",
        template="plotly_white"
    )
    st.plotly_chart(fig_wgt, use_container_width=True)

    st.write(f"### {selected_country} - {selected_hs} 월별 수출입 금액 추이")

    fig_dlr = px.line(
        monthly_df,
        x="month",
        y=["expDlr", "impDlr"],
        markers=True,
        title=f"{selected_country} {selected_hs} 월별 수출입 금액 추이",
        color_discrete_sequence=[COLOR_EXPORT, COLOR_IMPORT]
    )
    fig_dlr.update_layout(
        xaxis_title="월",
        yaxis_title="금액(USD)",
        legend_title_text="구분",
        template="plotly_white"
    )
    st.plotly_chart(fig_dlr, use_container_width=True)

    st.write("### 무역수지 추이")

    fig_bal = px.bar(
        monthly_df,
        x="month",
        y="balPayments",
        title=f"{selected_country} {selected_hs} 월별 무역수지"
    )
    fig_bal.update_traces(marker_color=COLOR_BALANCE)
    fig_bal.update_layout(
        xaxis_title="월",
        yaxis_title="무역수지(USD)",
        template="plotly_white"
    )
    st.plotly_chart(fig_bal, use_container_width=True)


# =========================================================
# 탭 3. AI 수요예측
# =========================================================
with tab3:
    st.write("### AI 기반 3개월 물류 수요예측")

    st.markdown(
        """
        <div style="
            background:#FFFFFF;
            border:0.5px solid #E2E8F0;
            border-left:5px solid #1D4ED8;
            border-radius:14px;
            padding:1rem 1.15rem;
            margin:0.75rem 0 1.1rem 0;
        ">
            <div style="font-size:0.95rem; font-weight:800; color:#0F172A; margin-bottom:0.35rem;">
                예측 구조
            </div>
            <div style="font-size:0.88rem; color:#334155; line-height:1.65;">
                월별 수출입 데이터 → 시간 인덱스 변환 → LinearRegression 기반 3개월 예측 →
                예측 변화율 및 예측 주의도 산출
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        현재 선택 조건인 **{selected_country} - {selected_hs} - {metric_option}**에 대해
        월별 시계열 데이터를 기반으로 향후 3개월 값을 예측합니다.

        - 사용 모델: **{forecast_model_type}**
        - 예측 대상 지표: **{metric_option}**
        - 단위: **{target_unit}**
        """
    )

    st.write("### 예측 모델 설명")

    st.markdown(
        """
        본 서비스는 월별 수출입 실적 데이터를 시간 인덱스로 변환한 뒤,
        `LinearRegression` 모델을 적용하여 향후 3개월 값을 예측합니다.

        - 입력 데이터: 선택 국가·품목군의 월별 수출입 중량, 금액 또는 무역수지
        - 예측 방식: 월별 시계열을 시간 순서 변수로 변환하여 추세 기반 예측
        - 예측 기간: 향후 3개월
        - 예외 처리: 데이터가 6개월 미만인 경우 최근 3개월 평균값으로 대체
        """
    )

    st.warning(
        """
       계절성, 환율, 운임지수, 지정학 리스크, 항만 처리량 등 외부 변수를 결합하면 더 정밀한 다변량 수요예측 모델로 확장할 수 있습니다.
        """
    )

    # -----------------------------------------------------
    # 예측 주의도 분석
    # -----------------------------------------------------
    forecast_caution = get_forecast_caution_level(
        volatility=volatility,
        forecast_change=forecast_change,
        monthly_count=len(monthly_df),
        forecast_model_type=forecast_model_type
    )

    caution_score = forecast_caution["예측 주의도 점수"]
    caution_level = forecast_caution["예측 주의도"]
    caution_message = forecast_caution["해석"]
    caution_reasons = forecast_caution["주의 요인"]

    st.write("### 예측 주의도")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "예측 주의도",
        caution_level
    )

    c2.metric(
        "주의도 점수",
        f"{caution_score:.1f}점"
    )

    c3.metric(
        "분석 데이터 기간",
        f"{len(monthly_df)}개월"
    )

    c4.metric(
        "최근 변동성",
        f"{volatility:.1f}%"
    )

    if caution_level == "높음":
        st.warning(caution_message)
    elif caution_level == "보통":
        st.info(caution_message)
    else:
        st.success(caution_message)

    caution_df = pd.DataFrame({
        "주의 요인": caution_reasons
    })

    with st.expander("예측 주의도 판단 근거 보기"):
        st.dataframe(
            caution_df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown(
            """
            예측 주의도는 실제 통계적 신뢰구간이 아니라,
            데이터 기간, 최근 변동성, 다음 달 예측 변화율, 예측 방식의 안정성을 바탕으로
            예측값을 얼마나 조심해서 해석해야 하는지 보여주는 보조 지표입니다.
            """
        )

    fig_forecast = px.line(
        forecast_plot_df,
        x="month",
        y="value",
        color="type",
        markers=True,
        title=f"{selected_country} {selected_hs} {metric_option} 실제값 및 예측값",
        color_discrete_map={
            "실제값": COLOR_EXPORT,
            "예측값": COLOR_FORECAST
        }
    )
    fig_forecast.update_layout(
        xaxis_title="월",
        yaxis_title=f"{metric_option}({target_unit})",
        legend_title_text="구분",
        template="plotly_white"
    )
    st.plotly_chart(fig_forecast, use_container_width=True)

    st.write("### 예측 결과")

    result_df = forecast_df.copy()
    if not result_df.empty:
        result_df["month"] = result_df["month"].dt.strftime("%Y-%m")
        result_df["forecast"] = result_df["forecast"].round(0).astype(int)

    st.dataframe(result_df, use_container_width=True)

    st.write("### 예측 해석")

    if forecast_change >= 10:
        st.warning(
            f"다음 달 예측값은 최근월 대비 약 {forecast_change:.1f}% 증가할 것으로 예상됩니다. "
            "수요 증가에 따른 선적 일정, 재고, 운송 여력 점검이 필요합니다."
        )
    elif forecast_change <= -10:
        st.info(
            f"다음 달 예측값은 최근월 대비 약 {forecast_change:.1f}% 감소할 것으로 예상됩니다. "
            "수요 둔화 가능성이 있으므로 물류 자원 배분을 보수적으로 검토할 필요가 있습니다."
        )
    else:
        st.success(
            f"다음 달 예측값은 최근월 대비 약 {forecast_change:.1f}% 수준으로, 급격한 변동 가능성은 크지 않습니다."
        )


# =========================================================
# 탭 4. 국가·품목 비교
# =========================================================
with tab4:
    st.write("### 국가·품목 비교 분석")

    st.markdown(
        f"""
        선택한 지표인 **{metric_option}**를 기준으로 국가별·품목군별 수출입 흐름을 비교합니다.

        국가별 비교는 특정 품목군이 어느 국가에서 크게 움직이는지 보여주고,
        품목군별 비교는 선택한 국가에서 어떤 품목군의 물류 수요가 상대적으로 큰지 확인하는 데 활용됩니다.
        """
    )

    st.write("### 국가별 선택 지표 비교")

    compare_country_df = (
        df[df["query_hs_name"] == selected_hs]
        .groupby(["month", "query_country_name"], as_index=False)[target_col]
        .sum()
    )

    fig_country = px.line(
        compare_country_df,
        x="month",
        y=target_col,
        color="query_country_name",
        markers=True,
        title=f"국가별 {selected_hs} {metric_option} 비교",
        color_discrete_sequence=PORTFLOW_PALETTE
    )
    fig_country.update_layout(
        xaxis_title="월",
        yaxis_title=f"{metric_option}({target_unit})",
        legend_title_text="국가",
        template="plotly_white"
    )
    st.plotly_chart(fig_country, use_container_width=True)

    st.write("### 품목군별 선택 지표 비교")

    compare_hs_df = (
        df[df["query_country_name"] == selected_country]
        .groupby(["month", "query_hs_name"], as_index=False)[target_col]
        .sum()
    )

    fig_hs = px.line(
        compare_hs_df,
        x="month",
        y=target_col,
        color="query_hs_name",
        markers=True,
        title=f"{selected_country} 품목군별 {metric_option} 비교",
        color_discrete_sequence=PORTFLOW_PALETTE
    )
    fig_hs.update_layout(
        xaxis_title="월",
        yaxis_title=f"{metric_option}({target_unit})",
        legend_title_text="품목군",
        template="plotly_white"
    )
    st.plotly_chart(fig_hs, use_container_width=True)

# =========================================================
# 탭 5. 리스크 분석
# =========================================================
with tab5:
    st.write("### 공급망 리스크 분석")

    st.markdown(
        """
        <div style="
            background:#FFFFFF;
            border:0.5px solid #E2E8F0;
            border-left:5px solid #EF4444;
            border-radius:14px;
            padding:1rem 1.15rem;
            margin:0.75rem 0 1.1rem 0;
        ">
            <div style="font-size:0.95rem; font-weight:800; color:#0F172A; margin-bottom:0.35rem;">
                리스크 산정 구조
            </div>
            <div style="font-size:0.88rem; color:#334155; line-height:1.65;">
                수요 변화 · 변동성 · 수출입 불균형 · AI 예측 변화율을 결합해 선택 조합 리스크를 계산하고,
                국가별 수입의존도와 총교역비중을 활용해 TOP 리스크 랭킹의 우선순위를 보정합니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        선택한 **{selected_country} - {selected_hs}** 조합의 수출입 흐름을 기반으로
        공급망 리스크 점수를 산정합니다.

        본 리스크 점수는 실제 지연이나 병목 발생 확률을 직접 의미하는 값이 아니라,
        수출입 수요 변화, 변동성, 수출입 불균형, AI 예측 변화율을 종합하여
        우선적으로 모니터링해야 할 국가·품목 조합을 선별하기 위한 조기경보 지표입니다.
        """
    )

    r1, r2 = st.columns([1.4, 1])

    with r1:
        st.write("### 주요 리스크 지표")

        risk_summary_df = pd.DataFrame({
            "항목": [
                "분석 국가",
                "분석 품목군",
                "선택 지표",
                "최근월 변화율",
                "다음 달 예측 변화율",
                "최근 변동성",
                "수출입 중량 불균형",
                "현재 리스크 점수",
                "예측 반영 리스크 점수"
            ],
            "값": [
                selected_country,
                selected_hs,
                metric_option,
                f"{mom_change:.1f}%",
                f"{forecast_change:.1f}%",
                f"{volatility:.1f}%",
                f"{imbalance_ratio:.1f}%",
                f"{risk_score}점",
                f"{forecast_based_risk_score}점"
            ],
            "해석": [
                "현재 선택한 분석 대상 국가",
                "현재 선택한 분석 대상 품목군",
                "리스크 산정에 사용한 기준 지표",
                "직전 월 대비 최근월 수출입 흐름 변화",
                "AI 예측값 기준 다음 달 수요 변화 가능성",
                "월별 수출입 흐름의 변동 폭",
                "수출중량과 수입중량 간 불균형 정도",
                "최근 데이터 기반 기본 리스크 점수",
                "AI 예측 변화율을 추가 반영한 최종 리스크 점수"
            ]
        })

        st.dataframe(risk_summary_df, use_container_width=True)

    with r2:
        st.write("### 최종 리스크 등급")

        if forecast_based_risk_score >= 70:
            st.error("높음")
            st.markdown(
                """
                급격한 수요 변화, 변동성 확대, 수출입 불균형이 나타날 가능성이 있습니다.  
                선적 일정, 재고, 통관 일정, 대체 운송경로를 우선 점검해야 합니다.
                """
            )
        elif forecast_based_risk_score >= 40:
            st.warning("보통")
            st.markdown(
                """
                즉각적인 고위험 구간은 아니지만, 수요 변화와 변동성이 관찰됩니다.  
                정기적인 모니터링과 사전 점검이 필요합니다.
                """
            )
        else:
            st.success("낮음")
            st.markdown(
                """
                현재 수출입 흐름은 비교적 안정적인 상태로 해석됩니다.  
                다만 특정 국가·품목 의존도가 높다면 정기 모니터링은 유지하는 것이 바람직합니다.
                """
            )

    st.write("### 리스크 등급 기준")

    risk_level_df = pd.DataFrame({
        "점수 구간": ["0~39점", "40~69점", "70~100점"],
        "등급": ["낮음", "보통", "높음"],
        "해석": [
            "수출입 물류 흐름이 비교적 안정적인 상태",
            "수요 변화 또는 변동성이 관찰되어 모니터링이 필요한 상태",
            "급격한 수요 변화·변동성·불균형이 나타나 선제 대응이 필요한 상태"
        ],
        "권장 대응": [
            "정기 모니터링 유지",
            "선적·재고·통관 일정 점검",
            "대체 운송경로 검토, 재고 확보, 납기 리스크 관리"
        ]
    })

    st.dataframe(risk_level_df, use_container_width=True)

    st.write("### 리스크 점수 산식")

    st.markdown(
        """
        본 서비스의 리스크 점수는 두 단계로 해석합니다.

        첫째, 현재 선택한 국가·품목 조합에 대해서는 수출입 수요 변화, 변동성,
        수출입 불균형, AI 예측 변화율을 결합하여 **예측 반영 리스크 점수**를 계산합니다.

        ```text
        선택 조합 리스크 점수 =
        수요 변화 리스크
        + 변동성 리스크
        + 수출입 불균형 리스크
        + AI 예측 변화 보정
        ```

        둘째, TOP 리스크 랭킹에서는 전체 국가·품목 조합을 비교하기 위해
        위 리스크 점수에 **국가별 공급망 의존도 보정**을 추가합니다.

        ```text
        TOP 랭킹 최종 리스크 점수 =
        선택 조합 리스크 점수
        + 국가별 공급망 의존도 보정
        ```

        이를 통해 단순히 최근 수출입 흐름이 크게 변한 조합뿐만 아니라,
        특정 국가에 대한 수입의존도와 총교역비중이 높은 조합까지
        우선 모니터링 대상으로 선별할 수 있습니다.
        """
    )

    risk_df = pd.DataFrame({
        "요인": [
            "수요 변화 리스크",
            "변동성 리스크",
            "수출입 불균형 리스크",
            "AI 예측 변화 보정"
        ],
        "점수": [
            demand_risk,
            volatility_risk,
            imbalance_risk,
            forecast_risk
        ],
        "설명": [
            "최근월 수출입 지표가 직전 월 대비 얼마나 급격히 변했는지 반영",
            "월별 수출입 흐름의 변동 폭을 반영",
            "수출중량과 수입중량 간 불균형 정도를 반영",
            "다음 달 AI 예측값이 최근월 대비 얼마나 변하는지 반영"
        ],
        "적용 위치": [
            "선택 조합 리스크 점수",
            "선택 조합 리스크 점수",
            "선택 조합 리스크 점수",
            "선택 조합 리스크 점수"
        ]
    })

    # -----------------------------------------------------
    # 리스크 원인 Top 요인 자동 해석
    # -----------------------------------------------------
    st.write("### 리스크 원인 Top 요인")

    top_factor_row = risk_df.sort_values("점수", ascending=False).iloc[0]

    top_factor_name = top_factor_row["요인"]
    top_factor_score = top_factor_row["점수"]
    top_factor_desc = top_factor_row["설명"]

    if top_factor_name == "수요 변화 리스크":
        top_factor_message = (
            "최근월 수출입 흐름이 직전 월 대비 크게 변하면서 리스크 점수에 가장 크게 기여했습니다. "
            "단기적으로 수요 급증 또는 급감 여부를 확인하고, 선적 일정과 재고 운영 계획을 점검할 필요가 있습니다."
        )
        top_factor_action = "대응 방향: 최근월 수요 변화 원인 확인, 선적·재고 계획 점검"

    elif top_factor_name == "변동성 리스크":
        top_factor_message = (
            "월별 수출입 흐름의 변동성이 크게 나타나 리스크 점수에 가장 크게 기여했습니다. "
            "일정한 흐름보다 월별 편차가 큰 상태이므로, 예측값을 단일 수치로 확정하기보다 변동 가능성을 함께 고려해야 합니다."
        )
        top_factor_action = "대응 방향: 모니터링 주기 단축, 월별 수요 편차 확인"

    elif top_factor_name == "수출입 불균형 리스크":
        top_factor_message = (
            "수출중량과 수입중량 간 차이가 크게 나타나 리스크 점수에 가장 크게 기여했습니다. "
            "이 경우 특정 방향의 물류 흐름에 치우쳐 장비 수급, 회송, 재고 운영 부담이 발생할 수 있습니다."
        )
        top_factor_action = "대응 방향: 수출입 균형 확인, 장비 수급 및 회송 가능성 점검"

    elif top_factor_name == "AI 예측 변화 보정":
        top_factor_message = (
            "다음 달 AI 예측값이 최근월 대비 크게 변하면서 리스크 점수에 영향을 주었습니다. "
            "예측 변화가 실제 수요 변화로 이어질 가능성에 대비해 운송 여력과 납기 리스크를 사전에 점검할 필요가 있습니다."
        )
        top_factor_action = "대응 방향: 예측 변화율 확인, 운송 여력과 납기 리스크 점검"

    else:
        top_factor_message = (
            "리스크 점수에 영향을 준 주요 요인을 확인했습니다. "
            "세부 지표를 함께 검토하여 우선 점검 대상을 판단할 필요가 있습니다."
        )
        top_factor_action = "대응 방향: 세부 리스크 요인 확인"

    f1, f2, f3 = st.columns(3)

    f1.metric(
        "최대 기여 요인",
        top_factor_name
    )

    f2.metric(
        "요인 점수",
        f"{top_factor_score:.1f}점"
    )

    f3.metric(
        "예측 반영 리스크",
        f"{forecast_based_risk_score}점",
        forecast_based_risk_level
    )

    if top_factor_score <= 0:
        st.success(
            "현재 선택 조건에서는 특정 리스크 요인이 크게 부각되지 않았습니다. "
            "다만 국가·품목별 수출입 흐름은 외부 변수에 따라 달라질 수 있으므로 정기 모니터링이 필요합니다."
        )
    elif forecast_based_risk_score >= 70:
        st.error(
            f"**{top_factor_name}**이 현재 리스크 점수에 가장 크게 기여했습니다. "
            f"{top_factor_message}"
        )
    elif forecast_based_risk_score >= 40:
        st.warning(
            f"**{top_factor_name}**이 현재 리스크 점수에 가장 크게 기여했습니다. "
            f"{top_factor_message}"
        )
    else:
        st.info(
            f"현재 리스크 등급은 낮은 편이지만, 상대적으로는 **{top_factor_name}**의 영향이 가장 큽니다. "
            f"{top_factor_message}"
        )

    st.caption(top_factor_action)

    with st.expander("리스크 요인별 세부 점수 보기"):
        st.dataframe(risk_df, use_container_width=True)

    fig_risk = px.bar(
        risk_df,
        x="요인",
        y="점수",
        hover_data=["설명", "적용 위치"],
        title="선택 조합 리스크 점수 구성 요인"
    )
    fig_risk.update_traces(marker_color=COLOR_EXPORT)
    fig_risk.update_layout(
        xaxis_title="리스크 요인",
        yaxis_title="점수",
        template="plotly_white"
    )
    st.plotly_chart(fig_risk, use_container_width=True)

    st.write("### 대응방안")

    st.info(
        get_action_recommendation(
            forecast_based_risk_score,
            forecast_change,
            selected_country,
            selected_hs
        )
    )
    st.write("### 국가별 공급망 의존도 분석")

    st.markdown(
        f"""
        선택한 품목군인 **{selected_hs}** 기준으로 국가별 수입 비중과 총교역 비중을 계산합니다.

        특정 국가의 수입 비중이 높을수록 해당 국가에 대한 공급망 의존도가 크다고 해석할 수 있으며,
        해당 국가에서 물류 차질, 통관 지연, 수급 변동이 발생할 경우 선택 품목군의 공급망 리스크가 확대될 수 있습니다.
        """
    )

    dependency_base_df = df[df["query_hs_name"] == selected_hs].copy()
    dependency_df = compute_country_dependency(dependency_base_df)

    if dependency_df.empty:
        st.warning("공급망 의존도 분석에 필요한 데이터가 부족합니다.")
    else:
        top_dependency = dependency_df.iloc[0]

        dep_country = top_dependency["국가"]
        dep_score = top_dependency["공급망의존도점수"]
        dep_level = top_dependency["의존도등급"]
        dep_import_share = top_dependency["수입의존도(%)"]
        dep_trade_share = top_dependency["총교역비중(%)"]

        dep_color = RISK_COLOR_MAP.get(dep_level, COLOR_MID)

        d1, d2, d3, d4 = st.columns(4)

        d1.metric(
            "최고 의존 국가",
            dep_country
        )

        d2.metric(
            "공급망 의존도 점수",
            f"{dep_score:.1f}점",
            dep_level
        )

        d3.metric(
            "수입의존도",
            f"{dep_import_share:.1f}%"
        )

        d4.metric(
            "총교역비중",
            f"{dep_trade_share:.1f}%"
        )

        st.write("#### 국가별 의존도 TOP 10")

        top_n_dependency = dependency_df.head(10).copy()

        fig_dependency = px.bar(
            top_n_dependency.sort_values("공급망의존도점수", ascending=True),
            x="공급망의존도점수",
            y="국가",
            orientation="h",
            color="의존도등급",
            color_discrete_map=RISK_COLOR_MAP,
            text="공급망의존도점수",
            title=f"{selected_hs} 국가별 공급망 의존도 TOP 10"
        )

        fig_dependency.update_traces(
            texttemplate="%{text:.1f}",
            textposition="outside"
        )

        fig_dependency.update_layout(
            xaxis_title="공급망 의존도 점수",
            yaxis_title="국가",
            legend_title_text="의존도 등급",
            template="plotly_white",
            height=460
        )

        st.plotly_chart(fig_dependency, use_container_width=True)

        st.write("#### 의존도 상세 테이블")

        dependency_display_df = dependency_df[
            [
                "국가",
                "수출금액",
                "수입금액",
                "총교역금액",
                "수입의존도(%)",
                "총교역비중(%)",
                "공급망의존도점수",
                "의존도등급",
                "해석"
            ]
        ].copy()

        st.dataframe(
            dependency_display_df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown(
            f"""
            현재 **{selected_hs}** 품목군에서 공급망 의존도가 가장 높은 국가는
            **{dep_country}**입니다.

            해당 국가는 전체 수입금액 중 **{dep_import_share:.1f}%**를 차지하며,
            총교역 비중은 **{dep_trade_share:.1f}%**입니다.
            따라서 해당 국가와 관련된 수급 변동, 통관 지연, 물류 차질이 발생할 경우
            선택 품목군의 공급망 리스크가 확대될 수 있습니다.
            """
        )

    st.write("### 데이터 해석상 한계")

    st.markdown(
        """
        - 본 서비스는 관세청 수출입 실적 데이터를 기반으로 하므로, 개별 컨테이너 이동이나 특정 항만의 실시간 혼잡도를 직접 추적하지는 않습니다.
        - 따라서 현재 리스크 점수는 항만 혼잡의 실시간 예측값이 아니라, 국가·품목별 수출입 수요 변화와 변동성에 기반한 조기경보용 의사결정 지원 지표입니다.
        - 항만운영 통계, 공컨테이너 수급 데이터, 선박 입출항 정보가 확보되면 향후 항만 운영 리스크 보정지표를 추가할 수 있습니다.
        - 현재 단계에서는 수출입 흐름의 이상 변화와 우선 점검 대상을 빠르게 찾는 기능에 초점을 둡니다.
        """
    )

# =========================================================
# 탭 6. What-if 시뮬레이션
# =========================================================
with tab6:
    st.write("### What-if 시뮬레이션")

    st.markdown(
        f"""
        선택한 **{selected_country} - {selected_hs}** 조합에 대해
        수출입 수요 변화와 변동성 확대 상황을 가정하고, 리스크 점수가 어떻게 달라지는지 확인합니다.

        이 기능은 실제 미래값을 확정적으로 예측하는 기능이 아니라,
        특정 상황이 발생했을 때 공급망 리스크가 어느 정도 민감하게 변하는지 검토하기 위한 시나리오 분석 도구입니다.
        """
    )

    st.write("### 시나리오 가정 입력")

    s1, s2 = st.columns(2)

    with s1:
        demand_change = st.slider(
            "가정 1: 다음 달 수요 변화율",
            min_value=-30,
            max_value=50,
            value=10,
            step=1,
            help="선택한 국가·품목군의 다음 달 수출입 수요가 얼마나 증가하거나 감소하는지 가정합니다."
        )

    with s2:
        volatility_change = st.slider(
            "가정 2: 변동성 변화율",
            min_value=-20,
            max_value=50,
            value=5,
            step=1,
            help="수출입 흐름의 불확실성 또는 월별 변동성이 얼마나 확대되거나 완화되는지 가정합니다."
        )

    scenario_risk = forecast_based_risk_score + demand_change * 0.5 + volatility_change * 0.3
    scenario_risk = max(0, min(100, round(scenario_risk, 1)))

    risk_gap = round(scenario_risk - forecast_based_risk_score, 1)

    st.write("### 시뮬레이션 결과")

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "기준 리스크 점수",
        f"{forecast_based_risk_score}점",
        forecast_based_risk_level
    )

    m2.metric(
        "시나리오 적용 후 리스크 점수",
        f"{scenario_risk}점",
        delta=f"{risk_gap}점"
    )

    if scenario_risk >= 70:
        scenario_level = "높음"
    elif scenario_risk >= 40:
        scenario_level = "보통"
    else:
        scenario_level = "낮음"

    m3.metric(
        "시나리오 리스크 등급",
        scenario_level
    )

    st.write("### 리스크 변화 해석")

    if scenario_risk >= 70:
        st.error(
            """
            시나리오 적용 결과 고위험 구간입니다.
            수요 변화 또는 변동성 확대가 실제로 발생할 경우 선적 지연, 재고 부족, 통관 일정 부담,
            대체 운송경로 검토 필요성이 커질 수 있습니다.
            """
        )
    elif scenario_risk >= 40:
        st.warning(
            """
            시나리오 적용 결과 중간 위험 구간입니다.
            즉각적인 고위험 상황은 아니지만, 수요 변화와 변동성이 확대될 경우
            정기 모니터링과 사전 점검이 필요합니다.
            """
        )
    else:
        st.success(
            """
            시나리오 적용 후에도 저위험 구간입니다.
            현재 가정에서는 급격한 물류 리스크 확대 가능성이 크지 않은 것으로 해석됩니다.
            """
        )

    st.write("### 시나리오 상세 요약")

    scenario_summary_df = pd.DataFrame({
        "항목": [
            "분석 국가",
            "분석 품목군",
            "기준 리스크 점수",
            "가정한 수요 변화율",
            "가정한 변동성 변화율",
            "시나리오 적용 후 리스크 점수",
            "리스크 점수 변화",
            "시나리오 등급"
        ],
        "값": [
            selected_country,
            selected_hs,
            f"{forecast_based_risk_score}점",
            f"{demand_change}%",
            f"{volatility_change}%",
            f"{scenario_risk}점",
            f"{risk_gap}점",
            scenario_level
        ],
        "해석": [
            "현재 선택한 분석 대상 국가",
            "현재 선택한 분석 대상 품목군",
            "AI 예측 변화율까지 반영한 기본 리스크 점수",
            "다음 달 수출입 수요가 증가하거나 감소하는 상황 가정",
            "월별 수출입 흐름의 불확실성이 확대되거나 완화되는 상황 가정",
            "사용자 가정값을 반영한 리스크 점수",
            "기준 리스크 대비 증가 또는 감소 폭",
            "시나리오 적용 후 최종 위험 수준"
        ]
    })

    st.dataframe(scenario_summary_df, use_container_width=True)

    st.write("### 시나리오 계산 방식")

    st.markdown(
        """
        본 What-if 시뮬레이션은 기존 예측 반영 리스크 점수에 사용자가 입력한 두 가지 가정값을 더해 계산합니다.

        ```text
        시나리오 리스크 점수 =
        예측 반영 기본 리스크 점수
        + 수요 변화율 × 0.5
        + 변동성 변화율 × 0.3
        ```

        계산 결과는 0점보다 작아지지 않고, 100점을 초과하지 않도록 제한합니다.
        따라서 이 점수는 실제 위험 확률이 아니라, 특정 상황에서 리스크가 얼마나 커지거나 줄어드는지 확인하는 민감도 지표입니다.
        """
    )

    with st.expander("What-if 시나리오 활용 예시 보기"):
        scenario_case_df = pd.DataFrame({
            "상황": [
                "수요 급증",
                "수요 감소",
                "변동성 확대",
                "수요 증가 + 변동성 확대"
            ],
            "의미": [
                "특정 국가·품목군의 물동량이 단기간에 증가하는 상황",
                "수출입 수요가 둔화되어 물류 자원 배분 조정이 필요한 상황",
                "월별 물동량 변동 폭이 커져 예측 불확실성이 확대되는 상황",
                "수요 증가와 불확실성이 동시에 발생해 운영 부담이 커지는 상황"
            ],
            "점검 포인트": [
                "선적 공간, 재고 확보, 운송 여력",
                "재고 조정, 운송계약 조정, 비용 관리",
                "수요 모니터링 주기 단축, 예외 상황 점검",
                "대체 운송경로, 선사·포워더 협의, 납기 리스크 관리"
            ]
        })

        st.dataframe(scenario_case_df, use_container_width=True)

# =========================================================
# 탭 7. TOP 리스크 랭킹
# =========================================================
with tab7:
    st.write("### TOP 리스크 국가·품목 랭킹")

    st.markdown(
        f"""
        전체 국가·품목 조합에 동일한 리스크 산식을 적용하여
        현재 선택한 분석 지표인 **{metric_option}** 기준으로 리스크 점수가 높은 조합을 자동으로 탐색합니다.

        이번 랭킹은 기존 수출입 흐름 기반 리스크 점수에 더해
        품목군별 국가 수입의존도와 총교역비중을 반영합니다.
        이를 통해 단순히 변동성이 큰 조합이 아니라,
        특정 국가에 대한 공급망 의존도가 높은 조합까지 함께 우선 모니터링 대상으로 선별합니다.
        """
    )

    st.write("### 랭킹 조건 설정")

    top_n = st.slider(
        "표시할 TOP 랭킹 개수",
        min_value=5,
        max_value=20,
        value=10,
        step=1,
        help="전체 국가·품목 조합 중 최종 리스크 점수가 높은 상위 N개 조합을 표시합니다."
    )

    top_risk_df = make_top_risk_ranking(
        df=df,
        target_col=target_col,
        top_n=top_n
    )

    if top_risk_df.empty:
        st.warning("리스크 랭킹을 계산할 수 있는 데이터가 부족합니다.")
    else:
        st.write("### 리스크 점수 상위 국가·품목 조합")

        display_cols = [
            "순위",
            "국가",
            "품목군",
            "기존 리스크 점수",
            "수입의존도(%)",
            "총교역비중(%)",
            "공급망의존도점수",
            "의존도등급",
            "의존도 보정점수",
            "최종 리스크 점수",
            "최종 리스크 등급",
            "최근월 변화율",
            "변동성",
            "수출입 불균형",
            "예측 변화율"
        ]

        available_display_cols = [
            col for col in display_cols if col in top_risk_df.columns
        ]

        st.dataframe(
            top_risk_df[available_display_cols],
            use_container_width=True,
            hide_index=True
        )

        # -----------------------------------------------------
        # TOP 리스크 랭킹 다운로드
        # -----------------------------------------------------
        download_df = top_risk_df[available_display_cols].copy()

        csv_data = download_df.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            label="TOP 리스크 랭킹 CSV 다운로드",
            data=csv_data,
            file_name=f"portflow_top_risk_{metric_option}.csv",
            mime="text/csv",
            help="현재 선택한 분석 지표 기준의 TOP 리스크 국가·품목 랭킹을 CSV 파일로 다운로드합니다."
        )

        highest_row = top_risk_df.iloc[0]

        st.write("### 최상위 리스크 조합 요약")

        r1, r2, r3, r4 = st.columns(4)

        r1.metric("1위 국가", highest_row["국가"])
        r2.metric("1위 품목군", highest_row["품목군"])
        r3.metric("최종 리스크 점수", f"{highest_row['최종 리스크 점수']}점")
        r4.metric("최종 리스크 등급", highest_row["최종 리스크 등급"])

        st.markdown(
            f"""
            현재 **{metric_option}** 기준으로 가장 높은 최종 리스크 점수를 보인 조합은
            **{highest_row["국가"]} - {highest_row["품목군"]}**입니다.

            이 조합은 최근월 변화율, 변동성, 수출입 불균형, 예측 변화율에 더해
            해당 품목군 내 국가별 수입의존도와 총교역비중을 함께 반영한 결과입니다.

            특히 이 조합의 수입의존도는 **{highest_row["수입의존도(%)"]}%**,
            총교역비중은 **{highest_row["총교역비중(%)"]}%**,
            공급망 의존도 점수는 **{highest_row["공급망의존도점수"]}점**이며,
            의존도 등급은 **{highest_row["의존도등급"]}**입니다.
            """
        )

        fig_top_risk = px.bar(
            top_risk_df,
            x="최종 리스크 점수",
            y="조합",
            color="최종 리스크 등급",
            orientation="h",
            hover_data=[
                "국가",
                "품목군",
                "기존 리스크 점수",
                "수입의존도(%)",
                "총교역비중(%)",
                "공급망의존도점수",
                "의존도등급",
                "의존도 보정점수",
                "최근월 변화율",
                "변동성",
                "수출입 불균형",
                "예측 변화율"
            ],
            title=f"TOP {len(top_risk_df)} 최종 리스크 국가·품목 조합",
            color_discrete_map=RISK_COLOR_MAP
        )

        fig_top_risk.update_layout(
            xaxis_title="최종 리스크 점수",
            yaxis_title="국가·품목 조합",
            yaxis={"categoryorder": "total ascending"},
            template="plotly_white",
            height=520
        )

        st.plotly_chart(fig_top_risk, use_container_width=True)

        with st.expander("랭킹 지표 해석 기준 보기"):
            ranking_guide_df = pd.DataFrame({
                "항목": [
                    "기존 리스크 점수",
                    "수입의존도(%)",
                    "총교역비중(%)",
                    "공급망의존도점수",
                    "의존도 보정점수",
                    "최종 리스크 점수",
                    "최근월 변화율",
                    "변동성",
                    "수출입 불균형",
                    "예측 변화율"
                ],
                "의미": [
                    "수요 변화, 변동성, 수출입 불균형, 예측 변화율을 종합한 기본 리스크 점수",
                    "선택 품목군에서 해당 국가가 차지하는 수입금액 비중",
                    "선택 품목군에서 해당 국가가 차지하는 총교역금액 비중",
                    "수입의존도와 총교역비중을 결합한 공급망 집중도 점수",
                    "공급망의존도점수에 따라 추가 반영되는 보정점수",
                    "기존 리스크 점수에 의존도 보정점수를 더한 최종 우선순위 점수",
                    "직전 월 대비 최근월 수출입 흐름 변화",
                    "월별 수출입 흐름의 변동 폭",
                    "수출중량과 수입중량의 차이",
                    "AI 예측값의 최근월 대비 변화"
                ],
                "해석 방향": [
                    "점수가 높을수록 수출입 흐름 자체의 변동성이 크다고 해석",
                    "값이 높을수록 해당 국가에 대한 수입 집중도가 큼",
                    "값이 높을수록 해당 국가와의 전체 교역 집중도가 큼",
                    "값이 높을수록 특정 국가에 대한 공급망 의존도가 큼",
                    "의존도가 높은 국가·품목 조합에 리스크 가중치를 부여",
                    "점수가 높을수록 우선 모니터링 필요성이 큼",
                    "절대값이 클수록 단기 수요 변화가 큼",
                    "값이 클수록 예측 불확실성과 운영 부담 가능성이 큼",
                    "값이 클수록 수출입 흐름의 균형이 약할 가능성이 있음",
                    "절대값이 클수록 향후 수요 변화 가능성이 큼"
                ]
            })

            st.dataframe(
                ranking_guide_df,
                use_container_width=True,
                hide_index=True
            )

    st.info(
        """
        리스크 랭킹은 전체 국가·품목 조합을 빠르게 비교하기 위한 탐색 기능입니다.
        현재 버전은 수출입 수요 변화, 변동성, 수출입 불균형, AI 예측 변화율에 더해
        품목군별 국가 수입의존도와 총교역비중을 반영하여 우선 모니터링 대상을 선별합니다.
        실제 운영 단계에서는 품목 중요도, 운임, 환율, 항만 처리량, 선박 스케줄 등 외부 변수를 함께 고려해야 합니다.
        """
    )

# =========================================================
# 탭 8. 확장 설계
# =========================================================
with tab8:
    st.write("### 확장 설계: 항만·선박 데이터 결합 방향")

    st.markdown(
        """
        <div style="
            background:#FFFFFF;
            border:0.5px solid #E2E8F0;
            border-left:5px solid #1D4ED8;
            border-radius:14px;
            padding:1.1rem 1.25rem;
            margin-bottom:1.25rem;
        ">
            <div style="font-size:1rem; font-weight:800; color:#0F172A; margin-bottom:0.4rem;">
                현재 구현 범위와 확장 설계의 구분
            </div>
            <div style="font-size:0.92rem; color:#334155; line-height:1.7;">
                현재 PortFlow AI는 관세청 품목별·국가별 수출입실적 데이터를 기반으로
                국가·품목별 물류 수요예측, 공급망 리스크 점수, 국가별 공급망 의존도 분석을 구현했습니다.
                항만운영 통계, 공컨테이너 수급 데이터, 선박 입출항 정보는 현재 직접 연동된 데이터가 아니라
                향후 확장 데이터로 설계한 항목입니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # 현재 구현 범위 요약
    # -----------------------------------------------------
    st.write("### 1. 현재 구현 범위")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div style="
                background:#EFF6FF;
                border:0.5px solid #BFDBFE;
                border-top:4px solid #1D4ED8;
                border-radius:14px;
                padding:1rem 1.1rem;
                min-height:170px;
            ">
                <div style="font-size:0.82rem; font-weight:700; color:#1D4ED8; margin-bottom:0.45rem;">
                    DATA
                </div>
                <div style="font-size:1rem; font-weight:800; color:#0F172A; margin-bottom:0.5rem;">
                    관세청 수출입 데이터
                </div>
                <div style="font-size:0.88rem; color:#334155; line-height:1.65;">
                    품목별·국가별 수출입실적을 기반으로 월별 수출입 중량, 금액, 무역수지를 분석합니다.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            """
            <div style="
                background:#F8FAFC;
                border:0.5px solid #CBD5E1;
                border-top:4px solid #64748B;
                border-radius:14px;
                padding:1rem 1.1rem;
                min-height:170px;
            ">
                <div style="font-size:0.82rem; font-weight:700; color:#475569; margin-bottom:0.45rem;">
                    MODEL
                </div>
                <div style="font-size:1rem; font-weight:800; color:#0F172A; margin-bottom:0.5rem;">
                    3개월 수요예측
                </div>
                <div style="font-size:0.88rem; color:#334155; line-height:1.65;">
                    LinearRegression 기반으로 선택 국가·품목군의 향후 수출입 수요 흐름을 예측합니다.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            """
            <div style="
                background:#F0FDF4;
                border:0.5px solid #BBF7D0;
                border-top:4px solid #22C55E;
                border-radius:14px;
                padding:1rem 1.1rem;
                min-height:170px;
            ">
                <div style="font-size:0.82rem; font-weight:700; color:#16A34A; margin-bottom:0.45rem;">
                    RISK
                </div>
                <div style="font-size:1rem; font-weight:800; color:#0F172A; margin-bottom:0.5rem;">
                    리스크 조기경보
                </div>
                <div style="font-size:0.88rem; color:#334155; line-height:1.65;">
                    수요 변화, 변동성, 수출입 불균형, 예측 변화율, 국가별 의존도를 결합해 우선 점검 대상을 선별합니다.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------------------------------------
    # 확장 데이터 카드
    # -----------------------------------------------------
    st.write("### 2. 향후 확장 데이터")

    e1, e2, e3 = st.columns(3)

    with e1:
        st.markdown(
            """
            <div style="
                background:#FFFFFF;
                border:0.5px solid #E2E8F0;
                border-radius:14px;
                padding:1rem 1.1rem;
                min-height:220px;
            ">
                <div style="
                    display:inline-block;
                    background:#EFF6FF;
                    color:#1D4ED8;
                    font-size:0.75rem;
                    font-weight:800;
                    padding:0.25rem 0.55rem;
                    border-radius:999px;
                    margin-bottom:0.65rem;
                ">
                    확장 데이터 01
                </div>
                <div style="font-size:1rem; font-weight:800; color:#0F172A; margin-bottom:0.5rem;">
                    항만운영 통계
                </div>
                <div style="font-size:0.88rem; color:#334155; line-height:1.65; margin-bottom:0.65rem;">
                    월별 컨테이너 처리량, 항만별 처리 흐름, 입항 실적 등을 활용하여 수출입 수요 변화가 항만 처리 부담으로 이어질 가능성을 보정합니다.
                </div>
                <div style="font-size:0.8rem; color:#64748B; line-height:1.6;">
                    활용 방향: 항만 처리 여력 · 물동량 증가 부담 · 운영 리스크 보정
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with e2:
        st.markdown(
            """
            <div style="
                background:#FFFFFF;
                border:0.5px solid #E2E8F0;
                border-radius:14px;
                padding:1rem 1.1rem;
                min-height:220px;
            ">
                <div style="
                    display:inline-block;
                    background:#FFFBEB;
                    color:#B45309;
                    font-size:0.75rem;
                    font-weight:800;
                    padding:0.25rem 0.55rem;
                    border-radius:999px;
                    margin-bottom:0.65rem;
                ">
                    확장 데이터 02
                </div>
                <div style="font-size:1rem; font-weight:800; color:#0F172A; margin-bottom:0.5rem;">
                    공컨테이너 수급 데이터
                </div>
                <div style="font-size:0.88rem; color:#334155; line-height:1.65; margin-bottom:0.65rem;">
                    월별 총 TEU, 20FT·40FT 컨테이너 수, 수출입 구분 정보를 활용해 공컨테이너 부족 또는 과잉 위험을 보정합니다.
                </div>
                <div style="font-size:0.8rem; color:#64748B; line-height:1.6;">
                    활용 방향: 장비 부족 · 컨테이너 회수 지연 · 수출 선적 차질 보정
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with e3:
        st.markdown(
            """
            <div style="
                background:#FFFFFF;
                border:0.5px solid #E2E8F0;
                border-radius:14px;
                padding:1rem 1.1rem;
                min-height:220px;
            ">
                <div style="
                    display:inline-block;
                    background:#F0FDF4;
                    color:#15803D;
                    font-size:0.75rem;
                    font-weight:800;
                    padding:0.25rem 0.55rem;
                    border-radius:999px;
                    margin-bottom:0.65rem;
                ">
                    확장 데이터 03
                </div>
                <div style="font-size:1rem; font-weight:800; color:#0F172A; margin-bottom:0.5rem;">
                    선박 입출항 정보
                </div>
                <div style="font-size:0.88rem; color:#334155; line-height:1.65; margin-bottom:0.65rem;">
                    입출항 일시, 선박명, 총톤수, 출항지, 차항지 정보를 활용해 운항 흐름 변화와 선박 공급 측면의 리스크를 보정합니다.
                </div>
                <div style="font-size:0.8rem; color:#64748B; line-height:1.6;">
                    활용 방향: 선복 변동 · 입항 흐름 변화 · 운항 리스크 보정
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------------------------------------
    # 결합 구조 로드맵
    # -----------------------------------------------------
    st.write("### 3. 데이터 결합 로드맵")

    st.markdown(
        """
        <div style="
            background:#FFFFFF;
            border:0.5px solid #E2E8F0;
            border-radius:16px;
            padding:1.2rem 1.3rem;
            margin-bottom:1rem;
        ">
            <div style="display:flex; align-items:stretch; gap:0.75rem; flex-wrap:wrap;">
                <div style="flex:1; min-width:170px; background:#EFF6FF; border-radius:12px; padding:0.9rem;">
                    <div style="font-size:0.75rem; color:#1D4ED8; font-weight:800; margin-bottom:0.35rem;">STEP 1</div>
                    <div style="font-size:0.95rem; color:#0F172A; font-weight:800; margin-bottom:0.35rem;">수출입 수요 신호</div>
                    <div style="font-size:0.8rem; color:#334155; line-height:1.55;">관세청 데이터 기반 국가·품목별 수요 변화 탐지</div>
                </div>
                <div style="display:flex; align-items:center; color:#94A3B8; font-weight:800;">→</div>
                <div style="flex:1; min-width:170px; background:#F8FAFC; border-radius:12px; padding:0.9rem;">
                    <div style="font-size:0.75rem; color:#475569; font-weight:800; margin-bottom:0.35rem;">STEP 2</div>
                    <div style="font-size:0.95rem; color:#0F172A; font-weight:800; margin-bottom:0.35rem;">운영 데이터 결합</div>
                    <div style="font-size:0.8rem; color:#334155; line-height:1.55;">항만운영·공컨테이너·선박 입출항 데이터를 월별 인덱스로 결합</div>
                </div>
                <div style="display:flex; align-items:center; color:#94A3B8; font-weight:800;">→</div>
                <div style="flex:1; min-width:170px; background:#FFFBEB; border-radius:12px; padding:0.9rem;">
                    <div style="font-size:0.75rem; color:#B45309; font-weight:800; margin-bottom:0.35rem;">STEP 3</div>
                    <div style="font-size:0.95rem; color:#0F172A; font-weight:800; margin-bottom:0.35rem;">운영 리스크 보정</div>
                    <div style="font-size:0.8rem; color:#334155; line-height:1.55;">처리량, 장비 수급, 선박 흐름을 반영해 리스크 점수 보정</div>
                </div>
                <div style="display:flex; align-items:center; color:#94A3B8; font-weight:800;">→</div>
                <div style="flex:1; min-width:170px; background:#F0FDF4; border-radius:12px; padding:0.9rem;">
                    <div style="font-size:0.75rem; color:#15803D; font-weight:800; margin-bottom:0.35rem;">STEP 4</div>
                    <div style="font-size:0.95rem; color:#0F172A; font-weight:800; margin-bottom:0.35rem;">의사결정 지원</div>
                    <div style="font-size:0.8rem; color:#334155; line-height:1.55;">우선 점검 국가·품목·운영 구간을 제시하는 조기경보 체계로 확장</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        현재 구현 데이터와 확장 데이터는 개별 화물 단위로 직접 연결하기보다,
        **월별 인덱스 방식**으로 결합하는 것이 현실적입니다.
        이를 통해 수출입 수요 변화와 항만 운영 부담 사이의 방향성을 비교할 수 있습니다.
        """
    )

    # -----------------------------------------------------
    # 확장 리스크 산식
    # -----------------------------------------------------
    st.write("### 4. 확장 리스크 산식")

    f1, f2 = st.columns([1.1, 1])

    with f1:
        st.markdown(
            """
            ```text
            확장 리스크 점수 =
            수요 변화 리스크
            + 변동성 리스크
            + 수출입 불균형 리스크
            + AI 예측 변화 보정
            + 국가별 공급망 의존도 보정
            + 항만 처리 리스크
            + 공컨테이너 수급 리스크
            + 선박 입출항 리스크
            ```
            """
        )

    with f2:
        st.markdown(
            """
            <div style="
                background:#F8FAFC;
                border:0.5px solid #CBD5E1;
                border-radius:14px;
                padding:1rem 1.1rem;
            ">
                <div style="font-size:0.95rem; font-weight:800; color:#0F172A; margin-bottom:0.5rem;">
                    설계 의도
                </div>
                <div style="font-size:0.88rem; color:#334155; line-height:1.65;">
                    현재 버전은 수출입 데이터 기반 리스크 신호를 먼저 탐지하고,
                    향후 항만·선박 운영 데이터를 결합해 실제 운영 부담 가능성을 보정하는 구조입니다.
                    따라서 현재 기능은 수요 기반 조기경보, 확장 기능은 운영 리스크 보정으로 구분됩니다.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # -----------------------------------------------------
    # 확장 시나리오 카드
    # -----------------------------------------------------
    st.write("### 5. 확장 시나리오 예시")

    s1, s2 = st.columns(2)

    with s1:
        st.markdown(
            """
            <div style="
                background:#FFFFFF;
                border:0.5px solid #E2E8F0;
                border-left:5px solid #F59E0B;
                border-radius:14px;
                padding:1rem 1.1rem;
                min-height:160px;
            ">
                <div style="font-size:0.95rem; font-weight:800; color:#0F172A; margin-bottom:0.5rem;">
                    수출 수요 증가 + 공컨테이너 감소
                </div>
                <div style="font-size:0.88rem; color:#334155; line-height:1.65;">
                    수출 수요가 증가하는데 공컨테이너 수급이 감소하면 선적 지연이나 장비 부족 리스크가 커질 수 있습니다.
                </div>
                <div style="font-size:0.8rem; color:#B45309; margin-top:0.7rem; font-weight:700;">
                    대응: 선사별 장비 확보 · 선적 일정 조정
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with s2:
        st.markdown(
            """
            <div style="
                background:#FFFFFF;
                border:0.5px solid #E2E8F0;
                border-left:5px solid #1D4ED8;
                border-radius:14px;
                padding:1rem 1.1rem;
                min-height:160px;
            ">
                <div style="font-size:0.95rem; font-weight:800; color:#0F172A; margin-bottom:0.5rem;">
                    입항 선박 수 감소 + 물동량 증가
                </div>
                <div style="font-size:0.88rem; color:#334155; line-height:1.65;">
                    물동량은 증가하지만 입항 선박 수가 감소하면 선복 부족, 일정 지연, 운임 상승 가능성이 커질 수 있습니다.
                </div>
                <div style="font-size:0.8rem; color:#1D4ED8; margin-top:0.7rem; font-weight:700;">
                    대응: 대체 항차 검토 · 운송 일정 재조정
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    s3, s4 = st.columns(2)

    with s3:
        st.markdown(
            """
            <div style="
                background:#FFFFFF;
                border:0.5px solid #E2E8F0;
                border-left:5px solid #22C55E;
                border-radius:14px;
                padding:1rem 1.1rem;
                min-height:160px;
            ">
                <div style="font-size:0.95rem; font-weight:800; color:#0F172A; margin-bottom:0.5rem;">
                    수입 수요 감소 + 수출 수요 증가
                </div>
                <div style="font-size:0.88rem; color:#334155; line-height:1.65;">
                    수입 컨테이너 유입이 줄어든 상태에서 수출 수요가 증가하면 수출용 공컨테이너 확보 부담이 커질 수 있습니다.
                </div>
                <div style="font-size:0.8rem; color:#15803D; margin-top:0.7rem; font-weight:700;">
                    대응: 공컨테이너 회수 계획 · 장비 재배치 검토
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with s4:
        st.markdown(
            """
            <div style="
                background:#FFFFFF;
                border:0.5px solid #E2E8F0;
                border-left:5px solid #64748B;
                border-radius:14px;
                padding:1rem 1.1rem;
                min-height:160px;
            ">
                <div style="font-size:0.95rem; font-weight:800; color:#0F172A; margin-bottom:0.5rem;">
                    40FT 컨테이너 비중 증가
                </div>
                <div style="font-size:0.88rem; color:#334155; line-height:1.65;">
                    특정 규격 컨테이너 수요가 증가하면 장비 수급 불균형과 배차·야드 운영 부담이 확대될 수 있습니다.
                </div>
                <div style="font-size:0.8rem; color:#475569; margin-top:0.7rem; font-weight:700;">
                    대응: 규격별 수급 관리 · 대체 규격 검토
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # -----------------------------------------------------
    # 표는 접기 처리
    # -----------------------------------------------------
    with st.expander("확장 데이터 상세 표 보기"):
        expansion_data = pd.DataFrame({
            "데이터": [
                "항만운영 통계",
                "공컨테이너 수급 데이터",
                "선박 입출항 정보"
            ],
            "활용 필드": [
                "월별 컨테이너 처리량, 항만별 처리 흐름, 입항 실적",
                "월별 총 TEU, 20FT/40FT 컨테이너 수, 수출입 구분",
                "입출항 일시, 선박명, 총톤수, 출항지, 차항지"
            ],
            "분석 활용": [
                "수출입 수요 증가 대비 항만 처리 여력 추정",
                "공컨테이너 부족·과잉 위험 보정",
                "선박 입출항 흐름 변화에 따른 운영 리스크 보정"
            ],
            "현재 적용 상태": [
                "확장 설계",
                "확장 설계",
                "확장 설계"
            ]
        })

        st.dataframe(
            expansion_data,
            use_container_width=True,
            hide_index=True
        )

    with st.expander("월별 결합 구조 예시 보기"):
        monthly_schema = pd.DataFrame({
            "월": ["2024-01", "2024-02", "2024-03"],
            "수출입 수요지수": [100, 108, 115],
            "AI 예측 수요지수": [None, None, 120],
            "컨테이너 처리지수": [100, 97, 94],
            "입항 선박 수 지수": [100, 98, 95],
            "공컨테이너 수급지수": [100, 96, 91],
            "운영 리스크": [42, 55, 68]
        })

        st.dataframe(
            monthly_schema,
            use_container_width=True,
            hide_index=True
        )

    st.info(
        """
        본 서비스는 관세청 수출입 실적 기반 AI 수요예측과 공급망 리스크 조기경보 기능을 구현했습니다.
        항만운영 통계, 공컨테이너 수급 데이터, 선박 입출항 정보는 향후 확장 데이터로 설계했으며,
        상세 API 명세와 안정적인 호출 구조가 확보되면 월별 인덱스 방식으로 결합할 수 있습니다.
        """
    )