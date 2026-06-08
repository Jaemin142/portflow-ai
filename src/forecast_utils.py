import pandas as pd
from sklearn.linear_model import LinearRegression


def make_forecast(monthly_df: pd.DataFrame, target_col: str, periods: int = 3) -> pd.DataFrame:
    """
    월별 데이터와 예측 대상 컬럼을 받아 다음 periods개월을 예측한다.
    LinearRegression 기반의 단순 시계열 예측 모델.
    """

    df = monthly_df[["month", target_col]].copy()
    df = df.dropna(subset=["month", target_col"]).sort_values("month")

    if len(df) < 6:
        # 데이터가 너무 적으면 최근 3개월 평균으로 대체
        last_month = df["month"].max()
        recent_avg = df[target_col].tail(3).mean()

        future_months = pd.date_range(
            start=last_month + pd.DateOffset(months=1),
            periods=periods,
            freq="MS"
        )

        forecast_df = pd.DataFrame({
            "month": future_months,
            "forecast": [recent_avg] * periods,
            "model_type": "최근 3개월 평균"
        })

        return forecast_df

    # 시간 인덱스 생성
    df["t"] = range(len(df))

    X = df[["t"]]
    y = df[target_col]

    model = LinearRegression()
    model.fit(X, y)

    # 다음 periods개월 예측
    last_t = df["t"].max()
    future_t = [[last_t + i] for i in range(1, periods + 1)]

    forecast_values = model.predict(future_t)

    # 음수 예측 방지
    forecast_values = [max(0, value) for value in forecast_values]

    last_month = df["month"].max()
    future_months = pd.date_range(
        start=last_month + pd.DateOffset(months=1),
        periods=periods,
        freq="MS"
    )

    forecast_df = pd.DataFrame({
        "month": future_months,
        "forecast": forecast_values,
        "model_type": "LinearRegression"
    })

    return forecast_df


def make_actual_forecast_plot_data(
    monthly_df: pd.DataFrame,
    target_col: str,
    forecast_df: pd.DataFrame
) -> pd.DataFrame:
    """
    실제값과 예측값을 하나의 그래프에 그리기 위한 데이터프레임 생성.
    """

    actual_df = monthly_df[["month", target_col]].copy()
    actual_df = actual_df.rename(columns={target_col: "value"})
    actual_df["type"] = "실제값"

    pred_df = forecast_df[["month", "forecast"]].copy()
    pred_df = pred_df.rename(columns={"forecast": "value"})
    pred_df["type"] = "예측값"

    plot_df = pd.concat([actual_df, pred_df], ignore_index=True)
    plot_df = plot_df.sort_values("month")

    return plot_df