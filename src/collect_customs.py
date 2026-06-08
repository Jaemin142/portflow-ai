import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


# =========================
# 기본 설정
# =========================
load_dotenv()

SERVICE_KEY = os.getenv("DATA_GO_KR_SERVICE_KEY")

BASE_URL = "http://apis.data.go.kr/1220000/nitemtrade/getNitemtradeList"

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = RAW_DIR / "customs_trade.csv"

# 수집할 국가
COUNTRIES = {
    "CN": "중국",
    "US": "미국",
    "JP": "일본",
    "VN": "베트남",
}

# 수집할 HS Code
# 관세청 API는 hsSgn을 선택값으로 받을 수 있음.
# 처음에는 2자리 품목군 중심으로 테스트.
HS_CODES = {
    "85": "전기기기",
    "84": "기계류",
    "39": "플라스틱",
    "33": "화장품",
}

# 수집 연도
# 문서상 조회기간은 1년 이내이므로 연도별로 나누어 호출.
YEARS = [2022, 2023, 2024, 2025]


# =========================
# 유틸 함수
# =========================
def check_service_key() -> None:
    if not SERVICE_KEY:
        raise ValueError(
            ".env 파일에서 DATA_GO_KR_SERVICE_KEY를 찾지 못했습니다. "
            ".env 파일을 확인하세요."
        )

    print("인증키 로드 성공")
    print("인증키 길이:", len(SERVICE_KEY))


def parse_xml_response(xml_content: bytes) -> tuple[str | None, str | None, list[dict]]:
    """
    XML 응답을 파싱해서 resultCode, resultMsg, item rows를 반환한다.
    """

    root = ET.fromstring(xml_content)

    # 공공데이터포털 공통 에러 응답
    if root.tag == "OpenAPI_ServiceResponse":
        error_info = {}
        for elem in root.iter():
            error_info[elem.tag] = elem.text

        print("공공데이터포털 에러 응답:")
        for k, v in error_info.items():
            print(f"  {k}: {v}")

        return None, error_info.get("returnAuthMsg"), []

    # 제공기관 응답 코드
    result_code = root.findtext(".//resultCode")
    result_msg = root.findtext(".//resultMsg")

    rows = []
    for item in root.findall(".//item"):
        row = {child.tag: child.text for child in item}
        rows.append(row)

    return result_code, result_msg, rows


def fetch_customs_data(
    country_code: str,
    country_name: str,
    hs_code: str,
    hs_name: str,
    year: int,
) -> list[dict]:
    """
    국가, HS Code, 연도 기준으로 관세청 API를 호출한다.
    """

    params = {
        "serviceKey": SERVICE_KEY,
        "strtYymm": f"{year}01",
        "endYymm": f"{year}12",
        "hsSgn": hs_code,
        "cntyCd": country_code,
    }

    print(
        f"호출 중: {year} / {country_code}({country_name}) / "
        f"HS {hs_code}({hs_name})"
    )

    try:
        response = requests.get(BASE_URL, params=params, timeout=30)
    except requests.RequestException as e:
        print("요청 실패:", e)
        return []

    if response.status_code != 200:
        print("HTTP 오류:", response.status_code)
        print(response.text[:500])
        return []

    try:
        result_code, result_msg, rows = parse_xml_response(response.content)
    except ET.ParseError as e:
        print("XML 파싱 실패:", e)
        print(response.text[:500])
        return []

    if result_code != "00":
        print(f"비정상 응답: resultCode={result_code}, resultMsg={result_msg}")
        return []

    if not rows:
        print("데이터 없음")
        return []

    # 메타정보 추가
    for row in rows:
        row["query_country_code"] = country_code
        row["query_country_name"] = country_name
        row["query_hs_code"] = hs_code
        row["query_hs_name"] = hs_name
        row["query_year"] = year

    print(f"수집 행 수: {len(rows)}")
    return rows


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    수집된 원본 데이터를 정리한다.
    """

    if df.empty:
        return df

    # 총계 행 제거
    if "year" in df.columns:
        df["year"] = df["year"].astype(str).str.strip()
        df = df[df["year"] != "총계"].copy()

    # 숫자 컬럼 변환
    numeric_cols = ["expWgt", "expDlr", "impWgt", "impDlr", "balPayments"]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .replace("-", "0")
            )
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # 날짜 컬럼 생성
    if "year" in df.columns:
        df["year_clean"] = (
            df["year"]
            .astype(str)
            .str.replace(".", "", regex=False)
            .str.replace("-", "", regex=False)
            .str.replace("/", "", regex=False)
            .str[:6]
        )
        df["month"] = pd.to_datetime(df["year_clean"], format="%Y%m", errors="coerce")

    # 컬럼 순서 정리
    preferred_cols = [
        "month",
        "year",
        "query_year",
        "query_country_code",
        "query_country_name",
        "query_hs_code",
        "query_hs_name",
        "statCdCntnKor1",
        "statCd",
        "statKor",
        "hsCd",
        "expWgt",
        "expDlr",
        "impWgt",
        "impDlr",
        "balPayments",
    ]

    existing_cols = [col for col in preferred_cols if col in df.columns]
    other_cols = [col for col in df.columns if col not in existing_cols]

    df = df[existing_cols + other_cols]

    return df


# =========================
# 메인 실행
# =========================
def main() -> None:
    check_service_key()

    all_rows = []

    total_tasks = len(COUNTRIES) * len(HS_CODES) * len(YEARS)
    current_task = 0

    for year in YEARS:
        for country_code, country_name in COUNTRIES.items():
            for hs_code, hs_name in HS_CODES.items():
                current_task += 1

                print("=" * 80)
                print(f"진행률: {current_task}/{total_tasks}")

                rows = fetch_customs_data(
                    country_code=country_code,
                    country_name=country_name,
                    hs_code=hs_code,
                    hs_name=hs_name,
                    year=year,
                )

                all_rows.extend(rows)

                # API 서버에 부담을 주지 않기 위해 짧게 대기
                time.sleep(0.2)

    print("=" * 80)
    print("전체 수집 완료")
    print("전체 원본 행 수:", len(all_rows))

    if not all_rows:
        print("수집된 데이터가 없습니다. 파라미터, 인증키, API 승인 상태를 확인하세요.")
        return

    df = pd.DataFrame(all_rows)
    df = clean_dataframe(df)

    print("정리 후 행 수:", len(df))
    print("컬럼:", df.columns.tolist())
    print(df.head())

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"CSV 저장 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()