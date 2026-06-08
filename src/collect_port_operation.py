import os
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

BASE_URL = "https://apis.data.go.kr/B551504/ipaPortStatistics"

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = RAW_DIR / "port_operation.csv"


def hide_service_key(url: str) -> str:
    if SERVICE_KEY:
        return url.replace(SERVICE_KEY, "SERVICE_KEY_HIDDEN")
    return url


def print_xml_tree(root, max_items=120):
    count = 0
    for elem in root.iter():
        text = elem.text.strip() if elem.text else ""
        print(f"{elem.tag}: {text[:120]}")
        count += 1
        if count >= max_items:
            break


def parse_items(root):
    rows = []

    # 일반적인 공공데이터 XML 구조: <item>...</item>
    for item in root.findall(".//item"):
        row = {}
        for child in item:
            row[child.tag] = child.text
        rows.append(row)

    return rows


def try_request(params):
    response = requests.get(BASE_URL, params=params, timeout=30)

    print("=" * 80)
    print("상태코드:", response.status_code)
    print("요청 URL:", hide_service_key(response.url))
    print("응답 앞부분:")
    print(response.text[:1500])

    return response


def main():
    if not SERVICE_KEY:
        print(".env에서 DATA_GO_KR_SERVICE_KEY를 찾지 못했습니다.")
        return

    print("인증키 로드 성공")
    print("인증키 길이:", len(SERVICE_KEY))

    # 파라미터명을 모르는 상태이므로 최소 공통 파라미터와 기간 추정 파라미터를 순차 테스트
    test_params_list = [
        # 1. 기본 공공데이터 방식
        {
            "serviceKey": SERVICE_KEY,
            "pageNo": "1",
            "numOfRows": "10",
        },
        # 2. 기본 + XML 명시
        {
            "serviceKey": SERVICE_KEY,
            "pageNo": "1",
            "numOfRows": "10",
            "resultType": "xml",
        },
        # 3. 연도/월 추정 1
        {
            "serviceKey": SERVICE_KEY,
            "pageNo": "1",
            "numOfRows": "10",
            "year": "2024",
            "month": "01",
        },
        # 4. 연도/월 추정 2
        {
            "serviceKey": SERVICE_KEY,
            "pageNo": "1",
            "numOfRows": "10",
            "stdrYr": "2024",
            "stdrMm": "01",
        },
        # 5. 연월 추정
        {
            "serviceKey": SERVICE_KEY,
            "pageNo": "1",
            "numOfRows": "10",
            "stdrYm": "202401",
        },
        # 6. 시작/종료 연월 추정
        {
            "serviceKey": SERVICE_KEY,
            "pageNo": "1",
            "numOfRows": "10",
            "strtYymm": "202401",
            "endYymm": "202412",
        },
        # 7. start/end 추정
        {
            "serviceKey": SERVICE_KEY,
            "pageNo": "1",
            "numOfRows": "10",
            "startYear": "2024",
            "startMonth": "01",
            "endYear": "2024",
            "endMonth": "12",
        },
    ]

    all_rows = []

    for idx, params in enumerate(test_params_list, start=1):
        print("\n")
        print(f"테스트 {idx} 실행")

        try:
            response = try_request(params)
        except requests.RequestException as e:
            print("요청 실패:", e)
            continue

        if response.status_code != 200:
            print("HTTP 상태코드가 200이 아닙니다.")
            continue

        try:
            root = ET.fromstring(response.content)
        except ET.ParseError as e:
            print("XML 파싱 실패:", e)
            print("응답이 XML이 아닐 수 있습니다.")
            continue

        print("\nXML 구조 일부:")
        print_xml_tree(root)

        rows = parse_items(root)
        print("\nitem 행 수:", len(rows))

        if rows:
            all_rows = rows
            print(f"테스트 {idx}에서 item 데이터 발견")
            break

    if not all_rows:
        print("\nitem 데이터를 찾지 못했습니다.")
        print("가능성:")
        print("1. End Point 뒤에 상세기능명이 더 필요할 수 있음")
        print("2. 필수 파라미터명이 현재 추정값과 다를 수 있음")
        print("3. 응답 구조가 item 태그가 아닐 수 있음")
        print("4. API 서버 내부 오류 또는 명세 미공개 문제일 수 있음")
        return

    df = pd.DataFrame(all_rows)

    print("\n수집 데이터 미리보기")
    print(df.head())
    print("\n컬럼")
    print(df.columns.tolist())

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"\nCSV 저장 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()