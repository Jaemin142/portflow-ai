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

BASE_URL = "https://apis.data.go.kr/B551504/ipaEmpConCargoInfo"

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = RAW_DIR / "empty_container.csv"


def print_xml_tree(root, max_items=80):
    """
    XML 구조를 확인하기 위한 디버그 출력 함수
    """
    count = 0
    for elem in root.iter():
        text = elem.text.strip() if elem.text else ""
        print(f"{elem.tag}: {text[:100]}")
        count += 1
        if count >= max_items:
            break


def parse_items(root):
    """
    XML 안의 item 태그를 자동으로 DataFrame row로 변환
    """
    rows = []

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
    print("요청 URL:", response.url.replace(SERVICE_KEY, "SERVICE_KEY_HIDDEN") if SERVICE_KEY else response.url)
    print("응답 앞부분:")
    print(response.text[:1000])

    return response


def main():
    if not SERVICE_KEY:
        print(".env에서 DATA_GO_KR_SERVICE_KEY를 찾지 못했습니다.")
        return

    print("인증키 로드 성공")
    print("인증키 길이:", len(SERVICE_KEY))

    # 1차 테스트: 가장 기본적인 공공데이터 파라미터만 사용
    test_params_list = [
        {
            "serviceKey": SERVICE_KEY,
            "pageNo": "1",
            "numOfRows": "10",
        },
        {
            "serviceKey": SERVICE_KEY,
            "page": "1",
            "perPage": "10",
        },
        {
            "serviceKey": SERVICE_KEY,
            "pageNo": "1",
            "numOfRows": "10",
            "resultType": "xml",
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
            continue

        print("\nXML 구조 일부:")
        print_xml_tree(root)

        rows = parse_items(root)

        print("\nitem 행 수:", len(rows))

        if rows:
            all_rows = rows
            break

    if not all_rows:
        print("\nitem 데이터를 찾지 못했습니다.")
        print("가능성:")
        print("1. End Point 뒤에 상세기능명이 더 필요할 수 있음")
        print("2. 필수 파라미터가 누락됐을 수 있음")
        print("3. 응답 구조가 item 태그가 아닐 수 있음")
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