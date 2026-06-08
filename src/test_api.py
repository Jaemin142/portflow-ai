import os
from dotenv import load_dotenv

load_dotenv()

service_key = os.getenv("DATA_GO_KR_SERVICE_KEY")

print("인증키 로드 여부:", service_key is not None)
print("인증키 길이:", len(service_key) if service_key else 0)
print("인증키 앞 10자리:", service_key[:10] if service_key else "없음")