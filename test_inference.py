import requests
import time

# Triton 서버가 완전히 시작될 때까지 대기
time.sleep(10)

url = "http://localhost:8080/api/v1/predict"
files = {"file": ("image.png", open("./truck.png", "rb"), "image/png")}

try:
    response = requests.post(url, files=files)
    response.raise_for_status()  # 에러 발생시 예외 발생
    print(response.json())
except requests.exceptions.RequestException as e:
    print(f"Error occurred: {e}")
