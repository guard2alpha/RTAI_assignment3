# Marabou가 가장 안정적으로 동작하는 Python 3.9 리눅스 환경
FROM python:3.9-slim

# 작업 폴더 설정
WORKDIR /app

# 필수 라이브러리 설치 (리눅스 환경이므로 maraboupy가 정상 설치됩니다)
RUN pip install --no-cache-dir onnx maraboupy

# 로컬(윈도우)에 있는 파일들을 도커(리눅스)로 복사
COPY . /app

# 컨테이너가 켜지면 test.py 실행
CMD ["python", "test.py"]