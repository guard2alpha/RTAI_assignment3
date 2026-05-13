# Reliable and Trustworthy AI - Assignment #3
## Marabou를 이용한 신경망 검증 실습

본 프로젝트는 SMT(Satisfiability Modulo Theories) 기반의 신경망 검증 도구인 **Marabou**를 설치하고, 외부 ONNX 모델(`my_mlp.onnx`)에 대해 입력 제약 조건을 설정하여 모델의 안전성을 검증하는 과정을 담고 있다.

---

##(File Structure)

* **my_mlp.onnx**: 검증 대상이 되는 외부 MLP 모델 파일.
* **test.py**: Marabou Python API를 사용하여 검증 쿼리를 실행하는 메인 스크립트.
* **requirements.txt**: 프로젝트 실행을 위해 필요한 Python 패키지 목록.
* **report.pdf**: Marabou 리소스 분석 및 검증 결과 해석 보고서.
* **README.md**: 프로젝트 개요 및 설치/실행 가이드.

---

##(Installation)

본 프로젝트는 **Linux (Ubuntu 22.04 / Google Colab)** 환경에서 빌드 및 테스트됐다. (Windows 환경에서는 빌드 오류가 발생할 수 있으므로 Linux 환경 사용을 권장)

### 1. 설치
```bash
sudo apt-get update
sudo apt-get install -y cmake wget build-essential git python3-dev

git clone [https://github.com/NeuralNetworkVerification/Marabou/](https://github.com/NeuralNetworkVerification/Marabou/)
cd Marabou
mkdir build
cd build
cmake .. -DBUILD_PYTHON=ON
make -j4

pip install -r requirements.txt
