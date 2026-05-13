import sys
import os
# 경로 설정
marabou_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "Marabou"))
# 경로 추가
sys.path.append(marabou_path)

from maraboupy import Marabou, MarabouCore

def main():
    print("Marabou 검증 시작 (Docker 환경)...")
    
    # 1. ONNX 모델 로드
    try:
        network = Marabou.read_onnx("my_mlp.onnx")
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        sys.exit(1)

    inputVars = network.inputVars[0][0] 
    outputVars = network.outputVars[0]

    # 2. 섭동(노이즈) 반경 설정 (L_infinity)
    epsilon = 0.01

    # 입력 제약 조건: -0.01 <= x <= 0.01
    for var in inputVars.flatten():
        network.setLowerBound(var, -epsilon)
        network.setUpperBound(var, epsilon)

    # 3. 출력 제약 조건: Class 1의 점수가 Class 0보다 높은 경우(반례)를 찾아라
    network.addInequality([outputVars[1], outputVars[0]], [1.0, -1.0], 0.001)

    # 4. 검증 실행
    print("🔍 Solving... (MarabouCore 동작 중)")
    vals, stats = network.solve()

    # 5. 결과 해석
    print("\n" + "="*40)
    if len(vals) > 0:
        print("결과: SAT (Satisfiable)")
        print("해석: 노이즈로 인해 예측이 바뀔 수 있는 반례(Adversarial input)가 존재.")
    else:
        print("결과: UNSAT (Unsatisfiable)")
        print("해석: 해당 노이즈 범위 내에서는 예측이 절대 바뀌지 않음이 증명.")
    print("="*40)

if __name__ == "__main__":
    main()