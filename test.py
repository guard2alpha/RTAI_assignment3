import sys
import os
<<<<<<< HEAD
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
=======

MARABOU_DIR = "/content/Marabou"
sys.path.append(MARABOU_DIR)

try:
    from maraboupy import Marabou
    from maraboupy import MarabouCore
    from maraboupy import MarabouUtils  # [수정된 부분 1] 파이썬 방정식 도구를 불러옵니다!
except ImportError:
    print("Marabou 모듈 불러오기 실패")
    sys.exit(1)

model_path = "my_mlp.onnx"
if os.path.exists(model_path):
    print(f"'{model_path}' 검증 시작\n")
    network = Marabou.read_onnx(model_path)

    inputVars = network.inputVars[0].flatten()
    outputVars = network.outputVars[0].flatten()

    # 1. 입력 제약 조건 (Input Constraints)
    for var in inputVars:
        network.setLowerBound(int(var), -0.1)
        network.setUpperBound(int(var), 0.1)

    # =================================================================
    # 2. 출력 제약 조건 (Output Constraints)
    # =================================================================
    target_class = 3
    attack_class = 8
    
    print(f"제약 조건 추가 중: Output[{attack_class}] >= Output[{target_class}] 가 되는 경우가 있는가?")
    
    # [수정된 부분 2] MarabouCore가 아닌 MarabouUtils를 사용하여 방정식을 만듭니다.
    eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.GE)
    
    # 부등식: (+1.0 * Output[attack_class]) + (-1.0 * Output[target_class]) >= 0.0
    eq.addAddend(1.0, int(outputVars[attack_class]))
    eq.addAddend(-1.0, int(outputVars[target_class]))
    eq.setScalar(0.0) 
    
    network.addEquation(eq)
    # =================================================================

    print("\n계산 중...")
    exitCode, vals, stats = network.solve()

    # 3. 결과 출력 및 반례(Counterexample) 분석
    print("\n" + "="*40)
    print(f"최종 검증 결과: {exitCode.upper()}")
    print("="*40)

    if exitCode == "sat":
        print(f"\n[!] 반례가 존재합니다 (SAT). 모델이 숫자 {attack_class}로 잘못 예측하는 취약점을 찾았습니다.")
        print("구체적인 값은 다음과 같습니다:\n")

        print("--- 입력 변수 (Input Variables) 샘플 ---")
        for i in range(min(5, len(inputVars))):
            var_idx = int(inputVars[i])
            print(f" Input[{i}] (변수번호 {var_idx}) = {vals[var_idx]:.6f}")
        print("... (나머지 생략) ...\n")

        print("--- 출력 변수 (Output Variables) ---")
        for i in range(len(outputVars)):
            var_idx = int(outputVars[i])
            print(f" Output Class {i} (변수번호 {var_idx}) = {vals[var_idx]:.6f}")

    elif exitCode == "unsat":
        print(f"\n[O] 해당 섭동 범위(-0.1 ~ 0.1) 내에서는 절대 숫자 {attack_class}로 잘못 예측하지 않습니다. (UNSAT = 안전함)")

else:
    print(f"에러: {model_path} 파일을 찾을 수 없습니다.")
>>>>>>> fe44f2d52abb320ac4d74adce5e46bb04fcd50aa
