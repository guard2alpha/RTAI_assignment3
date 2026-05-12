import sys
import os

MARABOU_DIR = "/content/Marabou"
sys.path.append(MARABOU_DIR)

try:
    from maraboupy import Marabou
    from maraboupy import MarabouCore
except ImportError:
    print("Marabou 모듈 불러오기 실패")
    sys.exit(1)

model_path = "my_mlp.onnx"
if os.path.exists(model_path):
    print(f"'{model_path}' 검증 시작\n")
    network = Marabou.read_onnx(model_path)
    

    inputVars = network.inputVars[0].flatten()
    outputVars = network.outputVars[0].flatten()

    for var in inputVars:
        network.setLowerBound(int(var), -0.1)
        network.setUpperBound(int(var), 0.1)
        

    print("계산 중...")
    exitCode, vals, stats = network.solve()
    
    # 5. 결과 출력 및 반례(Counterexample) 분석
    print("\n" + "="*40)
    print(f"최종 검증 결과: {exitCode.upper()}")
    print("="*40)
    
    if exitCode == "sat":
        print("\n[!] 반례(Adversarial Input)가 존재합니다 (SAT).")
        print("구체적인 값은 다음과 같습니다:\n")
        
        # 입력값 반례 중 첫 5개만 샘플로 출력 (보고서 작성용)
        print("--- 입력 변수 (Input Variables) ---")
        for i in range(min(5, len(inputVars))):
            var_idx = int(inputVars[i])
            print(f" Input[{i}] (변수번호 {var_idx}) = {vals[var_idx]:.6f}")
            
        print("... (나머지 입력값 생략) ...\n")
        
        # 이 반례를 넣었을 때 모델의 최종 출력값 확인
        print("--- 출력 변수 (Output Variables) ---")
        for i in range(len(outputVars)):
            var_idx = int(outputVars[i])
            print(f" Output Class {i} (변수번호 {var_idx}) = {vals[var_idx]:.6f}")
            
    elif exitCode == "unsat":
        print("\n[O] 해당 조건 내에서는 반례가 없습니다. (UNSAT = 모델이 안전함)")
        
else:
    print(f"에러: {model_path} 파일을 찾을 수 없습니다.")
