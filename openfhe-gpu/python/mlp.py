#!/usr/bin/env python3
"""
OpenFHE GPU 기반 MLP Layer 사용 예시 (수정 버전)
- GPU 결과는 gpu_to_cpu() -> decrypt() -> Plaintext.real() 로 확인
"""

import sys
import gc
import numpy as np

# Add the build directory to Python path
sys.path.insert(0, '/opt/openfhe-gpu/build/python')

import openfhe_gpu  # 수정된 GPU binding 모듈

def run_mlp_gpu_example():
    """GPU를 활용한 MLP layer 연산 예시"""

    # 1. CKKS context 생성
    print("Creating CKKS context...")
    ctx = openfhe_gpu.CkksContext()
    print("   ✅ CKKS context created successfully")

    # 2. 입력 데이터 준비 (MLP의 input features)
    print("Preparing input data...")
    num_features = 4  # 4개의 입력 feature
    feature_dim = 8   # 각 feature의 차원

    # 랜덤 입력 데이터 생성
    input_features = []
    for i in range(num_features):
        feature = np.random.random(feature_dim).astype(np.float64)
        input_features.append(feature)
        print(f"Input feature {i}: {feature}")

    # 3. 가중치(coefficients)와 bias 준비
    print("\nPreparing weights and bias...")
    coefficients = [0.1, 0.2, 0.3, 0.4]  # 각 feature에 대한 가중치
    bias = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float64)
    print(f"Coefficients: {coefficients}")
    print(f"Bias:         {bias}")

    # 4. 입력 데이터를 암호화하고 GPU로 변환
    print("\nEncrypting inputs and converting to GPU...")
    gpu_ct_list = []

    for i in range(num_features):
        # 4-1) encode (실수 -> CKKS)
        pt = ctx.encode_only_float(input_features[i].tolist())
        print(f"Encoded feature {i}")

        # 4-2) CPU에서 암호화
        cpu_ct = ctx.encrypt(pt)
        print(f"Encrypted feature {i} on CPU")

        # 4-3) GPU로 변환 (정확 매핑)
        gpu_ct = ctx.loadAccurate(cpu_ct)
        gpu_ct_list.append(gpu_ct)
        print(f"Converted feature {i} to GPU")

    # 5. Bias를 Plaintext로 인코딩
    bias_pt = ctx.encode_only_float(bias.tolist())
    print("Encoded bias as plaintext")

    # 6. GPU에서 MLP layer 연산 수행 (GPU에서 bias 더하기 포함)
    print("\nRunning MLP layer on GPU (with GPU bias addition)...")
    try:
        # run_layer는 GPU 암호문(Giphertext)을 반환
        gpu_result_ct = ctx.run_layer(gpu_ct_list, coefficients, bias_pt)
        print("MLP layer computation completed on GPU with bias addition")

        # GPU 결과를 CPU 암호문으로 가져와 복호
        cpu_res_ct = ctx.gpu_to_cpu(gpu_result_ct)
        pt_res = ctx.decrypt(cpu_res_ct)
        gpu_result = np.array(pt_res.real(), dtype=np.float64)
        print(f"Decrypted GPU result: {gpu_result}")

        # CPU에서 같은 연산 수행하여 예상 결과 계산
        print("\nVerifying with CPU computation...")
        cpu_result = np.zeros(feature_dim, dtype=np.float64)
        for i in range(num_features):
            cpu_result += coefficients[i] * input_features[i]
        cpu_result += bias
        print(f"Expected CPU result:  {cpu_result}")

        # 비교
        diff = np.abs(gpu_result - cpu_result)
        print(f"Abs diff:            {diff}")
        print(f"Max abs diff:        {diff.max():.6e}")
        print(f"Allclose (rtol=1e-3, atol=1e-3): {np.allclose(gpu_result, cpu_result, rtol=1e-3, atol=1e-3)}")

        # 안전하게 GPU 객체 먼저 정리(컨텍스트보다 앞서 파괴)
        del gpu_result_ct, gpu_ct_list
        gc.collect()

        return gpu_result, cpu_result

    except Exception as e:
        print(f"❌ Error during GPU computation: {e}")
        return None, None

def run_single_feature_example():
    """단일 feature를 사용한 간단한 예시"""

    print("\n" + "="*50)
    print("SINGLE FEATURE EXAMPLE")
    print("="*50)

    # CKKS context 생성
    ctx = openfhe_gpu.CkksContext()
    print("   ✅ CKKS context created successfully")

    # 입력 데이터
    input_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float64)
    coefficient = 0.5
    bias = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float64)

    print(f"Input:        {input_data}")
    print(f"Coefficient:  {coefficient}")
    print(f"Bias:         {bias}")

    # 암호화 및 GPU 변환
    pt = ctx.encode_only_float(input_data.tolist())
    cpu_ct = ctx.encrypt(pt)
    gpu_ct = ctx.loadAccurate(cpu_ct)
    bias_pt = ctx.encode_only_float(bias.tolist())

    # GPU에서 연산
    print("Attempting GPU computation...")
    try:
        result_ct = ctx.run_layer([gpu_ct], [coefficient], bias_pt)
        print("GPU computation completed")

        # 결과 확인: GPU -> CPU -> 복호 -> 실수부
        cpu_ct_out = ctx.gpu_to_cpu(result_ct)
        pt_out = ctx.decrypt(cpu_ct_out)
        result = np.array(pt_out.real(), dtype=np.float64)

        expected = coefficient * input_data + bias

        print(f"GPU result: {result}")
        print(f"Expected:   {expected}")
        print(f"Abs diff:   {np.abs(result - expected)}")
        print(f"Allclose (rtol=1e-3, atol=1e-3): {np.allclose(result, expected, rtol=1e-3, atol=1e-3)}")

        # 안전 정리
        del result_ct, gpu_ct
        gc.collect()

        return result, expected
    except Exception as e:
        print(f"Error during GPU computation: {e}")
        return None, None

def run_simple_gpu_test():
    """가장 간단한 GPU 테스트"""
    print("\n" + "="*50)
    print("SIMPLE GPU TEST")
    print("="*50)

    print("Creating CKKS context...")
    # CKKS context 생성
    try:
        ctx = openfhe_gpu.CkksContext()
        print("   ✅ CKKS context created successfully")
    except Exception as e:
        print(f"   ❌ CKKS context creation failed: {e}")
        return None, None

    # 간단한 데이터
    input_data = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    coefficient = 1.0
    bias = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    print(f"Input: {input_data}")
    print(f"Coefficient: {coefficient}")
    print(f"Bias: {bias}")

    # 암호화 및 GPU 변환
    pt = ctx.encode_only_float(input_data.tolist())
    cpu_ct = ctx.encrypt(pt)
    gpu_ct = ctx.loadAccurate(cpu_ct)
    bias_pt = ctx.encode_only_float(bias.tolist())

    print("Testing basic GPU operations...")

    # 1. 간단한 GPU -> CPU 변환 테스트
    print("Testing GPU->CPU conversion...")
    print("  - gpu_ct object created")
    print(f"  - gpu_ct type: {type(gpu_ct)}")
    try:
        print("  - Calling ctx.gpu_to_cpu()...")
        cpu_ct_out = ctx.gpu_to_cpu(gpu_ct)
        print("✅ GPU->CPU conversion successful")
    except Exception as e:
        print(f"❌ GPU->CPU conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

    # 2. 간단한 다항식 연산 테스트 - 일단 스킵
    print("Skipping polynomial operation test...")
    # try:
    #     result_ct = ctx.poly2_x2_plus_x(gpu_ct)
    #     print("✅ Polynomial operation successful")
    # except Exception as e:
    #     print(f"❌ Polynomial operation failed: {e}")
    #     import traceback
    #     traceback.print_exc()
    #     return None, None

    # 3. run_layer 테스트 (가장 간단한 버전)
    print("Testing run_layer...")
    try:
        result_ct = ctx.run_layer([gpu_ct], [coefficient], bias_pt)
        print("✅ Simple run_layer successful")

        # 결과 확인
        cpu_ct_out = ctx.gpu_to_cpu(result_ct)
        pt_out = ctx.decrypt(cpu_ct_out)
        result = np.array(pt_out.real(), dtype=np.float64)
        expected = coefficient * input_data + bias

        print(f"GPU result: {result}")
        print(f"Expected:   {expected}")
        print(f"Diff:       {np.abs(result - expected)}")

        return result, expected
    except Exception as e:
        print(f"❌ run_layer failed: {e}")
        return None, None

if __name__ == "__main__":
    print("OpenFHE GPU MLP Layer Example")
    print("="*40)

    # 간단한 GPU 테스트 먼저 실행
    try:
        simple_gpu, simple_cpu = run_simple_gpu_test()
        if simple_gpu is not None:
            print("\n✅ Simple GPU test completed successfully!")
        else:
            print("\n❌ Simple GPU test failed!")
    except Exception as e:
        print(f"❌ Simple GPU test failed with exception: {e}")

    print("\nExample completed!")

    # GPU 리소스 정리
    gc.collect()
    print("GPU resources cleaned up")