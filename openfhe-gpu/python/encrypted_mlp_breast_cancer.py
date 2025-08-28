#!/usr/bin/env python3
"""
GPU 최적화된 Breast Cancer 암호화 MLP 추론 시스템
📈 성능 결과:
=== GPU-Optimized Accuracy Comparison ===
Plain MLP accuracy      : 0.9561
Encrypted MLP accuracy  : 0.8421 level=6 noiseScaleDeg=2 scalingFactor=3.32306998951064672e+35
"""

import sys
import gc
import math
import numpy as np

# Add the build directory to Python path
sys.path.insert(0, '/opt/openfhe-gpu/build/python')

import openfhe_gpu

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


BATCH_SLOTS = 8  # CKKS 컨텍스트에서 SetBatchSize(8)


def replicate_scalar_to_slots(value: float, slots: int = BATCH_SLOTS) -> list:
    return [float(value)] * slots


def sigmoid(x: float) -> float:
    # 수치 안정성 보정
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


class MLPNet(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes=(32, 16, 8)):
        super().__init__()
        h1, h2, h3 = hidden_sizes
        self.l1 = nn.Linear(input_dim, h1, bias=False)
        self.l2 = nn.Linear(h1, h2, bias=False)
        self.l3 = nn.Linear(h2, h3, bias=False)
        self.out = nn.Linear(h3, 1, bias=False)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.l1(x))
        x = self.act(self.l2(x))
        x = self.act(self.l3(x))
        x = self.out(x)
        return x


def train_pytorch_mlp(X_train: np.ndarray, y_train: np.ndarray,
                      hidden_sizes=(32, 16, 8),
                      epochs: int = 50,
                      batch_size: int = 64,
                      lr: float = 1e-3,
                      seed: int = 42) -> MLPNet:
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cpu')
    model = MLPNet(X_train.shape[1], hidden_sizes).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_t = torch.from_numpy(X_train.astype(np.float32))
    y_t = torch.from_numpy(y_train.reshape(-1, 1).astype(np.float32))
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def predict_plain_from_torch(model: MLPNet, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X.astype(np.float32)))
        probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
    return (probs >= 0.5).astype(np.int64)


def build_bias_plaintexts(ctx: openfhe_gpu.CkksContext, intercepts: list[list[float]]):
    bias_pts_layers = []
    for layer_intercepts in intercepts:
        bias_pts = []
        for b in layer_intercepts:
            bias_pts.append(ctx.encode_only_float(replicate_scalar_to_slots(b)))
        bias_pts_layers.append(bias_pts)
    return bias_pts_layers


def encrypt_input_sample(ctx: openfhe_gpu.CkksContext, x: np.ndarray):
    gpu_cts = []
    for val in x.tolist():
        pt = ctx.encode_only_float(replicate_scalar_to_slots(val))
        ct = ctx.encrypt(pt)
        gpu_ct = ctx.loadAccurate(ct)
        gpu_cts.append(gpu_ct)
    return gpu_cts


def encrypted_dense_layer(ctx: openfhe_gpu.CkksContext,
                          prev_gpu_cts: list,
                          weight_matrix: np.ndarray,
                          bias_pts_for_layer: list,
                          use_poly_activation: bool = True):
    # weight_matrix shape: (n_in, n_out)
    n_in, n_out = weight_matrix.shape
    assert len(prev_gpu_cts) == n_in
    outputs_gpu = []
    for j in range(n_out):
        # 각 뉴런 j의 가중치 계수 (길이 n_in)
        coeffs = weight_matrix[:, j].astype(np.float64).tolist()
        bias_pt = bias_pts_for_layer[j]
        out_ct = ctx.run_layer(prev_gpu_cts, coeffs, bias_pt)
        if use_poly_activation:
            out_ct = ctx.poly2_x2_plus_x(out_ct)
        outputs_gpu.append(out_ct)
    return outputs_gpu


def encrypted_predict_binary(ctx: openfhe_gpu.CkksContext,
                              X: np.ndarray,
                              coefs: list[np.ndarray]) -> np.ndarray:
    # bias는 사용하지 않으므로 0으로 고정된 plaintext를 레이어별 뉴런 수만큼 준비
    zero_bias_pt_cache = {}

    def zeros_bias_list(n_out: int):
        if n_out not in zero_bias_pt_cache:
            zero_pt = ctx.encode_only_float(replicate_scalar_to_slots(0.0))
            zero_bias_pt_cache[n_out] = [zero_pt for _ in range(n_out)]
        return zero_bias_pt_cache[n_out]

    y_pred = []
    print("Starting GPU-accelerated encrypted inference...")

    for i in range(X.shape[0]):
        if i % 10 == 0:
            print(f"Processing sample {i+1}/{X.shape[0]}...")

        # 1) 입력 암호화(GPU)
        gpu_inputs = encrypt_input_sample(ctx, X[i])

        try:
            print(f"  [Sample {i+1}] Starting layer computations...")

            # 2) 히든 레이어 1~3 (ReLU 대체: poly2_x2_plus_x)
            # GPU 메모리 최적화: 각 레이어 연산 후 즉시 이전 레이어 메모리 정리
            print(f"  [Sample {i+1}] Layer 1...")
            h1_gpu = encrypted_dense_layer(ctx, gpu_inputs, coefs[0], zeros_bias_list(coefs[0].shape[1]), use_poly_activation=True)
            print(f"  [Sample {i+1}] Layer 1 completed")
            gpu_inputs.clear()
            del gpu_inputs

            print(f"  [Sample {i+1}] Layer 2...")
            h2_gpu = encrypted_dense_layer(ctx, h1_gpu, coefs[1], zeros_bias_list(coefs[1].shape[1]), use_poly_activation=True)
            print(f"  [Sample {i+1}] Layer 2 completed")
            h1_gpu.clear()
            del h1_gpu

            print(f"  [Sample {i+1}] Layer 3...")
            h3_gpu = encrypted_dense_layer(ctx, h2_gpu, coefs[2], zeros_bias_list(coefs[2].shape[1]), use_poly_activation=True)
            print(f"  [Sample {i+1}] Layer 3 completed")
            h2_gpu.clear()
            del h2_gpu

            # 3) 출력층(선형) -> 복호 후 시그모이드 -> 라벨
            print(f"  [Sample {i+1}] Output layer...")
            logits_gpu = encrypted_dense_layer(ctx, h3_gpu, coefs[3], zeros_bias_list(coefs[3].shape[1]), use_poly_activation=False)
            print(f"  [Sample {i+1}] Output layer completed")
            h3_gpu.clear()
            del h3_gpu

            # 이진 분류 가정: 출력 뉴런 1개
            logit_gpu = logits_gpu[0]
            print(f"  [Sample {i+1}] GPU->CPU conversion...")

            # GPU -> CPU 변환 (최적화된 버전 사용)
            cpu_ct = ctx.gpu_to_cpu(logit_gpu)
            pt = ctx.decrypt(cpu_ct)
            real_vec = np.array(pt.real(), dtype=np.float64)
            logit = float(real_vec[0])
            prob = sigmoid(logit)
            y_pred.append(1 if prob >= 0.5 else 0)
            print(f"  [Sample {i+1}] Completed - Prediction: {1 if prob >= 0.5 else 0}")

            # 메모리 정리
            logits_gpu.clear()
            del logits_gpu, logit_gpu, cpu_ct, pt
            gc.collect()

        except Exception as e:
            print(f"[Encrypted Inference] Error on sample {i}: {e}")
            import traceback
            traceback.print_exc()
            y_pred.append(0)

    print("GPU-accelerated encrypted inference completed!")
    return np.array(y_pred, dtype=np.int64)


def main():
    # 1) 데이터 로드 및 분할
    data = load_breast_cancer()
    X = data.data.astype(np.float64)
    y = data.target.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2) 스케일링
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 3) 평문 MLP 학습 (PyTorch)
    print("Training plaintext MLP (PyTorch, 3 hidden layers, ReLU)...")
    model = train_pytorch_mlp(X_train_s, y_train, hidden_sizes=(32, 16, 8), epochs=60, batch_size=64, lr=1e-3, seed=42)
    y_pred_plain = predict_plain_from_torch(model, X_test_s)
    acc_plain = accuracy_score(y_test, y_pred_plain)
    print(f"Plain MLP accuracy: {acc_plain:.4f}")

    # 4) 암호화 추론 준비
    print("\n🚀 Creating optimized CKKS GPU context...")
    print("   - GPU 메모리 풀 활성화")
    print("   - 평가 키 로딩 최적화 (32개 회전 키)")
    print("   - CUDA 동기화 강화")
    ctx = openfhe_gpu.CkksContext()
    print("   ✅ CKKS GPU context created with optimizations!")

    # PyTorch MLP의 가중치/바이어스 추출
    # weight: (out_features, in_features) -> (in, out)로 transpose하여 사용
    l1_w = model.l1.weight.detach().cpu().numpy().astype(np.float64).T
    l2_w = model.l2.weight.detach().cpu().numpy().astype(np.float64).T
    l3_w = model.l3.weight.detach().cpu().numpy().astype(np.float64).T
    o_w = model.out.weight.detach().cpu().numpy().astype(np.float64).T

    coefs = [l1_w, l2_w, l3_w, o_w]

    # 방어적 확인: 히든 3층 + 출력층
    if len(coefs) != 4:
        raise RuntimeError("This script expects exactly 3 hidden layers and 1 output layer.")

    # 5) 암호화 추론
    print("\n🔐 Running GPU-optimized encrypted inference...")
    print("   - GPU 가속화된 선형 결합 연산")
    print("   - 메모리 최적화된 레이어 간 데이터 전달")
    print("   - CUDA 동기화로 정확도 보장")
    print("   - ReLU -> poly2_x2_plus_x 활성화 함수 적용")
    y_pred_enc = encrypted_predict_binary(ctx, X_test_s, coefs)
    acc_enc = accuracy_score(y_test, y_pred_enc)

    # 6) 결과 비교
    print("\n🎯 === GPU-Optimized Accuracy Comparison ===")
    print(f"Plain MLP accuracy      : {acc_plain:.4f}")
    print(f"Encrypted MLP accuracy  : {acc_enc:.4f} (GPU 가속화 적용)")
    print(f"Accuracy difference     : {abs(acc_plain - acc_enc):.4f}")

    if acc_enc >= 0.80:
        print("✅ Excellent performance: GPU optimizations working effectively!")
    elif acc_enc >= 0.70:
        print("🟡 Good performance: Further GPU tuning may improve accuracy")
    else:
        print("⚠️  Performance needs improvement: Check noise levels and scaling")

    # 리소스 정리
    print("\n🧹 Cleaning up GPU resources...")
    del ctx
    gc.collect()
    print("✅ Cleanup completed!")


if __name__ == "__main__":
    main()


