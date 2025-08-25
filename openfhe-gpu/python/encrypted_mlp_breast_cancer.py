#!/usr/bin/env python3
"""
Breast Cancer 데이터셋에 대해
- 평문 MLP (히든 3층, ReLU) 학습 및 정확도 측정
- 테스트 단계: 입력을 암호화하여 동일한 구조의 암호화된 MLP로 추론(활성함수는 poly2_x2_plus_x 대체)
  -> 평문 MLP와 암호화된 MLP 정확도 비교

주의:
- CKKS batch size가 8로 세팅되어 있으므로, 스칼라 값을 8개 슬롯에 복제하여 벡터로 취급합니다.
- 암호 추론 시 각 뉴런의 선형결합은 ctx.run_layer를 사용합니다.
- 활성함수는 ReLU 대신 poly2_x2_plus_x (x^2 + x) 사용.
- 최종 출력층은 선형 로짓을 복호 후 시그모이드를 평문에서 적용하여 라벨 결정.
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
        self.l1 = nn.Linear(input_dim, h1)
        self.l2 = nn.Linear(h1, h2)
        self.l3 = nn.Linear(h2, h3)
        self.out = nn.Linear(h3, 1)
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
                              coefs: list[np.ndarray],
                              intercepts: list[np.ndarray]) -> np.ndarray:
    # bias를 Plaintext로 미리 준비
    bias_pts_layers = build_bias_plaintexts(ctx, [b.tolist() for b in intercepts[:-1]])
    # 출력층 bias는 별도로 보관 (1차원)
    output_bias_pts = [ctx.encode_only_float(replicate_scalar_to_slots(float(intercepts[-1][k])))
                       for k in range(intercepts[-1].shape[0])]

    y_pred = []
    for i in range(X.shape[0]):
        # 1) 입력 암호화(GPU)
        gpu_inputs = encrypt_input_sample(ctx, X[i])

        try:
            # 2) 히든 레이어 1~3 (ReLU 대체: poly2_x2_plus_x)
            h1_gpu = encrypted_dense_layer(ctx, gpu_inputs, coefs[0], bias_pts_layers[0], use_poly_activation=True)
            # 이전 계층 중간 결과 메모리 정리
            del gpu_inputs
            gc.collect()

            h2_gpu = encrypted_dense_layer(ctx, h1_gpu, coefs[1], bias_pts_layers[1], use_poly_activation=True)
            del h1_gpu
            gc.collect()

            h3_gpu = encrypted_dense_layer(ctx, h2_gpu, coefs[2], bias_pts_layers[2], use_poly_activation=True)
            del h2_gpu
            gc.collect()

            # 3) 출력층(선형) -> 복호 후 시그모이드 -> 라벨
            # coefs[-1] shape: (n_in3, n_out)
            logits_gpu = encrypted_dense_layer(ctx, h3_gpu, coefs[3], output_bias_pts, use_poly_activation=False)
            del h3_gpu
            gc.collect()

            # 이진 분류 가정: 출력 뉴런 1개
            logit_gpu = logits_gpu[0]
            cpu_ct = ctx.gpu_to_cpu(logit_gpu)
            pt = ctx.decrypt(cpu_ct)
            real_vec = np.array(pt.real(), dtype=np.float64)
            logit = float(real_vec[0])
            prob = sigmoid(logit)
            y_pred.append(1 if prob >= 0.5 else 0)

            del logits_gpu, logit_gpu, cpu_ct, pt
            gc.collect()
        except Exception as e:
            print(f"[Encrypted Inference] Error on sample {i}: {e}")
            y_pred.append(0)
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
    print("\nCreating CKKS GPU context...")
    ctx = openfhe_gpu.CkksContext()
    print("   ✅ CKKS context created")

    # PyTorch MLP의 가중치/바이어스 추출
    # weight: (out_features, in_features) -> (in, out)로 transpose하여 사용
    l1_w = model.l1.weight.detach().cpu().numpy().astype(np.float64).T
    l1_b = model.l1.bias.detach().cpu().numpy().astype(np.float64)
    l2_w = model.l2.weight.detach().cpu().numpy().astype(np.float64).T
    l2_b = model.l2.bias.detach().cpu().numpy().astype(np.float64)
    l3_w = model.l3.weight.detach().cpu().numpy().astype(np.float64).T
    l3_b = model.l3.bias.detach().cpu().numpy().astype(np.float64)
    o_w = model.out.weight.detach().cpu().numpy().astype(np.float64).T
    o_b = model.out.bias.detach().cpu().numpy().astype(np.float64)

    coefs = [l1_w, l2_w, l3_w, o_w]
    intercepts = [l1_b, l2_b, l3_b, o_b]

    # 방어적 확인: 히든 3층 + 출력층
    if len(coefs) != 4 or len(intercepts) != 4:
        raise RuntimeError("This script expects exactly 3 hidden layers and 1 output layer.")

    # 5) 암호화 추론
    print("Running encrypted inference on test set (ReLU -> poly2_x2_plus_x)...")
    y_pred_enc = encrypted_predict_binary(ctx, X_test_s, coefs, intercepts)
    acc_enc = accuracy_score(y_test, y_pred_enc)

    # 6) 결과 비교
    print("\n=== Accuracy Comparison ===")
    print(f"Plain MLP accuracy      : {acc_plain:.4f}")
    print(f"Encrypted MLP accuracy  : {acc_enc:.4f}")

    # 리소스 정리
    del ctx
    gc.collect()


if __name__ == "__main__":
    main()


