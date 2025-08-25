// openfhe_gpu_pybind.cpp
#include <memory>
#include <vector>
#include <complex>
#include <stdexcept>
#include <cmath>
#include <map>
#include <iostream>
#include <cuda_runtime.h>

#include "openfhe.h"
#include "gpu/Utils.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

namespace py = pybind11;
using namespace lbcrypto;

static inline void checkCudaSync(const char* where) {
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "[CUDA] " << where << " sync error: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(std::string("CUDA sync error at ") + where + ": " + cudaGetErrorString(err));
    }
}

// --------------------------- 래퍼 타입 ---------------------------

// CPU Ciphertext 래퍼
struct PyCiphertext {
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> value;
};

// GPU Ciphertext 래퍼 (CtAccurate 보유 + 컨텍스트 보유)
class PyGiphertext {
public:
    ckks::CtAccurate value;
    std::shared_ptr<ckks::Context> ctx; // 컨텍스트 수명 보장

    PyGiphertext() = default;
    explicit PyGiphertext(ckks::CtAccurate v, std::shared_ptr<ckks::Context> c = {})
        : value(std::move(v)), ctx(std::move(c)) {}

    // 복사 금지(중복 해제 방지)
    PyGiphertext(const PyGiphertext&) = delete;
    PyGiphertext& operator=(const PyGiphertext&) = delete;
    // 이동 허용
    PyGiphertext(PyGiphertext&&) noexcept = default;
    PyGiphertext& operator=(PyGiphertext&&) noexcept = default;
};

// Plaintext 래퍼
struct PyPlaintext {
    lbcrypto::Plaintext value;
};

// ------------------------- CKKS GPU 컨텍스트 -------------------------

class PyCkksContext {
public:
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc_;
    lbcrypto::KeyPair<lbcrypto::DCRTPoly> keyPair_;
    std::shared_ptr<ckks::Context> gpu_context_;         // shared_ptr 로 관리
    ckks::EvaluationKey evk_;
    std::map<uint32_t, ckks::EvaluationKey> loaded_rot_keys_;

    PyCkksContext() {
        // ===== 파라미터 설정 =====
        CCParams<CryptoContextCKKSRNS> parameters;
        parameters.SetScalingModSize(59);
        parameters.SetFirstModSize(60);

        SecretKeyDist secretKeyDist = UNIFORM_TERNARY;
        parameters.SetSecretKeyDist(secretKeyDist);

        parameters.SetSecurityLevel(HEStd_NotSet);
        parameters.SetRingDim(1 << 14);
        parameters.SetBatchSize(8);

        parameters.SetNumLargeDigits(3);
        parameters.SetKeySwitchTechnique(HYBRID);
        ScalingTechnique rescaleTech = FLEXIBLEAUTO;
        parameters.SetScalingTechnique(rescaleTech);

        std::vector<uint32_t> levelBudget = {3, 3};
        uint32_t levelsAvailableAfterBootstrap = 25;
        usint depth = levelsAvailableAfterBootstrap + FHECKKSRNS::GetBootstrapDepth(levelBudget, secretKeyDist);
        parameters.SetMultiplicativeDepth(depth);

        // ===== 컨텍스트 및 키 생성 =====
        cc_ = GenCryptoContext(parameters);
        cc_->Enable(PKE);
        cc_->Enable(KEYSWITCH);
        cc_->Enable(LEVELEDSHE);
        cc_->Enable(ADVANCEDSHE);
        cc_->Enable(FHE);

        keyPair_ = cc_->KeyGen();
        cc_->EvalMultKeyGen(keyPair_.secretKey);
        cc_->EvalAtIndexKeyGen(keyPair_.secretKey, {1, -1});

        // ===== GPU 컨텍스트 생성 =====
        const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc_->GetCryptoParameters());
        gpu_context_ = std::make_shared<ckks::Context>(GenGPUContext(cryptoParams));
        gpu_context_->EnableMemoryPool();

        // ===== 평가키 로드(재선형 키/회전 키) =====
        const std::string keyTag = keyPair_.publicKey->GetKeyTag();
        evk_ = LoadEvalMultRelinKey(cc_, keyTag);
        gpu_context_->preloaded_evaluation_key = &evk_;

        const std::map<usint, EvalKey<DCRTPoly>> evalKeys =
            cc_->GetEvalAutomorphismKeyMap(keyPair_.publicKey->GetKeyTag());
        const int num_loaded_keys = 16;  // GPU 메모리 절약을 위해 일부만 탑재
        for (const auto& pair : evalKeys) {
            if (static_cast<int>(loaded_rot_keys_.size()) >= num_loaded_keys && num_loaded_keys >= 0) break;
            loaded_rot_keys_[std::get<0>(pair)] = LoadRelinKey(std::get<1>(pair));
        }
        gpu_context_->preloaded_rotation_key_map = &loaded_rot_keys_;
    }

    ~PyCkksContext() = default;

    // ===== 인코딩 =====
    PyPlaintext encode(const std::vector<std::complex<double>>& input) {
        PyPlaintext pt;
        pt.value = cc_->MakeCKKSPackedPlaintext(input);
        pt.value->SetLength(static_cast<uint32_t>(input.size()));

        return pt;
    }

    PyPlaintext encode_only_float(const std::vector<double>& input) {
        std::vector<std::complex<double>> complexInput(input.begin(), input.end());
        PyPlaintext pt;
        pt.value = cc_->MakeCKKSPackedPlaintext(complexInput);
        pt.value->SetLength(static_cast<uint32_t>(complexInput.size()));

        return pt;
    }

    // ===== 암/복호 =====
    PyCiphertext encrypt(const PyPlaintext& pt) {
        PyCiphertext ct;
        ct.value = cc_->Encrypt(keyPair_.publicKey, pt.value);
        return ct;
    }

    PyPlaintext decrypt(const PyCiphertext& ct) {
        PyPlaintext pt;
        cc_->Decrypt(keyPair_.secretKey, ct.value, &pt.value);
        return pt;
    }

    // ===== CPU -> GPU 로드 (정확 매핑) =====
    PyGiphertext load(const PyCiphertext& ct, bool verbose=false) const {
        ckks::CtAccurate result = LoadAccurateCiphertext(ct.value, verbose);
        return PyGiphertext(std::move(result), gpu_context_);
    }

    PyGiphertext loadAccurate(const PyCiphertext& ct, bool verbose=false) const {
        ckks::CtAccurate result = LoadAccurateCiphertext(ct.value, verbose);
        return PyGiphertext(std::move(result), gpu_context_);
    }

    // ===== MLP 레이어: 선형 결합 + bias =====
    // 간단하고 안정적인 구현
    PyGiphertext run_layer(const std::vector<std::shared_ptr<PyGiphertext>>& gpu_ct_x,
        const std::vector<double>& coefficients,
        const PyPlaintext& bias_pt)
    {
    // --- 방어적 체크 ---
    if (!gpu_context_) throw std::runtime_error("GPU context is null");
    if (gpu_ct_x.empty()) throw std::invalid_argument("gpu_ct_x is empty");
    if (coefficients.size() != gpu_ct_x.size()) throw std::invalid_argument("coefficients size must match gpu_ct_x size");

    // element params (GPU<->CPU 변환에 필요)
    auto elementParams = cc_->GetCryptoParameters()->GetElementParams();

    try {
        // 1) GPU ciphertext들을 CPU로 변환 (한 번만 수행)
        std::vector<ConstCiphertext<DCRTPoly>> cpu_inputs;
        cpu_inputs.reserve(gpu_ct_x.size());

        for (const auto& gpu_ct : gpu_ct_x) {
            // 예제 패턴: 유효한 Ciphertext 객체를 먼저 생성 후 필드를 채움
            Ciphertext<DCRTPoly> cpu_ct = std::make_shared<CiphertextImpl<DCRTPoly>>(cc_);
            LoadCtAccurateFromGPU(cpu_ct, gpu_ct->value, elementParams);
            cpu_ct->SetEncodingType(CKKS_PACKED_ENCODING);
            cpu_ct->SetKeyTag(keyPair_.publicKey->GetKeyTag());
            cpu_ct->SetSlots(cc_->GetEncodingParams()->GetBatchSize());
            cpu_inputs.push_back(std::move(cpu_ct));
        }

        // 2) CPU에서 가중합 수행
        auto cpu_result = cc_->EvalLinearWSum(cpu_inputs, coefficients);
        cpu_result->SetSlots(cc_->GetEncodingParams()->GetBatchSize());

        // 3) CPU에서 bias와 더해 스케일/레벨을 안전히 정렬
        auto bias_cpu_ct = cc_->Encrypt(keyPair_.publicKey, bias_pt.value);
        bias_cpu_ct->SetEncodingType(CKKS_PACKED_ENCODING);
        bias_cpu_ct->SetKeyTag(keyPair_.publicKey->GetKeyTag());
        bias_cpu_ct->SetSlots(cc_->GetEncodingParams()->GetBatchSize());
        auto cpu_sum = cc_->EvalAdd(cpu_result, bias_cpu_ct);
        cpu_sum->SetSlots(cc_->GetEncodingParams()->GetBatchSize());

        // 4) CPU 결과를 GPU로 로드하여 반환
        ckks::CtAccurate acc = LoadAccurateCiphertext(cpu_sum);
        return PyGiphertext(std::move(acc), gpu_context_);

    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("run_layer failed: ") + e.what());
    }
}


    // ===== 다항식 연산: x^2 + x =====
    PyGiphertext poly2_x2_plus_x(const PyGiphertext& gpu_ct) {
        if (!gpu_context_) throw std::runtime_error("GPU context is null");

        try {
            // GPU에서 x^2 + x 계산
            ckks::CtAccurate x_squared = gpu_context_->EvalMultAndRelin(gpu_ct.value, gpu_ct.value, evk_);
            ckks::CtAccurate result = gpu_context_->Add(x_squared, gpu_ct.value);
            checkCudaSync("poly2_x2_plus_x");

            // 추가 변환 없이 바로 반환
            return PyGiphertext(std::move(result), gpu_context_);
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("poly2_x2_plus_x failed: ") + e.what());
        }
    }

    // ===== GPU -> CPU 변환 =====
    PyCiphertext gpu_to_cpu(const PyGiphertext& gpu_ct) {
        if (!gpu_context_) throw std::runtime_error("GPU context is null");

        // ckks-gpu-bookkeeping.cpp 방식으로 구현
        auto elementParams = cc_->GetCryptoParameters()->GetElementParams();

        PyCiphertext cpu_ct;
        // GPU -> CPU 변환 (정확 매핑)
        try {
            std::cerr << "[gpu_to_cpu] level=" << gpu_ct.value.level
                      << " noiseScaleDeg=" << gpu_ct.value.noiseScaleDeg
                      << " scalingFactor=" << std::setprecision(18) << gpu_ct.value.scalingFactor
                      << std::endl;

            cpu_ct.value = std::make_shared<CiphertextImpl<DCRTPoly>>(cc_);
            LoadCtAccurateFromGPU(cpu_ct.value, gpu_ct.value, elementParams);
            cpu_ct.value->SetEncodingType(CKKS_PACKED_ENCODING);
            cpu_ct.value->SetKeyTag(keyPair_.publicKey->GetKeyTag());
            cpu_ct.value->SetSlots(cc_->GetEncodingParams()->GetBatchSize());

            return cpu_ct;
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("gpu_to_cpu failed: ") + e.what());
        }
    }
};

// --------------------------- PYBIND11 바인딩 ---------------------------

PYBIND11_MODULE(openfhe_gpu, m) {
    m.doc() = "OpenFHE GPU CKKS helpers";

    // Plaintext: 실수부 추출 메서드 노출
    py::class_<PyPlaintext, std::shared_ptr<PyPlaintext>>(m, "Plaintext")
        .def("real", [](const PyPlaintext& pt) {
            std::vector<double> out;
            const auto& v = pt.value->GetCKKSPackedValue();
            out.reserve(v.size());
            for (const auto& z : v) out.push_back(std::real(z));
            return out;
        });

    py::class_<PyCiphertext, std::shared_ptr<PyCiphertext>>(m, "Ciphertext");

    // Giphertext 는 move-only 래퍼이지만 파이썬에서는 shared_ptr 홀더로 다룸
    py::class_<PyGiphertext, std::shared_ptr<PyGiphertext>>(m, "Giphertext");

    py::class_<PyCkksContext>(m, "CkksContext")
        .def(py::init<>())

        // 인코딩/암복호
        .def("encode", &PyCkksContext::encode, py::arg("input"),
             "Encode complex vector to plaintext")
        .def("encode_only_float", &PyCkksContext::encode_only_float, py::arg("input"),
             "Encode float vector to plaintext")
        .def("encrypt", &PyCkksContext::encrypt, py::arg("pt"),
             "Encrypt plaintext to ciphertext")
        .def("decrypt", &PyCkksContext::decrypt, py::arg("ct"),
             "Decrypt ciphertext to plaintext")

        // CPU->GPU 로드 (정확 매핑)
        .def("load", &PyCkksContext::load, py::arg("ct"), py::arg("verbose") = false,
             py::return_value_policy::move,
             py::keep_alive<0,1>(),  // 반환값(0)이 self(1)를 붙잡음
             "Load CPU ciphertext to GPU")
        .def("loadAccurate", &PyCkksContext::loadAccurate, py::arg("ct"), py::arg("verbose") = false,
             py::return_value_policy::move,
             py::keep_alive<0,1>(),
             "Load CPU ciphertext to GPU with accurate mapping")

        // MLP 레이어
        .def("run_layer", &PyCkksContext::run_layer,
             py::arg("gpu_ct_x"), py::arg("coefficients"), py::arg("bias_pt"),
             py::return_value_policy::move,
             py::keep_alive<0,1>(),
             "Run MLP layer on GPU with multiple encrypted inputs, returns GPU ciphertext")

        // 예시: 다항식
        .def("poly2_x2_plus_x", &PyCkksContext::poly2_x2_plus_x, py::arg("gpu_ct"),
             py::return_value_policy::move,
             py::keep_alive<0,1>(),
             "Compute x^2 + x on GPU ciphertext")

        // GPU->CPU 변환
        .def("gpu_to_cpu", &PyCkksContext::gpu_to_cpu, py::arg("gpu_ct"),
             py::return_value_policy::move,
             "Convert GPU ciphertext to CPU ciphertext");
}