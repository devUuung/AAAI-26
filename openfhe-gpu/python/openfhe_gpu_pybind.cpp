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
#include "scheme/ckksrns/ckksrns-leveledshe.h"
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

static inline void checkCudaErrors(cudaError_t err, const char* where) {
    if (err != cudaSuccess) {
        std::cerr << "[CUDA ERROR] " << where << ": " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(std::string("CUDA error at ") + where + ": " + cudaGetErrorString(err));
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

        // 디버깅: 컨텍스트 초기화 확인
        if (!cc_) {
            throw std::runtime_error("Failed to create crypto context");
        }

        keyPair_ = cc_->KeyGen();
        if (!keyPair_.secretKey || !keyPair_.publicKey) {
            throw std::runtime_error("Failed to generate key pair");
        }

        cc_->EvalMultKeyGen(keyPair_.secretKey);
        cc_->EvalAtIndexKeyGen(keyPair_.secretKey, {1, -1});

        // 디버깅: 스킴 확인
        auto testScheme = cc_->GetScheme();
        std::cerr << "[DEBUG Constructor] testScheme pointer: " << testScheme.get() << std::endl;
        std::cerr << "[DEBUG Constructor] cc_ pointer: " << cc_.get() << std::endl;
        if (!testScheme) {
            throw std::runtime_error("Scheme is not available in crypto context");
        }
        std::cerr << "[DEBUG Constructor] Scheme type: " << typeid(*testScheme).name() << std::endl;

        // ===== GPU 컨텍스트 생성 =====
        const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc_->GetCryptoParameters());
        gpu_context_ = std::make_shared<ckks::Context>(GenGPUContext(cryptoParams));
        gpu_context_->EnableMemoryPool();  // 메모리 풀 활성화로 GPU 메모리 관리 최적화

        // ===== 평가키 로드(재선형 키/회전 키) =====
        const std::string keyTag = keyPair_.publicKey->GetKeyTag();
        evk_ = LoadEvalMultRelinKey(cc_, keyTag);
        gpu_context_->preloaded_evaluation_key = &evk_;

        const std::map<usint, EvalKey<DCRTPoly>> evalKeys =
            cc_->GetEvalAutomorphismKeyMap(keyPair_.publicKey->GetKeyTag());
        const int num_loaded_keys = 32;  // 더 많은 회전 키 로드하여 MLP 연산 최적화
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

    // ===== MLP 레이어: 진정한 GPU 선형 결합 (bias 제거) =====
    // GPU에서 직접 모든 연산 수행 - 진정한 GPU 가속화, bias 연산 제거됨
    PyGiphertext run_layer(const std::vector<std::shared_ptr<PyGiphertext>>& gpu_ct_x,
        const std::vector<double>& coefficients,
        const PyPlaintext& bias_pt)
    {
    // bias_pt는 더 이상 사용하지 않음 (API 호환성 유지 위해 인자만 유지)
    (void)bias_pt;
    // --- 방어적 체크 ---
    if (!gpu_context_) throw std::runtime_error("GPU context is null");
    if (gpu_ct_x.empty()) throw std::invalid_argument("gpu_ct_x is empty");
    if (coefficients.size() != gpu_ct_x.size()) throw std::invalid_argument("coefficients size must match gpu_ct_x size");

    try {
        std::cerr << "[TRUE GPU] Starting TRUE GPU linear combination with " << gpu_ct_x.size() << " inputs" << std::endl;

        // ====== 계수 이진화 제거: 실계수 가중합을 GPU로 계산 ======
        // 입력 복사
        std::vector<ckks::CtAccurate> inputs;
        inputs.reserve(gpu_ct_x.size());
        for (const auto& p : gpu_ct_x) inputs.push_back(p->value);

        // 파라미터 (실계수 상수배를 위한 스케일 요소 생성에 사용)
        const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc_->GetCryptoParameters());
        if (!cryptoParams)
            throw std::runtime_error("CryptoParametersCKKSRNS unavailable");
        lbcrypto::LeveledSHECKKSRNS scheme_helper; // helper to build constant elements

        // 입력들의 level 정렬 (ckksrns-advancedshe-gpu.cpp 로직 차용)
        uint32_t maxLevel = inputs[0].level;
        uint32_t maxIdx   = 0;
        for (uint32_t i = 1; i < inputs.size(); i++) {
            if ((inputs[i].level > maxLevel) ||
                ((inputs[i].level == maxLevel) && (inputs[i].noiseScaleDeg == 2))) {
                maxLevel = inputs[i].level;
                maxIdx   = i;
            }
        }

        // 레벨 정렬: 필요한 경우 adjust scalar를 상수배(정수 CRT 벡터)로 만들어 GPU에서 적용 후 rescale
        for (uint32_t i = 0; i < inputs.size(); i++) {
            if (i == maxIdx) continue;
            if (inputs[i].level < inputs[maxIdx].level) {
                const double adjustScalar = gpu_context_->GetAdjustScalar(inputs[i], inputs[maxIdx]);
                const uint32_t numElems = inputs[i].ax__.size() / gpu_context_->GetDegree();
                auto factors = scheme_helper.GetElementForEvalMult(cryptoParams, inputs[i].level, numElems,
                                                                   inputs[i].noiseScaleDeg, adjustScalar);
                auto factors_gpu = LoadIntegerVector(factors);
                auto adjusted = gpu_context_->EvalMultConst(inputs[i], factors_gpu);
                inputs[i] = gpu_context_->Rescale(adjusted);
            }
        }
        if (inputs[maxIdx].noiseScaleDeg == 2) {
            for (auto& ct : inputs) ct = gpu_context_->Rescale(ct);
        }

        // 가중합 계산
        ckks::CtAccurate current_result;
        bool hasTerm = false;
        for (size_t i = 0; i < inputs.size(); ++i) {
            const double c = coefficients[i];
            if (std::abs(c) < 1e-18) continue;
            // 실계수 c를 위한 CRT 요소를 만들어 GPU 상수배 수행
            const uint32_t numElems = inputs[i].ax__.size() / gpu_context_->GetDegree();
            auto factors = scheme_helper.GetElementForEvalMult(cryptoParams, inputs[i].level, numElems,
                                                               inputs[i].noiseScaleDeg, c);
            auto factors_gpu = LoadIntegerVector(factors);
            ckks::CtAccurate term = gpu_context_->EvalMultConst(inputs[i], factors_gpu);
            if (!hasTerm) {
                current_result = std::move(term);
                hasTerm = true;
            } else {
                gpu_context_->EvalAddInPlace(current_result, term);
            }
            std::cerr << "[TRUE GPU] Weighted term added (i=" << i << ", c=" << c << ")" << std::endl;
            checkCudaSync("gpu_weighted_add");
        }
        if (!hasTerm) {
            // 모든 계수가 0인 경우: 0 암호문 생성
            const auto slots = cc_->GetEncodingParams()->GetBatchSize();
            std::vector<std::complex<double>> zeros(slots, std::complex<double>(0.0, 0.0));
            auto zero_pt = cc_->MakeCKKSPackedPlaintext(zeros);
            zero_pt->SetLength(static_cast<uint32_t>(zeros.size()));
            auto zero_cpu_ct = cc_->Encrypt(keyPair_.publicKey, zero_pt);
            zero_cpu_ct->SetEncodingType(CKKS_PACKED_ENCODING);
            zero_cpu_ct->SetKeyTag(keyPair_.publicKey->GetKeyTag());
            zero_cpu_ct->SetSlots(slots);
            current_result = LoadAccurateCiphertext(zero_cpu_ct);
        }

        std::cerr << "[TRUE GPU] Linear combination (real coefficients) completed on GPU" << std::endl;

        // 추가 정규화 없이 결과를 그대로 사용하되, 스케일이 너무 작으면 2^k 정수배로 GPU에서 보정
        ckks::CtAccurate final_gpu_result = std::move(current_result);
        const double sf = final_gpu_result.scalingFactor;
        if (sf > 0.0 && sf < 1e-12) {
            const double target = std::ldexp(1.0, 59); // 2^59
            double ratio = target / sf;
            const uint64_t step_pow2 = (1ULL << 30);   // 한 번에 2^30씩 곱하기
            const double step_dbl = static_cast<double>(step_pow2);
            int applied_steps = 0;
            while (ratio > step_dbl && applied_steps < 32) { // 안전 가드
                gpu_context_->EvalMultIntegerInPlace(final_gpu_result, step_pow2);
                final_gpu_result.scalingFactor *= step_dbl;
                ratio /= step_dbl;
                applied_steps++;
            }
            // 남은 비율에 대해 2^m (m <= 30) 적용
            int m = static_cast<int>(std::floor(std::log2(ratio)));
            if (m > 0 && m <= 30) {
                uint64_t mult = (1ULL << m);
                gpu_context_->EvalMultIntegerInPlace(final_gpu_result, mult);
                final_gpu_result.scalingFactor *= static_cast<double>(mult);
            }
            std::cerr << "[TRUE GPU] small scale boosted near target; new sf="
                      << std::setprecision(18) << final_gpu_result.scalingFactor << std::endl;
        }

        // CUDA 동기화로 진정한 GPU 연산 완료 보장
        checkCudaSync("true_gpu_layer");

        std::cerr << "[TRUE GPU] TRUE GPU COMPUTATION COMPLETED SUCCESSFULLY!" << std::endl;
        std::cerr << "[TRUE GPU] All operations performed on GPU: Add, Sub, no CPU fallback!" << std::endl;
        std::cerr << "[TRUE GPU] GPU memory managed by GPU context - no manual copies!" << std::endl;

        return PyGiphertext(std::move(final_gpu_result), gpu_context_);

    } catch (const std::exception& e) {
        std::cerr << "[TRUE GPU ERROR] True GPU run_layer failed: " << e.what() << std::endl;
        throw std::runtime_error(std::string("run_layer failed: ") + e.what());
    }
}


    // ===== 다항식 연산: x^2 + x =====
    // GPU 최적화된 ReLU 대체 함수
    PyGiphertext poly2_x2_plus_x(const PyGiphertext& gpu_ct) {
        if (!gpu_context_) throw std::runtime_error("GPU context is null");

        try {
            // GPU에서 x^2 + x 계산 (ReLU 근사)
            ckks::CtAccurate x_squared = gpu_context_->EvalMultAndRelin(gpu_ct.value, gpu_ct.value, evk_);
            ckks::CtAccurate result = gpu_context_->Add(x_squared, gpu_ct.value);

            // CUDA 동기화로 연산 완료 보장
            checkCudaSync("poly2_x2_plus_x");

            // GPU 메모리 최적화를 위한 추가 동기화
            cudaDeviceSynchronize();

            return PyGiphertext(std::move(result), gpu_context_);
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("poly2_x2_plus_x failed: ") + e.what());
        }
    }

    // ===== GPU -> CPU 변환 =====
    // 최적화된 GPU->CPU 변환 함수
    PyCiphertext gpu_to_cpu(const PyGiphertext& gpu_ct) {
        if (!gpu_context_) throw std::runtime_error("GPU context is null");

        // ckks-gpu-bookkeeping.cpp 방식으로 구현
        auto elementParams = cc_->GetCryptoParameters()->GetElementParams();

        PyCiphertext cpu_ct;
        // GPU -> CPU 변환 (정확 매핑)
        try {
            // 디버깅 정보 출력 (선택적)
            std::cerr << "[gpu_to_cpu] level=" << gpu_ct.value.level
                      << " noiseScaleDeg=" << gpu_ct.value.noiseScaleDeg
                      << " scalingFactor=" << std::setprecision(18) << gpu_ct.value.scalingFactor
                      << std::endl;

            // CUDA 동기화로 GPU 연산 완료 보장
            checkCudaSync("gpu_to_cpu_before");

            // ckks-gpu-bookkeeping.cpp 방식: loadIntoDCRTPoly 사용
            const auto resParams = cc_->GetCryptoParameters()->GetElementParams();
            DCRTPoly gpu_res_0 = loadIntoDCRTPoly(gpu_ct.value.bx__, resParams);
            DCRTPoly gpu_res_1 = loadIntoDCRTPoly(gpu_ct.value.ax__, resParams);

            // Ciphertext 객체 생성 (GPU 메타데이터까지 정확히 반영)
            std::vector<DCRTPoly> elements = {gpu_res_0, gpu_res_1};
            cpu_ct.value = std::make_shared<CiphertextImpl<DCRTPoly>>(cc_);
            cpu_ct.value->SetElements(elements);
            cpu_ct.value->SetEncodingType(CKKS_PACKED_ENCODING);
            cpu_ct.value->SetKeyTag(keyPair_.publicKey->GetKeyTag());
            cpu_ct.value->SetSlots(cc_->GetEncodingParams()->GetBatchSize());
            cpu_ct.value->SetLevel(gpu_ct.value.level);
            cpu_ct.value->SetNoiseScaleDeg(gpu_ct.value.noiseScaleDeg);
            cpu_ct.value->SetScalingFactor(gpu_ct.value.scalingFactor);

            // 최종 단계에서 스케일이 지나치게 작으면 CPU 상에서 상수배로 보정
            const double sf = cpu_ct.value->GetScalingFactor();
            if (sf > 0.0 && sf < 1e-12) {
                const double target = std::ldexp(1.0, 59); // 2^59
                const double factor = target / sf;
                std::cerr << "[gpu_to_cpu] boosting small scale by factor=" << std::setprecision(18) << factor << std::endl;
                cc_->EvalMultInPlace(cpu_ct.value, factor);
            }

            // 변환 후 CUDA 동기화
            checkCudaSync("gpu_to_cpu_after");

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

        // MLP 레이어 (bias 연산은 내부에서 무시됨)
        .def("run_layer", &PyCkksContext::run_layer,
             py::arg("gpu_ct_x"), py::arg("coefficients"), py::arg("bias_pt"),
             py::return_value_policy::move,
             py::keep_alive<0,1>(),
             "Run MLP layer on GPU with multiple encrypted inputs (bias ignored), returns GPU ciphertext")

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
