// cuBLASLt plans with explicit algorithm verification. FP16/BF16 operands, FP32 output.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>
#include <map>
#include <memory>
#include <vector>
using at::Tensor;
static void check(cublasStatus_t status) {
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cuBLASLt status ", int(status));
}

struct Layouts {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    int64_t m, k, n, sk, sn;
    int device;
    at::ScalarType dtype;
    Layouts(const Tensor &x, const Tensor &w) {
        TORCH_CHECK(x.is_cuda() && w.device() == x.device() && x.dim() == 2 && w.dim() == 2 &&
                        x.is_contiguous(),
                    "invalid operands");
        dtype = x.scalar_type();
        TORCH_CHECK(w.scalar_type() == dtype && (dtype == at::kHalf || dtype == at::kBFloat16),
                    "expected matching FP16/BF16 operands");
        m = x.size(0);
        k = x.size(1);
        n = w.size(1);
        sk = w.stride(0);
        sn = w.stride(1);
        device = x.get_device();
        TORCH_CHECK(m > 0 && n > 0 && k > 0 && w.size(0) == k, "invalid matrix sizes");
        TORCH_CHECK((sk == 1 && sn == k) || (sn == 1 && sk == n),
                    "expected dense row/column-major weights");
    }
    void initialize() {
        cudaDataType_t type = dtype == at::kHalf ? CUDA_R_16F : CUDA_R_16BF;
        check(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        cublasOperation_t trans = sk == 1 ? CUBLAS_OP_T : CUBLAS_OP_N;
        check(
            cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)));
        check(cublasLtMatrixLayoutCreate(&a, type, sk == 1 ? k : n, sk == 1 ? n : k,
                                         sk == 1 ? k : n));
        check(cublasLtMatrixLayoutCreate(&b, type, k, m, k));
        check(cublasLtMatrixLayoutCreate(&c, CUDA_R_32F, n, m, n));
    }
    ~Layouts() {
        if (c)
            cublasLtMatrixLayoutDestroy(c);
        if (b)
            cublasLtMatrixLayoutDestroy(b);
        if (a)
            cublasLtMatrixLayoutDestroy(a);
        if (op)
            cublasLtMatmulDescDestroy(op);
    }
};
struct Preference {
    cublasLtMatmulPreference_t value = nullptr;
    ~Preference() {
        if (value)
            cublasLtMatmulPreferenceDestroy(value);
    }
};
struct Plan {
    std::shared_ptr<Layouts> layout;
    cublasLtMatmulAlgo_t algorithm;
    size_t workspace;
    Plan(std::shared_ptr<Layouts> l, const cublasLtMatmulHeuristicResult_t &result)
        : layout(l), algorithm(result.algo), workspace(result.workspaceSize) {}
    Tensor run(const Tensor &x, const Tensor &w) {
        const auto &l = *layout;
        TORCH_CHECK(x.is_cuda() && x.get_device() == l.device && w.device() == x.device() &&
                        x.is_contiguous(),
                    "device/layout mismatch");
        TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && x.size(0) == l.m && x.size(1) == l.k &&
                        w.size(0) == l.k && w.size(1) == l.n && w.stride(0) == l.sk &&
                        w.stride(1) == l.sn,
                    "plan shape mismatch");
        TORCH_CHECK(x.scalar_type() == l.dtype && w.scalar_type() == l.dtype,
                    "plan dtype mismatch");
        c10::cuda::CUDAGuard guard(x.device());
        auto out = at::empty({l.m, l.n}, x.options().dtype(at::kFloat));
        auto scratch = at::empty({int64_t(workspace)}, x.options().dtype(at::kByte));
        const float alpha = 1.f, beta = 0.f;
        check(cublasLtMatmul(at::cuda::getCurrentCUDABlasLtHandle(), l.op, &alpha, w.data_ptr(),
                             l.a, x.data_ptr(), l.b, &beta, out.data_ptr(), l.c, out.data_ptr(),
                             l.c, &algorithm, scratch.data_ptr(), workspace,
                             c10::cuda::getCurrentCUDAStream()));
        return out;
    }
    std::shared_ptr<Plan> retarget(const Tensor &x, const Tensor &w) {
        auto next = std::make_shared<Layouts>(x, w);
        const auto &l = *layout;
        TORCH_CHECK(next->device == l.device && next->dtype == l.dtype && next->k == l.k &&
                        next->n == l.n && next->sk == l.sk && next->sn == l.sn,
                    "retarget may change only row count");
        c10::cuda::CUDAGuard guard(x.device());
        next->initialize();
        cublasLtMatmulHeuristicResult_t checked{};
        check(cublasLtMatmulAlgoCheck(at::cuda::getCurrentCUDABlasLtHandle(), next->op, next->a,
                                      next->b, next->c, next->c, &algorithm, &checked));
        check(checked.state);
        checked.algo = algorithm;
        TORCH_CHECK(checked.workspaceSize <= 256 * 1024 * 1024,
                    "retarget workspace exceeds 256 MiB");
        return std::make_shared<Plan>(next, checked);
    }
    std::shared_ptr<Plan> configure(int splits, int reduction, int tile, int stages) {
        const auto &l = *layout;
        c10::cuda::CUDAGuard guard(l.device);
        auto algo = algorithm;
        for (auto item : std::vector<std::pair<cublasLtMatmulAlgoConfigAttributes_t, int>>{
                 {CUBLASLT_ALGO_CONFIG_SPLITK_NUM, splits},
                 {CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, reduction},
                 {CUBLASLT_ALGO_CONFIG_TILE_ID, tile},
                 {CUBLASLT_ALGO_CONFIG_STAGES_ID, stages}})
            check(cublasLtMatmulAlgoConfigSetAttribute(&algo, item.first, &item.second, sizeof(int)));
        cublasLtMatmulHeuristicResult_t checked{};
        check(cublasLtMatmulAlgoCheck(at::cuda::getCurrentCUDABlasLtHandle(), l.op, l.a, l.b,
                                      l.c, l.c, &algo, &checked));
        check(checked.state);
        TORCH_CHECK(checked.workspaceSize <= 32 * 1024 * 1024, "configured workspace exceeds 32 MiB");
        checked.algo = algo;
        return std::make_shared<Plan>(layout, checked);
    }
    std::map<std::string, int64_t> info() {
        std::map<std::string, int64_t> result{{"workspace_bytes", int64_t(workspace)}};
        for (auto item : std::vector<std::pair<std::string, cublasLtMatmulAlgoConfigAttributes_t>>{
                 {"id", CUBLASLT_ALGO_CONFIG_ID},
                 {"tile", CUBLASLT_ALGO_CONFIG_TILE_ID},
                 {"split_k", CUBLASLT_ALGO_CONFIG_SPLITK_NUM},
                 {"reduction", CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME},
                 {"stages", CUBLASLT_ALGO_CONFIG_STAGES_ID}}) {
            int value = -1;
            size_t written = 0;
            if (cublasLtMatmulAlgoConfigGetAttribute(&algorithm, item.second, &value, sizeof(value),
                                                     &written) == CUBLAS_STATUS_SUCCESS)
                result[item.first] = value;
        }
        return result;
    }
};
std::vector<std::shared_ptr<Plan>> candidates(Tensor x, Tensor w, int64_t workspace, int count) {
    TORCH_CHECK(workspace >= 0 && count > 0 && count <= 256, "invalid search bounds");
    TORCH_CHECK(x.is_cuda(), "CUDA operands required");
    c10::cuda::CUDAGuard guard(x.device());
    auto layout = std::make_shared<Layouts>(x, w);
    layout->initialize();
    Preference pref;
    check(cublasLtMatmulPreferenceCreate(&pref.value));
    const size_t limit = workspace;
    check(cublasLtMatmulPreferenceSetAttribute(pref.value, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                               &limit, sizeof(limit)));
    std::vector<cublasLtMatmulHeuristicResult_t> found(count);
    int actual = 0;
    check(cublasLtMatmulAlgoGetHeuristic(at::cuda::getCurrentCUDABlasLtHandle(), layout->op,
                                         layout->a, layout->b, layout->c, layout->c, pref.value,
                                         count, found.data(), &actual));
    std::vector<std::shared_ptr<Plan>> result;
    for (int i = 0; i < actual; ++i)
        if (found[i].state == CUBLAS_STATUS_SUCCESS)
            result.push_back(std::make_shared<Plan>(layout, found[i]));
    return result;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<Plan, std::shared_ptr<Plan>>(m, "Plan")
        .def("run", &Plan::run)
        .def("info", &Plan::info)
        .def("configure", &Plan::configure)
        .def("retarget", &Plan::retarget);
    m.def("candidates", &candidates);
}
