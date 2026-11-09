#include <torch/extension.h>
#include <cmath>

void update(torch::Tensor parameter, torch::Tensor gradient,
            torch::Tensor momentum, torch::Tensor variance,
            double lr, double beta1, double beta2, double eps,
            double decay, int64_t step, int threads) {
    TORCH_CHECK(parameter.device().is_cpu() && gradient.device().is_cpu(),
                "CPU Adam requires CPU tensors");
    TORCH_CHECK(parameter.scalar_type() == torch::kFloat32 &&
                gradient.scalar_type() == torch::kFloat32 &&
                momentum.scalar_type() == torch::kFloat32 &&
                variance.scalar_type() == torch::kFloat32,
                "CPU Adam requires FP32 parameters, gradients and states");
    TORCH_CHECK(parameter.is_contiguous() && gradient.is_contiguous() &&
                momentum.is_contiguous() && variance.is_contiguous(),
                "CPU Adam requires contiguous tensors");
    const int64_t size = parameter.numel();
    TORCH_CHECK(gradient.numel() == size && momentum.numel() == size &&
                variance.numel() == size, "CPU Adam tensor sizes differ");
    float* p = parameter.data_ptr<float>();
    const float* g = gradient.data_ptr<float>();
    float* m = momentum.data_ptr<float>();
    float* v = variance.data_ptr<float>();
    const float b1 = beta1, b2 = beta2;
    const float one_b1 = 1.0-beta1, one_b2 = 1.0-beta2;
    const float step_size = lr/(1.0-std::pow(beta1,step));
    const float denominator_scale = std::sqrt(1.0-std::pow(beta2,step));
    const float weight_scale = 1.0-lr*decay, epsilon = eps;
    #pragma omp parallel for simd num_threads(threads)
    for (int64_t i=0; i<size; ++i) {
        m[i] = b1*m[i]+one_b1*g[i];
        v[i] = b2*v[i]+one_b2*g[i]*g[i];
        p[i] = weight_scale*p[i]-step_size*m[i]/(std::sqrt(v[i])/denominator_scale+epsilon);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("update", &update, pybind11::call_guard<pybind11::gil_scoped_release>());
}
