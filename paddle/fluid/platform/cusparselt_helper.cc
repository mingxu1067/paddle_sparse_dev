#include <cuda_runtime_api.h>
#include "paddle/fluid/platform/float16.h"
#include "paddle/fluid/platform/cusparselt_helper.h"

namespace paddle {
namespace platform {
Tensor CompressParameter(const platform::CUDADeviceContext& dev_ctx, Tensor param,
                         int m, int n, int k, int lda, int ldb, int ldc,
                         bool is_col_major) {

    cusparseLtHandle_t cusparselt_handle = dev_ctx.cusparselt_handle();
    cusparseLtMatDescriptor_t      matA, matB, matC;
    cusparseLtMatmulDescriptor_t   matmul;
    cusparseLtMatmulAlgSelection_t alg_sel;
    cusparseLtMatmulPlan_t         plan;

    auto opA   = CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto opB   = CUSPARSE_OPERATION_NON_TRANSPOSE;

    unsigned alignment = 16;
    auto type  = CUDA_R_16F;
    auto compute_type = CUSPARSE_COMPUTE_16F;
    auto order = is_col_major? CUSPARSE_ORDER_COL : CUSPARSE_ORDER_ROW;

    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cusparseLtStructuredDescriptorInit(&cusparselt_handle, &matA, m,
                                                            k, lda, alignment,
                                                            type, order, CUSPARSELT_SPARSITY_50_PERCENT));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cusparseLtDenseDescriptorInit(&cusparselt_handle, &matB, k,
                                                            n, ldb, alignment,
                                                            type, order));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cusparseLtDenseDescriptorInit(&cusparselt_handle, &matC, m,
                                                            n, ldc, alignment,
                                                            type, order));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cusparseLtMatmulDescriptorInit(
                                            &cusparselt_handle, &matmul, opA, opB,
                                            &matA, &matB, &matC, &matC,
                                            compute_type));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cusparseLtMatmulAlgSelectionInit(
                                            &cusparselt_handle, &alg_sel, &matmul,
                                            CUSPARSELT_MATMUL_ALG_DEFAULT));

    size_t workspace_size;
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cusparseLtMatmulGetWorkspace(
                                            &cusparselt_handle, &alg_sel, &workspace_size));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cusparseLtMatmulPlanInit(
                                            &cusparselt_handle, &plan, &matmul,
                                            &alg_sel, workspace_size));

    size_t compressed_size;
    cudaStream_t stream = nullptr;

    PADDLE_ENFORCE_CUDA_SUCCESS(
    platform::dynload::cusparseLtSpMMACompressedSize(
                                            &cusparselt_handle, &plan, &compressed_size));

    auto tmp_allocation_ptr = memory::Alloc(dev_ctx, compressed_size);
    auto& deleter = tmp_allocation_ptr.get_deleter();
    auto* allocation_ptr = tmp_allocation_ptr.release();
    auto shared_allocation = std::shared_ptr<memory::allocation::Allocation>(
                            allocation_ptr, deleter);

    Tensor temp_tensor(framework::proto::VarType::FP16);
    temp_tensor.Resize(param.dims());
    temp_tensor.ResetHolder(std::move(shared_allocation));

    const auto* x_data = param.data<float16>();
    auto* x_data_compressed = temp_tensor.data<float16>();
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cusparseLtSpMMACompress(
                                            &cusparselt_handle, &plan, x_data, x_data_compressed, stream));

    return temp_tensor;
}

}
}