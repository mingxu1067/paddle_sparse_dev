#pragma once

#include <cuda_runtime_api.h>
#include <cusparseLt.h>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class MulSparseKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto& dev_ctx = context.template device_context<platform::CUDADeviceContext>();

    const Tensor* x = context.Input<Tensor>("X");
    const Tensor* y = context.Input<Tensor>("Y");
    Tensor* z = context.Output<Tensor>("Out");
    const Tensor x_matrix =
        x->dims().size() > 2
            ? framework::ReshapeToMatrix(
                  *x, context.template Attr<int>("x_num_col_dims"))
            : *x;
    const Tensor y_matrix =
        y->dims().size() > 2
            ? framework::ReshapeToMatrix(
                  *y, context.template Attr<int>("y_num_col_dims"))
            : *y;

    z->mutable_data<T>(context.GetPlace());
    auto z_dim = z->dims();
    if (z_dim.size() != 2) {
      z->Resize({x_matrix.dims()[0], y_matrix.dims()[1]});
    }

    unsigned alignment = 16;
    auto type  = CUDA_R_16F;
    auto compute_type = CUSPARSE_COMPUTE_16F;
    auto          order = CUSPARSE_ORDER_ROW;
    auto          opA   = CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto          opB   = CUSPARSE_OPERATION_NON_TRANSPOSE;

    cusparseLtHandle_t cusparselt_handle = dev_ctx.cusparselt_handle();
    cusparseLtMatDescriptor_t      matA, matB, matC;
    cusparseLtMatmulDescriptor_t   matmul;
    cusparseLtMatmulAlgSelection_t alg_sel;
    cusparseLtMatmulPlan_t         plan;
    cudaStream_t                   stream = nullptr;

    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cusparseLtStructuredDescriptorInit(&cusparselt_handle, &matA, x_matrix.dims()[0],
                                                            x_matrix.dims()[1], x_matrix.dims()[1], alignment,
                                                            type, order, CUSPARSELT_SPARSITY_50_PERCENT));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cusparseLtDenseDescriptorInit(&cusparselt_handle, &matB, y_matrix.dims()[0],
                                                            y_matrix.dims()[1], y_matrix.dims()[1], alignment,
                                                            type, order));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cusparseLtDenseDescriptorInit(&cusparselt_handle, &matC, z->dims()[0],
                                                            z->dims()[1], z->dims()[1], alignment,
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

    size_t workspace_size, compressed_size;
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cusparseLtMatmulGetWorkspace(
                                            &cusparselt_handle, &alg_sel, &workspace_size));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cusparseLtMatmulPlanInit(
                                            &cusparselt_handle, &plan, &matmul,
                                            &alg_sel, workspace_size));

    // TODO: Do only once ---------------------------------------------------------------
    const T* x_data = x_matrix.data<T>();
    T *dA_pruned, *dA_compressed;
    PADDLE_ENFORCE_CUDA_SUCCESS( cudaMalloc((void**) &dA_pruned, (x_matrix.dims()[0]*x_matrix.dims()[1])) );
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cusparseLtSpMMAPrune(
                                            &cusparselt_handle, &matmul, x_data, dA_pruned,
                                            CUSPARSELT_PRUNE_SPMMA_TILE, stream));
    int is_valid;
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cusparseLtSpMMAPruneCheck(
                                            &cusparselt_handle, &matmul, dA_pruned,
                                            &is_valid, stream));
    if (is_valid != 0) {
        VLOG(0) << "!!!! The matrix has been pruned in a wrong way. " <<
                    "cusparseLtMatmul will not provided correct results\n";
    }

    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cusparseLtSpMMACompressedSize(
                                            &cusparselt_handle, &plan, &compressed_size));
    PADDLE_ENFORCE_CUDA_SUCCESS( cudaMalloc((void**) &dA_compressed, compressed_size));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cusparseLtSpMMACompress(
                                            &cusparselt_handle, &plan, dA_pruned,dA_compressed, stream));
    // --------------------------------------------------------------------------

    void*         d_workspace = nullptr;
    int           num_streams = 0;
    cudaStream_t* streams     = nullptr;
    float alpha = 1.0f;
    float beta  = 0.0f;

    const T* y_data = y_matrix.data<T>();
    T* output_data = z->data<T>();
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cusparseLtMatmul(&cusparselt_handle, &plan, &alpha, dA_compressed, y_data,
                                            &beta, output_data, output_data, d_workspace, streams,
                                            num_streams));

    PADDLE_ENFORCE_CUDA_SUCCESS( platform::dynload::cusparseLtMatmulPlanDestroy(&plan));
    PADDLE_ENFORCE_CUDA_SUCCESS( cudaFree(dA_pruned));
    PADDLE_ENFORCE_CUDA_SUCCESS( cudaFree(dA_compressed));

    if (z_dim.size() != 2) {
      z->Resize(z_dim);
    }
  }
};

}  // namespace operators
}  // namespace paddle