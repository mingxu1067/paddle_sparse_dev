#pragma once

#include <cuda_runtime_api.h>
#include <cusparseLt.h>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DDim = framework::DDim;

template <typename DeviceContext, typename T>
class MulSparseKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto& dev_ctx = context.template device_context<platform::CUDADeviceContext>();

    Tensor x_matrix, y_matrix;
    bool switch_xy = context.Attr<bool>("switch_XY");
    if (switch_xy) {
        const Tensor* x = context.Input<Tensor>("Y");
        const Tensor* y = context.Input<Tensor>("X");

        x_matrix =
            x->dims().size() > 2
                ? framework::ReshapeToMatrix(
                      *x, context.template Attr<int>("y_num_col_dims"))
                : *x;
        y_matrix =
            y->dims().size() > 2
                ? framework::ReshapeToMatrix(
                      *y, context.template Attr<int>("x_num_col_dims"))
                : *y;
    } else {
        const Tensor* x = context.Input<Tensor>("X");
        const Tensor* y = context.Input<Tensor>("Y");

        x_matrix =
            x->dims().size() > 2
                ? framework::ReshapeToMatrix(
                      *x, context.template Attr<int>("x_num_col_dims"))
                : *x;
        y_matrix =
            y->dims().size() > 2
                ? framework::ReshapeToMatrix(
                      *y, context.template Attr<int>("y_num_col_dims"))
                : *y;
    }

    Tensor* z = context.Output<Tensor>("Out");

    unsigned alignment = 16;
    auto type  = CUDA_R_16F;
    auto compute_type = CUSPARSE_COMPUTE_16F;
    bool is_col_major = context.Attr<bool>("is_col_major");
    auto order = is_col_major? CUSPARSE_ORDER_COL : CUSPARSE_ORDER_ROW;

    bool is_transpose_A = context.Attr<bool>("is_transpose_A");
    if (is_transpose_A) x_matrix.Resize({x_matrix.dims()[1], x_matrix.dims()[0]});
    bool is_transpose_B = context.Attr<bool>("is_transpose_B");
    if (is_transpose_B) y_matrix.Resize({y_matrix.dims()[1], y_matrix.dims()[0]});
    auto opA   = CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto opB   = CUSPARSE_OPERATION_NON_TRANSPOSE;

    int m = context.Attr<int>("m");
    int n = context.Attr<int>("n");
    int k = context.Attr<int>("k");
    m = m > 0? m:x_matrix.dims()[0];
    n = n > 0? n:y_matrix.dims()[1];
    k = k > 0? k:x_matrix.dims()[1];

    int lda = context.Attr<int>("lda");
    int ldb = context.Attr<int>("ldb");
    int ldc = context.Attr<int>("ldc");
    lda = lda > 0? lda:(is_col_major? x_matrix.dims()[0]:x_matrix.dims()[1]);
    ldb = ldb > 0? ldb:(is_col_major? y_matrix.dims()[0]:y_matrix.dims()[1]);
    ldc = ldc > 0? ldc:(is_col_major? z->dims()[0]:z->dims()[1]);

    z->mutable_data<T>(context.GetPlace());

    cusparseLtHandle_t cusparselt_handle = dev_ctx.cusparselt_handle();
    cusparseLtMatDescriptor_t      matA, matB, matC;
    cusparseLtMatmulDescriptor_t   matmul;
    cusparseLtMatmulAlgSelection_t alg_sel;
    cusparseLtMatmulPlan_t         plan;

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

    // TODO: Do only once ---------------------------------------------------------------
    const T* x_data = x_matrix.data<T>();
    __half *dA_compressed;
    size_t compressed_size;
    cudaStream_t stream = nullptr;

    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cusparseLtSpMMACompressedSize(
                                            &cusparselt_handle, &plan, &compressed_size));
    PADDLE_ENFORCE_CUDA_SUCCESS( cudaMalloc((void**) &dA_compressed, compressed_size));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cusparseLtSpMMACompress(
                                            &cusparselt_handle, &plan, x_data, dA_compressed, stream));
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
    PADDLE_ENFORCE_CUDA_SUCCESS( cudaFree(dA_compressed));
  }
};

template <typename DeviceContext, typename T>
class MulSparseGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    int x_num_col_dims = ctx.template Attr<int>("x_num_col_dims");
    int y_num_col_dims = ctx.template Attr<int>("y_num_col_dims");
    auto* x = ctx.Input<framework::LoDTensor>("X");
    auto* y = ctx.Input<framework::LoDTensor>("Y");
    auto x_matrix = x->dims().size() > 2
                        ? framework::ReshapeToMatrix(*x, x_num_col_dims)
                        : static_cast<const Tensor&>(*x);
    auto y_matrix = y->dims().size() > 2
                        ? framework::ReshapeToMatrix(*y, y_num_col_dims)
                        : static_cast<const Tensor&>(*y);
    auto* dout = ctx.Input<framework::LoDTensor>(framework::GradVarName("Out"));

    Tensor dout_mat;
    dout_mat.ShareDataWith(*dout);
    dout_mat.Resize({framework::flatten_to_2d(x->dims(), x_num_col_dims)[0],
                     framework::flatten_to_2d(y->dims(), y_num_col_dims)[1]});

    auto* dx = ctx.Output<framework::LoDTensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<framework::LoDTensor>(framework::GradVarName("Y"));

    if (dx != nullptr) {
      dx->set_lod(x->lod());
    }
    if (dy != nullptr) {
      dy->set_lod(y->lod());
    }

    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    auto blas = math::GetBlas<DeviceContext, T>(dev_ctx);
    if (dx) {
      dx->mutable_data<T>(ctx.GetPlace());
      Tensor dx_matrix = dx->dims().size() > 2
                             ? framework::ReshapeToMatrix(*dx, x_num_col_dims)
                             : *dx;

      // dx = dout * y'. dx: M x K, dout : M x N, y : K x N
      blas.MatMul(dout_mat, false, y_matrix, true, &dx_matrix);
    }
    if (dy) {
      dy->mutable_data<T>(ctx.GetPlace());
      Tensor dy_matrix = dy->dims().size() > 2
                             ? framework::ReshapeToMatrix(*dy, y_num_col_dims)
                             : *dy;
      // dy = x' * dout. dy K x N, dout : M x N, x : M x K
      blas.MatMul(x_matrix, true, dout_mat, false, &dy_matrix);
    }
  }
};

}  // namespace operators
}  // namespace paddle
