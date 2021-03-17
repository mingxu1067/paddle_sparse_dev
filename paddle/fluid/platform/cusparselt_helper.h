#pragma once

#include <cusparseLt.h>
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/framework.pb.h"


namespace paddle {
namespace platform {

using Tensor = framework::Tensor;
using DDim = framework::DDim;

Tensor CompressParameter(const platform::CUDADeviceContext& dev_ctx, Tensor param,
                         int m, int n, int k, int lda, int ldb, int ldc,
                         bool is_col_major);

}
}