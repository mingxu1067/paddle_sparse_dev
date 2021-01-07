/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <cusparseLt.h>
#include <cuda.h>
#include <mutex>  // NOLINT
#include <type_traits>

#include "paddle/fluid/platform/dynload/dynamic_loader.h"
#include "paddle/fluid/platform/port.h"

namespace paddle {
namespace platform {
namespace dynload {

extern std::once_flag cusparselt_dso_flag;
extern void *cusparselt_dso_handle;

/**
 * The following macro definition can generate structs
 * (for each function) to dynamic load cublas routine
 * via operator overloading.
 *
 * note: default dynamic linked libs
 */
#define DECLARE_DYNAMIC_LOAD_CUSPARSELT_WRAP(__name)                                    \
  struct DynLoad__##__name {                                                            \
    template <typename... Args>                                                         \
    inline auto operator()(Args... args) -> DECLARE_TYPE(__name, args...) {             \
      using cusparselt_func =                                                           \
          decltype(::__name(std::declval<Args>()...)) (*)(Args...);                     \
      std::call_once(cusparselt_dso_flag, []() {                                        \
        cusparselt_dso_handle = paddle::platform::dynload::GetCusparseltDsoHandle();    \
      });                                                                               \
      static void *p_##__name = dlsym(cusparselt_dso_handle, #__name);                  \
      return reinterpret_cast<cusparselt_func>(p_##__name)(args...);                    \
    }                                                                                   \
  };                                                                                    \
  extern DynLoad__##__name __name

// APIs available after CUDA 8.0
#if CUDA_VERSION >= 1100
#define CUSPARSELT_ROUTINE_EACH(__macro)        \
  __macro(cusparseLtInit);                      \
  __macro(cusparseLtStructuredDescriptorInit);  \
  __macro(cusparseLtDenseDescriptorInit);       \
  __macro(cusparseLtMatmulDescriptorInit);      \
  __macro(cusparseLtMatmulAlgSelectionInit);    \
  __macro(cusparseLtMatmulGetWorkspace);        \
  __macro(cusparseLtMatmulPlanInit);            \
  __macro(cusparseLtSpMMAPrune);                \
  __macro(cusparseLtSpMMAPruneCheck);           \
  __macro(cusparseLtSpMMACompressedSize);       \
  __macro(cusparseLtSpMMACompress);             \
  __macro(cusparseLtMatmulAlgGetAttribute);     \
  __macro(cusparseLtMatmulAlgSetAttribute);     \
  __macro(cusparseLtMatmulSearch);              \
  __macro(cusparseLtMatmul);                    \
  __macro(cusparseLtMatmulPlanDestroy);         \
  __macro(cusparseLtDestroy);

CUSPARSELT_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_CUSPARSELT_WRAP)
#endif

#undef DECLARE_DYNAMIC_LOAD_CUSPARSELT_WRAP
}  // namespace dynload
}  // namespace platform
}  // namespace paddle
