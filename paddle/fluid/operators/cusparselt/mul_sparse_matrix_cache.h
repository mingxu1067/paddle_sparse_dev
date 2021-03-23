#pragma once

#include <cuda_runtime_api.h>
#include <algorithm>
#include <cusparseLt.h>
#include <unordered_map>
#include <string>

namespace paddle {
namespace operators {

template <typename TMatrix>
class SparseMatrixCache {
 public:
  SparseMatrixCache() {
      hash_.clear();
  }
  ~SparseMatrixCache() {
        for (auto& it: hash_) {
            PADDLE_ENFORCE_CUDA_SUCCESS( cudaFree(it.second));
        }
  }

  TMatrix* GetMatrix(const std::string& matrix_name,
                     std::function<TMatrix*()> creat_maxtrix);

 private:
  std::unordered_map<std::string, TMatrix*> hash_;
  std::mutex cache_mutex;
};

template <typename TMatrix>
TMatrix* operators::SparseMatrixCache<TMatrix>::GetMatrix(
                        const std::string& matrix_name,
                        std::function<TMatrix*()> creat_maxtrix) {

  TMatrix* ret;
  auto it = hash_.end();
  bool have_found = false;
  if (matrix_name.length() > 0) {
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        it = hash_.find(matrix_name);

        if (it != hash_.end()) {
          ret = it->second;
          have_found = true;
        }
    }
  }

  if (!have_found) {
    ret = creat_maxtrix();
    std::lock_guard<std::mutex> lock(cache_mutex);
    hash_[matrix_name] = ret;
  }

  return ret;
}

class CompressedMatrixCache {
 public:
  static CompressedMatrixCache& Instance() {
    static CompressedMatrixCache instance;
    return instance;
  }

  SparseMatrixCache<__half>* GetMap() {
    return &cache_;
  }

 private:
  CompressedMatrixCache() {}
  ~CompressedMatrixCache() {}

  SparseMatrixCache<__half> cache_;
};

}  // namespace operators
}  // namespace paddle
