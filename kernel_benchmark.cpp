#include <immintrin.h>
#include <sys/time.h>
#include "tsimd/tsimd.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <thread>
#include <future>

using namespace std;
using vfloat_uptr = unique_ptr<float[], void(*)(void*)>;
constexpr int k_num_thread = 4;

struct Timer {
  struct timespec s, e;
  long accumulated = 0;
  void reset(void) {
    accumulated = 0;
  }
  void start(void) {
    clock_gettime(CLOCK_MONOTONIC, &s);
  }
  void stop(void) {
    clock_gettime(CLOCK_MONOTONIC, &e);
  }
  void accum(void) {
    accumulated += eval();
  }
  long eval(void) {
   return 1000000000 * (e.tv_sec - s.tv_sec) + (e.tv_nsec - s.tv_nsec);
  }
};

void* zmalloc_aligned(size_t size) {
  void* ptr;
  posix_memalign(&ptr, 64, size);
  return ptr;
}

// C - Column-based orientation / Dimension first
// R - Row-based orientation / Row first

vector<float*> createDataSetC(const size_t size, const size_t dim, const double min, const double max) {
  random_device rd;
  mt19937 gen(rd());
  uniform_real_distribution<> dis(min, max);
  vector<float*> ret(dim);
  for (auto& ptr : ret) {
    ptr = (float*) zmalloc_aligned(sizeof(float) * size);
  }
  for (int d = 0; d < dim; d++)
    for (int i = 0; i < size; i++)
      ret[d][i] = dis(gen);
  return ret;
}

vector<float*> createDataSetR(const size_t size, const size_t dim, const double min, const double max) {
  random_device rd;
  mt19937 gen(rd());
  uniform_real_distribution<> dis(min, max);
  vector<float*> ret(size);
  for (auto& ptr : ret) {
    ptr = (float*) zmalloc_aligned(sizeof(float) * dim);
  }
  for (int i = 0; i < size; i++)
    for (int d = 0; d < dim; d++)
      ret[i][d] = dis(gen);
  return ret;
}

vector<float*> loadDataSetC(const size_t size, const size_t dim, const string& path, const size_t offset) {
  vector<float*> ret(dim);
  for (auto& ptr : ret) {
    ptr = (float*) zmalloc_aligned(sizeof(float) * size);
  }
  ifstream ifs(path);
  if (!ifs) {
    cerr << "The file(" << path << ") does not exisit." << endl;
    return {};
  }
  long r = 0;
  string line, buf;
  while (ifs.good()) {
    if (!getline(ifs, line)) break;
    if (r >= offset && r-offset >= size) break;
    if (r < offset) {
      r++;
      continue;
    }
    stringstream ss(line);
    int i = 0;
    while (getline(ss, buf, ',')) {
      ret[i++][r-offset] = stof(buf);
    }
    r++;
  }
  return ret;
}

vector<float*> loadDataSetR(const size_t size, const size_t dim, const string& path, const size_t offset) {
  vector<float*> ret(size);
  for (auto& ptr : ret) {
    ptr = (float*) zmalloc_aligned(sizeof(float) * dim);
  }
  ifstream ifs(path);
  if (!ifs) {
    cerr << "The file(" << path << ") does not exist." << endl;
    return {};
  }
  long r = 0;
  string line, buf;
  while (ifs.good()) {
    if (!getline(ifs, line)) break;
    if (r >= offset && r-offset >= size) break;
    if (r < offset) {
      r++;
      continue;
    }
    stringstream ss(line);
    int i = 0;
    while (getline(ss, buf, ',')) {
      ret[r-offset][i++] = stof(buf);
    }
    r++;
  }
  return ret;
}

vfloat_uptr calcMagnitudeC (const vector<float*>& dataset, const size_t size, const size_t dim) {
  vfloat_uptr magnitude{(float*) zmalloc_aligned(sizeof(float) * size), free};
  fill(magnitude.get(), magnitude.get() + size, 0);

  for (int d = 0; d < dim; ++d) {
    for (int r = 0; r < size; ++r) {
      auto v = dataset[d][r];
      magnitude[r] += v * v;
    }
  }

  for (int r = 0; r < size; ++r) {
    magnitude[r] = sqrt(magnitude[r]);
  }

  return magnitude;
}

vfloat_uptr calcMagnitudeC (const float* dataset, const size_t size, const size_t dim) {
  vfloat_uptr magnitude{(float*) zmalloc_aligned(sizeof(float) * size), free};
  fill(magnitude.get(), magnitude.get() + size, 0);

  for (int d = 0; d < dim; ++d) {
    for (int r = 0; r < size; ++r) {
      auto v = dataset[d*size + r];
      magnitude[r] += v * v;
    }
  }

  for (int r = 0; r < size; ++r) {
    magnitude[r] = sqrt(magnitude[r]);
  }

  return magnitude;
}

vfloat_uptr calcMagnitudeR (const vector<float*>& dataset, const size_t size, const size_t dim) {
  vfloat_uptr magnitude{(float*) zmalloc_aligned(sizeof(float) * size), free};
  fill(magnitude.get(), magnitude.get() + size, 0);

  for (int r = 0; r < size; ++r) {
    for (int d = 0; d < dim; ++d) {
      auto v = dataset[r][d];
      magnitude[r] += v * v;
    }
  }

  for (int r = 0; r < size; ++r) {
    magnitude[r] = sqrt(magnitude[r]);
  }

  return magnitude;
}

vfloat_uptr calcMagnitudeRWithThread (const vector<float*>& dataset, const size_t size, const size_t dim, const size_t num_thread) {
  if (size < num_thread) return calcMagnitudeR(dataset, size,dim);
  vfloat_uptr magnitude{(float*) zmalloc_aligned(sizeof(float) * size), free};
  fill(magnitude.get(), magnitude.get() + size, 0);

  vector<thread> threads;
  auto calc = [&](int from, int to) {
    for (int r = from; r < to; ++r) {
      for (int d = 0; d < dim; ++d) {
        auto v = dataset[r][d];
        magnitude[r] += v * v;
      }
      magnitude[r] = sqrt(magnitude[r]);
    }
  };

  for (int i = 1; i < num_thread; ++i) {
    int from, to;
    from = size / num_thread * i;
    if (i != num_thread-1) to = size / num_thread * (i + 1);
    else to = size;
    threads.emplace_back(calc, from, to);
  }
  calc(0, size / num_thread);
  for (auto& thread : threads) thread.join();

  return magnitude;
}

vfloat_uptr calcMagnitudeR (const float* dataset, const size_t size, const size_t dim) {
  vfloat_uptr magnitude{(float*) zmalloc_aligned(sizeof(float) * size), free};
  fill(magnitude.get(), magnitude.get() + size, 0);

  for (int r = 0; r < size; ++r) {
    for (int d = 0; d < dim; ++d) {
      auto v = dataset[r * dim + d];
      magnitude[r] += v * v;
    }
  }

  for (int r = 0; r < size; ++r) {
    magnitude[r] = sqrt(magnitude[r]);
  }

  return magnitude;
}

vfloat_uptr calcMagnitudeRWithThread (const float* dataset, const size_t size, const size_t dim, const size_t num_thread) {
  if (size < num_thread) return calcMagnitudeR(dataset, size, dim);
  vfloat_uptr magnitude{(float*) zmalloc_aligned(sizeof(float) * size), free};
  fill(magnitude.get(), magnitude.get() + size, 0);

  vector<thread> threads;
  auto calc = [&](int from, int to) {
    for (int r = from; r < to; ++r) {
      for (int d = 0; d < dim; ++d) {
        auto v = dataset[r * dim + d];
        magnitude[r] += v * v;
      }
      magnitude[r] = sqrt(magnitude[r]);
    }
  };

  for (int i = 1; i < num_thread; ++i) {
    int from, to;
    from = size / num_thread * i;
    if (i != num_thread-1) to = size / num_thread * (i + 1);
    else to = size;
    threads.emplace_back(calc, from, to);
  }
  calc(0, size / num_thread);
  for (auto& thread : threads) thread.join();

  return magnitude;
}

vfloat_uptr calcMagnitudeCWithTsimd(const vector<float*>& dataset, const size_t size, const size_t dim) {
  using namespace tsimd;
  const size_t simd_unit = vfloat::static_size;

  if (size < simd_unit) return calcMagnitudeC(dataset, size, dim);
  const size_t simd_end = size - (size % simd_unit);

  vfloat_uptr magnitude{(float*) zmalloc_aligned(sizeof(float) * size), free};
  fill(magnitude.get(), magnitude.get() + size, 0);

  for (int d = 0; d < dim; ++d) {
    for (int r = 0; r < simd_end; r += simd_unit) {
      auto src = load<vfloat>(&dataset[d][r]);
      auto dst = load<vfloat>(&magnitude[r]) + (src * src);
      store(dst, &magnitude[r]);
    }
    for (int r = simd_end; r < size; ++r) {
      auto v = dataset[d][r];
      magnitude[r] += v * v;
    }
  }

  for (int r = 0; r < simd_end; r += simd_unit) {
    auto v = sqrt(load<vfloat>(&magnitude[r]));
    store(v, &magnitude[r]);
  }
  for (int r = simd_end; r < size; ++r) {
    magnitude[r] = sqrt(magnitude[r]);
  }

  return magnitude;
}

vfloat_uptr calcMagnitudeCWithTsimd(const float* dataset, const size_t size, const size_t dim) {
  using namespace tsimd;
  const size_t simd_unit = vfloat::static_size;

  if (size < simd_unit) return calcMagnitudeC(dataset, size, dim);
  const size_t simd_end = size - (size % simd_unit);

  vfloat_uptr magnitude{(float*) zmalloc_aligned(sizeof(float) * size), free};
  fill(magnitude.get(), magnitude.get() + size, 0);

  for (int d = 0; d < dim; ++d) {
    for (int r = 0; r < simd_end; r += simd_unit) {
      auto src = load<vfloat>(&dataset[d * size + r]);
      auto dst = load<vfloat>(&magnitude[r]) + (src * src);
      store(dst, &magnitude[r]);
    }
    for (int r = simd_end; r < size; ++r) {
      auto v = dataset[d * size + r];
      magnitude[r] += v * v;
    }
  }

  for (int r = 0; r < simd_end; r += simd_unit) {
    auto v = sqrt(load<vfloat>(&magnitude[r]));
    store(v, &magnitude[r]);
  }
  for (int r = simd_end; r < size; ++r) {
    magnitude[r] = sqrt(magnitude[r]);
  }

  return magnitude;
}

vfloat_uptr calcMagnitudeRWithTsimd(const vector<float*>& dataset, const size_t size, const size_t dim) {
  using namespace tsimd;
  const size_t simd_unit = vfloat::static_size;

  if (dim < simd_unit) return calcMagnitudeR(dataset, size, dim);
  const size_t dim_simd_end = dim - (dim % simd_unit);
  const size_t size_simd_end = size - (size % simd_unit);

  vfloat_uptr magnitude{(float*) zmalloc_aligned(sizeof(float) * size), free};
  fill(magnitude.get(), magnitude.get() + size, 0);

  for (int r = 0; r < size; ++r) {
    vfloat tmp(0);
    for (int d = 0; d < dim_simd_end; d += simd_unit) {
      auto v = load<vfloat>(&dataset[r][d]);
      tmp += v * v;
    }
    for (int i = 0; i < simd_unit; ++i) {
      magnitude[r] += tmp[i];
    }
    for (int d = dim_simd_end; d < dim; ++d) {
      auto v = dataset[r][d];
      magnitude[r] += v * v;
    }
  }

  for (int r = 0; r < size_simd_end; r += simd_unit) {
    auto v = sqrt(load<vfloat>(&magnitude[r]));
    store(v, &magnitude[r]);
  }
  for (int r = size_simd_end; r < size; ++r) {
    magnitude[r] = sqrt(magnitude[r]);
  }

  return magnitude;
}

vfloat_uptr calcMagnitudeRWithThreadAndTsimd(const vector<float*>& dataset, const size_t size, const size_t dim, const size_t num_thread) {
  using namespace tsimd;
  const size_t simd_unit = vfloat::static_size;

  if (dim < simd_unit) return calcMagnitudeRWithThread(dataset, size, dim, num_thread);
  if (size < num_thread) return calcMagnitudeRWithTsimd(dataset, size, dim);
  const size_t dim_simd_end = dim - (dim % simd_unit);
  const size_t size_simd_end = size - (size % simd_unit);

  vfloat_uptr magnitude{(float*) zmalloc_aligned(sizeof(float) * size), free};
  fill(magnitude.get(), magnitude.get() + size, 0);

  vector<thread> threads;
  auto calc = [&](int from, int to) {
    for (int r = from; r < to; ++r) {
      vfloat tmp(0);
      for (int d = 0; d < dim_simd_end; d += simd_unit) {
        auto v = load<vfloat>(&dataset[r][d]);
        tmp += v * v;
      }
      for (int i = 0; i < simd_unit; ++i) {
        magnitude[r] += tmp[i];
      }
      for (int d = dim_simd_end; d < dim; ++d) {
        auto v = dataset[r][d];
        magnitude[r] += v * v;
      }
    }
  };

  for (int i = 1; i < num_thread; ++i) {
    int from, to;
    from = size / num_thread * i;
    if (i != num_thread-1) to = size / num_thread * (i + 1);
    else to = size;
    threads.emplace_back(calc, from, to);
  }
  calc(0, size / num_thread);
  for (auto& thread : threads) thread.join();

  for (int r = 0; r < size_simd_end; r += simd_unit) {
    auto v = sqrt(load<vfloat>(&magnitude[r]));
    store(v, &magnitude[r]);
  }
  for (int r = size_simd_end; r < size; ++r) {
    magnitude[r] = sqrt(magnitude[r]);
  }

  return magnitude;
}

vfloat_uptr calcMagnitudeRWithTsimd(const float* dataset, const size_t size, const size_t dim) {
  using namespace tsimd;
  const size_t simd_unit = vfloat::static_size;

  if (dim < simd_unit || dim % simd_unit != 0) return calcMagnitudeR(dataset, size, dim);

  vfloat_uptr magnitude{(float*) zmalloc_aligned(sizeof(float) * size), free};
  fill(magnitude.get(), magnitude.get() + size, 0);

  for (int r = 0; r < size; ++r) {
    vfloat tmp(0);
    for (int d = 0; d < dim; d += simd_unit) {
      auto v = load<vfloat>(&dataset[r * dim + d]);
      tmp += v * v;
    }
    for (int i = 0; i < simd_unit; ++i) {
      magnitude[r] += tmp[i];
    }
  }

  const size_t simd_end = size - (size % simd_unit);
  for (int r = 0; r < simd_end; r += simd_unit) {
    auto v = sqrt(load<vfloat>(&magnitude[r]));
    store(v, &magnitude[r]);
  }
  for (int r = simd_end; r < size; ++r) {
    magnitude[r] = sqrt(magnitude[r]);
  }

  return magnitude;
}

vfloat_uptr calcMagnitudeRWithThreadAndTsimd(const float* dataset, const size_t size, const size_t dim, const size_t num_thread) {
  using namespace tsimd;
  const size_t simd_unit = vfloat::static_size;

  if (dim < simd_unit || dim % simd_unit != 0) return calcMagnitudeRWithThread(dataset, size, dim, num_thread);
  if (size < num_thread) return calcMagnitudeRWithTsimd(dataset, size, dim);

  vfloat_uptr magnitude{(float*) zmalloc_aligned(sizeof(float) * size), free};
  fill(magnitude.get(), magnitude.get() + size, 0);

  vector<thread> threads;
  auto calc = [&](int from, int to) {
    for (int r = from; r < to; ++r) {
      vfloat tmp(0);
      for (int d = 0; d < dim; d += simd_unit) {
        auto v = load<vfloat>(&dataset[r * dim + d]);
        tmp += v * v;
      }
      for (int i = 0; i < simd_unit; ++i) {
        magnitude[r] += tmp[i];
      }
    }
  };

  for (int i = 1; i < num_thread; ++i) {
    int from, to;
    from = size / num_thread * i;
    if (i != num_thread-1) to = size / num_thread * (i + 1);
    else to = size;
    threads.emplace_back(calc, from, to);
  }
  calc(0, size / num_thread);
  for (auto& thread : threads) thread.join();

  const size_t simd_end = size - (size % simd_unit);
  for (int r = 0; r < simd_end; r += simd_unit) {
    auto v = sqrt(load<vfloat>(&magnitude[r]));
    store(v, &magnitude[r]);
  }
  for (int r = simd_end; r < size; ++r) {
    magnitude[r] = sqrt(magnitude[r]);
  }

  return magnitude;
}

vfloat_uptr calcEucDistC (const vector<float*>& dataset, const size_t size, const size_t dim, const float* query) {
  vfloat_uptr dist{(float*) zmalloc_aligned(sizeof(float) * size), free};
  fill(dist.get(), dist.get() + size, 0);

  for (int d = 0; d < dim; d++) {
    float q = query[d];
    for (int r = 0; r < size; r++) {
      float v = dataset[d][r] - q;
      dist[r] += v * v;
    }
  }

  for (int r = 0; r < size; r++) {
    dist[r] = sqrt(dist[r]);
  }

  return dist;
}

vfloat_uptr calcCosDistC (const vector<float*>& dataset, const size_t size, const size_t dim, const float* query) {
  vfloat_uptr dist{(float*) zmalloc_aligned(sizeof(float) * size), free};
  fill(dist.get(), dist.get() + size, 0);
  vector<float> mag_d(size);

  for (int d = 0; d < dim; d++) {
    float q = query[d];
    for (int r = 0; r < size; r++) {
      float v = dataset[d][r];
      dist[r] += v * q;
      mag_d[r] += v * v;
    }
  }

  float mag_q = 0;
  for (int d = 0; d < dim; d++) {
    mag_q += query[d] * query[d];
  }
  mag_q = sqrt(mag_q);

  for (int r = 0; r < size; r++) {
    dist[r] /= sqrt(mag_d[r]) * mag_q;
  }

  return dist;
}

vfloat_uptr calcEucDistR (const vector<float*>& dataset, const size_t size, const size_t dim, const float* query) {
  vfloat_uptr dist{(float*) zmalloc_aligned(sizeof(float) * size), free};
  fill(dist.get(), dist.get() + size, 0);

  for (int r = 0; r < size; ++r) {
    for (int d = 0; d < dim; ++d) {
      float v = dataset[r][d] - query[d];
      dist[r] += v * v;
    }
  }

  for (int r = 0; r < size; ++r) {
    dist[r] = sqrt(dist[r]);
  }
  return dist;
}

vfloat_uptr calcEucDistRWithThread (const vector<float*>& dataset, const size_t size, const size_t dim, const float* query, const size_t num_thread) {
  if (size < num_thread) return calcEucDistR(dataset, size, dim, query);
  vfloat_uptr dist{(float*) zmalloc_aligned(sizeof(float) * size), free};
  fill(dist.get(), dist.get() + size, 0);

  vector<thread> threads;
  auto calc = [&](int from, int to) {
    for (int r = from; r < to; ++r) {
      for (int d = 0; d < dim; ++d) {
        float v = dataset[r][d] - query[d];
        dist[r] += v * v;
      }
      dist[r] = sqrt(dist[r]);
    }
  };

  for (int i = 1; i < num_thread; ++i) {
    int from, to;
    from = (size / num_thread) * i ;
    if (i != num_thread-1) to = (size / num_thread) * (i + 1);
    else to = size;
    threads.emplace_back(calc, from, to);
  }
  calc(0, size / num_thread);
  for (auto& thread : threads) thread.join();

  return dist;
}

vfloat_uptr calcCosDistR (const vector<float*>& dataset, const size_t size, const size_t dim, const float* query) {
  vfloat_uptr dist{(float*) zmalloc_aligned(sizeof(float) * size), free};
  fill(dist.get(), dist.get() + size, 0);
  float mag_q = 0;
  for (int d = 0; d < dim; d++) {
    mag_q += query[d] * query[d];
  }
  mag_q = sqrt(mag_q);

  float mag_d;
  for (int r = 0; r < size; r++) {
    mag_d = 0.0f;
    for (int d = 0; d < dim; d++) {
      float v = dataset[r][d];
      dist[r] += v * query[d];
      mag_d += v * v;
    }
    dist[r] /= sqrt(mag_d) * mag_q;
  }

  return dist;
}

vfloat_uptr calcCosDistRWithThread (const vector<float*>& dataset, const size_t size, const size_t dim, const float* query, const size_t num_thread) {
  if (size < num_thread) calcCosDistR(dataset, size, dim, query);
  vfloat_uptr dist{(float*) zmalloc_aligned(sizeof(float) * size), free};
  fill(dist.get(), dist.get() + size, 0);
  float mag_q = 0;
  for (int d = 0; d < dim; d++) {
    mag_q += query[d] * query[d];
  }
  mag_q = sqrt(mag_q);

  vector<thread> threads;
  auto calc = [&](int from, int to) {
    for (int r = from; r < to; r++) {
      float mag_d = 0.0f;
      for (int d = 0; d < dim; d++) {
        float v = dataset[r][d];
        dist[r] += v * query[d];
        mag_d += v * v;
      }
      dist[r] /= sqrt(mag_d) * mag_q;
    }
  };

  for (int i = 1; i < num_thread; ++i) {
    int from, to;
    from = (size / num_thread) * i;
    if (i != num_thread-1) to = (size / num_thread) * (i + 1);
    else to = size;
    threads.emplace_back(calc, from, to);
  }
  calc(0, size / num_thread);
  for (auto& thread : threads) thread.join();

  return dist;
}

vfloat_uptr calcEucDistCWithTsimd(const vector<float*>& dataset, const size_t size, const size_t dim, const float* query) {
  using namespace tsimd;
  const size_t simd_unit = vfloat::static_size;

  if (size < simd_unit) return calcEucDistC(dataset, size, dim, query);

  vfloat_uptr dist((float*) zmalloc_aligned(sizeof(float)*size), free);
  fill(dist.get(), dist.get() + size, 0);

  const size_t simd_end = size - (size % simd_unit);

  for (int d = 0; d < dim; ++d) {
    auto q = query[d];
    for (int r = 0; r < simd_end; r += simd_unit) {
      auto data_vec = load<vfloat>(&dataset[d][r]);
      auto sub_result = data_vec - q;
      sub_result = sub_result * sub_result;
      auto dist_vec = load<vfloat>(&dist[r]) + sub_result;
      store(dist_vec, &dist[r]);
    }
    for (int r = simd_end; r < size ; ++r) {
      auto v = dataset[d][r] - q;
      v = v * v;
      dist[r] += v;
    }
  }

  for (int r = 0; r < simd_end; r += simd_unit) {
    auto sqrt_result = sqrt(load<vfloat>(&dist[r]));
    store(sqrt_result, &dist[r]);
  }

  for (int r = simd_end; r < size; ++r) {
    dist[r] = sqrt(dist[r]);
  }

  return dist;
}

vfloat_uptr calcCosDistCWithTsimd(const vector<float*>& dataset, const size_t size, const size_t dim, const float* query) {
  using namespace tsimd;
  const size_t simd_unit = vfloat::static_size;

  if (size < simd_unit) return calcCosDistC(dataset, size, dim, query);

  vfloat_uptr dist{(float*) zmalloc_aligned(sizeof(float)*size), free};
  vfloat_uptr mag_d{(float*) zmalloc_aligned(sizeof(float)*size), free};
  fill(dist.get(), dist.get() + size, 0);
  fill(mag_d.get(), mag_d.get() + size, 0);

  const size_t dim_simd_end = dim - (dim % simd_unit);
  const size_t size_simd_end = size - (size % simd_unit);

  for (int d = 0; d < dim; d++) {
    auto q = query[d];
    for (int r = 0; r < size_simd_end; r += simd_unit) {
      auto v = load<vfloat>(&dataset[d][r]);

      auto dot_product = load<vfloat>(&dist[r]) + (v * q);
      store(dot_product, &dist[r]);

      auto mag = load<vfloat>(&mag_d[r]) + (v * v);
      store(mag, &mag_d[r]);
    }
    for (int r = size_simd_end; r < size; ++r) {
      auto v = dataset[d][r];
      auto dot_product = v * q;
      dist[r] += dot_product;
      mag_d[r] += v * v;
    }
  }

  vfloat tmp(0);
  for (int i = 0; i < dim_simd_end; i += simd_unit) {
    auto q = load<vfloat>(&query[i]);
    tmp += (q * q);
  }
  float mag_q = 0;
  for (int i = 0; i < simd_unit; ++i) {
    mag_q += tmp[i];
  }
  for (int i = dim_simd_end; i < dim; ++i) {
    mag_q += query[i] * query[i];
  }
  mag_q = sqrt(mag_q);

  for (int r = 0; r < size_simd_end; r += simd_unit) {
    auto md = sqrt(load<vfloat>(&mag_d[r])) * mag_q;
    auto dc = load<vfloat>(&dist[r]) / md;
    store(dc, &dist[r]);
  }
  for (int r = size_simd_end; r < size; ++r) {
    dist[r] /= sqrt(mag_d[r]) * mag_q;
  }

  return dist;
}

vfloat_uptr calcEucDistRWithTsimd(const vector<float*>& dataset, const size_t size, const size_t dim, const float* query) {
  using namespace tsimd;
  const size_t simd_unit = vfloat::static_size;

  if (dim < simd_unit) return calcEucDistR(dataset, size, dim, query);

  vfloat_uptr dist{(float*) zmalloc_aligned(sizeof(float) * size), free};
  fill(dist.get(), dist.get() + size, 0);

  const size_t dim_simd_end = dim - (dim % simd_unit);
  const size_t size_simd_end = size - (size % simd_unit);

  for (int r = 0; r < size; ++r) {
    vfloat sqr_result(0);
    for (int d = 0; d < dim_simd_end; d += simd_unit) {
      auto q = load<vfloat>(&query[d]);
      auto v = load<vfloat>(&dataset[r][d]);
      auto sub_result = q - v;
      sqr_result += sub_result * sub_result;
    }
    for (int i = 0; i < simd_unit; ++i) {
      dist[r] += sqr_result[i];
    }
    for (int d = dim_simd_end; d < dim; ++d) {
      auto sub_result = (query[d] - dataset[r][d]);
      dist[r] += sub_result * sub_result;
    }
  }

  for (int r = 0; r < size_simd_end; r += simd_unit) {
    auto sqrt_result = sqrt(load<vfloat>(&dist[r]));
    store(sqrt_result, &dist[r]);
  }
  for (int r = size_simd_end; r < size; ++r) {
    dist[r] = sqrt(dist[r]);
  }

  return dist;
}

vfloat_uptr calcEucDistRWithThreadAndTsimd(const vector<float*>& dataset, const size_t size, const size_t dim, const float* query, const size_t num_thread) {
  using namespace tsimd;
  const size_t simd_unit = vfloat::static_size;

  if (dim < simd_unit) return calcEucDistR(dataset, size, dim, query);
  if (size < num_thread) return calcEucDistRWithTsimd(dataset, size, dim, query);

  vfloat_uptr dist{(float*) zmalloc_aligned(sizeof(float) * size), free};
  fill(dist.get(), dist.get() + size, 0);

  const size_t dim_simd_end = dim - (dim % simd_unit);
  const size_t size_simd_end = size - (size % simd_unit);

  vector<thread> threads;
  auto calc = [&](int from, int to) {
    for (int r = from; r < to; ++r) {
      vfloat sqr_result(0);
      for (int d = 0; d < dim_simd_end; d += simd_unit) {
        auto q = load<vfloat>(&query[d]);
        auto v = load<vfloat>(&dataset[r][d]);
        auto sub_result = q - v;
        sqr_result += sub_result * sub_result;
      }
      for (int i = 0; i < simd_unit; ++i) {
        dist[r] += sqr_result[i];
      }
      for (int d = dim_simd_end; d < dim; ++d) {
        auto sub_result = (query[d] - dataset[r][d]);
        dist[r] += sub_result * sub_result;
      }
    }
  };

  for (int i = 1; i < num_thread; ++i) {
    int from, to;
    from = (size / num_thread) * i;
    if (i != num_thread-1) to = (size / num_thread) * (i + 1);
    else to = size;
    threads.emplace_back(calc, from, to);
  }
  calc(0, size / num_thread);
  for (auto& thread : threads) thread.join();

  for (int r = 0; r < size_simd_end; r += simd_unit) {
    auto sqrt_result = sqrt(load<vfloat>(&dist[r]));
    store(sqrt_result, &dist[r]);
  }
  for (int r = size_simd_end; r < size; ++r) {
    dist[r] = sqrt(dist[r]);
  }

  return dist;
}

vfloat_uptr calcCosDistRWithTsimd(const vector<float*>& dataset, const size_t size, const size_t dim, const float* query) {
  using namespace tsimd;
  const size_t simd_unit = vfloat::static_size;

  if (dim < simd_unit) return calcCosDistR(dataset, size, dim, query);

  vfloat_uptr dist{(float*) zmalloc_aligned(sizeof(float) * size), free};
  fill(dist.get(), dist.get() + size, 0);
  float mag_d;

  const size_t simd_end = dim - (dim % simd_unit);

  vfloat tmp(0);
  for (int i = 0; i < simd_end; i += simd_unit) {
    auto q = load<vfloat>(&query[i]);
    tmp += (q * q);
  }
  float mag_q = 0;
  for (int i = 0; i < simd_unit; ++i) {
    mag_q += tmp[i];
  }
  for (int i = simd_end; i < dim; ++i) {
    mag_q += query[i] * query[i];
  }
  mag_q = sqrt(mag_q);

  for (int r = 0; r < size; ++r) {
    mag_d = 0;
    vfloat dc(0), vv(0);
    for (int d = 0; d < simd_end; d += simd_unit) {
      auto q = load<vfloat>(&query[d]);
      auto v = load<vfloat>(&dataset[r][d]);
      vv += v * v;
      dc += v * q;
    }
    for (int i = 0; i < simd_unit; ++i) {
      dist[r] += dc[i];
      mag_d += vv[i];
    }
    for (int i = simd_end; i < dim; ++i) {
      dist[r] += dataset[r][i] * query[i];
      mag_d += dataset[r][i] * dataset[r][i];
    }
    mag_d = sqrt(mag_d);
    dist[r] /= mag_d * mag_q;
  }

  return dist;
}

vfloat_uptr calcCosDistRWithThreadAndTsimd(const vector<float*>& dataset, const size_t size, const size_t dim, const float* query, const size_t num_thread) {
  using namespace tsimd;
  const size_t simd_unit = vfloat::static_size;

  if (dim < simd_unit) return calcCosDistR(dataset, size, dim, query);
  if (size < num_thread) return calcCosDistRWithTsimd(dataset, size, dim, query);

  vfloat_uptr dist{(float*) zmalloc_aligned(sizeof(float) * size), free};
  fill(dist.get(), dist.get() + size, 0);

  const size_t simd_end = dim - (dim % simd_unit);

  vfloat tmp(0);
  for (int i = 0; i < simd_end; i += simd_unit) {
    auto q = load<vfloat>(&query[i]);
    tmp += (q * q);
  }
  float mag_q = 0;
  for (int i = 0; i < simd_unit; ++i) {
    mag_q += tmp[i];
  }
  for (int i = simd_end; i < dim; ++i) {
    mag_q += query[i] * query[i];
  }
  mag_q = sqrt(mag_q);

  vector<thread> threads;
  auto calc = [&](int from, int to) {
    for (int r = from; r < to; ++r) {
      float mag_d = 0;
      vfloat dc(0), vv(0);
      for (int d = 0; d < simd_end; d += simd_unit) {
        auto q = load<vfloat>(&query[d]);
        auto v = load<vfloat>(&dataset[r][d]);
        vv += v * v;
        dc += v * q;
      }
      for (int i = 0; i < simd_unit; ++i) {
        dist[r] += dc[i];
        mag_d += vv[i];
      }
      for (int i = simd_end; i < dim; ++i) {
        dist[r] += dataset[r][i] * query[i];
        mag_d += dataset[r][i] * dataset[r][i];
      }
      mag_d = sqrt(mag_d);
      dist[r] /= mag_d * mag_q;
    }
  };

  for (int i = 1; i < num_thread; ++i) {
    int from, to;
    from = (size / num_thread) * i;
    if (i != num_thread-1) to = (size / num_thread) * (i + 1);
    else to = size;
    threads.emplace_back(calc, from, to);
  }
  calc(0, size / num_thread);
  for (auto& thread : threads) thread.join();

  return dist;
}

vector<vfloat_uptr> batchCalcCosDistC(
    const vector<float*>& dataset,
    const float* magnitude_dataset,
    const size_t size,
    const size_t dim,
    const float* query,
    const float* magnitude_query,
    const size_t num_queries) {
  vector<vfloat_uptr> dist;
  dist.reserve(num_queries);
  for (int i = 0; i < num_queries; ++i) {
    dist.emplace_back((float*) zmalloc_aligned(sizeof(float) * size), free);
    fill(dist[i].get(), dist[i].get() + size, 0);
  }

  for (int i = 0; i < num_queries; ++i) {
    for (int d = 0; d < dim; ++d) {
      float q = query[i * dim + d];
      for (int r = 0; r < size; ++r) {
        dist[i][r] += dataset[d][r] * q;
      }
    }
  }

  for (int i = 0; i < num_queries; ++i) {
    auto q = magnitude_query[i];
    for (int r = 0; r < size; ++r) {
      dist[i][r] /= q * magnitude_dataset[r];
    }
  }

  return dist;
}

vector<vfloat_uptr> batchCalcCosDistR(
    const vector<float*>& dataset,
    const float* magnitude_dataset,
    const size_t size,
    const size_t dim,
    const float* query,
    const float* magnitude_query,
    const size_t num_queries) {
  vector<vfloat_uptr> dist;
  dist.reserve(num_queries);
  for (int i = 0; i < num_queries; ++i) {
    dist.emplace_back((float*) zmalloc_aligned(sizeof(float) * size), free);
    fill(dist[i].get(), dist[i].get() + size, 0);
  }

  for (int i = 0; i < num_queries; ++i) {
    for (int r = 0; r < size; ++r) {
      for (int d = 0; d < dim; ++d) {
        dist[i][r] += dataset[r][d] * query[i * dim + d];
      }
    }
  }

  for (int i = 0; i < num_queries; ++i) {
    auto q = magnitude_query[i];
    for (int r = 0; r < size; ++r) {
      dist[i][r] /= q * magnitude_dataset[r];
    }
  }

  return dist;
}

vector<vfloat_uptr> batchCalcCosDistRWithThread(
    const vector<float*>& dataset,
    const float* magnitude_dataset,
    const size_t size,
    const size_t dim,
    const float* query,
    const float* magnitude_query,
    const size_t num_queries,
    const size_t num_thread) {
  vector<vfloat_uptr> dist;
  dist.reserve(num_queries);
  for (int i = 0; i < num_queries; ++i) {
    dist.emplace_back((float*) zmalloc_aligned(sizeof(float) * size), free);
    fill(dist[i].get(), dist[i].get() + size, 0);
  }

  vector<thread> threads;
  auto calc = [&](int from, int to) {
    for (int i = from; i < to; ++i) {
      for (int r = 0; r < size; ++r) {
        for (int d = 0; d < dim; ++d) {
          dist[i][r] += dataset[r][d] * query[i * dim + d];
        }
      }
    }

    for (int i = from; i < to; ++i) {
      auto q = magnitude_query[i];
      for (int r = 0; r < size; ++r) {
        dist[i][r] /= q * magnitude_dataset[r];
      }
    }
  };

  size_t n_thread = num_queries < num_thread? num_queries : num_thread;
  for (int i = 1; i < n_thread; ++i) {
    int from, to;
    from = num_queries / n_thread * i;
    if (i != n_thread-1) to = num_queries / n_thread * (i + 1);
    else to = num_queries;
    threads.emplace_back(calc, from, to);
  }
  calc(0, num_queries / n_thread);
  for (auto& thread : threads) thread.join();

  return dist;
}

vector<vfloat_uptr> batchCalcCosDistCWithTsimd(
    const vector<float*>& dataset,
    const float* magnitude_dataset,
    const size_t size,
    const size_t dim,
    const float* query,
    const float* magnitude_query,
    const size_t num_queries) {
  using namespace tsimd;
  const size_t simd_unit = vfloat::static_size;

  if (size < simd_unit)
    return batchCalcCosDistC(dataset, magnitude_dataset, size, dim, query, magnitude_query, num_queries);

  const size_t simd_end = size - (size % simd_unit);

  vector<vfloat_uptr> dist;
  dist.reserve(num_queries);
  for (int i = 0; i < num_queries; ++i) {
    dist.emplace_back((float*) zmalloc_aligned(sizeof(float) * size), free);
    fill(dist[i].get(), dist[i].get() + size, 0);
  }

  for (int i = 0; i < num_queries; ++i) {
    for (int d = 0; d < dim; ++d) {
      auto q = query[i * dim + d];
      for (int r = 0; r < simd_end; r += simd_unit) {
        auto v = load<vfloat>(&dataset[d][r]);
        auto dot_product = load<vfloat>(&dist[i][r]) + (v * q);
        store(dot_product, &dist[i][r]);
      }
      for (int r = simd_end; r < size; ++r) {
        dist[i][r] += dataset[d][r] * q;
      }
    }
  }

  for (int i = 0; i < num_queries; ++i) {
    auto q = magnitude_query[i];
    for (int r = 0; r < simd_end; r += simd_unit) {
      auto md = load<vfloat>(&magnitude_dataset[r]) * q;
      auto dist_cos = load<vfloat>(&dist[i][r]) / md;
      store(dist_cos, &dist[i][r]);
    }
    for (int r = simd_end; r < size; ++r) {
      dist[i][r] /= q * magnitude_dataset[r];
    }
  }

  return dist;
}

vector<vfloat_uptr> batchCalcCosDistRWithTsimd(
    const vector<float*>& dataset,
    const float* magnitude_dataset,
    const size_t size,
    const size_t dim,
    const float* query,
    const float* magnitude_query,
    const size_t num_queries) {
  using namespace tsimd;
  const size_t simd_unit = vfloat::static_size;

  if (dim < simd_unit || dim % simd_unit != 0)
    return batchCalcCosDistR(dataset, magnitude_dataset, size, dim, query, magnitude_query, num_queries);

  vector<vfloat_uptr> dist;
  dist.reserve(num_queries);
  for (int i = 0; i < num_queries; ++i) {
    dist.emplace_back((float*) zmalloc_aligned(sizeof(float) * size), free);
    fill(dist[i].get(), dist[i].get() + size, 0);
  }

  for (int i = 0; i < num_queries; ++i) {
    for (int r = 0; r < size; ++r) {
      vfloat dot_product{0};
      for (int d = 0; d < dim; d += simd_unit) {
        dot_product += load<vfloat>(&query[i * dim + d]) * load<vfloat>(&dataset[r][d]);
      }
      for (int j = 0; j < simd_unit; ++j) {
        dist[i][r] += dot_product[j];
      }
      dist[i][r] /= magnitude_dataset[r] * magnitude_query[i];
    }
  }

  return dist;
}

vector<vfloat_uptr> batchCalcCosDistRWithThreadAndTsimd(
    const vector<float*>& dataset,
    const float* magnitude_dataset,
    const size_t size,
    const size_t dim,
    const float* query,
    const float* magnitude_query,
    const size_t num_queries,
    const size_t num_thread) {
  using namespace tsimd;
  const size_t simd_unit = vfloat::static_size;

  if (dim < simd_unit || dim % simd_unit != 0)
    return batchCalcCosDistRWithThread(dataset, magnitude_dataset, size, dim, query, magnitude_query, num_queries, num_thread);

  vector<vfloat_uptr> dist;
  dist.reserve(num_queries);
  for (int i = 0; i < num_queries; ++i) {
    dist.emplace_back((float*) zmalloc_aligned(sizeof(float) * size), free);
    fill(dist[i].get(), dist[i].get() + size, 0);
  }

  vector<thread> threads;
  auto calc = [&](int from, int to) {
    for (int i = from; i < to; ++i) {
      for (int r = 0; r < size; ++r) {
        vfloat dot_product{0};
        for (int d = 0; d < dim; d += simd_unit) {
          dot_product += load<vfloat>(&query[i * dim + d]) * load<vfloat>(&dataset[r][d]);
        }
        for (int j = 0; j < simd_unit; ++j) {
          dist[i][r] += dot_product[j];
        }
        dist[i][r] /= magnitude_dataset[r] * magnitude_query[i];
      }
    }
  };

  size_t n_thread = num_queries < num_thread? num_queries : num_thread;
  for (int i = 1; i < n_thread; ++i) {
    int from, to;
    from = num_queries / n_thread * i;
    if (i != n_thread - 1) to = num_queries / n_thread * (i + 1);
    else to = num_queries;
    threads.emplace_back(calc, from, to);
  }
  calc(0, num_queries / n_thread);
  for (auto& thread : threads) thread.join();

  return dist;
}

vector<vfloat_uptr> batchCalcEucDistC(
    const vector<float*>& dataset,
    const float* magnitude_dataset,
    const size_t size,
    const size_t dim,
    const float* query,
    const float* magnitude_query,
    const size_t num_queries) {
  vector<vfloat_uptr> dist;
  dist.reserve(num_queries);
  for (int i = 0; i < num_queries; ++i) {
    dist.emplace_back((float*) zmalloc_aligned(sizeof(float) * size), free);
    fill(dist[i].get(), dist[i].get() + size, 0);
  }

  for (int i = 0; i < num_queries; ++i) {
    for (int d = 0; d < dim; ++d) {
      float q = query[i * dim + d];
      for (int r = 0; r < size; ++r) {
        dist[i][r] -= dataset[d][r] * q;  // Sum(-XY)
      }
    }
    for (int r = 0; r < size; ++r) {
      dist[i][r] *= 2;  // Sum(-2XY)
      dist[i][r] += magnitude_dataset[r] * magnitude_dataset[r]
        + magnitude_query[i] * magnitude_query[i];  // Sum(X^2 - 2XY + Y^2) = Sum((X-Y)^2)
      dist[i][r] = sqrt(dist[i][r]);  // Sqrt(Sum((X-Y)^2))
    }
  }

  return dist;
}

vector<vfloat_uptr> batchCalcEucDistR(
    const vector<float*>& dataset,
    const float* magnitude_dataset,
    const size_t size,
    const size_t dim,
    const float* query,
    const float* magnitude_query,
    const size_t num_queries) {
  vector<vfloat_uptr> dist;
  dist.reserve(num_queries);
  for (int i = 0; i < num_queries; ++i) {
    dist.emplace_back((float*) zmalloc_aligned(sizeof(float) * size), free);
    fill(dist[i].get(), dist[i].get() + size, 0);
  }

  for (int i = 0; i < num_queries; ++i) {
    for(int r = 0; r < size; ++r) {
      for (int d = 0; d < dim; ++d) {
        dist[i][r] -= dataset[r][d] * query[i * dim + d];
      }
    }
    for (int r = 0; r < size; ++r) {
      dist[i][r] *= 2;
      dist[i][r] += magnitude_dataset[r] * magnitude_dataset[r]
        + magnitude_query[i] * magnitude_query[i];
      dist[i][r] = sqrt(dist[i][r]);
    }
  }

  return dist;
}

vector<vfloat_uptr> batchCalcEucDistRWithThread(
    const vector<float*>& dataset,
    const float* magnitude_dataset,
    const size_t size,
    const size_t dim,
    const float* query,
    const float* magnitude_query,
    const size_t num_queries,
    const size_t num_thread) {
  vector<vfloat_uptr> dist;
  dist.reserve(num_queries);
  for (int i = 0; i < num_queries; ++i) {
    dist.emplace_back((float*) zmalloc_aligned(sizeof(float) * size), free);
    fill(dist[i].get(), dist[i].get() + size, 0);
  }

  vector<thread> threads;
  auto calc = [&](int from, int to) {
    for (int i = from; i < to; ++i) {
      for(int r = 0; r < size; ++r) {
        for (int d = 0; d < dim; ++d) {
          dist[i][r] -= dataset[r][d] * query[i * dim + d];
        }
      }
      for (int r = 0; r < size; ++r) {
        dist[i][r] *= 2;
        dist[i][r] += magnitude_dataset[r] * magnitude_dataset[r]
          + magnitude_query[i] * magnitude_query[i];
        dist[i][r] = sqrt(dist[i][r]);
      }
    }
  };

  size_t n_thread = num_queries < num_thread? num_queries : num_thread;
  for (int i = 1; i < n_thread; ++i) {
    int from, to;
    from = num_queries / n_thread * i;
    if (i != n_thread-1) to = num_queries / n_thread * (i + 1);
    else to = num_queries;
    threads.emplace_back(calc, from, to);
  }
  calc(0, num_queries / n_thread);
  for (auto& thread : threads) thread.join();

  return dist;
}

vector<vfloat_uptr> batchCalcEucDistR(
    const vector<float*>& dataset,
    const size_t size,
    const size_t dim,
    const float* query,
    const size_t num_queries) {
  vector<vfloat_uptr> dist;
  dist.reserve(num_queries);
  for (int i = 0; i < num_queries; ++i) {
    dist.emplace_back((float*) zmalloc_aligned(sizeof(float) * size), free);
    fill(dist[i].get(), dist[i].get() + size, 0);
  }

  for (int i = 0; i < num_queries; ++i) {
    for(int r = 0; r < size; ++r) {
      for (int d = 0; d < dim; ++d) {
        float v = dataset[r][d] - query[i * dim + d];
        dist[i][r] += v * v;
      }
    }
    for (int r = 0; r < size; ++r) {
      dist[i][r] = sqrt(dist[i][r]);
    }
  }

  return dist;
}

vector<vfloat_uptr> batchCalcEucDistRWithThread(
    const vector<float*>& dataset,
    const size_t size,
    const size_t dim,
    const float* query,
    const size_t num_queries,
    const size_t num_thread) {
  vector<vfloat_uptr> dist;
  dist.reserve(num_queries);
  for (int i = 0; i < num_queries; ++i) {
    dist.emplace_back((float*) zmalloc_aligned(sizeof(float) * size), free);
    fill(dist[i].get(), dist[i].get() + size, 0);
  }

  vector<thread> threads;
  auto calc = [&](int from, int to) {
    for (int i = from; i < to; ++i) {
      for(int r = 0; r < size; ++r) {
        for (int d = 0; d < dim; ++d) {
          float v = dataset[r][d] - query[i * dim + d];
          dist[i][r] += v * v;
        }
      }
      for (int r = 0; r < size; ++r) {
        dist[i][r] = sqrt(dist[i][r]);
      }
    }
  };

  size_t n_thread = num_queries < num_thread? num_queries : num_thread;
  for (int i = 1; i < n_thread; ++i) {
    int from, to;
    from = num_queries / n_thread * i;
    if (i != n_thread-1) to = num_queries / n_thread * (i + 1);
    else to = num_queries;
    threads.emplace_back(calc, from, to);
  }
  calc(0, num_queries / n_thread);
  for (auto& thread : threads) thread.join();

  return dist;
}

vector<vfloat_uptr> batchCalcEucDistCWithTsimd(
    const vector<float*>& dataset,
    const float* magnitude_dataset,
    const size_t size,
    const size_t dim,
    const float* query,
    const float* magnitude_query,
    const size_t num_queries) {
  using namespace tsimd;
  const size_t simd_unit = vfloat::static_size;

  if (size < simd_unit)
    return batchCalcEucDistC(dataset, magnitude_dataset, size, dim, query, magnitude_query, num_queries);
  const size_t simd_end = size - (size % simd_unit);

  vector<vfloat_uptr> dist;
  dist.reserve(num_queries);
  for (int i = 0; i < num_queries; ++i) {
    dist.emplace_back((float*) zmalloc_aligned(sizeof(float) * size), free);
    fill(dist[i].get(), dist[i].get() + size, 0);
  }

  for (int i = 0; i < num_queries; ++i) {
    for (int d = 0; d < dim; ++d) {
      auto q = query[i * dim + d];
      for (int r = 0; r < simd_end; r += simd_unit) {
        auto res = load<vfloat>(&dist[i][r]);
        res -= load<vfloat>(&dataset[d][r]) * q;  // Sum(-XY)
        store(res, &dist[i][r]);
      }
      for (int r = simd_end; r < size; ++r) {
        dist[i][r] -= dataset[d][r] * q;  // Sum(-XY)
      }
    }
    for (int r = 0; r < simd_end; r += simd_unit) {
      auto res = load<vfloat>(&dist[i][r]) * 2;  // Sum(-2XY)
      auto vdss = load<vfloat>(&magnitude_dataset[r]);  // Sqrt(Sum(X^2))
      res += vdss * vdss + magnitude_query[i] * magnitude_query[i];  // Sum(X^2 - 2XY + Y^2)
      res = sqrt(res);
      store(res, &dist[i][r]);
    }
    for (int r = simd_end; r < size; ++r) {
      dist[i][r] *= 2;  // Sum(-2XY)
      dist[i][r] += magnitude_dataset[r] * magnitude_dataset[r]
        + magnitude_query[i] * magnitude_query[i];  // Sum(X^2 - 2XY + Y^2) = Sum((X-Y)^2)
      dist[i][r] = sqrt(dist[i][r]);  // Sqrt(Sum((X-Y)^2))
    }
  }

  return dist;
}

vector<vfloat_uptr> batchCalcEucDistRWithTsimd(
    const vector<float*>& dataset,
    const float* magnitude_dataset,
    const size_t size,
    const size_t dim,
    const float* query,
    const float* magnitude_query,
    const size_t num_queries) {
  using namespace tsimd;
  const size_t simd_unit = vfloat::static_size;

  if (dim < simd_unit || dim % simd_unit != 0)
    return batchCalcEucDistR(dataset, magnitude_dataset, size, dim, query, magnitude_query, num_queries);
  const size_t simd_end = size - (size % simd_unit);

  vector<vfloat_uptr> dist;
  dist.reserve(num_queries);
  for (int i = 0; i < num_queries; ++i) {
    dist.emplace_back((float*) zmalloc_aligned(sizeof(float) * size), free);
    fill(dist[i].get(), dist[i].get() + size, 0);
  }

  for (int i = 0; i < num_queries; ++i) {
    for (int r = 0; r < size; ++r) {
      vfloat res{0};
      for (int d = 0; d < dim; d += simd_unit) {
        res -= load<vfloat>(&dataset[r][d]) * load<vfloat>(&query[i * dim + d]);
      }
      res *= 2;
      for (int j = 0; j < simd_unit; ++j) {
        dist[i][r] += res[j];
      }
      dist[i][r] += magnitude_dataset[r] * magnitude_dataset[r]
        + magnitude_query[i] * magnitude_query[i];
      dist[i][r] = sqrt(dist[i][r]);
    }
  }

  return dist;
}

vector<vfloat_uptr> batchCalcEucDistRWithThreadAndTsimd(
    const vector<float*>& dataset,
    const float* magnitude_dataset,
    const size_t size,
    const size_t dim,
    const float* query,
    const float* magnitude_query,
    const size_t num_queries,
    const size_t num_thread) {
  using namespace tsimd;
  const size_t simd_unit = vfloat::static_size;

  if (dim < simd_unit || dim % simd_unit != 0)
    return batchCalcEucDistRWithThread(dataset, magnitude_dataset, size, dim, query, magnitude_query, num_queries, num_thread);
  const size_t simd_end = size - (size % simd_unit);

  vector<vfloat_uptr> dist;
  dist.reserve(num_queries);
  for (int i = 0; i < num_queries; ++i) {
    dist.emplace_back((float*) zmalloc_aligned(sizeof(float) * size), free);
    fill(dist[i].get(), dist[i].get() + size, 0);
  }

  vector<thread> threads;
  auto calc = [&](int from, int to) {
    for (int i = from; i < to; ++i) {
      for (int r = 0; r < size; ++r) {
        vfloat res{0};
        for (int d = 0; d < dim; d += simd_unit) {
          res -= load<vfloat>(&dataset[r][d]) * load<vfloat>(&query[i * dim + d]);
        }
        res *= 2;
        for (int j = 0; j < simd_unit; ++j) {
          dist[i][r] += res[j];
        }
        dist[i][r] += magnitude_dataset[r] * magnitude_dataset[r]
          + magnitude_query[i] * magnitude_query[i];
        dist[i][r] = sqrt(dist[i][r]);
      }
    }
  };

  size_t n_thread = num_queries < num_thread? num_queries : num_thread;
  for (int i = 1; i < n_thread; ++i) {
    int from, to;
    from = num_queries / n_thread * i;
    if (i != n_thread-1) to = num_queries / n_thread * (i + 1);
    else to = num_queries;
    threads.emplace_back(calc, from, to);
  }
  calc(0, num_queries / n_thread);
  for (auto& thread : threads) thread.join();

  return dist;
}

vector<vfloat_uptr> batchCalcEucDistRWithTsimd(
    const vector<float*>& dataset,
    const size_t size,
    const size_t dim,
    const float* query,
    const size_t num_queries) {
  using namespace tsimd;
  const size_t simd_unit = vfloat::static_size;

  if (dim < simd_unit || dim % simd_unit != 0)
    return batchCalcEucDistR(dataset, size, dim, query, num_queries);
  const size_t simd_end = size - (size % simd_unit);

  vector<vfloat_uptr> dist;
  dist.reserve(num_queries);
  for (int i = 0; i < num_queries; ++i) {
    dist.emplace_back((float*) zmalloc_aligned(sizeof(float) * size), free);
    fill(dist[i].get(), dist[i].get() + size, 0);
  }

  for (int i = 0; i < num_queries; ++i) {
    for (int r = 0; r < size; ++r) {
      vfloat tmp{0};
      for (int d = 0; d < dim; d += simd_unit) {
        auto v = load<vfloat>(&dataset[r][d]) - load<vfloat>(&query[i * dim + d]);
        tmp += v * v;
      }
      for (int j = 0; j < simd_unit; ++j) {
        dist[i][r] += tmp[j];
      }
    }
    for (int r = 0; r < simd_end; r += simd_unit) {
      auto v = sqrt(load<vfloat>(&dist[i][r]));
      store(v, &dist[i][r]);
    }
    for (int r = simd_end; r < size; ++r) {
      dist[i][r] = sqrt(dist[i][r]);
    }
  }

  return dist;
}

vector<vfloat_uptr> batchCalcEucDistRWithThreadAndTsimd(
    const vector<float*>& dataset,
    const size_t size,
    const size_t dim,
    const float* query,
    const size_t num_queries,
    const size_t num_thread) {
  using namespace tsimd;
  const size_t simd_unit = vfloat::static_size;

  if (dim < simd_unit || dim % simd_unit != 0)
    return batchCalcEucDistRWithThread(dataset, size, dim, query, num_queries, num_thread);
  const size_t simd_end = size - (size % simd_unit);

  vector<vfloat_uptr> dist;
  dist.reserve(num_queries);
  for (int i = 0; i < num_queries; ++i) {
    dist.emplace_back((float*) zmalloc_aligned(sizeof(float) * size), free);
    fill(dist[i].get(), dist[i].get() + size, 0);
  }

  vector<thread> threads;
  auto calc = [&](int from, int to) {
    for (int i = from; i < to; ++i) {
      for (int r = 0; r < size; ++r) {
        vfloat tmp{0};
        for (int d = 0; d < dim; d += simd_unit) {
          auto v = load<vfloat>(&dataset[r][d]) - load<vfloat>(&query[i * dim + d]);
          tmp += v * v;
        }
        for (int j = 0; j < simd_unit; ++j) {
          dist[i][r] += tmp[j];
        }
      }
      for (int r = 0; r < simd_end; r += simd_unit) {
        auto v = sqrt(load<vfloat>(&dist[i][r]));
        store(v, &dist[i][r]);
      }
      for (int r = simd_end; r < size; ++r) {
        dist[i][r] = sqrt(dist[i][r]);
      }
    }
  };

  size_t n_thread = num_queries < num_thread? num_queries : num_thread;
  for (int i = 1; i < n_thread; ++i) {
    int from, to;
    from = num_queries / n_thread * i;
    if (i != n_thread-1) to = num_queries / n_thread * (i + 1);
    else to = num_queries;
    threads.emplace_back(calc, from, to);
  }
  calc(0, num_queries / n_thread);
  for (auto& thread : threads) thread.join();

  return dist;
}

template <typename T>
inline void swap(vector<T>& arr, int x, int y) {
  T tmp = arr[x];
  arr[x] = arr[y];
  arr[y] = tmp;
}

// find k-nearest distances from `dist` with priority queue
// When the greater value show more similarity (e.g. cosine distance)
vector<int> findKNNG(const float* dist, const size_t size, const size_t k) {
  const size_t k_queue_size = k < size? k : size;
  vector<pair<float, int>> container;
  container.reserve(k_queue_size);
  priority_queue<pair<float, int>, vector<pair<float, int>>, greater<pair<float, int>>>
    pq{greater<pair<float, int>>(), container};
  for (int i = 0; i < k_queue_size; i++) {
    pq.emplace(dist[i], i);
  }
  for (int i = k_queue_size; i < size; i++) {
    if (dist[i] > pq.top().first) {
      pq.pop();
      pq.emplace(dist[i], i);
    }
  }

  vector<int> ids;
  ids.reserve(k_queue_size);
  while (!pq.empty()) {
    ids.push_back(pq.top().second);
    pq.pop();
  }
  sort(ids.begin(), ids.end());
  return ids;
}

// When the less value show more similarity (e.g. euclidean distance)
vector<int> findKNNL(const float* dist, const size_t size, const size_t k) {
  const size_t k_queue_size = k < size? k : size;
  vector<pair<float, int>> container;
  container.reserve(k_queue_size);
  priority_queue<pair<float, int>, vector<pair<float, int>>, less<pair<float, int>>>
    pq{less<pair<float, int>>(), container};
  for (int i = 0; i < k_queue_size; i++) {
    pq.emplace(dist[i], i);
  }
  for (int i = k_queue_size; i < size; i++) {
    if (dist[i] < pq.top().first) {
      pq.pop();
      pq.emplace(dist[i], i);
    }
  }

  vector<int> ids;
  ids.reserve(k_queue_size);
  while (!pq.empty()) {
    ids.push_back(pq.top().second);
    pq.pop();
  }
  sort(ids.begin(), ids.end());
  return ids;
}

template <typename T>
void logVector(const vector<T>& vec) {
  stringstream ss;
  for (auto& v : vec) {
    ss << v << " ";
  }
  cout << ss.str() << endl;
}

template <typename T>
void logVector(const vector<T>& vec, const vector<int>& ids) {
  stringstream ss;
  for (auto& id : ids) {
    ss << id << ":" << vec[id] << " ";
  }
  cout << ss.str() << endl;
}

void logVector(const vfloat_uptr& vec, const vector<int>& ids) {
  stringstream ss;
  for (auto& id : ids) {
    ss << id << ":" << vec[id] << " ";
  }
  cout << ss.str() << endl;
}

int single () {
  Timer t;
  const size_t test_size = 100000;
  const size_t test_dim  = 256;
  const size_t test_k    = 10;
  const size_t test_iter = 10;

  cout << "| configuration | value |" << endl;
  cout << "| ------------- | ----- |" << endl;
  cout << "| test size     | " << test_size <<  " |"  << endl;
  cout << "| test dim      | " << test_dim <<  " |"  << endl;
  cout << "| test k        | " << test_k <<  " |"  << endl;

  auto query = createDataSetR(1, test_dim, 0.0, 100.0)[0];
  auto dataset = createDataSetR(test_size, test_dim, 0.0, 100.0);

  t.start();
  for (int i = 0; i < test_iter-1; ++i) {
    auto dist_no_simd = calcCosDistR(dataset, test_size, test_dim, query);
  }
  auto dist_no_simd = calcCosDistR(dataset, test_size, test_dim, query);
  t.stop();
  cout << "| No SIMD | " << ((double)t.eval()) / test_iter / 1000000 << "ms |" << endl;

  t.start();
  for (int i = 0; i < test_iter-1; ++i) {
    auto dist_simd = calcCosDistRWithTsimd(dataset, test_size, test_dim, query);
  }
  auto dist_simd = calcCosDistRWithTsimd(dataset, test_size, test_dim, query);
  t.stop();
  cout << "| SIMD | " << ((double)t.eval()) / test_iter / 1000000 << "ms |" << endl;

  t.start();
  for (int i = 0; i < test_iter-1; ++i) {
    auto dist_thread = calcCosDistRWithThread(dataset, test_size, test_dim, query, k_num_thread);
  }
  auto dist_thread = calcCosDistRWithThread(dataset, test_size, test_dim, query, k_num_thread);
  t.stop();
  cout << "| thread | " << ((double)t.eval()) / test_iter / 1000000 << "ms |" << endl;

  auto ids_no = findKNNG(dist_no_simd.get(), test_size, test_k);
  auto ids_simd = findKNNG(dist_simd.get(), test_size, test_k);
  auto ids_thread = findKNNG(dist_thread.get(), test_size, test_k);

  // logVector(ids_no);
  // logVector(dist_no_simd, ids_no);
  // logVector(ids_simd);
  // logVector(ids_thread);

  return 0;
}

int batch() {
  Timer t;
  const size_t test_size  = 10000;
  const size_t test_dim   = 256;
  const size_t test_query = 128;
  const size_t test_k     = 10;
  const size_t test_iter  = 10;

  cout << "| configuration | value |" << endl;
  cout << "| ------------- | ----- |" << endl;
  cout << "| test size     | " << test_size <<  " |"  << endl;
  cout << "| test dim      | " << test_dim <<  " |"  << endl;
  cout << "| test query    | " << test_query <<  " |"  << endl;
  cout << "| test k        | " << test_k <<  " |"  << endl;

  auto query = createDataSetR(1, test_dim * test_query, 0.0, 100.0)[0];
  auto dataset = createDataSetR(test_size, test_dim, 0.0, 100.0);
  auto mag_d = calcMagnitudeR(dataset, test_size, test_dim);
  auto mag_q = calcMagnitudeR(query, test_query, test_dim);

  t.start();
  for (int i = 0; i < test_iter-1; ++i) {
    auto dist_default = batchCalcCosDistR(dataset, mag_d.get(), test_size, test_dim, query, mag_q.get(), test_query);
  }
  auto dist_default = batchCalcCosDistR(dataset, mag_d.get(), test_size, test_dim, query, mag_q.get(), test_query);
  t.stop();
  cout << "| Default | " << ((double)t.eval()) / test_iter / 1000000 << "ms |" << endl;

  t.start();
  for (int i = 0; i < test_iter-1; ++i) {
    auto dist_simd = batchCalcCosDistRWithTsimd(dataset, mag_d.get(), test_size, test_dim, query, mag_q.get(), test_query);
  }
  auto dist_simd = batchCalcCosDistRWithTsimd(dataset, mag_d.get(), test_size, test_dim, query, mag_q.get(), test_query);
  t.stop();
  cout << "| SIMD | " << ((double)t.eval()) / test_iter / 1000000 << "ms |" << endl;

  t.start();
  for (int i = 0; i < test_iter-1; ++i) {
    auto dist_thread = batchCalcCosDistRWithThread(dataset, mag_d.get(), test_size, test_dim, query, mag_q.get(), test_query, k_num_thread);
  }
  auto dist_thread = batchCalcCosDistRWithThread(dataset, mag_d.get(), test_size, test_dim, query, mag_q.get(), test_query, k_num_thread);
  t.stop();
  cout << "| thread | " << ((double)t.eval()) / test_iter / 1000000 << "ms |" << endl;

  t.start();
  for (int i = 0; i < test_iter-1; ++i) {
    vector<thread> threads;
    for (int j = 0; j < test_query; ++j) {
      threads.emplace_back(calcCosDistRWithTsimd, dataset, test_size, test_dim, &query[j * test_dim]);
    }
    for (auto& thread : threads) {
      thread.join();
    }
  }
  vector<thread> threads;
  for (int j = 0; j < test_query; ++j) {
    threads.emplace_back(calcCosDistRWithTsimd, dataset, test_size, test_dim, &query[j * test_dim]);
  }
  for (auto& thread : threads) {
    thread.join();
  }
  t.stop();
  cout << "| no batch thread | " << ((double)t.eval()) / test_iter / 1000000 << "ms |" << endl;

  auto ids_no = findKNNG(dist_default[0].get(), test_size, test_k);
  auto ids_simd = findKNNG(dist_simd[0].get(), test_size, test_k);
  auto ids_thread = findKNNG(dist_thread[0].get(), test_size, test_k);

  // logVector(ids_no);
  // logVector(dist_default[0], ids_no);
  // logVector(ids_simd);
  // logVector(ids_thread);

  ids_no = findKNNG(dist_default[test_query-1].get(), test_size, test_k);
  ids_simd = findKNNG(dist_simd[test_query-1].get(), test_size, test_k);
  ids_thread = findKNNG(dist_thread[test_query-1].get(), test_size, test_k);

  // cout << endl << " ----------------------- " << endl << endl;
  // logVector(ids_no);
  // logVector(dist_default[test_query-1], ids_no);
  // logVector(ids_simd);
  // logVector(ids_thread);

  return 0;
}

int main () {
  return single();
  // return batch();
}
