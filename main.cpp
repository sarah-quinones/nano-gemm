#include <eigen3/Eigen/Core>
#include <benchmark/benchmark.h>
#include <libxsmm.h>

void bench_eigen(benchmark::State& s) {
  long m = s.range(0);
  long n = s.range(1);
  long k = s.range(2);

  Eigen::Matrix<float, -1, -1> a(m, k);
  Eigen::Matrix<float, -1, -1> b(k, n);
  Eigen::Matrix<float, -1, -1> c(m, n);

  a.setOnes();
  b.setOnes();
  c.setOnes();

  benchmark::DoNotOptimize(a.data());
  benchmark::DoNotOptimize(b.data());
  benchmark::DoNotOptimize(c.data());
  for (auto _: s) {
    c.noalias() += 2.5 * a * b;
    benchmark::ClobberMemory();
  }
}

void bench_libxsmm(benchmark::State& s) {
  long m = s.range(0);
  long n = s.range(1);
  long k = s.range(2);

  std::vector<float> a(m * k, 1.0), b(k * n, 1.0), c(m * n, 1.0);

  typedef libxsmm_mmfunction<float> kernel_type;
  kernel_type kernel(LIBXSMM_GEMM_FLAG_NONE, m, n, k, 1.0 /*alpha*/, 1.0 /*beta*/);

  benchmark::DoNotOptimize(a.data());
  benchmark::DoNotOptimize(b.data());
  benchmark::DoNotOptimize(c.data());
  for (auto _: s) {
    kernel(&a[0], &b[0], &c[0]);
    benchmark::ClobberMemory();
  }
}

constexpr int MAX = 64;
BENCHMARK(bench_eigen)->Apply([](benchmark::internal::Benchmark* b) {
  for (int i = 1; i <= MAX; ++i) {
    b->Args({i, i, i});
  }
  for (int i = 1; i <= MAX; ++i) {
    b->Args({4, i, 4});
  }
  for (int i = 1; i <= MAX; ++i) {
    b->Args({i, 4, 4});
  }
});
BENCHMARK(bench_libxsmm)->Apply([](benchmark::internal::Benchmark* b) {
  for (int i = 1; i <= MAX; ++i) {
    b->Args({i, i, i});
  }
  for (int i = 1; i <= MAX; ++i) {
    b->Args({4, i, 4});
  }
  for (int i = 1; i <= MAX; ++i) {
    b->Args({i, 4, 4});
  }
});
BENCHMARK_MAIN();
