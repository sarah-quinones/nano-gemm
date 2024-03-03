#include <eigen3/Eigen/Core>
#include <benchmark/benchmark.h>
#include <mkl/mkl.h>

constexpr int M = 32;
constexpr int N = 32;
constexpr int K = 32;

void bench_eigen(benchmark::State& s) {
  Eigen::Matrix<float, M, K> a;
  Eigen::Matrix<float, K, N> b;
  Eigen::Matrix<float, M, N> c;

  a.setRandom();
  b.setRandom();
  c.setRandom();

  benchmark::DoNotOptimize(a.data());
  benchmark::DoNotOptimize(b.data());
  benchmark::DoNotOptimize(c.data());
  for (auto _: s) {
    c.noalias() += a * b;
    benchmark::ClobberMemory();
  }
}

void bench_mkl(benchmark::State& s) {
  Eigen::Matrix<float, M, K> a;
  Eigen::Matrix<float, K, N> b;
  Eigen::Matrix<float, M, N> c;

  a.setRandom();
  b.setRandom();
  c.setRandom();

  benchmark::DoNotOptimize(a.data());
  benchmark::DoNotOptimize(b.data());
  benchmark::DoNotOptimize(c.data());

  int m = M;
  int n = N;
  int k = K;
  float alpha = 1.0;
  float beta = 2.5;

  for (auto _: s) {
for (int i = 0; i < 16; ++i){    sgemm("N", "N", &m, &n, &k, &alpha, a.data(), &m, b.data(), &k, &beta, c.data(), &m);
}    benchmark::ClobberMemory();
  }
}

BENCHMARK(bench_eigen);
BENCHMARK(bench_mkl);
BENCHMARK_MAIN();
