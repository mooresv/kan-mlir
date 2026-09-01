#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <chrono>

struct MemRef2D {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
};

extern "C" MemRef2D kanlib_piecewise_benchmark(
    float *allocated,
    float *aligned,
    int64_t offset,
    int64_t size0,
    int64_t size1,
    int64_t stride0,
    int64_t stride1);

int main() {
  constexpr int64_t B = 256;
  constexpr int64_t Din = 2;
  constexpr int Iterations = 20;

  const size_t count = B * Din;
  const size_t bytes = count * sizeof(float);

  float *input = static_cast<float *>(std::malloc(bytes));

  if (!input) {
    std::fprintf(stderr, "input allocation failed\n");
    return 1;
  }

  // Deterministic values spanning approximately [-1, 1].
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t i = 0; i < Din; ++i) {
      const int64_t idx = b * Din + i;
      input[idx] =
          -0.95f +
          1.90f * static_cast<float>(idx % 101) / 100.0f;
    }
  }

  MemRef2D result{};

  for (int iter = 0; iter < Iterations; ++iter) {
    const auto start = std::chrono::steady_clock::now();

    result = kanlib_piecewise_benchmark(
        input,     // allocated
        input,     // aligned
        0,         // offset
        B,         // size 0
        Din,       // size 1
        Din,       // stride 0
        1);        // stride 1

    const auto stop = std::chrono::steady_clock::now();

    const double us =
        std::chrono::duration<double, std::micro>(
            stop - start).count();

    std::printf(
        "iteration %2d: %10.3f us   first=% .8f\n",
        iter,
        us,
        result.aligned[result.offset]);
  }

  std::free(input);

  return 0;
}
