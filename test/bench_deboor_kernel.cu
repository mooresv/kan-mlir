#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t err__ = (call);                                            \
    if (err__ != cudaSuccess) {                                            \
      std::fprintf(stderr,                                                 \
                   "CUDA error %s:%d: %s\n",                               \
                   __FILE__, __LINE__, cudaGetErrorString(err__));          \
      std::exit(EXIT_FAILURE);                                             \
    }                                                                      \
  } while (0)

static constexpr int DIN = 2;
static constexpr int DOUT = 8;

static constexpr int DEGREE = 3;
static constexpr int NCOEFF = 8;
static constexpr int NKNOTS = NCOEFF + DEGREE + 1;

static constexpr int THREADS = 256;

/*
 * Current KANLib knot grid:
 *
 * [-2.2, -1.8, -1.4, -1.0, -0.6, -0.2,
 *   0.2,  0.6,  1.0,  1.4,  1.8,  2.2]
 *
 * Uniform knot spacing:
 *
 * h = 0.4
 *
 * Evaluation domain:
 *
 * [-1.0, 1.0]
 *
 * Active spline spans:
 *
 * k = 3,4,5,6,7
 *
 * corresponding to:
 *
 * [-1.0,-0.6)
 * [-0.6,-0.2)
 * [-0.2, 0.2)
 * [ 0.2, 0.6)
 * [ 0.6, 1.0]
 */

static constexpr float DOMAIN_MIN = -1.0f;

static constexpr float H = 0.4f;
static constexpr float INV_H = 1.0f / H;

static constexpr float INV_2H = 1.0f / (2.0f * H);
static constexpr float INV_3H = 1.0f / (3.0f * H);

static constexpr int FIRST_SPAN = DEGREE;
static constexpr int NUM_ACTIVE_SPANS = 5;


template <typename T>
std::vector<T> read_binary(const std::string &path)
{
  std::ifstream file(path, std::ios::binary | std::ios::ate);

  if (!file) {
    std::fprintf(stderr, "Could not open %s\n", path.c_str());
    std::exit(EXIT_FAILURE);
  }

  std::streamsize bytes = file.tellg();

  if (bytes % sizeof(T) != 0) {
    std::fprintf(stderr,
                 "File size for %s is not a multiple of element size\n",
                 path.c_str());
    std::exit(EXIT_FAILURE);
  }

  file.seekg(0, std::ios::beg);

  std::vector<T> data(bytes / sizeof(T));

  if (!file.read(reinterpret_cast<char *>(data.data()), bytes)) {
    std::fprintf(stderr, "Could not read %s\n", path.c_str());
    std::exit(EXIT_FAILURE);
  }

  return data;
}


// -------------------------------------------------------------------------
// Optimized uniform-grid knot span selection
// -------------------------------------------------------------------------
//
// For x in [-1,1]:
//
//   piece = floor((x - DOMAIN_MIN) / H)
//
// and
//
//   span = FIRST_SPAN + piece
//
// Clamp the right endpoint x=1 to the final interval.
//
// This avoids a linear or binary knot search entirely.
//
__device__ __forceinline__
int find_span_uniform(float x)
{
  int piece =
      __float2int_rd((x - DOMAIN_MIN) * INV_H);

  piece = max(0, min(piece, NUM_ACTIVE_SPANS - 1));

  return FIRST_SPAN + piece;
}


// -------------------------------------------------------------------------
// Fully specialized cubic de Boor
// -------------------------------------------------------------------------
//
// Generic de Boor:
//
//   d_j^(0) = c[k-p+j]
//
//   alpha =
//     (x - t[j+k-p])
//     -------------------------------
//     t[j+1+k-r] - t[j+k-p]
//
//   d_j^(r) =
//     (1-alpha) d_{j-1}^{r-1}
//       + alpha d_j^{r-1}
//
// For p = 3:
//
//   r = 1 : 3 updates
//   r = 2 : 2 updates
//   r = 3 : 1 update
//
// total = 6 interpolation updates.
//
// Because the knots are uniformly spaced:
//
//   r = 1 denominator = 3h
//   r = 2 denominator = 2h
//   r = 3 denominator = h
//
// so all divisions can be replaced by multiplies with constants.
//
// Each interpolation:
//
//   (1-a)*left + a*right
//
// is evaluated as:
//
//   left + a*(right-left)
//
// using FMA.
//
// coeff points to the 8 coefficients of one learned edge.
// knots points to the 12-knot vector for one input feature.
//
__device__ __forceinline__
float deboor_cubic_uniform(float x,
                           const float *__restrict__ knots,
                           const float *__restrict__ coeff)
{
  const int k = find_span_uniform(x);

  float d0 = coeff[k - 3];
  float d1 = coeff[k - 2];
  float d2 = coeff[k - 1];
  float d3 = coeff[k];


  // ------------------------------------------------------------------
  // r = 1
  //
  // j = 3
  //
  // alpha =
  //   (x - t[k]) / (t[k+3] - t[k])
  // = (x - t[k]) / (3h)
  // ------------------------------------------------------------------

  float alpha =
      (x - knots[k]) * INV_3H;

  d3 = fmaf(alpha, d3 - d2, d2);


  // j = 2
  //
  // alpha =
  //   (x - t[k-1]) / (t[k+2] - t[k-1])
  // = (x - t[k-1]) / (3h)

  alpha =
      (x - knots[k - 1]) * INV_3H;

  d2 = fmaf(alpha, d2 - d1, d1);


  // j = 1
  //
  // alpha =
  //   (x - t[k-2]) / (t[k+1] - t[k-2])
  // = (x - t[k-2]) / (3h)

  alpha =
      (x - knots[k - 2]) * INV_3H;

  d1 = fmaf(alpha, d1 - d0, d0);


  // ------------------------------------------------------------------
  // r = 2
  //
  // j = 3
  //
  // denominator = 2h
  // ------------------------------------------------------------------

  alpha =
      (x - knots[k]) * INV_2H;

  d3 = fmaf(alpha, d3 - d2, d2);


  // j = 2

  alpha =
      (x - knots[k - 1]) * INV_2H;

  d2 = fmaf(alpha, d2 - d1, d1);


  // ------------------------------------------------------------------
  // r = 3
  //
  // j = 3
  //
  // denominator = h
  // ------------------------------------------------------------------

  alpha =
      (x - knots[k]) * INV_H;

  d3 = fmaf(alpha, d3 - d2, d2);

  return d3;
}


// -------------------------------------------------------------------------
// Fused KAN layer
// -------------------------------------------------------------------------
//
// One CUDA thread computes one:
//
//   output[b,o]
//
// and evaluates both learned edges:
//
//   y[b,o] =
//       phi[o,0](x[b,0])
//     + phi[o,1](x[b,1])
//
// Parameter layouts:
//
// input:
//   [batch, DIN]
//
// knots:
//   [DIN, NKNOTS]
//
// coeffs:
//   [DOUT, DIN, NCOEFF]
//
// output:
//   [batch, DOUT]
//
__global__
void kan_deboor_kernel(const float *__restrict__ input,
                       const float *__restrict__ knots,
                       const float *__restrict__ coeffs,
                       float *__restrict__ output,
                       long long batch)
{
  const long long tid =
      static_cast<long long>(blockIdx.x) * blockDim.x
      + threadIdx.x;

  const long long total =
      batch * static_cast<long long>(DOUT);

  if (tid >= total)
    return;

  const long long b =
      tid / DOUT;

  const int o =
      static_cast<int>(tid % DOUT);

  float acc = 0.0f;

#pragma unroll
  for (int i = 0; i < DIN; ++i) {

    const float x =
        input[b * DIN + i];

    const float *edge_knots =
        knots + i * NKNOTS;

    const float *edge_coeff =
        coeffs
        + (o * DIN + i) * NCOEFF;

    const float value =
        deboor_cubic_uniform(
            x,
            edge_knots,
            edge_coeff);

    acc += value;
  }

  output[b * DOUT + o] = acc;
}


int main(int argc, char **argv)
{
  if (argc < 4 || argc > 6) {
    std::fprintf(
        stderr,
        "Usage:\n"
        "  %s <data-dir> <batch-size> <expected-file>"
        " [warmup] [iterations]\n",
        argv[0]);

    return EXIT_FAILURE;
  }

  const std::string data_dir =
      argv[1];

  const long long batch =
      std::stoll(argv[2]);

  const std::string expected_path =
      argv[3];

  const int warmup =
      argc >= 5 ? std::stoi(argv[4]) : 100;

  const int iterations =
      argc >= 6 ? std::stoi(argv[5]) : 1000;


  // ------------------------------------------------------------------
  // Load benchmark data
  // ------------------------------------------------------------------

  auto input =
      read_binary<float>(
          data_dir + "/input.bin");

  auto knots =
      read_binary<float>(
          data_dir + "/grid.bin");

  auto coeffs =
      read_binary<float>(
          data_dir + "/source_coeffs.bin");

  auto expected =
      read_binary<float>(
          expected_path);


  const size_t expected_input =
      static_cast<size_t>(batch) * DIN;

  const size_t expected_output =
      static_cast<size_t>(batch) * DOUT;


  if (input.size() != expected_input) {

    std::fprintf(
        stderr,
        "input.bin has %zu floats; expected %zu\n",
        input.size(),
        expected_input);

    return EXIT_FAILURE;
  }


  if (expected.size() != expected_output) {

    std::fprintf(
        stderr,
        "Expected output has %zu floats; expected %zu\n",
        expected.size(),
        expected_output);

    return EXIT_FAILURE;
  }


  if (knots.size() != DIN * NKNOTS) {

    std::fprintf(
        stderr,
        "grid.bin has %zu floats; expected %d\n",
        knots.size(),
        DIN * NKNOTS);

    return EXIT_FAILURE;
  }


  if (coeffs.size() != DOUT * DIN * NCOEFF) {

    std::fprintf(
        stderr,
        "source_coeffs.bin has %zu floats; expected %d\n",
        coeffs.size(),
        DOUT * DIN * NCOEFF);

    return EXIT_FAILURE;
  }


  // ------------------------------------------------------------------
  // GPU
  // ------------------------------------------------------------------

  int device = 0;

  cudaDeviceProp prop{};

  CUDA_CHECK(
      cudaSetDevice(device));

  CUDA_CHECK(
      cudaGetDeviceProperties(
          &prop,
          device));

  std::printf(
      "GPU                : %s\n",
      prop.name);


  // ------------------------------------------------------------------
  // Persistent device buffers
  // ------------------------------------------------------------------

  float *d_input = nullptr;
  float *d_knots = nullptr;
  float *d_coeffs = nullptr;
  float *d_output = nullptr;


  CUDA_CHECK(
      cudaMalloc(
          &d_input,
          input.size() * sizeof(float)));


  CUDA_CHECK(
      cudaMalloc(
          &d_knots,
          knots.size() * sizeof(float)));


  CUDA_CHECK(
      cudaMalloc(
          &d_coeffs,
          coeffs.size() * sizeof(float)));


  CUDA_CHECK(
      cudaMalloc(
          &d_output,
          expected_output * sizeof(float)));


  CUDA_CHECK(
      cudaMemcpy(
          d_input,
          input.data(),
          input.size() * sizeof(float),
          cudaMemcpyHostToDevice));


  CUDA_CHECK(
      cudaMemcpy(
          d_knots,
          knots.data(),
          knots.size() * sizeof(float),
          cudaMemcpyHostToDevice));


  CUDA_CHECK(
      cudaMemcpy(
          d_coeffs,
          coeffs.data(),
          coeffs.size() * sizeof(float),
          cudaMemcpyHostToDevice));


  // ------------------------------------------------------------------
  // Launch geometry
  // ------------------------------------------------------------------

  const long long total_outputs =
      batch * DOUT;

  const int blocks =
      static_cast<int>(
          (total_outputs + THREADS - 1)
          / THREADS);


  auto launch = [&]() {

    kan_deboor_kernel<<<blocks, THREADS>>>(
        d_input,
        d_knots,
        d_coeffs,
        d_output,
        batch);
  };


  // ------------------------------------------------------------------
  // Warmup
  // ------------------------------------------------------------------

  for (int i = 0;
       i < warmup;
       ++i) {

    launch();
  }

  CUDA_CHECK(
      cudaGetLastError());

  CUDA_CHECK(
      cudaDeviceSynchronize());


  // ------------------------------------------------------------------
  // CUDA-event timing
  // ------------------------------------------------------------------

  cudaEvent_t start;
  cudaEvent_t stop;

  CUDA_CHECK(
      cudaEventCreate(&start));

  CUDA_CHECK(
      cudaEventCreate(&stop));


  CUDA_CHECK(
      cudaEventRecord(start));


  for (int i = 0;
       i < iterations;
       ++i) {

    launch();
  }


  CUDA_CHECK(
      cudaEventRecord(stop));

  CUDA_CHECK(
      cudaEventSynchronize(stop));


  float elapsed_ms = 0.0f;

  CUDA_CHECK(
      cudaEventElapsedTime(
          &elapsed_ms,
          start,
          stop));


  const double kernel_us =
      1000.0
      * static_cast<double>(elapsed_ms)
      / static_cast<double>(iterations);


  // ------------------------------------------------------------------
  // Validation
  // ------------------------------------------------------------------

  std::vector<float>
      actual(expected_output);


  CUDA_CHECK(
      cudaMemcpy(
          actual.data(),
          d_output,
          expected_output * sizeof(float),
          cudaMemcpyDeviceToHost));


  double max_abs = 0.0;
  double sum_sq = 0.0;
  size_t max_index = 0;


  for (size_t i = 0;
       i < expected_output;
       ++i) {

    const double err =
        std::abs(
            static_cast<double>(actual[i])
            - static_cast<double>(expected[i]));


    if (err > max_abs) {

      max_abs = err;
      max_index = i;
    }


    sum_sq += err * err;
  }


  const double rms =
      std::sqrt(
          sum_sq
          / static_cast<double>(
                expected_output));


  const double samples_per_second =
      static_cast<double>(batch)
      / (kernel_us * 1.0e-6);


  const double outputs_per_second =
      static_cast<double>(total_outputs)
      / (kernel_us * 1.0e-6);


  // ------------------------------------------------------------------
  // Results
  // ------------------------------------------------------------------

  std::printf(
      "batch              : %lld\n",
      batch);

  std::printf(
      "din                : %d\n",
      DIN);

  std::printf(
      "dout               : %d\n",
      DOUT);

  std::printf(
      "degree             : %d\n",
      DEGREE);

  std::printf(
      "coefficients/edge  : %d\n",
      NCOEFF);

  std::printf(
      "knots/input        : %d\n",
      NKNOTS);

  std::printf(
      "knot spacing       : %.9f\n",
      H);

  std::printf(
      "grid blocks        : %d\n",
      blocks);

  std::printf(
      "block              : %d\n",
      THREADS);

  std::printf(
      "warmup             : %d\n",
      warmup);

  std::printf(
      "iterations         : %d\n",
      iterations);

  std::printf(
      "kernel time us     : %.9f\n",
      kernel_us);

  std::printf(
      "samples/s          : %.9e\n",
      samples_per_second);

  std::printf(
      "outputs/s          : %.9e\n",
      outputs_per_second);

  std::printf(
      "actual[0]          : %.11g\n",
      actual[0]);

  std::printf(
      "expected[0]        : %.11g\n",
      expected[0]);

  std::printf(
      "max abs error      : %.9e\n",
      max_abs);

  std::printf(
      "RMS error          : %.9e\n",
      rms);

  std::printf(
      "max error index    : %zu\n",
      max_index);


  std::printf(
      "RESULT,%lld,%.9f,%.9e,%.9e,%.9e,%.9e\n",
      batch,
      kernel_us,
      samples_per_second,
      outputs_per_second,
      max_abs,
      rms);


  // ------------------------------------------------------------------
  // Cleanup
  // ------------------------------------------------------------------

  CUDA_CHECK(
      cudaEventDestroy(start));

  CUDA_CHECK(
      cudaEventDestroy(stop));

  CUDA_CHECK(
      cudaFree(d_input));

  CUDA_CHECK(
      cudaFree(d_knots));

  CUDA_CHECK(
      cudaFree(d_coeffs));

  CUDA_CHECK(
      cudaFree(d_output));

  return EXIT_SUCCESS;
}
