#include <cuda.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                           \
  do {                                                             \
    CUresult _err = (call);                                        \
    if (_err != CUDA_SUCCESS) {                                    \
      const char *name = nullptr;                                  \
      const char *msg = nullptr;                                   \
      cuGetErrorName(_err, &name);                                 \
      cuGetErrorString(_err, &msg);                                \
      std::fprintf(stderr,                                         \
                   "CUDA error at %s:%d: %s: %s\n",                \
                   __FILE__, __LINE__,                              \
                   name ? name : "?",                              \
                   msg ? msg : "?");                               \
      std::exit(1);                                                \
    }                                                              \
  } while (0)


static std::vector<unsigned char>
read_bytes(const std::string &path) {
  std::ifstream f(path, std::ios::binary);

  if (!f) {
    std::cerr << "Could not open " << path << "\n";
    std::exit(1);
  }

  f.seekg(0, std::ios::end);
  size_t n = static_cast<size_t>(f.tellg());
  f.seekg(0, std::ios::beg);

  std::vector<unsigned char> data(n);
  f.read(reinterpret_cast<char *>(data.data()), n);

  if (!f) {
    std::cerr << "Could not read " << path << "\n";
    std::exit(1);
  }

  return data;
}


static std::vector<float>
read_floats(const std::string &path,
            size_t expected_count) {
  std::ifstream f(path, std::ios::binary);

  if (!f) {
    std::cerr << "Could not open " << path << "\n";
    std::exit(1);
  }

  f.seekg(0, std::ios::end);
  size_t bytes = static_cast<size_t>(f.tellg());
  f.seekg(0, std::ios::beg);

  const size_t expected_bytes =
      expected_count * sizeof(float);

  if (bytes != expected_bytes) {
    std::cerr
        << "Unexpected size for " << path
        << ": got " << bytes
        << " bytes, expected "
        << expected_bytes << "\n";
    std::exit(1);
  }

  std::vector<float> data(expected_count);

  f.read(
      reinterpret_cast<char *>(data.data()),
      bytes);

  if (!f) {
    std::cerr << "Could not read " << path << "\n";
    std::exit(1);
  }

  return data;
}


// ================================================================
// Runtime representation of one compiled KAN layer.
//
// Static layer properties:
//   Din  = 2
//   Dout = 8
//   5 cubic pieces
//
// Bounds and coefficients are persistent GPU-resident layer
// parameters.
//
// Input and output are supplied as device pointers by the caller.
// ================================================================

class CompiledKANLayer {
public:
  CompiledKANLayer(CUfunction function,
                   CUstream stream,
                   int64_t batch,
                   CUdeviceptr bounds,
                   CUdeviceptr coeffs)
      : function_(function),
        stream_(stream),
        batch_(batch),
        bounds_(bounds),
        coeffs_(coeffs) {

    total_outputs_ = batch_ * DOUT;

    blocks_ =
        static_cast<unsigned int>(
            (total_outputs_ + THREADS - 1) / THREADS);
  }


  // Prevent compiler from folding this into the benchmark loop.
#if defined(__GNUC__) || defined(__clang__)
  __attribute__((noinline))
#endif
  CUresult invoke(CUdeviceptr input,
                  CUdeviceptr output) {

    // These correspond exactly to the 13 arguments in the
    // compiler-generated bare-pointer GPU ABI.

    int64_t arg_block_size = THREADS;
    int64_t arg_total_outputs = total_outputs_;
    int64_t arg_dout = DOUT;

    int64_t arg_one = 1;
    int64_t arg_zero = 0;
    int64_t arg_two = 2;
    int64_t arg_three = 3;
    int64_t arg_four = 4;

    float arg_zero_f = 0.0f;

    // Local aliases are intentional: this models a runtime layer
    // invocation receiving device-buffer operands.
    CUdeviceptr arg_input = input;
    CUdeviceptr arg_bounds = bounds_;
    CUdeviceptr arg_coeffs = coeffs_;
    CUdeviceptr arg_output = output;

    void *kernel_args[] = {
        &arg_block_size,
        &arg_total_outputs,
        &arg_dout,

        &arg_input,
        &arg_bounds,

        &arg_one,
        &arg_zero,
        &arg_two,
        &arg_three,
        &arg_four,

        &arg_coeffs,

        &arg_zero_f,

        &arg_output
    };

    return cuLaunchKernel(
        function_,
        blocks_, 1, 1,
        THREADS, 1, 1,
        0,
        stream_,
        kernel_args,
        nullptr);
  }


  unsigned int blocks() const {
    return blocks_;
  }


private:
  static constexpr int64_t DOUT = 8;
  static constexpr unsigned int THREADS = 256;

  CUfunction function_;
  CUstream stream_;

  int64_t batch_;
  int64_t total_outputs_;

  unsigned int blocks_;

  CUdeviceptr bounds_;
  CUdeviceptr coeffs_;
};


int main(int argc, char **argv) {
  if (argc < 4 || argc > 6) {
    std::cerr
        << "Usage: " << argv[0]
        << " <fatbin> <data-dir> <batch-size>"
        << " [warmup] [iterations]\n";
    return 1;
  }

  const std::string fatbin_path = argv[1];
  const std::string data_dir = argv[2];

  const int64_t B = std::stoll(argv[3]);

  const int warmup =
      argc >= 5 ? std::stoi(argv[4]) : 100;

  const int iterations =
      argc >= 6 ? std::stoi(argv[5]) : 1000;

  if (B <= 0 || warmup < 0 || iterations <= 0) {
    std::cerr << "Invalid benchmark arguments\n";
    return 1;
  }

  constexpr int64_t DIN = 2;
  constexpr int64_t DOUT = 8;
  constexpr int64_t PIECES = 5;
  constexpr int64_t NCOEFF = 4;

  // ------------------------------------------------------------
  // Read data.
  // ------------------------------------------------------------

  auto fatbin = read_bytes(fatbin_path);

  auto h_input = read_floats(
      data_dir + "/input.bin",
      B * DIN);

  auto h_bounds = read_floats(
      data_dir + "/bounds.bin",
      DIN * (PIECES + 1));

  auto h_coeffs = read_floats(
      data_dir + "/coeffs.bin",
      DOUT * DIN * PIECES * NCOEFF);

  auto h_expected = read_floats(
      data_dir + "/expected.bin",
      B * DOUT);

  std::vector<float> h_output(B * DOUT);

  // ------------------------------------------------------------
  // CUDA initialization.
  // ------------------------------------------------------------

  CUDA_CHECK(cuInit(0));

  CUdevice device;
  CUDA_CHECK(cuDeviceGet(&device, 0));

  char device_name[256];

  CUDA_CHECK(
      cuDeviceGetName(
          device_name,
          sizeof(device_name),
          device));

  int cc_major = 0;
  int cc_minor = 0;

  CUDA_CHECK(
      cuDeviceGetAttribute(
          &cc_major,
          CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
          device));

  CUDA_CHECK(
      cuDeviceGetAttribute(
          &cc_minor,
          CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
          device));

  CUcontext context;

  CUctxCreateParams ctx_params{};

  CUDA_CHECK(
      cuCtxCreate(
          &context,
          &ctx_params,
          0,
          device));

  CUstream stream;

  CUDA_CHECK(
      cuStreamCreate(
          &stream,
          CU_STREAM_DEFAULT));

  // ------------------------------------------------------------
  // Load compiler-generated layer module once.
  // ------------------------------------------------------------

  CUmodule module;

  CUDA_CHECK(
      cuModuleLoadData(
          &module,
          fatbin.data()));

  CUfunction function;

  CUDA_CHECK(
      cuModuleGetFunction(
          &function,
          module,
          "kanlib_piecewise_benchmark_kernel"));

  // ------------------------------------------------------------
  // Persistent layer storage.
  // ------------------------------------------------------------

  CUdeviceptr d_input = 0;
  CUdeviceptr d_bounds = 0;
  CUdeviceptr d_coeffs = 0;
  CUdeviceptr d_output = 0;

  CUDA_CHECK(
      cuMemAlloc(
          &d_input,
          B * DIN * sizeof(float)));

  CUDA_CHECK(
      cuMemAlloc(
          &d_bounds,
          DIN * (PIECES + 1) * sizeof(float)));

  CUDA_CHECK(
      cuMemAlloc(
          &d_coeffs,
          DOUT * DIN * PIECES *
              NCOEFF * sizeof(float)));

  CUDA_CHECK(
      cuMemAlloc(
          &d_output,
          B * DOUT * sizeof(float)));

  CUDA_CHECK(
      cuMemcpyHtoDAsync(
          d_input,
          h_input.data(),
          B * DIN * sizeof(float),
          stream));

  CUDA_CHECK(
      cuMemcpyHtoDAsync(
          d_bounds,
          h_bounds.data(),
          DIN * (PIECES + 1) * sizeof(float),
          stream));

  CUDA_CHECK(
      cuMemcpyHtoDAsync(
          d_coeffs,
          h_coeffs.data(),
          DOUT * DIN * PIECES *
              NCOEFF * sizeof(float),
          stream));

  CUDA_CHECK(cuStreamSynchronize(stream));

  // ------------------------------------------------------------
  // Instantiate compiled layer.
  // ------------------------------------------------------------

  CompiledKANLayer layer(
      function,
      stream,
      B,
      d_bounds,
      d_coeffs);

  // ------------------------------------------------------------
  // Warmup.
  // ------------------------------------------------------------

  for (int i = 0; i < warmup; ++i) {
    CUDA_CHECK(
        layer.invoke(
            d_input,
            d_output));
  }

  CUDA_CHECK(cuStreamSynchronize(stream));

  // ============================================================
  // Measurement 1:
  //
  // GPU elapsed time for a compiled layer invocation.
  //
  // This is the measurement comparable to CUDA-event timing of
  // the KANLib layer.
  // ============================================================

  CUevent start_event;
  CUevent stop_event;

  CUDA_CHECK(
      cuEventCreate(
          &start_event,
          CU_EVENT_DEFAULT));

  CUDA_CHECK(
      cuEventCreate(
          &stop_event,
          CU_EVENT_DEFAULT));

  CUDA_CHECK(
      cuEventRecord(
          start_event,
          stream));

  for (int i = 0; i < iterations; ++i) {
    CUDA_CHECK(
        layer.invoke(
            d_input,
            d_output));
  }

  CUDA_CHECK(
      cuEventRecord(
          stop_event,
          stream));

  CUDA_CHECK(
      cuEventSynchronize(
          stop_event));

  float gpu_elapsed_ms = 0.0f;

  CUDA_CHECK(
      cuEventElapsedTime(
          &gpu_elapsed_ms,
          start_event,
          stop_event));

  const double gpu_layer_us =
      static_cast<double>(gpu_elapsed_ms) *
      1000.0 /
      iterations;

  // ============================================================
  // Measurement 2:
  //
  // Host-side submission cost.
  //
  // Make sure the stream is idle before measuring. We measure
  // only the CPU work necessary to invoke the compiled layer.
  // GPU completion is synchronized after the CPU timer stops.
  // ============================================================

  CUDA_CHECK(cuStreamSynchronize(stream));

  auto host_start =
      std::chrono::steady_clock::now();

  for (int i = 0; i < iterations; ++i) {
    CUDA_CHECK(
        layer.invoke(
            d_input,
            d_output));
  }

  auto host_stop =
      std::chrono::steady_clock::now();

  // Finish queued GPU work after the host timer has stopped.
  CUDA_CHECK(cuStreamSynchronize(stream));

  const double host_total_us =
      std::chrono::duration<double, std::micro>(
          host_stop - host_start)
          .count();

  const double host_submit_us =
      host_total_us /
      static_cast<double>(iterations);

  // ------------------------------------------------------------
  // Correctness.
  // ------------------------------------------------------------

  CUDA_CHECK(
      cuMemcpyDtoH(
          h_output.data(),
          d_output,
          B * DOUT * sizeof(float)));

  double max_abs = 0.0;
  double sum_sq = 0.0;

  size_t max_index = 0;

  for (size_t i = 0; i < h_output.size(); ++i) {
    const double diff =
        std::abs(
            static_cast<double>(h_output[i]) -
            static_cast<double>(h_expected[i]));

    if (diff > max_abs) {
      max_abs = diff;
      max_index = i;
    }

    sum_sq += diff * diff;
  }

  const double rms =
      std::sqrt(
          sum_sq /
          static_cast<double>(h_output.size()));

  const double samples_per_second =
      static_cast<double>(B) /
      (gpu_layer_us * 1.0e-6);

  const double outputs_per_second =
      static_cast<double>(B * DOUT) /
      (gpu_layer_us * 1.0e-6);

  // ------------------------------------------------------------
  // Report.
  // ------------------------------------------------------------

  std::printf(
      "GPU               : %s sm_%d%d\n",
      device_name,
      cc_major,
      cc_minor);

  std::printf(
      "batch             : %lld\n",
      static_cast<long long>(B));

  std::printf(
      "grid              : %u\n",
      layer.blocks());

  std::printf(
      "block             : 256\n");

  std::printf(
      "warmup            : %d\n",
      warmup);

  std::printf(
      "iterations        : %d\n",
      iterations);

  std::printf("\n");

  std::printf(
      "GPU layer us      : %.6f\n",
      gpu_layer_us);

  std::printf(
      "host submit us    : %.6f\n",
      host_submit_us);

  std::printf(
      "samples/s         : %.6e\n",
      samples_per_second);

  std::printf(
      "outputs/s         : %.6e\n",
      outputs_per_second);

  std::printf("\n");

  std::printf(
      "actual[0]         : %.9g\n",
      h_output[0]);

  std::printf(
      "expected[0]       : %.9g\n",
      h_expected[0]);

  std::printf(
      "max abs error     : %.9e\n",
      max_abs);

  std::printf(
      "RMS error         : %.9e\n",
      rms);

  std::printf(
      "max error index   : %zu\n",
      max_index);

  // Machine-readable output.
  std::printf(
      "RESULT,%lld,%.9f,%.9f,%.9e,%.9e,%.9e,%.9e\n",
      static_cast<long long>(B),
      gpu_layer_us,
      host_submit_us,
      samples_per_second,
      outputs_per_second,
      max_abs,
      rms);

  // ------------------------------------------------------------
  // Cleanup.
  // ------------------------------------------------------------

  CUDA_CHECK(cuEventDestroy(start_event));
  CUDA_CHECK(cuEventDestroy(stop_event));

  CUDA_CHECK(cuMemFree(d_input));
  CUDA_CHECK(cuMemFree(d_bounds));
  CUDA_CHECK(cuMemFree(d_coeffs));
  CUDA_CHECK(cuMemFree(d_output));

  CUDA_CHECK(cuModuleUnload(module));

  CUDA_CHECK(cuStreamDestroy(stream));
  CUDA_CHECK(cuCtxDestroy(context));

  return 0;
}
