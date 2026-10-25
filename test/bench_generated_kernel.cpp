#include <cuda.h>

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

int main(int argc, char **argv) {
  if (argc < 4 || argc > 10 || (argc > 6 && argc != 10)) {
    std::cerr
        << "Usage: " << argv[0]
        << " <fatbin> <data-dir> <batch-size>"
        << " [warmup] [iterations]"
        << " [din dout pieces ncoeff]\n";
    return 1;
  }

  const std::string fatbin_path = argv[1];
  const std::string data_dir = argv[2];

  const int64_t B = std::stoll(argv[3]);

  const int warmup =
      argc >= 5 ? std::stoi(argv[4]) : 100;

  const int iterations =
      argc >= 6 ? std::stoi(argv[5]) : 1000;

  // Defaults preserve the original 2->8, 5-piece benchmark.
  const int64_t DIN = argc == 10 ? std::stoll(argv[6]) : 2;
  const int64_t DOUT = argc == 10 ? std::stoll(argv[7]) : 8;
  const int64_t PIECES = argc == 10 ? std::stoll(argv[8]) : 5;
  const int64_t NCOEFF = argc == 10 ? std::stoll(argv[9]) : 4;

  if ((DOUT & (DOUT - 1)) != 0) {
    std::cerr << "DOUT must be a power of two\n";
    return 1;
  }

  if (B <= 0 || warmup < 0 || iterations <= 0 ||
      DIN <= 0 || DOUT <= 0 || PIECES <= 0 || NCOEFF <= 0) {
    std::cerr << "Invalid benchmark arguments\n";
    return 1;
  }

  constexpr int threads = 256;

  const int64_t total_outputs = B * DOUT;

  const int blocks =
      static_cast<int>(
          (total_outputs + threads - 1) / threads);

  // ----------------------------------------------------------
  // Host data.
  // ----------------------------------------------------------

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

  // ----------------------------------------------------------
  // CUDA setup.
  // ----------------------------------------------------------

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

  // ----------------------------------------------------------
  // Persistent device buffers.
  // ----------------------------------------------------------

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

  CUstream stream;

  CUDA_CHECK(
      cuStreamCreate(
          &stream,
          CU_STREAM_DEFAULT));

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

  // ----------------------------------------------------------
  // 17-argument bare-pointer ABI for the boundary-search kernel.
  // ----------------------------------------------------------

  int64_t arg_block_size = 256;
  int64_t arg_total_outputs = total_outputs;

  // DOUT must be a power of two.
  int64_t arg_dout_shift = 0;
  for (int64_t n = DOUT; n > 1; n >>= 1)
    ++arg_dout_shift;

  int64_t arg_dout_mask = DOUT - 1;

  int64_t arg_zero = 0;
  int64_t arg_one = 1;
  int64_t arg_two = 2;
  int64_t arg_three = 3;
  int64_t arg_din = DIN;
  int64_t arg_pieces = PIECES;

  float arg_zero_f = 0.0f;

  void *kernel_args[] = {
      &arg_block_size,     //  0: 256
      &arg_total_outputs,  //  1: B * DOUT
      &arg_dout_shift,     //  2: log2(DOUT)
      &arg_dout_mask,      //  3: DOUT - 1

      &d_input,            //  4: input pointer
      &d_bounds,           //  5: boundaries pointer

      &arg_one,            //  6: 1
      &arg_pieces,         //  7: PIECES
      &arg_zero,           //  8: 0

      &d_coeffs,           //  9: coefficients pointer

      &arg_three,          // 10: 3
      &arg_two,            // 11: 2
      &arg_one,            // 12: 1
      &arg_zero,           // 13: 0
      &arg_din,            // 14: DIN

      &arg_zero_f,         // 15: 0.0f
      &d_output            // 16: output pointer
  };

  auto launch = [&]() {
    CUDA_CHECK(
        cuLaunchKernel(
            function,
            blocks, 1, 1,
            threads, 1, 1,
            0,
            stream,
            kernel_args,
            nullptr));
  };

  // ----------------------------------------------------------
  // Warmup.
  // ----------------------------------------------------------

  for (int i = 0; i < warmup; ++i)
    launch();

  CUDA_CHECK(cuStreamSynchronize(stream));

  // ----------------------------------------------------------
  // Timed launches.
  // ----------------------------------------------------------

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

  for (int i = 0; i < iterations; ++i)
    launch();

  CUDA_CHECK(
      cuEventRecord(
          stop_event,
          stream));

  CUDA_CHECK(
      cuEventSynchronize(
          stop_event));

  float elapsed_ms = 0.0f;

  CUDA_CHECK(
      cuEventElapsedTime(
          &elapsed_ms,
          start_event,
          stop_event));

  const double time_us =
      static_cast<double>(elapsed_ms) *
      1000.0 /
      iterations;

  const double samples_per_second =
      static_cast<double>(B) /
      (time_us * 1.0e-6);

  const double outputs_per_second =
      static_cast<double>(B * DOUT) /
      (time_us * 1.0e-6);

  // ----------------------------------------------------------
  // Validate.
  // ----------------------------------------------------------

  CUDA_CHECK(
      cuMemcpyDtoH(
          h_output.data(),
          d_output,
          B * DOUT * sizeof(float)));

  double max_abs = 0.0;
  double sum_sq = 0.0;
  size_t max_index = 0;

  for (size_t i = 0; i < h_output.size(); ++i) {
      if (!std::isfinite(h_output[i]) ||
          !std::isfinite(h_expected[i])) {
        std::cerr << "Nonfinite value at output index " << i
                  << ": actual=" << h_output[i]
                  << ", expected=" << h_expected[i] << "\n";
        return 1;
      }

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

  // Human-readable output.
  std::printf("GPU               : %s sm_%d%d\n",
              device_name, cc_major, cc_minor);
  std::printf("batch             : %lld\n",
              static_cast<long long>(B));
  std::printf("grid              : %d\n", blocks);
  std::printf("block             : %d\n", threads);
  std::printf("warmup            : %d\n", warmup);
  std::printf("iterations        : %d\n", iterations);
  std::printf("elapsed total ms  : %.6f\n", elapsed_ms);
  std::printf("kernel time us    : %.6f\n", time_us);
  std::printf("samples/s         : %.6e\n",
              samples_per_second);
  std::printf("outputs/s         : %.6e\n",
              outputs_per_second);
  std::printf("actual[0]         : %.9g\n",
              h_output[0]);
  std::printf("expected[0]       : %.9g\n",
              h_expected[0]);
  std::printf("max abs error     : %.9e\n",
              max_abs);
  std::printf("RMS error         : %.9e\n",
              rms);
  std::printf("max error index   : %zu\n",
              max_index);

  // Machine-readable line for collection script.
  std::printf(
      "RESULT,%lld,%.9f,%.9e,%.9e,%.9e,%.9e\n",
      static_cast<long long>(B),
      time_us,
      samples_per_second,
      outputs_per_second,
      max_abs,
      rms);

  CUDA_CHECK(cuEventDestroy(start_event));
  CUDA_CHECK(cuEventDestroy(stop_event));
  CUDA_CHECK(cuStreamDestroy(stream));

  CUDA_CHECK(cuMemFree(d_input));
  CUDA_CHECK(cuMemFree(d_bounds));
  CUDA_CHECK(cuMemFree(d_coeffs));
  CUDA_CHECK(cuMemFree(d_output));

  CUDA_CHECK(cuModuleUnload(module));
  CUDA_CHECK(cuCtxDestroy(context));

  return 0;
}
