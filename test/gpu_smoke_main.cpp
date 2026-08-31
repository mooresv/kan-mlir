#include <cstdint>
#include <cstdio>
#include <cstdlib>

// Function generated from gpu_smoke.mlir.
extern "C" void gpu_smoke(
    void *allocated,
    void *aligned,
    int64_t offset,
    int64_t size,
    int64_t stride);

// MLIR CUDA runtime wrappers.
extern "C" void *mgpuMemAlloc(
    uint64_t sizeBytes,
    void *stream,
    bool isHostShared);

extern "C" void mgpuMemFree(
    void *ptr,
    void *stream);

int main() {
    constexpr int64_t N = 1024;
    constexpr size_t bytes = N * sizeof(float);

    // isHostShared=true -> cuMemAllocManaged().
    float *data = static_cast<float *>(
        mgpuMemAlloc(bytes, nullptr, true));

    if (!data) {
        std::fprintf(stderr, "mgpuMemAlloc failed\n");
        return 1;
    }

    // Initialize from the CPU.
    for (int64_t i = 0; i < N; ++i)
        data[i] = static_cast<float>(i);

    // Rank-1 memref ABI:
    //
    //   allocated ptr
    //   aligned ptr
    //   offset
    //   size
    //   stride
    //
    gpu_smoke(
        data,       // allocated
        data,       // aligned
        0,          // offset
        N,          // size
        1);         // stride

    // gpu_smoke contains mgpuStreamSynchronize(), so the kernel has
    // completed before we inspect managed memory here.

    bool ok = true;

    for (int64_t i = 0; i < N; ++i) {
        const float expected = static_cast<float>(i) + 1.0f;

        if (data[i] != expected) {
            std::fprintf(stderr,
                         "Mismatch at index %ld: got %.8f expected %.8f\n",
                         static_cast<long>(i),
                         data[i],
                         expected);
            ok = false;
            break;
        }
    }

    if (ok) {
        std::printf("PASS: GPU kernel correctly incremented all %ld elements\n",
                    static_cast<long>(N));

        std::printf("First 8 values:");
        for (int i = 0; i < 8; ++i)
            std::printf(" %.1f", data[i]);
        std::printf("\n");

        std::printf("Last 8 values:");
        for (int64_t i = N - 8; i < N; ++i)
            std::printf(" %.1f", data[i]);
        std::printf("\n");
    }

    mgpuMemFree(data, nullptr);

    return ok ? 0 : 1;
}
