#include <cstdio>

extern "C" void print_f32(float x) {
  std::printf("%.8f\n", x);
}
