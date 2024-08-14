
/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip/hip_runtime.h>
#include <iostream>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define SAMPLE_VERSION "HIP-Examples-Application-v1.0"
#define SUCCESS 0
#define FAILURE 1

using namespace std;

typedef struct access_t {
  uint64_t thread;
  uint64_t access_time;
  bool is_write;
} Access;

constexpr uint64_t ACCESS_SIZE = (1 << 16);

__device__ Access accesses[ACCESS_SIZE];
__device__ uint64_t global_clock;
__device__ uint32_t write_lock;
__device__ bool race;

extern "C" __device__ void __tsan_func_entry(void *pc) {
  global_clock = 1;
  printf("Entered a function\n");
  race = false;
  write_lock = 0;
}

extern "C" __device__ void __tsan_write4(void *mem) {
  bool leaveLoop = false;
  uint64_t access_index = (((uint64_t) mem >> 2) & (ACCESS_SIZE - 1));
  uint64_t index = blockDim.x * blockIdx.x + threadIdx.x;
  while (!leaveLoop) {
    if (atomicExch(&write_lock, 1) == 0) {
      // critical section
      printf("Writing 4 bytes\n");
      if (accesses[access_index].access_time == global_clock &&
          accesses[access_index].thread != index) {
        race = true;
        const char *op;
        if (accesses[access_index].is_write) {
          op = "Written";
        } else {
          op = "Read";
        }
        uint64_t tidx = threadIdx.x;
        uint64_t bidx = blockIdx.x;
        printf("Data race at %p: \t\t%s by (%lu, %lu), \t\tWritten by (%lu, %lu)\n", mem, op, accesses[access_index].thread / blockDim.x, accesses[access_index].thread % blockDim.x, bidx, tidx);
      }
      accesses[access_index].access_time = global_clock;
      accesses[access_index].thread = index;
      accesses[access_index].is_write = true;
      leaveLoop = true;
      atomicExch(&write_lock, 0);
    }
  }
}

extern "C" __device__ void __tsan_read4(void *mem) {
  bool leaveLoop = false;
  uint64_t access_index = (((uint64_t) mem >> 2) & (ACCESS_SIZE - 1));
  uint64_t index = blockDim.x * blockIdx.x + threadIdx.x;
  while (!leaveLoop) {
    if (atomicExch(&write_lock, 1) == 0) {
      // critical section
      if (accesses[access_index].access_time == global_clock &&
          accesses[access_index].thread != index &&
          accesses[access_index].is_write) {
        race = true;
        uint64_t tidx = threadIdx.x;
        uint64_t bidx = blockIdx.x;
        printf("Data race at %p: \t\tWritten by (%lu, %lu), \t\tRead by (%lu, %lu)\n", mem, accesses[access_index].thread / blockDim.x, accesses[access_index].thread % blockDim.x, bidx, tidx);
      }
      accesses[access_index].access_time = global_clock;
      accesses[access_index].thread = index;
      accesses[access_index].is_write = false;
      leaveLoop = true;
      atomicExch(&write_lock, 0);
    }
  }
}

extern "C" __device__ void __tsan_func_exit() {
  printf("Exited a function\n");
  if (race) {
    printf("Data race detected!\n");
  }
}

// __device__ void __tsan_init() { return; }

// __device__ void __tsan_func_entry(void *pc) { return; }

// __device__ void __tsan_write4(void *mem) { return; }

// __device__ void __tsan_func_exit() { return; }

__global__ void helloworld(int *a) { *a += 5; }

#define HIP_CHECK(condition)                                                   \
  do {                                                                         \
    hipError_t error = condition;                                              \
    if (error != hipSuccess) {                                                 \
      std::cout << "HIP error: " << error << " line: " << __LINE__             \
                << std::endl;                                                  \
      exit(error);                                                             \
    }                                                                          \
  } while (false);

int main(int argc, char *argv[]) {
  hipDeviceProp_t devProp;
  HIP_CHECK(hipGetDeviceProperties(&devProp, 0));
  cout << " System minor " << devProp.minor << endl;
  cout << " System major " << devProp.major << endl;
  cout << " agent prop name " << devProp.name << endl;

  int *input = new int;
  *input = 5;

  int *d_inp;
  HIP_CHECK(hipMalloc(&d_inp, sizeof(int)));
  HIP_CHECK(hipMemcpy(d_inp, input, sizeof(int), hipMemcpyHostToDevice));

  //  int *d_accesses;
  //  HIP_CHECK(hipMalloc(&d_accesses, sizeof(Access)));
  //  HIP_CHECK(hipMemset(d_accesses, 0, sizeof(Access)));
  //  HIP_CHECK(hipMemcpyToSymbol("accesses", &d_accesses, sizeof(Access *)));

  // HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(gg), some_global, 5 * sizeof(int)));
  helloworld<<<dim3(1), dim3(18), 0, 0>>>(d_inp);
  HIP_CHECK(hipDeviceSynchronize());

  cout << "\noutput: " << *d_inp << endl;

  std::cout << "Passed!\n";
  return SUCCESS;
}
