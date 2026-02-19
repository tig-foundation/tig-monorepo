// IMPORTANT NOTES:
// 1. You can import any libraries available in nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04
//    Example:
//    #include <curand_kernel.h>
//    #include <stdint.h>
//    #include <math.h>
//    #include <float.h>
//
// 2. If you launch a kernel with multiple blocks, any writes should be to non-overlapping parts of the memory
//    Example:
//    arr[blockIdx.x] = 1; // This IS deterministic
//    arr[0] = 1; // This is NOT deterministic
//
// 3. Any kernel available in <challenge>.cu will be available here
//
// 4. If you need to use random numbers, you can use the CURAND library and seed it with challenge.seed.
//    Example rust:
//    let d_seed = stream.memcpy_stod(seed)?;
//    stream
//       .launch_builder(&my_kernel)
//       .arg(&d_seed)
//       ...
//
//    Example cuda:
//    extern "C" __global__ void my_kernel(
//        const uint8_t *seed,
//        ...
//    ) {
//        curandState state;
//        curand_init(((uint64_t *)(seed))[0], 0, 0, &state);
//        ...
//    }
