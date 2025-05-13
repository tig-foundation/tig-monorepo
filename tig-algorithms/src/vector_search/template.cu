/*!
Copyright [year copyright work created] [name of copyright owner]

Identity of Submitter [name of person or entity that submits the Work to TIG]

UAI [UAI (if applicable)]

Licensed under the TIG Inbound Game License v2.0 or (at your option) any later
version (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/

// REMOVE BELOW SECTION IF UNUSED
/*
REFERENCES AND ACKNOWLEDGMENTS

This implementation is based on or inspired by existing work. Citations and
acknowledgments below:

1. Academic Papers:
   - [Author(s), "Paper Title", DOI (if available)]

2. Code References:
   - [Author(s), URL]

3. Other:
   - [Author(s), Details]

*/
// License must be the same as the rust code

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
//    arr[0] = 1; // This is NOT deterministic as multiple blocks are writing to the same location
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