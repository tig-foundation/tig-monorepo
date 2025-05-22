#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

#define FUELUSAGE_OK        (0)
#define FUELUSAGE_EXCEEDED  (1)
#define CHECK_FUEL_LIMIT    asm("trap;")

__device__ u_int64_t gbl_SIGNATURE = 0; // Run-time signature
__device__ u_int64_t gbl_FUELUSAGE = 0; // Fuel usage
__device__ u_int64_t gbl_ERRORSTAT = 0; // Error status -- set to non-zero if fuel runs out

extern "C" __global__ void initialize_kernel(
    u_int64_t signature
)
{
    gbl_ERRORSTAT     = FUELUSAGE_OK;
    gbl_SIGNATURE     = signature; 
    gbl_FUELUSAGE     = 0;

    return;
}

extern "C" __global__ void finalize_kernel(
    u_int64_t *fuelusage_ptr,   // RETURNED: (64-bit) Fuel usage
    u_int64_t *signature_ptr,   // RETURNED: (64-bit) Run-time signature
    u_int64_t *errorstat_ptr    // RETURNED: (64-bit) Error status
)
{
    fuelusage_ptr[0] = gbl_FUELUSAGE; // RETURNED: (64-bit) Fuel usage
    signature_ptr[0] = gbl_SIGNATURE; // RETURNED: (64-bit) Run-time signature
    errorstat_ptr[0] = gbl_ERRORSTAT; // RETURNED: (64-bit) Error status -- set to non-zero if fuel runs out

    return;
}
