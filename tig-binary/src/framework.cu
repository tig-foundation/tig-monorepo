#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

//------------------------------------------------------------------------------
//------------ Required Framework Code Begins Here -- DO NOT CHANGE ------------
//------------------------------------------------------------------------------

//
// MACROS for signature and to check fuel usage
// (Python script will modify this to check fuel usage register.)
//
#define FUELUSAGE_OK        (0)
#define FUELUSAGE_EXCEEDED  (1)
#define CHECK_FUEL_LIMIT    asm("trap;")

//
// Required globals -- DO NOT CHANGE
//
__device__ unsigned long long gbl_SIGNATURE = 0; // Run-time signature
__device__ unsigned long long gbl_SIGNATURE_MOD = 0; // Run-time signature modifier
__device__ unsigned long long gbl_FUELUSAGE = 0; // Fuel usage
__device__ unsigned long long gbl_FUELUSAGE_MAX = 0; // Fuel usage maximum allowed
__device__ u_int64_t gbl_ERRORSTAT = 0; // Error status -- set to non-zero if fuel runs out

//
// Initialize -- DO NOT CHANGE
// 
extern "C" __global__ void initialize_kernel(
    u_int64_t fuelusage_max,   // (64-bit) Initialize Max allowed fuel usage
    u_int64_t signature_mod    // (64-bit) Final modifier for Runtime Signature
)
{
    gbl_ERRORSTAT     = FUELUSAGE_OK;
    gbl_SIGNATURE     = 0;
    gbl_SIGNATURE_MOD = signature_mod;
    gbl_FUELUSAGE     = 0;
    gbl_FUELUSAGE_MAX = fuelusage_max;

    //printf("Set fuel usage maximum to: %llu\n",gbl_FUELUSAGE_MAX);

    return;
} // End of initialize_kernel code

//
// Finalize -- DO NOT CHANGE
// 
extern "C" __global__ void finalize_kernel(

    // Special fuel usage and signature arguments (DO NOT CHANGE)
    u_int64_t *fuelusage_ptr,   // RETURNED: (64-bit) Fuel usage
    u_int64_t *signature_ptr,   // RETURNED: (64-bit) Run-time signature
    u_int64_t *errorstat_ptr    // RETURNED: (64-bit) Error status

)
{
    // Modify the runtime signature with the gbl_SIGNATURE_MOD value
    gbl_SIGNATURE ^= gbl_SIGNATURE_MOD;

    //printf("__finalize_kernel__:  fuel usage: %lu    signature: 0x%16.16lx \n",gbl_FUELUSAGE,gbl_SIGNATURE);
    fuelusage_ptr[0] = gbl_FUELUSAGE; // RETURNED: (64-bit) Fuel usage
    signature_ptr[0] = gbl_SIGNATURE; // RETURNED: (64-bit) Run-time signature
    errorstat_ptr[0] = gbl_ERRORSTAT; // RETURNED: (64-bit) Error status -- set to non-zero if fuel runs out

    return;
} // End of finalize_kernel code

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------