// Present because build_ptx requires a .cu file. Not launched.
extern "C" __global__ void even_keel_marker(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && data != 0) {
        data[i] = data[i];
    }
}
