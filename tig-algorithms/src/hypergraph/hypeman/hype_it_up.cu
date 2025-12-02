#include <stdint.h>
#include <cuda_runtime.h>

// --- HELPER: Fast Randomness ---
__device__ __forceinline__ uint32_t pcg_hash(uint64_t state) {
    uint64_t oldstate = state;
    state = oldstate * 6364136223846793005ULL + (oldstate | 1);
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

// --- KERNEL 1: INIT ---
extern "C" __global__ void init_solution_kernel(
    const int num_nodes,
    const int num_parts,
    const uint64_t seed,
    int *partition,
    int *nodes_in_part
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_nodes) {
        // Use high entropy hash to scatter nodes
        int part = pcg_hash(seed + idx * 0x9E3779B9) % num_parts;
        partition[idx] = part;
        atomicAdd(&nodes_in_part[part], 1);
    }
}

// --- KERNEL 2: ANNEALING MOVE COMPUTE ---
// This calculates gains but includes "Temperature" to accept uphill moves
extern "C" __global__ void compute_moves_annealing(
    const int num_nodes,
    const int num_parts,
    const int max_part_size,
    const int *node_hyperedges,
    const int *node_offsets,
    const int *hyperedge_nodes,
    const int *hyperedge_offsets,
    const int *partition,
    const int *nodes_in_part,
    int *move_nodes,
    int *move_parts,
    int *move_gains,
    int *num_valid_moves,
    const int round,
    const float temperature, // Controls randomness
    unsigned long long *global_fallback_buffer
) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes) return;

    move_nodes[node] = -1; 
    int current_part = partition[node];
    
    // Soft Balance Constraint
    if (nodes_in_part[current_part] <= 1) return;

    int start = node_offsets[node];
    int end = node_offsets[node + 1];
    int degree = end - start;

    // --- REGISTER BLOCKING (The Speed Secret) ---
    const int REG_LIMIT = 32; 
    unsigned long long local_flags[REG_LIMIT];
    unsigned long long* edge_flags;

    if (degree <= REG_LIMIT) {
        edge_flags = local_flags;
    } else {
        edge_flags = &global_fallback_buffer[node * 3000]; 
    }

    // 1. Build Connectivity
    for (int j = 0; j < degree; j++) {
        edge_flags[j] = 0; 
        int edge_idx = node_hyperedges[start + j];
        int h_start = hyperedge_offsets[edge_idx];
        int h_end = hyperedge_offsets[edge_idx + 1];

        for (int k = h_start; k < h_end; k++) {
            int neighbor = hyperedge_nodes[k];
            if (neighbor != node) {
                int n_part = partition[neighbor];
                edge_flags[j] |= (1ULL << (n_part & 63)); 
            }
        }
    }

    // 2. Current Cost
    int current_cost = 0;
    for (int j = 0; j < degree; j++) {
        unsigned long long mask = edge_flags[j] | (1ULL << (current_part & 63));
        int connectivity = __popcll(mask);
        if (connectivity > 1) current_cost += (connectivity - 1);
    }

    // 3. Evaluate Targets with ANNEALING
    int best_eval = -999999;
    int best_target = -1;
    
    uint32_t rng = pcg_hash(node + round * 19937);

    int t1 = (node + round) % num_parts;
    int t2 = (node + round + 1) % num_parts;
    int t3 = rng % num_parts;
    int trials[] = {t1, t2, t3};

    for (int t = 0; t < 3; t++) {
        int target = trials[t];
        if (target == current_part) continue;

        int target_size = nodes_in_part[target];
        int balance_penalty = 0;
        if (target_size >= max_part_size) {
            balance_penalty = 1000; 
        } else if (target_size >= max_part_size - 1) {
            balance_penalty = 2;    
        }

        int new_cost = 0;
        for (int j = 0; j < degree; j++) {
            unsigned long long mask = edge_flags[j] | (1ULL << (target & 63));
            int connectivity = __popcll(mask);
            if (connectivity > 1) new_cost += (connectivity - 1);
        }

        int gain = current_cost - new_cost;
        int eval = gain - balance_penalty;

        // ANNEALING: Add noise based on temperature
        float noise = 0.0f;
        if (temperature > 0.001f) {
            float r = (float)(rng % 1000) / 1000.0f; 
            noise = r * temperature; 
        }

        if ((float)eval + noise > (float)best_eval) {
            best_eval = eval;
            best_target = target;
        }
        rng = pcg_hash(rng);
    }

    // 4. Commit
    if (best_target != -1 && best_eval > -100) { 
        move_nodes[node] = node;
        move_parts[node] = best_target;
        move_gains[node] = best_eval;
        atomicAdd(num_valid_moves, 1);
    }
}

// --- KERNEL 3: APPLY MOVES ---
extern "C" __global__ void apply_moves_kernel(
    const int num_nodes,
    const int max_part_size,
    const int *move_nodes,
    const int *move_parts,
    int *partition,
    int *nodes_in_part
) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes) return;

    int target_node = move_nodes[node];
    if (target_node != -1) {
        int target_part = move_parts[node];
        int current_part = partition[node];

        if (nodes_in_part[target_part] < max_part_size + 2 && 
            nodes_in_part[current_part] > 1) {
            
            partition[node] = target_part;
            atomicSub(&nodes_in_part[current_part], 1);
            atomicAdd(&nodes_in_part[target_part], 1);
        }
    }
}