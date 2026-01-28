#include <cuda_runtime.h>
#include <float.h>

#define MAX_FLOAT FLT_MAX

__device__ __forceinline__ float euclidean_distance_high_bounded(const float* __restrict__ a, const float* __restrict__ b, int dims, float limit) {
    float sum=0.0f;
    int i = 0;

    if (dims >= 160) {
        float d0=a[0]-b[0], d1=a[1]-b[1], d2=a[2]-b[2], d3=a[3]-b[3];
        float d4=a[4]-b[4], d5=a[5]-b[5], d6=a[6]-b[6], d7=a[7]-b[7];
        float d8=a[8]-b[8], d9=a[9]-b[9], d10=a[10]-b[10], d11=a[11]-b[11];
        float d12=a[12]-b[12], d13=a[13]-b[13], d14=a[14]-b[14], d15=a[15]-b[15];
        float d16=a[16]-b[16], d17=a[17]-b[17], d18=a[18]-b[18], d19=a[19]-b[19];
        float d20=a[20]-b[20], d21=a[21]-b[21], d22=a[22]-b[22], d23=a[23]-b[23];
        float d24=a[24]-b[24], d25=a[25]-b[25], d26=a[26]-b[26], d27=a[27]-b[27];
        float d28=a[28]-b[28], d29=a[29]-b[29], d30=a[30]-b[30], d31=a[31]-b[31];
        float d32=a[32]-b[32], d33=a[33]-b[33], d34=a[34]-b[34], d35=a[35]-b[35];
        float d36=a[36]-b[36], d37=a[37]-b[37], d38=a[38]-b[38], d39=a[39]-b[39];
        float d40=a[40]-b[40], d41=a[41]-b[41], d42=a[42]-b[42], d43=a[43]-b[43];
        float d44=a[44]-b[44], d45=a[45]-b[45], d46=a[46]-b[46], d47=a[47]-b[47];
        float d48=a[48]-b[48], d49=a[49]-b[49], d50=a[50]-b[50], d51=a[51]-b[51];
        float d52=a[52]-b[52], d53=a[53]-b[53], d54=a[54]-b[54], d55=a[55]-b[55];
        float d56=a[56]-b[56], d57=a[57]-b[57], d58=a[58]-b[58], d59=a[59]-b[59];
        float d60=a[60]-b[60], d61=a[61]-b[61], d62=a[62]-b[62], d63=a[63]-b[63];
        float d64=a[64]-b[64], d65=a[65]-b[65], d66=a[66]-b[66], d67=a[67]-b[67];
        float d68=a[68]-b[68], d69=a[69]-b[69], d70=a[70]-b[70], d71=a[71]-b[71];
        float d72=a[72]-b[72], d73=a[73]-b[73], d74=a[74]-b[74], d75=a[75]-b[75];
        float d76=a[76]-b[76], d77=a[77]-b[77], d78=a[78]-b[78], d79=a[79]-b[79];
        float d80=a[80]-b[80], d81=a[81]-b[81], d82=a[82]-b[82], d83=a[83]-b[83];
        float d84=a[84]-b[84], d85=a[85]-b[85], d86=a[86]-b[86], d87=a[87]-b[87];
        float d88=a[88]-b[88], d89=a[89]-b[89], d90=a[90]-b[90], d91=a[91]-b[91];
        float d92=a[92]-b[92], d93=a[93]-b[93], d94=a[94]-b[94], d95=a[95]-b[95];
        float d96=a[96]-b[96], d97=a[97]-b[97], d98=a[98]-b[98], d99=a[99]-b[99];
        float d100=a[100]-b[100], d101=a[101]-b[101], d102=a[102]-b[102], d103=a[103]-b[103];
        float d104=a[104]-b[104], d105=a[105]-b[105], d106=a[106]-b[106], d107=a[107]-b[107];
        float d108=a[108]-b[108], d109=a[109]-b[109], d110=a[110]-b[110], d111=a[111]-b[111];
        float d112=a[112]-b[112], d113=a[113]-b[113], d114=a[114]-b[114], d115=a[115]-b[115];
        float d116=a[116]-b[116], d117=a[117]-b[117], d118=a[118]-b[118], d119=a[119]-b[119];
        float d120=a[120]-b[120], d121=a[121]-b[121], d122=a[122]-b[122], d123=a[123]-b[123];
        float d124=a[124]-b[124], d125=a[125]-b[125], d126=a[126]-b[126], d127=a[127]-b[127];
        float d128=a[128]-b[128], d129=a[129]-b[129], d130=a[130]-b[130], d131=a[131]-b[131];
        float d132=a[132]-b[132], d133=a[133]-b[133], d134=a[134]-b[134], d135=a[135]-b[135];
        float d136=a[136]-b[136], d137=a[137]-b[137], d138=a[138]-b[138], d139=a[139]-b[139];
        float d140=a[140]-b[140], d141=a[141]-b[141], d142=a[142]-b[142], d143=a[143]-b[143];
        float d144=a[144]-b[144], d145=a[145]-b[145], d146=a[146]-b[146], d147=a[147]-b[147];
        float d148=a[148]-b[148], d149=a[149]-b[149], d150=a[150]-b[150], d151=a[151]-b[151];
        float d152=a[152]-b[152], d153=a[153]-b[153], d154=a[154]-b[154], d155=a[155]-b[155];
        float d156=a[156]-b[156], d157=a[157]-b[157], d158=a[158]-b[158], d159=a[159]-b[159];
        
        sum = d0*d0+d1*d1+d2*d2+d3*d3+d4*d4+d5*d5+d6*d6+d7*d7+d8*d8+d9*d9+d10*d10+d11*d11+d12*d12+d13*d13+d14*d14+d15*d15
            +d16*d16+d17*d17+d18*d18+d19*d19+d20*d20+d21*d21+d22*d22+d23*d23+d24*d24+d25*d25+d26*d26+d27*d27+d28*d28+d29*d29+d30*d30+d31*d31
            +d32*d32+d33*d33+d34*d34+d35*d35+d36*d36+d37*d37+d38*d38+d39*d39+d40*d40+d41*d41+d42*d42+d43*d43+d44*d44+d45*d45+d46*d46+d47*d47
            +d48*d48+d49*d49+d50*d50+d51*d51+d52*d52+d53*d53+d54*d54+d55*d55+d56*d56+d57*d57+d58*d58+d59*d59+d60*d60+d61*d61+d62*d62+d63*d63
            +d64*d64+d65*d65+d66*d66+d67*d67+d68*d68+d69*d69+d70*d70+d71*d71+d72*d72+d73*d73+d74*d74+d75*d75+d76*d76+d77*d77+d78*d78+d79*d79
            +d80*d80+d81*d81+d82*d82+d83*d83+d84*d84+d85*d85+d86*d86+d87*d87+d88*d88+d89*d89+d90*d90+d91*d91+d92*d92+d93*d93+d94*d94+d95*d95
            +d96*d96+d97*d97+d98*d98+d99*d99+d100*d100+d101*d101+d102*d102+d103*d103+d104*d104+d105*d105+d106*d106+d107*d107+d108*d108+d109*d109+d110*d110+d111*d111
            +d112*d112+d113*d113+d114*d114+d115*d115+d116*d116+d117*d117+d118*d118+d119*d119+d120*d120+d121*d121+d122*d122+d123*d123+d124*d124+d125*d125+d126*d126+d127*d127
            +d128*d128+d129*d129+d130*d130+d131*d131+d132*d132+d133*d133+d134*d134+d135*d135+d136*d136+d137*d137+d138*d138+d139*d139+d140*d140+d141*d141+d142*d142+d143*d143
            +d144*d144+d145*d145+d146*d146+d147*d147+d148*d148+d149*d149+d150*d150+d151*d151+d152*d152+d153*d153+d154*d154+d155*d155+d156*d156+d157*d157+d158*d158+d159*d159;
        if (sum > limit - 20.0f) return limit + 1.0f;
        i = 160;
    }

    for (;i<dims-31;i+=32){
        float d0=a[i]-b[i], d1=a[i+1]-b[i+1], d2=a[i+2]-b[i+2], d3=a[i+3]-b[i+3];
        float d4=a[i+4]-b[i+4], d5=a[i+5]-b[i+5], d6=a[i+6]-b[i+6], d7=a[i+7]-b[i+7];
        float d8=a[i+8]-b[i+8], d9=a[i+9]-b[i+9], d10=a[i+10]-b[i+10], d11=a[i+11]-b[i+11];
        float d12=a[i+12]-b[i+12], d13=a[i+13]-b[i+13], d14=a[i+14]-b[i+14], d15=a[i+15]-b[i+15];
        sum += d0*d0+d1*d1+d2*d2+d3*d3+d4*d4+d5*d5+d6*d6+d7*d7+d8*d8+d9*d9+d10*d10+d11*d11+d12*d12+d13*d13+d14*d14+d15*d15;
        if (sum > limit) return sum;

        float d16=a[i+16]-b[i+16], d17=a[i+17]-b[i+17], d18=a[i+18]-b[i+18], d19=a[i+19]-b[i+19];
        float d20=a[i+20]-b[i+20], d21=a[i+21]-b[i+21], d22=a[i+22]-b[i+22], d23=a[i+23]-b[i+23];
        float d24=a[i+24]-b[i+24], d25=a[i+25]-b[i+25], d26=a[i+26]-b[i+26], d27=a[i+27]-b[i+27];
        float d28=a[i+28]-b[i+28], d29=a[i+29]-b[i+29], d30=a[i+30]-b[i+30], d31=a[i+31]-b[i+31];
        sum += d16*d16+d17*d17+d18*d18+d19*d19+d20*d20+d21*d21+d22*d22+d23*d23+d24*d24+d25*d25+d26*d26+d27*d27+d28*d28+d29*d29+d30*d30+d31*d31;
        if (sum > limit) return sum;
    }
    for (; i < dims - 15; i += 16) {
        float d0=a[i]-b[i], d1=a[i+1]-b[i+1], d2=a[i+2]-b[i+2], d3=a[i+3]-b[i+3];
        float d4=a[i+4]-b[i+4], d5=a[i+5]-b[i+5], d6=a[i+6]-b[i+6], d7=a[i+7]-b[i+7];
        float d8=a[i+8]-b[i+8], d9=a[i+9]-b[i+9], d10=a[i+10]-b[i+10], d11=a[i+11]-b[i+11];
        float d12=a[i+12]-b[i+12], d13=a[i+13]-b[i+13], d14=a[i+14]-b[i+14], d15=a[i+15]-b[i+15];
        sum += d0*d0+d1*d1+d2*d2+d3*d3+d4*d4+d5*d5+d6*d6+d7*d7+d8*d8+d9*d9+d10*d10+d11*d11+d12*d12+d13*d13+d14*d14+d15*d15;
        if (sum > limit) return sum;
    }
    for (; i < dims - 7; i += 8) {
        float d0=a[i]-b[i], d1=a[i+1]-b[i+1], d2=a[i+2]-b[i+2], d3=a[i+3]-b[i+3];
        float d4=a[i+4]-b[i+4], d5=a[i+5]-b[i+5], d6=a[i+6]-b[i+6], d7=a[i+7]-b[i+7];
        sum += d0*d0+d1*d1+d2*d2+d3*d3+d4*d4+d5*d5+d6*d6+d7*d7;
        if (sum > limit) return sum;
    }
    for (; i < dims - 3; i += 4) {
        float d0=a[i]-b[i], d1=a[i+1]-b[i+1], d2=a[i+2]-b[i+2], d3=a[i+3]-b[i+3];
        sum += d0*d0+d1*d1+d2*d2+d3*d3;
        if (sum > limit) return sum;
    }
    for (; i < dims; i++) {
        float diff=a[i]-b[i];
        sum += diff*diff;
        if (sum > limit) return sum;
    }
    return sum;
}

extern "C" __global__ __launch_bounds__(128) void batched_search(
    const float* __restrict__ query_vectors,
    const float* __restrict__ database_vectors,
    int* __restrict__ results,
    float* __restrict__ best_dists,
    int num_queries,
    int vector_dims,
    int batch_start,
    int batch_count,
    int is_first_batch
) {
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_idx >= num_queries) return;

    const size_t stride = (size_t)vector_dims;
    const float* query = query_vectors + (size_t)query_idx * stride;

    float min_dist = MAX_FLOAT;
    int best_idx = 0;
   
    int i0 = 0;
    if (is_first_batch) {
        if (batch_count <= 0) return;

        best_idx = batch_start;
        const float* first = database_vectors + (size_t)batch_start * stride;
        min_dist = euclidean_distance_high_bounded(query, first, vector_dims, MAX_FLOAT);
        i0 = 1;
    } else {
        min_dist = best_dists[query_idx];
        best_idx = results[query_idx];
    }

    for (int i = i0; i < batch_count; i++) {
        int vec_idx = batch_start + i;
        const float* db_vec = database_vectors + (size_t)vec_idx * stride;

        float dist = euclidean_distance_high_bounded(query, db_vec, vector_dims, min_dist);
        if (dist < min_dist) {
            min_dist = dist;
            best_idx = vec_idx;
        }
    }

    best_dists[query_idx] = min_dist;
    results[query_idx] = best_idx;
}