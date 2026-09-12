#include <cuda_runtime.h>
static __device__ __forceinline__ int list_grouped_add(int *counts,int pair){
    unsigned active=__activemask(),same=__match_any_sync(active,pair);int lane=threadIdx.x&31,first=__ffs(same)-1,base=0;
    if(lane==first)base=atomicAdd(counts+pair,__popc(same));
    base=__shfl_sync(active,base,first);return base+__popc(same&((1u<<lane)-1u));
}
extern "C" __global__ void gpt_count_swap_lists(int nn,int np,const int *partition,const int *gains,int *counts){
    for(int n=blockIdx.x*blockDim.x+threadIdx.x;n<nn;n+=blockDim.x*gridDim.x){
        int src=__ldg(partition+n);
        for(int k=0;k<3;++k){int value=__ldg(gains+n*3+k),target=value&0xffff;
            if((unsigned)src<(unsigned)np && value!=0 && target<np && target!=src)list_grouped_add(counts,src*np+target);
        }
    }
}
extern "C" __global__ void gpt_scatter_swap_lists(int nn,int np,const int *partition,const int *gains,const int *offsets,int *cursor,unsigned *ids){
    for(int n=blockIdx.x*blockDim.x+threadIdx.x;n<nn;n+=blockDim.x*gridDim.x){
        int src=__ldg(partition+n);
        for(int k=0;k<3;++k){int value=__ldg(gains+n*3+k),target=value&0xffff;
            if((unsigned)src<(unsigned)np && value!=0 && target<np && target!=src){int pair=src*np+target,pos=list_grouped_add(cursor,pair);ids[offsets[pair]+pos]=(unsigned)(n*3+k);}
        }
    }
}
extern "C" __global__ void gpt_sort_swap_lists(int np,const int *counts,const int *offsets,const unsigned *ids,const int *gains,unsigned long long *packed){
    __shared__ unsigned work[4096];int pair=blockIdx.x,tid=threadIdx.x,nb=blockDim.x;
    if(pair>=np*np)return;int n=__ldg(counts+pair),start=__ldg(offsets+pair);
    if(n<=0 || n>4096)return;
    int power=1;while(power<n)power<<=1;
    if(power<=nb){
        unsigned value=tid<n?__ldg(ids+start+tid):~0u;
        for(int size=2;size<=power;size<<=1)for(int dist=size>>1;dist;dist>>=1){
            unsigned other;
            if(dist<32)other=__shfl_xor_sync(0xffffffffu,value,dist);
            else{work[tid]=value;__syncthreads();other=work[tid^dist];__syncthreads();}
            bool small=((tid&dist)==0)==((tid&size)==0);value=small?min(value,other):max(value,other);
        }
        if(tid<n){int gain=__ldg(gains+value)>>16;packed[start+tid]=((unsigned long long)(unsigned)gain<<32)|(value/3);}
    }else{
        for(int i=tid;i<power;i+=nb)work[i]=i<n?__ldg(ids+start+i):~0u;
        __syncthreads();
        for(int size=2;size<=power;size<<=1)for(int dist=size>>1;dist;dist>>=1){
            for(int i=tid;i<power;i+=nb){int peer=i^dist;if(peer>i){unsigned a=work[i],b=work[peer];if((i&size)==0?a>b:a<b){work[i]=b;work[peer]=a;}}}
            __syncthreads();
        }
        for(int i=tid;i<n;i+=nb){unsigned value=work[i];int gain=__ldg(gains+value)>>16;packed[start+i]=((unsigned long long)(unsigned)gain<<32)|(value/3);}
    }
}
