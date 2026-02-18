/*!Copyright 2025 Rootz

Identity of Submitter Rootz

UAI null

Licensed under the TIG Inbound Game License v2.0 or (at your option) any later
version (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/
#include <cuda_runtime.h>
#include <float.h>

#define MAX_FLOAT 3.402823466e+38F

__device__ __forceinline__ float euclidean_distance(const float* __restrict__ a, const float* __restrict__ b, int dims) {
    float sum = 0.0f;
    float c = 0.0f;
    int i;
    
    for (i = 0; i < dims - 15; i += 16) {
        float d0=a[i]-b[i], d1=a[i+1]-b[i+1], d2=a[i+2]-b[i+2], d3=a[i+3]-b[i+3];
        float d4=a[i+4]-b[i+4], d5=a[i+5]-b[i+5], d6=a[i+6]-b[i+6], d7=a[i+7]-b[i+7];
        float d8=a[i+8]-b[i+8], d9=a[i+9]-b[i+9], d10=a[i+10]-b[i+10], d11=a[i+11]-b[i+11];
        float d12=a[i+12]-b[i+12], d13=a[i+13]-b[i+13], d14=a[i+14]-b[i+14], d15=a[i+15]-b[i+15];

        float s0 = d0*d0 + d1*d1 + d2*d2 + d3*d3;
        float s1 = d4*d4 + d5*d5 + d6*d6 + d7*d7;
        float s2 = d8*d8 + d9*d9 + d10*d10 + d11*d11;
        float s3 = d12*d12 + d13*d13 + d14*d14 + d15*d15;

        float partial = s0 + s1 + s2 + s3;
        float y = partial - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    for (; i < dims - 7; i += 8) {
        float d0=a[i]-b[i], d1=a[i+1]-b[i+1], d2=a[i+2]-b[i+2], d3=a[i+3]-b[i+3];
        float d4=a[i+4]-b[i+4], d5=a[i+5]-b[i+5], d6=a[i+6]-b[i+6], d7=a[i+7]-b[i+7];
        float v,y,t;
        v=d0*d0; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d1*d1; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d2*d2; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d3*d3; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d4*d4; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d5*d5; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d6*d6; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d7*d7; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
    }

    for (; i < dims - 3; i += 4) {
        float d0=a[i]-b[i], d1=a[i+1]-b[i+1], d2=a[i+2]-b[i+2], d3=a[i+3]-b[i+3];
        float v,y,t;
        v=d0*d0; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d1*d1; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d2*d2; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d3*d3; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
    }

    for (; i < dims; i++) {
        float diff = a[i] - b[i];
        float squared = diff * diff;
        float y = squared - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return sum;
}

__device__ __forceinline__ float euclidean_distance_high(const float* __restrict__ a, const float* __restrict__ b, int dims) {
    float sum = 0.0f;
    float c = 0.0f;
    int i;

    for (i = 0; i < dims - 31; i += 32) {
        float d0=a[i]-b[i], d1=a[i+1]-b[i+1], d2=a[i+2]-b[i+2], d3=a[i+3]-b[i+3];
        float d4=a[i+4]-b[i+4], d5=a[i+5]-b[i+5], d6=a[i+6]-b[i+6], d7=a[i+7]-b[i+7];
        float d8=a[i+8]-b[i+8], d9=a[i+9]-b[i+9], d10=a[i+10]-b[i+10], d11=a[i+11]-b[i+11];
        float d12=a[i+12]-b[i+12], d13=a[i+13]-b[i+13], d14=a[i+14]-b[i+14], d15=a[i+15]-b[i+15];
        float d16=a[i+16]-b[i+16], d17=a[i+17]-b[i+17], d18=a[i+18]-b[i+18], d19=a[i+19]-b[i+19];
        float d20=a[i+20]-b[i+20], d21=a[i+21]-b[i+21], d22=a[i+22]-b[i+22], d23=a[i+23]-b[i+23];
        float d24=a[i+24]-b[i+24], d25=a[i+25]-b[i+25], d26=a[i+26]-b[i+26], d27=a[i+27]-b[i+27];
        float d28=a[i+28]-b[i+28], d29=a[i+29]-b[i+29], d30=a[i+30]-b[i+30], d31=a[i+31]-b[i+31];
        float v,y,t;
        v=d0*d0; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d1*d1; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d2*d2; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d3*d3; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d4*d4; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d5*d5; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d6*d6; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d7*d7; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d8*d8; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d9*d9; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d10*d10; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d11*d11; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d12*d12; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d13*d13; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d14*d14; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d15*d15; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d16*d16; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d17*d17; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d18*d18; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d19*d19; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d20*d20; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d21*d21; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d22*d22; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d23*d23; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d24*d24; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d25*d25; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d26*d26; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d27*d27; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d28*d28; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d29*d29; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d30*d30; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d31*d31; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
    }

    for (; i < dims - 15; i += 16) {
        float d0=a[i]-b[i], d1=a[i+1]-b[i+1], d2=a[i+2]-b[i+2], d3=a[i+3]-b[i+3];
        float d4=a[i+4]-b[i+4], d5=a[i+5]-b[i+5], d6=a[i+6]-b[i+6], d7=a[i+7]-b[i+7];
        float d8=a[i+8]-b[i+8], d9=a[i+9]-b[i+9], d10=a[i+10]-b[i+10], d11=a[i+11]-b[i+11];
        float d12=a[i+12]-b[i+12], d13=a[i+13]-b[i+13], d14=a[i+14]-b[i+14], d15=a[i+15]-b[i+15];

        float v,y,t;
        v=d0*d0; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d1*d1; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d2*d2; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d3*d3; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d4*d4; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d5*d5; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d6*d6; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d7*d7; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d8*d8; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d9*d9; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d10*d10; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d11*d11; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d12*d12; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d13*d13; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d14*d14; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d15*d15; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
    }

    for (; i < dims - 7; i += 8) {
        float d0=a[i]-b[i], d1=a[i+1]-b[i+1], d2=a[i+2]-b[i+2], d3=a[i+3]-b[i+3];
        float d4=a[i+4]-b[i+4], d5=a[i+5]-b[i+5], d6=a[i+6]-b[i+6], d7=a[i+7]-b[i+7];
        float v,y,t;
        v=d0*d0; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d1*d1; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d2*d2; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d3*d3; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d4*d4; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d5*d5; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d6*d6; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d7*d7; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
    }

    for (; i < dims - 3; i += 4) {
        float d0=a[i]-b[i], d1=a[i+1]-b[i+1], d2=a[i+2]-b[i+2], d3=a[i+3]-b[i+3];
        float v,y,t;
        v=d0*d0; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d1*d1; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d2*d2; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d3*d3; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
    }

    for (; i < dims; i++) {
        float diff = a[i] - b[i];
        float squared = diff * diff;
        float y = squared - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return sum;
}

__device__ __forceinline__ float euclidean_distance_bounded(const float* __restrict__ a, const float* __restrict__ b, int dims, float limit) {
    float sum = 0.0f;
    float c = 0.0f;
    float margin = fmaxf(1e-6f, 1.0e-4f * (1.0f + limit));
    int i;
    for (i = 0; i < dims - 15; i += 16) {
        float d0=a[i]-b[i], d1=a[i+1]-b[i+1], d2=a[i+2]-b[i+2], d3=a[i+3]-b[i+3];
        float d4=a[i+4]-b[i+4], d5=a[i+5]-b[i+5], d6=a[i+6]-b[i+6], d7=a[i+7]-b[i+7];
        float d8=a[i+8]-b[i+8], d9=a[i+9]-b[i+9], d10=a[i+10]-b[i+10], d11=a[i+11]-b[i+11];
        float d12=a[i+12]-b[i+12], d13=a[i+13]-b[i+13], d14=a[i+14]-b[i+14], d15=a[i+15]-b[i+15];
        float s0=d0*d0+d1*d1+d2*d2+d3*d3;
        float s1=d4*d4+d5*d5+d6*d6+d7*d7;
        float s2=d8*d8+d9*d9+d10*d10+d11*d11;
        float s3=d12*d12+d13*d13+d14*d14+d15*d15;
        float partial=s0+s1+s2+s3;
        float y=partial-c;
        float t=sum+y;
        c=(t-sum)-y;
        sum=t;
        if (sum > limit + margin) return sum;
    }
    for (; i < dims - 7; i += 8) {
        float d0=a[i]-b[i], d1=a[i+1]-b[i+1], d2=a[i+2]-b[i+2], d3=a[i+3]-b[i+3];
        float d4=a[i+4]-b[i+4], d5=a[i+5]-b[i+5], d6=a[i+6]-b[i+6], d7=a[i+7]-b[i+7];
        float v,y,t;
        v=d0*d0; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d1*d1; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d2*d2; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d3*d3; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d4*d4; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d5*d5; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d6*d6; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d7*d7; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        if (sum > limit + margin) return sum;
    }
    for (; i < dims - 3; i += 4) {
        float d0=a[i]-b[i], d1=a[i+1]-b[i+1], d2=a[i+2]-b[i+2], d3=a[i+3]-b[i+3];
        float v,y,t;
        v=d0*d0; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d1*d1; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d2*d2; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d3*d3; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        if (sum > limit + margin) return sum;
    }
    for (; i < dims; i++) {
        float diff=a[i]-b[i];
        float squared=diff*diff;
        float y=squared-c;
        float t=sum+y;
        c=(t-sum)-y;
        sum=t;
        if (sum > limit + margin) return sum;
    }
    return sum;
}

__device__ __forceinline__ float euclidean_distance_high_bounded(const float* __restrict__ a, const float* __restrict__ b, int dims, float limit) {
    float sum=0.0f;
    float c=0.0f;
    float margin = fmaxf(1e-6f, 1.0e-4f * (1.0f + limit));
    int i;
    for (i=0;i<dims-31;i+=32){
        float d0=a[i]-b[i], d1=a[i+1]-b[i+1], d2=a[i+2]-b[i+2], d3=a[i+3]-b[i+3];
        float d4=a[i+4]-b[i+4], d5=a[i+5]-b[i+5], d6=a[i+6]-b[i+6], d7=a[i+7]-b[i+7];
        float d8=a[i+8]-b[i+8], d9=a[i+9]-b[i+9], d10=a[i+10]-b[i+10], d11=a[i+11]-b[i+11];
        float d12=a[i+12]-b[i+12], d13=a[i+13]-b[i+13], d14=a[i+14]-b[i+14], d15=a[i+15]-b[i+15];
        float d16=a[i+16]-b[i+16], d17=a[i+17]-b[i+17], d18=a[i+18]-b[i+18], d19=a[i+19]-b[i+19];
        float d20=a[i+20]-b[i+20], d21=a[i+21]-b[i+21], d22=a[i+22]-b[i+22], d23=a[i+23]-b[i+23];
        float d24=a[i+24]-b[i+24], d25=a[i+25]-b[i+25], d26=a[i+26]-b[i+26], d27=a[i+27]-b[i+27];
        float d28=a[i+28]-b[i+28], d29=a[i+29]-b[i+29], d30=a[i+30]-b[i+30], d31=a[i+31]-b[i+31];
        float v,y,t;
        v=d0*d0; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d1*d1; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d2*d2; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d3*d3; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d4*d4; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d5*d5; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d6*d6; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d7*d7; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d8*d8; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d9*d9; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d10*d10; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d11*d11; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d12*d12; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d13*d13; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d14*d14; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d15*d15; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d16*d16; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d17*d17; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d18*d18; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d19*d19; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d20*d20; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d21*d21; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d22*d22; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d23*d23; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d24*d24; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d25*d25; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d26*d26; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d27*d27; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d28*d28; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d29*d29; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d30*d30; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d31*d31; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        if (sum > limit + margin) return sum;
    }
    for (; i < dims - 15; i += 16) {
        float d0=a[i]-b[i], d1=a[i+1]-b[i+1], d2=a[i+2]-b[i+2], d3=a[i+3]-b[i+3];
        float d4=a[i+4]-b[i+4], d5=a[i+5]-b[i+5], d6=a[i+6]-b[i+6], d7=a[i+7]-b[i+7];
        float d8=a[i+8]-b[i+8], d9=a[i+9]-b[i+9], d10=a[i+10]-b[i+10], d11=a[i+11]-b[i+11];
        float d12=a[i+12]-b[i+12], d13=a[i+13]-b[i+13], d14=a[i+14]-b[i+14], d15=a[i+15]-b[i+15];

        float v,y,t;
        v=d0*d0; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d1*d1; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d2*d2; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d3*d3; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d4*d4; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d5*d5; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d6*d6; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d7*d7; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d8*d8; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d9*d9; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d10*d10; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d11*d11; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d12*d12; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d13*d13; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d14*d14; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d15*d15; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        if (sum > limit + margin) return sum;
    }
    for (; i < dims - 7; i += 8) {
        float d0=a[i]-b[i], d1=a[i+1]-b[i+1], d2=a[i+2]-b[i+2], d3=a[i+3]-b[i+3];
        float d4=a[i+4]-b[i+4], d5=a[i+5]-b[i+5], d6=a[i+6]-b[i+6], d7=a[i+7]-b[i+7];
        float v,y,t;
        v=d0*d0; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d1*d1; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d2*d2; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d3*d3; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d4*d4; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d5*d5; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d6*d6; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d7*d7; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        if (sum > limit + margin) return sum;
    }
    for (; i < dims - 3; i += 4) {
        float d0=a[i]-b[i], d1=a[i+1]-b[i+1], d2=a[i+2]-b[i+2], d3=a[i+3]-b[i+3];
        float v,y,t;
        v=d0*d0; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d1*d1; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d2*d2; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        v=d3*d3; y=v-c; t=sum+y; c=(t-sum)-y; sum=t;
        if (sum > limit + margin) return sum;
    }
    for (; i < dims; i++) {
        float diff=a[i]-b[i];
        float squared=diff*diff;
        float y=squared-c;
        float t=sum+y;
        c=(t-sum)-y;
        sum=t;
        if (sum > limit + margin) return sum;
    }
    return sum;
}

__device__ __forceinline__ float euclidean_distance_precise_bounded(const float* __restrict__ a, const float* __restrict__ b, int dims, float limit) {
    double acc = 0.0;
    double lim = (double)limit;
    for (int i = 0; i < dims; i++) {
        double d = (double)a[i] - (double)b[i];
        acc += d * d;
        if (acc > lim) return (float)acc;
    }
    return (float)acc;
}

extern "C" __global__ void deterministic_clustering(
    const float* __restrict__ database_vectors,
    float* __restrict__ cluster_centers,
    int* __restrict__ cluster_assignments,
    int* __restrict__ cluster_sizes,
    int database_size,
    int vector_dims,
    int num_clusters,
    int num_queries
) {
    int cluster_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (cluster_idx >= num_clusters) return;

    long long seed_idx = ((long long)cluster_idx * 982451653LL + 1566083941LL) % (long long)database_size;
    int stride = max(1, database_size / (num_clusters * 37));
    long long start_idx = seed_idx;

    for (int d = tid; d < vector_dims; d += blockDim.x) {
        float acc = 0.0f;
        long long idx = start_idx;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int pos = (int)(idx % (long long)database_size);
            acc += database_vectors[pos * vector_dims + d];
            idx += stride;
        }
        cluster_centers[cluster_idx * vector_dims + d] = acc * 0.25f;
    }

    if (tid == 0) {
        cluster_sizes[cluster_idx] = 0;
    }
}

extern "C" __global__ void assign_clusters(
    const float* __restrict__ database_vectors,
    const float* __restrict__ cluster_centers,
    int* __restrict__ cluster_assignments,
    int* __restrict__ cluster_sizes,
    int database_size,
    int vector_dims,
    int num_clusters,
    int num_queries
) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const bool use_high = (num_queries > 3000) || (vector_dims >= 700);
    if (thread_id < database_size) {
        int vec_idx = thread_id;
        const float* vector = database_vectors + vec_idx * vector_dims;
        float min_dist = MAX_FLOAT;
        int best_cluster = 0;
        for (int c = 0; c < num_clusters; c++) {
            const float* c_center = cluster_centers + c * vector_dims;
            float dist = use_high ? euclidean_distance_high(vector, c_center, vector_dims)
                                  : euclidean_distance(vector, c_center, vector_dims);
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = c;
            }
        }
        cluster_assignments[vec_idx] = best_cluster;
    }
}

extern "C" __global__ void exclusive_scan_sizes(
    const int* cluster_sizes,
    int* cluster_offsets,
    int* write_offsets,
    int num_clusters
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int acc = 0;
        for (int i = 0; i < num_clusters; i++) {
            cluster_offsets[i] = acc;
            write_offsets[i] = acc;
            acc += cluster_sizes[i];
        }
    }
}

extern "C" __global__ void build_cluster_index(
    const int* cluster_assignments,
    int* write_offsets,
    int* cluster_indices,
    int database_size
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int vec_idx = 0; vec_idx < database_size; vec_idx++) {
            int cluster = cluster_assignments[vec_idx];
            int pos = write_offsets[cluster];
            cluster_indices[pos] = vec_idx;
            write_offsets[cluster]++;
        }
    }
}

extern "C" __global__ void count_block_cluster_sizes(
    const int* cluster_assignments,
    int* block_counts,
    int database_size,
    int num_clusters
) {
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int block = blockIdx.x;
    int base = block * blockDim.x;
    int vec_idx = base + tid;
    __shared__ int s_len;
    if (tid == 0) {
        int rem = database_size - base;
        s_len = rem > blockDim.x ? blockDim.x : (rem > 0 ? rem : 0);
    }
    __syncthreads();
    if (s_len == 0) {
        if (tid == 0) {
            for (int c = 0; c < num_clusters; c++) {
                block_counts[block * num_clusters + c] = 0;
            }
        }
        return;
    }
    int cid = -1;
    if (tid < s_len) cid = cluster_assignments[vec_idx];

    for (int c = 0; c < num_clusters; c++) {
        int* buf = sdata + c * blockDim.x;
        if (tid < s_len) {
            buf[tid] = (cid == c) ? 1 : 0;
        } else if (tid < blockDim.x) {
            buf[tid] = 0;
        }
    }
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            int limit = (tid + stride < s_len) ? 1 : 0;
            if (limit) {
                for (int c = 0; c < num_clusters; c++) {
                    int* buf = sdata + c * blockDim.x;
                    buf[tid] += buf[tid + stride];
                }
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        for (int c = 0; c < num_clusters; c++) {
            int* buf = sdata + c * blockDim.x;
            block_counts[block * num_clusters + c] = buf[0];
        }
    }
}

extern "C" __global__ void exclusive_scan_block_counts(
    const int* cluster_offsets,
    const int* block_counts,
    int* block_offsets,
    int num_blocks,
    int num_clusters
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int c = 0; c < num_clusters; c++) {
            int acc = cluster_offsets[c];
            for (int b = 0; b < num_blocks; b++) {
                block_offsets[b * num_clusters + c] = acc;
                acc += block_counts[b * num_clusters + c];
            }
        }
    }
}

extern "C" __global__ void reduce_block_counts(
    const int* block_counts,
    int* cluster_sizes,
    int num_blocks,
    int num_clusters
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int c = 0; c < num_clusters; c++) {
            int acc = 0;
            for (int b = 0; b < num_blocks; b++) {
                acc += block_counts[b * num_clusters + c];
            }
            cluster_sizes[c] = acc;
        }
    }
}

extern "C" __global__ void parallel_build_cluster_index(
    const int* cluster_assignments,
    const int* block_offsets,
    int* cluster_indices,
    int database_size,
    int num_clusters
) {
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int block = blockIdx.x;
    int base = block * blockDim.x;
    int vec_idx = base + tid;
    __shared__ int s_len;
    if (tid == 0) {
        int rem = database_size - base;
        s_len = rem > blockDim.x ? blockDim.x : (rem > 0 ? rem : 0);
    }
    __syncthreads();
    if (s_len == 0) return;
    int cid = -1;
    if (tid < s_len) cid = cluster_assignments[vec_idx];
    for (int c = 0; c < num_clusters; c++) {
        int* flags = sdata + c * blockDim.x;
        if (tid < s_len) flags[tid] = (cid == c) ? 1 : 0;
        else if (tid < blockDim.x) flags[tid] = 0;
    }
    __syncthreads();
    for (int c = 0; c < num_clusters; c++) {
        int* flags = sdata + c * blockDim.x;
        for (int offset = 1; offset < s_len; offset <<= 1) {
            int v = 0;
            if (tid >= offset && tid < s_len) v = flags[tid - offset];
            __syncthreads();
            if (tid < s_len) flags[tid] += v;
            __syncthreads();
        }
        if (tid < s_len && cid == c) {
            int local_rank = flags[tid] - 1;
            int base_off = block_offsets[block * num_clusters + c];
            cluster_indices[base_off + local_rank] = vec_idx;
        }
        __syncthreads();
    }
}

extern "C" __global__ void cluster_search(
    const float* __restrict__ query_vectors,
    const float* __restrict__ database_vectors,
    const float* __restrict__ cluster_centers,
    const int* __restrict__ cluster_assignments,
    const int* __restrict__ cluster_sizes,
    const int* __restrict__ cluster_indices,
    const int* __restrict__ cluster_offsets,
    int* __restrict__ results,
    int num_queries,
    int database_size,
    int vector_dims,
    int num_clusters
) {
    if (num_queries <= 3000) {
        int query_idx = blockIdx.x;
        if (query_idx >= num_queries) return;

        const float* query = query_vectors + query_idx * vector_dims;

        float cluster_dists[16];
        int cluster_order[16];

        for (int cluster = 0; cluster < num_clusters; cluster++) {
            const float* center = cluster_centers + cluster * vector_dims;
            cluster_dists[cluster] = euclidean_distance(query, center, vector_dims);
            cluster_order[cluster] = cluster;
        }

        int clusters_to_search = (num_queries <= 1000) ? num_clusters :
                                (num_queries <= 2000) ? min(num_clusters, (num_clusters * 3) / 4) :
                                (num_queries <= 2800) ? min(num_clusters, (num_clusters * 2) / 3) :
                                min(num_clusters, max(2, num_clusters / 2));
        if (vector_dims >= 700) {
            int target = max(3, clusters_to_search);
            clusters_to_search = min(num_clusters, target);
        }
        for (int i = 0; i < clusters_to_search; i++) {
            int best = i;
            for (int j = i + 1; j < num_clusters; j++) {
                if (cluster_dists[cluster_order[j]] < cluster_dists[cluster_order[best]]) {
                    best = j;
                }
            }
            int temp = cluster_order[i];
            cluster_order[i] = cluster_order[best];
            cluster_order[best] = temp;
        }

        float min_dist = MAX_FLOAT;
        int best_idx = -1;

        for (int c_idx = 0; c_idx < clusters_to_search; c_idx++) {
            int target_cluster = cluster_order[c_idx];
            if (cluster_sizes[target_cluster] <= 0) continue;

            int start = cluster_offsets[target_cluster];
            int end = start + cluster_sizes[target_cluster];
            for (int p = start; p < end; p++) {
                int vec_idx = cluster_indices[p];
                const float* db_vector = database_vectors + vec_idx * vector_dims;
                float dist = euclidean_distance_bounded(query, db_vector, vector_dims, min_dist);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_idx = vec_idx;
                } else if (vector_dims >= 720 && num_queries <= 5000 && best_idx != -1 && dist <= min_dist + 0.0015f) {
                    float d2 = euclidean_distance_precise_bounded(query, db_vector, vector_dims, min_dist);
                    if (d2 < min_dist) {
                        min_dist = d2;
                        best_idx = vec_idx;
                    }
                }
            }
        }

        if (min_dist == MAX_FLOAT) {
            int base_stride = max(5, database_size / 2000);
            int max_checks = min(database_size / base_stride, 2000);

            for (int phase = 0; phase < 2; phase++) {
                int offset = phase * (base_stride / 2);
                for (int i = 0; i < max_checks / 2; i++) {
                    int db_idx = (offset + i * base_stride) % database_size;

                    const float* db_vector = database_vectors + db_idx * vector_dims;
                    float dist = euclidean_distance_bounded(query, db_vector, vector_dims, min_dist);
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_idx = db_idx;
                    }
                }
            }

            if (best_idx != -1) {
                int radius = min(25, base_stride);
                int start_local = max(0, best_idx - radius);
                int end_local = min(database_size, best_idx + radius + 1);

                for (int i = start_local; i < end_local; i++) {
                    if (i == best_idx) continue;
                    const float* db_vector = database_vectors + i * vector_dims;
                    float dist = euclidean_distance_bounded(query, db_vector, vector_dims, min_dist);
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_idx = i;
                    }
                }
            }
        }

        if (min_dist == MAX_FLOAT) {
            best_idx = 0;
        }

        results[query_idx] = best_idx;
    } else {
        int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (query_idx >= num_queries) return;

        const float* query = query_vectors + query_idx * vector_dims;

        float cluster_dists[16];
        int cluster_order[16];

        for (int cluster = 0; cluster < num_clusters; cluster++) {
            const float* center = cluster_centers + cluster * vector_dims;
            cluster_dists[cluster] = euclidean_distance_high(query, center, vector_dims);
            cluster_order[cluster] = cluster;
        }

        int clusters_to_search = (num_queries <= 3500) ? min(num_clusters, 5) :
                                (num_queries <= 6000) ? min(num_clusters, 4) :
                                (num_queries <= 8000) ? min(num_clusters, 3) :
                                2;
        if (num_queries <= 5000 && vector_dims >= 720) {
            clusters_to_search = num_clusters;
        } else if (vector_dims >= 720) {
            clusters_to_search = num_clusters;
        } else if (vector_dims >= 700) {
            clusters_to_search = max(clusters_to_search, min(num_clusters, (num_clusters * 3) / 4 + 1));
        }
        for (int i = 0; i < clusters_to_search; i++) {
            int best = i;
            for (int j = i + 1; j < num_clusters; j++) {
                if (cluster_dists[cluster_order[j]] < cluster_dists[cluster_order[best]]) {
                    best = j;
                }
            }
            int temp = cluster_order[i];
            cluster_order[i] = cluster_order[best];
            cluster_order[best] = temp;
        }

        float min_dist = MAX_FLOAT;
        int best_idx = -1;

        for (int c_idx = 0; c_idx < clusters_to_search; c_idx++) {
            int target_cluster = cluster_order[c_idx];
            if (cluster_sizes[target_cluster] <= 0) continue;

            int start = cluster_offsets[target_cluster];
            int end = start + cluster_sizes[target_cluster];
            for (int p = start; p < end; p++) {
                int vec_idx = cluster_indices[p];
                const float* db_vector = database_vectors + vec_idx * vector_dims;
                float dist = euclidean_distance_high_bounded(query, db_vector, vector_dims, min_dist);
                if (vector_dims >= 720 && num_queries <= 5000 && dist <= min_dist + 0.0015f) {
                    float d2 = euclidean_distance_precise_bounded(query, db_vector, vector_dims, min_dist);
                    if (d2 < dist) dist = d2;
                }
                if (dist < min_dist) {
                    min_dist = dist;
                    best_idx = vec_idx;
                }
            }
        }

        if (min_dist == MAX_FLOAT) {
            int base_stride = (vector_dims >= 720) ? max(7, database_size / 900) : max(9, database_size / 1200);
            int max_checks = min(database_size / base_stride, (vector_dims >= 720) ? 1600 : 1200);

            for (int phase = 0; phase < 2; phase++) {
                int offset = phase * (base_stride / 3);
                int phase_checks = max_checks / 2;

                for (int i = 0; i < phase_checks; i++) {
                    int db_idx = (offset + i * base_stride) % database_size;

                    const float* db_vector = database_vectors + db_idx * vector_dims;
                    float dist = euclidean_distance_high_bounded(query, db_vector, vector_dims, min_dist);
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_idx = db_idx;
                    }
                }
            }

            if (best_idx != -1) {
                int radius = (vector_dims >= 720) ? min(32, (base_stride * 2) / 3) : min(18, base_stride / 2);
                int start_local = max(0, best_idx - radius);
                int end_local = min(database_size, best_idx + radius + 1);

                for (int i = start_local; i < end_local; i++) {
                    if (i == best_idx) continue;
                    const float* db_vector = database_vectors + i * vector_dims;
                    float dist = euclidean_distance_high_bounded(query, db_vector, vector_dims, min_dist);
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_idx = i;
                    }
                }
            }
        }

        if (min_dist == MAX_FLOAT) {
            best_idx = 0;
        }

        results[query_idx] = best_idx;
    }
}
