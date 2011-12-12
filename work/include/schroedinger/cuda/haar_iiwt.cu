#include <algorithm>
#include <cassert>
#include "common.h"
#include "wavelet_common.h"

#define BROWS (2*BSVY)

static __global__ void s_transform_h( DATATYPE* data, int width, int stride, int shift )
{
    extern __shared__ DATATYPE shared[];  

    const int bid = blockIdx.x;    // row
    const int tid = threadIdx.x;   // thread id within row
    const int tidu16 = ((tid&16)>>4)|((tid&15)<<1)|(tid&~31);
    //const int BSH = blockDim.x;

    // Row offset
    const unsigned int idata = __mul24(bid, stride);

    // Load entire line into shared memory
    // Deinterleave right here
    int half = (width>>1);

    // Make sure half is odd, to prevent bank conflicts
    unsigned int ofs;

    /// Write line back to global memory, don't interleave again
    uint32_t *row = (uint32_t*)&data[idata];
    for(ofs = tid; ofs < (width>>1); ofs += BSH)
         *((uint32_t*)&shared[ofs<<1]) = row[ofs];

    __syncthreads();

    // Now apply wavelet lifting to entire line at once
    const int end = (width>>1);
    for(ofs = tidu16; ofs < end; ofs += BSH)
    {
        int val1 = shared[ofs];
        int val2 = shared[half+ofs];
        val1 -= (val2+1)>>1;
        val2 += val1;
        shared[ofs] = val1;
        shared[half+ofs] = val2;
    }

    __syncthreads();

    // Shared memory output offset for this thread
    int od = (1<<shift)>>1;
    for(ofs = tid; ofs < (width>>1); ofs += BSH)
    {
        uint16_t a = (uint16_t)((shared[ofs]+od)>>shift);
        uint16_t b = (uint16_t)((shared[ofs + half]+od)>>shift);
        row[ofs] = a|(b<<16);
    }

}

__device__ void doRevTransform()
{
    const int tidx = threadIdx.x<<1;   // column
    const int tidy = threadIdx.y;   // row

    extern __shared__ DATATYPE shared[];
    int ofs;

    ofs = (((tidy<<1))<<BCOLS_SHIFT) + tidx;

    {
        int val1 = shared[ofs];
        int val2 = shared[ofs+BCOLS];
        val1 -= (val2+1)>>1;
        val2 += val1;
        shared[ofs] = val1;
        shared[ofs+BCOLS] = val2;
    }
    {
        int val1 = shared[ofs+1];
        int val2 = shared[ofs+BCOLS+1];
        val1 -= (val2+1)>>1;
        val2 += val1;
        shared[ofs+1] = val1;
        shared[ofs+BCOLS+1] = val2;
    }

}

static __global__ void s_transform_v( DATATYPE* data, int width, int height, int stride )
{
    extern __shared__ DATATYPE shared[];  

    const unsigned int bid = blockIdx.x;    // slab (BCOLS columns)
    const unsigned int tidx = threadIdx.x<<1;   // column
    const unsigned int tidy = threadIdx.y;   // row    
    const unsigned int swidth = min(width-(bid<<BCOLS_SHIFT), BCOLS); // Width of this slab, usually BCOLS but can be less

    // Element offset in global memory
    int idata = tidx + (bid<<BCOLS_SHIFT) + __mul24(tidy<<1, stride);
    
    const unsigned int sdata = tidx + ((tidy<<1)<<BCOLS_SHIFT);
    
    unsigned int leftover = height % BROWS; /// How far to fill buffer on last read
    unsigned int blocks = height / BROWS;

    const unsigned int data_inc = __mul24(BROWS, stride);

    for(unsigned int block=0; block<blocks; ++block)
    {
        if(tidx < swidth)
        {
            UINT(shared[sdata]) = UINT(data[idata]);
            UINT(shared[sdata + BCOLS]) = UINT(data[idata + stride]);
        }

        __syncthreads();
        
        doRevTransform();
        
        __syncthreads();

        if(tidx < swidth)
        {
            UINT(data[idata]) = UINT(shared[sdata]);
            UINT(data[idata + stride]) = UINT(shared[sdata + BCOLS]);
        }
        
        idata += data_inc;
    }

    if(tidx < swidth && tidy < leftover)
    {
        UINT(shared[sdata]) = UINT(data[idata]);
        UINT(shared[sdata + BCOLS]) = UINT(data[idata + stride]);
    }

    __syncthreads();

    doRevTransform();
    
    __syncthreads();
    
    /// Write back leftover
    if(tidx < swidth && (tidy<<1) < leftover)
    {
        UINT(data[idata]) = UINT(shared[sdata]);
        UINT(data[idata + stride]) = UINT(shared[sdata + BCOLS]);
    }
}

void cuda_iiwt_haar(int shift, int16_t *d_data, int lwidth, int lheight, int stride, cudaStream_t stream)
{
    /** Invoke kernel */
    dim3 block_size;
    dim3 grid_size;
    int shared_size;

#ifdef VERTICAL
    block_size.x = BSVX;
    block_size.y = BSVY;
    block_size.z = 1;
    grid_size.x = (lwidth+BCOLS-1)/BCOLS;
    grid_size.y = 1;
    grid_size.z = 1;
    shared_size = BCOLS*BROWS*2; 
	
    s_transform_v<<<grid_size, block_size, shared_size, stream>>>(d_data, lwidth, lheight, stride);
#endif
#ifdef HORIZONTAL
    block_size.x = BSH;
    block_size.y = 1;
    block_size.z = 1;
    grid_size.x = lheight;
    grid_size.y = 1;
    grid_size.z = 1;
    shared_size = lwidth * sizeof(int16_t); 
    s_transform_h<<<grid_size, block_size, shared_size, stream>>>(d_data, lwidth, stride, shift);
#endif
}
