/** 
Overlapped block motion compensation for CUDA

Efficient implementation that divides up the image into regions based on the 
amount of blocks that overlap it, which is 1 (in the middle), 2 (horizontal 
or vertical overlap) or 4 (diagonal overlap). 

By processing these regions in different cuda blocks, any divergence between
threads is prevented.

*/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
//#include <cutil.h>
#include <cuda.h>

#include <algorithm>
#include <cassert>

#define SPECIAL

//#define USING_GLOBAL

#define THREADSX_LOG2 4
#define THREADSY_LOG2 4

#ifdef SPECIAL
#define WIDTHX_LOG2 (THREADSX_LOG2+1)
#define WIDTHY_LOG2 THREADSY_LOG2
#else
#define WIDTHX_LOG2 THREADSX_LOG2
#define WIDTHY_LOG2 THREADSY_LOG2
#endif

#define THREADSX (1<<THREADSX_LOG2)
#define THREADSY (1<<THREADSY_LOG2)

#define WIDTHX (1<<WIDTHX_LOG2)
#define WIDTHY (1<<WIDTHY_LOG2)

#include "common.h"
#include "cudamotion.h"

#include "motion_kernel_tex.h"

static inline int div_roundup(int x, int y)
{
    return (x+y-1)/y;
}

/// Private object
struct _CudaMotion
{
  cudaStream_t stream;
  struct _MotionVector *vectors;
  struct cudaArray *bdata;
};

__global__ void tex_copy(
    uint16_t* dest, int dstride, int dwidth, int dheight
)
{
  worktype2 val;
  int xofs = (blockIdx.x << WIDTHX_LOG2) + (threadIdx.x<<1);
  int yofs = (blockIdx.y << WIDTHY_LOG2) + threadIdx.y;
  dest = OFFSET_U16(dest, __mul24(dstride, yofs) + (xofs<<1));

  val.x = (worktype)(tex2D(ref1, xofs*2.0f+0.5f, yofs*2.0f+0.5f)*256.0f);
  val.y = (worktype)(tex2D(ref1, xofs*2.0f+3.5f, yofs*2.0f+0.5f)*256.0f);

  int16_2 sval;
  sval.x = val.x - 128;
  sval.y = val.y - 128;
  *((int16_2*)dest) = sval;
}

extern "C" {

CudaMotion *cuda_motion_init(cudaStream_t stream)
{
    CudaMotion *rv;

    rv = new CudaMotion;
    rv->vectors = 0;
    rv->bdata = 0;
    rv->stream = stream;

    return rv;
}

void cuda_motion_free(CudaMotion *rv)
{
    cudaFreeArray(rv->bdata);
    cudaFreeHost((void*)rv->vectors);

    delete rv;
}

struct _MotionVector *cuda_motion_reserve(CudaMotion *self, int width, int height)
{
    /// XXX check for dimension changes!
    cudaChannelFormatDesc bdesc = cudaCreateChannelDesc<short4>();
    if(!self->vectors)
        cudaMallocHost((void**)&self->vectors, width*height*sizeof(struct _MotionVector));
    if(!self->bdata)
        cudaMallocArray(&self->bdata, &bdesc, width, height);
    return self->vectors;
}

void cuda_motion_begin(CudaMotion *self, CudaMotionData *d)
{
    /// Upload motion vectors
    cudaChannelFormatDesc bdesc = cudaCreateChannelDesc<short4>();

    cudaMemcpy2DToArrayAsync(self->bdata, 0, 0, self->vectors, d->obmc.blocksx*8,
                     d->obmc.blocksx*8, d->obmc.blocksy, cudaMemcpyHostToDevice, self->stream);

    bt1.addressMode[0] = cudaAddressModeClamp;
    bt1.addressMode[1] = cudaAddressModeClamp;
    bt1.filterMode = cudaFilterModePoint;
    bt1.normalized = false;

    ref1.addressMode[0] = cudaAddressModeClamp;
    ref1.addressMode[1] = cudaAddressModeClamp;
    ref1.filterMode = cudaFilterModeLinear;
    ref1.normalized = false;

    ref2.addressMode[0] = cudaAddressModeClamp;
    ref2.addressMode[1] = cudaAddressModeClamp;
    ref2.filterMode = cudaFilterModeLinear;
    ref2.normalized = false;

    /// Bind motion vector texture
    cudaBindTextureToArray(bt1, self->bdata, bdesc);
}

void cuda_motion_copy(CudaMotion *self, CudaMotionData *d, int16_t *output, int ostride, int width, int height, int component, int xshift, int yshift, struct cudaArray *aref1, struct cudaArray *aref2)
{
    /// Bind reference texture
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
    cudaBindTextureToArray(ref1, aref1, channelDesc);
    if(aref2)
        cudaBindTextureToArray(ref2, aref2, channelDesc);

    /// Execute the kernel
    dim3 block_size, grid_size;
    int shared_size;

    /// Determine execution parameters
    block_size.x = THREADSX;
    block_size.y = THREADSY;
    block_size.z = 1;

    int blocksX = div_roundup(width,  d->obmc.x_sep>>xshift);
    int blocksY = div_roundup(height, d->obmc.y_sep>>yshift);
    int xB = div_roundup(blocksX * (d->obmc.x_mid>>xshift), WIDTHX);
    int yB = div_roundup(blocksY * (d->obmc.y_mid>>yshift), WIDTHY);
    int xC = div_roundup((blocksX+1) * (d->obmc.x_ramp>>xshift), WIDTHX);
    int yC = div_roundup((blocksY+1) * (d->obmc.y_ramp>>yshift), WIDTHY);
/*
    printf("%i %i : %i %i %i %i %i %i : %i %i %i %i %i %i : %i %i\n", width, height, 
                                  d->obmc.x_sep>>xshift, d->obmc.y_sep>>yshift, d->obmc.x_mid>>xshift, d->obmc.y_mid>>yshift, d->obmc.x_ramp>>xshift, d->obmc.y_ramp>>yshift,
                                  blocksX, blocksY, xB, yB, xC, yC,
                                  xB + xC, yB + yB);
*/
    grid_size.x = xB + xC;
    grid_size.y = yB + yC;
    grid_size.z = 1;
    shared_size = 0;

/*
    printf("%ix%i comp %i grid %ix%i scale %fx%f ramp %i %i sep %i %i mid %i %i\n", 
        width, height,
        component, grid_size.x, grid_size.y, sxscale, syscale,
        d->obmc.x_ramp_log2 - xshift, d->obmc.y_ramp_log2 - yshift, 
        d->obmc.x_sep_log2 - xshift,  d->obmc.y_sep_log2 - yshift, 
        d->obmc.x_mid_log2 - xshift,  d->obmc.y_mid_log2 - yshift
    );
*/
#if 1
    if((d->obmc.x_ramp_log2 - xshift - 1) == 0)
    {
      /** Slower variant, use 2 byte writes */
      motion_copy_2ref_2b<<<grid_size, block_size, shared_size, self->stream>>>(
          output, ostride, width, height, xB, yB,
          component*8,
          d->obmc.x_ramp_log2 - xshift, d->obmc.y_ramp_log2 - yshift, 
          d->obmc.x_sep_log2 - xshift,  d->obmc.y_sep_log2 - yshift, 
          d->obmc.x_mid_log2 - xshift,  d->obmc.y_mid_log2 - yshift,
          d->obmc.weight1, d->obmc.weight2, d->obmc.weight_shift,
          xshift, yshift
          );
    }
    else
    {
      /** Fast variant, use 4 byte writes */
      motion_copy_2ref_4b<<<grid_size, block_size, shared_size, self->stream>>>(
          output, ostride, width, height, xB, yB,
          component*8,
          d->obmc.x_ramp_log2 - xshift, d->obmc.y_ramp_log2 - yshift, 
          d->obmc.x_sep_log2 - xshift,  d->obmc.y_sep_log2 - yshift, 
          d->obmc.x_mid_log2 - xshift,  d->obmc.y_mid_log2 - yshift,
          d->obmc.weight1, d->obmc.weight2, d->obmc.weight_shift,
          xshift, yshift
          );
    }
#else
    tex_copy<<<dim3(div_roundup(width,WIDTHX),div_roundup(height,WIDTHY),1), block_size, 0>>>(
        output, ostride, width, height);
#endif
    /// XXX unbind textures?
}


}
