/// XXX speed up by using shared memory / coalescing?

#ifdef _WIN32
#  define NOMINMAX 
#endif

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
//#include <cutil.h>
#include <cuda.h>

#include <algorithm>
#include <cassert>

#define THREADS 256

#include "common.h"
#include "convert_base_coalesce.h"
//#include "convert_base.h"
#include "convert_packed.h"
#include "arith_coalesce.h"
//#include "arith.h"



extern "C" {

void cuda_convert_u8_s16(uint8_t* dst, int dstride, int dwidth, int dheight, int16_t* src, int sstride, int swidth, int sheight, cudaStream_t stream)
{
    dim3 block_size, grid_size;
    int shared_size;

    block_size.x = THREADS;
    block_size.y = block_size.z = 1;
    grid_size.x = dheight;
    grid_size.y = grid_size.z = 1;
    shared_size = 0;

    convert_u8_s16<<<grid_size, block_size, shared_size, stream>>>(dst, dstride, dwidth, src, sstride, swidth, sheight);
}
void cuda_convert_s16_u8(int16_t* dst, int dstride, int dwidth, int dheight, uint8_t* src, int sstride, int swidth, int sheight, cudaStream_t stream)
{
    dim3 block_size, grid_size;
    int shared_size;

    block_size.x = THREADS;
    block_size.y = block_size.z = 1;
    grid_size.x = dheight;
    grid_size.y = grid_size.z = 1;
    shared_size = 0;

    convert_s16_u8<<<grid_size, block_size, shared_size, stream>>>(dst, dstride, dwidth, src, sstride, swidth, sheight);
}
void cuda_convert_u8_u8(uint8_t* dst, int dstride, int dwidth, int dheight, uint8_t* src, int sstride, int swidth, int sheight, cudaStream_t stream)
{
    dim3 block_size, grid_size;
    int shared_size;

    block_size.x = THREADS;
    block_size.y = block_size.z = 1;
    grid_size.x = dheight;
    grid_size.y = grid_size.z = 1;
    shared_size = 0;

    convert_u8_u8<<<grid_size, block_size, shared_size, stream>>>(dst, dstride, dwidth, src, sstride, swidth, sheight);
}
void cuda_convert_s16_s16(int16_t* dst, int dstride, int dwidth, int dheight, int16_t* src, int sstride, int swidth, int sheight, cudaStream_t stream)
{
    dim3 block_size, grid_size;
    int shared_size;

    block_size.x = THREADS;
    block_size.y = block_size.z = 1;
    grid_size.x = dheight;
    grid_size.y = grid_size.z = 1;
    shared_size = 0;

    convert_s16_s16<<<grid_size, block_size, shared_size, stream>>>(dst, dstride, dwidth, src, sstride, swidth, sheight);
}
void cuda_convert_u8_422_yuyv(uint8_t* dsty, int ystride, uint8_t* dstu, int ustride, uint8_t* dstv, int vstride, int dwidth, int dheight, uint8_t* _src, int sstride, int swidth, int sheight, cudaStream_t stream)
{
    dim3 block_size, grid_size;
    int shared_size;

    dwidth >>= 1;  swidth >>= 1;

    block_size.x = THREADS;
    block_size.y = block_size.z = 1;
    grid_size.x = dheight;
    grid_size.y = grid_size.z = 1;
    shared_size = 0;

    convert_u8_422_yuyv<<<grid_size, block_size, shared_size, stream>>>(dsty, ystride, dstu, ustride, dstv, vstride, dwidth, _src, sstride, swidth, sheight);
}
void cuda_convert_u8_422_uyvy(uint8_t* dsty, int ystride, uint8_t* dstu, int ustride, uint8_t* dstv, int vstride, int dwidth, int dheight, uint8_t* _src, int sstride, int swidth, int sheight, cudaStream_t stream)
{
    dim3 block_size, grid_size;
    int shared_size;

    dwidth >>= 1;  swidth >>= 1;

    block_size.x = THREADS;
    block_size.y = block_size.z = 1;
    grid_size.x = dheight;
    grid_size.y = grid_size.z = 1;
    shared_size = 0;

    convert_u8_422_uyvy<<<grid_size, block_size, shared_size, stream>>>(dsty, ystride, dstu, ustride, dstv, vstride, dwidth, _src, sstride, swidth, sheight);
}
void cuda_convert_u8_444_ayuv(uint8_t* dsty, int ystride, uint8_t* dstu, int ustride, uint8_t* dstv, int vstride, int dwidth, int dheight, uint8_t* _src, int sstride, int swidth, int sheight, cudaStream_t stream)
{
    dim3 block_size, grid_size;
    int shared_size;

    block_size.x = THREADS;
    block_size.y = block_size.z = 1;
    grid_size.x = dheight;
    grid_size.y = grid_size.z = 1;
    shared_size = 0;

    convert_u8_444_ayuv<<<grid_size, block_size, shared_size, stream>>>(dsty, ystride, dstu, ustride, dstv, vstride, dwidth, _src, sstride, swidth, sheight);
}
void cuda_convert_yuyv_u8_422 (uint8_t* _dst, int dstride, int dwidth, int dheight, uint8_t* srcy, int ystride, uint8_t* srcu, int ustride, uint8_t* srcv, int vstride, int swidth, int sheight, cudaStream_t stream)
{
    dim3 block_size, grid_size;
    int shared_size;

    dwidth >>= 1;  swidth >>= 1;

    block_size.x = THREADS;
    block_size.y = block_size.z = 1;
    grid_size.x = dheight;
    grid_size.y = grid_size.z = 1;
    shared_size = 0;

    convert_yuyv_u8_422<<<grid_size, block_size, shared_size, stream>>>(_dst, dstride, dwidth, srcy, ystride, srcu, ustride, srcv, vstride, swidth, sheight);
}
void cuda_convert_uyvy_u8_422 (uint8_t* _dst, int dstride, int dwidth, int dheight, uint8_t* srcy, int ystride, uint8_t* srcu, int ustride, uint8_t* srcv, int vstride, int swidth, int sheight, cudaStream_t stream)
{
    dim3 block_size, grid_size;
    int shared_size;

    dwidth >>= 1;  swidth >>= 1;

    block_size.x = THREADS;
    block_size.y = block_size.z = 1;
    grid_size.x = dheight;
    grid_size.y = grid_size.z = 1;
    shared_size = 0;

    convert_uyvy_u8_422<<<grid_size, block_size, shared_size, stream>>>(_dst, dstride, dwidth, srcy, ystride, srcu, ustride, srcv, vstride, swidth, sheight);
}
void cuda_convert_ayuv_u8_444 (uint8_t* _dst, int dstride, int dwidth, int dheight, uint8_t* srcy, int ystride, uint8_t* srcu, int ustride, uint8_t* srcv, int vstride, int swidth, int sheight, cudaStream_t stream)
{
    dim3 block_size, grid_size;
    int shared_size;

    block_size.x = THREADS;
    block_size.y = block_size.z = 1;
    grid_size.x = dheight;
    grid_size.y = grid_size.z = 1;
    shared_size = 0;

    convert_ayuv_u8_444<<<grid_size, block_size, shared_size, stream>>>(_dst, dstride, dwidth, srcy, ystride, srcu, ustride, srcv, vstride, swidth, sheight);
}
void cuda_subtract_s16_u8(int16_t* dst, int dstride, int dwidth, int dheight, uint8_t* src, int sstride, int swidth, int sheight, cudaStream_t stream)
{
    dim3 block_size, grid_size;
    int shared_size;

    block_size.x = THREADS;
    block_size.y = block_size.z = 1;
    grid_size.x = std::min(dheight, sheight);
    grid_size.y = grid_size.z = 1;
    shared_size = 0;

    subtract_s16_u8<<<grid_size, block_size, shared_size, stream>>>(dst, dstride, src, sstride, std::min(swidth, dwidth));
}
void cuda_subtract_s16_s16(int16_t* dst, int dstride, int dwidth, int dheight, int16_t* src, int sstride, int swidth, int sheight, cudaStream_t stream)
{
    dim3 block_size, grid_size;
    int shared_size;

    block_size.x = THREADS;
    block_size.y = block_size.z = 1;
    grid_size.x = std::min(dheight, sheight);
    grid_size.y = grid_size.z = 1;
    shared_size = 0;

    subtract_s16_s16<<<grid_size, block_size, shared_size, stream>>>(dst, dstride, src, sstride, std::min(swidth, dwidth));
}
void cuda_add_s16_u8(int16_t* dst, int dstride, int dwidth, int dheight, uint8_t* src, int sstride, int swidth, int sheight, cudaStream_t stream)
{
    dim3 block_size, grid_size;
    int shared_size;

    block_size.x = THREADS;
    block_size.y = block_size.z = 1;
    grid_size.x = std::min(dheight, sheight);
    grid_size.y = grid_size.z = 1;
    shared_size = 0;

    add_s16_u8<<<grid_size, block_size, shared_size, stream>>>(dst, dstride, src, sstride, std::min(swidth, dwidth));
}

void cuda_add_s16_s16(int16_t* dst, int dstride, int dwidth, int dheight, int16_t* src, int sstride, int swidth, int sheight, cudaStream_t stream)
{
    dim3 block_size, grid_size;
    int shared_size;

    block_size.x = THREADS;
    block_size.y = block_size.z = 1;
    grid_size.x = std::min(dheight, sheight);
    grid_size.y = grid_size.z = 1;
    shared_size = 0;

    add_s16_s16<<<grid_size, block_size, shared_size, stream>>>(dst, dstride, src, sstride, std::min(swidth, dwidth));
}

}

