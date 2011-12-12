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
#include "upsample_kernel.h"

extern "C" {

void cuda_upsample_horizontal(uint8_t *output, int ostride, uint8_t *input, int istride, int width, int height, cudaStream_t stream)
{
    dim3 block_size, grid_size;
    int shared_size;

    block_size.x = THREADS;
    block_size.y = block_size.z = 1;
    grid_size.x = height;
    grid_size.y = grid_size.z = 1;
    shared_size = ROUNDUP4(BLEFT+width+BRIGHT) + ROUNDUP4(width);

    upsample_horizontal<<<grid_size, block_size, shared_size, stream>>>(output, ostride, input, istride, width);
}

void cuda_upsample_vertical(uint8_t *output, int ostride, uint8_t *input, int istride, int width, int height, cudaStream_t stream)
{
    dim3 block_size, grid_size;
    int shared_size;

    block_size.x = COLUMN_THREAD_W;
    block_size.y = COLUMN_THREAD_H;
    block_size.z = 1;
    grid_size.x = (width+COLUMN_TILE_W-1)/COLUMN_TILE_W;
    grid_size.y = (height+COLUMN_TILE_H-1)/COLUMN_TILE_H;
    grid_size.z = 1;
    shared_size = COLUMN_TILE_W*(BLEFT+COLUMN_TILE_H+BRIGHT) + COLUMN_TILE_W*COLUMN_TILE_H;

    upsample_vertical<<<grid_size, block_size, shared_size, stream>>>(output, ostride, input, istride, width, height);
}


}
