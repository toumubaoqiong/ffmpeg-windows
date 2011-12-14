#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>
#include <schroedinger/schrogpumotion.h>

#include <stdlib.h>
#include <string.h>

#include <cuda_runtime_api.h>
#include "cudawavelet.h"
#include "cudaframe.h"
#include "cudaupsample.h"
#include "cudamotion.h"
#include <stdio.h>

struct _SchroGPUMotion
{
  CudaMotion *cm;
  CudaMotionData md;
  struct _MotionVector *vectors;
};

static inline int
ilog2 (unsigned int x)
{
  int i;

  if (x == 0)
    return -1;

  for (i = 0; x > 1; i++) {
    x >>= 1;
  }
  return i;
}

SchroGPUMotion *
schro_gpumotion_new (SchroCUDAStream stream)
{
  SchroGPUMotion *ret;

  ret = schro_malloc (sizeof (SchroGPUMotion));
  ret->cm = cuda_motion_init (stream);
  ret->vectors = NULL;
  return ret;
}

void
schro_gpumotion_free (SchroGPUMotion * rv)
{
  cuda_motion_free (rv->cm);
  schro_free (rv);
}

void
schro_gpumotion_init (SchroGPUMotion * self, SchroMotion * motion)
{
  /* Create texture */
  self->vectors =
      cuda_motion_reserve (self->cm, motion->params->x_num_blocks,
      motion->params->y_num_blocks);
}

#define md self->md
#define vectors self->vectors
void
schro_gpumotion_copy (SchroGPUMotion * self, SchroMotion * motion)
{
  int i;
  int numv;
  int precision;

  SCHRO_ASSERT (vectors);

  SCHRO_DEBUG ("schro_gpuframe_copy_with_motion");

  md.obmc.blocksx = motion->params->x_num_blocks;
  md.obmc.blocksy = motion->params->y_num_blocks;
  //printf("%i %i %i\n", md.obmc.blocksx, md.obmc.blocksy, md.obmc.blocksx*md.obmc.blocksy);
  md.obmc.weight1 = motion->params->picture_weight_1;
  md.obmc.weight2 = motion->params->picture_weight_2;
  //printf("%i %i %i\n", md.obmc.weight1, md.obmc.weight2, motion->params->picture_weight_bits);

  md.obmc.x_len = motion->params->xblen_luma;
  md.obmc.y_len = motion->params->yblen_luma;
  md.obmc.x_sep = motion->params->xbsep_luma;
  md.obmc.y_sep = motion->params->ybsep_luma;
  md.obmc.weight_shift = motion->params->picture_weight_bits;

  // Overlapped size
  md.obmc.x_ramp = md.obmc.x_len - md.obmc.x_sep;
  md.obmc.y_ramp = md.obmc.y_len - md.obmc.y_sep;

  // Non-overlapped size
  md.obmc.x_mid = md.obmc.x_sep - md.obmc.x_ramp;
  md.obmc.y_mid = md.obmc.y_sep - md.obmc.y_ramp;

  //md.obmc.shift = ilog2(md.obmc.x_ramp) + ilog2(md.obmc.y_ramp) + motion->params->picture_weight_bits;

  // Convert to powers of two, for fast arithmetic
  md.obmc.x_sep_log2 = ilog2 (md.obmc.x_sep);
  md.obmc.y_sep_log2 = ilog2 (md.obmc.y_sep);
  md.obmc.x_ramp_log2 = ilog2 (md.obmc.x_ramp);
  md.obmc.y_ramp_log2 = ilog2 (md.obmc.y_ramp);
  md.obmc.x_mid_log2 = ilog2 (md.obmc.x_mid);
  md.obmc.y_mid_log2 = ilog2 (md.obmc.y_mid);

  // Transfer vectors
  numv = md.obmc.blocksx * md.obmc.blocksy;
//
  // make sure we always have 3 bits of precision
#ifndef TESTMODE
  precision = (3 - motion->params->mv_precision);
#endif
  md.obmc.mv_precision = motion->params->mv_precision;
  for (i = 0; i < numv; ++i) {

#ifndef TESTMODE
    if (motion->motion_vectors[i].pred_mode == 0) {
      SchroMotionVectorDC *mvdc =
          (SchroMotionVectorDC *) & motion->motion_vectors[i];
      // DC
      vectors[i].x1 = MOTION_NONE;
      vectors[i].x2 = MOTION_NONE;
      // YUV color code is encoded in these
      vectors[i].y1 =
          ((mvdc->dc[0] + 128) & 0xFF) | (((mvdc->dc[1] + 128) & 0xFF) << 8);
      vectors[i].y2 = (mvdc->dc[2] + 128) & 0xFF;
    } else {
      // Reference 1
      if (motion->motion_vectors[i].pred_mode & 1) {
        vectors[i].x1 = motion->motion_vectors[i].dx[0] << precision;
        vectors[i].y1 = motion->motion_vectors[i].dy[0] << precision;
      } else {
        vectors[i].x1 = MOTION_NONE;
        vectors[i].y1 = MOTION_NONE;
      }
      // Reference 2
      if (motion->motion_vectors[i].pred_mode & 2) {
        vectors[i].x2 = motion->motion_vectors[i].dx[1] << precision;
        vectors[i].y2 = motion->motion_vectors[i].dy[1] << precision;
      } else {
        vectors[i].x2 = MOTION_NONE;
        vectors[i].y2 = MOTION_NONE;
      }
    }
    //md.flags[i] = (motion->motion_vectors[i].using_global << 2) | motion->motion_vectors[i].pred_mode;
#else
    {
      vectors[i].x1 = 0;
      vectors[i].y1 = 0;
      vectors[i].x2 = MOTION_NONE;
      vectors[i].y2 = MOTION_NONE;
    }
#endif
  }
  //printf("%i blocks (%ix%i)\n", numv, md.obmc.blocksx, md.obmc.blocksy);
}

void
schro_gpumotion_render (SchroGPUMotion * self, SchroMotion * motion,
    SchroFrame * gdest)
{
  CudaMotion *cm = self->cm;
  SchroUpsampledFrame *ref1 = (SchroUpsampledFrame *) motion->src1;
  SchroUpsampledFrame *ref2 = (SchroUpsampledFrame *) motion->src2;
  int fwidth;
  int fheight;
  int hshift;
  int vshift;

  cuda_motion_begin (cm, &md);

  fwidth = motion->params->video_format->width;
  fheight = motion->params->video_format->height;
  hshift =
      SCHRO_CHROMA_FORMAT_H_SHIFT (motion->params->video_format->chroma_format);
  vshift =
      SCHRO_CHROMA_FORMAT_V_SHIFT (motion->params->video_format->chroma_format);

  SCHRO_ASSERT (schro_async_get_exec_domain () == SCHRO_EXEC_DOMAIN_CUDA);

  SCHRO_ASSERT (gdest->domain->flags == SCHRO_MEMORY_DOMAIN_CUDA);

  if (ref2) {
    cuda_motion_copy (cm, &md, (int16_t *) gdest->components[0].data,
        gdest->components[0].stride, fwidth, fheight, 0, 0, 0,
        ref1->components[0], ref2->components[0]);
    cuda_motion_copy (cm, &md, (int16_t *) gdest->components[1].data,
        gdest->components[1].stride, fwidth >> hshift, fheight >> vshift, 1,
        hshift, vshift, ref1->components[1], ref2->components[1]);
    cuda_motion_copy (cm, &md, (int16_t *) gdest->components[2].data,
        gdest->components[2].stride, fwidth >> hshift, fheight >> vshift, 2,
        hshift, vshift, ref1->components[2], ref2->components[2]);
  } else {
    cuda_motion_copy (cm, &md, (int16_t *) gdest->components[0].data,
        gdest->components[0].stride, fwidth, fheight, 0, 0, 0,
        ref1->components[0], NULL);
    cuda_motion_copy (cm, &md, (int16_t *) gdest->components[1].data,
        gdest->components[1].stride, fwidth >> hshift, fheight >> vshift, 1,
        hshift, vshift, ref1->components[1], NULL);
    cuda_motion_copy (cm, &md, (int16_t *) gdest->components[2].data,
        gdest->components[2].stride, fwidth >> hshift, fheight >> vshift, 2,
        hshift, vshift, ref1->components[2], NULL);
  }
}

#undef md
#undef vectors
