
#ifndef __SCHRO_GPUFRAME_H__
#define __SCHRO_GPUFRAME_H__

#include <schroedinger/schro-stdint.h>
#include <schroedinger/schroframe.h>

SCHRO_BEGIN_DECLS

#ifdef SCHRO_ENABLE_UNSTABLE_API

#define SCHRO_FRAME_IS_CUDA(frame) ((frame)->domain && ((frame)->domain->flags & SCHRO_MEMORY_DOMAIN_CUDA))

void schro_gpuframe_to_cpu (SchroFrame *dest, SchroFrame *src);
void schro_frame_to_gpu (SchroFrame *dest, SchroFrame *src);

void schro_gpuframe_convert (SchroFrame *dest, SchroFrame *src);
void schro_gpuframe_add (SchroFrame *dest, SchroFrame *src);
void schro_gpuframe_subtract (SchroFrame *dest, SchroFrame *src);

void schro_gpuframe_iwt_transform (SchroFrame *frame, SchroParams *params);
void schro_gpuframe_inverse_iwt_transform (SchroFrame *frame, SchroParams *params);

void schro_gpuframe_compare (SchroFrame *a, SchroFrame *b);

void schro_gpuframe_upsample(SchroFrame *dst, SchroFrame *src);

SchroUpsampledFrame *schro_upsampled_gpuframe_new(SchroVideoFormat *fmt);
void schro_upsampled_gpuframe_upsample(SchroUpsampledFrame *rv);
void schro_upsampled_gpuframe_free(SchroUpsampledFrame *x);

void schro_gpuframe_zero (SchroFrame *dest);

#endif

SCHRO_END_DECLS

#endif

