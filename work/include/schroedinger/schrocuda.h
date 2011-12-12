
#ifndef __SCHRO_CUDA_H__
#define __SCHRO_CUDA_H__

SCHRO_BEGIN_DECLS

#ifdef SCHRO_ENABLE_UNSTABLE_API

void schro_cuda_init (void);
SchroMemoryDomain *schro_memory_domain_new_cuda (void);


void schro_motion_render_cuda (SchroMotion *motion, SchroFrame *dest);

void schro_frame_inverse_iwt_transform_cuda (SchroFrame *frame,
    SchroFrame *transform_frame, SchroParams *params);

#endif

SCHRO_END_DECLS

#endif

