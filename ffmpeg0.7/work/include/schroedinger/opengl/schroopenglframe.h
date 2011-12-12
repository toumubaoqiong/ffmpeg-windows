
#ifndef __SCHRO_OPENGL_FRAME_H__
#define __SCHRO_OPENGL_FRAME_H__

#include <schroedinger/schroframe.h>
#include <schroedinger/schroparams.h>
#include <schroedinger/opengl/schroopengltypes.h>

SCHRO_BEGIN_DECLS

#define SCHRO_FRAME_IS_OPENGL(_frame) \
    ((_frame)->domain && ((_frame)->domain->flags & SCHRO_MEMORY_DOMAIN_OPENGL))

void schro_opengl_frame_setup (SchroOpenGL *opengl, SchroFrame *frame);
void schro_opengl_frame_cleanup (SchroFrame *frame);

SchroFrame *schro_opengl_frame_new (SchroOpenGL *opengl,
    SchroMemoryDomain *opengl_domain, SchroFrameFormat format, int width,
    int height);
SchroFrame *schro_opengl_frame_clone (SchroFrame *opengl_frame);
SchroFrame *schro_opengl_frame_clone_and_push (SchroOpenGL *opengl,
    SchroMemoryDomain *opengl_domain, SchroFrame *cpu_frame);

void schro_opengl_frame_push (SchroFrame *dest, SchroFrame *src); // CPU -> GPU
void schro_opengl_frame_pull (SchroFrame *dest, SchroFrame *src); // CPU <- GPU

void schro_opengl_frame_convert (SchroFrame *dest, SchroFrame *src);
void schro_opengl_frame_add (SchroFrame *dest, SchroFrame *src);
void schro_opengl_frame_subtract (SchroFrame *dest, SchroFrame *src);

void schro_opengl_frame_inverse_iwt_transform (SchroFrame *frame,
    SchroParams *params);

void schro_opengl_upsampled_frame_upsample (SchroUpsampledFrame *upsampled_frame);

void schro_frame_print (SchroFrame *frame, const char* name);

SCHRO_END_DECLS

#endif

