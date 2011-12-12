#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>
#include <schroedinger/schrogpuframe.h>

#include <stdlib.h>
#include <string.h>

#include <cuda_runtime_api.h>
#include "cudawavelet.h"
#include "cudaframe.h"
#include "cudaupsample.h"
#include "cudamotion.h"
#include <stdio.h>

#define CUDA_STREAM 0

int
schro_bpp (int format)
{
  int bytes_pp;

  if (SCHRO_FRAME_IS_PACKED (format)) {
    switch (format) {
      case SCHRO_FRAME_FORMAT_YUYV:
      case SCHRO_FRAME_FORMAT_UYVY:
        bytes_pp = 2;
        break;
      case SCHRO_FRAME_FORMAT_AYUV:
        bytes_pp = 4;
        break;
      default:
        SCHRO_ASSERT (0);
    }
  } else {
    switch (SCHRO_FRAME_FORMAT_DEPTH (format)) {
      case SCHRO_FRAME_FORMAT_DEPTH_U8:
        bytes_pp = 1;
        break;
      case SCHRO_FRAME_FORMAT_DEPTH_S16:
        bytes_pp = 2;
        break;
      case SCHRO_FRAME_FORMAT_DEPTH_S32:
        bytes_pp = 4;
        break;
      default:
        SCHRO_ASSERT (0);
        bytes_pp = 0;
        break;
    }
  }
  return bytes_pp;
}

int
schro_components (int format)
{
  int comp;

  if (SCHRO_FRAME_IS_PACKED (format)) {
    // packed
    return 1;
  } else {
    // planar
    return 3;
  }
  return comp;
}

#if 0
SchroFrame *
schro_gpuframe_new_clone (SchroFrame * src)
{
  SchroFrame *frame = schro_frame_new ();
  int i, length;
  void *ptr;

  SCHRO_ASSERT (!SCHRO_FRAME_IS_CUDA (src));

  frame->format = src->format;
  frame->width = src->width;
  frame->height = src->height;

  length =
      src->components[0].length + src->components[1].length +
      src->components[2].length;
  cudaMalloc ((void **) &frame->gregions[0], length);

  SCHRO_DEBUG ("schro_gpuframe_new_clone %i %i %i (%i)", frame->format,
      frame->width, frame->height, length);

  /** Copy components and allocate space */
  ptr = frame->gregions[0];
  for (i = 0; i < 3; ++i) {
    frame->components[i].width = src->components[i].width;
    frame->components[i].height = src->components[i].height;
    frame->components[i].stride = src->components[i].stride;
    frame->components[i].length = src->components[i].length;
    frame->components[i].v_shift = src->components[i].v_shift;
    frame->components[i].h_shift = src->components[i].h_shift;

    if (frame->components[i].length) {
      frame->components[i].data = ptr;
      cudaMemcpy (ptr, src->components[i].data, frame->components[i].length,
          cudaMemcpyHostToDevice);

      ptr += frame->components[i].length;
    }
  }

  return frame;
}

void
_schro_gpuframe_free (SchroFrame * frame)
{
  if (frame->gregions[0]) {
    cudaFree (frame->gregions[0]);
  }
}
#endif

void
schro_gpuframe_convert (SchroFrame * dest, SchroFrame * src)
{
  int i;
  int ret;

  SCHRO_ASSERT (dest != NULL);
  SCHRO_ASSERT (src != NULL);
  SCHRO_ASSERT (SCHRO_FRAME_IS_CUDA (dest));
  SCHRO_ASSERT (SCHRO_FRAME_IS_CUDA (src));

  SCHRO_DEBUG ("schro_gpuframe_convert %ix%i(format %i) <- %ix%i(format %i)",
      dest->width, dest->height, dest->format, src->width, src->height,
      src->format);

  if ((src->format == SCHRO_FRAME_FORMAT_S16_444
          && dest->format == SCHRO_FRAME_FORMAT_U8_444)
      || (src->format == SCHRO_FRAME_FORMAT_S16_422
          && dest->format == SCHRO_FRAME_FORMAT_U8_422)
      || (src->format == SCHRO_FRAME_FORMAT_S16_420
          && dest->format == SCHRO_FRAME_FORMAT_U8_420)) {
    // S16 to U8
    for (i = 0; i < 3; ++i)
      cuda_convert_u8_s16 (dest->components[i].data, dest->components[i].stride,
          dest->components[i].width, dest->components[i].height,
          src->components[i].data, src->components[i].stride,
          src->components[i].width, src->components[i].height, CUDA_STREAM);
  } else if ((src->format == SCHRO_FRAME_FORMAT_U8_444
          && dest->format == SCHRO_FRAME_FORMAT_S16_444)
      || (src->format == SCHRO_FRAME_FORMAT_U8_422
          && dest->format == SCHRO_FRAME_FORMAT_S16_422)
      || (src->format == SCHRO_FRAME_FORMAT_U8_420
          && dest->format == SCHRO_FRAME_FORMAT_S16_420)) {
    // U8 to S16
    for (i = 0; i < 3; ++i)
      cuda_convert_s16_u8 (dest->components[i].data, dest->components[i].stride,
          dest->components[i].width, dest->components[i].height,
          src->components[i].data, src->components[i].stride,
          src->components[i].width, src->components[i].height, CUDA_STREAM);

  } else if ((src->format == SCHRO_FRAME_FORMAT_U8_444
          && dest->format == SCHRO_FRAME_FORMAT_U8_444)
      || (src->format == SCHRO_FRAME_FORMAT_U8_422
          && dest->format == SCHRO_FRAME_FORMAT_U8_422)
      || (src->format == SCHRO_FRAME_FORMAT_U8_420
          && dest->format == SCHRO_FRAME_FORMAT_U8_420)) {
    // U8 to U8
    for (i = 0; i < 3; ++i)
      cuda_convert_u8_u8 (dest->components[i].data, dest->components[i].stride,
          dest->components[i].width, dest->components[i].height,
          src->components[i].data, src->components[i].stride,
          src->components[i].width, src->components[i].height, CUDA_STREAM);
  } else if ((src->format == SCHRO_FRAME_FORMAT_S16_444
          && dest->format == SCHRO_FRAME_FORMAT_S16_444)
      || (src->format == SCHRO_FRAME_FORMAT_S16_422
          && dest->format == SCHRO_FRAME_FORMAT_S16_422)
      || (src->format == SCHRO_FRAME_FORMAT_S16_420
          && dest->format == SCHRO_FRAME_FORMAT_S16_420)) {
    // S16 to S16
    for (i = 0; i < 3; ++i)
      cuda_convert_s16_s16 (dest->components[i].data,
          dest->components[i].stride, dest->components[i].width,
          dest->components[i].height, src->components[i].data,
          src->components[i].stride, src->components[i].width,
          src->components[i].height, CUDA_STREAM);
  } else if (src->format == SCHRO_FRAME_FORMAT_YUYV
      && dest->format == SCHRO_FRAME_FORMAT_U8_422) {
    // deinterleave YUYV
    cuda_convert_u8_422_yuyv (dest->components[0].data,
        dest->components[0].stride, dest->components[1].data,
        dest->components[1].stride, dest->components[2].data,
        dest->components[2].stride, dest->width, dest->height,
        src->components[0].data, src->components[0].stride, src->width,
        src->height, CUDA_STREAM);
  } else if (src->format == SCHRO_FRAME_FORMAT_UYVY
      && dest->format == SCHRO_FRAME_FORMAT_U8_422) {
    // deinterleave UYVY
    cuda_convert_u8_422_uyvy (dest->components[0].data,
        dest->components[0].stride, dest->components[1].data,
        dest->components[1].stride, dest->components[2].data,
        dest->components[2].stride, dest->width, dest->height,
        src->components[0].data, src->components[0].stride, src->width,
        src->height, CUDA_STREAM);
  } else if (src->format == SCHRO_FRAME_FORMAT_AYUV
      && dest->format == SCHRO_FRAME_FORMAT_U8_444) {
    // deinterleave AYUV
    cuda_convert_u8_444_ayuv (dest->components[0].data,
        dest->components[0].stride, dest->components[1].data,
        dest->components[1].stride, dest->components[2].data,
        dest->components[2].stride, dest->width, dest->height,
        src->components[0].data, src->components[0].stride, src->width,
        src->height, CUDA_STREAM);
  } else if (src->format == SCHRO_FRAME_FORMAT_U8_422
      && dest->format == SCHRO_FRAME_FORMAT_YUYV) {
    // interleave YUYV
    cuda_convert_yuyv_u8_422 (dest->components[0].data,
        dest->components[0].stride, dest->width, dest->height,
        src->components[0].data, src->components[0].stride,
        src->components[1].data, src->components[1].stride,
        src->components[2].data, src->components[2].stride, src->width,
        src->height, CUDA_STREAM);
  } else if (src->format == SCHRO_FRAME_FORMAT_U8_422
      && dest->format == SCHRO_FRAME_FORMAT_UYVY) {
    // interleave UYVY
    cuda_convert_uyvy_u8_422 (dest->components[0].data,
        dest->components[0].stride, dest->width, dest->height,
        src->components[0].data, src->components[0].stride,
        src->components[1].data, src->components[1].stride,
        src->components[2].data, src->components[2].stride, src->width,
        src->height, CUDA_STREAM);
  } else if (src->format == SCHRO_FRAME_FORMAT_U8_444
      && dest->format == SCHRO_FRAME_FORMAT_AYUV) {
    // interleave AYUV
    cuda_convert_ayuv_u8_444 (dest->components[0].data,
        dest->components[0].stride, dest->width, dest->height,
        src->components[0].data, src->components[0].stride,
        src->components[1].data, src->components[1].stride,
        src->components[2].data, src->components[2].stride, src->width,
        src->height, CUDA_STREAM);
  } else {
    SCHRO_ERROR ("conversion unimplemented");
    SCHRO_ASSERT (0);
  }

  ret = cudaThreadSynchronize ();
  if (ret != 0) {
    SCHRO_ERROR ("thread sync %d", ret);
  }
  SCHRO_ASSERT (ret == 0);
}

void
schro_gpuframe_add (SchroFrame * dest, SchroFrame * src)
{
  int i;
  int ret;

  //SCHRO_ASSERT(schro_async_get_exec_domain () == SCHRO_EXEC_DOMAIN_CUDA);
  SCHRO_ASSERT (dest != NULL);
  SCHRO_ASSERT (src != NULL);
  SCHRO_ASSERT (SCHRO_FRAME_IS_CUDA (dest));
  SCHRO_ASSERT (SCHRO_FRAME_IS_CUDA (src));

  SCHRO_DEBUG ("schro_gpuframe_add %ix%i(format %i) <- %ix%i(format %i)",
      dest->width, dest->height, dest->format, src->width, src->height,
      src->format);

  if ((src->format == SCHRO_FRAME_FORMAT_U8_444
          && dest->format == SCHRO_FRAME_FORMAT_S16_444)
      || (src->format == SCHRO_FRAME_FORMAT_U8_422
          && dest->format == SCHRO_FRAME_FORMAT_S16_422)
      || (src->format == SCHRO_FRAME_FORMAT_U8_420
          && dest->format == SCHRO_FRAME_FORMAT_S16_420)) {
    // U8 to S16
    for (i = 0; i < 3; ++i)
      cuda_add_s16_u8 (dest->components[i].data, dest->components[i].stride,
          dest->components[i].width, dest->components[i].height,
          src->components[i].data, src->components[i].stride,
          src->components[i].width, src->components[i].height, CUDA_STREAM);
  } else if ((src->format == SCHRO_FRAME_FORMAT_S16_444
          && dest->format == SCHRO_FRAME_FORMAT_S16_444)
      || (src->format == SCHRO_FRAME_FORMAT_S16_422
          && dest->format == SCHRO_FRAME_FORMAT_S16_422)
      || (src->format == SCHRO_FRAME_FORMAT_S16_420
          && dest->format == SCHRO_FRAME_FORMAT_S16_420)) {
    // S16 to S16
    for (i = 0; i < 3; ++i)
      cuda_add_s16_s16 (dest->components[i].data, dest->components[i].stride,
          dest->components[i].width, dest->components[i].height,
          src->components[i].data, src->components[i].stride,
          src->components[i].width, src->components[i].height, CUDA_STREAM);
  } else {
    SCHRO_ERROR ("add function unimplemented");
    SCHRO_ASSERT (0);
  }

  ret = cudaThreadSynchronize ();
  if (ret != 0) {
    SCHRO_ERROR ("thread sync %d", ret);
  }
  SCHRO_ASSERT (ret == 0);
}

void
schro_gpuframe_subtract (SchroFrame * dest, SchroFrame * src)
{
  int i;

  //SCHRO_ASSERT(schro_async_get_exec_domain () == SCHRO_EXEC_DOMAIN_CUDA);
  SCHRO_ASSERT (dest != NULL);
  SCHRO_ASSERT (src != NULL);
  SCHRO_ASSERT (SCHRO_FRAME_IS_CUDA (dest));
  SCHRO_ASSERT (SCHRO_FRAME_IS_CUDA (src));

  SCHRO_DEBUG ("schro_gpuframe_subtract %ix%i(format %i) <- %ix%i(format %i)",
      dest->width, dest->height, dest->format, src->width, src->height,
      src->format);

  if ((src->format == SCHRO_FRAME_FORMAT_U8_444
          && dest->format == SCHRO_FRAME_FORMAT_S16_444)
      || (src->format == SCHRO_FRAME_FORMAT_U8_422
          && dest->format == SCHRO_FRAME_FORMAT_S16_422)
      || (src->format == SCHRO_FRAME_FORMAT_U8_420
          && dest->format == SCHRO_FRAME_FORMAT_S16_420)) {
    // U8 to S16
    for (i = 0; i < 3; ++i)
      cuda_subtract_s16_u8 (dest->components[i].data,
          dest->components[i].stride, dest->components[i].width,
          dest->components[i].height, src->components[i].data,
          src->components[i].stride, src->components[i].width,
          src->components[i].height, CUDA_STREAM);
  } else if ((src->format == SCHRO_FRAME_FORMAT_S16_444
          && dest->format == SCHRO_FRAME_FORMAT_S16_444)
      || (src->format == SCHRO_FRAME_FORMAT_S16_422
          && dest->format == SCHRO_FRAME_FORMAT_S16_422)
      || (src->format == SCHRO_FRAME_FORMAT_S16_420
          && dest->format == SCHRO_FRAME_FORMAT_S16_420)) {
    // S16 to S16
    for (i = 0; i < 3; ++i)
      cuda_subtract_s16_s16 (dest->components[i].data,
          dest->components[i].stride, dest->components[i].width,
          dest->components[i].height, src->components[i].data,
          src->components[i].stride, src->components[i].width,
          src->components[i].height, CUDA_STREAM);
  } else {
    SCHRO_ERROR ("add function unimplemented");
    SCHRO_ASSERT (0);
  }
}


void
schro_gpuframe_iwt_transform (SchroFrame * frame, SchroParams * params)
{
  int16_t *frame_data;
  int component;
  int width;
  int height;
  int level;

  //SCHRO_ASSERT(schro_async_get_exec_domain () == SCHRO_EXEC_DOMAIN_CUDA);

  SCHRO_DEBUG ("schro_gpuframe_iwt_transform %ix%i (%i levels)", frame->width,
      frame->height, params->transform_depth);

  SCHRO_ASSERT (SCHRO_FRAME_IS_CUDA (frame));

  for (component = 0; component < 3; component++) {
    SchroFrameData *comp = &frame->components[component];

    if (component == 0) {
      width = params->iwt_luma_width;
      height = params->iwt_luma_height;
    } else {
      width = params->iwt_chroma_width;
      height = params->iwt_chroma_height;
    }

    frame_data = (int16_t *) comp->data;
    for (level = 0; level < params->transform_depth; level++) {
      int w;
      int h;
      int stride;

      w = width >> level;
      h = height >> level;
      stride = comp->stride << level;

      cuda_wavelet_transform_2d (params->wavelet_filter_index, frame_data,
          stride, w, h, CUDA_STREAM);
    }
  }
}

void
schro_gpuframe_inverse_iwt_transform (SchroFrame * frame, SchroParams * params)
{
  int16_t *frame_data;
  int width;
  int height;
  int level;
  int component;
  int ret;

  //SCHRO_ASSERT(schro_async_get_exec_domain () == SCHRO_EXEC_DOMAIN_CUDA);

  SCHRO_DEBUG
      ("schro_gpuframe_inverse_iwt_transform %ix%i, filter %i, %i levels",
      frame->width, frame->height, params->wavelet_filter_index,
      params->transform_depth);
#ifdef TEST
  int16_t *c_data;
  int x;

  c_data = schro_malloc (frame->components[0].stride * params->iwt_luma_height);
#endif

  SCHRO_ASSERT (SCHRO_FRAME_IS_CUDA (frame));

  for (component = 0; component < 3; component++) {
    SchroFrameData *comp = &frame->components[component];

    if (component == 0) {
      width = params->iwt_luma_width;
      height = params->iwt_luma_height;
    } else {
      width = params->iwt_chroma_width;
      height = params->iwt_chroma_height;
    }

    frame_data = (int16_t *) comp->data;
#ifdef TEST
    /// Copy frame from GPU
    cudaMemcpy (c_data, frame_data, height * comp->stride,
        cudaMemcpyDeviceToHost);
    for (x = 0; x < width; ++x) {
      fprintf (stderr, "%i ", c_data[x]);
    }
    fprintf (stderr, "\n");
#endif

    for (level = params->transform_depth - 1; level >= 0; level--) {
      int w;
      int h;
      int stride;

      w = width >> level;
      h = height >> level;
      stride = comp->stride << level;

      cuda_wavelet_inverse_transform_2d (params->wavelet_filter_index,
          frame_data, stride, w, h, CUDA_STREAM);
    }
#ifdef TEST
    /// Copy frame from GPU
    cudaMemcpy (c_data, frame_data, height * comp->stride,
        cudaMemcpyDeviceToHost);
    for (x = 0; x < width; ++x) {
      fprintf (stderr, "%i ", c_data[x]);
    }
    fprintf (stderr, "\n");
    fprintf (stderr, "---------------------\n");
#endif
  }
#ifdef TEST
  schro_free (c_data);
#endif

  ret = cudaThreadSynchronize ();
  if (ret != 0) {
    SCHRO_ERROR ("thread sync %d", ret);
  }
  SCHRO_ASSERT (ret == 0);
}

void
schro_gpuframe_to_cpu (SchroFrame * dest, SchroFrame * src)
{
  int i;
  int bpp;
  int ret;

  //SCHRO_ASSERT(schro_async_get_exec_domain () == SCHRO_EXEC_DOMAIN_CUDA);
  SCHRO_ASSERT (SCHRO_FRAME_IS_CUDA (src));
  SCHRO_ASSERT (!SCHRO_FRAME_IS_CUDA (dest));

  bpp = schro_bpp (dest->format);
  SCHRO_DEBUG ("schro_gpuframe_to_cpu %ix%i (%d)", dest->width, dest->height);

    /** Format, components and dimensions must match exactly */
  SCHRO_ASSERT (src->format == dest->format);
  for (i = 0; i < 3; ++i) {
    if (src->components[i].data) {
      SCHRO_ASSERT (dest->components[i].data);
      //SCHRO_ASSERT(dest->components[i].stride==src->components[i].stride && dest->components[i].length==src->components[i].length);
      SCHRO_ASSERT (dest->components[i].width == src->components[i].width
          && dest->components[i].height == src->components[i].height);
    }
  }

  /** If the buffer is consecutive, move it in one pass */
  if (src->components[1].data ==
      (src->components[0].data + src->components[0].length)
      && src->components[2].data ==
      (src->components[1].data + src->components[1].length)
      && dest->components[1].data ==
      (dest->components[0].data + dest->components[0].length)
      && dest->components[2].data ==
      (dest->components[1].data + dest->components[1].length)
      && src->components[0].length == dest->components[0].length
      && src->components[1].length == dest->components[1].length
      && src->components[2].length == dest->components[2].length) {
    SCHRO_DEBUG ("consecutive %i+%i+%i", src->components[0].length,
        src->components[1].length, src->components[2].length);
    cudaMemcpy (dest->components[0].data, src->components[0].data,
        src->components[0].length + src->components[1].length +
        src->components[2].length, cudaMemcpyDeviceToHost);
  } else {
    for (i = 0; i < 3; ++i) {
      if (src->components[i].data) {
        SCHRO_DEBUG ("component %i: %p %i %p %i %i %i",
            i,
            dest->components[i].data, dest->components[i].stride,
            src->components[i].data, src->components[i].stride,
            src->components[i].width * bpp, src->components[i].height);

        cudaMemcpy2D (dest->components[i].data, dest->components[i].stride,
            src->components[i].data, src->components[i].stride,
            src->components[i].width * bpp, src->components[i].height,
            cudaMemcpyDeviceToHost);
      }
    }
  }

  ret = cudaThreadSynchronize ();
  if (ret != 0) {
    SCHRO_ERROR ("thread sync %d", ret);
  }
  SCHRO_ASSERT (ret == 0);
}

void
schro_frame_to_gpu (SchroFrame * dest, SchroFrame * src)
{
  int i;
  int bpp;
  int ret;

  //SCHRO_ASSERT(schro_async_get_exec_domain () == SCHRO_EXEC_DOMAIN_CUDA);
  SCHRO_ASSERT (!SCHRO_FRAME_IS_CUDA (src));
  SCHRO_ASSERT (SCHRO_FRAME_IS_CUDA (dest));

  bpp = schro_bpp (dest->format);
  SCHRO_DEBUG ("schro_frame_to_gpu %ix%i (%d)", dest->width, dest->height);

    /** Format, components and dimensions must match exactly */
  SCHRO_ASSERT (src->format == dest->format);
  for (i = 0; i < 3; ++i) {
    if (src->components[i].data) {
      SCHRO_ASSERT (dest->components[i].data);
      //SCHRO_ASSERT(dest->components[i].stride==src->components[i].stride && dest->components[i].length==src->components[i].length);
      SCHRO_ASSERT (dest->components[i].width == src->components[i].width
          && dest->components[i].height == src->components[i].height);
    }
  }
    /** If the buffer is consecutive, move it in one pass */
  if (src->components[1].data ==
      (src->components[0].data + src->components[0].length)
      && src->components[2].data ==
      (src->components[1].data + src->components[1].length)
      && dest->components[1].data ==
      (dest->components[0].data + dest->components[0].length)
      && dest->components[2].data ==
      (dest->components[1].data + dest->components[1].length)
      && src->components[0].length == dest->components[0].length
      && src->components[1].length == dest->components[1].length
      && src->components[2].length == dest->components[2].length) {
    SCHRO_DEBUG ("consecutive %i+%i+%i", src->components[0].length,
        src->components[1].length, src->components[2].length);
    cudaMemcpy (dest->components[0].data, src->components[0].data,
        src->components[0].length + src->components[1].length +
        src->components[2].length, cudaMemcpyHostToDevice);
  } else {
    for (i = 0; i < 3; ++i) {
      if (src->components[i].data) {
        //cudaMemcpy(dest->components[i].data, src->components[i].data, src->components[i].length, cudaMemcpyHostToDevice);
        SCHRO_DEBUG ("component %i: %p %i %p %i %i %i",
            i,
            dest->components[i].data, dest->components[i].stride,
            src->components[i].data, src->components[i].stride,
            src->components[i].width * bpp, src->components[i].height);
        cudaMemcpy2D (dest->components[i].data, dest->components[i].stride,
            src->components[i].data, src->components[i].stride,
            src->components[i].width * bpp, src->components[i].height,
            cudaMemcpyHostToDevice);
      }
    }
  }

  ret = cudaThreadSynchronize ();
  if (ret != 0) {
    SCHRO_ERROR ("thread sync %d", ret);
  }
  SCHRO_ASSERT (ret == 0);
}

void
schro_gpuframe_compare (SchroFrame * a, SchroFrame * b)
{
  void *temp;
  int i, bpp;

  SCHRO_ASSERT (a->format == b->format);
  /// Temp buffer
  temp = schro_malloc (b->components[0].length);
  bpp = schro_bpp (a->format);

  SCHRO_ASSERT (SCHRO_FRAME_IS_CUDA (a));
  SCHRO_ASSERT (!SCHRO_FRAME_IS_CUDA (b));

  SCHRO_DEBUG ("schro_gpuframe_compare %ix%ix%i", a->width, a->height, bpp);
  for (i = 0; i < 3; ++i) {
    int y;

    if (a->components[i].data == NULL)
      continue;
    SCHRO_ASSERT (a->components[i].length <= a->components[0].length);
    SCHRO_ASSERT (a->components[i].length == b->components[i].length
        && a->components[i].width == b->components[i].width);

    cudaMemcpy (temp, a->components[i].data, a->components[i].length,
        cudaMemcpyDeviceToHost);

    for (y = 0; y < a->components[i].height; ++y) {
      void *bofs = b->components[i].data + y * b->components[i].stride;
      void *aofs = temp + y * a->components[i].stride;
      int diff = memcmp (bofs, aofs, a->components[i].width * bpp);

      if (diff != 0) {
        int x;

        for (x = 0; x < a->components[i].width; ++x) {
          fprintf (stderr, "%i ", ((int16_t *) aofs)[x]);
        }
        fprintf (stderr, "\n");
        for (x = 0; x < a->components[i].width; ++x) {
          fprintf (stderr, "%i ", ((int16_t *) bofs)[x]);
        }
        fprintf (stderr, "\n");

        SCHRO_ERROR ("Error on line %i of component %i", y, i);
      }
      SCHRO_ASSERT (diff == 0);
    }
  }

  schro_free (temp);
}

void
schro_gpuframe_zero (SchroFrame * dest)
{
  int i;

  SCHRO_ASSERT (SCHRO_FRAME_IS_CUDA (dest));

    /** If the buffer is consecutive, fill it in one pass */
  if (dest->components[1].data ==
      (dest->components[0].data + dest->components[0].length)
      && dest->components[2].data ==
      (dest->components[1].data + dest->components[1].length)) {
    cudaMemset (dest->components[0].data, 0,
        dest->components[0].length + dest->components[1].length +
        dest->components[2].length);
  } else {
        /** Otherwise, fill per component */
    for (i = 0; i < 3; ++i) {
      if (dest->components[i].data)
        cudaMemset (dest->components[i].data, 0, dest->components[i].length);
    }
  }
}


void
schro_gpuframe_upsample (SchroFrame * dst, SchroFrame * src)
{
  int i;

  SCHRO_ASSERT (SCHRO_FRAME_IS_CUDA (dst));
  SCHRO_ASSERT (SCHRO_FRAME_IS_CUDA (src));
  SCHRO_ASSERT (dst->width == src->width * 2 && dst->height == src->height * 2);
  SCHRO_ASSERT (SCHRO_FRAME_FORMAT_DEPTH (src->format) ==
      SCHRO_FRAME_FORMAT_DEPTH_U8);
  SCHRO_ASSERT (src->format == dst->format);

  for (i = 0; i < 3; ++i) {
    uint8_t *dst_data = (uint8_t *) dst->components[i].data;
    int dst_stride = dst->components[i].stride;
    uint8_t *src_data = (uint8_t *) src->components[i].data;
    int src_stride = src->components[i].stride;
    int width = src->components[i].width;
    int height = src->components[i].height;

    cuda_upsample_horizontal (dst_data, dst_stride * 2, src_data, src_stride,
        width, height, CUDA_STREAM);
    cuda_upsample_vertical (dst_data + dst_stride, dst_stride * 2, dst_data,
        dst_stride * 2, width * 2, height, CUDA_STREAM);
  }
}

SchroUpsampledFrame *
schro_upsampled_gpuframe_new (SchroVideoFormat * fmt)
{
  SchroUpsampledFrame *rv;

  SCHRO_DEBUG ("schro_upsampled_gpuframe_new");
  rv = schro_malloc0 (sizeof (SchroUpsampledFrame));


  return rv;
}

void
schro_upsampled_gpuframe_upsample (SchroUpsampledFrame * uf)
{
  struct cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc (8, 0, 0, 0, cudaChannelFormatKindUnsigned);
  SchroFrame *tmp_frame;
  SchroFrame *src = uf->frames[0];
  struct cudaArray *ca;
  int k;

  //SCHRO_ASSERT(schro_async_get_exec_domain () == SCHRO_EXEC_DOMAIN_CUDA);
  SCHRO_ASSERT (SCHRO_FRAME_IS_CUDA (src));

  tmp_frame = schro_frame_new_and_alloc (src->domain, src->format,
      src->width * 2, src->height * 2);

  /** Make an 8 bit texture for each component */
  for (k = 0; k < 3; k++) {
    cudaMallocArray (&ca, &channelDesc,
        src->components[k].width * 2, src->components[k].height * 2);
    uf->components[k] = ca;
  }

  /** Temporary texture must have two times the size of a frame in each dimension */
  schro_gpuframe_upsample (tmp_frame, src);

  /** Copy data to texture */
  for (k = 0; k < 3; k++) {
    cudaMemcpy2DToArray (uf->components[k], 0, 0, tmp_frame->components[k].data,
        tmp_frame->components[k].stride, tmp_frame->components[k].width,
        tmp_frame->components[k].height, cudaMemcpyDeviceToDevice);
  }

  schro_frame_unref (tmp_frame);
}

void
schro_upsampled_gpuframe_free (SchroUpsampledFrame * x)
{
  int i;

  SCHRO_DEBUG ("schro_upsampled_gpuframe_free -- freed");
  for (i = 0; i < 3; ++i)
    cudaFreeArray (x->components[i]);

  //active --;
  schro_free (x);
  //SCHRO_DEBUG("active is now %i", active);
}

#if 0
SchroFrame *
schro_frame_new_and_alloc_locked (SchroFrameFormat format, int width,
    int height)
{
  SchroFrame *frame = schro_frame_new ();
  int bytes_pp;
  int h_shift, v_shift;
  int chroma_width;
  int chroma_height;

  SCHRO_ASSERT (width > 0);
  SCHRO_ASSERT (height > 0);

  /* FIXME this function allocates with cudaMallocHost() but doesn't
   * set the free() function, which means it will be freed using free(). */

  frame->format = format;
  frame->width = width;
  frame->height = height;
  frame->is_cuda_frame = FALSE;
  frame->is_cuda_shared = TRUE;

  switch (SCHRO_FRAME_FORMAT_DEPTH (format)) {
    case SCHRO_FRAME_FORMAT_DEPTH_U8:
      bytes_pp = 1;
      break;
    case SCHRO_FRAME_FORMAT_DEPTH_S16:
      bytes_pp = 2;
      break;
    case SCHRO_FRAME_FORMAT_DEPTH_S32:
      bytes_pp = 4;
      break;
    default:
      SCHRO_ASSERT (0);
      bytes_pp = 0;
      break;
  }

  h_shift = SCHRO_FRAME_FORMAT_H_SHIFT (format);
  v_shift = SCHRO_FRAME_FORMAT_V_SHIFT (format);
  chroma_width = ROUND_UP_SHIFT (width, h_shift);
  chroma_height = ROUND_UP_SHIFT (height, v_shift);

  frame->components[0].width = width;
  frame->components[0].height = height;
  frame->components[0].stride = ROUND_UP_64 (width * bytes_pp);
  frame->components[0].length =
      frame->components[0].stride * frame->components[0].height;
  frame->components[0].v_shift = 0;
  frame->components[0].h_shift = 0;

  frame->components[1].width = chroma_width;
  frame->components[1].height = chroma_height;
  frame->components[1].stride = ROUND_UP_64 (chroma_width * bytes_pp);
  frame->components[1].length =
      frame->components[1].stride * frame->components[1].height;
  frame->components[1].v_shift = v_shift;
  frame->components[1].h_shift = h_shift;

  frame->components[2].width = chroma_width;
  frame->components[2].height = chroma_height;
  frame->components[2].stride = ROUND_UP_64 (chroma_width * bytes_pp);
  frame->components[2].length =
      frame->components[2].stride * frame->components[2].height;
  frame->components[2].v_shift = v_shift;
  frame->components[2].h_shift = h_shift;

  cudaMallocHost ((void **) &frame->regions[0],
      frame->components[0].length + frame->components[1].length +
      frame->components[2].length);
  //frame->regions[0] = schro_malloc (frame->components[0].length +
  //    frame->components[1].length + frame->components[2].length);

  frame->components[0].data = frame->regions[0];
  frame->components[1].data = frame->components[0].data +
      frame->components[0].length;
  frame->components[2].data = frame->components[1].data +
      frame->components[1].length;

  return frame;
}
#endif
