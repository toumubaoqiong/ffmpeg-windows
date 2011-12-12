
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#define SCHRO_ENABLE_UNSTABLE_API 1

#include "schrovirtframe.h"
#include <schroedinger/schro.h>
#include <schroedinger/schroutils.h>
#include <string.h>
#include <math.h>
#include <orc/orc.h>

#include <schroedinger/schroorc.h>


SchroFrame *
schro_frame_new_virtual (SchroMemoryDomain * domain, SchroFrameFormat format,
    int width, int height)
{
  SchroFrame *frame = schro_frame_new ();
  int bytes_pp;
  int h_shift, v_shift;
  int chroma_width;
  int chroma_height;
  int i;

  frame->format = format;
  frame->width = width;
  frame->height = height;
  frame->domain = domain;

  if (SCHRO_FRAME_IS_PACKED (format)) {
    frame->components[0].format = format;
    frame->components[0].width = width;
    frame->components[0].height = height;
    if (format == SCHRO_FRAME_FORMAT_AYUV) {
      frame->components[0].stride = width * 4;
    } else if (format == SCHRO_FRAME_FORMAT_v216) {
      frame->components[0].stride = ROUND_UP_POW2 (width, 1) * 4;
    } else if (format == SCHRO_FRAME_FORMAT_v210) {
      frame->components[0].stride = ((width + 47) / 48) * 128;
    } else {
      frame->components[0].stride = ROUND_UP_POW2 (width, 1) * 2;
    }
    frame->components[0].length = frame->components[0].stride * height;

    frame->components[0].data = frame->regions[0];
    frame->components[0].v_shift = 0;
    frame->components[0].h_shift = 0;

    frame->regions[0] =
        malloc (frame->components[0].stride * SCHRO_FRAME_CACHE_SIZE);
    for (i = 0; i < SCHRO_FRAME_CACHE_SIZE; i++) {
      frame->cached_lines[0][i] = 0;
    }
    frame->is_virtual = TRUE;

    return frame;
  }

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

  frame->components[0].format = format;
  frame->components[0].width = width;
  frame->components[0].height = height;
  frame->components[0].stride = ROUND_UP_4 (width * bytes_pp);
  frame->components[0].length =
      frame->components[0].stride * frame->components[0].height;
  frame->components[0].v_shift = 0;
  frame->components[0].h_shift = 0;

  frame->components[1].format = format;
  frame->components[1].width = chroma_width;
  frame->components[1].height = chroma_height;
  frame->components[1].stride = ROUND_UP_4 (chroma_width * bytes_pp);
  frame->components[1].length =
      frame->components[1].stride * frame->components[1].height;
  frame->components[1].v_shift = v_shift;
  frame->components[1].h_shift = h_shift;

  frame->components[2].format = format;
  frame->components[2].width = chroma_width;
  frame->components[2].height = chroma_height;
  frame->components[2].stride = ROUND_UP_4 (chroma_width * bytes_pp);
  frame->components[2].length =
      frame->components[2].stride * frame->components[2].height;
  frame->components[2].v_shift = v_shift;
  frame->components[2].h_shift = h_shift;

  for (i = 0; i < 3; i++) {
    SchroFrameData *comp = &frame->components[i];
    int j;

    frame->regions[i] = malloc (comp->stride * SCHRO_FRAME_CACHE_SIZE);
    for (j = 0; j < SCHRO_FRAME_CACHE_SIZE; j++) {
      frame->cached_lines[i][j] = 0;
    }
  }
  frame->is_virtual = TRUE;

  return frame;
}

static void
schro_virt_frame_prep_cache_line (SchroFrame * frame, int component, int i)
{
  int j;

  if (i < frame->cache_offset[component]) {
    SCHRO_ERROR ("cache failure: %d outside [%d,%d]", i,
        frame->cache_offset[component],
        frame->cache_offset[component] + SCHRO_FRAME_CACHE_SIZE - 1);

    frame->cache_offset[component] = i;
    for (j = 0; j < SCHRO_FRAME_CACHE_SIZE; j++) {
      frame->cached_lines[component][j] = 0;
    }
  }

  while (i >= frame->cache_offset[component] + SCHRO_FRAME_CACHE_SIZE) {
    j = frame->cache_offset[component] & (SCHRO_FRAME_CACHE_SIZE - 1);
    frame->cached_lines[component][j] = 0;

    frame->cache_offset[component]++;
  }
}

void *
schro_virt_frame_get_line_unrendered (SchroFrame * frame, int component, int i)
{
  SchroFrameData *comp = &frame->components[component];
  int j;

  //SCHRO_ASSERT(i >= 0);
  //SCHRO_ASSERT(i < comp->height);

  if (!frame->is_virtual) {
    return SCHRO_FRAME_DATA_GET_LINE (&frame->components[component], i);
  }

  schro_virt_frame_prep_cache_line (frame, component, i);
  j = i & (SCHRO_FRAME_CACHE_SIZE - 1);

  return SCHRO_OFFSET (frame->regions[component], comp->stride * j);
}

void *
schro_virt_frame_get_line (SchroFrame * frame, int component, int i)
{
  SchroFrameData *comp = &frame->components[component];
  int j;

  //SCHRO_ASSERT(i >= 0);
  //SCHRO_ASSERT(i < comp->height);

  if (!frame->is_virtual) {
    return SCHRO_FRAME_DATA_GET_LINE (&frame->components[component], i);
  }

  schro_virt_frame_prep_cache_line (frame, component, i);
  j = i & (SCHRO_FRAME_CACHE_SIZE - 1);

  if (!frame->cached_lines[component][j]) {
    schro_virt_frame_render_line (frame,
        SCHRO_OFFSET (frame->regions[component], comp->stride * j), component,
        i);
    frame->cached_lines[component][j] = 1;
  }

  return SCHRO_OFFSET (frame->regions[component], comp->stride * j);
}

void
schro_virt_frame_set_line_rendered (SchroFrame * frame, int component, int i)
{
  int j;

  //SCHRO_ASSERT(i >= 0);
  //SCHRO_ASSERT(i < comp->height);
  //SCHRO_ASSERT(frame->is_virtual);

  j = i & (SCHRO_FRAME_CACHE_SIZE - 1);
  frame->cached_lines[component][j] = 1;
}

void
schro_virt_frame_render_line (SchroFrame * frame, void *dest,
    int component, int i)
{
  frame->render_line (frame, dest, component, i);
}

static void
copy (SchroFrame * frame, void *_dest, int component, int i)
{
  uint8_t *dest = _dest;
  uint8_t *src;

  src = schro_virt_frame_get_line (frame, component, i);
  switch (SCHRO_FRAME_FORMAT_DEPTH (frame->format)) {
    case SCHRO_FRAME_FORMAT_DEPTH_U8:
      orc_memcpy (dest, src, frame->components[component].width);
      break;
    case SCHRO_FRAME_FORMAT_DEPTH_S16:
      orc_memcpy (dest, src, frame->components[component].width * 2);
      break;
    default:
      SCHRO_ASSERT (0);
      break;
  }
}

void
schro_virt_frame_render (SchroFrame * frame, SchroFrame * dest)
{
  int i, k;

  SCHRO_ASSERT (frame->width == dest->width);
  SCHRO_ASSERT (frame->height >= dest->height);

  if (frame->is_virtual) {
    for (k = 0; k < 3; k++) {
      SchroFrameData *comp = dest->components + k;

      for (i = 0; i < dest->components[k].height; i++) {
        schro_virt_frame_render_line (frame,
            SCHRO_FRAME_DATA_GET_LINE (comp, i), k, i);
      }
    }
  } else {
    for (k = 0; k < 3; k++) {
      SchroFrameData *comp = dest->components + k;

      for (i = 0; i < dest->components[k].height; i++) {
        copy (frame, SCHRO_FRAME_DATA_GET_LINE (comp, i), k, i);
      }
    }
  }
}

#ifdef unused
void
schro_virt_frame_render_downsample_horiz_cosite (SchroFrame * frame,
    void *_dest, int component, int i)
{
  uint8_t *dest = _dest;
  uint8_t *src;
  int j;
  int n_src;

  src = schro_virt_frame_get_line (frame->virt_frame1, component, i);
  n_src = frame->virt_frame1->components[component].width;

  for (j = 0; j < frame->components[component].width; j++) {
    int x = 0;
    x += 1 * src[CLAMP (j * 2 - 1, 0, n_src - 1)];
    x += 2 * src[CLAMP (j * 2 + 0, 0, n_src - 1)];
    x += 1 * src[CLAMP (j * 2 + 1, 0, n_src - 1)];
    dest[j] = CLAMP ((x + 2) >> 2, 0, 255);
  }
}
#endif

#ifdef unused
void
schro_virt_frame_render_downsample_horiz_halfsite (SchroFrame * frame,
    void *_dest, int component, int i)
{
  uint8_t *dest = _dest;
  uint8_t *src;
  int j;
  int n_src;
  int taps = 4;
  int k;

  src = schro_virt_frame_get_line (frame->virt_frame1, component, i);
  n_src = frame->virt_frame1->components[component].width;

  switch (taps) {
    case 4:
      for (j = 0; j < frame->components[component].width; j++) {
        int x = 0;
        x += 6 * src[CLAMP (j * 2 - 1, 0, n_src - 1)];
        x += 26 * src[CLAMP (j * 2 + 0, 0, n_src - 1)];
        x += 26 * src[CLAMP (j * 2 + 1, 0, n_src - 1)];
        x += 6 * src[CLAMP (j * 2 + 2, 0, n_src - 1)];
        dest[j] = CLAMP ((x + 32) >> 6, 0, 255);
      }
      break;
    case 6:
      for (j = 0; j < frame->components[component].width; j++) {
        int x = 0;
        x += -3 * src[CLAMP (j * 2 - 2, 0, n_src - 1)];
        x += 8 * src[CLAMP (j * 2 - 1, 0, n_src - 1)];
        x += 27 * src[CLAMP (j * 2 + 0, 0, n_src - 1)];
        x += 27 * src[CLAMP (j * 2 + 1, 0, n_src - 1)];
        x += 8 * src[CLAMP (j * 2 + 2, 0, n_src - 1)];
        x += -3 * src[CLAMP (j * 2 + 3, 0, n_src - 1)];
        dest[j] = CLAMP ((x + 32) >> 6, 0, 255);
      }
    case 8:
      for (j = 0; j < frame->components[component].width; j++) {
        int x = 0;
        const int taps8[8] = { -2, -4, 9, 29, 29, 9, -4, -2 };
        for (k = 0; k < 8; k++) {
          x += taps8[k] * src[CLAMP (j * 2 - 3 + k, 0, n_src - 1)];
        }
        dest[j] = CLAMP ((x + 32) >> 6, 0, 255);
      }
      break;
    case 10:
      for (j = 0; j < frame->components[component].width; j++) {
        int x = 0;
        const int taps10[10] = { 1, -2, -5, 9, 29, 29, 9, -5, -2, 1 };
        for (k = 0; k < 10; k++) {
          x += taps10[k] * src[CLAMP (j * 2 - 4 + k, 0, n_src - 1)];
        }
        dest[j] = CLAMP ((x + 32) >> 6, 0, 255);
      }
      break;
    default:
      break;
  }
}
#endif

#ifdef unused
SchroFrame *
schro_virt_frame_new_horiz_downsample (SchroFrame * vf, int cosite)
{
  SchroFrame *virt_frame;

  virt_frame =
      schro_frame_new_virtual (NULL, vf->format, vf->width / 2, vf->height);
  virt_frame->virt_frame1 = vf;
  if (cosite) {
    virt_frame->render_line = schro_virt_frame_render_downsample_horiz_cosite;
  } else {
    virt_frame->render_line = schro_virt_frame_render_downsample_horiz_halfsite;
  }

  return virt_frame;
}
#endif

#ifdef unused
void
schro_virt_frame_render_downsample_vert_cosite (SchroFrame * frame,
    void *_dest, int component, int i)
{
  uint8_t *dest = _dest;
  uint8_t *src1;
  uint8_t *src2;
  uint8_t *src3;
  int j;
  int n_src;

  n_src = frame->virt_frame1->components[component].height;
  src1 = schro_virt_frame_get_line (frame->virt_frame1, component,
      CLAMP (i * 2 - 1, 0, n_src - 1));
  src2 = schro_virt_frame_get_line (frame->virt_frame1, component,
      CLAMP (i * 2 + 0, 0, n_src - 1));
  src3 = schro_virt_frame_get_line (frame->virt_frame1, component,
      CLAMP (i * 2 + 1, 0, n_src - 1));

  for (j = 0; j < frame->components[component].width; j++) {
    int x = 0;
    x += 1 * src1[j];
    x += 2 * src2[j];
    x += 1 * src3[j];
    dest[j] = CLAMP ((x + 2) >> 2, 0, 255);
  }
}
#endif

#ifdef unused
void
schro_virt_frame_render_downsample_vert_halfsite (SchroFrame * frame,
    void *_dest, int component, int i)
{
  uint8_t *dest = _dest;
  uint8_t *src[10];
  int j;
  int n_src;
  int taps = 4;
  int k;

  n_src = frame->virt_frame1->components[component].height;
  for (j = 0; j < taps; j++) {
    src[j] = schro_virt_frame_get_line (frame->virt_frame1, component,
        CLAMP (i * 2 - (taps - 2) / 2 + j, 0, n_src - 1));
  }

  switch (taps) {
    case 4:
      for (j = 0; j < frame->components[component].width; j++) {
        int x = 0;
        x += 6 * src[0][j];
        x += 26 * src[1][j];
        x += 26 * src[2][j];
        x += 6 * src[3][j];
        dest[j] = CLAMP ((x + 32) >> 6, 0, 255);
      }
      break;
    case 6:
      for (j = 0; j < frame->components[component].width; j++) {
        int x = 0;
        x += -3 * src[0][j];
        x += 8 * src[1][j];
        x += 27 * src[2][j];
        x += 27 * src[3][j];
        x += 8 * src[4][j];
        x += -3 * src[5][j];
        dest[j] = CLAMP ((x + 32) >> 6, 0, 255);
      }
      break;
    case 8:
      for (j = 0; j < frame->components[component].width; j++) {
        int x = 0;
        const int taps8[8] = { -2, -4, 9, 29, 29, 9, -4, -2 };
        for (k = 0; k < 8; k++) {
          x += taps8[k] * src[k][j];
        }
        dest[j] = CLAMP ((x + 32) >> 6, 0, 255);
      }
      break;
    case 10:
      for (j = 0; j < frame->components[component].width; j++) {
        int x = 0;
        const int taps10[10] = { 1, -2, -5, 9, 29, 29, 9, -5, -2, 1 };
        //const int taps10[10] = { -1, 1, 6, 11, 15, 15, 11, 6, 1, -1 };
        for (k = 0; k < 10; k++) {
          x += taps10[k] * src[k][j];
        }
        dest[j] = CLAMP ((x + 32) >> 6, 0, 255);
      }
      break;
    default:
      SCHRO_ASSERT (0);
      break;
  }
}
#endif

#ifdef unused
SchroFrame *
schro_virt_frame_new_vert_downsample (SchroFrame * vf, int cosite)
{
  SchroFrame *virt_frame;

  virt_frame =
      schro_frame_new_virtual (NULL, vf->format, vf->width, vf->height / 2);
  virt_frame->virt_frame1 = vf;
  if (cosite) {
    virt_frame->render_line = schro_virt_frame_render_downsample_vert_cosite;
  } else {
    virt_frame->render_line = schro_virt_frame_render_downsample_vert_halfsite;
  }

  return virt_frame;
}
#endif

#ifdef unused
void
get_taps (double *taps, double x)
{
  taps[3] = x * x * (x - 1);
  taps[2] = x * (-x * x + x + 1);
  x = 1 - x;
  taps[1] = x * (-x * x + x + 1);
  taps[0] = x * x * (x - 1);
}
#endif

#ifdef unused
void
schro_virt_frame_render_resample_vert (SchroFrame * frame, void *_dest,
    int component, int i)
{
  uint8_t *dest = _dest;
  uint8_t *src1;
  uint8_t *src2;
  uint8_t *src3;
  uint8_t *src4;
  int j;
  int n_src;
  double taps[4];
  double *scale = (double *) frame->virt_priv;
  double x;
  int src_i;

  x = (*scale) * i;
  src_i = floor (x);
  get_taps (taps, x - floor (x));

  n_src = frame->virt_frame1->components[component].height;
  src1 = schro_virt_frame_get_line (frame->virt_frame1, component,
      CLAMP (src_i - 1, 0, n_src - 1));
  src2 = schro_virt_frame_get_line (frame->virt_frame1, component,
      CLAMP (src_i + 0, 0, n_src - 1));
  src3 = schro_virt_frame_get_line (frame->virt_frame1, component,
      CLAMP (src_i + 1, 0, n_src - 1));
  src4 = schro_virt_frame_get_line (frame->virt_frame1, component,
      CLAMP (src_i + 2, 0, n_src - 1));

  for (j = 0; j < frame->components[component].width; j++) {
    double x = 0;
    x += taps[0] * src1[j];
    x += taps[1] * src2[j];
    x += taps[2] * src3[j];
    x += taps[3] * src4[j];
    dest[j] = CLAMP (rint (x), 0, 255);
  }
}
#endif

#ifdef unused
SchroFrame *
schro_virt_frame_new_vert_resample (SchroFrame * vf, int height)
{
  SchroFrame *virt_frame;
  double *scale;

  virt_frame = schro_frame_new_virtual (NULL, vf->format, vf->width, height);
  virt_frame->virt_frame1 = vf;
  virt_frame->render_line = schro_virt_frame_render_resample_vert;

  scale = malloc (sizeof (double));
  virt_frame->virt_priv = scale;

  *scale = (double) vf->height / height;

  return virt_frame;
}
#endif

#ifdef unused
void
schro_virt_frame_render_resample_horiz (SchroFrame * frame, void *_dest,
    int component, int i)
{
  uint8_t *dest = _dest;
  uint8_t *src;
  int j;
  int n_src;
  double taps[4];
  double *scale = (double *) frame->virt_priv;
  int src_i;

  n_src = frame->virt_frame1->components[component].width;
  src = schro_virt_frame_get_line (frame->virt_frame1, component, i);

  for (j = 0; j < frame->components[component].width; j++) {
    double x;
    double y = 0;

    x = (*scale) * j;
    src_i = floor (x);
    get_taps (taps, x - floor (x));

    y = 0;
    y += taps[0] * src[CLAMP (src_i - 1, 0, n_src - 1)];
    y += taps[1] * src[CLAMP (src_i + 0, 0, n_src - 1)];
    y += taps[2] * src[CLAMP (src_i + 1, 0, n_src - 1)];
    y += taps[3] * src[CLAMP (src_i + 2, 0, n_src - 1)];
    dest[j] = CLAMP (rint (y), 0, 255);
  }
}
#endif

#ifdef unused
SchroFrame *
schro_virt_frame_new_horiz_resample (SchroFrame * vf, int width)
{
  SchroFrame *virt_frame;
  double *scale;

  virt_frame = schro_frame_new_virtual (NULL, vf->format, width, vf->height);
  virt_frame->virt_frame1 = vf;
  virt_frame->render_line = schro_virt_frame_render_resample_horiz;

  scale = malloc (sizeof (double));
  virt_frame->virt_priv = scale;

  *scale = (double) vf->width / width;

  return virt_frame;
}
#endif

static void
unpack_yuyv (SchroFrame * frame, void *_dest, int component, int i)
{
  uint8_t *dest = _dest;
  uint8_t *src;

  src = schro_virt_frame_get_line (frame->virt_frame1, 0, i);

  switch (component) {
    case 0:
      orc_unpack_yuyv_y (dest, (void *) src, frame->width);
      break;
    case 1:
      orc_unpack_yuyv_u (dest, (void *) src, frame->width / 2);
      break;
    case 2:
      orc_unpack_yuyv_v (dest, (void *) src, frame->width / 2);
      break;
    default:
      SCHRO_ASSERT (0);
  }
}

static void
unpack_uyvy (SchroFrame * frame, void *_dest, int component, int i)
{
  uint8_t *dest = _dest;
  uint8_t *src;

  src = schro_virt_frame_get_line (frame->virt_frame1, 0, i);

  switch (component) {
    case 0:
      orc_unpack_uyvy_y (dest, (void *) src, frame->width);
      break;
    case 1:
      orc_unpack_uyvy_u (dest, (void *) src, frame->width / 2);
      break;
    case 2:
      orc_unpack_uyvy_v (dest, (void *) src, frame->width / 2);
      break;
    default:
      SCHRO_ASSERT (0);
  }
}

static void
unpack_ayuv (SchroFrame * frame, void *_dest, int component, int i)
{
  uint8_t *dest = _dest;
  uint8_t *src;
  int j;

  src = schro_virt_frame_get_line (frame->virt_frame1, 0, i);

  switch (component) {
    case 0:
      for (j = 0; j < frame->width; j++) {
        dest[j] = src[j * 4 + 1];
      }
      break;
    case 1:
      for (j = 0; j < frame->width; j++) {
        dest[j] = src[j * 4 + 2];
      }
      break;
    case 2:
      for (j = 0; j < frame->width; j++) {
        dest[j] = src[j * 4 + 3];
      }
      break;
    default:
      SCHRO_ASSERT (0);
  }
}

static void
unpack_v210 (SchroFrame * frame, void *_dest, int component, int i)
{
  uint8_t *dest = _dest;
  uint8_t *src;
  int j;

  src = schro_virt_frame_get_line (frame->virt_frame1, 0, i);

#define READ_UINT32_LE(a) (((uint8_t *)(a))[0] | (((uint8_t *)(a))[1]<<8) | \
  (((uint8_t *)(a))[2]<<16) | (((uint8_t *)(a))[3]<<24))
  switch (component) {
    case 0:
      for (j = 0; j < frame->width / 6; j++) {
        dest[j * 6 + 0] =
            ((READ_UINT32_LE (src + j * 16 + 0) >> 10) & 0x3ff) >> 2;
        dest[j * 6 + 1] =
            ((READ_UINT32_LE (src + j * 16 + 4) >> 0) & 0x3ff) >> 2;
        dest[j * 6 + 2] =
            ((READ_UINT32_LE (src + j * 16 + 4) >> 20) & 0x3ff) >> 2;
        dest[j * 6 + 3] =
            ((READ_UINT32_LE (src + j * 16 + 8) >> 10) & 0x3ff) >> 2;
        dest[j * 6 + 4] =
            ((READ_UINT32_LE (src + j * 16 + 12) >> 0) & 0x3ff) >> 2;
        dest[j * 6 + 5] =
            ((READ_UINT32_LE (src + j * 16 + 12) >> 20) & 0x3ff) >> 2;
      }
      if (j * 6 + 0 < frame->width) {
        dest[j * 6 + 0] =
            ((READ_UINT32_LE (src + j * 16 + 0) >> 10) & 0x3ff) >> 2;
      }
      if (j * 6 + 1 < frame->width) {
        dest[j * 6 + 1] =
            ((READ_UINT32_LE (src + j * 16 + 4) >> 0) & 0x3ff) >> 2;
      }
      if (j * 6 + 2 < frame->width) {
        dest[j * 6 + 2] =
            ((READ_UINT32_LE (src + j * 16 + 4) >> 20) & 0x3ff) >> 2;
      }
      if (j * 6 + 3 < frame->width) {
        dest[j * 6 + 3] =
            ((READ_UINT32_LE (src + j * 16 + 8) >> 10) & 0x3ff) >> 2;
      }
      if (j * 6 + 4 < frame->width) {
        dest[j * 6 + 4] =
            ((READ_UINT32_LE (src + j * 16 + 12) >> 0) & 0x3ff) >> 2;
      }
      if (j * 6 + 5 < frame->width) {
        dest[j * 6 + 5] =
            ((READ_UINT32_LE (src + j * 16 + 12) >> 20) & 0x3ff) >> 2;
      }
      break;
    case 1:
      for (j = 0; j < frame->width / 6; j++) {
        dest[j * 3 + 0] =
            ((READ_UINT32_LE (src + j * 16 + 0) >> 0) & 0x3ff) >> 2;
        dest[j * 3 + 1] =
            ((READ_UINT32_LE (src + j * 16 + 4) >> 10) & 0x3ff) >> 2;
        dest[j * 3 + 2] =
            ((READ_UINT32_LE (src + j * 16 + 8) >> 20) & 0x3ff) >> 2;
      }
      if (j * 6 + 0 < frame->width) {
        dest[j * 3 + 0] =
            ((READ_UINT32_LE (src + j * 16 + 0) >> 0) & 0x3ff) >> 2;
      }
      if (j * 6 + 2 < frame->width) {
        dest[j * 3 + 1] =
            ((READ_UINT32_LE (src + j * 16 + 4) >> 10) & 0x3ff) >> 2;
      }
      if (j * 6 + 4 < frame->width) {
        dest[j * 3 + 2] =
            ((READ_UINT32_LE (src + j * 16 + 8) >> 20) & 0x3ff) >> 2;
      }
      break;
    case 2:
      for (j = 0; j < frame->width / 6; j++) {
        dest[j * 3 + 0] =
            ((READ_UINT32_LE (src + j * 16 + 0) >> 20) & 0x3ff) >> 2;
        dest[j * 3 + 1] =
            ((READ_UINT32_LE (src + j * 16 + 8) >> 0) & 0x3ff) >> 2;
        dest[j * 3 + 2] =
            ((READ_UINT32_LE (src + j * 16 + 12) >> 10) & 0x3ff) >> 2;
      }
      if (j * 6 + 0 < frame->width) {
        dest[j * 3 + 0] =
            ((READ_UINT32_LE (src + j * 16 + 0) >> 20) & 0x3ff) >> 2;
      }
      if (j * 6 + 2 < frame->width) {
        dest[j * 3 + 1] =
            ((READ_UINT32_LE (src + j * 16 + 8) >> 0) & 0x3ff) >> 2;
      }
      if (j * 6 + 4 < frame->width) {
        dest[j * 3 + 2] =
            ((READ_UINT32_LE (src + j * 16 + 12) >> 10) & 0x3ff) >> 2;
      }
      break;
    default:
      SCHRO_ASSERT (0);
  }
}

static void
unpack_v216 (SchroFrame * frame, void *_dest, int component, int i)
{
  uint8_t *dest = _dest;
  uint8_t *src;
  int j;

  src = schro_virt_frame_get_line (frame->virt_frame1, 0, i);

  switch (component) {
    case 0:
      for (j = 0; j < frame->width; j++) {
        dest[j] = src[j * 4 + 2 + 1];
      }
      break;
    case 1:
      for (j = 0; j < frame->width / 2; j++) {
        dest[j] = src[j * 8 + 0 + 1];
      }
      break;
    case 2:
      for (j = 0; j < frame->width / 2; j++) {
        dest[j] = src[j * 8 + 4 + 1];
      }
      break;
    default:
      SCHRO_ASSERT (0);
  }
}

SchroFrame *
schro_virt_frame_new_unpack (SchroFrame * vf)
{
  SchroFrame *virt_frame;
  SchroFrameFormat format;
  SchroFrameRenderFunc render_line;

  switch (vf->format) {
    case SCHRO_FRAME_FORMAT_YUYV:
      format = SCHRO_FRAME_FORMAT_U8_422;
      render_line = unpack_yuyv;
      break;
    case SCHRO_FRAME_FORMAT_UYVY:
      format = SCHRO_FRAME_FORMAT_U8_422;
      render_line = unpack_uyvy;
      break;
    case SCHRO_FRAME_FORMAT_AYUV:
      format = SCHRO_FRAME_FORMAT_U8_444;
      render_line = unpack_ayuv;
      break;
    case SCHRO_FRAME_FORMAT_v210:
      format = SCHRO_FRAME_FORMAT_U8_422;
      render_line = unpack_v210;
      break;
    case SCHRO_FRAME_FORMAT_v216:
      format = SCHRO_FRAME_FORMAT_U8_422;
      render_line = unpack_v216;
      break;
    default:
      return vf;
  }

  virt_frame = schro_frame_new_virtual (NULL, format, vf->width, vf->height);
  virt_frame->virt_frame1 = vf;
  virt_frame->render_line = render_line;

  return virt_frame;
}


static void
pack_yuyv (SchroFrame * frame, void *_dest, int component, int i)
{
  uint32_t *dest = _dest;
  uint8_t *src_y;
  uint8_t *src_u;
  uint8_t *src_v;

  src_y = schro_virt_frame_get_line (frame->virt_frame1, 0, i);
  src_u = schro_virt_frame_get_line (frame->virt_frame1, 1, i);
  src_v = schro_virt_frame_get_line (frame->virt_frame1, 2, i);

  orc_packyuyv (dest, src_y, src_u, src_v, frame->width / 2);
}


SchroFrame *
schro_virt_frame_new_pack_YUY2 (SchroFrame * vf)
{
  SchroFrame *virt_frame;

  virt_frame = schro_frame_new_virtual (NULL, SCHRO_FRAME_FORMAT_YUYV,
      vf->width, vf->height);
  virt_frame->virt_frame1 = vf;
  virt_frame->render_line = pack_yuyv;

  return virt_frame;
}

static void
pack_uyvy (SchroFrame * frame, void *_dest, int component, int i)
{
  uint8_t *dest = _dest;
  uint8_t *src_y;
  uint8_t *src_u;
  uint8_t *src_v;
  int j;

  src_y = schro_virt_frame_get_line (frame->virt_frame1, 0, i);
  src_u = schro_virt_frame_get_line (frame->virt_frame1, 1, i);
  src_v = schro_virt_frame_get_line (frame->virt_frame1, 2, i);

  for (j = 0; j < frame->width / 2; j++) {
    dest[j * 4 + 1] = src_y[j * 2 + 0];
    dest[j * 4 + 3] = src_y[j * 2 + 1];
    dest[j * 4 + 0] = src_u[j];
    dest[j * 4 + 2] = src_v[j];
  }
}

SchroFrame *
schro_virt_frame_new_pack_UYVY (SchroFrame * vf)
{
  SchroFrame *virt_frame;

  virt_frame = schro_frame_new_virtual (NULL, SCHRO_FRAME_FORMAT_YUYV,
      vf->width, vf->height);
  virt_frame->virt_frame1 = vf;
  virt_frame->render_line = pack_uyvy;

  return virt_frame;
}

static void
pack_v216 (SchroFrame * frame, void *_dest, int component, int i)
{
  uint8_t *dest = _dest;
  uint8_t *src_y;
  uint8_t *src_u;
  uint8_t *src_v;
  int j;

  src_y = schro_virt_frame_get_line (frame->virt_frame1, 0, i);
  src_u = schro_virt_frame_get_line (frame->virt_frame1, 1, i);
  src_v = schro_virt_frame_get_line (frame->virt_frame1, 2, i);

  for (j = 0; j < frame->width / 2; j++) {
    dest[j * 8 + 0] = src_u[j];
    dest[j * 8 + 1] = src_u[j];
    dest[j * 8 + 2] = src_y[j * 2 + 0];
    dest[j * 8 + 3] = src_y[j * 2 + 0];
    dest[j * 8 + 4] = src_v[j];
    dest[j * 8 + 5] = src_v[j];
    dest[j * 8 + 6] = src_y[j * 2 + 1];
    dest[j * 8 + 7] = src_y[j * 2 + 1];
  }
}

SchroFrame *
schro_virt_frame_new_pack_v216 (SchroFrame * vf)
{
  SchroFrame *virt_frame;

  virt_frame = schro_frame_new_virtual (NULL, SCHRO_FRAME_FORMAT_v216,
      vf->width, vf->height);
  virt_frame->virt_frame1 = vf;
  virt_frame->render_line = pack_v216;

  return virt_frame;
}

static void
pack_v210_s16 (SchroFrame * frame, void *_dest, int component, int i)
{
  uint8_t *dest = _dest;
  int16_t *src_y;
  int16_t *src_u;
  int16_t *src_v;
  int j;
  uint32_t val;

  src_y = schro_virt_frame_get_line (frame->virt_frame1, 0, i);
  src_u = schro_virt_frame_get_line (frame->virt_frame1, 1, i);
  src_v = schro_virt_frame_get_line (frame->virt_frame1, 2, i);

#define TO_10(x) (((x) + 512)&0x3ff)
#define WRITE_UINT32_LE(a,b) do { \
  ((uint8_t *)(a))[0] = (b)&0xff; \
  ((uint8_t *)(a))[1] = ((b)>>8)&0xff; \
  ((uint8_t *)(a))[2] = ((b)>>16)&0xff; \
  ((uint8_t *)(a))[3] = ((b)>>24)&0xff; \
} while(0)
  for (j = 0; j < frame->width / 6; j++) {
    int y0, y1, y2, y3, y4, y5;
    int cr0, cr1, cr2;
    int cb0, cb1, cb2;

    y0 = TO_10 (src_y[j * 6 + 0]);
    y1 = TO_10 (src_y[j * 6 + 1]);
    y2 = TO_10 (src_y[j * 6 + 2]);
    y3 = TO_10 (src_y[j * 6 + 3]);
    y4 = TO_10 (src_y[j * 6 + 4]);
    y5 = TO_10 (src_y[j * 6 + 5]);
    cb0 = TO_10 (src_u[j * 3 + 0]);
    cb1 = TO_10 (src_u[j * 3 + 1]);
    cb2 = TO_10 (src_u[j * 3 + 2]);
    cr0 = TO_10 (src_v[j * 3 + 0]);
    cr1 = TO_10 (src_v[j * 3 + 1]);
    cr2 = TO_10 (src_v[j * 3 + 2]);

    val = (cr0 << 20) | (y0 << 10) | (cb0);
    WRITE_UINT32_LE (dest + j * 16 + 0, val);

    val = (y2 << 20) | (cb1 << 10) | (y1);
    WRITE_UINT32_LE (dest + j * 16 + 4, val);

    val = (cb2 << 20) | (y3 << 10) | (cr1);
    WRITE_UINT32_LE (dest + j * 16 + 8, val);

    val = (y5 << 20) | (cr2 << 10) | (y4);
    WRITE_UINT32_LE (dest + j * 16 + 12, val);
  }
  if (j * 6 < frame->width) {
    int y0, y1, y2, y3, y4, y5;
    int cr0, cr1, cr2;
    int cb0, cb1, cb2;

    y0 = ((j * 6 + 0) < frame->width) ? TO_10 (src_y[j * 6 + 0]) : 0;
    y1 = ((j * 6 + 1) < frame->width) ? TO_10 (src_y[j * 6 + 1]) : 0;
    y2 = ((j * 6 + 2) < frame->width) ? TO_10 (src_y[j * 6 + 2]) : 0;
    y3 = ((j * 6 + 3) < frame->width) ? TO_10 (src_y[j * 6 + 3]) : 0;
    y4 = ((j * 6 + 4) < frame->width) ? TO_10 (src_y[j * 6 + 4]) : 0;
    y5 = ((j * 6 + 5) < frame->width) ? TO_10 (src_y[j * 6 + 5]) : 0;
    cb0 = ((j * 6 + 0) < frame->width) ? TO_10 (src_u[j * 3 + 0]) : 0;
    cb1 = ((j * 6 + 2) < frame->width) ? TO_10 (src_u[j * 3 + 1]) : 0;
    cb2 = ((j * 6 + 4) < frame->width) ? TO_10 (src_u[j * 3 + 2]) : 0;
    cr0 = ((j * 6 + 0) < frame->width) ? TO_10 (src_v[j * 3 + 0]) : 0;
    cr1 = ((j * 6 + 2) < frame->width) ? TO_10 (src_v[j * 3 + 1]) : 0;
    cr2 = ((j * 6 + 4) < frame->width) ? TO_10 (src_v[j * 3 + 2]) : 0;

    val = (cr0 << 20) | (y0 << 10) | (cb0);
    WRITE_UINT32_LE (dest + j * 16 + 0, val);

    val = (y2 << 20) | (cb1 << 10) | (y1);
    WRITE_UINT32_LE (dest + j * 16 + 4, val);

    val = (cb2 << 20) | (y3 << 10) | (cr1);
    WRITE_UINT32_LE (dest + j * 16 + 8, val);

    val = (y5 << 20) | (cr2 << 10) | (y4);
    WRITE_UINT32_LE (dest + j * 16 + 12, val);
  }
#undef TO_10

}

static void
pack_v210 (SchroFrame * frame, void *_dest, int component, int i)
{
  uint8_t *dest = _dest;
  uint8_t *src_y;
  uint8_t *src_u;
  uint8_t *src_v;
  int j;
  uint32_t val;

  src_y = schro_virt_frame_get_line (frame->virt_frame1, 0, i);
  src_u = schro_virt_frame_get_line (frame->virt_frame1, 1, i);
  src_v = schro_virt_frame_get_line (frame->virt_frame1, 2, i);

#define TO_10(x) (((x)<<2) | ((x)>>6))
#define WRITE_UINT32_LE(a,b) do { \
  ((uint8_t *)(a))[0] = (b)&0xff; \
  ((uint8_t *)(a))[1] = ((b)>>8)&0xff; \
  ((uint8_t *)(a))[2] = ((b)>>16)&0xff; \
  ((uint8_t *)(a))[3] = ((b)>>24)&0xff; \
} while(0)
  for (j = 0; j < frame->width / 6; j++) {
    int y0, y1, y2, y3, y4, y5;
    int cr0, cr1, cr2;
    int cb0, cb1, cb2;

    y0 = TO_10 (src_y[j * 6 + 0]);
    y1 = TO_10 (src_y[j * 6 + 1]);
    y2 = TO_10 (src_y[j * 6 + 2]);
    y3 = TO_10 (src_y[j * 6 + 3]);
    y4 = TO_10 (src_y[j * 6 + 4]);
    y5 = TO_10 (src_y[j * 6 + 5]);
    cb0 = TO_10 (src_u[j * 3 + 0]);
    cb1 = TO_10 (src_u[j * 3 + 1]);
    cb2 = TO_10 (src_u[j * 3 + 2]);
    cr0 = TO_10 (src_v[j * 3 + 0]);
    cr1 = TO_10 (src_v[j * 3 + 1]);
    cr2 = TO_10 (src_v[j * 3 + 2]);

    val = (cr0 << 20) | (y0 << 10) | (cb0);
    WRITE_UINT32_LE (dest + j * 16 + 0, val);

    val = (y2 << 20) | (cb1 << 10) | (y1);
    WRITE_UINT32_LE (dest + j * 16 + 4, val);

    val = (cb2 << 20) | (y3 << 10) | (cr1);
    WRITE_UINT32_LE (dest + j * 16 + 8, val);

    val = (y5 << 20) | (cr2 << 10) | (y4);
    WRITE_UINT32_LE (dest + j * 16 + 12, val);
  }
  if (j * 6 < frame->width) {
    int y0, y1, y2, y3, y4, y5;
    int cr0, cr1, cr2;
    int cb0, cb1, cb2;

    y0 = ((j * 6 + 0) < frame->width) ? TO_10 (src_y[j * 6 + 0]) : 0;
    y1 = ((j * 6 + 1) < frame->width) ? TO_10 (src_y[j * 6 + 1]) : 0;
    y2 = ((j * 6 + 2) < frame->width) ? TO_10 (src_y[j * 6 + 2]) : 0;
    y3 = ((j * 6 + 3) < frame->width) ? TO_10 (src_y[j * 6 + 3]) : 0;
    y4 = ((j * 6 + 4) < frame->width) ? TO_10 (src_y[j * 6 + 4]) : 0;
    y5 = ((j * 6 + 5) < frame->width) ? TO_10 (src_y[j * 6 + 5]) : 0;
    cb0 = ((j * 6 + 0) < frame->width) ? TO_10 (src_u[j * 3 + 0]) : 0;
    cb1 = ((j * 6 + 2) < frame->width) ? TO_10 (src_u[j * 3 + 1]) : 0;
    cb2 = ((j * 6 + 4) < frame->width) ? TO_10 (src_u[j * 3 + 2]) : 0;
    cr0 = ((j * 6 + 0) < frame->width) ? TO_10 (src_v[j * 3 + 0]) : 0;
    cr1 = ((j * 6 + 2) < frame->width) ? TO_10 (src_v[j * 3 + 1]) : 0;
    cr2 = ((j * 6 + 4) < frame->width) ? TO_10 (src_v[j * 3 + 2]) : 0;

    val = (cr0 << 20) | (y0 << 10) | (cb0);
    WRITE_UINT32_LE (dest + j * 16 + 0, val);

    val = (y2 << 20) | (cb1 << 10) | (y1);
    WRITE_UINT32_LE (dest + j * 16 + 4, val);

    val = (cb2 << 20) | (y3 << 10) | (cr1);
    WRITE_UINT32_LE (dest + j * 16 + 8, val);

    val = (y5 << 20) | (cr2 << 10) | (y4);
    WRITE_UINT32_LE (dest + j * 16 + 12, val);
  }

}

SchroFrame *
schro_virt_frame_new_pack_v210 (SchroFrame * vf)
{
  SchroFrame *virt_frame;

  virt_frame = schro_frame_new_virtual (NULL, SCHRO_FRAME_FORMAT_v210,
      vf->width, vf->height);
  virt_frame->virt_frame1 = vf;
  if (vf->format == SCHRO_FRAME_FORMAT_S16_422) {
    virt_frame->render_line = pack_v210_s16;
  } else {
    virt_frame->render_line = pack_v210;
  }

  return virt_frame;
}

static void
pack_ayuv (SchroFrame * frame, void *_dest, int component, int i)
{
  uint8_t *dest = _dest;
  uint8_t *src_y;
  uint8_t *src_u;
  uint8_t *src_v;
  int j;

  src_y = schro_virt_frame_get_line (frame->virt_frame1, 0, i);
  src_u = schro_virt_frame_get_line (frame->virt_frame1, 1, i);
  src_v = schro_virt_frame_get_line (frame->virt_frame1, 2, i);

  for (j = 0; j < frame->width; j++) {
    dest[j * 4 + 0] = 0xff;
    dest[j * 4 + 1] = src_y[j];
    dest[j * 4 + 2] = src_u[j];
    dest[j * 4 + 3] = src_v[j];
  }
}

SchroFrame *
schro_virt_frame_new_pack_AYUV (SchroFrame * vf)
{
  SchroFrame *virt_frame;

  virt_frame = schro_frame_new_virtual (NULL, SCHRO_FRAME_FORMAT_AYUV,
      vf->width, vf->height);
  virt_frame->virt_frame1 = vf;
  virt_frame->render_line = pack_ayuv;

  return virt_frame;
}

#ifdef unused
static void
pack_rgb (SchroFrame * frame, void *_dest, int component, int i)
{
  uint8_t *dest = _dest;
  uint8_t *src_y;
  uint8_t *src_u;
  uint8_t *src_v;
  int j;

  src_y = schro_virt_frame_get_line (frame->virt_frame1, 0, i);
  src_u = schro_virt_frame_get_line (frame->virt_frame1, 1, i);
  src_v = schro_virt_frame_get_line (frame->virt_frame1, 2, i);

  for (j = 0; j < frame->width; j++) {
    dest[j * 3 + 0] = src_y[j];
    dest[j * 3 + 1] = src_u[j];
    dest[j * 3 + 2] = src_v[j];
  }
}
#endif

#ifdef unused
SchroFrame *
schro_virt_frame_new_pack_RGB (SchroFrame * vf)
{
  SchroFrame *virt_frame;

  virt_frame = schro_frame_new_virtual (NULL, SCHRO_FRAME_FORMAT_RGB,
      vf->width, vf->height);
  virt_frame->virt_frame1 = vf;
  virt_frame->render_line = pack_rgb;

  return virt_frame;
}
#endif

#ifdef unused
static void
color_matrix (SchroFrame * frame, void *_dest, int component, int i)
{
  uint8_t *dest = _dest;
  uint8_t *src1;
  uint8_t *src2;
  uint8_t *src3;
  double m1, m2, m3;
  double offset;
  int j;

  src1 = schro_virt_frame_get_line (frame->virt_frame1, 0, i);
  src2 = schro_virt_frame_get_line (frame->virt_frame1, 1, i);
  src3 = schro_virt_frame_get_line (frame->virt_frame1, 2, i);

  switch (component) {
    case 0:
      m1 = 0.25679;
      m2 = 0.50413;
      m3 = 0.097906;
      offset = 16;
      break;
    case 1:
      m1 = -0.14822;
      m2 = -0.29099;
      m3 = 0.43922;
      offset = 128;
      break;
    case 2:
      m1 = 0.43922;
      m2 = -0.36779;
      m3 = -0.071427;
      offset = 128;
      break;
    default:
      m1 = 0.0;
      m2 = 0.0;
      m3 = 0.0;
      offset = 0;
      break;
  }

  for (j = 0; j < frame->width; j++) {
    dest[j] = floor (src1[j] * m1 + src2[j] * m2 + src3[j] * m3 + offset + 0.5);
  }

}
#endif

#ifdef unused
SchroFrame *
schro_virt_frame_new_color_matrix (SchroFrame * vf)
{
  SchroFrame *virt_frame;

  virt_frame = schro_frame_new_virtual (NULL, SCHRO_FRAME_FORMAT_U8_444,
      vf->width, vf->height);
  virt_frame->virt_frame1 = vf;
  virt_frame->render_line = color_matrix;

  return virt_frame;
}
#endif

static void
convert_444_422 (SchroFrame * frame, void *_dest, int component, int i)
{
  uint8_t *dest = _dest;
  uint8_t *src;
  int j;

  src = schro_virt_frame_get_line (frame->virt_frame1, component, i);

  if (component == 0) {
    orc_memcpy (dest, src, frame->width);
  } else {
    for (j = 0; j < frame->components[component].width; j++) {
      dest[j] = src[j * 2];
    }
  }
}

static void
convert_444_420 (SchroFrame * frame, void *_dest, int component, int i)
{
  uint8_t *dest = _dest;
  uint8_t *src;
  int j;

  if (component == 0) {
    src = schro_virt_frame_get_line (frame->virt_frame1, component, i);
    orc_memcpy (dest, src, frame->components[component].width);
  } else {
    src = schro_virt_frame_get_line (frame->virt_frame1, component, i * 2);
    for (j = 0; j < frame->components[component].width; j++) {
      dest[j] = src[j * 2];
    }
  }
}

static void
convert_422_420 (SchroFrame * frame, void *_dest, int component, int i)
{
  uint8_t *dest = _dest;
  uint8_t *src;

  if (component == 0) {
    src = schro_virt_frame_get_line (frame->virt_frame1, component, i);
  } else {
    src = schro_virt_frame_get_line (frame->virt_frame1, component, i * 2);
  }
  orc_memcpy (dest, src, frame->components[component].width);
}

/* up */

static void
convert_422_444 (SchroFrame * frame, void *_dest, int component, int i)
{
  uint8_t *dest = _dest;
  uint8_t *src;
  int j;

  src = schro_virt_frame_get_line (frame->virt_frame1, component, i);

  if (component == 0) {
    orc_memcpy (dest, src, frame->width);
  } else {
    for (j = 0; j < frame->components[component].width; j++) {
      dest[j] = src[j >> 1];
    }
  }
}

static void
convert_420_444 (SchroFrame * frame, void *_dest, int component, int i)
{
  uint8_t *dest = _dest;
  uint8_t *src;
  int j;

  if (component == 0) {
    src = schro_virt_frame_get_line (frame->virt_frame1, component, i);
    orc_memcpy (dest, src, frame->components[component].width);
  } else {
    src = schro_virt_frame_get_line (frame->virt_frame1, component, i >> 1);
    for (j = 0; j < frame->components[component].width; j++) {
      dest[j] = src[j >> 1];
    }
  }
}

static void
convert_420_422 (SchroFrame * frame, void *_dest, int component, int i)
{
  uint8_t *dest = _dest;
  uint8_t *src;

  if (component == 0) {
    src = schro_virt_frame_get_line (frame->virt_frame1, component, i);
  } else {
    src = schro_virt_frame_get_line (frame->virt_frame1, component, i >> 1);
  }
  orc_memcpy (dest, src, frame->components[component].width);
}

SchroFrame *
schro_virt_frame_new_subsample (SchroFrame * vf, SchroFrameFormat format)
{
  SchroFrame *virt_frame;
  SchroFrameRenderFunc render_line;

  if (vf->format == format) {
    return vf;
  }
  if (vf->format == SCHRO_FRAME_FORMAT_U8_422 &&
      format == SCHRO_FRAME_FORMAT_U8_420) {
    render_line = convert_422_420;
  } else if (vf->format == SCHRO_FRAME_FORMAT_U8_444 &&
      format == SCHRO_FRAME_FORMAT_U8_420) {
    render_line = convert_444_420;
  } else if (vf->format == SCHRO_FRAME_FORMAT_U8_444 &&
      format == SCHRO_FRAME_FORMAT_U8_422) {
    render_line = convert_444_422;
  } else if (vf->format == SCHRO_FRAME_FORMAT_U8_420 &&
      format == SCHRO_FRAME_FORMAT_U8_422) {
    render_line = convert_420_422;
  } else if (vf->format == SCHRO_FRAME_FORMAT_U8_420 &&
      format == SCHRO_FRAME_FORMAT_U8_444) {
    render_line = convert_420_444;
  } else if (vf->format == SCHRO_FRAME_FORMAT_U8_422 &&
      format == SCHRO_FRAME_FORMAT_U8_444) {
    render_line = convert_422_444;
  } else {
    SCHRO_ASSERT (0);
    return NULL;
  }
  virt_frame = schro_frame_new_virtual (NULL, format, vf->width, vf->height);
  virt_frame->virt_frame1 = vf;
  virt_frame->render_line = render_line;

  return virt_frame;
}


#if 0
SchroFrame *
schro_virt_frame_new_horiz_downsample_take (SchroFrame * vf, int cosite)
{
  SchroFrame *virt_frame;
  virt_frame = schro_virt_frame_new_horiz_downsample (vf, cosite);
  schro_frame_unref (vf);
  return virt_frame;
}

SchroFrame *
schro_virt_frame_new_vert_downsample_take (SchroFrame * vf, int cosite)
{
  SchroFrame *virt_frame;
  virt_frame = schro_virt_frame_new_vert_downsample (vf, cosite);
  schro_frame_unref (vf);
  return virt_frame;
}

SchroFrame *
schro_virt_frame_new_vert_resample_take (SchroFrame * vf, int height)
{
  SchroFrame *virt_frame;
  virt_frame = schro_virt_frame_new_vert_resample (vf, height);
  schro_frame_unref (vf);
  return virt_frame;
}

SchroFrame *
schro_virt_frame_new_horiz_resample_take (SchroFrame * vf, int width)
{
  SchroFrame *virt_frame;
  virt_frame = schro_virt_frame_new_horiz_resample (vf, width);
  schro_frame_unref (vf);
  return virt_frame;
}

SchroFrame *
schro_virt_frame_new_unpack_take (SchroFrame * vf)
{
  SchroFrame *virt_frame;
  virt_frame = schro_virt_frame_new_unpack (vf);
  schro_frame_unref (vf);
  return virt_frame;
}

SchroFrame *
schro_virt_frame_new_pack_YUY2_take (SchroFrame * vf)
{
  SchroFrame *virt_frame;
  virt_frame = schro_virt_frame_new_pack_YUY2 (vf);
  schro_frame_unref (vf);
  return virt_frame;
}

SchroFrame *
schro_virt_frame_new_pack_UYVY_take (SchroFrame * vf)
{
  SchroFrame *virt_frame;
  virt_frame = schro_virt_frame_new_pack_UYVY (vf);
  schro_frame_unref (vf);
  return virt_frame;
}

SchroFrame *
schro_virt_frame_new_pack_v216_take (SchroFrame * vf)
{
  SchroFrame *virt_frame;
  virt_frame = schro_virt_frame_new_pack_v216 (vf);
  schro_frame_unref (vf);
  return virt_frame;
}

SchroFrame *
schro_virt_frame_new_pack_v210_take (SchroFrame * vf)
{
  SchroFrame *virt_frame;
  virt_frame = schro_virt_frame_new_pack_v210 (vf);
  schro_frame_unref (vf);
  return virt_frame;
}

SchroFrame *
schro_virt_frame_new_pack_AYUV_take (SchroFrame * vf)
{
  SchroFrame *virt_frame;
  virt_frame = schro_virt_frame_new_pack_AYUV (vf);
  schro_frame_unref (vf);
  return virt_frame;
}

SchroFrame *
schro_virt_frame_new_pack_RGB_take (SchroFrame * vf)
{
  SchroFrame *virt_frame;
  virt_frame = schro_virt_frame_new_pack_RGB (vf);
  schro_frame_unref (vf);
  return virt_frame;
}

SchroFrame *
schro_virt_frame_new_subsample_take (SchroFrame * vf, SchroFrameFormat format)
{
  SchroFrame *virt_frame;
  virt_frame = schro_virt_frame_new_subsample (vf, format);
  schro_frame_unref (vf);
  return virt_frame;
}
#endif

static void
convert_u8_s16 (SchroFrame * frame, void *_dest, int component, int i)
{
  uint8_t *dest = _dest;
  int16_t *src;

  src = schro_virt_frame_get_line (frame->virt_frame1, component, i);
  orc_offsetconvert_u8_s16 (dest, src, frame->components[component].width);
}

SchroFrame *
schro_virt_frame_new_convert_u8 (SchroFrame * vf)
{
  SchroFrame *virt_frame;
  SchroFrameFormat format;

  format = (vf->format & 3) | SCHRO_FRAME_FORMAT_U8_444;

  virt_frame = schro_frame_new_virtual (NULL, format, vf->width, vf->height);
  virt_frame->virt_frame1 = vf;
  virt_frame->render_line = convert_u8_s16;
  virt_frame->virt_priv = schro_malloc (sizeof (int16_t) * vf->width);

  return virt_frame;
}

static void
convert_s16_u8 (SchroFrame * frame, void *_dest, int component, int i)
{
  int16_t *dest = _dest;
  uint8_t *src;

  src = schro_virt_frame_get_line (frame->virt_frame1, component, i);

  orc_offsetconvert_s16_u8 (dest, src, frame->components[component].width);
}

SchroFrame *
schro_virt_frame_new_convert_s16 (SchroFrame * vf)
{
  SchroFrame *virt_frame;
  SchroFrameFormat format;

  format = (vf->format & 3) | SCHRO_FRAME_FORMAT_S16_444;

  virt_frame = schro_frame_new_virtual (NULL, format, vf->width, vf->height);
  virt_frame->virt_frame1 = vf;
  virt_frame->render_line = convert_s16_u8;

  return virt_frame;
}

static void
crop_u8 (SchroFrame * frame, void *_dest, int component, int i)
{
  uint8_t *dest = _dest;
  uint8_t *src;

  src = schro_virt_frame_get_line (frame->virt_frame1, component, i);
  orc_memcpy (dest, src, frame->components[component].width);
}

static void
crop_s16 (SchroFrame * frame, void *_dest, int component, int i)
{
  int16_t *dest = _dest;
  int16_t *src;

  src = schro_virt_frame_get_line (frame->virt_frame1, component, i);
  orc_memcpy (dest, src, frame->components[component].width * sizeof (int16_t));
}

SchroFrame *
schro_virt_frame_new_crop (SchroFrame * vf, int width, int height)
{
  SchroFrame *virt_frame;

  if (width == vf->width && height == vf->height)
    return vf;

  SCHRO_ASSERT (width <= vf->width);
  SCHRO_ASSERT (height <= vf->height);

  virt_frame = schro_frame_new_virtual (NULL, vf->format, width, height);
  virt_frame->virt_frame1 = vf;
  switch (SCHRO_FRAME_FORMAT_DEPTH (vf->format)) {
    case SCHRO_FRAME_FORMAT_DEPTH_U8:
      virt_frame->render_line = crop_u8;
      break;
    case SCHRO_FRAME_FORMAT_DEPTH_S16:
      virt_frame->render_line = crop_s16;
      break;
    default:
      SCHRO_ASSERT (0);
      break;
  }

  return virt_frame;
}

static void
edge_extend_u8 (SchroFrame * frame, void *_dest, int component, int i)
{
  uint8_t *dest = _dest;
  uint8_t *src;
  SchroFrame *srcframe = frame->virt_frame1;

  src = schro_virt_frame_get_line (frame->virt_frame1, component,
      MIN (i, srcframe->components[component].height - 1));
  orc_memcpy (dest, src, srcframe->components[component].width);
  orc_splat_u8_ns (dest + srcframe->components[component].width,
      dest[srcframe->components[component].width - 1],
      frame->components[component].width -
      srcframe->components[component].width);
}

static void
edge_extend_s16 (SchroFrame * frame, void *_dest, int component, int i)
{
  int16_t *dest = _dest;
  int16_t *src;
  SchroFrame *srcframe = frame->virt_frame1;

  src = schro_virt_frame_get_line (frame->virt_frame1, component,
      MIN (i, srcframe->components[component].height - 1));
  orc_memcpy (dest, src,
      srcframe->components[component].width * sizeof (int16_t));
  orc_splat_s16_ns (dest + srcframe->components[component].width,
      dest[srcframe->components[component].width - 1],
      frame->components[component].width -
      srcframe->components[component].width);
}

SchroFrame *
schro_virt_frame_new_edgeextend (SchroFrame * vf, int width, int height)
{
  SchroFrame *virt_frame;

  if (width == vf->width && height == vf->height)
    return vf;

  SCHRO_ASSERT (width >= vf->width);
  SCHRO_ASSERT (height >= vf->height);

  virt_frame = schro_frame_new_virtual (NULL, vf->format, width, height);
  virt_frame->virt_frame1 = vf;
  switch (SCHRO_FRAME_FORMAT_DEPTH (vf->format)) {
    case SCHRO_FRAME_FORMAT_DEPTH_U8:
      virt_frame->render_line = edge_extend_u8;
      break;
    case SCHRO_FRAME_FORMAT_DEPTH_S16:
      virt_frame->render_line = edge_extend_s16;
      break;
    default:
      SCHRO_ASSERT (0);
      break;
  }

  return virt_frame;
}
