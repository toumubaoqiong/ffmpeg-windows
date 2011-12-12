
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <string.h>

int _schro_motion_ref = FALSE;

static int
get_pixel (SchroMotion * motion, int k, SchroUpsampledFrame * upframe,
    int x, int y, int dx, int dy);

int
schro_motion_pixel_predict_block (SchroMotion * motion, int x, int y, int k,
    int i, int j);

static void
schro_motion_get_global_vector (SchroMotion * motion, int ref, int x, int y,
    int *dx, int *dy)
{
  SchroParams *params = motion->params;
  SchroGlobalMotion *gm = params->global_motion + ref;
  int alpha, beta;
  int scale;

  alpha = gm->a_exp;
  beta = gm->c_exp;

  scale = (1 << beta) - (gm->c0 * x + gm->c1 * y);

  *dx = scale * (gm->a00 * x + gm->a01 * y + (1 << alpha) * gm->b0);
  *dy = scale * (gm->a10 * x + gm->a11 * y + (1 << alpha) * gm->b1);

  *dx >>= (alpha + beta);
  *dy >>= (alpha + beta);
}


static int
schro_motion_pixel_predict (SchroMotion * motion, int x, int y, int k)
{
  int i, j;
  int value;

  i = (x + motion->xoffset) / motion->xbsep - 1;
  j = (y + motion->yoffset) / motion->ybsep - 1;

  value = schro_motion_pixel_predict_block (motion, x, y, k, i, j);
  value += schro_motion_pixel_predict_block (motion, x, y, k, i + 1, j);
  value += schro_motion_pixel_predict_block (motion, x, y, k, i, j + 1);
  value += schro_motion_pixel_predict_block (motion, x, y, k, i + 1, j + 1);

  return ROUND_SHIFT (value, 6);
}

static int
get_dc_pixel (SchroMotion * motion, int i, int j, int k, int x, int y)
{
  SchroParams *params = motion->params;
  SchroMotionVector *mv;

  mv = &motion->motion_vectors[j * params->x_num_blocks + i];

  return mv->u.dc.dc[k] + 128;
}

static int
get_ref1_pixel (SchroMotion * motion, int i, int j, int k, int x, int y)
{
  SchroParams *params = motion->params;
  SchroMotionVector *mv;
  int value;
  int dx, dy;

  mv = &motion->motion_vectors[j * params->x_num_blocks + i];
  if (mv->using_global) {
    schro_motion_get_global_vector (motion, 0, x, y, &dx, &dy);
  } else {
    dx = mv->u.vec.dx[0];
    dy = mv->u.vec.dy[0];
  }

  value = (motion->ref1_weight + motion->ref2_weight) *
      get_pixel (motion, k, motion->src1, x, y, dx, dy);

  return ROUND_SHIFT (value, motion->ref_weight_precision);
}

static int
get_ref2_pixel (SchroMotion * motion, int i, int j, int k, int x, int y)
{
  SchroParams *params = motion->params;
  SchroMotionVector *mv;
  int value;
  int dx, dy;

  mv = &motion->motion_vectors[j * params->x_num_blocks + i];
  if (mv->using_global) {
    schro_motion_get_global_vector (motion, 1, x, y, &dx, &dy);
  } else {
    dx = mv->u.vec.dx[1];
    dy = mv->u.vec.dy[1];
  }

  value = (motion->ref1_weight + motion->ref2_weight) *
      get_pixel (motion, k, motion->src2, x, y, dx, dy);

  return ROUND_SHIFT (value, motion->ref_weight_precision);
}

static int
get_biref_pixel (SchroMotion * motion, int i, int j, int k, int x, int y)
{
  SchroParams *params = motion->params;
  SchroMotionVector *mv;
  int value;
  int dx0, dx1, dy0, dy1;

  mv = &motion->motion_vectors[j * params->x_num_blocks + i];
  if (mv->using_global) {
    schro_motion_get_global_vector (motion, 0, x, y, &dx0, &dy0);
    schro_motion_get_global_vector (motion, 1, x, y, &dx1, &dy1);
  } else {
    dx0 = mv->u.vec.dx[0];
    dy0 = mv->u.vec.dy[0];
    dx1 = mv->u.vec.dx[1];
    dy1 = mv->u.vec.dy[1];
  }

  value = motion->ref1_weight *
      get_pixel (motion, k, motion->src1, x, y, dx0, dy0);
  value += motion->ref2_weight *
      get_pixel (motion, k, motion->src2, x, y, dx1, dy1);

  return ROUND_SHIFT (value, motion->ref_weight_precision);
}

static int
get_pixel (SchroMotion * motion, int k, SchroUpsampledFrame * upframe,
    int x, int y, int dx, int dy)
{
  int px, py;

  if (k > 0) {
    dx >>= SCHRO_CHROMA_FORMAT_H_SHIFT (motion->params->
        video_format->chroma_format);
    dy >>= SCHRO_CHROMA_FORMAT_V_SHIFT (motion->params->
        video_format->chroma_format);
  }

  px = (x << motion->mv_precision) + dx;
  py = (y << motion->mv_precision) + dy;

  return schro_upsampled_frame_get_pixel_precN (upframe, k, px, py,
      motion->mv_precision);
}

static int
get_ramp (int x, int offset)
{
  if (offset == 1) {
    if (x == 0)
      return 3;
    return 5;
  }
  return 1 + (6 * x + offset - 1) / (2 * offset - 1);
}

int
schro_motion_pixel_predict_block (SchroMotion * motion, int x, int y, int k,
    int i, int j)
{
  SchroParams *params = motion->params;
  SchroMotionVector *mv;
  int xmin, xmax, ymin, ymax;
  int wx, wy;
  int value;
  int width, height;

  if (i < 0 || j < 0)
    return 0;
  if (i >= params->x_num_blocks || j >= params->y_num_blocks)
    return 0;

  width = motion->xbsep * params->x_num_blocks;
  height = motion->ybsep * params->y_num_blocks;

  xmin = i * motion->xbsep - motion->xoffset;
  ymin = j * motion->ybsep - motion->yoffset;
  xmax = (i + 1) * motion->xbsep + motion->xoffset;
  ymax = (j + 1) * motion->ybsep + motion->yoffset;

  if (x < xmin || y < ymin || x >= xmax || y >= ymax)
    return 0;

  if (motion->xoffset == 0) {
    wx = 8;
  } else if (x < motion->xoffset || x >= width - motion->xoffset) {
    wx = 8;
  } else if (x - xmin < 2 * motion->xoffset) {
    wx = get_ramp (x - xmin, motion->xoffset);
  } else if (xmax - 1 - x < 2 * motion->xoffset) {
    wx = get_ramp (xmax - 1 - x, motion->xoffset);
  } else {
    wx = 8;
  }

  if (motion->yoffset == 0) {
    wy = 8;
  } else if (y < motion->yoffset || y >= height - motion->yoffset) {
    wy = 8;
  } else if (y - ymin < 2 * motion->yoffset) {
    wy = get_ramp (y - ymin, motion->yoffset);
  } else if (ymax - 1 - y < 2 * motion->yoffset) {
    wy = get_ramp (ymax - 1 - y, motion->yoffset);
  } else {
    wy = 8;
  }

  mv = &motion->motion_vectors[j * params->x_num_blocks + i];

  switch (mv->pred_mode) {
    case 0:
      value = get_dc_pixel (motion, i, j, k, x, y);
      break;
    case 1:
      value = get_ref1_pixel (motion, i, j, k, x, y);
      break;
    case 2:
      value = get_ref2_pixel (motion, i, j, k, x, y);
      break;
    case 3:
      value = get_biref_pixel (motion, i, j, k, x, y);
      break;
    default:
      value = 0;
      break;
  }

  return value * wx * wy;
}


void
schro_motion_render_ref (SchroMotion * motion, SchroFrame * dest,
    SchroFrame * addframe, int add, SchroFrame * output_frame)
{
  SchroParams *params = motion->params;
  int k;
  int x, y;
  int16_t *line;
  int16_t *addline;
  uint8_t *oline;

  if (params->num_refs == 1) {
    SCHRO_ASSERT (params->picture_weight_2 == 1);
  }

  motion->ref_weight_precision = params->picture_weight_bits;
  motion->ref1_weight = params->picture_weight_1;
  motion->ref2_weight = params->picture_weight_2;

  motion->mv_precision = params->mv_precision;

  for (k = 0; k < 3; k++) {
    SchroFrameData *comp = dest->components + k;

    if (k == 0) {
      motion->xbsep = params->xbsep_luma;
      motion->ybsep = params->ybsep_luma;
      motion->xblen = params->xblen_luma;
      motion->yblen = params->yblen_luma;
    } else {
      motion->xbsep = params->xbsep_luma >>
          SCHRO_CHROMA_FORMAT_H_SHIFT (motion->params->
          video_format->chroma_format);
      motion->ybsep =
          params->ybsep_luma >> SCHRO_CHROMA_FORMAT_V_SHIFT (motion->
          params->video_format->chroma_format);
      motion->xblen =
          params->xblen_luma >> SCHRO_CHROMA_FORMAT_H_SHIFT (motion->
          params->video_format->chroma_format);
      motion->yblen =
          params->yblen_luma >> SCHRO_CHROMA_FORMAT_V_SHIFT (motion->
          params->video_format->chroma_format);
    }
    motion->xoffset = (motion->xblen - motion->xbsep) / 2;
    motion->yoffset = (motion->yblen - motion->ybsep) / 2;

    for (y = 0; y < comp->height; y++) {
      line = SCHRO_FRAME_DATA_GET_LINE (comp, y);
      for (x = 0; x < comp->width; x++) {
        line[x] = CLAMP (schro_motion_pixel_predict (motion, x, y, k), 0, 255);

        /* Note: the 128 offset converts the 0-255 range of the reference
         * pictures into the bipolar range used for Dirac signal processing */
        line[x] -= 128;
      }
    }

    if (add) {
      for (y = 0; y < comp->height; y++) {
        line = SCHRO_FRAME_DATA_GET_LINE (comp, y);
        addline = SCHRO_FRAME_DATA_GET_LINE (addframe->components + k, y);
        oline = SCHRO_FRAME_DATA_GET_LINE (output_frame->components + k, y);

        for (x = 0; x < comp->width; x++) {
          oline[x] = CLAMP (addline[x] + line[x] + 128, 0, 255);
        }
      }
    } else {
      for (y = 0; y < comp->height; y++) {
        line = SCHRO_FRAME_DATA_GET_LINE (comp, y);
        addline = SCHRO_FRAME_DATA_GET_LINE (addframe->components + k, y);

        for (x = 0; x < comp->width; x++) {
          addline[x] -= line[x];
        }
      }
    }
  }
}
