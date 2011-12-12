
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <string.h>
#include <schroedinger/schroorc.h>

/*
 * This is a slimmed-down implementation of normal OBMC for non-overlapped
 * 8x8 blocks with motion vector precision of 0.
 */

static void
get_block (SchroMotion * motion, int k, int ref, int i, int j, int dx, int dy)
{
  int px, py;
  int x, y;
  SchroUpsampledFrame *upframe;
  int exp;

  if (k > 0) {
    dx >>= SCHRO_CHROMA_FORMAT_H_SHIFT (motion->params->
        video_format->chroma_format);
    dy >>= SCHRO_CHROMA_FORMAT_V_SHIFT (motion->params->
        video_format->chroma_format);
  }
  if (ref) {
    upframe = motion->src2;
  } else {
    upframe = motion->src1;
  }

  x = motion->xbsep * i - motion->xoffset;
  y = motion->ybsep * j - motion->yoffset;
  px = (x << motion->mv_precision) + dx;
  py = (y << motion->mv_precision) + dy;
  exp = 32 << motion->mv_precision;

  px = CLAMP (px, -exp, motion->max_fast_x + exp - 1);
  py = CLAMP (py, -exp, motion->max_fast_y + exp - 1);

  schro_upsampled_frame_get_subdata_prec0 (upframe, k, px, py,
      &motion->block_ref[ref]);
}

static void
get_dc_block (SchroMotion * motion, int i, int j, int k, int x, int y)
{
  SchroParams *params = motion->params;
  SchroMotionVector *mv;
  int value;
  int ii, jj;

  mv = &motion->motion_vectors[j * params->x_num_blocks + i];

  memcpy (&motion->block, &motion->alloc_block, sizeof (SchroFrameData));
  value = mv->u.dc.dc[k];
  for (jj = 0; jj < motion->yblen; jj++) {
    uint8_t *data = SCHRO_FRAME_DATA_GET_LINE (&motion->block, jj);
    /* FIXME splat */
    for (ii = 0; ii < motion->xblen; ii++) {
      data[ii] = value + 128;
    }
  }
}

static void
get_ref1_block (SchroMotion * motion, int i, int j, int k, int x, int y)
{
  SchroParams *params = motion->params;
  SchroMotionVector *mv;

  mv = &motion->motion_vectors[j * params->x_num_blocks + i];
  SCHRO_ASSERT (mv->using_global == FALSE);

  get_block (motion, k, 0, i, j, mv->u.vec.dx[0], mv->u.vec.dy[0]);

  memcpy (&motion->block, &motion->block_ref[0], sizeof (SchroFrameData));
}

static void
get_ref2_block (SchroMotion * motion, int i, int j, int k, int x, int y)
{
  SchroParams *params = motion->params;
  SchroMotionVector *mv;

  mv = &motion->motion_vectors[j * params->x_num_blocks + i];
  SCHRO_ASSERT (mv->using_global == FALSE);

  get_block (motion, k, 1, i, j, mv->u.vec.dx[1], mv->u.vec.dy[1]);

  memcpy (&motion->block, &motion->block_ref[1], sizeof (SchroFrameData));
}

static void
get_biref_block (SchroMotion * motion, int i, int j, int k, int x, int y)
{
  SchroParams *params = motion->params;
  SchroMotionVector *mv;

  mv = &motion->motion_vectors[j * params->x_num_blocks + i];
  SCHRO_ASSERT (mv->using_global == FALSE);

  get_block (motion, k, 0, i, j, mv->u.vec.dx[0], mv->u.vec.dy[0]);
  get_block (motion, k, 1, i, j, mv->u.vec.dx[1], mv->u.vec.dy[1]);

  memcpy (&motion->block, &motion->alloc_block, sizeof (SchroFrameData));
  orc_avg2_8xn_u8 (motion->block.data, motion->block.stride,
      motion->block_ref[0].data, motion->block_ref[0].stride,
      motion->block_ref[1].data, motion->block_ref[1].stride, motion->yblen);
}

static void
schro_motion_block_predict_block (SchroMotion * motion, int x, int y, int k,
    int i, int j)
{
  SchroParams *params = motion->params;
  SchroMotionVector *mv;

  mv = &motion->motion_vectors[j * params->x_num_blocks + i];

  switch (mv->pred_mode) {
    case 0:
      get_dc_block (motion, i, j, k, x, y);
      break;
    case 1:
      get_ref1_block (motion, i, j, k, x, y);
      break;
    case 2:
      get_ref2_block (motion, i, j, k, x, y);
      break;
    case 3:
      get_biref_block (motion, i, j, k, x, y);
      break;
    default:
      SCHRO_ASSERT (0);
      break;
  }
}

static void
schro_motion_block_accumulate_slow (SchroMotion * motion, SchroFrameData * comp,
    int x, int y)
{
  int i, j;

  for (j = 0; j < motion->yblen; j++) {
    int16_t *dest = SCHRO_FRAME_DATA_GET_PIXEL_S16 (comp, x, y + j);
    uint8_t *src = SCHRO_FRAME_DATA_GET_LINE (&motion->block, j);

    if (y + j < 0 || y + j >= comp->height)
      continue;

    for (i = 0; i < motion->xblen; i++) {
      if (x + i < 0 || x + i >= comp->width)
        continue;

      dest[i] = src[i] - 128;
    }
  }
}

static void
schro_motion_block_accumulate (SchroMotion * motion, SchroFrameData * comp,
    int x, int y)
{
  int i, j;

  for (j = 0; j < motion->yblen; j++) {
    int16_t *dest = SCHRO_FRAME_DATA_GET_PIXEL_S16 (comp, x, y + j);
    uint8_t *src = SCHRO_FRAME_DATA_GET_LINE (&motion->block, j);
    for (i = 0; i < motion->xblen; i++) {
      dest[i] = src[i] - 128;
    }
  }
}

int
schro_motion_render_fast_allowed (SchroMotion * motion)
{
  SchroParams *params = motion->params;

  if (params->have_global_motion)
    return FALSE;

  if (params->xblen_luma != 8 || params->yblen_luma != 8 ||
      params->xbsep_luma != 8 || params->ybsep_luma != 8) {
    return FALSE;
  }

  if (params->picture_weight_bits != 1 ||
      params->picture_weight_1 != 1 || params->picture_weight_2 != 1) {
    return FALSE;
  }

  if (params->mv_precision != 0) {
    return FALSE;
  }

  return TRUE;
}

void
schro_motion_render_fast (SchroMotion * motion, SchroFrame * dest,
    SchroFrame * addframe, int add, SchroFrame * output_frame)
{
  int i, j;
  int x, y;
  int k;
  SchroParams *params = motion->params;
  int max_x_blocks;
  int max_y_blocks;

  SCHRO_ASSERT (schro_motion_render_fast_allowed (motion));

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
      motion->width = comp->width;
      motion->height = comp->height;
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
      motion->width = comp->width;
      motion->height = comp->height;
    }
    motion->xoffset = (motion->xblen - motion->xbsep) / 2;
    motion->yoffset = (motion->yblen - motion->ybsep) / 2;
    motion->max_fast_x =
        (motion->width - motion->xblen) << motion->mv_precision;
    motion->max_fast_y =
        (motion->height - motion->yblen) << motion->mv_precision;

    motion->alloc_block.data =
        schro_malloc (motion->xblen * motion->yblen * sizeof (uint8_t));
    motion->alloc_block.stride = motion->xblen * sizeof (uint8_t);
    motion->alloc_block.width = motion->xblen;
    motion->alloc_block.height = motion->yblen;

    orc_splat_s16_2d (comp->data, comp->stride, 0, comp->width, comp->height);

    max_x_blocks = MIN (params->x_num_blocks,
        (motion->width - motion->xoffset) / motion->xbsep);
    max_y_blocks = MIN (params->y_num_blocks,
        (motion->height - motion->yoffset) / motion->ybsep);

    j = 0;
    for (j = 0; j < max_y_blocks; j++) {
      y = motion->ybsep * j - motion->yoffset;

      for (i = 0; i < max_x_blocks; i++) {
        x = motion->xbsep * i - motion->xoffset;

        schro_motion_block_predict_block (motion, x, y, k, i, j);
        schro_motion_block_accumulate (motion, comp, x, y);
      }

      for (; i < params->x_num_blocks; i++) {
        x = motion->xbsep * i - motion->xoffset;

        schro_motion_block_predict_block (motion, x, y, k, i, j);
        schro_motion_block_accumulate_slow (motion, comp, x, y);
      }
    }
    for (j = max_y_blocks; j < params->y_num_blocks; j++) {
      y = motion->ybsep * j - motion->yoffset;
      for (i = 0; i < params->x_num_blocks; i++) {
        x = motion->xbsep * i - motion->xoffset;

        schro_motion_block_predict_block (motion, x, y, k, i, j);
        schro_motion_block_accumulate_slow (motion, comp, x, y);
      }
    }

    schro_free (motion->alloc_block.data);
  }

}
