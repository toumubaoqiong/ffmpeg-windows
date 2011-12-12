
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <schroedinger/schrophasecorrelation.h>
#include <string.h>
#include <math.h>

#define DC_BIAS 50
#define DC_METRIC 50
#define BIDIR_LIMIT (10*8*8)

#define SCHRO_METRIC_INVALID_2 0x7fffffff

#define motion_field_get(mf,x,y) \
  ((mf)->motion_vectors + (y)*(mf)->x_num_blocks + (x))

void schro_encoder_bigblock_estimation (SchroMotionEst * me);
static SchroFrame *get_downsampled (SchroEncoderFrame * frame, int i);


SchroMotionEst *
schro_motionest_new (SchroEncoderFrame * frame)
{
  SchroMotionEst *me;

  me = schro_malloc0 (sizeof (SchroMotionEst));

  me->encoder_frame = frame;
  me->params = &frame->params;

  me->downsampled_src0[0] = frame->ref_frame[0]->filtered_frame;
  me->downsampled_src0[1] = frame->ref_frame[0]->downsampled_frames[0];
  me->downsampled_src0[2] = frame->ref_frame[0]->downsampled_frames[1];
  me->downsampled_src0[3] = frame->ref_frame[0]->downsampled_frames[2];
  me->downsampled_src0[4] = frame->ref_frame[0]->downsampled_frames[3];

  if (me->params->num_refs > 1) {
    me->downsampled_src1[0] = frame->ref_frame[1]->filtered_frame;
    me->downsampled_src1[1] = frame->ref_frame[1]->downsampled_frames[0];
    me->downsampled_src1[2] = frame->ref_frame[1]->downsampled_frames[1];
    me->downsampled_src1[3] = frame->ref_frame[1]->downsampled_frames[2];
    me->downsampled_src1[4] = frame->ref_frame[1]->downsampled_frames[3];
  }

  me->scan_distance = frame->encoder->magic_scan_distance;

  return me;
}

void
schro_motionest_free (SchroMotionEst * me)
{
  schro_free (me);
}


void
schro_encoder_motion_predict_rough (SchroEncoderFrame * frame)
{
  SchroParams *params = &frame->params;
  SchroEncoder *encoder = frame->encoder;
  int ref;

  SCHRO_ASSERT (params->x_num_blocks != 0);
  SCHRO_ASSERT (params->y_num_blocks != 0);
  SCHRO_ASSERT (params->num_refs > 0);

  if (encoder->enable_hierarchical_estimation) {
    for (ref = 0; ref < params->num_refs; ref++) {
      if (encoder->enable_bigblock_estimation) {
        frame->rme[ref] = schro_rough_me_new (frame, frame->ref_frame[ref]);
        schro_rough_me_heirarchical_scan (frame->rme[ref]);
      } else if (encoder->enable_deep_estimation) {
        frame->hier_bm[ref] = schro_hbm_new (frame, ref);
        schro_hbm_scan (frame->hier_bm[ref]);
      }

      if (encoder->enable_phasecorr_estimation) {
        frame->phasecorr[ref] = schro_phasecorr_new (frame,
            frame->ref_frame[ref]);
        schro_encoder_phasecorr_estimation (frame->phasecorr[ref]);
      }
    }
    if (encoder->enable_global_motion) {
      schro_encoder_global_estimation (frame);
    }
  }

  if (encoder->enable_bigblock_estimation) {
    frame->me = schro_motionest_new (frame);
  } else if (encoder->enable_deep_estimation) {
    frame->deep_me = schro_me_new (frame);
  }

  frame->motion = schro_motion_new (params, NULL, NULL);
  if (encoder->enable_bigblock_estimation) {
    frame->me->motion = frame->motion;
  }

}

void
schro_encoder_motion_predict_pel (SchroEncoderFrame * frame)
{
  SchroParams *params = &frame->params;
  int ref;

  SCHRO_ASSERT (params->x_num_blocks != 0);
  SCHRO_ASSERT (params->y_num_blocks != 0);
  SCHRO_ASSERT (params->num_refs > 0);

  if (frame->encoder->enable_bigblock_estimation) {
    schro_encoder_bigblock_estimation (frame->me);

    schro_motion_calculate_stats (frame->motion, frame);
    frame->estimated_mc_bits = schro_motion_estimate_entropy (frame->motion);

    frame->badblock_ratio =
        (double) frame->me->badblocks / (params->x_num_blocks *
        params->y_num_blocks / 16);
  } else if (frame->encoder->enable_deep_estimation) {
    for (ref = 0; params->num_refs > ref; ++ref) {
      SCHRO_ASSERT (frame->hier_bm[ref]);
      schro_hierarchical_bm_scan_hint (frame->hier_bm[ref], 0, 3);
    }
  } else
    SCHRO_ASSERT (0);
}

static void
schro_encoder_motion_refine_block_subpel (SchroEncoderFrame * frame,
    SchroBlock * block, int i, int j)
{
  SchroParams *params = &frame->params;
  int skip;
  int ii, jj;

  skip = 4 >> block->mv[0][0].split;
  for (jj = 0; jj < 4; jj += skip) {
    for (ii = 0; ii < 4; ii += skip) {
      if (block->mv[jj][ii].pred_mode & 1) {
        block->mv[jj][ii].u.vec.dx[0] <<= params->mv_precision;
        block->mv[jj][ii].u.vec.dy[0] <<= params->mv_precision;
      }
      if (block->mv[jj][ii].pred_mode & 2) {
        block->mv[jj][ii].u.vec.dx[1] <<= params->mv_precision;
        block->mv[jj][ii].u.vec.dy[1] <<= params->mv_precision;
      }
    }
  }

  if (block->mv[0][0].split < 3) {
    for (jj = 0; jj < 4; jj += skip) {
      for (ii = 0; ii < 4; ii += skip) {
        if (SCHRO_METRIC_INVALID == block->mv[jj][ii].metric) {
          continue;
        }
        if (block->mv[jj][ii].pred_mode == 1
            || block->mv[jj][ii].pred_mode == 2) {
          SchroUpsampledFrame *ref_upframe;
          SchroFrameData orig;
          SchroFrameData ref_fd;
          int dx, dy;
          int x, y;
          int metric = SCHRO_METRIC_INVALID_2;
          int width, height;
          int min_metric;
          int min_dx, min_dy;
          int ref;

          ref = block->mv[jj][ii].pred_mode - 1;
          ref_upframe = frame->ref_frame[ref]->upsampled_original_frame;

          x = MAX ((i + ii) * frame->params.xbsep_luma, 0);
          y = MAX ((j + jj) * frame->params.ybsep_luma, 0);

          schro_frame_get_subdata (get_downsampled (frame, 0), &orig, 0, x, y);

          width = MIN (skip * frame->params.xbsep_luma, orig.width);
          height = MIN (skip * frame->params.ybsep_luma, orig.height);


          min_metric = 0x7fffffff;
          min_dx = 0;
          min_dy = 0;
          for (dx = -1; dx <= 1; dx++) {
            for (dy = -1; dy <= 1; dy++) {
              schro_upsampled_frame_get_subdata_prec1 (ref_upframe, 0,
                  2 * x + block->mv[jj][ii].u.vec.dx[ref] + dx,
                  2 * y + block->mv[jj][ii].u.vec.dy[ref] + dy, &ref_fd);

              metric = schro_metric_get (&orig, &ref_fd, width, height);
              if (metric < min_metric) {
                min_dx = dx;
                min_dy = dy;
                min_metric = metric;
              }
            }
          }
          if (SCHRO_METRIC_INVALID > min_metric) {
            block->mv[ii][ii].u.vec.dx[ref] += min_dx;
            block->mv[jj][ii].u.vec.dy[ref] += min_dy;
            block->mv[jj][ii].metric = min_metric;
          }
        }
      }
    }
  }

}

void
schro_encoder_motion_predict_subpel (SchroEncoderFrame * frame)
{
  SchroParams *params = &frame->params;
  int i;
  int j;

  SCHRO_ASSERT (frame->upsampled_original_frame);
  SCHRO_ASSERT (frame->ref_frame[0]->upsampled_original_frame);
  if (frame->ref_frame[1]) {
    SCHRO_ASSERT (frame->ref_frame[1]->upsampled_original_frame);
  }

  for (j = 0; j < params->y_num_blocks; j += 4) {
    for (i = 0; i < params->x_num_blocks; i += 4) {
      SchroBlock block = { 0 };

      schro_motion_copy_from (frame->me->motion, i, j, &block);
      schro_encoder_motion_refine_block_subpel (frame, &block, i, j);
      schro_block_fixup (&block);
      schro_motion_copy_to (frame->me->motion, i, j, &block);
    }
  }
}

typedef struct
{
  int dx;
  int dy;
} MatchPos;


void
schro_encoder_motion_predict_subpel_deep (SchroMe * me)
{
  SchroParams *params = schro_me_params (me);
  double lambda = schro_me_lambda (me);
  int mvprec = 0, ref;
  int xblen = params->xbsep_luma, yblen = params->ybsep_luma;
  int i, j;
  int x_min, x_max, y_min, y_max;
  SchroMotionField *mf = NULL;
  SchroMotionVector *mv;
  SchroFrameData fd;
  SchroFrame *orig_frame = schro_me_src (me);
  const MatchPos sp_matches[] = { {-1, -1}, {0, -1}, {1, -1}
  , {-1, 0}, {1, 0}
  , {-1, 1}, {0, 1}, {1, 1}
  };


  x_min = y_min = -orig_frame->extension;

  if (1 < params->mv_precision) {
    fd.data = schro_malloc (xblen * yblen * sizeof (uint8_t));
    fd.stride = xblen;
    fd.height = yblen;
    fd.width = xblen;
    fd.format = SCHRO_FRAME_FORMAT_U8_420;

  }

  while (!(params->mv_precision < ++mvprec)) {
    x_max = (orig_frame->width << mvprec) + orig_frame->extension;
    y_max = (orig_frame->height << mvprec) + orig_frame->extension;
    for (ref = 0; ref < params->num_refs; ++ref) {
      SchroUpsampledFrame *upframe = schro_me_ref (me, ref);
      mf = schro_me_subpel_mf (me, ref);
      for (j = 0; params->y_num_blocks > j; ++j) {
        for (i = 0; params->x_num_blocks > i; ++i) {
          int error, entropy, min_error = INT_MAX, m = -1;
          double score, min_score = HUGE_VAL;
          int x, y, k;
          int dx, dy;
          int pred_x, pred_y;
          SchroFrameData orig, ref_data;
          int width, height;

          mv = &mf->motion_vectors[j * params->x_num_blocks + i];

          /* fetch source data (only process valid MVs) */
          if (!schro_frame_get_data (orig_frame, &orig, 0, i * xblen,
                  j * yblen)) {
            continue;
          }
          width = MIN (xblen, orig.width);
          height = MIN (yblen, orig.height);
          /* adjust MV precision */
          mv->u.vec.dx[ref] <<= 1;
          mv->u.vec.dy[ref] <<= 1;

          /* calculate score for current MV */
          schro_mf_vector_prediction (mf, i, j, &pred_x, &pred_y, ref + 1);
          entropy = schro_pack_estimate_sint (mv->u.vec.dx[ref] - pred_x);
          entropy += schro_pack_estimate_sint (mv->u.vec.dy[ref] - pred_y);
          min_score = entropy + lambda * mv->metric;
          x = i * (xblen << mvprec) + mv->u.vec.dx[ref];
          y = j * (yblen << mvprec) + mv->u.vec.dy[ref];
          /* check what matches are valid */
          for (k = 0; sizeof (sp_matches) / sizeof (sp_matches[0]) > k; ++k) {
            dx = x + sp_matches[k].dx;
            dy = y + sp_matches[k].dy;
            if (!(x_min < dx) || !(x_max > dx + xblen - 1)
                || !(y_min < dy) || !(y_max > dy + yblen - 1)) {
              continue;
            }
            fd.width = width;
            fd.height = height;
            schro_upsampled_frame_get_block_fast_precN (upframe, 0, dx, dy,
                mvprec, &ref_data, &fd);
            error =
                schro_metric_absdiff_u8 (orig.data, orig.stride, ref_data.data,
                ref_data.stride, width, height);
            /* calculate score */
            entropy =
                schro_pack_estimate_sint (mv->u.vec.dx[ref] + sp_matches[k].dx -
                pred_x);
            entropy +=
                schro_pack_estimate_sint (mv->u.vec.dy[ref] + sp_matches[k].dy -
                pred_y);
            score = entropy + lambda * error;
            if (min_score > score) {
              min_score = score;
              min_error = error;
              m = k;
            }
          }
          if (-1 != m) {
            mv->u.vec.dx[ref] += sp_matches[m].dx;
            mv->u.vec.dy[ref] += sp_matches[m].dy;
            mv->metric = min_error;
          }
        }
      }

    }
  }
  if (1 < params->mv_precision) {
    schro_free (fd.data);
  }
}

void
schro_motion_calculate_stats (SchroMotion * motion, SchroEncoderFrame * frame)
{
  int i, j;
  SchroMotionVector *mv;
  int ref1 = 0;
  int ref2 = 0;
  int bidir = 0;

  frame->stats_dc = 0;
  frame->stats_global = 0;
  frame->stats_motion = 0;
  for (j = 0; j < motion->params->y_num_blocks; j++) {
    for (i = 0; i < motion->params->x_num_blocks; i++) {
      mv = SCHRO_MOTION_GET_BLOCK (motion, i, j);
      if (mv->pred_mode == 0) {
        frame->stats_dc++;
      } else {
        if (mv->using_global) {
          frame->stats_global++;
        } else {
          frame->stats_motion++;
        }
        if (mv->pred_mode == 1) {
          ref1++;
        } else if (mv->pred_mode == 2) {
          ref2++;
        } else {
          bidir++;
        }
      }
    }
  }
  SCHRO_DEBUG ("dc %d global %d motion %d ref1 %d ref2 %d bidir %d",
      frame->stats_dc, frame->stats_global, frame->stats_motion,
      ref1, ref2, bidir);
}


SchroMotionField *
schro_motion_field_new (int x_num_blocks, int y_num_blocks)
{
  SchroMotionField *mf;

  mf = schro_malloc0 (sizeof (SchroMotionField));
  mf->x_num_blocks = x_num_blocks;
  mf->y_num_blocks = y_num_blocks;
  mf->motion_vectors = schro_malloc0 (sizeof (SchroMotionVector) *
      x_num_blocks * y_num_blocks);

  return mf;
}

void
schro_motion_field_free (SchroMotionField * field)
{
  schro_free (field->motion_vectors);
  schro_free (field);
}

void
schro_motion_field_set (SchroMotionField * field, int split, int pred_mode)
{
  SchroMotionVector *mv;
  int i;
  int j;

  for (j = 0; j < field->y_num_blocks; j++) {
    for (i = 0; i < field->x_num_blocks; i++) {
      mv = field->motion_vectors + j * field->x_num_blocks + i;
      memset (mv, 0, sizeof (*mv));
      mv->split = split;
      mv->pred_mode = pred_mode;
      mv->metric = 0;
    }
  }
}

void
schro_motion_field_copy (SchroMotionField * field, SchroMotionField * parent)
{
  SchroMotionVector *mv;
  SchroMotionVector *pv;
  int i;
  int j;

  for (j = 0; j < field->y_num_blocks; j++) {
    for (i = 0; i < field->x_num_blocks; i++) {
      mv = field->motion_vectors + j * field->x_num_blocks + i;
      pv = parent->motion_vectors + (j >> 1) * parent->x_num_blocks + (i >> 1);
      *mv = *pv;
    }
  }
}

#if 0
void
schro_motion_field_dump (SchroMotionField * field)
{
  SchroMotionVector *mv;
  int i;
  int j;

  for (j = 0; j < field->y_num_blocks; j++) {
    for (i = 0; i < field->x_num_blocks; i++) {
      mv = field->motion_vectors + j * field->x_num_blocks + i;
      printf ("%d %d %d %d\n", i, j, mv->u.vec.dx[0], mv->u.vec.dy[0]);
    }
  }
  exit (0);
}
#endif

static SchroFrame *
get_downsampled (SchroEncoderFrame * frame, int i)
{
  SCHRO_ASSERT (frame->have_downsampling);

  if (i == 0) {
    return frame->filtered_frame;
  }
  return frame->downsampled_frames[i - 1];
}

static int
schro_block_average (int16_t * dest, SchroFrameData * comp,
    int x, int y, int w, int h)
{
  int xmax = MIN (x + w, comp->width);
  int ymax = MIN (y + h, comp->height);
  int i, j;
  int n = 0;
  int sum = 0;
  int ave;

  if (x >= comp->width || y >= comp->height)
    return SCHRO_METRIC_INVALID_2;

  for (j = y; j < ymax; j++) {
    for (i = x; i < xmax; i++) {
      sum += SCHRO_GET (comp->data, j * comp->stride + i, uint8_t);
    }
    n += xmax - x;
  }

  if (n == 0) {
    return SCHRO_METRIC_INVALID_2;
  }

  ave = (sum + n / 2) / n;

  sum = 0;
  for (j = y; j < ymax; j++) {
    for (i = x; i < xmax; i++) {
      sum += abs (ave - SCHRO_GET (comp->data, j * comp->stride + i, uint8_t));
    }
  }

  *dest = ave - 128;
  return sum;
}


static void
schro_motionest_superblock_scan_one (SchroMotionEst * me, int ref, int distance,
    SchroBlock * block, int i, int j)
{
  SchroParams *params = me->params;
  SchroMotionVector *mv;
  SchroMetricScan scan;
  SchroMotionField *hint_mf;
  SchroMotionVector *hint_mv;
  int dx, dy;
  uint32_t dummy;

  scan.frame = get_downsampled (me->encoder_frame, 0);
  scan.ref_frame = get_downsampled (me->encoder_frame->ref_frame[ref], 0);

  //hint_mf = me->downsampled_mf[ref][2];
  hint_mf = me->encoder_frame->rme[ref]->motion_fields[2];

  scan.x = i * params->xbsep_luma;
  scan.y = j * params->ybsep_luma;
  scan.block_width = MIN (4 * params->xbsep_luma, scan.frame->width - scan.x);
  scan.block_height = MIN (4 * params->ybsep_luma, scan.frame->height - scan.y);
  scan.gravity_scale = 0;
  scan.gravity_x = 0;
  scan.gravity_y = 0;

  mv = &block->mv[0][0];
  hint_mv = motion_field_get (hint_mf, i, j);

  dx = hint_mv->u.vec.dx[ref];
  dy = hint_mv->u.vec.dy[ref];
  scan.gravity_x = dx;
  scan.gravity_y = dy;

  schro_metric_scan_setup (&scan, dx, dy, distance, FALSE);
  if (scan.scan_width <= 0 || scan.scan_height <= 0) {
    mv->u.vec.dx[ref] = 0;
    mv->u.vec.dy[ref] = 0;
    mv->metric = SCHRO_METRIC_INVALID;
    block->valid = FALSE;
    return;
  }

  schro_metric_scan_do_scan (&scan);
  block->error = schro_metric_scan_get_min (&scan, &dx, &dy, &dummy);
  mv->metric = block->error / 16;

  mv->split = 0;
  mv->pred_mode = 1 << ref;
  mv->using_global = 0;
  mv->u.vec.dx[ref] = dx;
  mv->u.vec.dy[ref] = dy;

  schro_block_fixup (block);
  block->entropy = schro_motion_superblock_try_estimate_entropy (me->motion,
      i, j, block);
  block->valid = TRUE;
}

static void
schro_motionest_superblock_predicted (SchroMotionEst * me, int ref,
    SchroBlock * block, int i, int j)
{
  SchroMotionVector *mv;
  int pred_x, pred_y;

  schro_motion_vector_prediction (me->motion, i, j, &pred_x, &pred_y,
      (1 << ref));

  mv = &block->mv[0][0];
  mv->split = 0;
  mv->pred_mode = 1 << ref;
  mv->using_global = 0;
  mv->u.vec.dx[ref] = pred_x;
  mv->u.vec.dy[ref] = pred_y;
  block->error = schro_motionest_superblock_get_metric (me, block, i, j);
  block->entropy = 0;
  schro_block_fixup (block);

  block->valid = (block->error != SCHRO_METRIC_INVALID_2);
}

static void
schro_motionest_superblock_biref_zero (SchroMotionEst * me,
    SchroBlock * block, int i, int j)
{
  SchroMotionVector *mv;

  mv = &block->mv[0][0];
  mv->split = 0;
  mv->pred_mode = 3;
  mv->using_global = 0;
  mv->u.vec.dx[0] = 0;
  mv->u.vec.dy[0] = 0;
  mv->u.vec.dx[1] = 0;
  mv->u.vec.dy[1] = 0;
  block->error = schro_motionest_superblock_get_metric (me, block, i, j);
  schro_block_fixup (block);
  block->entropy = schro_motion_superblock_try_estimate_entropy (me->motion,
      i, j, block);
  block->valid = (block->error != SCHRO_METRIC_INVALID_2);
}

static void
schro_motionest_superblock_dc (SchroMotionEst * me,
    SchroBlock * block, int i, int j)
{
  SchroParams *params = me->params;
  SchroMotionVector *mv;
  int chroma_w, chroma_h;
  SchroFrame *frame;
  int metric;

  frame = get_downsampled (me->encoder_frame, 0);

  mv = &block->mv[0][0];
  mv->split = 0;
  mv->pred_mode = 0;

  metric = schro_block_average (&mv->u.dc.dc[0], frame->components + 0,
      i * params->xbsep_luma, j * params->ybsep_luma,
      4 * params->xbsep_luma, 4 * params->ybsep_luma);
  if (metric == SCHRO_METRIC_INVALID_2) {
    block->valid = FALSE;
    return;
  }
  chroma_w =
      params->xbsep_luma >> SCHRO_CHROMA_FORMAT_H_SHIFT (params->
      video_format->chroma_format);
  chroma_h =
      params->ybsep_luma >> SCHRO_CHROMA_FORMAT_V_SHIFT (params->
      video_format->chroma_format);
  schro_block_average (&mv->u.dc.dc[1], frame->components + 1, i * chroma_w,
      j * chroma_h, 4 * chroma_w, 4 * chroma_h);
  schro_block_average (&mv->u.dc.dc[2], frame->components + 2, i * chroma_w,
      j * chroma_h, 4 * chroma_w, 4 * chroma_h);

  mv->metric = metric;
  block->error = metric;
  block->error += 4 * 2 * me->params->xbsep_luma *
      me->encoder_frame->encoder->magic_dc_metric_offset;

  schro_block_fixup (block);

  block->entropy = schro_motion_superblock_try_estimate_entropy (me->motion,
      i, j, block);
  block->valid = TRUE;
}

static void
schro_motionest_superblock_dc_predicted (SchroMotionEst * me,
    SchroBlock * block, int i, int j)
{
  SchroMotionVector *mv;
  int pred[3];

  schro_motion_dc_prediction (me->motion, i, j, pred);

  mv = &block->mv[0][0];
  mv->split = 0;
  mv->pred_mode = 0;
  mv->u.dc.dc[0] = pred[0];
  mv->u.dc.dc[1] = pred[1];
  mv->u.dc.dc[2] = pred[2];

  block->error = schro_motionest_superblock_get_metric (me, block, i, j);
  mv->metric = block->error;
  block->error += 4 * 2 * me->params->xbsep_luma *
      me->encoder_frame->encoder->magic_dc_metric_offset;

  schro_block_fixup (block);
  block->entropy = 0;
  block->valid = TRUE;
}

#ifdef unused
static void
schro_motion_splat_4x4 (SchroMotion * motion, int i, int j)
{
  SchroMotionVector *mv;

  mv = SCHRO_MOTION_GET_BLOCK (motion, i, j);
  mv[1] = mv[0];
  mv[2] = mv[0];
  mv[3] = mv[0];
  memcpy (SCHRO_MOTION_GET_BLOCK (motion, i, j + 1), mv, 4 * sizeof (*mv));
  memcpy (SCHRO_MOTION_GET_BLOCK (motion, i, j + 2), mv, 4 * sizeof (*mv));
  memcpy (SCHRO_MOTION_GET_BLOCK (motion, i, j + 3), mv, 4 * sizeof (*mv));
}
#endif

#ifdef unused
static void
motion_field_splat_4x4 (SchroMotionField * mf, int i, int j)
{
  SchroMotionVector *mv;

  mv = motion_field_get (mf, i, j);
  mv[1] = mv[0];
  mv[2] = mv[0];
  mv[3] = mv[0];
  memcpy (motion_field_get (mf, i, j + 1), mv, 4 * sizeof (*mv));
  memcpy (motion_field_get (mf, i, j + 2), mv, 4 * sizeof (*mv));
  memcpy (motion_field_get (mf, i, j + 3), mv, 4 * sizeof (*mv));
}
#endif

#ifdef unused
static void
schro_motionest_block_scan_one (SchroMotionEst * me, int ref, int distance,
    SchroBlock * block, int i, int j)
{
  SchroParams *params = me->params;
  SchroMotionVector *mv;
  SchroMetricScan scan;
  SchroMotionField *hint_mf;
  SchroMotionVector *hint_mv;
  int dx, dy;
  int ii, jj;

  scan.frame = get_downsampled (me->encoder_frame, 0);
  scan.ref_frame = get_downsampled (me->encoder_frame->ref_frame[ref], 0);

  hint_mf = me->encoder_frame->rme[ref]->motion_fields[1];

  scan.block_width = params->xbsep_luma;
  scan.block_height = params->ybsep_luma;
  scan.gravity_scale = 0;
  scan.gravity_x = 0;
  scan.gravity_y = 0;

  block->error = 0;
  block->valid = TRUE;
  for (jj = 0; jj < 4; jj++) {
    for (ii = 0; ii < 4; ii++) {
      mv = &block->mv[jj][ii];
      hint_mv = motion_field_get (hint_mf, i + (ii & 2), j + (jj & 2));

      dx = hint_mv->u.vec.dx[ref];
      dy = hint_mv->u.vec.dy[ref];
      scan.gravity_x = dx;
      scan.gravity_y = dy;

      scan.x = (i + ii) * params->xbsep_luma;
      scan.y = (j + jj) * params->ybsep_luma;
      schro_metric_scan_setup (&scan, dx, dy, distance, FALSE);
      if (scan.scan_width <= 0 || scan.scan_height <= 0) {
        mv->u.vec.dx[ref] = 0;
        mv->u.vec.dy[ref] = 0;
        mv->metric = SCHRO_METRIC_INVALID;
        block->error += mv->metric;
        block->valid = FALSE;
        continue;
      }

      schro_metric_scan_do_scan (&scan);
      uint32_t dummy;
      mv->metric = schro_metric_scan_get_min (&scan, &dx, &dy, &dummy);
      block->error += mv->metric;
      block->valid &= (mv->metric != SCHRO_METRIC_INVALID);

      mv->split = 2;
      mv->pred_mode = 1 << ref;
      mv->using_global = 0;
      mv->u.vec.dx[ref] = dx;
      mv->u.vec.dy[ref] = dy;
    }
  }

  schro_block_fixup (block);
  block->entropy = schro_motion_superblock_try_estimate_entropy (me->motion,
      i, j, block);
}
#endif


#define MAGIC_SUPERBLOCK_METRIC 5
#define MAGIC_BLOCK_METRIC 50

#define TRYBLOCK \
      score = tryblock.entropy + me->lambda * tryblock.error; \
      if (tryblock.valid && score < min_score) { \
        memcpy (&block, &tryblock, sizeof(block)); \
        min_score = score; \
      }

static void
schro_motionest_block_scan (SchroMotionEst * me, int ref, int distance,
    SchroBlock * block, int i, int j, int ii, int jj)
{
  SchroParams *params = me->params;
  SchroMotionVector *mv;
  SchroMetricScan scan;
  SchroMotionField *hint_mf;
  SchroMotionVector *hint_mv;
  int dx, dy;
  uint32_t dummy;

  scan.frame = get_downsampled (me->encoder_frame, 0);
  scan.ref_frame = get_downsampled (me->encoder_frame->ref_frame[ref], 0);

  hint_mf = me->encoder_frame->rme[ref]->motion_fields[1];

  scan.block_width = params->xbsep_luma;
  scan.block_height = params->ybsep_luma;

  scan.gravity_scale = 0;
  scan.gravity_x = 0;
  scan.gravity_y = 0;

  mv = &block->mv[jj][ii];
  hint_mv = motion_field_get (hint_mf, i + (ii & 2), j + (jj & 2));

  dx = hint_mv->u.vec.dx[ref];
  dy = hint_mv->u.vec.dy[ref];
  scan.gravity_x = dx;
  scan.gravity_y = dy;

  scan.x = (i + ii) * params->xbsep_luma;
  scan.y = (j + jj) * params->ybsep_luma;
  if (!(scan.x < scan.frame->width) || !(scan.y < scan.frame->height)) {
    mv->u.vec.dx[ref] = 0;
    mv->u.vec.dy[ref] = 0;
    mv->metric = SCHRO_METRIC_INVALID;
    block->error += mv->metric;
    block->valid = FALSE;
    return;
  }
  scan.block_width = MIN (params->xbsep_luma, scan.frame->width - scan.x);
  scan.block_height = MIN (params->ybsep_luma, scan.frame->height - scan.y);
  schro_metric_scan_setup (&scan, dx, dy, distance, FALSE);
  if (scan.scan_width <= 0 || scan.scan_height <= 0) {
    mv->u.vec.dx[ref] = 0;
    mv->u.vec.dy[ref] = 0;
    mv->metric = SCHRO_METRIC_INVALID;
    block->error += mv->metric;
    block->valid = FALSE;
    return;
  }

  schro_metric_scan_do_scan (&scan);
  mv->metric = schro_metric_scan_get_min (&scan, &dx, &dy, &dummy);
  block->error = mv->metric;
  block->valid = (mv->metric != SCHRO_METRIC_INVALID);

  mv->split = 2;
  mv->pred_mode = 1 << ref;
  mv->using_global = 0;
  mv->u.vec.dx[ref] = dx;
  mv->u.vec.dy[ref] = dy;

  schro_block_fixup (block);

  mv = SCHRO_MOTION_GET_BLOCK (me->motion, i + ii, j + jj);
  *mv = block->mv[jj][ii];
  block->entropy = schro_motion_block_estimate_entropy (me->motion,
      i + ii, j + jj);
}

static void
schro_motionest_block_dc (SchroMotionEst * me,
    SchroBlock * block, int i, int j, int ii, int jj)
{
  SchroParams *params = me->params;
  SchroMotionVector *mv;
  int chroma_w, chroma_h;
  SchroFrame *frame;
  int metric;

  frame = get_downsampled (me->encoder_frame, 0);

  mv = &(block->mv[jj][ii]);
  mv->split = 2;
  mv->pred_mode = 0;

  metric = schro_block_average (&mv->u.dc.dc[0], frame->components + 0,
      (i + ii) * params->xbsep_luma, (j + jj) * params->ybsep_luma,
      params->xbsep_luma, params->ybsep_luma);
  if (metric == SCHRO_METRIC_INVALID_2) {
    block->valid = FALSE;
    return;
  }
  chroma_w =
      params->xbsep_luma >> SCHRO_CHROMA_FORMAT_H_SHIFT (params->
      video_format->chroma_format);
  chroma_h =
      params->ybsep_luma >> SCHRO_CHROMA_FORMAT_V_SHIFT (params->
      video_format->chroma_format);
  schro_block_average (&mv->u.dc.dc[1], frame->components + 1,
      (i + ii) * chroma_w, (j + jj) * chroma_h, chroma_w, chroma_h);
  schro_block_average (&mv->u.dc.dc[2], frame->components + 2,
      (i + ii) * chroma_w, (j + jj) * chroma_h, chroma_w, chroma_h);

  mv->metric = metric;
  block->error = metric;
  block->error += 4 * 2 * me->params->xbsep_luma *
      me->encoder_frame->encoder->magic_dc_metric_offset;

  block->entropy = schro_motion_block_estimate_entropy (me->motion,
      i + ii, j + jj);
  block->valid = TRUE;
}

static void
schro_motionest_superblock_block (SchroMotionEst * me,
    SchroBlock * p_block, int i, int j)
{
  SchroParams *params = me->params;
  int ii, jj;
  SchroBlock block = { 0 };
  int total_error = 0;

  for (jj = 0; jj < 4; jj++) {
    for (ii = 0; ii < 4; ii++) {
      block.mv[jj][ii].split = 2;
      block.mv[jj][ii].pred_mode = 1;
      block.mv[jj][ii].u.vec.dx[0] = 0;
      block.mv[jj][ii].u.vec.dy[0] = 0;
    }
  }
  schro_motion_copy_to (me->motion, i, j, &block);

  for (jj = 0; jj < 4; jj++) {
    for (ii = 0; ii < 4; ii++) {
      SchroBlock tryblock = { 0 };
      double score;
      double min_score;

      /* FIXME use better default than DC */
      schro_motionest_block_dc (me, &tryblock, i, j, ii, jj);
      min_score = block.entropy + me->lambda * block.error;

      if (me->encoder_frame->encoder->enable_hierarchical_estimation) {
        memcpy (&tryblock, &block, sizeof (block));
        schro_motionest_block_scan (me, 0, me->scan_distance, &block, i, j, ii,
            jj);
        min_score = block.entropy + me->lambda * block.error;

        if (params->num_refs > 1) {
          memcpy (&tryblock, &block, sizeof (block));
          schro_motionest_block_scan (me, 1, me->scan_distance, &tryblock, i, j,
              ii, jj);
        TRYBLOCK}
      }

      memcpy (&tryblock, &block, sizeof (block));
      schro_motionest_block_dc (me, &tryblock, i, j, ii, jj);
      TRYBLOCK total_error += block.error;
    }
  }
  block.entropy = schro_motion_superblock_try_estimate_entropy (me->motion,
      i, j, &block);
  block.error = total_error;

  memcpy (p_block, &block, sizeof (block));
}

static void
schro_motionest_subsuperblock_scan (SchroMotionEst * me, int ref, int distance,
    SchroBlock * block, int i, int j, int ii, int jj)
{
  SchroParams *params = me->params;
  SchroMotionVector *mv;
  SchroMetricScan scan;
  SchroMotionField *hint_mf;
  SchroMotionVector *hint_mv;
  int dx, dy;
  uint32_t dummy;

  scan.frame = get_downsampled (me->encoder_frame, 0);
  scan.ref_frame = get_downsampled (me->encoder_frame->ref_frame[ref], 0);

  hint_mf = me->encoder_frame->rme[ref]->motion_fields[1];

  scan.block_width = 2 * params->xbsep_luma;
  scan.block_height = 2 * params->ybsep_luma;

  scan.gravity_scale = 0;
  scan.gravity_x = 0;
  scan.gravity_y = 0;

  mv = &block->mv[jj][ii];
  hint_mv = motion_field_get (hint_mf, i + (ii & 2), j + (jj & 2));

  dx = hint_mv->u.vec.dx[ref];
  dy = hint_mv->u.vec.dy[ref];
  scan.gravity_x = dx;
  scan.gravity_y = dy;

  scan.x = (i + ii) * params->xbsep_luma;
  scan.y = (j + jj) * params->ybsep_luma;
  if (!(scan.x < scan.frame->width) || !(scan.y < scan.frame->height)) {
    mv->u.vec.dx[ref] = mv->u.vec.dy[ref] = 0;
    mv->metric = SCHRO_METRIC_INVALID;
    block->error += mv->metric;
    block->valid = FALSE;
    return;
  }
  scan.block_width = MIN (2 * params->xbsep_luma, scan.frame->width - scan.x);
  scan.block_height = MIN (2 * params->ybsep_luma, scan.frame->height - scan.y);
  schro_metric_scan_setup (&scan, dx, dy, distance, FALSE);
  if (scan.scan_width <= 0 || scan.scan_height <= 0) {
    mv->u.vec.dx[ref] = 0;
    mv->u.vec.dy[ref] = 0;
    mv->metric = SCHRO_METRIC_INVALID;
    block->error += mv->metric;
    block->valid = FALSE;
    return;
  }

  schro_metric_scan_do_scan (&scan);
  mv->metric = schro_metric_scan_get_min (&scan, &dx, &dy, &dummy);
  block->error = mv->metric;
  block->valid = (mv->metric != SCHRO_METRIC_INVALID);

  mv->split = 1;
  mv->pred_mode = 1 << ref;
  mv->using_global = 0;
  mv->u.vec.dx[ref] = dx;
  mv->u.vec.dy[ref] = dy;

  schro_block_fixup (block);

  mv = SCHRO_MOTION_GET_BLOCK (me->motion, i + ii, j + jj);
  *mv = block->mv[jj][ii];
  block->entropy = schro_motion_block_estimate_entropy (me->motion,
      i + ii, j + jj);
}

static void
schro_motionest_subsuperblock_dc (SchroMotionEst * me,
    SchroBlock * block, int i, int j, int ii, int jj)
{
  SchroParams *params = me->params;
  SchroMotionVector *mv;
  int chroma_w, chroma_h;
  SchroFrame *frame;
  int metric;

  frame = get_downsampled (me->encoder_frame, 0);

  mv = (SchroMotionVector *) & block->mv[jj][ii];
  mv->split = 1;
  mv->pred_mode = 0;

  metric = schro_block_average (&mv->u.dc.dc[0], frame->components + 0,
      (i + ii) * params->xbsep_luma, (j + jj) * params->ybsep_luma,
      2 * params->xbsep_luma, 2 * params->ybsep_luma);
  if (metric == SCHRO_METRIC_INVALID_2) {
    block->valid = FALSE;
    return;
  }
  chroma_w =
      params->xbsep_luma >> SCHRO_CHROMA_FORMAT_H_SHIFT (params->
      video_format->chroma_format);
  chroma_h =
      params->ybsep_luma >> SCHRO_CHROMA_FORMAT_V_SHIFT (params->
      video_format->chroma_format);
  schro_block_average (&mv->u.dc.dc[1], frame->components + 1,
      (i + ii) * chroma_w, (j + jj) * chroma_h, 2 * chroma_w, 2 * chroma_h);
  schro_block_average (&mv->u.dc.dc[2], frame->components + 2,
      (i + ii) * chroma_w, (j + jj) * chroma_h, 2 * chroma_w, 2 * chroma_h);

  mv->metric = metric;
  block->error = metric;
  block->error += 4 * 2 * me->params->xbsep_luma *
      me->encoder_frame->encoder->magic_dc_metric_offset;

  block->entropy = schro_motion_block_estimate_entropy (me->motion,
      i + ii, j + jj);
  block->valid = TRUE;
}

static void
schro_motionest_superblock_subsuperblock (SchroMotionEst * me,
    SchroBlock * p_block, int i, int j)
{
  SchroParams *params = me->params;
  int ii, jj;
  SchroBlock block = { 0 };
  int total_error = 0;

  for (jj = 0; jj < 4; jj++) {
    for (ii = 0; ii < 4; ii++) {
      block.mv[jj][ii].split = 1;
      block.mv[jj][ii].pred_mode = 1;
      block.mv[jj][ii].u.vec.dx[0] = 0;
      block.mv[jj][ii].u.vec.dy[0] = 0;
    }
  }
  schro_motion_copy_to (me->motion, i, j, &block);

  for (jj = 0; jj < 4; jj += 2) {
    for (ii = 0; ii < 4; ii += 2) {
      SchroBlock tryblock = { 0 };
      double score;
      double min_score;

      /* FIXME use better default than DC */
      schro_motionest_subsuperblock_dc (me, &block, i, j, ii, jj);
      min_score = block.entropy + me->lambda * block.error;

      if (me->encoder_frame->encoder->enable_hierarchical_estimation) {
        memcpy (&tryblock, &block, sizeof (block));
        schro_motionest_subsuperblock_scan (me, 0, me->scan_distance, &tryblock,
            i, j, ii, jj);
        TRYBLOCK if (params->num_refs > 1)
        {
          memcpy (&tryblock, &block, sizeof (block));
          schro_motionest_subsuperblock_scan (me, 1, me->scan_distance,
              &tryblock, i, j, ii, jj);
          TRYBLOCK
#if 0
              memcpy (&tryblock, &block, sizeof (block));
          schro_motionest_block_biref_zero (me, 1, &tryblock, i, j, ii, jj);
          TRYBLOCK
#endif
        }
      }

      memcpy (&tryblock, &block, sizeof (block));
      schro_motionest_subsuperblock_dc (me, &tryblock, i, j, ii, jj);
      TRYBLOCK total_error += block.error;
    }
  }
  block.entropy = schro_motion_superblock_try_estimate_entropy (me->motion,
      i, j, &block);
  block.error = total_error;

  memcpy (p_block, &block, sizeof (block));
}


void
schro_encoder_bigblock_estimation (SchroMotionEst * me)
{
  SchroParams *params = me->params;
  int i, j;
  double total_error = 0;
  int block_size;
  int block_threshold;

  me->lambda = me->encoder_frame->frame_me_lambda;

  block_size = 16 * params->xbsep_luma * params->ybsep_luma;
  block_threshold = params->xbsep_luma * params->ybsep_luma *
      me->encoder_frame->encoder->magic_block_search_threshold;

  for (j = 0; j < params->y_num_blocks; j += 4) {
    for (i = 0; i < params->x_num_blocks; i += 4) {
      SchroBlock block = { 0 };
      SchroBlock tryblock = { 0 };
      double score;
      double min_score;

      /* base 119 s */
      schro_motionest_superblock_predicted (me, 0, &block, i, j);
      min_score = block.entropy + me->lambda * block.error;
      if (params->num_refs > 1) {
        schro_motionest_superblock_predicted (me, 1, &tryblock, i, j);
      TRYBLOCK}

      if (me->encoder_frame->encoder->enable_hierarchical_estimation) {
        /* 16 s */
        schro_motionest_superblock_scan_one (me, 0, me->scan_distance,
            &tryblock, i, j);
        TRYBLOCK if (params->num_refs > 1)
        {
          schro_motionest_superblock_scan_one (me, 1, me->scan_distance,
              &tryblock, i, j);
        TRYBLOCK}
      }

      /* 2.5 s */
      schro_motionest_superblock_dc_predicted (me, &tryblock, i, j);
      TRYBLOCK schro_motionest_superblock_dc (me, &tryblock, i, j);
      TRYBLOCK
          /* 3.0 s */
          if (params->num_refs > 1) {
        schro_motionest_superblock_biref_zero (me, &tryblock, i, j);
      TRYBLOCK}

      if (min_score > block_threshold || block.mv[0][0].pred_mode == 0) {
        schro_motionest_superblock_subsuperblock (me, &tryblock, i, j);
        TRYBLOCK schro_motionest_superblock_block (me, &tryblock, i, j);
      TRYBLOCK}

      if (me->encoder_frame->encoder->enable_phasecorr_estimation) {
        schro_motionest_superblock_phasecorr1 (me, 0, &tryblock, i, j);
        TRYBLOCK if (params->num_refs > 1)
        {
          schro_motionest_superblock_phasecorr1 (me, 1, &tryblock, i, j);
        TRYBLOCK}
      }

      if (me->encoder_frame->encoder->enable_global_motion) {
        schro_motionest_superblock_global (me, 0, &tryblock, i, j);
        TRYBLOCK if (params->num_refs > 1)
        {
          schro_motionest_superblock_global (me, 1, &tryblock, i, j);
        TRYBLOCK}
      }

      if (block.error > 10 * block_size) {
        me->badblocks++;
      }

      schro_block_fixup (&block);
      schro_motion_copy_to (me->motion, i, j, &block);

      total_error +=
          (double) block.error * block.error / (double) (block_size *
          block_size);
    }
  }

  me->encoder_frame->mc_error = total_error / (240.0 * 240.0) /
      (params->x_num_blocks * params->y_num_blocks / 16);

  /* magic parameter */
  me->encoder_frame->mc_error *= 2.5;
}

int
schro_motion_block_estimate_entropy (SchroMotion * motion, int i, int j)
{
  SchroMotionVector *mv;
  int entropy = 0;

  mv = SCHRO_MOTION_GET_BLOCK (motion, i, j);

  if (mv->split == 0 && (i & 3 || j & 3))
    return 0;
  if (mv->split == 1 && (i & 1 || j & 1))
    return 0;

  if (mv->pred_mode == 0) {
    int pred[3];

    schro_motion_dc_prediction (motion, i, j, pred);

    entropy += schro_pack_estimate_sint (mv->u.dc.dc[0] - pred[0]);
    entropy += schro_pack_estimate_sint (mv->u.dc.dc[1] - pred[1]);
    entropy += schro_pack_estimate_sint (mv->u.dc.dc[2] - pred[2]);

    return entropy;
  }
  if (mv->using_global)
    return 0;
  if (mv->pred_mode & 1) {
    int pred_x, pred_y;
    schro_motion_vector_prediction (motion, i, j, &pred_x, &pred_y, 1);
    entropy += schro_pack_estimate_sint (mv->u.vec.dx[0] - pred_x);
    entropy += schro_pack_estimate_sint (mv->u.vec.dy[0] - pred_y);
  }
  if (mv->pred_mode & 2) {
    int pred_x, pred_y;
    schro_motion_vector_prediction (motion, i, j, &pred_x, &pred_y, 2);
    entropy += schro_pack_estimate_sint (mv->u.vec.dx[1] - pred_x);
    entropy += schro_pack_estimate_sint (mv->u.vec.dy[1] - pred_y);
  }
  return entropy;
}

int
schro_motion_estimate_entropy (SchroMotion * motion)
{
  SchroParams *params = motion->params;
  int i, j;
  int entropy = 0;

  for (j = 0; j < params->y_num_blocks; j++) {
    for (i = 0; i < params->x_num_blocks; i++) {
      entropy += schro_motion_block_estimate_entropy (motion, i, j);
    }
  }

  return entropy;
}

#ifdef unused
int
schro_motion_superblock_estimate_entropy (SchroMotion * motion, int i, int j)
{
  int ii, jj;
  int entropy = 0;

  for (jj = j; jj < j + 4; jj++) {
    for (ii = i; ii < i + 4; ii++) {
      entropy += schro_motion_block_estimate_entropy (motion, ii, jj);
    }
  }

  return entropy;
}
#endif

int
schro_motion_superblock_try_estimate_entropy (SchroMotion * motion, int i,
    int j, SchroBlock * block)
{
  int ii, jj;
  int entropy = 0;
  SchroBlock save_block;

  schro_motion_copy_from (motion, i, j, &save_block);
  schro_motion_copy_to (motion, i, j, block);
  for (jj = 0; jj < 4; jj++) {
    for (ii = 0; ii < 4; ii++) {
      entropy += schro_motion_block_estimate_entropy (motion, i + ii, j + jj);
    }
  }
  schro_motion_copy_to (motion, i, j, &save_block);

  return entropy;
}

int
schro_motionest_superblock_get_metric (SchroMotionEst * me,
    SchroBlock * block, int i, int j)
{
  SchroMotionVector *mv;
  SchroFrameData orig;
  int width, height;
  int xmin, ymin;
  int xmax, ymax;

  xmin = MAX (i * me->params->xbsep_luma, 0);
  ymin = MAX (j * me->params->ybsep_luma, 0);
  xmax =
      MIN ((i + 4) * me->params->xbsep_luma,
      me->encoder_frame->filtered_frame->width);
  ymax =
      MIN ((j + 4) * me->params->ybsep_luma,
      me->encoder_frame->filtered_frame->height);

  schro_frame_get_subdata (get_downsampled (me->encoder_frame, 0), &orig,
      0, xmin, ymin);

  width = xmax - xmin;
  height = ymax - ymin;

  mv = &block->mv[0][0];

  if (mv->pred_mode == 0) {
    return schro_metric_get_dc (&orig, mv->u.dc.dc[0], width, height);
  }
  if (mv->pred_mode == 1 || mv->pred_mode == 2) {
    SchroFrame *ref_frame;
    SchroFrameData ref_data;
    int ref;

    ref = mv->pred_mode - 1;

    ref_frame = get_downsampled (me->encoder_frame->ref_frame[ref], 0);

    if (xmin + mv->u.vec.dx[ref] < -ref_frame->extension ||
        ymin + mv->u.vec.dy[ref] < -ref_frame->extension ||
        xmax + mv->u.vec.dx[ref] >
        me->encoder_frame->filtered_frame->width + ref_frame->extension
        || ymax + mv->u.vec.dy[ref] >
        me->encoder_frame->filtered_frame->height + ref_frame->extension) {
      /* bailing because it's "hard" */
      return SCHRO_METRIC_INVALID_2;
    }

    schro_frame_get_subdata (ref_frame, &ref_data,
        0, xmin + mv->u.vec.dx[ref], ymin + mv->u.vec.dy[ref]);

    return schro_metric_get (&orig, &ref_data, width, height);
  }

  if (mv->pred_mode == 3) {
    SchroFrame *ref0_frame;
    SchroFrame *ref1_frame;
    SchroFrameData ref0_data;
    SchroFrameData ref1_data;

    ref0_frame = get_downsampled (me->encoder_frame->ref_frame[0], 0);
    ref1_frame = get_downsampled (me->encoder_frame->ref_frame[1], 0);

    if (xmin + mv->u.vec.dx[0] < -ref0_frame->extension ||
        ymin + mv->u.vec.dy[0] < -ref0_frame->extension ||
        xmax + mv->u.vec.dx[0] >
        me->encoder_frame->filtered_frame->width + ref0_frame->extension
        || ymax + mv->u.vec.dy[0] >
        me->encoder_frame->filtered_frame->height + ref0_frame->extension
        || xmin + mv->u.vec.dx[1] < -ref1_frame->extension
        || ymin + mv->u.vec.dy[1] < -ref1_frame->extension
        || xmax + mv->u.vec.dx[1] >
        me->encoder_frame->filtered_frame->width + ref1_frame->extension
        || ymax + mv->u.vec.dy[1] >
        me->encoder_frame->filtered_frame->height + ref1_frame->extension) {
      /* bailing because it's "hard" */
      return SCHRO_METRIC_INVALID_2;
    }

    schro_frame_get_subdata (ref0_frame,
        &ref0_data, 0, xmin + mv->u.vec.dx[0], ymin + mv->u.vec.dy[0]);
    schro_frame_get_subdata (ref1_frame,
        &ref1_data, 0, xmin + mv->u.vec.dx[1], ymin + mv->u.vec.dy[1]);

    return schro_metric_get_biref (&orig, &ref0_data, 1, &ref1_data, 1, 1,
        width, height);
  }

  SCHRO_ASSERT (0);

  return SCHRO_METRIC_INVALID_2;
}

#ifdef unused
int
schro_block_check (SchroBlock * block)
{
  SchroMotionVector *sbmv;
  SchroMotionVector *bmv;
  SchroMotionVector *mv;
  int i, j;

  sbmv = &block->mv[0][0];
  for (j = 0; j < 4; j++) {
    for (i = 0; i < 4; i++) {
      mv = &block->mv[j][i];

      switch (sbmv->split) {
        case 0:
          if (!schro_motion_vector_is_equal (mv, sbmv)) {
            SCHRO_ERROR ("mv(%d,%d) not equal to superblock mv", i, j);
            return 0;
          }
          break;
        case 1:
          bmv = &block->mv[(j & ~1)][(i & ~1)];
          if (!schro_motion_vector_is_equal (mv, bmv)) {
            SCHRO_ERROR ("mv(%d,%d) not equal to 2-block mv", i, j);
            return 0;
          }
          break;
        case 2:
          break;
        default:
          SCHRO_ERROR ("mv(%d,%d) has bad split", i, j);
          return 0;
          break;
      }
    }
  }

  return 1;
}
#endif

void
schro_block_fixup (SchroBlock * block)
{
  SchroMotionVector *mv;

  mv = &block->mv[0][0];
  if (mv->split == 0) {
    mv[1] = mv[0];
    mv[2] = mv[0];
    mv[3] = mv[0];
    memcpy (mv + 4, mv, 4 * sizeof (*mv));
    memcpy (mv + 8, mv, 4 * sizeof (*mv));
    memcpy (mv + 12, mv, 4 * sizeof (*mv));
  }
  if (mv->split == 1) {
    mv[1] = mv[0];
    mv[3] = mv[2];
    memcpy (mv + 4, mv, 4 * sizeof (*mv));
    mv[9] = mv[8];
    mv[11] = mv[10];
    memcpy (mv + 12, mv + 8, 4 * sizeof (*mv));
  }
}

void
schro_motion_copy_from (SchroMotion * motion, int i, int j, SchroBlock * block)
{
  SchroMotionVector *mv;
  int ii, jj;

  for (jj = 0; jj < 4; jj++) {
    for (ii = 0; ii < 4; ii++) {
      mv = SCHRO_MOTION_GET_BLOCK (motion, i + ii, j + jj);
      block->mv[jj][ii] = *mv;
    }
  }
}

void
schro_motion_copy_to (SchroMotion * motion, int i, int j, SchroBlock * block)
{
  SchroMotionVector *mv;
  int ii, jj;

  for (jj = 0; jj < 4; jj++) {
    for (ii = 0; ii < 4; ii++) {
      mv = SCHRO_MOTION_GET_BLOCK (motion, i + ii, j + jj);
      *mv = block->mv[jj][ii];
    }
  }
}

/* performs single position block matching for split2 
 * to calculate SAD */
static void
schro_get_split2_metric (SchroMe * me, int ref, int i, int j,
    SchroMotionVector * mv, int *metric, SchroFrameData * fd)
{
  SchroParams *params = schro_me_params (me);
  SchroFrameData orig[2], ref_data[2];
  SchroFrame *frame = schro_me_src (me);
  SchroUpsampledFrame *upframe;
  int mv_prec = params->mv_precision;
  int fd_width, fd_height;
  int width[3], height[3], dx, dy;
  int k;
  int xmin, xmax, ymin, ymax, tmp_x, tmp_y;
  int block_x[3], block_y[3];

  xmin = ymin = -frame->extension;
  xmax = (frame->width << mv_prec) + frame->extension;
  ymax = (frame->height << mv_prec) + frame->extension;
  /* calculates split2 block sizes for all components */
  block_x[0] = params->xbsep_luma;
  block_y[0] = params->ybsep_luma;
  block_x[1] = block_x[2] =
      block_x[0] >> SCHRO_CHROMA_FORMAT_H_SHIFT (params->
      video_format->chroma_format);
  block_y[1] = block_y[2] =
      block_y[0] >> SCHRO_CHROMA_FORMAT_V_SHIFT (params->
      video_format->chroma_format);

  /* get source data if possible */
  for (k = 1; 3 > k; ++k) {
    if (!schro_frame_get_data (frame, &orig[k - 1], k, i * block_x[k],
            j * block_y[k])) {
      *metric = INT_MAX;
      return;
    }
    width[k] = MIN (block_x[k], orig[k - 1].width);
    height[k] = MIN (block_y[k], orig[k - 1].height);
  }
  upframe = schro_me_ref (me, ref);
  *metric = 0;
  for (k = 1; 3 > k; ++k) {
    /* get ref data, if possible */
    tmp_x = (i * block_x[k]) << mv_prec;
    tmp_y = (j * block_y[k]) << mv_prec;
    dx = mv->u.vec.dx[ref];
    dx >>= SCHRO_CHROMA_FORMAT_H_SHIFT (params->video_format->chroma_format);
    dx += tmp_x;
    dy = mv->u.vec.dy[ref];
    dy >>= SCHRO_CHROMA_FORMAT_V_SHIFT (params->video_format->chroma_format);
    dy += tmp_y;
    if (INT_MAX == mv->metric) {
      *metric = INT_MAX;
      return;
    }
    /* I need to save the original value of fd width and height */
    fd_width = fd->width;
    fd_height = fd->height;
    fd->width = width[k];
    fd->height = height[k];

    schro_upsampled_frame_get_block_fast_precN (upframe, k, dx, dy, mv_prec,
        &ref_data[k - 1], fd);
    fd->width = fd_width;
    fd->height = fd_height;
    *metric +=
        schro_metric_absdiff_u8 (orig[k - 1].data, orig[k - 1].stride,
        ref_data[k - 1].data, ref_data[k - 1].stride, width[k], height[k]);
  }
  mv->chroma_metric = *metric;
  *metric += mv->metric;
  return;
}


/* performs mode decision for a superblock, split level 2
 * Note that a SchroMotion object is required to estimate
 * the cost of the different prediction modes. */
static void
schro_do_split2 (SchroMe * me, int i, int j, SchroBlock * block,
    SchroFrameData * fd)
{
  int total_entropy = 0, total_error = 0, ii, jj, ref;
  double lambda = schro_me_lambda (me); /* frame->frame_me_lambda; */
  int comp_w[3], comp_h[3];
  SchroParams *params = schro_me_params (me);
  SchroMotion *motion = schro_me_motion (me);
  int mvprec = params->mv_precision;
  int xnum_blocks = params->x_num_blocks;
  int fd_width, fd_height, biref;
  int xblen = params->xbsep_luma, yblen = params->ybsep_luma;
  SchroFrame *orig_frame = schro_me_src (me);
  SchroFrameData ref_data[3][2], orig[3];
  SchroUpsampledFrame *upframe[2];
  SchroMotionField *mf;
  int xmin, ymin, xmax, ymax;

  xmin = ymin = -orig_frame->extension;
  xmax = (orig_frame->width << mvprec) + orig_frame->extension;
  ymax = (orig_frame->height << mvprec) + orig_frame->extension;

  comp_w[0] = xblen;
  comp_h[0] = yblen;
  comp_w[1] = comp_w[2] = xblen
      >> SCHRO_CHROMA_FORMAT_H_SHIFT (params->video_format->chroma_format);
  comp_h[1] = comp_h[2] = yblen
      >> SCHRO_CHROMA_FORMAT_V_SHIFT (params->video_format->chroma_format);


  for (jj = 0; 4 > jj; ++jj) {
    for (ii = 0; 4 > ii; ++ii) {
      double score, min_score = HUGE_VAL;
      int entropy[2], error;
      int width[3], height[3];
      int dx[2], dy[2];
      int k;
      int best_entropy = INT_MAX, best_error = INT_MAX;
      SchroMotionVector *mv, *mv_ref[2], best_mv = { 0 };
      best_mv.split = 2;
      best_mv.pred_mode = 1;

      mv_ref[0] = mv_ref[1] = NULL;
      memset (entropy, 0, sizeof (entropy));
      memset (width, 0, sizeof (width));
      memset (height, 0, sizeof (height));
      mv = motion->motion_vectors + (j + jj) * xnum_blocks + i + ii;
      /* check that the block lies whitin the frame */
      if (!(orig_frame->width > (i + ii) * xblen)
          || !(orig_frame->height > (j + jj) * yblen)) {
        /* Note: blocks outside frame are encoded pred_mode 1, zero MVs */
        int pred_x, pred_y;
        *mv = best_mv;
        mv->pred_mode =
            schro_motion_get_mode_prediction (motion, i + ii, j + jj);
        if (mv->pred_mode != 1 && mv->pred_mode != 2)
          mv->pred_mode = 1;
        schro_motion_vector_prediction (motion, i + ii, j + jj, &pred_x,
            &pred_y, 1);
        mv->u.vec.dx[mv->pred_mode - 1] = pred_x;
        mv->u.vec.dy[mv->pred_mode - 1] = pred_y;
        block->mv[jj][ii] = best_mv;
        total_entropy += 2;
        continue;
      }
      mv->metric = INT_MAX;

      /* do the 2 references, if available */
      for (ref = 0; params->num_refs > ref; ++ref) {
        mf = schro_me_split2_mf (me, ref);
        SCHRO_ASSERT (mf);
        mv_ref[ref] = mf->motion_vectors + (j + jj) * xnum_blocks + i + ii;
        *mv = *mv_ref[ref];
        mv->split = 2;
        mv->pred_mode = ref + 1;
        mv->using_global = 0;
        entropy[ref] =
            schro_motion_block_estimate_entropy (motion, i + ii, j + jj);
        schro_get_split2_metric (me, ref, i + ii, j + jj, mv, &error, fd);
        score = entropy[ref] + error * lambda;
        if (min_score > score) {
          min_score = score;
          best_mv = *mv;
          best_entropy = entropy[ref];
          best_error = mv->metric;
        }
      }
      /* do biref, if available */
      if (1 < params->num_refs) {
        /* Note: I need to calculate the cost of biref prediction */
        biref = TRUE;           /* flag - either or both MVs could be outside frame+ext */
        mv->u.vec.dx[0] = mv_ref[0]->u.vec.dx[0];
        mv->u.vec.dy[0] = mv_ref[0]->u.vec.dy[0];
        mv->u.vec.dx[1] = mv_ref[1]->u.vec.dx[1];
        mv->u.vec.dy[1] = mv_ref[1]->u.vec.dy[1];
        mv->pred_mode = 3;
        mv->using_global = 0;
        for (k = 0; 3 > k; ++k) {
          int tmp_x;
          int tmp_y;

          /* fetch dource data for all components */
          schro_frame_get_data (orig_frame, &orig[k], k, (i + ii) * comp_w[k],
              (j + jj) * comp_h[k]);
          width[k] = MIN (comp_w[k], orig[k].width);
          height[k] = MIN (comp_h[k], orig[k].height);
          tmp_x = (i + ii) * (comp_w[k] << mvprec);
          tmp_y = (j + jj) * (comp_h[k] << mvprec);
          for (ref = 0; params->num_refs > ref; ++ref) {
            dx[ref] = mv->u.vec.dx[ref];
            dx[ref] >>= 0 == k ? 0
                : SCHRO_CHROMA_FORMAT_H_SHIFT (params->
                video_format->chroma_format);
            dx[ref] += tmp_x;
            dy[ref] = mv->u.vec.dy[ref];
            dy[ref] >>= 0 == k ? 0
                : SCHRO_CHROMA_FORMAT_V_SHIFT (params->
                video_format->chroma_format);
            dy[ref] += tmp_y;
            if (0 == k && biref && (xmin > dx[ref] || ymin > dy[ref]
                    || !(xmax > dx[ref] + width[k] - 1)
                    || !(ymax > dy[ref] + height[k] - 1))) {
              biref = FALSE;
              break;
            }
            /* I need to save the original value of fd width and height */
            fd_width = fd[ref].width;
            fd_height = fd[ref].height;
            fd[ref].width = width[k];
            fd[ref].height = height[k];
            upframe[ref] = schro_me_ref (me, ref);
            schro_upsampled_frame_get_block_fast_precN (upframe[ref], k, dx[ref]
                , dy[ref], mvprec, &ref_data[k][ref], &fd[ref]);
            fd[ref].width = fd_width;
            fd[ref].height = fd_height;
          }
        }
        if (biref) {
          mv->metric =
              schro_metric_get_biref (&orig[0], &ref_data[0][0], 1,
              &ref_data[0][1], 1, 1, width[0], height[0]);
          mv->chroma_metric = 0;
          for (k = 1; 3 > k; ++k) {
            mv->chroma_metric +=
                schro_metric_get_biref (&orig[k], &ref_data[k][0], 1,
                &ref_data[k][1], 1, 1, width[k], height[k]);
          }
          score =
              entropy[0] + entropy[1] + (mv->metric +
              mv->chroma_metric) * lambda;
          if (min_score > score) {
            best_error = mv->metric + mv->chroma_metric;
            best_entropy = entropy[0] + entropy[1];
            best_mv = *mv;
            min_score = score;
          }
        }
      }

      /* FIXME: magic used for considering DC prediction */
      if (4 * (width[0] * height[0] + 2 * width[1] * height[1]) < best_error) {
        /* let's consider DC prediction */
        SchroMotionVector *mvdc;
        int k;
        int ok = TRUE;

        mvdc = (SchroMotionVector *) mv;
        mvdc->pred_mode = 0;
        mvdc->split = 2;
        mvdc->using_global = 0;
        error = 0;
        for (k = 0; 3 > k; ++k) {
          int tmp = schro_block_average (&mvdc->u.dc.dc[k],
              orig_frame->components + k, (i + ii) * comp_w[k],
              (j + jj) * comp_h[k], comp_w[k], comp_h[k]);
          if (SCHRO_METRIC_INVALID_2 == tmp) {
            SCHRO_DEBUG ("Invalid DC metric");
            mvdc->metric = INT_MAX;
            ok = FALSE;
          } else {
            error += tmp;
          }
        }
        if (ok) {
          mvdc->metric = error;
          /* FIXME: we're assuming that the block doesn't have any predictor */
          entropy[0] = schro_pack_estimate_sint (mvdc->u.dc.dc[0]);
          entropy[0] += schro_pack_estimate_sint (mvdc->u.dc.dc[1]);
          entropy[0] += schro_pack_estimate_sint (mvdc->u.dc.dc[2]);
          if (error < best_error) {
            best_mv = *(SchroMotionVector *) mvdc;
            best_error = mvdc->metric;
            best_entropy = entropy[0];
          }
        }
      }
      *mv = best_mv;
      total_error += best_error;
      total_entropy += best_entropy;
      block->mv[jj][ii] = best_mv;
    }
  }
  block->valid = TRUE;
  block->error = total_error;
  block->entropy = total_entropy;
  block->score = total_entropy + lambda * total_error;
}

static void
set_split1_motion (SchroMotion * motion, int i, int j)
{
  SchroParams *params = motion->params;
  int xnum_blocks = params->x_num_blocks;
  SchroMotionVector *mv = motion->motion_vectors + j * xnum_blocks + i;

  *(mv + 1) = *mv;
  *(mv + xnum_blocks) = *mv;
  *(mv + xnum_blocks + 1) = *mv;
}

static int
mv_already_in_list (SchroMotionVector * hint_list[], int len,
    SchroMotionVector * candidate_mv, int ref, int shift)
{
  int i;

  for (i = 0; len > i; ++i) {
    if ((candidate_mv->u.vec.dx[ref] << shift) == hint_list[i]->u.vec.dx[ref]
        && (candidate_mv->u.vec.dy[ref] << shift) ==
        hint_list[i]->u.vec.dy[ref]) {
      return TRUE;
    }
  }
  return FALSE;
}

/* select a MV for a split1 block based on 5 seeds
   the four split 2 MVs and the MV from rough ME, level 1.
   The selection of the MV is based on actual cost, not just SAD */
static void
schro_get_best_mv_split1 (SchroMe * me, int ref, int i, int j,
    SchroMotionVector * mv_ref, int *error, int *entropy, SchroFrameData * fd)
{
  SchroMotionVector *hint_mv[6], *mv_motion;
  SchroMotionVector *mv, temp_mv = { 0 };
  SchroMotionField *mf;
  SchroParams *params = schro_me_params (me);
  SchroFrameData orig[3], ref_data[3];
  SchroFrame *frame = schro_me_src (me);
  SchroUpsampledFrame *upframe;
  SchroMotion *motion = schro_me_motion (me);
  int n = 0, ii, jj, m = 0, min_m, metric, ent;
  int mv_prec = params->mv_precision;
  int fd_width, fd_height;
  double score, min_score, lambda = schro_me_lambda (me);
  int xblen, yblen, width[3], height[3], dx, dy;
  int comp_w[3], comp_h[3];
  int k;
  int xmin, xmax, ymin, ymax, tmp_x, tmp_y;
  int best_error = INT_MAX, best_entropy = INT_MAX;
  int best_chroma_error = INT_MAX;
  int block_x[3], block_y[3];
  SchroHierBm *hbm;

  xblen = params->xbsep_luma << 1;
  yblen = params->ybsep_luma << 1;
  xmin = ymin = -frame->extension;
  xmax = (frame->width << mv_prec) + frame->extension;
  ymax = (frame->height << mv_prec) + frame->extension;
  /* calculate split1 block sizes for all components */
  comp_w[0] = xblen;
  comp_h[0] = yblen;
  comp_w[1] = comp_w[2] = xblen
      >> SCHRO_CHROMA_FORMAT_H_SHIFT (params->video_format->chroma_format);
  comp_h[1] = comp_h[2] = yblen
      >> SCHRO_CHROMA_FORMAT_V_SHIFT (params->video_format->chroma_format);
  /* calculates split2 block sizes for all components */
  block_x[0] = params->xbsep_luma;
  block_y[0] = params->ybsep_luma;
  block_x[1] = block_x[2] =
      block_x[0] >> SCHRO_CHROMA_FORMAT_H_SHIFT (params->
      video_format->chroma_format);
  block_y[1] = block_y[2] =
      block_y[0] >> SCHRO_CHROMA_FORMAT_V_SHIFT (params->
      video_format->chroma_format);

  SCHRO_ASSERT (motion);
  mv_motion = motion->motion_vectors + j * params->x_num_blocks + i;

  for (k = 0; 3 > k; ++k) {
    /* get source data if possible */
    if (!schro_frame_get_data (frame, &orig[k], k, i * block_x[k],
            j * block_y[k])) {
      *mv_ref = temp_mv;
      mv_ref->metric = INT_MAX;
      mv_ref->pred_mode = ref + 1;
      *error = INT_MAX;
      *mv_motion = *mv_ref;
      *entropy = schro_motion_block_estimate_entropy (motion, i, j);
      return;
    }
    width[k] = MIN (comp_w[k], orig[k].width);
    height[k] = MIN (comp_h[k], orig[k].height);
  }
  /* inherit from split 2 level MV */
  mf = schro_me_subpel_mf (me, ref);
  SCHRO_ASSERT (mf);
  for (jj = 0; 2 > jj; ++jj) {
    for (ii = 0; 2 > ii; ++ii) {
      mv = mf->motion_vectors + (j + jj) * mf->x_num_blocks + i + ii;
      if (SCHRO_METRIC_INVALID != mv->metric) {
        if ((0 < n && !mv_already_in_list (hint_mv, n, mv, ref, mv_prec))
            || 0 == n) {
          hint_mv[n++] = mv;
        }
      }
    }
  }
  /* inherit from stage 1 of hier. BM */
  hbm = schro_me_hbm (me, ref);
  SCHRO_ASSERT (hbm);
  mf = schro_hbm_motion_field (hbm, 1);
  SCHRO_ASSERT (mf);
  mv = mf->motion_vectors + j * mf->x_num_blocks + i;
  if (INT_MAX != mv->metric) {
    if ((0 < n && !mv_already_in_list (hint_mv, n, mv, ref, mv_prec)) || 0 == n) {
      temp_mv = *mv;
      temp_mv.u.vec.dx[ref] <<= mv_prec;
      temp_mv.u.vec.dy[ref] <<= mv_prec;
      hint_mv[n++] = &temp_mv;
    }
  }

  /* now pick the best candidate */
  min_m = -1;
  min_score = HUGE_VAL;
  upframe = schro_me_ref (me, ref);
  for (m = 0; n > m; ++m) {
    int chroma_metric = 0;
    int ok = TRUE;

    metric = 0;
    for (k = 0; 3 > k; ++k) {
      tmp_x = (i * block_x[k]) << mv_prec;
      tmp_y = (j * block_y[k]) << mv_prec;
      dx = hint_mv[m]->u.vec.dx[ref];
      dx >>= 0 == k ? 0
          : SCHRO_CHROMA_FORMAT_H_SHIFT (params->video_format->chroma_format);
      dx += tmp_x;
      dy = hint_mv[m]->u.vec.dy[ref];
      dy >>= 0 == k ? 0
          : SCHRO_CHROMA_FORMAT_V_SHIFT (params->video_format->chroma_format);
      dy += tmp_y;
      if (0 == k && (xmin > dx || ymin > dy
              || !(xmax > dx + width[k] - 1) || !(ymax > dy + height[k] - 1))) {
        k = 3;
        ok = FALSE;
        continue;
      }
      /* I need to save the original value of fd width and height */
      fd_width = fd->width;
      fd_height = fd->height;
      fd->width = width[k];
      fd->height = height[k];

      schro_upsampled_frame_get_block_fast_precN (upframe, k, dx, dy, mv_prec,
          &ref_data[k], fd);
      fd->width = fd_width;
      fd->height = fd_height;
      if (0 == k) {
        metric =
            schro_metric_absdiff_u8 (orig[0].data, orig[0].stride,
            ref_data[0].data, ref_data[0].stride, width[0], height[0]);
      } else {
        chroma_metric +=
            schro_metric_absdiff_u8 (orig[k].data, orig[k].stride,
            ref_data[k].data, ref_data[k].stride, width[k], height[k]);
      }
    }
    if (ok) {
      *mv_motion = *hint_mv[m];
      mv_motion->split = 1;
      mv_motion->pred_mode = ref + 1;
      ent = schro_motion_block_estimate_entropy (motion, i, j);
      score = ent + (metric + chroma_metric) * lambda;
      if (min_score > score) {
        min_score = score;
        min_m = m;
        best_entropy = ent;
        best_error = metric;
        best_chroma_error = chroma_metric;
      }
    }
  }
  if (-1 < min_m) {
    *error = best_error + best_chroma_error;
    *entropy = best_entropy;
    *mv_ref = *hint_mv[min_m];
    mv_ref->metric = best_error >> 2;
    mv_ref->chroma_metric = best_chroma_error >> 2;
    mv_ref->split = 1;
    mv_ref->pred_mode = ref + 1;
  } else
    mv_ref->metric = INT_MAX;
}

/* performs mode decision for a superblock, split level 1 */
static void
schro_do_split1 (SchroMe * me, int i, int j, SchroBlock * block,
    SchroFrameData * fd)
{
  SchroParams *params = schro_me_params (me);
  SchroMotion *motion = schro_me_motion (me);
  SchroFrame *orig_frame = schro_me_src (me);
  SchroFrameData ref_data[3][2], orig[3];
  SchroUpsampledFrame *upframe[2];
  SchroMotionField *mf;
  double lambda = schro_me_lambda (me);
  int ii, jj, ref, biref, k;
  int total_entropy = 0, total_error = 0;
  int mvprec = params->mv_precision;
  int xblen = params->xbsep_luma * 2, yblen = params->ybsep_luma * 2;
  int width[3], height[3], dx[2], dy[2], tmp_x, tmp_y;
  int fd_width, fd_height;
  int xmin, xmax, ymin, ymax;
  int comp_h[3], comp_w[3];
  int block_x[3], block_y[3];

  xmin = ymin = -orig_frame->extension;
  xmax = (orig_frame->width << mvprec) + orig_frame->extension;
  ymax = (orig_frame->height << mvprec) + orig_frame->extension;

  comp_w[0] = xblen;
  comp_h[0] = yblen;
  comp_w[1] = comp_w[2] = xblen
      >> SCHRO_CHROMA_FORMAT_H_SHIFT (params->video_format->chroma_format);
  comp_h[1] = comp_h[2] = yblen
      >> SCHRO_CHROMA_FORMAT_V_SHIFT (params->video_format->chroma_format);
  block_x[0] = params->xbsep_luma;
  block_y[0] = params->ybsep_luma;
  block_x[1] = block_x[2] =
      block_x[0] >> SCHRO_CHROMA_FORMAT_H_SHIFT (params->
      video_format->chroma_format);
  block_y[1] = block_y[2] =
      block_y[0] >> SCHRO_CHROMA_FORMAT_V_SHIFT (params->
      video_format->chroma_format);

  block->valid = TRUE;

  for (jj = 0; 4 > jj; jj += 2) {
    for (ii = 0; 4 > ii; ii += 2) {
      double score, min_score = HUGE_VAL;
      int entropy[2], error = INT_MAX;
      int best_entropy = INT_MAX, best_chroma_error = INT_MAX, best_error =
          INT_MAX;
      /* Note that the metric for the split 1 block will be stored in best_mv
       * but divided by 4 */
      SchroMotionVector *mv, best_mv = { 0 }, mv_ref[2], *mv_split1;
      best_mv.split = 1;
      best_mv.pred_mode = 1;

      mv = motion->motion_vectors + (j + jj) * params->x_num_blocks + i + ii;
      /* check that the block lies within the frame */
      if (!(orig_frame->width > (i + ii) * params->xbsep_luma)
          || !(orig_frame->height > (j + jj) * params->ybsep_luma)) {
        /* the block lies outside the frame, set both block and motion
         * to a zero mv, split 1, forward predicted */
        int pred_x, pred_y;
        *mv = best_mv;
        mv->pred_mode =
            schro_motion_get_mode_prediction (motion, i + ii, j + jj);
        if (mv->pred_mode != 1 && mv->pred_mode != 2)
          mv->pred_mode = 1;
        schro_motion_vector_prediction (motion, i + ii, j + jj, &pred_x,
            &pred_y, 1);
        mv->u.vec.dx[mv->pred_mode - 1] = pred_x;
        mv->u.vec.dy[mv->pred_mode - 1] = pred_y;
        block->mv[jj][ii] = *mv;
        set_split1_motion (motion, i + ii, j + jj);
        total_entropy += 2;
        mf = schro_me_split1_mf (me, 0);
        SCHRO_ASSERT (mf);
        mv_split1 =
            mf->motion_vectors + (j + jj) * params->x_num_blocks + i + ii;
        *mv_split1 = best_mv;
        if (1 < params->num_refs) {
          mf = schro_me_split1_mf (me, 1);
          SCHRO_ASSERT (mf);
          mv_split1 =
              mf->motion_vectors + (j + jj) * params->x_num_blocks + i + ii;
          *mv_split1 = best_mv;
          mv_split1->pred_mode = 2;
        }
        continue;
      }
      mv->metric = INT_MAX;
      mv->chroma_metric = INT_MAX;
      best_mv.metric = INT_MAX;
      best_mv.chroma_metric = INT_MAX;

      /* do the 2 references, if available */
      for (ref = 0; params->num_refs > ref; ++ref) {
        mf = schro_me_split1_mf (me, ref);
        SCHRO_ASSERT (mf);
        mv_split1 =
            mf->motion_vectors + (j + jj) * params->x_num_blocks + i + ii;
        schro_get_best_mv_split1 (me, ref, i + ii, j + jj, &mv_ref[ref]
            , &error, &entropy[ref], fd);
        *mv_split1 = mv_ref[ref];
        if (INT_MAX != mv_ref[ref].metric) {
          score = entropy[ref] + lambda * error;
          if (min_score > score) {
            min_score = score;
            best_mv = mv_ref[ref];
            best_entropy = entropy[ref];
            best_error = error;
          }
        }
      }
      /* do biref if available */
      if (1 < params->num_refs
          && SCHRO_METRIC_INVALID != mv_ref[0].metric
          && SCHRO_METRIC_INVALID != mv_ref[1].metric) {
        /* I'm going to use the two best MVs from previous steps
         * Note: I need to calculate the cost and entropy of biref */
        mv_ref[0].u.vec.dx[1] = mv_ref[1].u.vec.dx[1];
        mv_ref[0].u.vec.dy[1] = mv_ref[1].u.vec.dy[1];
        mv_ref[0].pred_mode = 3;
        biref = TRUE;
        for (k = 0; 3 > k; ++k) {
          schro_frame_get_data (orig_frame, &orig[k], k, (i + ii) * block_x[k],
              (j + jj) * block_y[k]);
          width[k] = MIN (comp_w[k], orig[k].width);
          height[k] = MIN (comp_h[k], orig[k].height);
          tmp_x = (i + ii) * (block_x[k] << mvprec);
          tmp_y = (j + jj) * (block_y[k] << mvprec);
          for (ref = 0; params->num_refs > ref; ++ref) {
            dx[ref] = mv_ref[0].u.vec.dx[ref];
            dx[ref] >>= 0 == k ? 0
                : SCHRO_CHROMA_FORMAT_H_SHIFT (params->
                video_format->chroma_format);
            dx[ref] += tmp_x;
            dy[ref] = mv_ref[0].u.vec.dy[ref];
            dy[ref] >>= 0 == k ? 0
                : SCHRO_CHROMA_FORMAT_V_SHIFT (params->
                video_format->chroma_format);
            dy[ref] += tmp_y;
            /* check whether we can extract reference blocks */
            if (0 == k && biref && (xmin > dx[ref] || ymin > dy[ref]
                    || !(xmax > dx[ref] + width[k] - 1)
                    || !(ymax > dy[ref] + height[k] - 1))) {
              biref = FALSE;
              k = 3;
              break;
            }
            /* I need to save the original value of fd width and height */
            fd_width = fd[ref].width;
            fd_height = fd[ref].height;
            fd[ref].width = width[k];
            fd[ref].height = height[k];
            upframe[ref] = schro_me_ref (me, ref);
            schro_upsampled_frame_get_block_fast_precN (upframe[ref], k,
                dx[ref], dy[ref], mvprec, &ref_data[k][ref], &fd[ref]);
            fd[ref].width = fd_width;
            fd[ref].height = fd_height;
          }
        }
        if (biref) {
          int chroma_error = 0;

          error = 0;
          error =
              schro_metric_get_biref (&orig[0], &ref_data[0][0], 1,
              &ref_data[0][1], 1, 1, width[0], height[0]);
          for (k = 1; 3 > k; ++k) {
            chroma_error +=
                schro_metric_get_biref (&orig[k], &ref_data[k][0], 1,
                &ref_data[k][1], 1, 1, width[k], height[k]);
          }
          score = entropy[0] + entropy[1] + lambda * (error + chroma_error);
          mv_ref[0].metric = error >> 2;
          mv_ref[0].chroma_metric = chroma_error >> 2;
          if (min_score > score) {
            best_error = error;
            best_chroma_error = chroma_error;
            best_entropy = entropy[0] + entropy[1];
            best_mv = mv_ref[0];
            min_score = score;
          }
        }
      }
      if (SCHRO_METRIC_INVALID == best_mv.metric) {
        block->valid = FALSE;
      } else {
        *mv = best_mv;
        total_error += best_error + best_chroma_error;
        total_entropy += best_entropy;
        block->mv[jj][ii] = best_mv;
        set_split1_motion (motion, i + ii, j + jj);
      }
    }
  }
  block->error = total_error;
  block->entropy = total_entropy;
  block->score = total_entropy + lambda * total_error;
}

static void
schro_get_best_split0_mv (SchroMe * me, int ref, int i, int j,
    SchroMotionVector * mv_ref, int *error, int *entropy, SchroFrameData * fd)
{
  SchroMotionVector *hint_mv[6], *mv_motion;
  SchroMotionVector *mv, temp_mv = { 0 };
  SchroMotionField *mf;
  SchroParams *params = schro_me_params (me);
  SchroFrameData orig[3], ref_data[3];
  SchroFrame *frame = schro_me_src (me);
  SchroUpsampledFrame *upframe;
  SchroHierBm *hbm;
  SchroMotion *motion = schro_me_motion (me);
  int n = 0, m = 0, min_m = -1, metric = 0, ent;
  int mv_prec = params->mv_precision;
  int fd_width, fd_height, jj, ii;
  int k;
  double score, min_score = HUGE_VAL, lambda = schro_me_lambda (me);
  int width[3], height[3], dx, dy, xmin, xmax, ymin, ymax, tmp_x, tmp_y;
  int best_error = INT_MAX, best_chroma_error = INT_MAX, best_entropy = INT_MAX;
  int comp_w[3], comp_h[3];
  /* split2 block sizes for all components */
  int block_x[3], block_y[3];

  block_x[0] = params->xbsep_luma;
  block_y[0] = params->ybsep_luma;
  block_x[1] = block_x[2] =
      block_x[0] >> SCHRO_CHROMA_FORMAT_H_SHIFT (params->
      video_format->chroma_format);
  block_y[1] = block_y[2] =
      block_y[0] >> SCHRO_CHROMA_FORMAT_V_SHIFT (params->
      video_format->chroma_format);
  /* split0 block sizes for all components */
  comp_w[0] = block_x[0] << 2;
  comp_h[0] = block_y[0] << 2;
  comp_w[1] = comp_w[2] =
      comp_w[0] >> SCHRO_CHROMA_FORMAT_H_SHIFT (params->
      video_format->chroma_format);
  comp_h[1] = comp_h[2] =
      comp_h[0] >> SCHRO_CHROMA_FORMAT_V_SHIFT (params->
      video_format->chroma_format);

  xmin = -frame->extension;
  ymin = -frame->extension;
  xmax = (frame->width << mv_prec) + frame->extension;
  ymax = (frame->height << mv_prec) + frame->extension;

  /* get source data if possible */
  for (k = 0; 3 > k; ++k) {
    if (!schro_frame_get_data (frame, &orig[k], k, i * block_x[k],
            j * block_y[k])) {
      /* this should never happen */
      SCHRO_ASSERT (0);
    }
    width[k] = MIN (comp_w[k], orig[k].width);
    height[k] = MIN (comp_h[k], orig[k].height);
  }
  mv_motion = motion->motion_vectors + j * params->x_num_blocks + i;

  /* inherit from split 1 level MV */
  mf = schro_me_split1_mf (me, ref);
  SCHRO_ASSERT (mf);
  for (jj = 0; 4 > jj; jj += 2) {
    for (ii = 0; 4 > ii; ii += 2) {
      mv = mf->motion_vectors + (j + jj) * mf->x_num_blocks + i + ii;
      if (INT_MAX != mv->metric) {
        if ((0 < n && !mv_already_in_list (hint_mv, n, mv, ref, 0)) || 0 == n) {
          hint_mv[n++] = mv;
        }
      }
    }
  }
  /* inherit from stage 2 of hier. BM */
  hbm = schro_me_hbm (me, ref);
  SCHRO_ASSERT (hbm);
  mf = schro_hbm_motion_field (hbm, 2);
  mv = mf->motion_vectors + j * mf->x_num_blocks + i;
  if (INT_MAX != mv->metric) {
    if ((0 < n && !mv_already_in_list (hint_mv, n, mv, ref, mv_prec)) || 0 == n) {
      temp_mv = *mv;
      temp_mv.u.vec.dx[ref] <<= mv_prec;
      temp_mv.u.vec.dy[ref] <<= mv_prec;
      hint_mv[n++] = &temp_mv;
    }
  }
  /* now pick the best candidate */
  tmp_x = (i * params->xbsep_luma) << mv_prec;
  tmp_y = (j * params->ybsep_luma) << mv_prec;
  upframe = schro_me_ref (me, ref);
  for (m = 0; n > m; ++m) {
    int ok = TRUE;
    int chroma_metric = 0;
    for (k = 0; 3 > k; ++k) {
      tmp_x = (i * block_x[k]) << mv_prec;
      tmp_y = (j * block_y[k]) << mv_prec;
      dx = hint_mv[m]->u.vec.dx[ref];
      dx >>= 0 == k ? 0
          : SCHRO_CHROMA_FORMAT_H_SHIFT (params->video_format->chroma_format);
      dx += tmp_x;
      dy = hint_mv[m]->u.vec.dy[ref];
      dy >>= 0 == k ? 0
          : SCHRO_CHROMA_FORMAT_V_SHIFT (params->video_format->chroma_format);
      dy += tmp_y;
      if (0 == k && (xmin > dx || ymin > dy
              || !(xmax > dx + width[k] - 1) || !(ymax > dy + height[k] - 1))) {
        ok = FALSE;
        k = 3;
        continue;
      }
      if (ok) {
        /* I need to save the original values of fd width and height */
        fd_width = fd->width;
        fd_height = fd->height;
        fd->width = width[k];
        fd->height = height[k];
        schro_upsampled_frame_get_block_fast_precN (upframe, k, dx, dy, mv_prec,
            &ref_data[k], fd);
        fd->width = fd_width;
        fd->height = fd_height;
        if (0 == k) {
          metric =
              schro_metric_absdiff_u8 (orig[0].data, orig[0].stride,
              ref_data[0].data, ref_data[0].stride, width[0], height[0]);
        } else {
          chroma_metric +=
              schro_metric_absdiff_u8 (orig[k].data, orig[k].stride,
              ref_data[k].data, ref_data[k].stride, width[k], height[k]);
        }
      }
    }
    if (ok) {
      *mv_motion = *hint_mv[m];
      mv_motion->split = 0;
      mv_motion->pred_mode = ref + 1;
      ent = schro_motion_block_estimate_entropy (motion, i, j);
      score = ent + lambda * (metric + chroma_metric);
      if (min_score > score) {
        min_score = score;
        min_m = m;
        best_entropy = ent;
        best_error = metric;
        best_chroma_error = chroma_metric;
      }
    }
  }
  if (-1 < min_m) {
    *error = best_error + best_chroma_error;
    *entropy = best_entropy;
    *mv_ref = *hint_mv[min_m];
    mv_ref->metric = best_error >> 4;
    mv_ref->chroma_metric = best_chroma_error >> 4;
    mv_ref->split = 0;
    mv_ref->pred_mode = ref + 1;
  } else
    mv_ref->metric = INT_MAX;
}

static void
schro_do_split0_biref (SchroMe * me, int i, int j, SchroBlock * block,
    SchroFrameData * fd)
{
  SchroParams *params = schro_me_params (me);
  int entropy = 0;
  SchroMotion *motion = schro_me_motion (me);
  SchroMotionVector *mv_motion;
  SchroFrame *orig_frame = schro_me_src (me);
  SchroFrameData ref_data[3][2], orig[3];
  SchroUpsampledFrame *upframe[2];
  double score, lambda = schro_me_lambda (me);
  int mv_prec = params->mv_precision;
  int width[3], height[3], dx[2], dy[2], tmp_x, tmp_y;
  int ref, error = INT_MAX;
  int xmin = -orig_frame->extension;
  int ymin = -orig_frame->extension;
  int xmax = (orig_frame->width << mv_prec) + orig_frame->extension;
  int ymax = (orig_frame->height << mv_prec) + orig_frame->extension;
  int biref, fd_width, fd_height;
  int k;
  /* split2 block sizes for all components */
  int block_x[3], block_y[3];
  int comp_w[3], comp_h[3];
  SchroMotionVector *mv;


  block->valid = FALSE;

  mv_motion = motion->motion_vectors + j * params->x_num_blocks + i;

  mv = &block->mv[0][0];
  mv->split = 0;
  mv->pred_mode = 3;
  mv->using_global = 0;

  *mv_motion = *mv;
  entropy = schro_motion_block_estimate_entropy (motion, i, j);

  block_x[0] = params->xbsep_luma;
  block_y[0] = params->ybsep_luma;
  block_x[1] = block_x[2] =
      block_x[0] >> SCHRO_CHROMA_FORMAT_H_SHIFT (params->
      video_format->chroma_format);
  block_y[1] = block_y[2] =
      block_y[0] >> SCHRO_CHROMA_FORMAT_V_SHIFT (params->
      video_format->chroma_format);
  /* split0 block sizes for all components */
  comp_w[0] = block_x[0] << 2;
  comp_h[0] = block_y[0] << 2;
  comp_w[1] = comp_w[2] =
      comp_w[0] >> SCHRO_CHROMA_FORMAT_H_SHIFT (params->
      video_format->chroma_format);
  comp_h[1] = comp_h[2] =
      comp_h[0] >> SCHRO_CHROMA_FORMAT_V_SHIFT (params->
      video_format->chroma_format);

  /* do biref, if available */
  biref = TRUE;
  /* fetch source data */
  for (k = 0; 3 > k; ++k) {
    if (schro_frame_get_data (orig_frame, &orig[k], k, i * block_x[k],
            j * block_y[k])) {
      width[k] = MIN (orig[k].width, comp_w[k]);
      height[k] = MIN (orig[k].height, comp_h[k]);
    } else {
      biref = FALSE;
      break;
    }
  }
  if (biref) {
    /* fetch reference data, if possible */
    for (ref = 0; params->num_refs > ref; ++ref) {
      for (k = 0; 3 > k && biref; ++k) {
        tmp_x = (i * block_x[k]) << mv_prec;
        tmp_y = (j * block_y[k]) << mv_prec;
        dx[ref] = mv->u.vec.dx[ref];
        dx[ref] >>= 0 == k ? 0
            : SCHRO_CHROMA_FORMAT_H_SHIFT (params->video_format->chroma_format);
        dx[ref] += tmp_x;
        dy[ref] = mv->u.vec.dy[ref];
        dy[ref] >>= 0 == k ? 0
            : SCHRO_CHROMA_FORMAT_V_SHIFT (params->video_format->chroma_format);
        dy[ref] += tmp_y;
        /* check whether we can extract reference blocks */
        if (0 == k && (xmin > dx[ref] || ymin > dy[ref]
                || !(xmax > dx[ref] + width[k] - 1)
                || !(ymax > dy[ref] + height[k] - 1))) {
          biref = FALSE;
          break;
        } else {
          fd_width = fd[ref].width;
          fd_height = fd[ref].height;
          fd[ref].width = width[k];
          fd[ref].height = height[k];
          upframe[ref] = schro_me_ref (me, ref);
          schro_upsampled_frame_get_block_fast_precN (upframe[ref], k, dx[ref],
              dy[ref], mv_prec, &ref_data[k][ref], &fd[ref]);
          fd[ref].width = fd_width;
          fd[ref].height = fd_height;
        }
      }
    }
    if (biref) {
      int chroma_error = 0;
      error =
          schro_metric_get_biref (&orig[0], &ref_data[0][0], 1, &ref_data[0][1],
          1, 1, width[0], height[0]);
      for (k = 1; 3 > k; ++k) {
        chroma_error +=
            schro_metric_get_biref (&orig[k], &ref_data[k][0], 1,
            &ref_data[k][1], 1, 1, width[k], height[k]);
      }
      score = entropy + lambda * (error + chroma_error);
      mv->metric = error >> 4;
      mv->chroma_metric = chroma_error >> 4;
      block->entropy = entropy;
      block->error = error + chroma_error;
      block->score = block->entropy + lambda * block->error;
      block->valid = TRUE;
    }
  }
}

static void
schro_do_split0_biref_zero (SchroMe * me, int i, int j, SchroBlock * block,
    SchroFrameData * fd)
{
  SchroParams *params = schro_me_params (me);
  SchroMotionVector *mv;

  SCHRO_ASSERT (1 < params->num_refs);

  mv = &block->mv[0][0];
  mv->split = 0;
  mv->pred_mode = 3;
  mv->using_global = 0;
  mv->u.vec.dx[0] = 0;
  mv->u.vec.dy[0] = 0;
  mv->u.vec.dx[1] = 0;
  mv->u.vec.dy[1] = 0;

  schro_do_split0_biref (me, i, j, block, fd);
}

/* performs mode decision for a superblock, split level 0 */
static void
schro_do_split0 (SchroMe * me, int i, int j, SchroBlock * block,
    SchroFrameData * fd)
{
  SchroParams *params = schro_me_params (me);
  SchroMotionField *mf;
  SchroMotionVector mv_ref[2], best_mv = { 0 }, *mv_split0;
  double lambda = schro_me_lambda (me);
  int ref;
  int error = INT_MAX;
  int entropy[2];
  int best_error = INT_MAX;
  int best_entropy = INT_MAX;
  double score, min_score = HUGE_VAL;

  best_mv.metric = INT_MAX;

  block->valid = FALSE;

  /* do the 2 references, if available */
  for (ref = 0; params->num_refs > ref; ++ref) {
    mf = schro_me_split0_mf (me, ref);
    SCHRO_ASSERT (mf);
    mv_split0 = mf->motion_vectors + j * params->x_num_blocks + i;
    schro_get_best_split0_mv (me, ref, i, j, &mv_ref[ref], &error,
        &entropy[ref], fd);
    *mv_split0 = mv_ref[ref];
    if (INT_MAX != mv_ref[ref].metric) {
      score = entropy[ref] + lambda * error;
      if (min_score > score) {
        min_score = score;
        best_mv = mv_ref[ref];
        best_entropy = entropy[ref];
        best_error = error;
      }
    }
  }
  /* do biref, if available */
  if (1 < params->num_refs
      && INT_MAX != mv_ref[0].metric && INT_MAX != mv_ref[1].metric) {
    SchroBlock biref_block = { 0 };

    SchroMotionVector *mv;
    mv = &biref_block.mv[0][0];
    mv->split = 0;
    mv->pred_mode = 3;
    mv->using_global = 0;
    mv->u.vec.dx[0] = mv_ref[0].u.vec.dx[0];
    mv->u.vec.dy[0] = mv_ref[0].u.vec.dy[0];
    mv->u.vec.dx[1] = mv_ref[1].u.vec.dx[1];
    mv->u.vec.dy[1] = mv_ref[1].u.vec.dy[1];
    mv->metric = INT_MAX;

    schro_do_split0_biref (me, i, j, &biref_block, fd);

    if (biref_block.valid && min_score > biref_block.score) {
      min_score = biref_block.score;
      best_mv = biref_block.mv[0][0];
      best_error = biref_block.error;
      best_entropy = biref_block.entropy;
    }
  }

  if (INT_MAX == best_mv.metric) {
    block->valid = FALSE;
  } else {
    block->valid = TRUE;
    block->error = best_error;
    block->entropy = best_entropy;
    block->score = best_entropy + lambda * best_error;
    block->mv[0][0] = best_mv;
  }
}


/* performs mode decision and block/superblock splitting */
void
schro_mode_decision (SchroMe * me)
{
  SchroParams *params = schro_me_params (me);
  SchroFrameData fd[2];
  SchroMotion *motion = schro_me_motion (me);
  int i, j, ref;
  double total_error = 0.0;
  int block_size;
  int badblocks = 0, dcblocks = 0;
  double min_score;
  int k, l;

  block_size = 16 * params->xbsep_luma * params->ybsep_luma;
  fd[0].data = fd[1].data = NULL;
  if (1 < params->mv_precision) {
    for (ref = 0; params->num_refs > ref; ++ref) {
      fd[ref].data = schro_malloc (block_size * sizeof (uint8_t));
      fd[ref].stride = fd[ref].width = params->xbsep_luma << 2;
      fd[ref].height = params->ybsep_luma << 2;
      fd[ref].length = block_size * sizeof (uint8_t);
      fd[ref].h_shift = fd[ref].v_shift = 0;
      fd[ref].format = SCHRO_FRAME_FORMAT_U8_420;
    }
  }
  /* we're now using chroma info for MS, increase block size accordingly */
  block_size = block_size * 2 / 3;
  /* loop over all superblocks. The indices are set to the value of the
   * top-left block in the SB, just like in Dave's bigblock estimation.
   * We'll start considering split 2 first, then split 1 and finally split 0 */
  for (j = 0; params->y_num_blocks > j; j += 4) {
    for (i = 0; params->x_num_blocks > i; i += 4) {
      SchroBlock block = { 0 }
      , tryblock = {
      0};

      schro_do_split2 (me, i, j, &block, fd);
      min_score = block.score;

      /* note: do_split1 writes to motion */
      schro_do_split1 (me, i, j, &tryblock, fd);
      if (tryblock.valid) {
        if (min_score > tryblock.score) {
          memcpy (&block, &tryblock, sizeof (block));
          schro_block_fixup (&block);
          /* I need to overwrite motion because it's needed to estimate
           * entropy at split 0 */
          schro_motion_copy_to (motion, i, j, &block);
          min_score = block.score;
          /* Note: only do split0 if split1 better than split2 */
          schro_do_split0 (me, i, j, &tryblock, fd);
          if (tryblock.valid) {
            if (min_score > tryblock.score) {
              memcpy (&block, &tryblock, sizeof (block));
              schro_block_fixup (&block);
            }
          }
        }
      }
      if (1 < params->num_refs) {
        schro_do_split0_biref_zero (me, i, j, &tryblock, fd);
        if (tryblock.valid) {
          if (min_score > tryblock.score) {
            memcpy (&block, &tryblock, sizeof (block));
            schro_block_fixup (&block);
          }
        }
      }
      schro_motion_copy_to (motion, i, j, &block);
      if (block.error > 10 * block_size) {
        ++badblocks;
      }

      for (k = 0; 4 > k; ++k) {
        for (l = 0; 4 > l; ++l) {
          if (0 == block.mv[k][l].pred_mode) {
            ++dcblocks;
          }
        }
      }

      total_error += (double) block.error * block.error /
          (double) (block_size * block_size);
    }
  }

  schro_me_set_mc_error (me,
      total_error / (240.0 * 240.0) /
      params->x_num_blocks * params->y_num_blocks / 16);

  schro_me_set_badblock_ratio (me,
      ((double) badblocks) / (params->x_num_blocks * params->y_num_blocks /
          16));

  schro_me_set_dcblock_ratio (me,
      ((double) dcblocks) / (params->x_num_blocks * params->y_num_blocks));

  if (1 < params->mv_precision) {
    for (ref = 0; params->num_refs > ref; ++ref) {
      schro_free (fd[ref].data);
    }
  }
}

struct SchroMeElement
{
  SchroUpsampledFrame *ref;

  SchroMotionField *subpel_mf;
  SchroMotionField *split2_mf;
  SchroMotionField *split1_mf;
  SchroMotionField *split0_mf;

  SchroHierBm *hbm;
};


typedef struct SchroMeElement *SchroMeElement;

/* supports motion estimation */
struct _SchroMe
{
  SchroFrame *src;

  SchroParams *params;
  double lambda;
  SchroMotion *motion;

  double mc_error;
  double badblocks_ratio;
  double dcblock_ratio;

  SchroMeElement meElement[2];
};

static SchroMeElement
schro_me_element_new (SchroEncoderFrame * frame, int ref_number)
{
  SchroMeElement me;

  me = schro_malloc0 (sizeof (struct SchroMeElement));

  SCHRO_ASSERT (frame && (0 == ref_number || 1 == ref_number));
  SCHRO_ASSERT (me);

  me->ref = frame->ref_frame[ref_number]->upsampled_original_frame;
  me->hbm = schro_hbm_ref (frame->hier_bm[ref_number]);
  return me;
}

static void
schro_me_element_free (SchroMeElement * pme)
{
  SchroMeElement me = *pme;

  if (me) {
    if (me->hbm)
      schro_hbm_unref (me->hbm);
    if (me->subpel_mf)
      schro_motion_field_free (me->subpel_mf);
    if (me->split2_mf)
      schro_motion_field_free (me->split2_mf);
    if (me->split1_mf)
      schro_motion_field_free (me->split1_mf);
    if (me->split0_mf)
      schro_motion_field_free (me->split0_mf);
    schro_free (me);
    *pme = NULL;
  }
}


SchroMe *
schro_me_new (SchroEncoderFrame * frame)
{
  int ref;
  SchroMe *me = schro_malloc0 (sizeof (struct _SchroMe));

  SCHRO_ASSERT (me);

  me->src = frame->filtered_frame;
  me->params = &frame->params;
  me->motion = frame->motion;
  me->lambda = frame->frame_me_lambda;
  for (ref = 0; me->params->num_refs > ref; ++ref) {
    me->meElement[ref] = schro_me_element_new (frame, ref);
  }
  return me;
}

void
schro_me_free (SchroMe * me)
{
  int ref;

  if (me) {
    for (ref = 0; me->params->num_refs > ref; ++ref) {
      schro_me_element_free (&me->meElement[ref]);
    }
  }
  schro_free (me);
}

SchroFrame *
schro_me_src (SchroMe * me)
{
  SCHRO_ASSERT (me);
  return me->src;
}

SchroUpsampledFrame *
schro_me_ref (SchroMe * me, int ref_number)
{
  SCHRO_ASSERT (me && (0 == ref_number || 1 == ref_number));
  return me->meElement[ref_number]->ref;
}

SchroMotionField *
schro_me_subpel_mf (SchroMe * me, int ref_number)
{
  SCHRO_ASSERT (me && (0 == ref_number || 1 == ref_number));
  return me->meElement[ref_number]->subpel_mf;
}

void
schro_me_set_subpel_mf (SchroMe * me, SchroMotionField * mf, int ref_number)
{
  SCHRO_ASSERT (me && (0 == ref_number || 1 == ref_number));
  me->meElement[ref_number]->subpel_mf = mf;
}

SchroMotionField *
schro_me_split2_mf (SchroMe * me, int ref_number)
{
  SCHRO_ASSERT (me && (0 == ref_number || 1 == ref_number));
  return me->meElement[ref_number]->split2_mf;
}

void
schro_me_set_split2_mf (SchroMe * me, SchroMotionField * mf, int ref_number)
{
  SCHRO_ASSERT (me && (0 == ref_number || 1 == ref_number));
  me->meElement[ref_number]->split2_mf = mf;
}

SchroMotionField *
schro_me_split1_mf (SchroMe * me, int ref_number)
{
  SCHRO_ASSERT (me && (0 == ref_number || 1 == ref_number));
  return me->meElement[ref_number]->split1_mf;
}

void
schro_me_set_split1_mf (SchroMe * me, SchroMotionField * mf, int ref_number)
{
  SCHRO_ASSERT (me && (0 == ref_number || 1 == ref_number));
  me->meElement[ref_number]->split1_mf = mf;
}

SchroMotionField *
schro_me_split0_mf (SchroMe * me, int ref_number)
{
  SCHRO_ASSERT (me && (0 == ref_number || 1 == ref_number));
  return me->meElement[ref_number]->split0_mf;
}

void
schro_me_set_split0_mf (SchroMe * me, SchroMotionField * mf, int ref_number)
{
  SCHRO_ASSERT (me && (0 == ref_number || 1 == ref_number));
  me->meElement[ref_number]->split0_mf = mf;
}

SchroHierBm *
schro_me_hbm (SchroMe * me, int ref_number)
{
  SCHRO_ASSERT (me && (0 == ref_number || 1 == ref_number));
  return me->meElement[ref_number]->hbm;
}

void
schro_me_set_lambda (SchroMe * me, double lambda)
{
  SCHRO_ASSERT (me);
  me->lambda = lambda;
}

double
schro_me_lambda (SchroMe * me)
{
  SCHRO_ASSERT (me);
  return me->lambda;
}

SchroParams *
schro_me_params (SchroMe * me)
{
  SCHRO_ASSERT (me);
  return me->params;
}

SchroMotion *
schro_me_motion (SchroMe * me)
{
  SCHRO_ASSERT (me);
  return me->motion;
}

void
schro_me_set_motion (SchroMe * me, SchroMotion * motion)
{
  SCHRO_ASSERT (me);
  me->motion = motion;
}

void
schro_me_set_mc_error (SchroMe * me, double mc_error)
{
  SCHRO_ASSERT (me);
  me->mc_error = mc_error;
}

double
schro_me_mc_error (SchroMe * me)
{
  SCHRO_ASSERT (me);
  return me->mc_error;
}

void
schro_me_set_badblock_ratio (SchroMe * me, double badblocks_ratio)
{
  SCHRO_ASSERT (me);
  me->badblocks_ratio = badblocks_ratio;
}

double
schro_me_badblocks_ratio (SchroMe * me)
{
  SCHRO_ASSERT (me);
  return me->badblocks_ratio;
}

void
schro_me_set_dcblock_ratio (SchroMe * me, double dcblock_ratio)
{
  SCHRO_ASSERT (me);
  me->dcblock_ratio = dcblock_ratio;
}

double
schro_me_dcblock_ratio (SchroMe * me)
{
  SCHRO_ASSERT (me);
  return me->dcblock_ratio;
}
