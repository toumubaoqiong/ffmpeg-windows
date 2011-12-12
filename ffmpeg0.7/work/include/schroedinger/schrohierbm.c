
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <string.h>
#include <limits.h>

/***
struct _SchroHierBm {
  int                ref_count;
  int                ref;
  int                hierarchy_levels;
  SchroParams*       params;
  SchroFrame**       downsampled_src;
  SchroFrame**       downsampled_ref;
  SchroMotionField** downsampled_mf;
};
***/

static int get_hier_levels (SchroHierBm * schro_hbm);
static int schro_hbm_ref_number (SchroHierBm * schro_hbm);


SchroHierBm *
schro_hbm_new (SchroEncoderFrame * frame, int ref)
{
  int i;
  SchroEncoderFrame *ref_frame = frame->ref_frame[ref];
  SchroHierBm *schro_hbm;

  SCHRO_ASSERT (ref_frame);

  schro_hbm = schro_malloc0 (sizeof (struct _SchroHierBm));
  schro_hbm->ref_count = 1;
  schro_hbm->hierarchy_levels = frame->encoder->downsample_levels;
  if (frame->encoder->enable_chroma_me)
    schro_hbm->use_chroma = TRUE;
  else
    schro_hbm->use_chroma = FALSE;
  schro_hbm->hierarchy_levels = frame->encoder->downsample_levels;
  schro_hbm->params = &frame->params;
  schro_hbm->ref = ref;

  schro_hbm->downsampled_src =
      schro_malloc0 (sizeof (SchroFrame *) * (schro_hbm->hierarchy_levels + 1));
  schro_hbm->downsampled_ref =
      schro_malloc0 (sizeof (SchroFrame *) * (schro_hbm->hierarchy_levels + 1));
  schro_hbm->downsampled_mf =
      schro_malloc0 (sizeof (SchroMotionField *) *
      (schro_hbm->hierarchy_levels + 1));

  schro_hbm->downsampled_src[0] = schro_frame_ref (frame->filtered_frame);
  schro_hbm->downsampled_ref[0] = schro_frame_ref (ref_frame->filtered_frame);
  for (i = 0; schro_hbm->hierarchy_levels > i; ++i) {
    SCHRO_ASSERT (frame->downsampled_frames[i]
        && ref_frame->downsampled_frames[i]);
    schro_hbm->downsampled_src[i + 1] =
        schro_frame_ref (frame->downsampled_frames[i]);
    schro_hbm->downsampled_ref[i + 1] =
        schro_frame_ref (ref_frame->downsampled_frames[i]);
  }
  return schro_hbm;
}

SchroHierBm *
schro_hbm_ref (SchroHierBm * src)
{
  SCHRO_ASSERT (src && src->ref_count > 0);
  ++src->ref_count;
  return src;
}

static void
schro_hbm_free (SchroHierBm * hbm)
{
  int i;

  for (i = 0; hbm->hierarchy_levels + 1 > i; ++i) {
    if (hbm->downsampled_src[i]) {
      schro_frame_unref (hbm->downsampled_src[i]);
    }
    if (hbm->downsampled_ref[i]) {
      schro_frame_unref (hbm->downsampled_ref[i]);
    }
    if (hbm->downsampled_mf[i]) {
      schro_motion_field_free (hbm->downsampled_mf[i]);
    }
  }
  schro_free (hbm->downsampled_mf);
  schro_free (hbm->downsampled_ref);
  schro_free (hbm->downsampled_src);
  schro_free (hbm);
}

/* unreferences a SchroHierBm structure - if it is the last reference
 * it frees the structure and sets its pointer to NULL
 * if the pointer was already NULL it's a no-op */
void
schro_hbm_unref (SchroHierBm * schro_hbm)
{
  if (0 < --schro_hbm->ref_count)
    return;
  schro_hbm_free (schro_hbm);
}

static SchroFrame *
schro_hbm_src_frame (SchroHierBm * hbm, int level)
{
  SCHRO_ASSERT (hbm && 0 < hbm->ref_count && !(get_hier_levels (hbm) < level));
  return hbm->downsampled_src[level];
}

static SchroFrame *
schro_hbm_ref_frame (SchroHierBm * hbm, int level)
{
  SCHRO_ASSERT (hbm && 0 < hbm->ref_count && !(get_hier_levels (hbm) < level));
  return hbm->downsampled_ref[level];
}

/* Note: it doesn't check whether the requested mf is not NULL */
SchroMotionField *
schro_hbm_motion_field (SchroHierBm * schro_hbm, int level)
{
  SCHRO_ASSERT (schro_hbm && schro_hbm->ref_count > 0
      && !(get_hier_levels (schro_hbm) < level));
  return schro_hbm->downsampled_mf[level];
}

static void
schro_hbm_set_motion_field (SchroHierBm * hbm, SchroMotionField * mf, int level)
{
  SCHRO_ASSERT (hbm && 0 < hbm->ref_count && !(get_hier_levels (hbm) < level));
  hbm->downsampled_mf[level] = mf;
}

static SchroParams *
schro_hbm_params (SchroHierBm * schro_hbm)
{
  SCHRO_ASSERT (schro_hbm && 0 < schro_hbm->ref_count);
  return schro_hbm->params;
}

static int
schro_hbm_ref_number (SchroHierBm * schro_hbm)
{
  SCHRO_ASSERT (schro_hbm && schro_hbm->ref_count > 0);
  return schro_hbm->ref;
}

static int
get_hier_levels (SchroHierBm * schro_hbm)
{
  SCHRO_ASSERT (schro_hbm);
  return schro_hbm->hierarchy_levels;
}

void
schro_hbm_scan (SchroHierBm * schro_hbm)
{
  int i;
  int half_scan_range = 20;
  int n_levels = get_hier_levels (schro_hbm);

  SCHRO_ASSERT (n_levels > 0);

  schro_hierarchical_bm_scan_hint (schro_hbm, n_levels, half_scan_range);
  half_scan_range >>= 1;
  for (i = n_levels - 1; 1 <= i; --i, half_scan_range >>= 1) {
    schro_hierarchical_bm_scan_hint (schro_hbm, i, MAX (3, half_scan_range));
  }
}

void
schro_hierarchical_bm_scan_hint (SchroHierBm * schro_hbm, int shift,
    int h_range)
{
  SchroMetricScan scan;
  SchroMotionVector *mv;
  SchroMotionField *mf, *hint_mf = NULL;
  SchroParams *params = schro_hbm_params (schro_hbm);
  SchroMotionVector zero_mv;
  SchroMetricInfo info;

  int xblen = params->xbsep_luma, yblen = params->ybsep_luma;
  int i;
  int j;
  int skip;
  int shift_w[3], shift_h[3];
  int split = 1 < shift ? 0 : (1 == shift ? 1 : 2);
#define LIST_LENGTH 9
  SchroMotionVector *temp_hint_mv[LIST_LENGTH], *hint_mv[LIST_LENGTH];
  unsigned int hint_mask;
  int ref = schro_hbm_ref_number (schro_hbm);
  int comp_w[3], comp_h[3];

  /* sets up for block matching */
  scan.frame = schro_hbm_src_frame (schro_hbm, shift);
  scan.ref_frame = schro_hbm_ref_frame (schro_hbm, shift);

  schro_metric_info_init (&info,
      schro_hbm_src_frame (schro_hbm, shift),
      schro_hbm_ref_frame (schro_hbm, shift),
      xblen, yblen);

  mf = schro_motion_field_new (params->x_num_blocks, params->y_num_blocks);
  schro_motion_field_set (mf, split, ref + 1);

  if (shift < get_hier_levels (schro_hbm)) {
    hint_mf = schro_hbm_motion_field (schro_hbm, shift + 1);
  }

  memset (&zero_mv, 0, sizeof (zero_mv));
  zero_mv.pred_mode = ref + 1;
  zero_mv.split = split;

  hint_mask = ~((1 << (shift + 1)) - 1);
  skip = 1 << shift;

  shift_w[0] = shift_h[0] = shift;
  shift_w[1] = shift_w[2] =
      shift + SCHRO_FRAME_FORMAT_H_SHIFT (scan.frame->format);
  shift_h[1] = shift_h[2] =
      shift + SCHRO_FRAME_FORMAT_V_SHIFT (scan.frame->format);
  comp_w[0] = xblen;
  comp_h[0] = yblen;
  comp_w[1] = comp_w[2] =
      xblen >> SCHRO_FRAME_FORMAT_H_SHIFT (scan.frame->format);
  comp_h[1] = comp_h[2] =
      yblen >> SCHRO_FRAME_FORMAT_V_SHIFT (scan.frame->format);

  for (j = 0; j < params->y_num_blocks; j += skip) {
    for (i = 0; i < params->x_num_blocks; i += skip) {
      SchroFrameData orig[3];
      int m = 0;
      int n = 0;
      int dx, dy;
      int min_m;
      int min_metric;
      int width[3], height[3];
      int k;

      /* get source data, if possible */
      if (!(scan.frame->width > (i * xblen) >> shift)
          || !(scan.frame->height > (j * yblen) >> shift)) {
        continue;
      }
      for (k = 0; 3 > k; ++k) {
        schro_frame_get_subdata (scan.frame, &orig[k], k,
            (i * xblen) >> shift_w[k], (j * yblen) >> shift_h[k]);
        width[k] = MIN (orig[k].width, comp_w[k]);
        height[k] = MIN (orig[k].height, comp_h[k]);
        if (0 == k) {
          SCHRO_ASSERT (0 < width[k] && 0 < height[k]);
        }
      }

      /* now check all candidates */
      /* always test the zero vector */
      temp_hint_mv[n] = &zero_mv;
      n++;

      /* inherit from nearby parents (star-like selection) */
      if (NULL != hint_mf) {
        int l = i & hint_mask, k = j & hint_mask, ll, kk;
        int offset[5][2] = { {0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1} };
        for (m = 0; m < 5; m++) {
          ll = l + offset[m][0] * skip * 2;
          kk = k + offset[m][1] * skip * 2;
          if (ll >= 0 && ll < params->x_num_blocks &&
              kk >= 0 && kk < params->y_num_blocks) {
            mv = hint_mf->motion_vectors + kk * params->x_num_blocks + ll;
            temp_hint_mv[n] = mv;
            n++;
          }
        }
      }

      /* inherit from neighbours (only towards SE) */
      if (i > 0) {
        mv = mf->motion_vectors + j * params->x_num_blocks + i - skip;
        temp_hint_mv[n] = mv;
        n++;
      }
      if (j > 0) {
        mv = mf->motion_vectors + (j - skip) * params->x_num_blocks + i;
        temp_hint_mv[n] = mv;
        n++;
      }
      if (i > 0 && j > 0) {
        mv = mf->motion_vectors + (j - skip) * params->x_num_blocks + i - skip;
        temp_hint_mv[n] = mv;
        n++;
      }

      SCHRO_ASSERT (n <= LIST_LENGTH);

      /* remove duplicates */
      m = 0;
      if (1 < n) {
        int k, s;
        SchroMotionVector *mv1, *mv2;
        for (k = 0; n - 1 > k; ++k) {
          int skip = 0;
          mv1 = temp_hint_mv[k];
          for (s = k + 1; n > s && !skip; ++s) {
            mv2 = temp_hint_mv[s];
            if (mv1->u.vec.dx[ref] == mv2->u.vec.dx[ref]
                && mv1->u.vec.dy[ref] == mv2->u.vec.dy[ref]) {
              skip = 1;
            }
          }
          if (!skip) {
            hint_mv[m++] = mv1;
          }
        }
        hint_mv[m++] = temp_hint_mv[n - 1];
        n = m;
      } else {
        hint_mv[0] = temp_hint_mv[0];
      }

      min_m = -1;
      min_metric = INT_MAX;
      /* choose best candidate for refinement based on SAD only. */
      for (m = 0; m < n; m++) {
        int metric = 0;

        dx = hint_mv[m]->u.vec.dx[ref];
        dx >>= shift;
        dx = CLAMP (dx + ((i*xblen) >> shift), -width[0],
            (scan.ref_frame->components + 0)->width);
        dx -= ((i*xblen) >> shift);
        dy = hint_mv[m]->u.vec.dy[ref];
        dy >>= shift;
        dy = CLAMP (dy + ((j*yblen) >> shift), -height[0],
            (scan.ref_frame->components + 0)->height);
        dy -= ((j*yblen) >> shift);
        metric = schro_metric_fast_block (&info, (i*xblen)>>shift,
            (j*yblen)>>shift, dx, dy);

        if (metric < min_metric) {
          min_metric = metric;
          min_m = m;
        }
      }
      SCHRO_ASSERT (-1 < min_m);

      /* finally do block-matching around chosen candidate MV */
      dx = hint_mv[min_m]->u.vec.dx[ref] >> shift;
      dy = hint_mv[min_m]->u.vec.dy[ref] >> shift;

      scan.block_width = width[0];
      scan.block_height = height[0];
      scan.x = i * xblen >> shift;
      scan.y = j * yblen >> shift;
      dx = MAX (-width[0] - scan.x, MIN (scan.ref_frame->width - scan.x, dx));
      dy = MAX (-height[0] - scan.y, MIN (scan.ref_frame->height - scan.y, dy));
      scan.gravity_x = dx;
      scan.gravity_y = dy;
      schro_metric_scan_setup (&scan, dx, dy, h_range, schro_hbm->use_chroma);
      SCHRO_ASSERT (!(0 > scan.scan_width) && !(0 > scan.scan_height));

      mv = mf->motion_vectors + j * params->x_num_blocks + i;

      schro_metric_scan_do_scan (&scan);
      mv->metric =
          schro_metric_scan_get_min (&scan, &dx, &dy, &mv->chroma_metric);
      dx <<= shift;
      dy <<= shift;

      mv->u.vec.dx[ref] = dx;
      mv->u.vec.dy[ref] = dy;

      mv->using_global = 0;

      mv->pred_mode = ref + 1;
    }
  }

  schro_hbm_set_motion_field (schro_hbm, mf, shift);
#undef LIST_LENGTH
}
