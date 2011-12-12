
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

static SchroFrame *get_downsampled (SchroEncoderFrame * frame, int i);



SchroRoughME *
schro_rough_me_new (SchroEncoderFrame * frame, SchroEncoderFrame * ref)
{
  SchroRoughME *rme;

  rme = schro_malloc0 (sizeof (SchroRoughME));

  rme->encoder_frame = frame;
  rme->ref_frame = ref;

  return rme;
}

void
schro_rough_me_free (SchroRoughME * rme)
{
  int i;
  for (i = 0; i < SCHRO_MAX_HIER_LEVELS; i++) {
    if (rme->motion_fields[i])
      schro_motion_field_free (rme->motion_fields[i]);
  }
  schro_free (rme);
}

void
schro_rough_me_heirarchical_scan (SchroRoughME * rme)
{
  SchroParams *params = &rme->encoder_frame->params;
  int i;
  int n_levels = rme->encoder_frame->encoder->downsample_levels;

  SCHRO_ASSERT (params->x_num_blocks != 0);
  SCHRO_ASSERT (params->y_num_blocks != 0);
  SCHRO_ASSERT (params->num_refs > 0);

  schro_rough_me_heirarchical_scan_nohint (rme, n_levels, 12);
  for (i = n_levels - 1; i >= 1; i--) {
    schro_rough_me_heirarchical_scan_hint (rme, i, 4);
  }
}

void
schro_rough_me_heirarchical_scan_nohint (SchroRoughME * rme, int shift,
    int distance)
{
  SchroMetricScan scan;
  SchroMotionVector *mv;
  SchroMotionField *mf;
  SchroParams *params = &rme->encoder_frame->params;
  int i;
  int j;
  int skip;
  /* I need to determine which reference I'm working on
   * to process the candidates MVs - note I've already checked
   * that ref_frame != NULL */
  int ref = rme->ref_frame == rme->encoder_frame->ref_frame[0] ? 0
      : (rme->ref_frame == rme->encoder_frame->ref_frame[1] ? 1 : -1);
  SCHRO_ASSERT (ref != -1);

  scan.frame = get_downsampled (rme->encoder_frame, shift);
  scan.ref_frame = get_downsampled (rme->ref_frame, shift);

  mf = schro_motion_field_new (params->x_num_blocks, params->y_num_blocks);

  schro_motion_field_set (mf, 0, 1);

  scan.block_width = params->xbsep_luma;
  scan.block_height = params->ybsep_luma;
  scan.gravity_scale = 0;
  scan.gravity_x = 0;
  scan.gravity_y = 0;

  skip = 1 << shift;
  for (j = 0; j < params->y_num_blocks; j += skip) {
    for (i = 0; i < params->x_num_blocks; i += skip) {
      int dx, dy;
      uint32_t dummy;

      scan.x = (i >> shift) * params->xbsep_luma;
      scan.y = (j >> shift) * params->ybsep_luma;
      scan.block_width = MIN (scan.frame->width - scan.x, params->xbsep_luma);
      scan.block_height = MIN (scan.frame->height - scan.y, params->ybsep_luma);
      schro_metric_scan_setup (&scan, 0, 0, distance, FALSE);
      dx = scan.ref_x + 0 - scan.x;
      dy = scan.ref_y + 0 - scan.y;
      scan.gravity_x = dx;
      scan.gravity_y = dy;

      mv = motion_field_get (mf, i, j);
      if (scan.scan_width <= 0 || scan.scan_height <= 0) {
        mv->u.vec.dx[0] = 0 << shift;
        mv->u.vec.dy[0] = 0 << shift;
        mv->metric = SCHRO_METRIC_INVALID;
        continue;
      }
#if 0
      /* this code skips blocks that are off the edge.  Instead, we
       * scan smaller block sizes */
      if (scan.x + scan.block_width >= scan.ref_frame->width ||
          scan.y + scan.block_height >= scan.ref_frame->height) {
        mv->u.vec.dx[0] = 0 << shift;
        mv->u.vec.dy[0] = 0 << shift;
        mv->metric = SCHRO_METRIC_INVALID;
        continue;
      }
#endif

      schro_metric_scan_do_scan (&scan);
      mv->metric = schro_metric_scan_get_min (&scan, &dx, &dy, &dummy);
      dx <<= shift;
      dy <<= shift;

      mv->u.vec.dx[ref] = dx;
      mv->u.vec.dy[ref] = dy;
    }
  }

  rme->motion_fields[shift] = mf;
}

void
schro_rough_me_heirarchical_scan_hint (SchroRoughME * rme, int shift,
    int distance)
{
  SchroMetricScan scan;
  SchroMotionVector *mv;
  SchroMotionField *mf;
  SchroMotionField *hint_mf;
  SchroParams *params = &rme->encoder_frame->params;
  SchroMotionVector zero_mv;
  int i;
  int j;
  int skip;
  unsigned int hint_mask;
  /* I need to determine which reference I'm working on
   * to process the candidates MVs - note I've already checked
   * that ref_frame != NULL */
  int ref = rme->ref_frame == rme->encoder_frame->ref_frame[0] ? 0
      : (rme->ref_frame == rme->encoder_frame->ref_frame[1] ? 1 : -1);
  SCHRO_ASSERT (ref != -1);


  scan.frame = get_downsampled (rme->encoder_frame, shift);
  scan.ref_frame = get_downsampled (rme->ref_frame, shift);

  mf = schro_motion_field_new (params->x_num_blocks, params->y_num_blocks);
  hint_mf = rme->motion_fields[shift + 1];

  schro_motion_field_set (mf, 0, 1);

  scan.block_width = params->xbsep_luma;
  scan.block_height = params->ybsep_luma;
  scan.gravity_scale = 0;
  scan.gravity_x = 0;
  scan.gravity_y = 0;

  memset (&zero_mv, 0, sizeof (zero_mv));

  hint_mask = ~((1 << (shift + 1)) - 1);
  skip = 1 << shift;
  for (j = 0; j < params->y_num_blocks; j += skip) {
    for (i = 0; i < params->x_num_blocks; i += skip) {
      SchroFrameData orig;
      SchroFrameData ref_data;
#define LIST_LENGTH 10
      SchroMotionVector *hint_mv[LIST_LENGTH];
      int m;
      int n = 0;
      int dx, dy;
      int min_m;
      int min_metric;
      uint32_t dummy;

      schro_frame_get_subdata (scan.frame, &orig,
          0, i * params->xbsep_luma >> shift, j * params->ybsep_luma >> shift);

      /* always test the zero vector */
      hint_mv[n] = &zero_mv;
      n++;

      /* inherit from nearby parents */
      /* This overly clever bit of code checks the parents of the diagonal
       * neighbors, which corresponds to the nearest parents. */
      for (m = 0; m < 4; m++) {
        int l = (i + skip * (-1 + 2 * (m & 1))) & hint_mask;
        int k = (j + skip * (-1 + (m & 2))) & hint_mask;
        if (l >= 0 && l < params->x_num_blocks &&
            k >= 0 && k < params->y_num_blocks) {
          hint_mv[n] = motion_field_get (hint_mf, l, k);
          n++;
        }
      }

      /* inherit from neighbors (only towards SE) */
      if (i > 0) {
        hint_mv[n] = motion_field_get (mf, i - skip, j);
        n++;
      }
      if (j > 0) {
        hint_mv[n] = motion_field_get (mf, i, j - skip);
        n++;
      }
      if (i > 0 && j > 0) {
        hint_mv[n] = motion_field_get (mf, i - skip, j - skip);
        n++;
      }

      SCHRO_ASSERT (n <= LIST_LENGTH);

      min_m = 0;
      min_metric = SCHRO_METRIC_INVALID;
      for (m = 0; m < n; m++) {
        int metric;
        int width, height;
        int x, y;

        dx = hint_mv[m]->u.vec.dx[ref];
        dy = hint_mv[m]->u.vec.dy[ref];


        x = (i * params->xbsep_luma + dx) >> shift;
        y = (j * params->ybsep_luma + dy) >> shift;
        if (x < 0 || y < 0) {
          //SCHRO_ERROR("ij %d %d dx dy %d %d", i, j, dx, dy);
          continue;
        }

        schro_frame_get_subdata (scan.ref_frame,
            &ref_data, 0,
            (i * params->xbsep_luma + dx) >> shift,
            (j * params->ybsep_luma + dy) >> shift);

        width = MIN (params->xbsep_luma, orig.width);
        height = MIN (params->ybsep_luma, orig.height);
        if (width == 0 || height == 0)
          continue;
        if (ref_data.width < width || ref_data.height < height)
          continue;

        metric = schro_metric_get (&orig, &ref_data, width, height);

        if (metric < min_metric) {
          min_metric = metric;
          min_m = m;
        }
      }

      dx = hint_mv[min_m]->u.vec.dx[ref] >> shift;
      dy = hint_mv[min_m]->u.vec.dy[ref] >> shift;
      scan.gravity_x = dx;
      scan.gravity_y = dy;

      scan.x = (i >> shift) * params->xbsep_luma;
      scan.y = (j >> shift) * params->ybsep_luma;
      scan.block_width = MIN (scan.frame->width - scan.x, params->xbsep_luma);
      scan.block_height = MIN (scan.frame->height - scan.y, params->ybsep_luma);
      schro_metric_scan_setup (&scan, dx, dy, distance, FALSE);

      mv = motion_field_get (mf, i, j);
      if (scan.scan_width <= 0 || scan.scan_height <= 0) {
        mv->u.vec.dx[ref] = 0;
        mv->u.vec.dy[ref] = 0;
        mv->metric = SCHRO_METRIC_INVALID;
        continue;
      }

      schro_metric_scan_do_scan (&scan);
      mv->metric = schro_metric_scan_get_min (&scan, &dx, &dy, &dummy);
      dx <<= shift;
      dy <<= shift;

      mv->u.vec.dx[ref] = dx;
      mv->u.vec.dy[ref] = dy;
    }
  }

  rme->motion_fields[shift] = mf;
}


static SchroFrame *
get_downsampled (SchroEncoderFrame * frame, int i)
{
  SCHRO_ASSERT (frame->have_downsampling);

  if (i == 0) {
    return frame->filtered_frame;
  }
  return frame->downsampled_frames[i - 1];
}
