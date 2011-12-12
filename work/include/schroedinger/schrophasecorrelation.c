
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <schroedinger/schro.h>
#include <schroedinger/schrofft.h>
#include <schroedinger/schrophasecorrelation.h>
#include <math.h>
#include <string.h>


#define COMPLEX_MULT_R(a,b,c,d) ((a)*(c) - (b)*(d))
#define COMPLEX_MULT_I(a,b,c,d) ((a)*(d) + (b)*(c))


static void
complex_mult_f32 (float *d1, float *d2, float *s1, float *s2,
    float *s3, float *s4, int n)
{
  int i;
  for (i = 0; i < n; i++) {
    d1[i] = COMPLEX_MULT_R (s1[i], s2[i], s3[i], s4[i]);
    d2[i] = COMPLEX_MULT_I (s1[i], s2[i], s3[i], s4[i]);
  }
}

static void
complex_normalize_f32 (float *i1, float *i2, int n)
{
  int i;
  float x;
  for (i = 0; i < n; i++) {
    x = sqrt (i1[i] * i1[i] + i2[i] * i2[i]);
    if (x > 0)
      x = 1 / x;
    i1[i] *= x;
    i2[i] *= x;
  }
}

static void
negate_f32 (float *i1, int n)
{
  int j;
  for (j = 0; j < n; j++) {
    i1[j] = -i1[j];
  }
}

static int
get_max_f32 (float *src, int n)
{
  int i;
  float max;
  int max_i;

  max = src[0];
  max_i = 0;

  for (i = 1; i < n; i++) {
    if (src[i] > max) {
      max_i = i;
      max = src[i];
    }
  }

  return max_i;
}

static void
generate_weights (float *weight, int width, int height)
{
  int i;
  int j;
  double d2;
  double mid_x, mid_y;
  double scale_x, scale_y;
  double sum;
  double weight2;

  mid_x = 0.5 * (width - 1);
  mid_y = 0.5 * (height - 1);
  scale_x = 1.0 / mid_x;
  scale_y = 1.0 / mid_y;

  sum = 0;
  for (j = 0; j < height; j++) {
    for (i = 0; i < width; i++) {
      d2 = (i - mid_x) * (i - mid_x) * scale_x * scale_x +
          (j - mid_y) * (j - mid_y) * scale_y * scale_y;
      weight[j * width + i] = exp (-2 * d2);
      sum += weight[j * width + i];
    }
  }
  weight2 = 1.0 / sum;
  for (j = 0; j < height; j++) {
    for (i = 0; i < width; i++) {
      weight[j * width + i] *= weight2;
    }
  }
}

static void
get_image (float *image, SchroFrame * frame, int x, int y, int width,
    int height, float *weight)
{
  double sum;
  int i, j;
  uint8_t *line;
  double weight2;

  sum = 0;
  for (j = 0; j < height; j++) {
    line = SCHRO_FRAME_DATA_GET_LINE (&frame->components[0], j + y);
    for (i = 0; i < width; i++) {
      sum += line[i + x] * weight[j * width + i];
    }
  }
  weight2 = 1.0 / sum;
  for (j = 0; j < height; j++) {
    line = SCHRO_FRAME_DATA_GET_LINE (&frame->components[0], j + y);
    for (i = 0; i < width; i++) {
      image[j * width + i] = line[i + x] * weight[j * width + i] * weight2;
    }
  }
}

static void
find_peak (float *ccorr, int hshift, int vshift, double *dx, double *dy)
{
  int idx, idy;
  int width = 1 << hshift;
  int height = 1 << vshift;
  int i;
  float peak;
  float a, b;

  i = get_max_f32 (ccorr, width * height);
  peak = ccorr[i];
  if (peak == 0) {
    *dx = 0;
    *dy = 0;
  }

  idx = i & (width - 1);
  if (idx >= width / 2)
    idx -= width;
  idy = i >> hshift;
  if (idy >= height / 2)
    idy -= height;

#define get_ccorr_value(x,y) ccorr[((x)&(width-1)) + (((y)&(height-1))<<hshift)]
  a = get_ccorr_value (idx + 1, idy);
  b = get_ccorr_value (idx - 1, idy);
  if (a > b) {
    *dx = idx + 0.5 * a / peak;
  } else {
    *dx = idx - 0.5 * b / peak;
  }

  a = get_ccorr_value (idx, idy + 1);
  b = get_ccorr_value (idx, idy - 1);
  if (a > b) {
    *dy = idy + 0.5 * a / peak;
  } else {
    *dy = idy - 0.5 * b / peak;
  }

  get_ccorr_value (idx - 1, idy - 1) = 0;
  get_ccorr_value (idx, idy - 1) = 0;
  get_ccorr_value (idx + 1, idy - 1) = 0;
  get_ccorr_value (idx - 1, idy) = 0;
  get_ccorr_value (idx, idy) = 0;
  get_ccorr_value (idx + 1, idy) = 0;
  get_ccorr_value (idx - 1, idy + 1) = 0;
  get_ccorr_value (idx, idy + 1) = 0;
  get_ccorr_value (idx + 1, idy + 1) = 0;
}

#define motion_field_get(mf,x,y) \
  ((mf)->motion_vectors + (y)*(mf)->x_num_blocks + (x))

static SchroFrame *
get_downsampled (SchroEncoderFrame * frame, int i)
{
  SCHRO_ASSERT (frame);
  SCHRO_ASSERT (frame->have_downsampling);

  if (i == 0) {
    return frame->filtered_frame;
  }
  return frame->downsampled_frames[i - 1];
}

#if 0
typedef struct _SchroMVComp SchroMVComp;
struct _SchroMVComp
{
  int metric;
  SchroFrame *frame;
  SchroFrame *ref;
  int dx, dy;
};

static void
schro_mvcomp_init (SchroMVComp * mvcomp, SchroFrame * frame, SchroFrame * ref)
{
  memset (mvcomp, 0, sizeof (*mvcomp));

  mvcomp->metric = 32000;
  mvcomp->frame = frame;
  mvcomp->ref = ref;
}

static void
schro_mvcomp_add (SchroMVComp * mvcomp, int i, int j, int dx, int dy)
{
  int metric;

#if 0
  metric = schro_frame_get_metric (mvcomp->frame,
      i * 8, j * 8, mvcomp->ref, i * 8 + dx, j * 8 + dy);
#endif
  metric = 0x7fffffff;
  if (metric < mvcomp->metric) {
    mvcomp->metric = metric;
    mvcomp->dx = dx;
    mvcomp->dy = dy;
  }
}
#endif

SchroPhaseCorr *
schro_phasecorr_new (SchroEncoderFrame * frame, SchroEncoderFrame * ref)
{
  SchroPhaseCorr *pc;

  pc = schro_malloc0 (sizeof (SchroPhaseCorr));

  pc->frame = frame;
  pc->ref = ref;

  return pc;
}

void
schro_phasecorr_free (SchroPhaseCorr * pc)
{
  int i;

  for (i = 0; i < pc->n_levels; i++) {
    schro_free (pc->levels[i].vecs_dx);
    schro_free (pc->levels[i].vecs_dy);
    schro_free (pc->levels[i].vecs2_dx);
    schro_free (pc->levels[i].vecs2_dy);
  }

  schro_free (pc);
}

static void
schro_phasecorr_cleanup (SchroPhaseCorr * pc)
{
  schro_free (pc->s);
  schro_free (pc->c);
  schro_free (pc->weight);
  schro_free (pc->zero);

  schro_free (pc->image1);
  schro_free (pc->image2);

  schro_free (pc->ft1r);
  schro_free (pc->ft1i);
  schro_free (pc->ft2r);
  schro_free (pc->ft2i);
  schro_free (pc->conv_r);
  schro_free (pc->conv_i);
  schro_free (pc->resr);
  schro_free (pc->resi);
}


static void
schro_phasecorr_setup (SchroPhaseCorr * pc,
    int level, int picture_shift, int hshift, int vshift)
{
  pc->picture_shift = picture_shift;

  pc->levels[level].hshift = hshift;
  pc->levels[level].vshift = vshift;
  pc->levels[level].width = 1 << hshift;
  pc->levels[level].height = 1 << vshift;
  pc->shift = hshift + vshift;
  pc->n = 1 << pc->shift;

  pc->s = schro_malloc (pc->n * sizeof (float));
  pc->c = schro_malloc (pc->n * sizeof (float));
  pc->weight = schro_malloc (pc->n * sizeof (float));
  pc->zero = schro_malloc (pc->n * sizeof (float));
  memset (pc->zero, 0, pc->n * sizeof (float));

  pc->image1 = schro_malloc (pc->n * sizeof (float));
  pc->image2 = schro_malloc (pc->n * sizeof (float));

  pc->ft1r = schro_malloc (pc->n * sizeof (float));
  pc->ft1i = schro_malloc (pc->n * sizeof (float));
  pc->ft2r = schro_malloc (pc->n * sizeof (float));
  pc->ft2i = schro_malloc (pc->n * sizeof (float));
  pc->conv_r = schro_malloc (pc->n * sizeof (float));
  pc->conv_i = schro_malloc (pc->n * sizeof (float));
  pc->resr = schro_malloc (pc->n * sizeof (float));
  pc->resi = schro_malloc (pc->n * sizeof (float));

  generate_weights (pc->weight, pc->levels[level].width,
      pc->levels[level].height);
  schro_fft_generate_tables_f32 (pc->c, pc->s, pc->shift);

  pc->levels[level].num_x =
      ((pc->frame->filtered_frame->width >> picture_shift) -
      pc->levels[level].width) / (pc->levels[level].width / 2) + 2;
  pc->levels[level].num_y =
      ((pc->frame->filtered_frame->height >> picture_shift) -
      pc->levels[level].height) / (pc->levels[level].height / 2) + 2;
  pc->levels[level].vecs_dx =
      schro_malloc (sizeof (int) * pc->levels[level].num_x *
      pc->levels[level].num_y);
  pc->levels[level].vecs_dy =
      schro_malloc (sizeof (int) * pc->levels[level].num_x *
      pc->levels[level].num_y);
  pc->levels[level].vecs2_dx =
      schro_malloc (sizeof (int) * pc->levels[level].num_x *
      pc->levels[level].num_y);
  pc->levels[level].vecs2_dy =
      schro_malloc (sizeof (int) * pc->levels[level].num_x *
      pc->levels[level].num_y);
}

static void
do_phase_corr (SchroPhaseCorr * pc, int level)
{
  int ix, iy;
  int x, y;
  SchroFrame *src_frame;
  SchroFrame *ref_frame;

  src_frame = get_downsampled (pc->frame, pc->picture_shift);
  ref_frame = get_downsampled (pc->ref, pc->picture_shift);

  for (iy = 0; iy < pc->levels[level].num_y; iy++) {
    for (ix = 0; ix < pc->levels[level].num_x; ix++) {
      double dx, dy;

      x = ((src_frame->width -
              pc->levels[level].width) * ix) / (pc->levels[level].num_x - 1);
      y = ((src_frame->height -
              pc->levels[level].height) * iy) / (pc->levels[level].num_y - 1);

      get_image (pc->image1, src_frame, x, y, pc->levels[level].width,
          pc->levels[level].height, pc->weight);
      get_image (pc->image2, ref_frame, x, y, pc->levels[level].width,
          pc->levels[level].height, pc->weight);

      schro_fft_fwd_f32 (pc->ft1r, pc->ft1i, pc->image1, pc->zero, pc->c, pc->s,
          pc->shift);
      schro_fft_fwd_f32 (pc->ft2r, pc->ft2i, pc->image2, pc->zero, pc->c, pc->s,
          pc->shift);

      negate_f32 (pc->ft2i, pc->n);

      complex_mult_f32 (pc->conv_r, pc->conv_i, pc->ft1r, pc->ft1i, pc->ft2r,
          pc->ft2i, pc->n);
      complex_normalize_f32 (pc->conv_r, pc->conv_i, pc->n);

      schro_fft_rev_f32 (pc->resr, pc->resi, pc->conv_r, pc->conv_i, pc->c,
          pc->s, pc->shift);

      find_peak (pc->resr, pc->levels[level].hshift, pc->levels[level].vshift,
          &dx, &dy);

#if 0
      schro_dump (SCHRO_DUMP_PHASE_CORR, "%d %d %d %g %g\n",
          frame->frame_number, x, y, dx, dy);
#endif

      pc->levels[level].vecs_dx[iy * pc->levels[level].num_x + ix] =
          rint (-dx * (1 << pc->picture_shift));
      pc->levels[level].vecs_dy[iy * pc->levels[level].num_x + ix] =
          rint (-dy * (1 << pc->picture_shift));

      find_peak (pc->resr, pc->levels[level].hshift, pc->levels[level].vshift,
          &dx, &dy);

      pc->levels[level].vecs2_dx[iy * pc->levels[level].num_x + ix] =
          rint (-dx * (1 << pc->picture_shift));
      pc->levels[level].vecs2_dy[iy * pc->levels[level].num_x + ix] =
          rint (-dy * (1 << pc->picture_shift));
    }
  }

}

#if 0
static void
do_motion_field (SchroPhaseCorr * pc, int level)
{
  SchroParams *params = &pc->frame->params;
  SchroMotionField *mf;
  SchroFrame *ref;
  SchroFrame *src;
  int x, y;
  int ix, iy;
  int k, l;

  mf = schro_motion_field_new (params->x_num_blocks, params->y_num_blocks);
  src = get_downsampled (pc->frame, 0);
  ref = get_downsampled (pc->ref, 0);
  for (l = 0; l < params->y_num_blocks; l++) {
    for (k = 0; k < params->x_num_blocks; k++) {
      SchroMotionVector *mv;
      int ymin, ymax;
      int xmin, xmax;
      SchroMVComp mvcomp;

      /* FIXME real block params */
      xmin = k * 8 - 2;
      xmax = k * 8 + 10;
      ymin = l * 8 - 2;
      ymax = l * 8 + 10;

      schro_mvcomp_init (&mvcomp, src, ref);

      for (iy = 0; iy < pc->levels[level].num_y; iy++) {
        for (ix = 0; ix < pc->levels[level].num_x; ix++) {
          x = ((src->width -
                  (pc->levels[level].width << pc->picture_shift)) * ix) /
              (pc->levels[level].num_x - 1);
          y = ((src->height -
                  (pc->levels[level].height << pc->picture_shift)) * iy) /
              (pc->levels[level].num_y - 1);

          if (xmax < x || ymax < y ||
              xmin >= x + (pc->levels[level].width << pc->picture_shift) ||
              ymin >= y + (pc->levels[level].height << pc->picture_shift)) {
            continue;
          }

          schro_mvcomp_add (&mvcomp, k, l,
              pc->levels[level].vecs_dx[iy * pc->levels[level].num_x + ix],
              pc->levels[level].vecs_dy[iy * pc->levels[level].num_x + ix]);
          schro_mvcomp_add (&mvcomp, k, l,
              pc->levels[level].vecs2_dx[iy * pc->levels[level].num_x + ix],
              pc->levels[level].vecs2_dy[iy * pc->levels[level].num_x + ix]);
        }
      }

      mv = motion_field_get (mf, k, l);
      mv->split = 2;
      mv->pred_mode = 1;
      mv->dx[0] = mvcomp.dx;
      mv->dy[0] = mvcomp.dy;
      mv->metric = mvcomp.metric;
    }
  }

  //schro_motion_field_scan (mf, params, src, ref, 2);

  //schro_motion_field_lshift (mf, params->mv_precision);

  schro_list_append (pc->frame->motion_field_list, mf);
}
#endif

void
schro_encoder_phasecorr_estimation (SchroPhaseCorr * pc)
{
  SchroParams *params = &pc->frame->params;
  int ref;
  int i;

  for (i = 0; i < 4; i++) {
    SCHRO_DEBUG ("block size %dx%d", 1 << (2 + 5 + i), 1 << (2 + 4 + i));
    if (pc->frame->filtered_frame->width < 1 << (2 + 5 + i) ||
        pc->frame->filtered_frame->height < 1 << (2 + 4 + i)) {
      continue;
    }

    pc->n_levels = i + 1;
    schro_phasecorr_setup (pc, i, 2, 5 + i, 4 + i);

    for (ref = 0; ref < params->num_refs; ref++) {
      do_phase_corr (pc, i);
      //do_motion_field (pc, i);
    }

    schro_phasecorr_cleanup (pc);
  }
}

#define SCHRO_METRIC_INVALID_2 0x7fffffff

void
schro_motionest_superblock_phasecorr1 (SchroMotionEst * me, int ref,
    SchroBlock * block, int i, int j)
{
  SchroMotionVector *mv;
  SchroParams *params = &me->encoder_frame->params;
  int dx, dy;
  SchroPhaseCorr *pc = me->encoder_frame->phasecorr[ref];
  int xmin, xmax, ymin, ymax;
  int ix, iy;
  int level;
  int x, y;
  int width, height;

  xmin = i * params->xbsep_luma;
  xmax = (i + 4) * params->xbsep_luma;
  ymin = j * params->ybsep_luma;
  ymax = (j + 4) * params->ybsep_luma;

  level = 0;

  width = params->video_format->width;
  height = params->video_format->height;
  for (iy = 0; iy < pc->levels[level].num_y; iy++) {
    for (ix = 0; ix < pc->levels[level].num_x; ix++) {
      x = ((width -
              (pc->levels[level].width << pc->picture_shift)) * ix) /
          (pc->levels[level].num_x - 1);
      y = ((height -
              (pc->levels[level].height << pc->picture_shift)) * iy) /
          (pc->levels[level].num_y - 1);

      if (xmax < x || ymax < y ||
          xmin >= x + (pc->levels[level].width << pc->picture_shift) ||
          ymin >= y + (pc->levels[level].height << pc->picture_shift)) {
        continue;
      }

      dx = pc->levels[level].vecs_dx[iy * pc->levels[level].num_x + ix];
      dy = pc->levels[level].vecs_dy[iy * pc->levels[level].num_x + ix];
      goto out;
    }
  }
  block->valid = FALSE;
  return;

out:

  mv = &block->mv[0][0];
  mv->split = 0;
  mv->pred_mode = 1 << ref;
  mv->using_global = 0;
  mv->u.vec.dx[ref] = dx;
  mv->u.vec.dy[ref] = dy;
  block->error = schro_motionest_superblock_get_metric (me, block, i, j);
  block->entropy = 0;
  schro_block_fixup (block);

  block->valid = (block->error != SCHRO_METRIC_INVALID_2);
}
