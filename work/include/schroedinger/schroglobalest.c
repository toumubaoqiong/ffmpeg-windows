
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <string.h>
#include <math.h>


static void schro_motion_global_metric (SchroMotionField * field,
    SchroFrame * frame, SchroFrame * ref_frame, int ref);
static void schro_motion_field_global_estimation (SchroMotionField * mf,
    SchroGlobalMotion * gm, int mv_precision, int ref, SchroParams * params);

void
schro_encoder_global_estimation (SchroEncoderFrame * frame)
{
  SchroParams *params = &frame->params;
  SchroMotionField *mf, *mf_orig;
  int i;

  SCHRO_ERROR ("global motion is broken");

  for (i = 0; i < params->num_refs; i++) {
    mf_orig = frame->rme[i]->motion_fields[1];
    mf = schro_motion_field_new (mf_orig->x_num_blocks, mf_orig->y_num_blocks);

    SCHRO_DEBUG ("ref %d", i);
    memcpy (mf->motion_vectors, mf_orig->motion_vectors,
        sizeof (SchroMotionVector) * mf->x_num_blocks * mf->y_num_blocks);
    schro_motion_field_global_estimation (mf,
        &frame->params.global_motion[i], params->mv_precision, i, params);
    schro_motion_global_metric (mf, frame->filtered_frame,
        frame->ref_frame[i]->filtered_frame, i);
  }
}

static void
schro_motion_global_metric (SchroMotionField * field, SchroFrame * frame,
    SchroFrame * ref_frame, int ref)
{
  SchroMotionVector *mv;
  int i;
  int j;
  int x, y;

  for (j = 0; j < field->y_num_blocks; j++) {
    for (i = 0; i < field->x_num_blocks; i++) {
      mv = field->motion_vectors + j * field->x_num_blocks + i;

      x = i * 8 + mv->u.vec.dx[ref];
      y = j * 8 + mv->u.vec.dy[ref];
#if 0
      mv->metric =
          schro_metric_absdiff_u8 (frame->components[0].data + x +
          y * frame->components[0].stride, frame->components[0].stride,
          ref->components[0].data + i * 8 + j * 8 * ref->components[0].stride,
          ref->components[0].stride, 8, 8);
#endif
      mv->metric = 0;
    }
  }
}

void
schro_motion_field_global_estimation (SchroMotionField * mf,
    SchroGlobalMotion * gm, int mv_precision, int ref, SchroParams * params)
{
  int i;
  int j;
  int k;
  SchroMotionVector *mv;
  int xbsep = 2 * params->xbsep_luma;
  int ybsep = 2 * params->ybsep_luma;
  double a00, a01, a10, a11;
  double pan_x, pan_y;

  for (j = 0; j < mf->y_num_blocks; j++) {
    for (i = 0; i < mf->x_num_blocks; i++) {
      mv = mf->motion_vectors + j * mf->x_num_blocks + i;

      mv->using_global = 1;

      /* HACK */
      if (j >= mf->y_num_blocks - 4 || i >= mf->x_num_blocks - 4) {
        mv->using_global = 0;
      }
      if (j < 4 || i < 4) {
        mv->using_global = 0;
      }
    }
  }

  for (k = 0; k < 4; k++) {
    double m_x, m_y;
    double m_f, m_g;
    double ave_x, ave_y;
    double m_fx, m_fy, m_gx, m_gy;
    double m_xx, m_yy;
    double sum2;
    double stddev2;
    int n = 0;

    SCHRO_DEBUG ("step %d", k);
    m_x = 0;
    m_y = 0;
    m_f = 0;
    m_g = 0;
    for (j = 0; j < mf->y_num_blocks; j++) {
      for (i = 0; i < mf->x_num_blocks; i++) {
        mv = mf->motion_vectors + j * mf->x_num_blocks + i;
        if (mv->using_global) {
          m_f += mv->u.vec.dx[ref];
          m_g += mv->u.vec.dy[ref];
          m_x += i * xbsep;
          m_y += j * ybsep;
          n++;
        }
      }
    }
    pan_x = m_f / n;
    pan_y = m_g / n;
    ave_x = m_x / n;
    ave_y = m_y / n;

    SCHRO_DEBUG ("pan %f %f ave %f %f n %d", pan_x, pan_y, ave_x, ave_y, n);

    m_fx = 0;
    m_fy = 0;
    m_gx = 0;
    m_gy = 0;
    m_xx = 0;
    m_yy = 0;
    n = 0;
    for (j = 0; j < mf->y_num_blocks; j++) {
      for (i = 0; i < mf->x_num_blocks; i++) {
        mv = mf->motion_vectors + j * mf->x_num_blocks + i;
        if (mv->using_global) {
          m_fx += (mv->u.vec.dx[ref] - pan_x) * (i * xbsep - ave_x);
          m_fy += (mv->u.vec.dx[ref] - pan_x) * (j * ybsep - ave_y);
          m_gx += (mv->u.vec.dy[ref] - pan_y) * (i * xbsep - ave_x);
          m_gy += (mv->u.vec.dy[ref] - pan_y) * (j * ybsep - ave_y);
          m_xx += (i * xbsep - ave_x) * (i * xbsep - ave_x);
          m_yy += (j * ybsep - ave_y) * (j * ybsep - ave_y);
          n++;
        }
      }
    }
    SCHRO_DEBUG ("m_fx %f m_gx %f m_xx %f n %d", m_fx, m_gx, m_xx, n);
    a00 = m_fx / m_xx;
    a01 = m_fy / m_yy;
    a10 = m_gx / m_xx;
    a11 = m_gy / m_yy;

    pan_x -= a00 * ave_x + a01 * ave_y;
    pan_y -= a10 * ave_x + a11 * ave_y;

    SCHRO_DEBUG ("pan %f %f a[] %f %f %f %f", pan_x, pan_y, a00, a01, a10, a11);

    sum2 = 0;
    for (j = 0; j < mf->y_num_blocks; j++) {
      for (i = 0; i < mf->x_num_blocks; i++) {
        mv = mf->motion_vectors + j * mf->x_num_blocks + i;
        if (mv->using_global) {
          double dx, dy;
          dx = mv->u.vec.dx[ref] - (pan_x + a00 * i + a01 * j);
          dy = mv->u.vec.dy[ref] - (pan_y + a10 * i + a11 * j);
          sum2 += dx * dx + dy * dy;
        }
      }
    }

    stddev2 = sum2 / n;
    SCHRO_DEBUG ("stddev %f", sqrt (sum2 / n));

    if (stddev2 < 1)
      stddev2 = 1;

    n = 0;
    for (j = 0; j < mf->y_num_blocks; j++) {
      for (i = 0; i < mf->x_num_blocks; i++) {
        double dx, dy;
        mv = mf->motion_vectors + j * mf->x_num_blocks + i;
        dx = mv->u.vec.dx[ref] - (pan_x + a00 * i + a01 * j);
        dy = mv->u.vec.dy[ref] - (pan_y + a10 * i + a11 * j);
        mv->using_global = (dx * dx + dy * dy < stddev2 * 16);
        n += mv->using_global;
      }
    }
    SCHRO_DEBUG ("using n = %d", n);
  }

  SCHRO_DEBUG ("pan %f %f a[] %f %f %f %f", pan_x, pan_y, a00, a01, a10, a11);

  pan_x *= 4.0;
  pan_y *= 4.0;
  a00 *= 8;
  a01 *= 8;
  a10 *= 8;
  a11 *= 8;

  gm->b0 = rint (pan_x * (1 << mv_precision));
  gm->b1 = rint (pan_y * (1 << mv_precision));
  gm->a_exp = 16;
  gm->a00 = rint (a00 * (1 << (gm->a_exp + mv_precision)));
  gm->a01 = rint (a01 * (1 << (gm->a_exp + mv_precision)));
  gm->a10 = rint (a10 * (1 << (gm->a_exp + mv_precision)));
  gm->a11 = rint (a11 * (1 << (gm->a_exp + mv_precision)));

  for (j = 0; j < mf->y_num_blocks; j++) {
    for (i = 0; i < mf->x_num_blocks; i++) {
      mv = mf->motion_vectors + j * mf->x_num_blocks + i;
      mv->using_global = 1;
      //mv->u.vec.dx[ref] = gm->b0 + ((gm->a00 * (i*8) + gm->a01 * (j*8))>>gm->a_exp) - i*8;
      //mv->u.vec.dy[ref] = gm->b1 + ((gm->a10 * (i*8) + gm->a11 * (j*8))>>gm->a_exp) - j*8;
      mv->u.vec.dx[ref] = 0;
      mv->u.vec.dy[ref] = 0;
    }
  }
}

void
schro_motionest_superblock_global (SchroMotionEst * me, int ref,
    SchroBlock * block, int i, int j)
{
  SchroMotionVector *mv;

  mv = &block->mv[0][0];
  mv->split = 0;
  mv->pred_mode = 1 << ref;
  mv->using_global = 1;
  mv->u.vec.dx[ref] = 0;
  mv->u.vec.dy[ref] = 0;
  //block->error = schro_motionest_superblock_get_metric (me, block, i, j);
  block->error = (ref == 1) ? -1000 : 1000;
  block->entropy = 0;
  schro_block_fixup (block);

  //block->valid = (block->error != SCHRO_METRIC_INVALID_2);
  block->valid = TRUE;
}
