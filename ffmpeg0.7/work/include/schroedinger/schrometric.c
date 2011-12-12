
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include "schroorc.h"
#include <string.h>


int
schro_metric_absdiff_u8 (uint8_t * a, int a_stride, uint8_t * b, int b_stride,
    int width, int height)
{
  uint32_t metric = 0;

  if (height == 8 && width == 8) {
    orc_sad_8x8_u8 (&metric, a, a_stride, b, b_stride);
  } else if (height == 12 && width == 12) {
    orc_sad_12x12_u8 (&metric, a, a_stride, b, b_stride);
  } else if (width == 16) {
    orc_sad_16xn_u8 (&metric, a, a_stride, b, b_stride, height);
  } else if (width == 32) {
    orc_sad_32xn_u8 (&metric, a, a_stride, b, b_stride, height);
  } else {
    orc_sad_nxm_u8 (&metric, a, a_stride, b, b_stride, width, height);
  }

  return metric;
}

void
schro_metric_scan_do_scan (SchroMetricScan * scan)
{
  SchroFrameData *fd;
  SchroFrameData *fd_ref;
  int i, j;

  SCHRO_ASSERT (scan->ref_x + scan->block_width + scan->scan_width - 1 <=
      scan->frame->width + scan->frame->extension);
  SCHRO_ASSERT (scan->ref_y + scan->block_height + scan->scan_height - 1 <=
      scan->frame->height + scan->frame->extension);
  SCHRO_ASSERT (scan->ref_x >= -scan->frame->extension);
  SCHRO_ASSERT (scan->ref_y >= -scan->frame->extension);
  SCHRO_ASSERT (scan->scan_width > 0);
  SCHRO_ASSERT (scan->scan_height > 0);

  /* do luma first */
  fd = scan->frame->components + 0;
  fd_ref = scan->ref_frame->components + 0;

  if (scan->block_width == 8 && scan->block_height == 8) {
    for (j = 0; j < scan->scan_height; j++) {
      for (i = 0; i < scan->scan_width; i++) {
        orc_sad_8x8_u8 (scan->metrics + i * scan->scan_height + j,
            SCHRO_FRAME_DATA_GET_PIXEL_U8 (fd, scan->x, scan->y),
            fd->stride,
            SCHRO_FRAME_DATA_GET_PIXEL_U8 (fd_ref, scan->ref_x + i,
                scan->ref_y + j), fd_ref->stride);
      }
    }
  } else {
    for (i = 0; i < scan->scan_width; i++) {
      for (j = 0; j < scan->scan_height; j++) {
        scan->metrics[i * scan->scan_height + j] =
            schro_metric_absdiff_u8 (SCHRO_FRAME_DATA_GET_PIXEL_U8 (fd, scan->x,
                scan->y), fd->stride, SCHRO_FRAME_DATA_GET_PIXEL_U8 (fd_ref,
                scan->ref_x + i, scan->ref_y + j), fd_ref->stride,
            scan->block_width, scan->block_height);
      }
    }
  }
  memset (scan->chroma_metrics, 0, sizeof (scan->chroma_metrics));
  if (scan->use_chroma) {
    /* now do chroma ME */
    int skip_h = 1 << SCHRO_FRAME_FORMAT_H_SHIFT (scan->frame->format)
        , skip_v = 1 << SCHRO_FRAME_FORMAT_V_SHIFT (scan->frame->format);
    int x = scan->x / skip_h, y = scan->y / skip_v, ref_x =
        scan->ref_x / skip_h, ref_y = scan->ref_y / skip_v;
    int block_width = scan->block_width / skip_h, block_height =
        scan->block_height / skip_v;
    int scan_width = (scan->scan_width / skip_h) + (scan->scan_width % skip_h)
        , scan_height =
        (scan->scan_height / skip_v) + (scan->scan_height % skip_v);
    uint32_t metrics[SCHRO_LIMIT_METRIC_SCAN * SCHRO_LIMIT_METRIC_SCAN];
    int k;
    for (k = 1; 3 > k; ++k) {
      fd = scan->frame->components + k;
      fd_ref = scan->ref_frame->components + k;
      for (i = 0; i < scan_width; ++i) {
        for (j = 0; j < scan_height; ++j) {
          metrics[i * 2 * scan->scan_height + j * 2] =
              schro_metric_absdiff_u8 (SCHRO_FRAME_DATA_GET_PIXEL_U8 (fd, x, y),
              fd->stride, SCHRO_FRAME_DATA_GET_PIXEL_U8 (fd_ref, ref_x + i,
                  ref_y + j)
              , fd_ref->stride, block_width, block_height);
          if (skip_v > 1) {
            metrics[i * 2 * scan->scan_height + 1 + j * 2] =
                metrics[i * 2 * scan->scan_height + j * 2];
          }
        }
        if (skip_h > 1) {
          for (j = 0; j < scan->scan_height; ++j) {
            metrics[(i * 2 + 1) * scan->scan_height + j] =
                metrics[i * 2 * scan->scan_height + j];
          }
        }
      }
      for (j = 0; j < scan->scan_height; ++j) {
        for (i = 0; i < scan->scan_width; ++i) {
          scan->chroma_metrics[i * scan->scan_height + j] +=
              metrics[i * scan->scan_height + j];
        }
      }
    }
  }
}


/* note that gravity_x and gravity_y should contain the seed MV
 * we use to bias our search */
int
schro_metric_scan_get_min (SchroMetricScan * scan, int *dx, int *dy,
    uint32_t * chroma_error)
{
  int i, j;
  uint32_t min_metric;
  int min_gravity;
  uint32_t metric;
  uint32_t chroma_metric;
  uint32_t min_chroma_metric = 0;
  uint32_t min_total_metric = 0;
  uint32_t tmp;
  int gravity;
  int x, y;

  SCHRO_ASSERT (scan->scan_width > 0);
  SCHRO_ASSERT (scan->scan_height > 0);

  i = scan->gravity_x + scan->x - scan->ref_x;
  j = scan->gravity_y + scan->y - scan->ref_y;
  min_metric = scan->metrics[j + i * scan->scan_height];
  if (scan->use_chroma) {
    min_chroma_metric = scan->chroma_metrics[j + i * scan->scan_height];
    min_total_metric = min_metric + min_chroma_metric;
  }
  min_gravity = scan->gravity_scale *
      (abs (*dx - scan->gravity_x) + abs (*dy - scan->gravity_y));

  for (i = 0; i < scan->scan_width; i++) {
    for (j = 0; j < scan->scan_height; j++) {
      metric = scan->metrics[i * scan->scan_height + j];
      chroma_metric = scan->chroma_metrics[i * scan->scan_height + j];
      x = scan->ref_x + i - scan->x;
      y = scan->ref_y + j - scan->y;
      gravity = scan->gravity_scale *
          (abs (x - scan->gravity_x) + abs (y - scan->gravity_y));
      if (scan->use_chroma) {
        tmp = metric + chroma_metric;
        if (tmp < min_total_metric) {
          min_total_metric = tmp;
          min_metric = metric;
          min_chroma_metric = chroma_metric;
          min_gravity = gravity;
          *dx = x;
          *dy = y;
        }
      } else {
        if (metric < min_metric) {
          min_metric = metric;
          min_gravity = gravity;
          *dx = x;
          *dy = y;
        }
      }
    }
  }
  *chroma_error = min_chroma_metric;
  return min_metric;
}


void
schro_metric_scan_setup (SchroMetricScan * scan, int dx, int dy, int dist,
    int use_chroma)
{
  int xmin;
  int xmax;
  int ymin;
  int ymax;
  int xrange;
  int yrange;

  SCHRO_ASSERT (scan && scan->frame && scan->ref_frame && dist > 0);

  xmin = MAX (-scan->block_width, scan->x + dx - dist);
  xmax = MIN (scan->frame->width, scan->x + dx + dist);
  ymin = MAX (-scan->block_height, scan->y + dy - dist);
  ymax = MIN (scan->frame->height, scan->y + dy + dist);
  xmin = MAX (xmin, -scan->frame->extension);
  ymin = MAX (ymin, -scan->frame->extension);
  xmax =
      MIN (xmax,
      scan->frame->width - scan->block_width + scan->frame->extension);
  ymax =
      MIN (ymax,
      scan->frame->height - scan->block_height + scan->frame->extension);

  xrange = xmax - xmin;
  yrange = ymax - ymin;

  /* sets ref_x and ref_y */
  scan->ref_x = xmin;
  scan->ref_y = ymin;

  scan->scan_width = xrange + 1;
  scan->scan_height = yrange + 1;

  scan->use_chroma = use_chroma;

  SCHRO_ASSERT (scan->scan_width <= SCHRO_LIMIT_METRIC_SCAN);
  SCHRO_ASSERT (scan->scan_height <= SCHRO_LIMIT_METRIC_SCAN);
}



int
schro_metric_get (SchroFrameData * src1, SchroFrameData * src2, int width,
    int height)
{
  uint32_t metric = 0;

#if 0
  SCHRO_ASSERT (src1->width >= width);
  SCHRO_ASSERT (src1->height >= height);
  SCHRO_ASSERT (src2->width >= width);
  SCHRO_ASSERT (src2->height >= height);
#endif

  if (height == 8 && width == 8) {
    orc_sad_8x8_u8 (&metric,
        src1->data, src1->stride, src2->data, src2->stride);
  } else if (height == 12 && width == 12) {
    orc_sad_12x12_u8 (&metric,
        src1->data, src1->stride, src2->data, src2->stride);
  } else if (width == 16) {
    orc_sad_16xn_u8 (&metric,
        src1->data, src1->stride, src2->data, src2->stride, height);
#if 0
  } else if (width == 32) {
    orc_sad_32xn_u8 (&metric,
        src1->data, src1->stride, src2->data, src2->stride, height);
#endif
  } else {
    orc_sad_nxm_u8 (&metric,
        src1->data, src1->stride, src2->data, src2->stride, width, height);
  }

  return metric;
}

int
schro_metric_get_dc (SchroFrameData * src, int value, int width, int height)
{
  int i, j;
  int metric = 0;
  uint8_t *line;

  SCHRO_ASSERT (src->width >= width);
  SCHRO_ASSERT (src->height >= height);

  for (j = 0; j < height; j++) {
    line = SCHRO_FRAME_DATA_GET_LINE (src, j);
    for (i = 0; i < width; i++) {
      metric += abs (value - line[i]);
    }
  }
  return metric;
}

int
schro_metric_get_biref (SchroFrameData * fd, SchroFrameData * src1,
    int weight1, SchroFrameData * src2, int weight2, int shift, int width,
    int height)
{
  int i, j;
  int metric = 0;
  uint8_t *line;
  uint8_t *src1_line;
  uint8_t *src2_line;
  int offset = (1 << (shift - 1));
  int x;

#if 0
  SCHRO_ASSERT (fd->width >= width);
  SCHRO_ASSERT (fd->height >= height);
  SCHRO_ASSERT (src1->width >= width);
  SCHRO_ASSERT (src1->height >= height);
  SCHRO_ASSERT (src2->width >= width);
  SCHRO_ASSERT (src2->height >= height);
#endif

  for (j = 0; j < height; j++) {
    line = SCHRO_FRAME_DATA_GET_LINE (fd, j);
    src1_line = SCHRO_FRAME_DATA_GET_LINE (src1, j);
    src2_line = SCHRO_FRAME_DATA_GET_LINE (src2, j);
    for (i = 0; i < width; i++) {
      x = (src1_line[i] * weight1 + src2_line[i] * weight2 + offset) >> shift;
      metric += abs (line[i] - x);
    }
  }
  return metric;
}


static int
schro_frame_block_is_valid (SchroFrame *frame, int x, int y, int sx, int sy)
{
#if 0
  SCHRO_ERROR("block %d %d %d %d, frame %d %d %d %d",
      x, y, x + sx, y + sy,
      -frame->extension, -frame->extension,
      frame->width + frame->extension,
      frame->height + frame->extension);
#endif

  if (x < -frame->extension || y < -frame->extension ||
      x + sx > frame->width + frame->extension ||
      y + sy > frame->height + frame->extension) {
    SCHRO_ERROR("block %d %d %d %d, frame %d %d %d %d",
        x, y, x + sx, y + sy,
        -frame->extension, -frame->extension,
        frame->width + frame->extension,
        frame->height + frame->extension);
    return FALSE;
  }

  return TRUE;
}

static int
schro_metric_block_sad_slow (SchroMetricInfo *info, int x, int y,
    int dx, int dy)
{
  int i,j;
  int k;
  SchroFrameData fd;
  SchroFrameData fd_ref;
  int metric = 0;

#if 0
  SCHRO_ASSERT (schro_frame_block_is_valid (info->frame, x, y,
        info->block_width, info->block_height));
  SCHRO_ASSERT (schro_frame_block_is_valid (info->ref_frame, x + dx, y + dy,
        info->block_width, info->block_height));
#endif
  if (!schro_frame_block_is_valid (info->frame, x, y,
        info->block_width[0], info->block_height[0])) return INT_MAX;
  if (!schro_frame_block_is_valid (info->ref_frame, x + dx, y + dy,
        info->block_width[0], info->block_height[0])) return INT_MAX;

  for(k=0;k<3;k++){
    int width, height;

    schro_frame_get_subdata (info->frame, &fd, k,
        x>>info->h_shift[k], y>>info->v_shift[k]);
    schro_frame_get_subdata (info->ref_frame, &fd_ref, k,
        (x + dx)>>info->h_shift[k], (y + dy)>>info->v_shift[k]);

    width = MIN(fd.width, info->block_width[k]);
    height = MIN(fd.height, info->block_height[k]);

    for(j=0;j<height;j++){
      uint8_t *line = SCHRO_FRAME_DATA_GET_LINE (&fd, j);
      uint8_t *line_ref = SCHRO_FRAME_DATA_GET_LINE (&fd_ref, j);

      for(i=0;i<width;i++){
        metric += abs (line[i] - line_ref[i]);
      }
    }
  }

  return metric;
}




void
schro_metric_info_init (SchroMetricInfo *info, SchroFrame *frame,
    SchroFrame *ref_frame, int block_width, int block_height)
{
  memset (info, 0, sizeof(*info));

  info->frame = frame;
  info->ref_frame = ref_frame;

  info->block_width[0] = block_width;
  info->block_height[0] = block_height;
  info->h_shift[0] = 0;
  info->v_shift[0] = 0;
  info->h_shift[1] = SCHRO_FRAME_FORMAT_H_SHIFT (frame->format);
  info->v_shift[1] = SCHRO_FRAME_FORMAT_V_SHIFT (frame->format);
  info->block_width[1] = block_width >>
    SCHRO_FRAME_FORMAT_H_SHIFT (frame->format);
  info->block_height[1] = block_height >>
    SCHRO_FRAME_FORMAT_V_SHIFT (frame->format);
  info->h_shift[2] = SCHRO_FRAME_FORMAT_H_SHIFT (frame->format);
  info->v_shift[2] = SCHRO_FRAME_FORMAT_V_SHIFT (frame->format);
  info->block_width[2] = info->block_width[1];
  info->block_height[2] = info->block_height[1];

  info->metric = schro_metric_block_sad_slow;
  info->metric_right = schro_metric_block_sad_slow;
  info->metric_bottom = schro_metric_block_sad_slow;
  info->metric_corner = schro_metric_block_sad_slow;
}

int schro_metric_fast_block (SchroMetricInfo *info, int x, int y,
        int dx, int dy)
{
  return info->metric (info, x, y, dx, dy);
}

