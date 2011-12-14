
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>
#include <schroedinger/schroorc.h>

void
schro_encoder_frame_downsample (SchroEncoderFrame * frame)
{
  int i;
  SchroFrame *last;

  SCHRO_DEBUG ("downsampling frame %d", frame->frame_number);

  last = frame->filtered_frame;
  for (i = 0; i < frame->encoder->downsample_levels; i++) {
    frame->downsampled_frames[i] =
        schro_frame_new_and_alloc_extended (NULL, frame->filtered_frame->format,
        ROUND_UP_SHIFT (frame->filtered_frame->width, i + 1),
        ROUND_UP_SHIFT (frame->filtered_frame->height, i + 1),
        MAX (frame->params.xbsep_luma, frame->params.ybsep_luma));
    schro_frame_downsample (frame->downsampled_frames[i], last);
    schro_frame_mc_edgeextend (frame->downsampled_frames[i]);
    last = frame->downsampled_frames[i];
  }
}

void
schro_encoder_frame_upsample (SchroEncoderFrame * frame)
{
  SCHRO_ASSERT (frame);
  SCHRO_DEBUG ("upsampling frame %d", frame->frame_number);

  if (frame->upsampled_original_frame) {
    return;
  }
  schro_frame_ref (frame->filtered_frame);
  frame->upsampled_original_frame =
      schro_upsampled_frame_new (frame->filtered_frame);
  schro_upsampled_frame_upsample (frame->upsampled_original_frame);
}

static double
schro_frame_component_squared_error (SchroFrameData * a, SchroFrameData * b)
{
  int j;
  double sum;

  SCHRO_ASSERT (a->width == b->width);
  SCHRO_ASSERT (a->height == b->height);

  sum = 0;
  for (j = 0; j < a->height; j++) {
    int32_t linesum;

    orc_sum_square_diff_u8 (&linesum,
        SCHRO_FRAME_DATA_GET_LINE (a, j),
        SCHRO_FRAME_DATA_GET_LINE (b, j), a->width);
    sum += linesum;
  }
  return sum;
}

void
schro_frame_mean_squared_error (SchroFrame * a, SchroFrame * b, double *mse)
{
  double sum, n;

  sum = schro_frame_component_squared_error (&a->components[0],
      &b->components[0]);
  n = a->components[0].width * a->components[0].height;
  mse[0] = sum / n;

  sum = schro_frame_component_squared_error (&a->components[1],
      &b->components[1]);
  n = a->components[1].width * a->components[1].height;
  mse[1] = sum / n;

  sum = schro_frame_component_squared_error (&a->components[2],
      &b->components[2]);
  n = a->components[2].width * a->components[2].height;
  mse[2] = sum / n;
}
