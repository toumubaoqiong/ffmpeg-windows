
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>

#include <math.h>

int schro_engine_get_scene_change_score (SchroEncoder * encoder, int i);
void schro_encoder_calculate_allocation (SchroEncoderFrame * frame);

/**
 * schro_engine_check_new_sequence_header:
 * @encoder: encoder
 * @frame: encoder frame
 *
 * Checks if the current picture should be the start of a new access
 * unit.
 */
static void
schro_engine_check_new_sequence_header (SchroEncoder * encoder,
    SchroEncoderFrame * frame)
{
  if (encoder->force_sequence_header ||
      frame->frame_number >= encoder->au_frame + encoder->au_distance) {
    frame->start_sequence_header = TRUE;
    encoder->au_frame = frame->frame_number;
    encoder->force_sequence_header = FALSE;
  }
}

/**
 * schro_engine_code_picture:
 * @frame: encoder frame
 * @is_ref:
 * @retire:
 * @num_refs:
 * @ref0:
 * @ref1:
 *
 * Used to set coding order and coding parameters for a picture.
 */
static void
schro_engine_code_picture (SchroEncoderFrame * frame,
    int is_ref, int retire, int num_refs, int ref0, int ref1)
{
  SchroEncoder *encoder = frame->encoder;

  SCHRO_DEBUG
      ("preparing %d as is_ref=%d retire=%d num_refs=%d ref0=%d ref1=%d",
      frame->frame_number, is_ref, retire, num_refs, ref0, ref1);

  frame->is_ref = is_ref;
  frame->retired_picture_number = retire;
  frame->num_refs = num_refs;
  frame->picture_number_ref[0] = ref0;
  frame->picture_number_ref[1] = ref1;

  frame->stages[SCHRO_ENCODER_FRAME_STAGE_HAVE_GOP].is_done = TRUE;
  frame->slot = encoder->next_slot++;

  if (num_refs > 0) {
    SCHRO_ASSERT (ref0 >= encoder->au_frame);
    frame->ref_frame[0] = schro_encoder_reference_get (encoder, ref0);
    SCHRO_ASSERT (frame->ref_frame[0]);
    schro_encoder_frame_ref (frame->ref_frame[0]);
  }
  if (num_refs > 1) {
    SCHRO_ASSERT (ref0 >= encoder->au_frame);
    frame->ref_frame[1] = schro_encoder_reference_get (encoder, ref1);
    SCHRO_ASSERT (frame->ref_frame[1]);
    schro_encoder_frame_ref (frame->ref_frame[1]);
  }
  if (is_ref) {
    int i;
    for (i = 0; i < SCHRO_LIMIT_REFERENCE_FRAMES; i++) {
      if (encoder->reference_pictures[i] == NULL)
        break;
      if (encoder->reference_pictures[i]->frame_number == retire) {
        schro_encoder_frame_unref (encoder->reference_pictures[i]);
        encoder->reference_pictures[i] = NULL;
        break;
      }
    }
    SCHRO_ASSERT (i < SCHRO_LIMIT_REFERENCE_FRAMES);
    encoder->reference_pictures[i] = frame;
    schro_encoder_frame_ref (frame);
  }
}

static int
subgroup_ready (SchroQueue * queue, int index, int subgroup_length,
    SchroEncoderFrameStateEnum gop_state)
{
  size_t i = index;
  SchroEncoderFrame *f;

  for (; index + subgroup_length > i; ++i) {
    f = queue->elements[i].data;
    SCHRO_ASSERT (!f->stages[gop_state].is_done);
    if (!f->stages[gop_state - 1].is_done) {
      return 0;
    }
  }
  return 1;
}

/**
 * schro_engine_code_intra_bailout_picture:
 * @frame:
 *
 * Sets up coding parameters for encoding as a completely independent
 * non-ref intra picture.
 */
#if 0
static void
schro_engine_code_intra (SchroEncoderFrame * frame, double weight)
{
  schro_engine_code_picture (frame, FALSE, -1, 0, -1, -1);
  frame->presentation_frame = frame->frame_number;
  frame->picture_weight = weight;
  frame->gop_length = 1;
}
#endif

static void
schro_encoder_pick_refs (SchroEncoderFrame * frame,
    SchroPictureNumber * ptr_ref0, SchroPictureNumber * ptr_ref1)
{
  SchroEncoder *encoder = frame->encoder;
  SchroPictureNumber ref0;
  SchroPictureNumber ref1;
  int i;

  ref0 = SCHRO_PICTURE_NUMBER_INVALID;
  /* pick the most recent back ref */
  for (i = 0; i < SCHRO_LIMIT_REFERENCE_FRAMES; i++) {
    if (encoder->reference_pictures[i] == NULL)
      continue;
    if (encoder->reference_pictures[i]->frame_number < frame->frame_number &&
        (ref0 == SCHRO_PICTURE_NUMBER_INVALID ||
            encoder->reference_pictures[i]->frame_number > ref0)) {
      ref0 = encoder->reference_pictures[i]->frame_number;
    }
  }
  SCHRO_ASSERT (ref0 != SCHRO_PICTURE_NUMBER_INVALID);

  /* pick the earliest forward ref */
  ref1 = SCHRO_PICTURE_NUMBER_INVALID;
  for (i = 0; i < SCHRO_LIMIT_REFERENCE_FRAMES; i++) {
    if (encoder->reference_pictures[i] == NULL)
      continue;
    if (!encoder->reference_pictures[i]->expired_reference &&
        encoder->reference_pictures[i]->frame_number > frame->frame_number &&
        (ref1 == SCHRO_PICTURE_NUMBER_INVALID ||
            encoder->reference_pictures[i]->frame_number < ref1)) {
      ref1 = encoder->reference_pictures[i]->frame_number;
    }
  }

  if (ref1 == SCHRO_PICTURE_NUMBER_INVALID) {
    /* if there's no fwd ref, pick an older back ref */
    for (i = 0; i < SCHRO_LIMIT_REFERENCE_FRAMES; i++) {
      if (encoder->reference_pictures[i] == NULL)
        continue;
      if (!encoder->reference_pictures[i]->expired_reference &&
          encoder->reference_pictures[i]->frame_number < ref0 &&
          (ref1 == SCHRO_PICTURE_NUMBER_INVALID ||
              encoder->reference_pictures[i]->frame_number > ref1)) {
        ref1 = encoder->reference_pictures[i]->frame_number;
      }
    }
  }

  *ptr_ref0 = ref0;
  *ptr_ref1 = ref1;
}

static void
schro_encoder_pick_retire (SchroEncoderFrame * frame,
    SchroPictureNumber * ptr_retire)
{
  SchroEncoder *encoder = frame->encoder;
  SchroPictureNumber retire;
  int n_refs = 0;
  int i;

  retire = SCHRO_PICTURE_NUMBER_INVALID;
  /* pick the oldest expired ref */
  for (i = 0; i < SCHRO_LIMIT_REFERENCE_FRAMES; i++) {
    if (encoder->reference_pictures[i] == NULL)
      continue;
    n_refs++;
    if (encoder->reference_pictures[i]->expired_reference &&
        (retire == SCHRO_PICTURE_NUMBER_INVALID ||
            encoder->reference_pictures[i]->frame_number < retire)) {
      retire = encoder->reference_pictures[i]->frame_number;
    }
  }

  if (retire == SCHRO_PICTURE_NUMBER_INVALID && n_refs == 3) {
    /* if we have a full queue, forceably retire something */
    for (i = 0; i < SCHRO_LIMIT_REFERENCE_FRAMES; i++) {
      if (encoder->reference_pictures[i] == NULL)
        continue;
      if (retire == SCHRO_PICTURE_NUMBER_INVALID ||
          encoder->reference_pictures[i]->frame_number < retire) {
        retire = encoder->reference_pictures[i]->frame_number;
      }
    }
    SCHRO_ASSERT (retire != SCHRO_PICTURE_NUMBER_INVALID);
  }

  *ptr_retire = retire;
}

static void
schro_encoder_expire_reference (SchroEncoder * encoder, SchroPictureNumber ref)
{
  int i;

  for (i = 0; i < SCHRO_LIMIT_REFERENCE_FRAMES; i++) {
    if (encoder->reference_pictures[i] == NULL)
      continue;
    if (encoder->reference_pictures[i]->frame_number == ref) {
      encoder->reference_pictures[i]->expired_reference = TRUE;
    }
  }
}

static void
schro_encoder_expire_refs_before (SchroEncoder * encoder,
    SchroPictureNumber ref)
{
  int i;

  for (i = 0; i < SCHRO_LIMIT_REFERENCE_FRAMES; i++) {
    if (encoder->reference_pictures[i] == NULL)
      continue;
    if (encoder->reference_pictures[i]->frame_number < ref) {
      encoder->reference_pictures[i]->expired_reference = TRUE;
    }
  }
}

static void
schro_engine_code_BBBP (SchroEncoder * encoder, int i, int gop_length)
{
  SchroEncoderFrame *frame;
  SchroEncoderFrame *f;
  int j;
  SchroPictureNumber ref0;
  SchroPictureNumber ref1;
  SchroPictureNumber retire;

  frame = encoder->frame_queue->elements[i].data;

  /* BBBP */
  frame->gop_length = gop_length;

  f = encoder->frame_queue->elements[i + gop_length - 1].data;
  if (f->start_sequence_header) {
    schro_encoder_pick_retire (f, &retire);

    schro_engine_code_picture (f, TRUE, retire, 0, -1, -1);
    f->picture_weight = encoder->magic_keyframe_weight;
  } else {
    schro_encoder_pick_retire (f, &retire);
    schro_encoder_pick_refs (f, &ref0, &ref1);

    schro_engine_code_picture (f, TRUE, retire,
        (ref1 == SCHRO_PICTURE_NUMBER_INVALID) ? 1 : 2, ref0, ref1);
    f->picture_weight = encoder->magic_inter_p_weight;

    schro_encoder_expire_reference (encoder, encoder->last_ref);
    encoder->last_ref = f->frame_number;
  }

  for (j = 0; j < gop_length - 1; j++) {
    f = encoder->frame_queue->elements[i + j].data;
    schro_encoder_pick_refs (f, &ref0, &ref1);

    schro_engine_code_picture (f, FALSE, -1, 2, ref0, ref1);
    f->presentation_frame = f->frame_number;
    if (j == gop_length - 2) {
      f->presentation_frame++;
    }
    f->picture_weight = encoder->magic_inter_b_weight;
  }

  f = encoder->frame_queue->elements[i + gop_length - 1].data;
  if (f->start_sequence_header) {
    schro_encoder_expire_refs_before (encoder, f->frame_number);
  }
}

/**
 * schro_engine_get_scene_change_score:
 * @frame: encoder frame
 * @i: index
 *
 * Calculates scene change score for two pictures.
 */
int
schro_engine_get_scene_change_score (SchroEncoder * encoder, int i)
{
  SchroEncoderFrame *frame1;
  SchroEncoderFrame *frame2;
  double luma;

  frame1 = encoder->frame_queue->elements[i].data;
  if (frame1->have_scene_change_score)
    return TRUE;

  frame2 = frame1->previous_frame;
  if (frame2 == NULL) {
    frame1->scene_change_score = 1.0;
    frame1->have_scene_change_score = TRUE;
    return TRUE;
  }
  if (!(frame2->stages[SCHRO_ENCODER_FRAME_STAGE_ANALYSE].is_done)) {
    return FALSE;
  }

  SCHRO_DEBUG ("%g %g", frame1->average_luma, frame2->average_luma);

  luma = frame1->average_luma - 16.0;
  if (luma > 0.01) {
    double mse[3];
    schro_frame_mean_squared_error (frame1->
        downsampled_frames[encoder->downsample_levels - 1],
        frame2->downsampled_frames[encoder->downsample_levels - 1], mse);
    frame1->scene_change_score = mse[0] / (luma * luma);
  } else {
    frame1->scene_change_score = 1.0;
  }

  SCHRO_DEBUG ("scene change score %g", frame1->scene_change_score);

  schro_encoder_frame_unref (frame1->previous_frame);
  frame1->previous_frame = NULL;

  frame1->have_scene_change_score = TRUE;
  return TRUE;
}


/**
 * schro_engine_pick_output_buffer_size:
 * @encoder: encoder
 * @frame: encoder frame
 *
 * Calculates allocated size of output buffer for a picture.  Horribly
 * inefficient and outdated.
 */
static int
schro_engine_pick_output_buffer_size (SchroEncoder * encoder,
    SchroEncoderFrame * frame)
{
  int size;

  size = encoder->video_format.width * encoder->video_format.height;
  switch (encoder->video_format.chroma_format) {
    case SCHRO_CHROMA_444:
      size *= 3;
      break;
    case SCHRO_CHROMA_422:
      size *= 2;
      break;
    case SCHRO_CHROMA_420:
      size += size / 2;
      break;
    default:
      SCHRO_ASSERT (0);
  }

  /* random scale factor of 2 in order to be safe */
  size *= 2;

  return size;
}

/**
 * init_params:
 * @frame: encoder frame
 *
 * Initializes params structure for picture based on encoder parameters
 * and some heuristics.
 */
void
init_params (SchroEncoderFrame * frame)
{
  SchroParams *params = &frame->params;
  SchroEncoder *encoder = frame->encoder;
  SchroVideoFormat *video_format = params->video_format;
  int shift;
  int i;
  int size;
  int overlap;

  params->video_format = &encoder->video_format;

  schro_params_init (params, params->video_format->index);

  if ((encoder->enable_noarith && frame->num_refs == 0) || params->is_lowdelay) {
    params->is_noarith = TRUE;
  }

  params->transform_depth = encoder->transform_depth;

  size = encoder->motion_block_size;
  if (size == 0) {
    if (video_format->width * video_format->height >= 1920 * 1080) {
      size = 3;
    } else if (video_format->width * video_format->height >= 960 * 540) {
      size = 2;
    } else {
      size = 1;
    }
  }
  switch (size) {
    default:
    case 1:
      params->xbsep_luma = 8;
      params->ybsep_luma = 8;
      break;
    case 2:
      params->xbsep_luma = 12;
      params->ybsep_luma = 12;
      break;
    case 3:
      params->xbsep_luma = 16;
      params->ybsep_luma = 16;
      break;
  }
  overlap = encoder->motion_block_overlap;
  if (overlap == 0) {
    overlap = 3;
  }
  switch (overlap) {
    case 1:
      params->xblen_luma = params->xbsep_luma;
      params->yblen_luma = params->ybsep_luma;
      break;
    default:
    case 2:
      params->xblen_luma = (params->xbsep_luma * 3 / 2) & (~3);
      params->yblen_luma = (params->ybsep_luma * 3 / 2) & (~3);
      break;
    case 3:
      params->xblen_luma = 2 * params->xbsep_luma;
      params->yblen_luma = 2 * params->ybsep_luma;
      break;
  }

  schro_params_calculate_mc_sizes (params);
  schro_params_calculate_iwt_sizes (params);

  switch (encoder->codeblock_size) {
    case 1:                    /* small (blocks of size 5x5) */
      shift = params->transform_depth;
      params->horiz_codeblocks[0] =
          MAX (1, (params->iwt_luma_width >> shift) / 5);
      params->vert_codeblocks[0] =
          MAX (1, (params->iwt_luma_height >> shift) / 5);
      for (i = 1; i < params->transform_depth + 1; i++) {
        shift = params->transform_depth + 1 - i;
        /* These values are empirically derived from fewer than 2 test results */
        params->horiz_codeblocks[i] =
            MAX (1, (params->iwt_luma_width >> shift) / 5);
        params->vert_codeblocks[i] =
            MAX (1, (params->iwt_luma_height >> shift) / 5);
        SCHRO_DEBUG ("codeblocks %d %d %d", i, params->horiz_codeblocks[i],
            params->vert_codeblocks[i]);
      }
      break;
    case 0:
    default:
    case 2:                    /* medium (blocks of size 8x8) */
      shift = params->transform_depth;
      params->horiz_codeblocks[0] =
          MAX (1, (params->iwt_luma_width >> shift) / 8);
      params->vert_codeblocks[0] =
          MAX (1, (params->iwt_luma_height >> shift) / 8);
      for (i = 1; i < params->transform_depth + 1; i++) {
        shift = params->transform_depth + 1 - i;
        params->horiz_codeblocks[i] =
            MAX (1, (params->iwt_luma_width >> shift) / 8);
        params->vert_codeblocks[i] =
            MAX (1, (params->iwt_luma_height >> shift) / 8);
        SCHRO_DEBUG ("codeblocks %d %d %d", i, params->horiz_codeblocks[i],
            params->vert_codeblocks[i]);
      }
      break;
    case 3:                    /* large (uses spec defaults) */
      break;
    case 4:                    /* full (codeblocks are entire subband) */
      params->horiz_codeblocks[0] = 1;
      params->vert_codeblocks[0] = 1;
      for (i = 1; i < params->transform_depth + 1; i++) {
        params->horiz_codeblocks[i] = 1;
        params->vert_codeblocks[i] = 1;
      }
      break;
  }

  if (!encoder->enable_dc_multiquant) {
    /* This is to work around a bug in the decoder that was fixed 8/2008. */
    params->horiz_codeblocks[0] = 1;
    params->vert_codeblocks[0] = 1;
  }

  params->mv_precision = encoder->mv_precision;
  if (encoder->enable_global_motion) {
    params->have_global_motion = TRUE;
  }
  if (encoder->enable_multiquant) {
    params->codeblock_mode_index = 1;
  } else {
    params->codeblock_mode_index = 0;
  }
}


void
schro_frame_set_wavelet_params (SchroEncoderFrame * frame)
{
  SchroParams *params = &frame->params;
  SchroEncoder *encoder = frame->encoder;

  if (params->num_refs > 0) {
    params->wavelet_filter_index = encoder->inter_wavelet;
  } else {
    params->wavelet_filter_index = encoder->intra_wavelet;
  }

  /* overrides for near-lossless */
  if (encoder->rate_control == 0) {
    if (encoder->noise_threshold < 40.0) {
      /* do nothing */
    } else if (encoder->noise_threshold < 47.0) {
      params->wavelet_filter_index = 1;
    } else {
      params->wavelet_filter_index = 3;
    }
  } else if (encoder->rate_control == SCHRO_ENCODER_RATE_CONTROL_LOSSLESS) {
    params->wavelet_filter_index = 3;
  }
}

static double
get_alloc (SchroEncoder * encoder, double requested_bits)
{
  double x;
  double y;
  int must_use_bits;
  double alloc;

  must_use_bits = MAX (0, encoder->buffer_level + encoder->bits_per_picture
      - encoder->buffer_size);

  x = MAX (0, requested_bits - must_use_bits) /
      MAX (0, encoder->buffer_size - encoder->bits_per_picture);

  y = 1 - exp (-x);

  alloc = must_use_bits + (encoder->buffer_level - must_use_bits) * y;

  SCHRO_DEBUG ("request %g, level %d/%d, must use %d -> x %g y %g alloc %g",
      requested_bits,
      encoder->buffer_level, encoder->buffer_size, must_use_bits, x, y, alloc);

  return alloc;
}

/**
 * schro_encoder_calculate_allocation:
 * @frame:
 *
 * Calculates the number of bits to allocate to a picture.
 */
void
schro_encoder_calculate_allocation (SchroEncoderFrame * frame)
{
  SchroEncoder *encoder = frame->encoder;

  if (encoder->rate_control != SCHRO_ENCODER_RATE_CONTROL_CONSTANT_BITRATE) {
    /* FIXME this function shouldn't be called for CBR */

    frame->hard_limit_bits = frame->output_buffer_size * 8;
    frame->allocated_mc_bits = frame->hard_limit_bits;
    frame->allocated_residual_bits = frame->hard_limit_bits;
    return;
  }

  /* FIXME should be fixed elsewhere */
  if (frame->picture_weight == 0.0)
    frame->picture_weight = 1.0;

  if (frame->num_refs == 0) {
    frame->allocated_mc_bits = 0;
    frame->allocated_residual_bits = get_alloc (encoder,
        encoder->bits_per_picture * frame->picture_weight *
        encoder->magic_allocation_scale);
    frame->hard_limit_bits = encoder->buffer_level;
  } else {
    double weight;

    frame->allocated_mc_bits = frame->estimated_mc_bits;

    weight = frame->picture_weight;
    if (frame->is_ref) {
      weight += frame->badblock_ratio * encoder->magic_badblock_multiplier_ref;
    } else {
      weight +=
          frame->badblock_ratio * encoder->magic_badblock_multiplier_nonref;
    }

    frame->allocated_residual_bits = get_alloc (encoder,
        encoder->bits_per_picture * weight * encoder->magic_allocation_scale);
    frame->allocated_residual_bits -= frame->estimated_mc_bits;
    if (frame->allocated_residual_bits < 0) {
      SCHRO_DEBUG ("allocated residual bits less than 0");
      frame->allocated_residual_bits = 0;
    }
    frame->hard_limit_bits = encoder->buffer_level;
  }
}

/**
 * init_frame:
 * @frame:
 *
 * Initializes a frame prior to any analysis.
 */
void
schro_encoder_init_frame (SchroEncoderFrame * frame)
{
  SchroEncoder *encoder = frame->encoder;

  frame->params.video_format = &encoder->video_format;

  frame->need_filtering = (encoder->filtering != 0);
  switch (encoder->gop_structure) {
    case SCHRO_ENCODER_GOP_INTRA_ONLY:
      frame->need_downsampling = FALSE;
      frame->need_upsampling = FALSE;
      frame->need_average_luma = FALSE;
      frame->need_mad = FALSE;
      break;
    case SCHRO_ENCODER_GOP_ADAPTIVE:
    case SCHRO_ENCODER_GOP_BACKREF:
    case SCHRO_ENCODER_GOP_CHAINED_BACKREF:
      frame->need_downsampling = TRUE;
      frame->need_upsampling = (encoder->mv_precision > 0);
      frame->need_average_luma = TRUE;
      frame->need_extension = TRUE;
      frame->need_mad = encoder->enable_scene_change_detection;
      break;
    case SCHRO_ENCODER_GOP_BIREF:
    case SCHRO_ENCODER_GOP_CHAINED_BIREF:
      frame->need_downsampling = TRUE;
      frame->need_upsampling = (encoder->mv_precision > 0);
      frame->need_average_luma = TRUE;
      frame->need_extension = TRUE;
      frame->need_mad = encoder->enable_scene_change_detection;
      break;
    default:
      SCHRO_ASSERT (0);
  }
}


/***** tworef *****/

/**
 * handle_gop_tworef:
 * @encoder:
 * @i:
 *
 * Sets up a minor group of pictures for the tworef engine.
 */
void
schro_encoder_handle_gop_tworef (SchroEncoder * encoder, int i)
{
  SchroEncoderFrame *frame;
  SchroEncoderFrame *f;
  int j;
  int gop_length;
  double scs_sum;

  frame = encoder->frame_queue->elements[i].data;

  SCHRO_ASSERT (frame->stages[SCHRO_ENCODER_FRAME_STAGE_HAVE_GOP].is_done ==
      FALSE);

  if (frame->busy || !frame->stages[SCHRO_ENCODER_FRAME_STAGE_ANALYSE].is_done)
    return;

  schro_engine_check_new_sequence_header (encoder, frame);

  gop_length = encoder->magic_subgroup_length;
  SCHRO_DEBUG ("handling gop from %d to %d (index %d)", encoder->gop_picture,
      encoder->gop_picture + gop_length - 1, i);

  if (encoder->end_of_stream) {
    gop_length = MIN (gop_length, encoder->frame_queue->n - i);
  }
  //intra_start = frame->start_sequence_header;
  scs_sum = 0;
  for (j = 0; j < gop_length; j++) {
    if (i + j >= encoder->frame_queue->n) {
      SCHRO_DEBUG ("not enough pictures in queue");
      return;
    }

    f = encoder->frame_queue->elements[i + j].data;

    SCHRO_ASSERT (f->stages[SCHRO_ENCODER_FRAME_STAGE_HAVE_GOP].is_done ==
        FALSE);

    if (f->busy || !f->stages[SCHRO_ENCODER_FRAME_STAGE_ANALYSE].is_done) {
      SCHRO_DEBUG ("picture %d not ready", i + j);
      return;
    }

    if (f->start_sequence_header ||
        f->frame_number >= encoder->au_frame + encoder->au_distance) {
      f->start_sequence_header = TRUE;
      if (encoder->open_gop || j == 0) {
        gop_length = j + 1;
      } else {
        gop_length = j;
      }
      break;
    }

    if (encoder->enable_scene_change_detection) {
      if (!subgroup_ready (encoder->frame_queue, i, gop_length,
              SCHRO_ENCODER_FRAME_STAGE_HAVE_GOP))
        return;                 /*not all frames in subgroup have scene change score calculated */
    } else {
      schro_engine_get_scene_change_score (encoder, i + j);
    }
    schro_dump (SCHRO_DUMP_SCENE_CHANGE, "%d %g %g\n",
        f->frame_number, f->scene_change_score, f->average_luma);
    SCHRO_DEBUG ("scene change score %g", f->scene_change_score);

    if (f->scene_change_score > encoder->magic_scene_change_threshold) {
      SCHRO_DEBUG ("Scene change detected: score %g for picture %d",
          f->scene_change_score, f->frame_number);
      if (j == 0) {
        /* If the first picture of the proposed subgroup is first
         * picture of a new shot, we want to encode a sequence header
         * and an I frame. */
        f->start_sequence_header = TRUE;
        gop_length = 1;
        break;
      } else {
        /* If there's a shot change in the middle of the proposed
         * subgroup, terminate the subgroup early.  Also flag that
         * picture as a new sequence header (not really necessary). */
        f->start_sequence_header = TRUE;
        gop_length = j;
      }
    }
#if 0
    scs_sum += f->scene_change_score;
    if (scs_sum > encoder->magic_scene_change_threshold) {
      /* matching is getting bad.  terminate gop */
      gop_length = j;
    }
#endif
  }

  SCHRO_DEBUG ("gop length %d", gop_length);

  for (j = 0; j < gop_length - 1; j++) {
    f = encoder->frame_queue->elements[i + j].data;
    SCHRO_ASSERT (f->start_sequence_header == FALSE);
  }

  if (gop_length == 1) {
    schro_engine_code_BBBP (encoder, i, gop_length);
  } else {
    schro_engine_code_BBBP (encoder, i, gop_length);
  }

  f = encoder->frame_queue->elements[i + gop_length - 1].data;
  if (f->start_sequence_header) {
    encoder->au_frame = f->frame_number;
  }

  encoder->gop_picture += gop_length;
}

int
schro_encoder_setup_frame_tworef (SchroEncoderFrame * frame)
{
  SchroEncoder *encoder = frame->encoder;

  frame->output_buffer_size =
      schro_engine_pick_output_buffer_size (encoder, frame);
  SCHRO_ASSERT (frame->output_buffer_size != 0);

  /* set up params - num_refs only */
  frame->params.num_refs = frame->num_refs;

  return TRUE;
}

int
schro_encoder_handle_quants (SchroEncoder * encoder, int i)
{
  SchroEncoderFrame *frame;

  frame = encoder->frame_queue->elements[i].data;

  if (frame->busy
      || !frame->stages[SCHRO_ENCODER_FRAME_STAGE_MODE_DECISION].is_done)
    return FALSE;

  schro_encoder_calculate_allocation (frame);
  schro_encoder_choose_quantisers (frame);
  schro_encoder_estimate_entropy (frame);

  frame->stages[SCHRO_ENCODER_FRAME_STAGE_HAVE_QUANTS].is_done = TRUE;

  return TRUE;
}

/**** backref ****/

/**
 * handle_gop_backref:
 * @encoder:
 * @i:
 *
 * Sets up a minor group of pictures for the backref engine.
 */
void
schro_encoder_handle_gop_backref (SchroEncoder * encoder, int i)
{
  SchroEncoderFrame *frame;
  SchroPictureNumber retire;
  SchroPictureNumber ref0;
  SchroPictureNumber ref1;

  frame = encoder->frame_queue->elements[i].data;

  if (frame->busy || !frame->stages[SCHRO_ENCODER_FRAME_STAGE_ANALYSE].is_done)
    return;

  schro_engine_check_new_sequence_header (encoder, frame);

  //schro_engine_code_BBBP (encoder, i, 1);
  if (frame->start_sequence_header) {
    schro_encoder_pick_retire (frame, &retire);
    schro_engine_code_picture (frame, TRUE, retire, 0, -1, -1);
    frame->picture_weight = encoder->magic_keyframe_weight;
  } else {
    schro_encoder_pick_retire (frame, &retire);
    schro_encoder_pick_refs (frame, &ref0, &ref1);

    schro_engine_code_picture (frame, TRUE, retire,
        (ref1 == SCHRO_PICTURE_NUMBER_INVALID) ? 1 : 2, ref0, ref1);
    frame->picture_weight = encoder->magic_inter_p_weight;
  }
  schro_encoder_expire_reference (encoder, frame->frame_number - 2);
  frame->presentation_frame = frame->frame_number;
  frame->picture_weight = 1;
  encoder->last_ref = frame->frame_number;

  encoder->gop_picture += 1;
  if (frame->start_sequence_header) {
    schro_encoder_expire_refs_before (encoder, frame->frame_number);
  }
}

int
schro_encoder_setup_frame_backref (SchroEncoderFrame * frame)
{
  SchroEncoder *encoder = frame->encoder;

  frame->output_buffer_size =
      schro_engine_pick_output_buffer_size (encoder, frame);

  /* set up params */
  frame->params.num_refs = frame->num_refs;

  return TRUE;
}

/*** intra-only ***/

/**
 * handle_gop_intra_only:
 * @encoder:
 * @i:
 *
 * Sets up GOP structure for an intra picture.
 */
void
schro_encoder_handle_gop_intra_only (SchroEncoder * encoder, int i)
{
  SchroEncoderFrame *frame;

  frame = encoder->frame_queue->elements[i].data;

  if (frame->busy || !frame->stages[SCHRO_ENCODER_FRAME_STAGE_ANALYSE].is_done)
    return;

  schro_engine_check_new_sequence_header (encoder, frame);

  SCHRO_DEBUG ("handling gop from %d to %d (index %d)", encoder->gop_picture,
      encoder->gop_picture, i);

  if (frame->busy || !frame->stages[SCHRO_ENCODER_FRAME_STAGE_ANALYSE].is_done) {
    SCHRO_DEBUG ("picture %d not ready", i);
    return;
  }

  schro_engine_code_picture (frame, FALSE, -1, 0, -1, -1);
  frame->presentation_frame = frame->frame_number;
  frame->picture_weight = 1.0;

  encoder->gop_picture++;
}

/**
 * setup_params_intra_only:
 * @frame:
 *
 * sets up parameters for a picture for intra-only encoding.
 */
int
schro_encoder_setup_frame_intra_only (SchroEncoderFrame * frame)
{
  SchroEncoder *encoder = frame->encoder;

  frame->output_buffer_size =
      schro_engine_pick_output_buffer_size (encoder, frame);

  frame->params.num_refs = frame->num_refs;

  /* set up params */

  return TRUE;
}


/*** lossless ***/

/**
 * setup_params_lossless:
 * @frame:
 *
 * sets up parameters for a picture for intra-only encoding.
 */
int
schro_encoder_setup_frame_lossless (SchroEncoderFrame * frame)
{
  SchroEncoder *encoder = frame->encoder;
  SchroParams *params;

  frame->output_buffer_size =
      schro_engine_pick_output_buffer_size (encoder, frame);

  frame->params.num_refs = frame->num_refs;

  /* set up params */
  params = &frame->params;

  params->wavelet_filter_index = SCHRO_WAVELET_HAAR_0;
  params->transform_depth = 3;

  params->num_refs = frame->num_refs;
  params->video_format = &encoder->video_format;
  init_params (frame);

  params->xbsep_luma = 8;
  params->xblen_luma = 8;
  params->ybsep_luma = 8;
  params->yblen_luma = 8;
  schro_params_calculate_mc_sizes (params);

  return TRUE;
}

void
schro_encoder_handle_gop_lossless (SchroEncoder * encoder, int i)
{
  schro_encoder_handle_gop_backref (encoder, i);
}

/*** low delay ***/

void
schro_encoder_handle_gop_lowdelay (SchroEncoder * encoder, int i)
{
  SchroEncoderFrame *frame;

  frame = encoder->frame_queue->elements[i].data;

  if (frame->busy || !frame->stages[SCHRO_ENCODER_FRAME_STAGE_ANALYSE].is_done)
    return;

  schro_engine_check_new_sequence_header (encoder, frame);

  SCHRO_DEBUG ("handling gop from %d to %d (index %d)", encoder->gop_picture,
      encoder->gop_picture, i);

  schro_engine_code_picture (frame, FALSE, -1, 0, -1, -1);
  frame->presentation_frame = frame->frame_number;
  frame->picture_weight = 1.0;

  encoder->gop_picture++;
}

int
schro_encoder_setup_frame_lowdelay (SchroEncoderFrame * frame)
{
  SchroEncoder *encoder = frame->encoder;
  SchroParams *params = &frame->params;
  int num;
  int denom;

  frame->output_buffer_size =
      schro_engine_pick_output_buffer_size (encoder, frame);

  /* set up params */
  params->num_refs = frame->num_refs;
  params->is_lowdelay = TRUE;

  if (encoder->horiz_slices != 0 && encoder->vert_slices != 0) {
    params->n_horiz_slices = encoder->horiz_slices;
    params->n_vert_slices = encoder->vert_slices;
  } else {
    params->n_horiz_slices =
        params->iwt_chroma_width >> params->transform_depth;
    params->n_vert_slices =
        params->iwt_chroma_height >> params->transform_depth;
  }
  schro_params_set_default_quant_matrix (params);

  num = muldiv64 (encoder->bitrate,
      encoder->video_format.frame_rate_denominator,
      encoder->video_format.frame_rate_numerator * 8);
  denom = params->n_horiz_slices * params->n_vert_slices;
  if (encoder->video_format.interlaced_coding) {
    denom *= 2;
  }
  SCHRO_ASSERT (denom != 0);
  schro_utils_reduce_fraction (&num, &denom);
  params->slice_bytes_num = num;
  params->slice_bytes_denom = denom;

  return TRUE;
}
