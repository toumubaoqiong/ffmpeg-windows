
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#define SCHRO_ARITH_DEFINE_INLINE
#include <schroedinger/schro.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <stddef.h>
#include <orc/orc.h>
#include "schroorc.h"

#if 0
/* Used for testing bitstream */
#define MARKER(pack) schro_pack_encode_uint (pack, 1234567)
#else
#define MARKER(pack)
#endif

void schro_encoder_render_picture (SchroEncoderFrame * frame);
static void
schro_encoder_encode_picture_prediction_parameters (SchroEncoderFrame * frame);
static void schro_encoder_encode_superblock_split (SchroEncoderFrame * frame);
static void schro_encoder_encode_prediction_modes (SchroEncoderFrame * frame);
static void schro_encoder_encode_vector_data (SchroEncoderFrame * frame,
    int ref, int xy);
static void schro_encoder_encode_dc_data (SchroEncoderFrame * frame, int comp);
static void schro_encoder_encode_transform_parameters (SchroEncoderFrame *
    frame);
static void schro_encoder_encode_transform_data (SchroEncoderFrame * frame);
static int schro_encoder_pull_is_ready_locked (SchroEncoder * encoder);
static void schro_encoder_encode_codec_comment (SchroEncoder * encoder);
static void schro_encoder_encode_bitrate_comment (SchroEncoder * encoder,
    unsigned int bitrate);
static int schro_encoder_encode_padding (SchroEncoder * encoder, int n);
static void schro_encoder_clean_up_transform_subband (SchroEncoderFrame * frame,
    int component, int index);
static void schro_encoder_fixup_offsets (SchroEncoder * encoder,
    SchroBuffer * buffer, schro_bool is_eos);
static void schro_encoder_frame_complete (SchroAsyncStage * stage);
static int schro_encoder_async_schedule (SchroEncoder * encoder,
    SchroExecDomain exec_domain);
static void schro_encoder_init_perceptual_weighting (SchroEncoder * encoder);
void schro_encoder_encode_sequence_header_header (SchroEncoder * encoder,
    SchroPack * pack);
static schro_bool schro_frame_data_is_zero (SchroFrameData * fd);
static void schro_encoder_setting_set_defaults (SchroEncoder * encoder);

/*
 * Set a frame lambda based on the encoder lambda
 */
void
schro_encoder_set_frame_lambda (SchroEncoderFrame * frame)
{
  SchroEncoder *encoder = frame->encoder;

  SCHRO_ASSERT (frame);
  SCHRO_ASSERT (frame->encoder);
  switch (frame->encoder->rate_control) {
    case SCHRO_ENCODER_RATE_CONTROL_CONSTANT_BITRATE:
      if (frame->encoder->enable_rdo_cbr) {
        double q;

        //frame->frame_lambda = pow( 10.0 , -(12.0-frame->encoder->qf )/2.5 )/16.0;
        frame->frame_lambda = exp (0.921034 * encoder->qf - 13.825);

        q = (log (frame->frame_lambda) + 16.2826) / 1.6447;

        frame->frame_me_lambda = frame->encoder->magic_me_lambda_scale *
            sqrt (frame->frame_lambda);

        frame->frame_me_lambda = MIN (0.002 * exp (q * 0.2 * M_LN10), 1.0);
        if (frame->frame_me_lambda > 1.0) {
          frame->frame_me_lambda = 1.0;
        }
        frame->frame_me_lambda *= encoder->magic_me_lambda_scale;
      } else {
        /* overwritten in schro_encoder_choose_quantisers_rdo_bit_allocation */
        frame->frame_lambda = 0;
        frame->frame_me_lambda = 0.1;   /* used to be magic_mc_lambda */
      }
      break;
    case SCHRO_ENCODER_RATE_CONTROL_CONSTANT_QUALITY:
    {
      double q = encoder->quality;

      q += -3.5 * (frame->encoder->magic_error_power - 4);
      q *= 1.0 + (frame->encoder->magic_error_power - 4) * 0.2;
      if (frame->encoder->magic_error_power < 2.5) {
        q += 2;
      }
      //frame->frame_lambda = exp(((q-5)/0.7 - 7.0)*M_LN10*0.5);
      frame->frame_lambda = exp (1.6447 * q - 16.2826);

      frame->frame_me_lambda = MIN (0.002 * exp (q * 0.2 * M_LN10), 1.0);
      if (frame->frame_me_lambda > 1.0) {
        frame->frame_me_lambda = 1.0;
      }
      frame->frame_me_lambda *= encoder->magic_me_lambda_scale;

    }
      break;
    case SCHRO_ENCODER_RATE_CONTROL_LOSSLESS:
      frame->frame_me_lambda = 10;
      break;
    default:
      /* others don't use lambda */
      frame->frame_lambda = 1.0;
      frame->frame_me_lambda = 0.1;
      break;
  }
  if ((frame->num_refs != 0)) {
    if (schro_encoder_frame_is_B_frame (frame)) {
      frame->frame_lambda *= frame->encoder->magic_B_lambda_scale;
    } else {
      frame->frame_lambda *= frame->encoder->magic_P_lambda_scale;
    }
  } else {
    if (frame->encoder->rate_control ==
        SCHRO_ENCODER_RATE_CONTROL_CONSTANT_BITRATE) {
      if (frame->encoder->intra_cbr_lambda != -1) {
        frame->frame_lambda = sqrt (frame->frame_lambda *
            frame->encoder->intra_cbr_lambda);
      }
      frame->encoder->intra_cbr_lambda = frame->frame_lambda;
      SCHRO_DEBUG ("Using filtered CBR value for intra lambda %g (picture %d)",
          frame->frame_lambda, frame->frame_number);
    } else {
      frame->frame_lambda *= frame->encoder->magic_I_lambda_scale;
    }
  }
}

/*
 * Return 1 if the last subgroup has been coded, 0 otherwise
 *
 */
#if 0
int
schro_encoder_last_subgroup_coded (SchroEncoder * encoder, int picnum)
{
  int subgroup_coded = 1;
  int pos, fnum;

  int last_P_frame =
      (picnum + encoder->subgroup_length - 1) / encoder->subgroup_length;
  last_P_frame *= encoder->subgroup_length;
  last_P_frame -= encoder->subgroup_length;

  if (picnum > encoder->subgroup_length) {
    for (pos = 0; pos < encoder->frame_queue->n; ++pos) {
      SchroEncoderFrame *frame = encoder->frame_queue->elements[pos].data;
      fnum = frame->frame_number;
      if (fnum <= last_P_frame
          && fnum >= last_P_frame - encoder->subgroup_length + 1) {
        if (!(frame->state & SCHRO_ENCODER_FRAME_STATE_ENCODING))
          subgroup_coded = 0;
      }
    }
  } else {
    for (pos = 0; pos < encoder->frame_queue->n; ++pos) {
      SchroEncoderFrame *frame = encoder->frame_queue->elements[pos].data;
      fnum = frame->frame_number;
      if (fnum == 0 && !(frame->state & SCHRO_ENCODER_FRAME_STATE_ENCODING))
        subgroup_coded = 0;
    }
  }

  SCHRO_DEBUG ("Last subgroup coded value %d at %d", subgroup_coded, picnum);

  return subgroup_coded;

}
#endif

/*
 * schro_encoder_init_rc_buffer
 *
 * Initialises the buffer model for rate control
 *
 */
static void
schro_encoder_init_rc_buffer (SchroEncoder * encoder)
{
  int gop_length;

  SCHRO_ASSERT (encoder);

  gop_length = encoder->au_distance;
  if (encoder->buffer_size == 0) {
    encoder->buffer_size = 3 * encoder->bitrate;
  }
  // Set initial level at 100%
  if (encoder->buffer_level == 0) {
    encoder->buffer_level = encoder->buffer_size;
  }
  encoder->bits_per_picture = muldiv64 (encoder->bitrate,
      encoder->video_format.frame_rate_denominator,
      encoder->video_format.frame_rate_numerator);
  encoder->gop_target = muldiv64 (encoder->bitrate * gop_length,
      encoder->video_format.frame_rate_denominator,
      encoder->video_format.frame_rate_numerator);

  if (encoder->video_format.interlaced_coding) {
    encoder->bits_per_picture /= 2;
  }

  encoder->B_complexity_sum = 0;

  // Set up an initial allocation
  if (encoder->gop_structure == SCHRO_ENCODER_GOP_INTRA_ONLY) {
    encoder->I_frame_alloc = encoder->bits_per_picture;
    encoder->P_frame_alloc = 0;
    encoder->B_frame_alloc = 0;
  } else {
    int num_P_frames =
        encoder->au_distance / encoder->magic_subgroup_length - 1;
    int num_B_frames = gop_length - num_P_frames - 1;
    int total;
    encoder->I_frame_alloc = 2 ^ 24;
    encoder->P_frame_alloc = encoder->I_frame_alloc / 3;
    encoder->B_frame_alloc = encoder->P_frame_alloc / 3;
    total = encoder->I_frame_alloc + num_P_frames * encoder->P_frame_alloc +
        num_B_frames * encoder->B_frame_alloc;
    encoder->I_frame_alloc =
        (encoder->I_frame_alloc * encoder->gop_target) / total;
    encoder->P_frame_alloc =
        (encoder->P_frame_alloc * encoder->gop_target) / total;
    encoder->B_frame_alloc =
        (encoder->B_frame_alloc * encoder->gop_target) / total;
  }
  encoder->I_complexity = encoder->I_frame_alloc;
  encoder->P_complexity = encoder->P_frame_alloc;
  encoder->B_complexity = encoder->B_frame_alloc;

  SCHRO_DEBUG ("Initialising buffer with allocations (I, B, P) %d, %d, %d",
      encoder->I_frame_alloc, encoder->P_frame_alloc, encoder->B_frame_alloc);

  encoder->subgroup_position = 1;
}

/*
 * schro_encoder_projected_subgroup_bits
 *
 * Returns the total number of bits expected for the next subgroup
 *
 */
static int
schro_encoder_projected_subgroup_bits (SchroEncoder * encoder)
{
  // FIXME: take account of subgroups with an I instead of a P??
  int bits = encoder->P_complexity +
      (encoder->magic_subgroup_length - 1) * encoder->B_complexity;
  return bits;
}

/*
 * schro_encoder_target_subgroup_bits
 *
 * Returns the target for the next subgroup
 *
 */
static int
schro_encoder_target_subgroup_bits (SchroEncoder * encoder)
{
  // FIXME: take account of subgroups with an I instead of a P??
  int bits = encoder->P_frame_alloc +
      (encoder->magic_subgroup_length - 1) * encoder->B_frame_alloc;
  return bits;
}

/*
 * schro_encoder_cbr_allocate:
 *
 * TM5-style bit allocation routine
 *
 */
static void
schro_encoder_cbr_allocate (SchroEncoder * encoder, int fnum)
{
  int gop_length;
  int Icty;
  int Pcty;
  int Bcty;
  int num_I_frames;
  int num_P_frames;
  int num_B_frames;
  int total_gop_bits;
  int sg_len;
  double buffer_occ;
  int min_bits;

  SCHRO_ASSERT (encoder);

  gop_length = encoder->au_distance;
  Icty = encoder->I_complexity;
  Pcty = encoder->P_complexity;
  Bcty = encoder->B_complexity;
  num_I_frames = 1;
  num_P_frames = encoder->au_distance / encoder->magic_subgroup_length - 1;
  num_B_frames = gop_length - num_I_frames - num_P_frames;
  total_gop_bits = muldiv64 (encoder->bitrate * gop_length,
      encoder->video_format.frame_rate_denominator,
      encoder->video_format.frame_rate_numerator);
  sg_len = encoder->magic_subgroup_length;
  buffer_occ =
      ((double) encoder->buffer_level) / ((double) encoder->buffer_size);

  if (encoder->gop_structure != SCHRO_ENCODER_GOP_INTRA_ONLY) {
    double correction;
    if (buffer_occ < 0.9 && ((fnum + 1) % 4 * sg_len) == 0) {
      // If we're undershooting buffer target, correct slowly
      correction = MIN (0.25, 0.25 * (0.9 - buffer_occ) / 0.9);
      encoder->gop_target =
          (long int) ((double) (total_gop_bits) * (1.0 - correction));
    } else if (buffer_occ > 0.9 && ((fnum + 1) % sg_len) == 0) {
      // If we're overshooting buffer target, correct quickly
      correction = MIN (0.5, 0.5 * (buffer_occ - 0.9) / 0.9);
      encoder->gop_target =
          (long int) ((double) (total_gop_bits) * (1.0 + correction));
    }
  }

  min_bits = total_gop_bits / (100 * gop_length);

  encoder->I_frame_alloc = (long int) (encoder->gop_target
      / (num_I_frames
          + (double) (num_P_frames * Pcty) / Icty
          + (double) (num_B_frames * Bcty) / Icty));

  encoder->I_frame_alloc = MAX (min_bits, encoder->I_frame_alloc);

  encoder->P_frame_alloc = (long int) (encoder->gop_target
      / (num_P_frames
          + (double) (num_I_frames * Icty) / Pcty
          + (double) (num_B_frames * Bcty) / Pcty));

  encoder->P_frame_alloc = MAX (min_bits, encoder->P_frame_alloc);

  encoder->B_frame_alloc = (long int) (encoder->gop_target
      / (num_B_frames
          + (double) (num_I_frames * Icty) / Bcty
          + (double) (num_P_frames * Pcty) / Bcty));

  encoder->B_frame_alloc = MAX (min_bits, encoder->B_frame_alloc);

}

/*
 * schro_encoder_cbr_update
 *
 * Sets the qf (and hence lambdas) for the next subgroup to be coded
 *
 */
static void
schro_encoder_cbr_update (SchroEncoderFrame * frame, int num_bits)
{
  SchroEncoder *encoder = frame->encoder;
  double target_ratio;
  double actual_ratio;
  double filter_tap;
  int P_separation;
  int field_factor;
  int emergency_realloc;
  int target;
  double tbits, pbits;

  // The target buffer occupancy
  target_ratio = 0.9;
  actual_ratio = (double) (encoder->buffer_level) /
      (double) (encoder->buffer_size);
  P_separation = encoder->magic_subgroup_length;

  // 1 is coding frames, 2 if coding fields
  field_factor = 1;
  if (encoder->video_format.interlaced_coding)
    field_factor = 2;

  emergency_realloc = 0;

  // Decrement the subgroup frame counter. This is zero just after the last
  // B frame before the next P frame i.e. before the start of a subgroup
  encoder->subgroup_position--;

  /* Determine the filter tap for adjusting lambda */
  if ((frame->frame_number / field_factor) <= 3 * P_separation) {
    // Adjust immediately at the beginning of the sequence
    filter_tap = 1.0;
  } else {
    if (actual_ratio > target_ratio)
      filter_tap = (actual_ratio - target_ratio) / (1.0 - target_ratio);
    else
      filter_tap = (target_ratio - actual_ratio) / target_ratio;

    filter_tap = CLAMP (filter_tap, 0.25, 1.0);
  }

  // Now for the actual update
  if (encoder->gop_structure != SCHRO_ENCODER_GOP_INTRA_ONLY) {
    // Long-GOP coding

    if (frame->num_refs == 0) {
      encoder->I_complexity = num_bits;
      target = encoder->I_frame_alloc;

      if (num_bits < target / 2 || num_bits > 3 * target)
        emergency_realloc = 1;

      if ((frame->frame_number / field_factor) == 0) {  //FIXME: needed?
        // We've just coded the very first frame, which is a special
        // case as the B frames which normally follow are missing
        encoder->subgroup_position = P_separation;
      }
    }

    if (((frame->frame_number) / field_factor) % P_separation != 0) {
      // Scheduled B picture
      encoder->B_complexity_sum += num_bits;
      target = encoder->B_frame_alloc;

      if (num_bits < target / 2 || num_bits > 3 * target) {
        emergency_realloc = 1;
      }

    } else if (frame->num_refs != 0) {
      // Scheduled P picture (if inserted I picture, don't change the complexity)
      encoder->P_complexity = num_bits;
      target = encoder->P_frame_alloc;

      if (num_bits < target / 2 || num_bits > 3 * target) {
        emergency_realloc = 1;
      }

    }

    if (encoder->subgroup_position == 0 || emergency_realloc == 1) {
      double K;
      double new_qf;

      if (emergency_realloc == 1)
        SCHRO_DEBUG ("Major undershoot of frame bit rate: Reallocating");

      // We recompute allocations for the next subgroup
      if (P_separation > 1 && encoder->subgroup_position < P_separation - 1) {
        encoder->B_complexity =
            encoder->B_complexity_sum / (P_separation - 1 -
            encoder->subgroup_position);
      }
      schro_encoder_cbr_allocate (encoder, frame->frame_number / field_factor);

      // We work out what this means for the quality factor and set it

      tbits = (double) (schro_encoder_target_subgroup_bits (encoder));
      pbits = (double) (schro_encoder_projected_subgroup_bits (encoder));

      SCHRO_DEBUG ("Reallocating: target bits = %g, projected bits = %g", tbits,
          pbits);

      // Determine K value in model
      K = pow (pbits, 2) * pow (10.0,
          ((double) 2 / 5 * (12 - encoder->qf))) / 16;

      // Determine a new qf from K
      new_qf = 12 - (double) 5 / 2 * log10 (16 * K / pow (tbits, 2));

      if ((abs (encoder->qf - new_qf) >= 0.25 || new_qf <= 4.0)
          && new_qf <= 8.0)
        new_qf = filter_tap * new_qf + (1.0 - filter_tap) * encoder->qf;

      if (new_qf <= 8.0) {
        if (pbits < 2 * tbits) {
          new_qf = MAX (new_qf, encoder->qf - 1.0);
        } else {
          new_qf = MAX (new_qf, encoder->qf - 2.0);
        }
      }
      new_qf =
          MIN (new_qf,
          5 + 10 * ((double) encoder->buffer_level) / encoder->buffer_size);

      encoder->qf = new_qf;
      SCHRO_DEBUG ("Setting qf for next subgroup to %g, bits %d", encoder->qf,
          encoder->buffer_level);

      // Reset the frame counter
      if (encoder->subgroup_position == 0) {
        encoder->subgroup_position = encoder->magic_subgroup_length;
        encoder->B_complexity_sum = 0;
      }

    }
  } else {
    // We're doing intraonly coding

    double tbits = (double) encoder->bits_per_picture;
    double pbits = (double) num_bits;

    // Determine K value
    double K =
        pow (pbits, 2) * pow (10.0, ((double) 2 / 5 * (12 - encoder->qf))) / 16;

    // Determine a new QF
    double new_qf = 12 - (double) 5 / 2 * log10 (16 * K / pow (tbits, 2));

    // Adjust the QF to meet the target
    double abs_delta = abs (new_qf - encoder->qf);
    if (abs_delta > 0.01) {
      // Rate of convergence to new QF
      double r;

      // Use an Sshaped curve to compute r
      //   Where the qf difference is less than 1/2, r decreases to zero
      //   exponentially, so for small differences in QF we jump straight
      //   to the target value. For large differences in QF, r converges
      //   exponentially to 0.75, so we converge to the target value at
      //   a fixed rate.

      //   Overall behaviour is to converge steadily for 2 or 3 frames until
      //   close and then lock to the correct value. This avoids very rapid
      //   changes in quality.

      //   Actual parameters may be adjusted later. Some applications may
      //   require instant lock.

      double lg_diff = log (abs_delta / 2.0);
      if (lg_diff < 0.0)
        r = 0.5 * exp (-lg_diff * lg_diff / 2.0);
      else
        r = 1.0 - 0.5 * exp (-lg_diff * lg_diff / 2.0);

      r *= 0.75;
      encoder->qf = r * encoder->qf + (1.0 - r) * new_qf;
      SCHRO_DEBUG ("Setting qf for next subgroup to %g", encoder->qf);
    }
  }

}

/**
 * schro_encoder_new:
 *
 * Create a new encoder object.
 *
 * Returns: a new encoder object
 */
SchroEncoder *
schro_encoder_new (void)
{
  SchroEncoder *encoder;
  int c, b;

  encoder = schro_malloc0 (sizeof (SchroEncoder));

  encoder->version_major = 2;
  encoder->version_minor = 2;

  encoder->au_frame = -1;

  encoder->last_ref = -1;
  encoder->qf = 7.0;

  schro_encoder_setting_set_defaults (encoder);

  schro_video_format_set_std_video_format (&encoder->video_format,
      SCHRO_VIDEO_FORMAT_CUSTOM);

  encoder->inserted_buffers =
      schro_list_new_full ((SchroListFreeFunc) schro_buffer_unref, NULL);

  for (c = 0; c < 3; ++c) {
    for (b = 0; b < SCHRO_LIMIT_SUBBANDS; ++b) {
      encoder->average_arith_context_ratios_intra[c][b] = 1.0;
      encoder->average_arith_context_ratios_inter[c][b] = 1.0;
    }
  }

  return encoder;
}

static void
handle_gop_enum (SchroEncoder * encoder)
{
  switch (encoder->gop_structure) {
    case SCHRO_ENCODER_GOP_BACKREF:
    case SCHRO_ENCODER_GOP_CHAINED_BACKREF:
      SCHRO_DEBUG ("Setting backref\n");
      encoder->init_frame = schro_encoder_init_frame;
      encoder->handle_gop = schro_encoder_handle_gop_backref;
      encoder->handle_quants = schro_encoder_handle_quants;
      encoder->setup_frame = schro_encoder_setup_frame_backref;
      break;
    case SCHRO_ENCODER_GOP_INTRA_ONLY:
      SCHRO_DEBUG ("Setting intra only\n");
      encoder->init_frame = schro_encoder_init_frame;
      encoder->handle_gop = schro_encoder_handle_gop_intra_only;
      encoder->handle_quants = schro_encoder_handle_quants;
      encoder->setup_frame = schro_encoder_setup_frame_intra_only;
      break;
    case SCHRO_ENCODER_GOP_ADAPTIVE:
    case SCHRO_ENCODER_GOP_BIREF:
    case SCHRO_ENCODER_GOP_CHAINED_BIREF:
      SCHRO_DEBUG ("Setting tworef engine\n");
      encoder->init_frame = schro_encoder_init_frame;
      encoder->handle_gop = schro_encoder_handle_gop_tworef;
      encoder->handle_quants = schro_encoder_handle_quants;
      encoder->setup_frame = schro_encoder_setup_frame_tworef;
      break;
    default:
      SCHRO_ASSERT (0);
  }

}

/**
 * schro_encoder_start:
 * @encoder: an encoder object
 *
 * Locks in encoder configuration and causes the encoder to start
 * encoding pictures.  At this point, the encoder will start worker
 * threads to do the actual encoding.
 */
void
schro_encoder_start (SchroEncoder * encoder)
{
  encoder->frame_queue = schro_queue_new (encoder->queue_depth,
      (SchroQueueFreeFunc) schro_encoder_frame_unref);
  SCHRO_DEBUG ("queue depth %d", encoder->queue_depth);

  encoder->engine_init = 1;
  encoder->force_sequence_header = TRUE;

  /* add check on 'enable' switches */
  if (encoder->enable_scene_change_detection) {
    encoder->magic_scene_change_threshold = 3.25;
  } else {
    encoder->magic_scene_change_threshold = 0.2;
  }
  if (encoder->enable_bigblock_estimation && encoder->enable_deep_estimation) {
    /* only one can be set at any one time */
    encoder->enable_bigblock_estimation = FALSE;
  } else if (!encoder->enable_bigblock_estimation
      && !encoder->enable_deep_estimation) {
    SCHRO_ERROR ("no motion estimation selected!");
    SCHRO_ASSERT (0);
  }
  if (!encoder->enable_deep_estimation && encoder->enable_chroma_me) {
    encoder->enable_chroma_me = FALSE;
  }
  /* Global motion is broken */
  encoder->enable_global_motion = FALSE;

  if (encoder->video_format.luma_excursion >= 256 ||
      encoder->video_format.chroma_excursion >= 256) {
    SCHRO_ERROR ("luma or chroma excursion is too large for 8 bit");
  }

  encoder->video_format.interlaced_coding = encoder->interlaced_coding;

  schro_encoder_encode_codec_comment (encoder);

  schro_tables_init ();
  schro_encoder_init_perceptual_weighting (encoder);
  /* special value indicating invalid */
  encoder->intra_cbr_lambda = -1;

  schro_encoder_init_error_tables (encoder);

  encoder->async = schro_async_new (0,
      (SchroAsyncScheduleFunc) schro_encoder_async_schedule,
      (SchroAsyncCompleteFunc) schro_encoder_frame_complete, encoder);

  if (encoder->force_profile == SCHRO_ENCODER_PROFILE_AUTO) {
    if (encoder->rate_control == SCHRO_ENCODER_RATE_CONTROL_LOW_DELAY) {
      encoder->force_profile = SCHRO_ENCODER_PROFILE_VC2_LOW_DELAY;
    } else if (encoder->enable_noarith) {
      encoder->force_profile = SCHRO_ENCODER_PROFILE_VC2_SIMPLE;
    } else if (encoder->gop_structure == SCHRO_ENCODER_GOP_INTRA_ONLY) {
      encoder->force_profile = SCHRO_ENCODER_PROFILE_VC2_MAIN;
    } else {
      encoder->force_profile = SCHRO_ENCODER_PROFILE_MAIN;
    }
  }

  if (encoder->bitrate == 0) {
    encoder->bitrate = encoder->video_format.width *
        encoder->video_format.height *
        encoder->video_format.frame_rate_numerator /
        encoder->video_format.frame_rate_denominator;
    if (encoder->force_profile == SCHRO_ENCODER_PROFILE_MAIN) {
      encoder->bitrate /= 10;
    } else {
      encoder->bitrate *= 2;
    }
  }

  switch (encoder->force_profile) {
    case SCHRO_ENCODER_PROFILE_VC2_LOW_DELAY:
      encoder->profile = SCHRO_PROFILE_LOW_DELAY;
      encoder->rate_control = SCHRO_ENCODER_RATE_CONTROL_LOW_DELAY;
      encoder->gop_structure = SCHRO_ENCODER_GOP_INTRA_ONLY;
      encoder->au_distance = 1;
      break;
    case SCHRO_ENCODER_PROFILE_VC2_SIMPLE:
      encoder->profile = SCHRO_PROFILE_SIMPLE;
      encoder->enable_noarith = TRUE;
      encoder->gop_structure = SCHRO_ENCODER_GOP_INTRA_ONLY;
      encoder->au_distance = 1;
      break;
    case SCHRO_ENCODER_PROFILE_VC2_MAIN:
      encoder->profile = SCHRO_PROFILE_MAIN_INTRA;
      encoder->enable_noarith = FALSE;
      encoder->gop_structure = SCHRO_ENCODER_GOP_INTRA_ONLY;
      encoder->au_distance = 1;
      break;
    case SCHRO_ENCODER_PROFILE_MAIN:
      encoder->profile = SCHRO_PROFILE_MAIN;
      encoder->enable_noarith = FALSE;
      break;
    case SCHRO_ENCODER_PROFILE_AUTO:
    default:
      SCHRO_ASSERT (0);
  }

  switch (encoder->rate_control) {
    case SCHRO_ENCODER_RATE_CONTROL_CONSTANT_NOISE_THRESHOLD:
      handle_gop_enum (encoder);
      encoder->quantiser_engine = SCHRO_QUANTISER_ENGINE_SIMPLE;
      break;
    case SCHRO_ENCODER_RATE_CONTROL_CONSTANT_BITRATE:
      handle_gop_enum (encoder);
      if (encoder->enable_rdo_cbr) {
        encoder->quantiser_engine = SCHRO_QUANTISER_ENGINE_CBR;
      } else {
        encoder->quantiser_engine = SCHRO_QUANTISER_ENGINE_RDO_BIT_ALLOCATION;
      }
      schro_encoder_init_rc_buffer (encoder);

      schro_encoder_encode_bitrate_comment (encoder, encoder->bitrate);
      break;
    case SCHRO_ENCODER_RATE_CONTROL_LOW_DELAY:
      encoder->quantiser_engine = SCHRO_QUANTISER_ENGINE_LOWDELAY;

      encoder->init_frame = schro_encoder_init_frame;
      encoder->handle_gop = schro_encoder_handle_gop_lowdelay;
      encoder->handle_quants = schro_encoder_handle_quants;
      encoder->setup_frame = schro_encoder_setup_frame_lowdelay;

      schro_encoder_encode_bitrate_comment (encoder, encoder->bitrate);
      break;
    case SCHRO_ENCODER_RATE_CONTROL_LOSSLESS:
      if (encoder->force_profile == SCHRO_ENCODER_PROFILE_MAIN) {
        encoder->handle_gop = schro_encoder_handle_gop_lossless;
      } else {
        encoder->handle_gop = schro_encoder_handle_gop_intra_only;
      }
      encoder->quantiser_engine = SCHRO_QUANTISER_ENGINE_LOSSLESS;
      encoder->init_frame = schro_encoder_init_frame;
      encoder->handle_quants = schro_encoder_handle_quants;
      encoder->setup_frame = schro_encoder_setup_frame_tworef;
      break;
    case SCHRO_ENCODER_RATE_CONTROL_CONSTANT_LAMBDA:
      handle_gop_enum (encoder);
      encoder->quantiser_engine = SCHRO_QUANTISER_ENGINE_RDO_LAMBDA;
      break;
    case SCHRO_ENCODER_RATE_CONTROL_CONSTANT_ERROR:
      handle_gop_enum (encoder);
      encoder->quantiser_engine = SCHRO_QUANTISER_ENGINE_CONSTANT_ERROR;
      break;
    case SCHRO_ENCODER_RATE_CONTROL_CONSTANT_QUALITY:
      handle_gop_enum (encoder);
      encoder->quantiser_engine = SCHRO_QUANTISER_ENGINE_RDO_LAMBDA;
      break;
    default:
      SCHRO_ASSERT (0);
  }

  encoder->level = 0;
  encoder->video_format.index =
      schro_video_format_get_std_video_format (&encoder->video_format);
  switch (encoder->profile) {
    case SCHRO_PROFILE_LOW_DELAY:
    case SCHRO_PROFILE_SIMPLE:
    case SCHRO_PROFILE_MAIN_INTRA:
      if (schro_video_format_check_VC2_DL (&encoder->video_format)) {
        encoder->level = 1;
      }
      break;
    case SCHRO_PROFILE_MAIN:
      if (schro_video_format_check_MP_DL (&encoder->video_format)) {
        encoder->level = 128;
      }
      break;
    default:
      SCHRO_ASSERT (0);
  }

  {
    /* 1, 3, 4, 6 have overflows at transform-depth>=5
     * 5 has errors at transform-depth=any */
    const int max_transform_depths[] = { 5, 4, 5, 4, 4, 3, 4 };

    encoder->transform_depth = MIN (encoder->transform_depth,
        max_transform_depths[encoder->intra_wavelet]);
    if (encoder->gop_structure != SCHRO_ENCODER_GOP_INTRA_ONLY) {
      encoder->transform_depth = MIN (encoder->transform_depth,
          max_transform_depths[encoder->inter_wavelet]);
    }
  }

  encoder->start_time = schro_utils_get_time ();
}

/**
 * schro_encoder_free:
 * @encoder: an encoder object
 *
 * Frees an encoder object and all its resources.
 */
void
schro_encoder_free (SchroEncoder * encoder)
{
  int i;

  if (encoder->async) {
    schro_async_free (encoder->async);
  }

  if (encoder->last_frame) {
    schro_encoder_frame_unref (encoder->last_frame);
    encoder->last_frame = NULL;
  }

  for (i = 0; i < SCHRO_LIMIT_REFERENCE_FRAMES; i++) {
    if (encoder->reference_pictures[i]) {
      schro_encoder_frame_unref (encoder->reference_pictures[i]);
      encoder->reference_pictures[i] = NULL;
    }
  }
  if (encoder->frame_queue) {
    schro_queue_free (encoder->frame_queue);
  }

  if (encoder->inserted_buffers) {
    schro_list_free (encoder->inserted_buffers);
  }

  schro_free (encoder);
}

static void
schro_encoder_init_perceptual_weighting (SchroEncoder * encoder)
{
  encoder->cycles_per_degree_vert = 0.5 * encoder->video_format.height /
      (2.0 * atan (0.5 / encoder->perceptual_distance) * 180 / M_PI);
  encoder->cycles_per_degree_horiz = encoder->cycles_per_degree_vert *
      encoder->video_format.aspect_ratio_denominator /
      encoder->video_format.aspect_ratio_numerator;

  if (encoder->video_format.interlaced_coding) {
    encoder->cycles_per_degree_vert *= 0.5;
  }

  SCHRO_DEBUG ("cycles per degree horiz=%g vert=%g",
      encoder->cycles_per_degree_horiz, encoder->cycles_per_degree_vert);

  switch (encoder->perceptual_weighting) {
    default:
    case SCHRO_ENCODER_PERCEPTUAL_CONSTANT:
      schro_encoder_calculate_subband_weights (encoder,
          schro_encoder_perceptual_weight_constant);
      break;
    case SCHRO_ENCODER_PERCEPTUAL_CCIR959:
      schro_encoder_calculate_subband_weights (encoder,
          schro_encoder_perceptual_weight_ccir959);
      break;
    case SCHRO_ENCODER_PERCEPTUAL_MOO:
      schro_encoder_calculate_subband_weights (encoder,
          schro_encoder_perceptual_weight_moo);
      break;
    case SCHRO_ENCODER_PERCEPTUAL_MANOS_SAKRISON:
      schro_encoder_calculate_subband_weights (encoder,
          schro_encoder_perceptual_weight_manos_sakrison);
      break;
  }
}

/**
 * schro_encoder_get_video_format:
 * @encoder: an encoder object
 *
 * Creates a new SchroVideoFormat structure and copies the
 * video format information of @decoder into it.
 *
 * When no longer needed, the returned pointer should be
 * freed using free().
 *
 * Returns: a pointer to a SchroVideoFormat structure
 */
SchroVideoFormat *
schro_encoder_get_video_format (SchroEncoder * encoder)
{
  SchroVideoFormat *format;

  format = malloc (sizeof (SchroVideoFormat));
  memcpy (format, &encoder->video_format, sizeof (SchroVideoFormat));

  return format;
}

/**
 * schro_encoder_set_video_format:
 * @encoder: an encoder object
 * @format: the video format to use
 *
 * Sets the video format used by @encoder to the values specified
 * in @format.  This function may only be called before schro_encoder_start()
 * is called on the encoder.
 */
void
schro_encoder_set_video_format (SchroEncoder * encoder,
    SchroVideoFormat * format)
{
  /* FIXME check that we're in the right state to do this */

  memcpy (&encoder->video_format, format, sizeof (SchroVideoFormat));

  schro_video_format_validate (&encoder->video_format);
}

/**
 * schro_encoder_set_packet_assembly:
 * @encoder: an encoder object
 * @value:
 *
 * If @value is TRUE, all subsequent calls to schro_encoder_pull()
 * will return a buffer that contains all Dirac packets related to
 * a frame.  If @value is FALSE, each buffer will be one Dirac packet.
 *
 * It is recommended that users always call this function with TRUE
 * immediately after creating an encoder object.
 */
void
schro_encoder_set_packet_assembly (SchroEncoder * encoder, int value)
{
  encoder->assemble_packets = value;
}

static int
schro_encoder_push_is_ready_locked (SchroEncoder * encoder)
{
  int n;

  if (encoder->end_of_stream) {
    return FALSE;
  }

  n = schro_queue_slots_available (encoder->frame_queue);

  if (encoder->video_format.interlaced_coding) {
    return (n >= 2);
  } else {
    return (n >= 1);
  }
}

/**
 * schro_encoder_push_ready:
 * @encoder: an encoder object
 *
 * Returns true if the encoder has available space for additional
 * video frames.
 *
 * Returns: TRUE if the encoder is ready for another video frame to
 * be pushed.
 */
int
schro_encoder_push_ready (SchroEncoder * encoder)
{
  int ret;

  schro_async_lock (encoder->async);
  ret = schro_encoder_push_is_ready_locked (encoder);
  schro_async_unlock (encoder->async);

  return ret;
}

/**
 * schro_encoder_force_sequence_header:
 * @encoder: an encoder object
 *
 * Indicates to the encoder that the next frame pushed should be
 * encoded with a sequence header.
 */
void
schro_encoder_force_sequence_header (SchroEncoder * encoder)
{
  encoder->force_sequence_header = TRUE;
}

/**
 * schro_encoder_push_frame:
 * @encoder: an encoder object
 * @frame: a frame to encode
 *
 * Provides a frame to the encoder to encode.
 */
void
schro_encoder_push_frame (SchroEncoder * encoder, SchroFrame * frame)
{
  schro_encoder_push_frame_full (encoder, frame, NULL);
}

/**
 * schro_encoder_push_frame_full:
 * @encoder: an encoder object
 * @frame: a frame to encode
 * @priv: a private tag
 *
 * Provides a frame to the encoder to encode.  The value of @priv is
 * returned when schro_encoder_pull_full() is called for the encoded
 * frame.
 */
void
schro_encoder_push_frame_full (SchroEncoder * encoder, SchroFrame * frame,
    void *priv)
{
  schro_async_lock (encoder->async);
  if (encoder->video_format.interlaced_coding == 0) {
    SchroEncoderFrame *encoder_frame;
    SchroFrameFormat format;

    encoder_frame = schro_encoder_frame_new (encoder);
    encoder_frame->encoder = encoder;

    encoder_frame->priv = priv;

    encoder_frame->previous_frame = encoder->last_frame;
    schro_encoder_frame_ref (encoder_frame);
    encoder->last_frame = encoder_frame;

    format =
        schro_params_get_frame_format (8, encoder->video_format.chroma_format);
    if (format == frame->format) {
      encoder_frame->original_frame = frame;
    } else {
      encoder_frame->original_frame = schro_frame_new_and_alloc (NULL, format,
          encoder->video_format.width, encoder->video_format.height);
      schro_frame_convert (encoder_frame->original_frame, frame);
      schro_frame_unref (frame);
      frame = NULL;
    }

    encoder_frame->frame_number = encoder->next_frame_number++;

    if (schro_queue_is_full (encoder->frame_queue)) {
      SCHRO_ERROR ("push when queue full");
      SCHRO_ASSERT (0);
    }
    schro_queue_add (encoder->frame_queue, encoder_frame,
        encoder_frame->frame_number);
    schro_async_signal_scheduler (encoder->async);
    schro_async_unlock (encoder->async);
  } else {
    SchroEncoderFrame *encoder_frame1;
    SchroEncoderFrame *encoder_frame2;
    SchroFrameFormat format;
    int width, height;

    encoder_frame1 = schro_encoder_frame_new (encoder);
    encoder_frame1->encoder = encoder;
    encoder_frame1->priv = priv;
    encoder_frame2 = schro_encoder_frame_new (encoder);
    encoder_frame2->encoder = encoder;

    encoder_frame1->previous_frame = encoder->last_frame;
    schro_encoder_frame_ref (encoder_frame1);
    encoder_frame2->previous_frame = encoder_frame1;
    schro_encoder_frame_ref (encoder_frame2);
    encoder->last_frame = encoder_frame2;

    schro_video_format_get_picture_luma_size (&encoder->video_format,
        &width, &height);
    format = schro_params_get_frame_format (8,
        encoder->video_format.chroma_format);

    encoder_frame1->original_frame = schro_frame_new_and_alloc (NULL, format,
        width, height);
    encoder_frame2->original_frame = schro_frame_new_and_alloc (NULL, format,
        width, height);
    schro_frame_split_fields (encoder_frame1->original_frame,
        encoder_frame2->original_frame, frame);
    schro_frame_unref (frame);
    frame = NULL;

    encoder_frame1->frame_number = encoder->next_frame_number++;
    encoder_frame2->frame_number = encoder->next_frame_number++;

    if (schro_queue_slots_available (encoder->frame_queue) < 2) {
      SCHRO_ERROR ("push when queue full");
      SCHRO_ASSERT (0);
    }
    schro_queue_add (encoder->frame_queue, encoder_frame1,
        encoder_frame1->frame_number);
    schro_queue_add (encoder->frame_queue, encoder_frame2,
        encoder_frame2->frame_number);
    schro_async_signal_scheduler (encoder->async);
    schro_async_unlock (encoder->async);
  }
}

static int
schro_encoder_pull_is_ready_locked (SchroEncoder * encoder)
{
  int i;

  for (i = 0; i < encoder->frame_queue->n; i++) {
    SchroEncoderFrame *frame;
    frame = encoder->frame_queue->elements[i].data;
    if (frame->slot == encoder->output_slot &&
        (frame->stages[SCHRO_ENCODER_FRAME_STAGE_DONE].is_done)) {
      return TRUE;
    }
  }

  if (schro_queue_is_empty (encoder->frame_queue) && encoder->end_of_stream
      && !encoder->end_of_stream_pulled) {
    return TRUE;
  }

  return FALSE;
}

static void
schro_encoder_shift_frame_queue (SchroEncoder * encoder)
{
  SchroEncoderFrame *frame;

  while (!schro_queue_is_empty (encoder->frame_queue)) {
    frame = encoder->frame_queue->elements[0].data;
    if (!(frame->stages[SCHRO_ENCODER_FRAME_STAGE_FREE].is_done)) {
      break;
    }

    schro_queue_pop (encoder->frame_queue);
  }
}

/**
 * schro_encoder_pull:
 * @encoder: an encoder object
 * @presentation_frame: (output) latest decodable frame
 *
 * Pulls a buffer of compressed video from the encoder.  If
 * @presentation_frame is not NULL, the frame number of the
 * latest decodable frame is returned.
 *
 * Returns: a buffer containing compressed video
 */
SchroBuffer *
schro_encoder_pull (SchroEncoder * encoder, int *presentation_frame)
{
  return schro_encoder_pull_full (encoder, presentation_frame, NULL);
}

static int
schro_encoder_frame_get_encoded_size (SchroEncoderFrame * frame)
{
  SchroBuffer *buffer;
  int size = 0;
  int i;

  if (frame->sequence_header_buffer) {
    size += frame->sequence_header_buffer->length;
  }

  for (i = 0; i < schro_list_get_size (frame->inserted_buffers); i++) {
    buffer = schro_list_get (frame->inserted_buffers, i);
    size += buffer->length;
  }
  for (i = 0; i < schro_list_get_size (frame->encoder->inserted_buffers); i++) {
    buffer = schro_list_get (frame->encoder->inserted_buffers, i);
    size += buffer->length;
  }

  size += frame->output_buffer->length;

  return size;
}

static void
schro_encoder_frame_assemble_buffer (SchroEncoderFrame * frame,
    SchroBuffer * buffer)
{
  SchroBuffer *buf;
  int offset = 0;
  int i;

  if (frame->sequence_header_buffer) {
    buf = frame->sequence_header_buffer;
    schro_encoder_fixup_offsets (frame->encoder, buf, FALSE);
    orc_memcpy (buffer->data + offset, buf->data, buf->length);
    offset += frame->sequence_header_buffer->length;
  }

  for (i = 0; i < schro_list_get_size (frame->inserted_buffers); i++) {
    buf = schro_list_get (frame->inserted_buffers, i);
    schro_encoder_fixup_offsets (frame->encoder, buf, FALSE);
    orc_memcpy (buffer->data + offset, buf->data, buf->length);
    offset += buf->length;
  }
  while (schro_list_get_size (frame->encoder->inserted_buffers) > 0) {
    buf = schro_list_remove (frame->encoder->inserted_buffers, 0);
    schro_encoder_fixup_offsets (frame->encoder, buf, FALSE);
    orc_memcpy (buffer->data + offset, buf->data, buf->length);
    offset += buf->length;
    schro_buffer_unref (buf);
    buf = NULL;
  }

  buf = frame->output_buffer;
  schro_encoder_fixup_offsets (frame->encoder, buf, FALSE);
  orc_memcpy (buffer->data + offset, buf->data, buf->length);
  offset += buf->length;
}

int
schro_encoder_get_frame_stats_size (SchroEncoder * encoder)
{
  return 21;
}

void
schro_encoder_get_frame_stats (SchroEncoder * encoder, double *dest, int n)
{
  memcpy (dest, encoder->frame_stats, sizeof (double) * MIN (n, 21));
}

/**
 * schro_encoder_pull_full:
 * @encoder: an encoder object
 * @presentation_frame: (output) latest decodable frame
 * @priv: (output)
 *
 * Pulls a buffer of compressed video from the encoder.  If
 * @presentation_frame is not NULL, the frame number of the
 * latest decodable frame is returned.  If @priv is not NULL,
 * the private tag attached to the pushed uncompressed frame
 * is returned.
 *
 * Returns: a buffer containing compressed video
 */
SchroBuffer *
schro_encoder_pull_full (SchroEncoder * encoder, int *presentation_frame,
    void **priv)
{
  SchroBuffer *buffer;
  int i;
  int done = FALSE;

  SCHRO_DEBUG ("pulling slot %d", encoder->output_slot);

  schro_async_lock (encoder->async);
  for (i = 0; i < encoder->frame_queue->n; i++) {
    SchroEncoderFrame *frame;
    frame = encoder->frame_queue->elements[i].data;
    if (frame->slot == encoder->output_slot &&
        (frame->stages[SCHRO_ENCODER_FRAME_STAGE_DONE].is_done)) {
      int is_picture = FALSE;

      if (presentation_frame) {
        *presentation_frame = frame->presentation_frame;
      }

      if (encoder->assemble_packets) {
        int size;

        size = schro_encoder_frame_get_encoded_size (frame);
        buffer = schro_buffer_new_and_alloc (size);
        schro_encoder_frame_assemble_buffer (frame, buffer);

        if (priv) {
          *priv = frame->priv;
        }
        done = TRUE;
      } else {
        if (frame->sequence_header_buffer) {
          buffer = frame->sequence_header_buffer;
          frame->sequence_header_buffer = NULL;
        } else if (schro_list_get_size (frame->inserted_buffers) > 0) {
          buffer = schro_list_remove (frame->inserted_buffers, 0);
        } else if (schro_list_get_size (encoder->inserted_buffers) > 0) {
          buffer = schro_list_remove (encoder->inserted_buffers, 0);
        } else {
          if (priv) {
            *priv = frame->priv;
          }
          buffer = frame->output_buffer;
          frame->output_buffer = NULL;

          done = TRUE;
        }
      }
      if (done) {
        double elapsed_time;

        is_picture = TRUE;
        frame->stages[SCHRO_ENCODER_FRAME_STAGE_FREE].is_done = TRUE;
        encoder->output_slot++;

        elapsed_time = schro_utils_get_time () - encoder->start_time;

        if (frame->num_refs == 0) {
          frame->badblock_ratio = 0;
          frame->dcblock_ratio = 0;
          frame->mc_error = 0;
        }

        schro_dump (SCHRO_DUMP_PICTURE, "%d %d %d %d %d %g %d %d %d %d %g %d %g %g %g %g %g %g %g %g\n", frame->frame_number,   /* 0 */
            frame->num_refs, frame->is_ref, frame->allocated_mc_bits, frame->allocated_residual_bits, frame->picture_weight,    /* 5 */
            frame->estimated_mc_bits, frame->estimated_residual_bits, frame->actual_mc_bits, frame->actual_residual_bits, frame->scene_change_score,    /* 10 */
            encoder->buffer_level, frame->frame_lambda, frame->mc_error, frame->mean_squared_error_luma, frame->mean_squared_error_chroma,      /* 15 */
            elapsed_time,
            frame->badblock_ratio, frame->dcblock_ratio, frame->hist_slope);

        encoder->frame_stats[0] = frame->frame_number;
        encoder->frame_stats[1] = frame->num_refs;
        encoder->frame_stats[2] = frame->is_ref;
        encoder->frame_stats[3] = frame->allocated_mc_bits;
        encoder->frame_stats[4] = frame->allocated_residual_bits;
        encoder->frame_stats[5] = frame->picture_weight;
        encoder->frame_stats[6] = frame->estimated_mc_bits;
        encoder->frame_stats[7] = frame->estimated_residual_bits;
        encoder->frame_stats[8] = frame->actual_mc_bits;
        encoder->frame_stats[8] = frame->actual_residual_bits;
        encoder->frame_stats[10] = frame->scene_change_score;
        encoder->frame_stats[11] = encoder->buffer_level;
        encoder->frame_stats[12] = frame->frame_lambda;
        encoder->frame_stats[13] = frame->mc_error;
        encoder->frame_stats[14] = frame->mean_squared_error_luma;
        encoder->frame_stats[15] = frame->mean_squared_error_chroma;
        encoder->frame_stats[16] = elapsed_time;
        encoder->frame_stats[17] = frame->badblock_ratio;
        encoder->frame_stats[18] = frame->dcblock_ratio;
        encoder->frame_stats[19] = frame->hist_slope;
        encoder->frame_stats[20] = frame->mssim;

        schro_encoder_shift_frame_queue (encoder);
      }

      if (encoder->rate_control == SCHRO_ENCODER_RATE_CONTROL_CONSTANT_BITRATE) {
        encoder->buffer_level -= buffer->length * 8;
        if (is_picture) {
          if (encoder->buffer_level < 0) {
            SCHRO_WARNING ("buffer underrun by %d bits",
                -encoder->buffer_level);
            encoder->buffer_level = 0;
          }
          encoder->buffer_level += encoder->bits_per_picture;
          if (encoder->buffer_level > encoder->buffer_size) {
            int n;

            n = (encoder->buffer_level - encoder->buffer_size + 7) / 8;
            SCHRO_DEBUG ("buffer overrun, adding padding of %d bytes", n);
            n = schro_encoder_encode_padding (encoder, n);
            encoder->buffer_level -= n * 8;
          }
          SCHRO_DEBUG ("buffer level %d of %d bits", encoder->buffer_level,
              encoder->buffer_size);
        }
      }

      if (!encoder->assemble_packets) {
        schro_encoder_fixup_offsets (encoder, buffer, FALSE);
      }

      SCHRO_DEBUG ("got buffer length=%d", buffer->length);
      schro_async_unlock (encoder->async);
      return buffer;
    }
  }

  if (schro_queue_is_empty (encoder->frame_queue) && encoder->end_of_stream) {
    buffer = schro_encoder_encode_end_of_stream (encoder);
    schro_encoder_fixup_offsets (encoder, buffer, TRUE);
    encoder->end_of_stream_pulled = TRUE;

    schro_async_unlock (encoder->async);
    return buffer;
  }
  schro_async_unlock (encoder->async);

  SCHRO_DEBUG ("got nothing");
  return NULL;
}

/**
 * schro_encoder_end_of_stream:
 * @encoder: an encoder object
 *
 * Tells the encoder that the end of the stream has been reached, and
 * no more frames are available to encode.  The encoder will then
 * finish encoding.
 */
void
schro_encoder_end_of_stream (SchroEncoder * encoder)
{
  encoder->end_of_stream = TRUE;
  schro_async_lock (encoder->async);
  if (encoder->frame_queue->n > 0) {
    SchroEncoderFrame *encoder_frame;

    encoder_frame =
        encoder->frame_queue->elements[encoder->frame_queue->n - 1].data;
    encoder_frame->last_frame = TRUE;
  }
  schro_async_unlock (encoder->async);
}

static void
schro_encoder_fixup_offsets (SchroEncoder * encoder, SchroBuffer * buffer,
    schro_bool is_eos)
{
  uint8_t *data = buffer->data;
  unsigned int next_offset;

  if (buffer->length < 13) {
    SCHRO_ERROR ("packet too short (%d < 13)", buffer->length);
  }

  if (is_eos) {
    next_offset = 0;
  } else {
    next_offset = buffer->length;
  }

  data[5] = (next_offset >> 24) & 0xff;
  data[6] = (next_offset >> 16) & 0xff;
  data[7] = (next_offset >> 8) & 0xff;
  data[8] = (next_offset >> 0) & 0xff;
  data[9] = (encoder->prev_offset >> 24) & 0xff;
  data[10] = (encoder->prev_offset >> 16) & 0xff;
  data[11] = (encoder->prev_offset >> 8) & 0xff;
  data[12] = (encoder->prev_offset >> 0) & 0xff;

  encoder->prev_offset = next_offset;
}

static int
schro_encoder_encode_padding (SchroEncoder * encoder, int n)
{
  SchroBuffer *buffer;
  SchroPack *pack;

  if (n < SCHRO_PARSE_HEADER_SIZE)
    n = SCHRO_PARSE_HEADER_SIZE;

  buffer = schro_buffer_new_and_alloc (n);

  pack = schro_pack_new ();
  schro_pack_encode_init (pack, buffer);

  schro_encoder_encode_parse_info (pack, SCHRO_PARSE_CODE_PADDING);

  schro_pack_append_zero (pack, n - SCHRO_PARSE_HEADER_SIZE);

  schro_pack_free (pack);

  schro_encoder_insert_buffer (encoder, buffer);

  return n;
}

static void
schro_encoder_encode_codec_comment (SchroEncoder * encoder)
{
  const char *s = "Schroedinger " VERSION;
  SchroBuffer *buffer;

  buffer = schro_encoder_encode_auxiliary_data (encoder,
      SCHRO_AUX_DATA_ENCODER_STRING, s, strlen (s));

  schro_encoder_insert_buffer (encoder, buffer);
}

static void
schro_encoder_encode_bitrate_comment (SchroEncoder * encoder,
    unsigned int bitrate)
{
  uint8_t s[4];
  SchroBuffer *buffer;

  s[0] = (bitrate >> 24) & 0xff;
  s[1] = (bitrate >> 16) & 0xff;
  s[2] = (bitrate >> 8) & 0xff;
  s[3] = (bitrate >> 0) & 0xff;
  buffer = schro_encoder_encode_auxiliary_data (encoder,
      SCHRO_AUX_DATA_BITRATE, s, 4);

  schro_encoder_insert_buffer (encoder, buffer);
}

static void
schro_encoder_encode_md5_checksum (SchroEncoderFrame * frame)
{
  SchroBuffer *buffer;
  uint32_t checksum[4];

  schro_frame_md5 (frame->reconstructed_frame->frames[0], checksum);
  buffer = schro_encoder_encode_auxiliary_data (frame->encoder,
      SCHRO_AUX_DATA_MD5_CHECKSUM, checksum, 16);

  schro_encoder_frame_insert_buffer (frame, buffer);
}

/**
 * schro_encoder_insert_buffer:
 * @encoder: an encoder object
 * @buffer: a buffer
 *
 * Inserts an application-provided buffer into the encoded video stream
 * with the next frame that is pushed.
 */
void
schro_encoder_insert_buffer (SchroEncoder * encoder, SchroBuffer * buffer)
{
  schro_list_append (encoder->inserted_buffers, buffer);
}

/**
 * schro_encoder_frame_insert_buffer:
 * @frame: an encoder frame
 * @buffer: a buffer
 *
 * Inserts a buffer into an encoder frame.
 */
void
schro_encoder_frame_insert_buffer (SchroEncoderFrame * frame,
    SchroBuffer * buffer)
{
  schro_list_append (frame->inserted_buffers, buffer);
}

/**
 * schro_encoder_encode_auxiliary_data:
 * @encoder:
 * @id:
 * @data:
 * @size:
 *
 * Packs data into a Dirac auxiliary data packet.
 *
 * Returns: a buffer
 */
SchroBuffer *
schro_encoder_encode_auxiliary_data (SchroEncoder * encoder,
    SchroAuxiliaryDataID id, const void *data, int size)
{
  SchroPack *pack;
  SchroBuffer *buffer;

  buffer = schro_buffer_new_and_alloc (size + SCHRO_PARSE_HEADER_SIZE + 1);

  pack = schro_pack_new ();
  schro_pack_encode_init (pack, buffer);

  schro_encoder_encode_parse_info (pack, SCHRO_PARSE_CODE_AUXILIARY_DATA);
  schro_pack_encode_bits (pack, 8, id);
  schro_pack_append (pack, data, size);

  schro_pack_free (pack);

  return buffer;
}

/**
 * schro_encoder_encode_sequence_header:
 * @encoder: an encoder object
 *
 * Creates a buffer containing a sequence header.
 *
 * Returns: a buffer
 */
SchroBuffer *
schro_encoder_encode_sequence_header (SchroEncoder * encoder)
{
  SchroPack *pack;
  SchroBuffer *buffer;
  SchroBuffer *subbuffer;
  int next_offset;

  buffer = schro_buffer_new_and_alloc (0x100);

  pack = schro_pack_new ();
  schro_pack_encode_init (pack, buffer);

  schro_encoder_encode_sequence_header_header (encoder, pack);

  schro_pack_flush (pack);

  next_offset = schro_pack_get_offset (pack);
  buffer->data[5] = (next_offset >> 24) & 0xff;
  buffer->data[6] = (next_offset >> 16) & 0xff;
  buffer->data[7] = (next_offset >> 8) & 0xff;
  buffer->data[8] = (next_offset >> 0) & 0xff;

  subbuffer = schro_buffer_new_subbuffer (buffer, 0,
      schro_pack_get_offset (pack));
  schro_pack_free (pack);
  schro_buffer_unref (buffer);
  buffer = NULL;

  return subbuffer;
}

/**
 * schro_encoder_encode_end_of_stream:
 * @encoder:
 *
 * Creates an end-of-stream packet.
 *
 * Returns: a buffer
 */
SchroBuffer *
schro_encoder_encode_end_of_stream (SchroEncoder * encoder)
{
  SchroPack *pack;
  SchroBuffer *buffer;

  buffer = schro_buffer_new_and_alloc (SCHRO_PARSE_HEADER_SIZE);

  pack = schro_pack_new ();
  schro_pack_encode_init (pack, buffer);

  schro_encoder_encode_parse_info (pack, SCHRO_PARSE_CODE_END_OF_SEQUENCE);

  schro_pack_free (pack);

  return buffer;
}

/**
 * schro_encoder_wait:
 * @encoder: an encoder object
 *
 * Checks the state of the encoder.  If the encoder requires the
 * application to do something, an appropriate state code is returned.
 * Otherwise, this function waits until the encoder requires the
 * application to do something.
 *
 * Returns: a state code
 */
SchroStateEnum
schro_encoder_wait (SchroEncoder * encoder)
{
  SchroStateEnum ret = SCHRO_STATE_AGAIN;

  schro_async_lock (encoder->async);
  while (1) {
    if (schro_encoder_pull_is_ready_locked (encoder)) {
      SCHRO_DEBUG ("have buffer");
      ret = SCHRO_STATE_HAVE_BUFFER;
      break;
    }
    if (schro_encoder_push_is_ready_locked (encoder)) {
      SCHRO_DEBUG ("need frame");
      ret = SCHRO_STATE_NEED_FRAME;
      break;
    }
    if (schro_queue_is_empty (encoder->frame_queue)
        && encoder->end_of_stream_pulled) {
      ret = SCHRO_STATE_END_OF_STREAM;
      break;
    }

    SCHRO_DEBUG ("encoder waiting");
    ret = schro_async_wait_locked (encoder->async);
    if (!ret) {
      int i;

      SCHRO_WARNING ("deadlock?  kicking scheduler");
      for (i = 0; i < encoder->frame_queue->n; i++) {
        SchroEncoderFrame *frame = encoder->frame_queue->elements[i].data;
        SCHRO_WARNING ("%d: %d %d %d %d %04x", i, frame->frame_number,
            frame->picture_number_ref[0], frame->picture_number_ref[1],
            frame->busy, 0 /*frame->state */ );
      }
      for (i = 0; i < SCHRO_LIMIT_REFERENCE_FRAMES; i++) {
        SchroEncoderFrame *frame = encoder->reference_pictures[i];
        if (frame) {
          SCHRO_WARNING ("ref %d: %d %d %04x", i, frame->frame_number,
              frame->busy, 0 /*frame->state */ );
        } else {
          SCHRO_WARNING ("ref %d: NULL", i);
        }
      }
      //SCHRO_ASSERT(0);
      schro_async_signal_scheduler (encoder->async);
      ret = SCHRO_STATE_AGAIN;
      break;
    }
  }
  schro_async_unlock (encoder->async);

  return ret;
}

int
schro_encoder_frame_is_B_frame (SchroEncoderFrame * frame)
{
  int is_B_frame = 0;

  if (frame->num_refs == 2 &&
      ((frame->picture_number_ref[0] < frame->frame_number
              && frame->picture_number_ref[1] > frame->frame_number)
          || (frame->picture_number_ref[1] < frame->frame_number
              && frame->picture_number_ref[0] > frame->frame_number))) {
    is_B_frame = 1;
  }

  return is_B_frame;

}


static void
schro_encoder_frame_complete (SchroAsyncStage * stage)
{
  SchroEncoderFrame *frame = (SchroEncoderFrame *) stage->priv;
  int i;

  SCHRO_INFO ("completing task, picture %d working %02x in state %02x",
      frame->frame_number, frame->working, 0 /*frame->state */ );

  SCHRO_ASSERT (frame->busy == TRUE);

  frame->busy = FALSE;
  stage->is_done = TRUE;
  frame->working = 0;

  if (stage == frame->stages + SCHRO_ENCODER_FRAME_STAGE_POSTANALYSE) {
    frame->stages[SCHRO_ENCODER_FRAME_STAGE_DONE].is_done = TRUE;

    SCHRO_ASSERT (frame->output_buffer_size > 0);

    if (frame->previous_frame) {
      schro_encoder_frame_unref (frame->previous_frame);
      frame->previous_frame = NULL;
    }
    if (frame->motion) {
      schro_motion_free (frame->motion);
      frame->motion = NULL;
    }
    if (frame->me) {
      schro_motionest_free (frame->me);
      frame->me = NULL;
    }
    if (frame->ref_frame[0]) {
      schro_encoder_frame_unref (frame->ref_frame[0]);
      frame->ref_frame[0] = NULL;
    }
    if (frame->ref_frame[1]) {
      schro_encoder_frame_unref (frame->ref_frame[1]);
      frame->ref_frame[1] = NULL;
    }

    if (frame->deep_me) {
      schro_me_free (frame->deep_me);
      frame->deep_me = NULL;
    }

    for (i = 0; 2 > i; ++i) {
      if (frame->hier_bm[i]) {
        schro_hbm_unref (frame->hier_bm[i]);
        frame->hier_bm[i] = NULL;
      }
    }

    if (!frame->is_ref) {
      for (i = 0; i < 5; i++) {
        if (frame->downsampled_frames[i]) {
          schro_frame_unref (frame->downsampled_frames[i]);
          frame->downsampled_frames[i] = NULL;
        }
      }
    }

    if (frame->start_sequence_header) {
      frame->sequence_header_buffer =
          schro_encoder_encode_sequence_header (frame->encoder);
    }
    if (frame->last_frame) {
      frame->encoder->completed_eos = TRUE;
    }
  }
}

static void
schro_encoder_sc_detect_1 (SchroAsyncStage * stage)
{
  SchroEncoderFrame *frame = stage->priv;
  SchroFrameData *comp1;
  SchroFrameData *comp2;

  SCHRO_ASSERT (frame->stages[SCHRO_ENCODER_FRAME_STAGE_ANALYSE].is_done
      && frame->previous_frame
      && frame->previous_frame->
      stages[SCHRO_ENCODER_FRAME_STAGE_ANALYSE].is_done);

  comp1 = &frame->downsampled_frames[0]->components[0];
  comp2 = &frame->previous_frame->downsampled_frames[0]->components[0];

  frame->sc_mad =
      schro_metric_absdiff_u8 (comp1->data, comp1->stride, comp2->data,
      comp2->stride, comp1->width, comp1->height);
  frame->sc_mad /= (comp1->width * comp1->height);
  frame->have_mad = 1;
}


/**
 * run_stage:
 * @frame:
 * @state:
 *
 * Runs a stage in the encoding process.
 */
static void
run_stage (SchroEncoderFrame * frame, int stage)
{
  void *func;

  SCHRO_ASSERT (frame->stages[stage].is_done == FALSE);

  frame->busy = TRUE;
  frame->working = stage;
  switch (stage) {
    case SCHRO_ENCODER_FRAME_STAGE_ANALYSE:
      func = schro_encoder_analyse_picture;
      break;
    case SCHRO_ENCODER_FRAME_STAGE_PREDICT_ROUGH:
      func = schro_encoder_predict_rough_picture;
      break;
    case SCHRO_ENCODER_FRAME_STAGE_PREDICT_PEL:
      func = schro_encoder_predict_pel_picture;
      break;
    case SCHRO_ENCODER_FRAME_STAGE_PREDICT_SUBPEL:
      func = schro_encoder_predict_subpel_picture;
      break;
    case SCHRO_ENCODER_FRAME_STAGE_MODE_DECISION:
      func = schro_encoder_mode_decision;
      break;
    case SCHRO_ENCODER_FRAME_STAGE_ENCODING:
      func = schro_encoder_encode_picture;
      break;
    case SCHRO_ENCODER_FRAME_STAGE_RECONSTRUCT:
      func = schro_encoder_reconstruct_picture;
      break;
    case SCHRO_ENCODER_FRAME_STAGE_POSTANALYSE:
      func = schro_encoder_postanalyse_picture;
      break;
    case SCHRO_ENCODER_FRAME_STAGE_SC_DETECT_1:
      func = schro_encoder_sc_detect_1;
      break;

    default:
      SCHRO_ASSERT (0);
  }
  frame->stages[stage].task_func = func;
  frame->stages[stage].priv = frame;
  schro_async_run_stage_locked (frame->encoder->async, frame->stages + stage);
}

/**
 * check_refs:
 * @frame: encoder frame
 *
 * Checks whether reference pictures are available to be used for motion
 * rendering.
 */
static int
check_refs (SchroEncoderFrame * frame)
{
  if (frame->num_refs == 0)
    return TRUE;

  if (frame->num_refs > 0 &&
      !(frame->ref_frame[0]->stages[SCHRO_ENCODER_FRAME_STAGE_DONE].is_done)) {
    return FALSE;
  }
  if (frame->num_refs > 1 &&
      !(frame->ref_frame[1]->stages[SCHRO_ENCODER_FRAME_STAGE_DONE].is_done)) {
    return FALSE;
  }

  return TRUE;
}

static void
calculate_sc_score (SchroEncoder * encoder)
{
#define SC_THRESHOLD_RANGE 2
#define MAD_ARRAY_LEN SC_THRESHOLD_RANGE * 2 + 1
#define BIG_INT 0xffffffff
  int i, j, end;
  int mad_array[MAD_ARRAY_LEN];
  SchroEncoderFrame *frame, *f, *f2;

  /* calculate threshold first */
  for (i = SC_THRESHOLD_RANGE; encoder->frame_queue->n - SC_THRESHOLD_RANGE > i;
      ++i) {
    int mad_max = 0, mad_min = BIG_INT;

    frame = encoder->frame_queue->elements[i].data;
    if (!frame->need_mad)
      continue;
    if (frame->have_scene_change_score)
      continue;
    if (0 > frame->sc_threshold) {
      memset (mad_array, -1, sizeof (mad_array));
      mad_array[SC_THRESHOLD_RANGE] = frame->sc_mad;
      for (j = 0; SC_THRESHOLD_RANGE > j; ++j) {
        f = encoder->frame_queue->elements[i - j - 1].data;
        mad_array[SC_THRESHOLD_RANGE - j - 1] = f->sc_mad;
        f = encoder->frame_queue->elements[i + j + 1].data;
        mad_array[SC_THRESHOLD_RANGE + j + 1] = f->sc_mad;
      }
      for (j = 0; MAD_ARRAY_LEN > j; ++j) {
        if (mad_array[j] > mad_max)
          mad_max = mad_array[j];
        if (mad_array[j] < mad_min)
          mad_min = mad_array[j];
      }
      frame->sc_threshold = mad_max + 0.8 * (mad_max - mad_min);
    }
  }
  /* and now calculate the score */
  for (i = SC_THRESHOLD_RANGE + 3;
      encoder->frame_queue->n - SC_THRESHOLD_RANGE - 3 > i; ++i) {
    frame = encoder->frame_queue->elements[i].data;
    if (!frame->need_mad)
      continue;
    if (frame->have_scene_change_score)
      continue;
    f = encoder->frame_queue->elements[i - 3].data;
    f2 = encoder->frame_queue->elements[i + 3].data;
    frame->scene_change_score =
        frame->sc_mad - (f->sc_threshold + f2->sc_threshold) / 2;
    frame->have_scene_change_score = 1;
  }
  /* finally we update the state of the processed items */
  if (encoder->queue_depth > encoder->frame_queue->n && encoder->end_of_stream) {
    /* end of sequence */
    end = encoder->frame_queue->n;
  } else {
    end = encoder->frame_queue->n - SC_THRESHOLD_RANGE - 3;
  }
  for (i = 0; end > i; ++i) {
    f = encoder->frame_queue->elements[i].data;
    f->stages[SCHRO_ENCODER_FRAME_STAGE_SC_DETECT_2].is_done = TRUE;
    SCHRO_DEBUG ("calculate sc score for i=%d picture=%d state=%d", i,
        f->frame_number, 0 /* f->state */ );
  }
}

#undef BIG_INT


static int
schro_encoder_async_schedule (SchroEncoder * encoder,
    SchroExecDomain exec_domain)
{
  SchroEncoderFrame *frame;
  int i;
  int ref;

  SCHRO_INFO ("iterate %d", encoder->completed_eos);

  for (i = 0; i < encoder->frame_queue->n; i++) {
    frame = encoder->frame_queue->elements[i].data;
    SCHRO_DEBUG ("analyse i=%d picture=%d state=%d busy=%d", i,
        frame->frame_number, 0 /*frame->state */ , frame->busy);

    if (frame->busy)
      continue;

#define TODO(stage) (frame->stages[(stage)].is_needed && !frame->stages[(stage)].is_done)

    if (TODO (SCHRO_ENCODER_FRAME_STAGE_ANALYSE)) {
      encoder->init_frame (frame);
      init_params (frame);
      run_stage (frame, SCHRO_ENCODER_FRAME_STAGE_ANALYSE);
      return TRUE;
    }
  }

  if (encoder->enable_scene_change_detection) {
    /* calculate MAD for entire queue of frames */
    for (i = 0; encoder->frame_queue->n > i; ++i) {
      SchroEncoderFrame *prev_frame;

      frame = encoder->frame_queue->elements[i].data;
      SCHRO_DEBUG
          ("phase I of shot change detection for i=%d picture=%d state=%d", i,
          frame->frame_number, 0 /* frame->state */ );

      if (frame->busy)
        continue;

      if (0 == i || FALSE == frame->need_mad) {
        /* we can't calculate a MAD for first frame in the queue */
        frame->stages[SCHRO_ENCODER_FRAME_STAGE_SC_DETECT_1].is_done = TRUE;
        continue;
      }
      prev_frame = frame->previous_frame;

      if (!frame->stages[SCHRO_ENCODER_FRAME_STAGE_SC_DETECT_1].is_done
          && prev_frame
          && prev_frame->stages[SCHRO_ENCODER_FRAME_STAGE_ANALYSE].is_done
          && frame->stages[SCHRO_ENCODER_FRAME_STAGE_ANALYSE].is_done) {
        run_stage (frame, SCHRO_ENCODER_FRAME_STAGE_SC_DETECT_1);
        return TRUE;
      }
    }

    /* calculate new scene change score */
    for (i = 0; encoder->frame_queue->n > i; ++i) {
      frame = encoder->frame_queue->elements[i].data;
      if (!frame->stages[SCHRO_ENCODER_FRAME_STAGE_SC_DETECT_1].is_done) {
        return TRUE;
      }
    }
    calculate_sc_score (encoder);
  } else {
    for (i = 0; encoder->frame_queue->n > i; ++i) {
      frame = encoder->frame_queue->elements[i].data;
      if (frame->stages[SCHRO_ENCODER_FRAME_STAGE_ANALYSE].is_done) {
        frame->stages[SCHRO_ENCODER_FRAME_STAGE_SC_DETECT_1].is_done = TRUE;
        frame->stages[SCHRO_ENCODER_FRAME_STAGE_SC_DETECT_2].is_done = TRUE;
      }
    }
  }

  for (i = 0; i < encoder->frame_queue->n; i++) {
    frame = encoder->frame_queue->elements[i].data;
    if (frame->frame_number == encoder->gop_picture) {
      encoder->handle_gop (encoder, i);
      break;
    }
  }

  /* Reference pictures are higher priority, so we pass over the list
   * first for reference pictures, then for non-ref. */
  for (ref = 1; ref >= 0; ref--) {
    for (i = 0; i < encoder->frame_queue->n; i++) {
      frame = encoder->frame_queue->elements[i].data;
      SCHRO_DEBUG ("backref i=%d picture=%d state=%d busy=%d", i,
          frame->frame_number, 0 /*frame->state */ , frame->busy);

      if (frame->busy)
        continue;

      if (frame->is_ref != ref)
        continue;

      if (TODO (SCHRO_ENCODER_FRAME_STAGE_HAVE_PARAMS) &&
          frame->stages[SCHRO_ENCODER_FRAME_STAGE_HAVE_GOP].is_done) {

        if (encoder->setup_frame (frame)) {
          frame->stages[SCHRO_ENCODER_FRAME_STAGE_HAVE_PARAMS].is_done = TRUE;
        }
      }

      if (TODO (SCHRO_ENCODER_FRAME_STAGE_PREDICT_ROUGH) &&
          frame->stages[SCHRO_ENCODER_FRAME_STAGE_HAVE_PARAMS].is_done) {
        if (!check_refs (frame))
          continue;
        run_stage (frame, SCHRO_ENCODER_FRAME_STAGE_PREDICT_ROUGH);
        return TRUE;
      }

      if (TODO (SCHRO_ENCODER_FRAME_STAGE_PREDICT_PEL) &&
          frame->stages[SCHRO_ENCODER_FRAME_STAGE_PREDICT_ROUGH].is_done) {
        schro_frame_set_wavelet_params (frame);
        run_stage (frame, SCHRO_ENCODER_FRAME_STAGE_PREDICT_PEL);
        return TRUE;
      }

    }
  }

  for (ref = 1; ref >= 0; ref--) {
    for (i = 0; encoder->frame_queue->n > i; ++i) {
      frame = encoder->frame_queue->elements[i].data;
      SCHRO_DEBUG ("serialised stage i=%d picture=%d state=%d slot=%d quant_slot=%d", i, frame->frame_number, 0 /* frame>state */
          , frame->slot, encoder->quant_slot);

      if (frame->busy)
        continue;

      if (encoder->enable_deep_estimation) {
        if (frame->slot != encoder->quant_slot) {
          continue;
        }
      } else if (encoder->enable_bigblock_estimation) {
        if (frame->is_ref != ref)
          continue;
      }

      if (TODO (SCHRO_ENCODER_FRAME_STAGE_PREDICT_SUBPEL) &&
          frame->stages[SCHRO_ENCODER_FRAME_STAGE_PREDICT_PEL].is_done) {
        schro_encoder_set_frame_lambda (frame);
        run_stage (frame, SCHRO_ENCODER_FRAME_STAGE_PREDICT_SUBPEL);
        return TRUE;
      }

      if (TODO (SCHRO_ENCODER_FRAME_STAGE_MODE_DECISION) &&
          frame->stages[SCHRO_ENCODER_FRAME_STAGE_PREDICT_SUBPEL].is_done) {
        run_stage (frame, SCHRO_ENCODER_FRAME_STAGE_MODE_DECISION);
        return TRUE;
      }
    }
  }

  for (i = 0; i < encoder->frame_queue->n; i++) {
    frame = encoder->frame_queue->elements[i].data;
    if (frame->slot == encoder->quant_slot) {
      int ret;
      ret = encoder->handle_quants (encoder, i);
      if (!ret)
        break;
    }
  }

  for (ref = 1; ref >= 0; ref--) {
    for (i = 0; encoder->frame_queue->n > i; ++i) {
      frame = encoder->frame_queue->elements[i].data;

      if (frame->busy)
        continue;

      if (encoder->enable_deep_estimation) {
        if (frame->slot != encoder->quant_slot)
          continue;
      } else if (encoder->enable_bigblock_estimation) {
        if (frame->is_ref != ref)
          continue;
      }

      if (TODO (SCHRO_ENCODER_FRAME_STAGE_ENCODING) &&
          frame->stages[SCHRO_ENCODER_FRAME_STAGE_HAVE_QUANTS].is_done) {
        run_stage (frame, SCHRO_ENCODER_FRAME_STAGE_ENCODING);
        return TRUE;
      }
    }
  }

  for (ref = 1; ref >= 0; ref--) {
    for (i = 0; i < encoder->frame_queue->n; i++) {
      frame = encoder->frame_queue->elements[i].data;
      SCHRO_DEBUG ("backref i=%d picture=%d state=%d busy=%d", i,
          frame->frame_number, 0 /*frame->state */ , frame->busy);

      if (frame->busy)
        continue;
      if (frame->is_ref != ref)
        continue;

      if (TODO (SCHRO_ENCODER_FRAME_STAGE_RECONSTRUCT) &&
          frame->stages[SCHRO_ENCODER_FRAME_STAGE_ENCODING].is_done) {
        run_stage (frame, SCHRO_ENCODER_FRAME_STAGE_RECONSTRUCT);
        return TRUE;
      }
      if (TODO (SCHRO_ENCODER_FRAME_STAGE_POSTANALYSE) &&
          frame->stages[SCHRO_ENCODER_FRAME_STAGE_RECONSTRUCT].is_done) {
        run_stage (frame, SCHRO_ENCODER_FRAME_STAGE_POSTANALYSE);
        return TRUE;
      }
    }
  }

  return FALSE;
}


void
schro_encoder_analyse_picture (SchroAsyncStage * stage)
{
  SchroEncoderFrame *frame = (SchroEncoderFrame *) stage->priv;

  if (frame->encoder->filtering != 0 || frame->need_extension) {
    if (frame->encoder->enable_deep_estimation) {
      frame->filtered_frame =
          schro_frame_dup_extended (frame->original_frame,
          MAX (frame->params.xbsep_luma * 4, frame->params.ybsep_luma * 4));
    } else if (frame->encoder->enable_bigblock_estimation) {
      frame->filtered_frame = schro_frame_dup_extended (frame->original_frame,
          32);
    } else
      SCHRO_ASSERT (0);
    switch (frame->encoder->filtering) {
      case 1:
        schro_frame_filter_cwmN (frame->filtered_frame,
            frame->encoder->filter_value);
        break;
      case 2:
        schro_frame_filter_lowpass2 (frame->filtered_frame,
            frame->encoder->filter_value);
        break;
      case 3:
        schro_frame_filter_addnoise (frame->filtered_frame,
            frame->encoder->filter_value);
        break;
      case 4:
        schro_frame_filter_adaptive_lowpass (frame->filtered_frame);
        break;
      case 5:
        schro_frame_filter_lowpass (frame->filtered_frame,
            frame->encoder->filter_value);
        break;
      default:
        /* do nothing */
        break;
    }
    schro_frame_mc_edgeextend (frame->filtered_frame);
  } else {
    frame->filtered_frame = schro_frame_ref (frame->original_frame);
  }

  if (frame->need_downsampling) {
    schro_encoder_frame_downsample (frame);
    frame->have_downsampling = TRUE;
  }
  schro_frame_ref (frame->filtered_frame);
  frame->upsampled_original_frame =
      schro_upsampled_frame_new (frame->filtered_frame);
  if (frame->need_upsampling) {
    schro_upsampled_frame_upsample (frame->upsampled_original_frame);
    frame->have_upsampling = TRUE;
  }

  if (frame->need_average_luma) {
    if (frame->have_downsampling) {
      frame->average_luma =
          schro_frame_calculate_average_luma (frame->
          downsampled_frames[frame->encoder->downsample_levels - 1]);
    } else {
      frame->average_luma =
          schro_frame_calculate_average_luma (frame->filtered_frame);
    }
    frame->have_average_luma = TRUE;
  }
}

void
schro_encoder_predict_rough_picture (SchroAsyncStage * stage)
{
  SchroEncoderFrame *frame = (SchroEncoderFrame *) stage->priv;

  SCHRO_INFO ("predict picture %d", frame->frame_number);

  if (frame->params.num_refs > 0) {
    schro_encoder_motion_predict_rough (frame);
  }
}

/* should perform fullpel ME without "rendering",
 * ie without mode decision and motion compensation and DWT */
void
schro_encoder_predict_pel_picture (SchroAsyncStage * stage)
{
  SchroEncoderFrame *frame = (SchroEncoderFrame *) stage->priv;

  SCHRO_ASSERT (frame
      && frame->stages[SCHRO_ENCODER_FRAME_STAGE_PREDICT_ROUGH].is_done);
  SCHRO_INFO ("fullpel predict picture %d", frame->frame_number);

  if (frame->params.num_refs > 0) {
    schro_encoder_motion_predict_pel (frame);
  }
}

void
schro_encoder_predict_subpel_picture (SchroAsyncStage * stage)
{
  SchroEncoderFrame *frame = (SchroEncoderFrame *) stage->priv;

  if (frame->encoder->enable_bigblock_estimation) {
    if (frame->params.num_refs > 0 && frame->params.mv_precision > 0) {
      schro_encoder_motion_predict_subpel (frame);
    }
  } else if (frame->encoder->enable_deep_estimation) {
    int ref, xnum_blocks, ynum_blocks;
    SchroMotionField *mf_dest, *mf_src;

    if (frame->params.num_refs > 0) {
      xnum_blocks = frame->params.x_num_blocks;
      ynum_blocks = frame->params.y_num_blocks;
      for (ref = 0; frame->params.num_refs > ref; ++ref) {
        mf_dest = schro_motion_field_new (xnum_blocks, ynum_blocks);
        mf_src = schro_hbm_motion_field (frame->hier_bm[ref], 0);
        memcpy (mf_dest->motion_vectors, mf_src->motion_vectors,
            xnum_blocks * ynum_blocks * sizeof (SchroMotionVector));
        schro_me_set_subpel_mf (frame->deep_me, mf_dest, ref);
      }
    }
    if (frame->params.num_refs > 0 && frame->params.mv_precision > 0) {
      schro_me_set_lambda (frame->deep_me, frame->frame_me_lambda);
      schro_encoder_motion_predict_subpel_deep (frame->deep_me);
    }
  }
}

/* performs mode decision and superblock splitting
 * finally it does DWT */
void
schro_encoder_mode_decision (SchroAsyncStage * stage)
{
  SchroEncoderFrame *frame = (SchroEncoderFrame *) stage->priv;

  SCHRO_ASSERT (frame->stages[SCHRO_ENCODER_FRAME_STAGE_PREDICT_PEL].is_done);
  SCHRO_INFO ("mode decision and superblock splitting picture %d",
      frame->frame_number);

  if (frame->encoder->enable_deep_estimation) {
    SchroMotionField *mf, *mf_src;
    int xnum_blocks, ynum_blocks, ref;
    if (frame->params.num_refs > 0) {
      xnum_blocks = frame->params.x_num_blocks;
      ynum_blocks = frame->params.y_num_blocks;
      for (ref = 0; frame->params.num_refs > ref; ++ref) {
        mf = schro_motion_field_new (xnum_blocks, ynum_blocks);
        schro_motion_field_set (mf, 2, ref + 1);
        /* copy split2 from subpel */
        mf_src = schro_me_subpel_mf (frame->deep_me, ref);
        SCHRO_ASSERT (mf_src);
        memcpy (mf->motion_vectors, mf_src->motion_vectors,
            xnum_blocks * ynum_blocks * sizeof (SchroMotionVector));
        schro_me_set_split2_mf (frame->deep_me, mf, ref);

        mf = schro_motion_field_new (xnum_blocks, ynum_blocks);
        schro_motion_field_set (mf, 1, ref + 1);
        schro_me_set_split1_mf (frame->deep_me, mf, ref);

        mf = schro_motion_field_new (xnum_blocks, ynum_blocks);
        schro_motion_field_set (mf, 0, ref + 1);
        schro_me_set_split0_mf (frame->deep_me, mf, ref);
      }
      SCHRO_INFO ("mode decision and superblock splitting picture %d",
          frame->frame_number);
      schro_me_set_motion (frame->deep_me, frame->motion);
      schro_me_set_lambda (frame->deep_me, frame->frame_me_lambda);
      schro_mode_decision (frame->deep_me);
      schro_motion_calculate_stats (frame->motion, frame);
      frame->estimated_mc_bits = schro_motion_estimate_entropy (frame->motion);
      frame->badblock_ratio = schro_me_badblocks_ratio (frame->deep_me);
      frame->dcblock_ratio = schro_me_dcblock_ratio (frame->deep_me);
      frame->mc_error = schro_me_mc_error (frame->deep_me);

      SCHRO_DEBUG ("DC block ratio for frame %d s %g", frame->frame_number,
          frame->dcblock_ratio);

      if (frame->dcblock_ratio > frame->encoder->magic_me_bailout_limit) {
        if (frame->deep_me) {
          /* Free the motion estimation info if we are inserting an I frame */
          schro_me_free (frame->deep_me);
          frame->deep_me = NULL;
        }
        frame->params.num_refs = 0;
        frame->num_refs = 0;
        SCHRO_WARNING
            ("DC block ratio too high for frame %d, inserting an intra  picture",
            frame->frame_number);
      }
    }
  }

  schro_encoder_render_picture (frame);
}


void
schro_encoder_render_picture (SchroEncoderFrame * frame)
{
  SCHRO_INFO ("render picture %d", frame->frame_number);

  if (frame->params.num_refs > 0) {
    frame->motion->src1 = frame->ref_frame[0]->reconstructed_frame;
    if (frame->params.num_refs > 1) {
      frame->motion->src2 = frame->ref_frame[1]->reconstructed_frame;
    }

    SCHRO_ASSERT (schro_motion_verify (frame->motion));

  }

  if (frame->params.num_refs > 0) {
    schro_frame_convert (frame->iwt_frame, frame->filtered_frame);

    schro_motion_render (frame->motion, frame->prediction_frame,
        frame->iwt_frame, FALSE, NULL);

    schro_frame_zero_extend (frame->iwt_frame,
        frame->params.video_format->width,
        schro_video_format_get_picture_height (frame->params.video_format));
  } else {
    schro_frame_convert (frame->iwt_frame, frame->filtered_frame);
  }

  schro_frame_iwt_transform (frame->iwt_frame, &frame->params);

  schro_encoder_clean_up_transform (frame);
}

void
schro_encoder_encode_picture (SchroAsyncStage * stage)
{
  SchroEncoderFrame *frame = (SchroEncoderFrame *) stage->priv;
  SchroBuffer *subbuffer;
  int picture_chroma_width, picture_chroma_height;
  int width, height;
  int total_frame_bits;
  double factor;
  double est_subband_bits;

  SCHRO_INFO ("encode picture %d", frame->frame_number);

  frame->output_buffer = schro_buffer_new_and_alloc (frame->output_buffer_size);

  schro_video_format_get_iwt_alloc_size (&frame->encoder->video_format,
      &width, &height, frame->encoder->transform_depth);
  frame->subband_size = width * height / 4;
  frame->subband_buffer =
      schro_buffer_new_and_alloc (frame->subband_size * sizeof (int16_t));

  schro_video_format_get_picture_chroma_size (&frame->encoder->video_format,
      &picture_chroma_width, &picture_chroma_height);

  /* FIXME this is only used for lowdelay, and is way too big */
  frame->quant_data = schro_malloc (sizeof (int16_t) * frame->subband_size);

  frame->pack = schro_pack_new ();
  schro_pack_encode_init (frame->pack, frame->output_buffer);

  total_frame_bits = -schro_pack_get_offset (frame->pack) * 8;

  /* encode header */
  schro_encoder_encode_parse_info (frame->pack,
      SCHRO_PARSE_CODE_PICTURE (frame->is_ref, frame->params.num_refs,
          frame->params.is_lowdelay, frame->params.is_noarith));
  schro_encoder_encode_picture_header (frame);

  if (frame->params.num_refs > 0) {
    schro_pack_sync (frame->pack);
    schro_encoder_encode_picture_prediction_parameters (frame);
    schro_pack_sync (frame->pack);
    frame->actual_mc_bits = -schro_pack_get_offset (frame->pack) * 8;
    schro_encoder_encode_superblock_split (frame);
    schro_encoder_encode_prediction_modes (frame);
    schro_encoder_encode_vector_data (frame, 0, 0);
    schro_encoder_encode_vector_data (frame, 0, 1);
    if (frame->params.num_refs > 1) {
      schro_encoder_encode_vector_data (frame, 1, 0);
      schro_encoder_encode_vector_data (frame, 1, 1);
    }
    schro_encoder_encode_dc_data (frame, 0);
    schro_encoder_encode_dc_data (frame, 1);
    schro_encoder_encode_dc_data (frame, 2);
    frame->actual_mc_bits += schro_pack_get_offset (frame->pack) * 8;
  }

  if (schro_pack_get_offset (frame->pack) * 8 > frame->hard_limit_bits) {
    SCHRO_ERROR ("over hard_limit_bits after MC (%d>%d)",
        schro_pack_get_offset (frame->pack) * 8, frame->hard_limit_bits);
  }

  schro_pack_sync (frame->pack);
  schro_encoder_encode_transform_parameters (frame);

  if (schro_pack_get_offset (frame->pack) * 8 > frame->hard_limit_bits) {
    SCHRO_ERROR ("over hard_limit_bits after transform params (%d>%d)",
        schro_pack_get_offset (frame->pack) * 8, frame->hard_limit_bits);
  }

  frame->actual_residual_bits = -schro_pack_get_offset (frame->pack) * 8;

  schro_pack_sync (frame->pack);
  if (frame->params.is_lowdelay) {
    schro_encoder_encode_lowdelay_transform_data (frame);
  } else {
    schro_encoder_encode_transform_data (frame);
  }

  schro_pack_flush (frame->pack);
  frame->actual_residual_bits += schro_pack_get_offset (frame->pack) * 8;
  total_frame_bits += schro_pack_get_offset (frame->pack) * 8;

  SCHRO_DEBUG ("Actual frame %d residual bits : %d", frame->frame_number,
      frame->actual_residual_bits);
  // Update the fiddle factors for estimating entropy
  if (frame->num_refs == 0) {
    int component, b;
    for (component = 0; component < 3; ++component) {
      for (b = 0; b < 1 + 3 * frame->params.transform_depth; ++b) {
        est_subband_bits =
            frame->
            est_entropy[component][b][frame->quant_indices[component][b][0]];
        SCHRO_DEBUG ("Actual versus estimated band bits : %d %d %g %g",
            component, b, frame->actual_subband_bits[component][b],
            est_subband_bits);
        if (est_subband_bits > 200.0) {
          double ratio;
          factor = frame->actual_subband_bits[component][b] / est_subband_bits;
          ratio =
              frame->encoder->average_arith_context_ratios_intra[component][b];
          ratio = ratio * 0.9 + factor * 0.1;
          frame->encoder->average_arith_context_ratios_intra[component][b] =
              ratio;
        }
      }
    }
  } else {
    int component, b;
    for (component = 0; component < 3; ++component) {
      for (b = 0; b < 1 + 3 * frame->params.transform_depth; ++b) {
        est_subband_bits =
            frame->
            est_entropy[component][b][frame->quant_indices[component][b][0]];
        SCHRO_DEBUG ("Actual versus estimated band bits : %d %d %g %g",
            component, b, frame->actual_subband_bits[component][b],
            est_subband_bits);
        if (est_subband_bits > 200.0) {
          double ratio;
          factor = frame->actual_subband_bits[component][b] / est_subband_bits;
          ratio =
              frame->encoder->average_arith_context_ratios_inter[component][b];
          ratio = ratio * 0.9 + factor * 0.1;
          frame->encoder->average_arith_context_ratios_inter[component][b] =
              ratio;
        }
      }
    }
  }

  // Update the buffer model
  if (frame->encoder->rate_control ==
      SCHRO_ENCODER_RATE_CONTROL_CONSTANT_BITRATE) {
    SchroEncoder *encoder = frame->encoder;
    encoder->buffer_level -= total_frame_bits;
    encoder->buffer_level += encoder->bits_per_picture;

    if (encoder->buffer_level < 0) {
      SCHRO_DEBUG ("buffer underrun by %d bytes", -encoder->buffer_level);
      encoder->buffer_level = 0;
    }

    if (encoder->buffer_level > encoder->buffer_size) {
      int n;

      n = (encoder->buffer_level - encoder->buffer_size + 7) / 8;
      SCHRO_DEBUG ("buffer overrun, adding padding of %d bytes", n);
      n = schro_encoder_encode_padding (encoder, n);
      encoder->buffer_level -= n * 8;
    }
    SCHRO_DEBUG ("At frame %d, buffer level %d of %d bits", frame->frame_number,
        encoder->buffer_level, encoder->buffer_size);

    schro_encoder_cbr_update (frame, total_frame_bits);
  }

  if (schro_pack_get_offset (frame->pack) * 8 > frame->hard_limit_bits) {
    SCHRO_ERROR ("over hard_limit_bits after residual (%d>%d)",
        schro_pack_get_offset (frame->pack) * 8, frame->hard_limit_bits);
  }

  subbuffer = schro_buffer_new_subbuffer (frame->output_buffer, 0,
      schro_pack_get_offset (frame->pack));
  schro_buffer_unref (frame->output_buffer);
  frame->output_buffer = subbuffer;

  if (frame->subband_buffer) {
    schro_buffer_unref (frame->subband_buffer);
  }
  if (frame->quant_data) {
    free (frame->quant_data);
  }
  if (frame->quant_frame) {
    schro_frame_unref (frame->quant_frame);
  }
  if (frame->pack) {
    schro_pack_free (frame->pack);
    frame->pack = NULL;
  }

  frame->encoder->quant_slot++;
}

void
schro_encoder_reconstruct_picture (SchroAsyncStage * stage)
{
  SchroEncoderFrame *encoder_frame = (SchroEncoderFrame *) stage->priv;
  SchroFrameFormat frame_format;
  SchroFrame *frame;

  schro_frame_inverse_iwt_transform (encoder_frame->iwt_frame,
      &encoder_frame->params);

  if (encoder_frame->params.num_refs > 0) {
    schro_frame_add (encoder_frame->iwt_frame, encoder_frame->prediction_frame);
  }

  frame_format = schro_params_get_frame_format (8,
      encoder_frame->encoder->video_format.chroma_format);
  frame = schro_frame_new_and_alloc_extended (NULL, frame_format,
      encoder_frame->encoder->video_format.width,
      schro_video_format_get_picture_height (&encoder_frame->
          encoder->video_format), 32);
  schro_frame_convert (frame, encoder_frame->iwt_frame);
  schro_frame_mc_edgeextend (frame);
  encoder_frame->reconstructed_frame = schro_upsampled_frame_new (frame);

  if (encoder_frame->encoder->enable_md5) {
    schro_encoder_encode_md5_checksum (encoder_frame);
  }

  if (encoder_frame->is_ref && encoder_frame->encoder->mv_precision > 0) {
    schro_upsampled_frame_upsample (encoder_frame->reconstructed_frame);
  }
}

void
schro_encoder_postanalyse_picture (SchroAsyncStage * stage)
{
  SchroEncoderFrame *frame = (SchroEncoderFrame *) stage->priv;
  SchroVideoFormat *video_format = frame->params.video_format;

  if (frame->encoder->enable_psnr) {
    double mse[3];

    schro_frame_mean_squared_error (frame->filtered_frame,
        frame->reconstructed_frame->frames[0], mse);

    frame->mean_squared_error_luma = mse[0] /
        (video_format->luma_excursion * video_format->luma_excursion);
    frame->mean_squared_error_chroma = 0.5 * (mse[1] + mse[2]) /
        (video_format->chroma_excursion * video_format->chroma_excursion);
  }

  if (frame->encoder->enable_ssim) {
    frame->mssim = schro_frame_ssim (frame->original_frame,
        frame->reconstructed_frame->frames[0]);
    schro_dump (SCHRO_DUMP_SSIM, "%d %g\n", frame->frame_number, frame->mssim);
  }
}

static void
schro_encoder_encode_picture_prediction_parameters (SchroEncoderFrame * frame)
{
  SchroParams *params = &frame->params;
  int index;

  /* block parameters */
  index = schro_params_get_block_params (params);
  schro_pack_encode_uint (frame->pack, index);
  if (index == 0) {
    schro_pack_encode_uint (frame->pack, params->xblen_luma);
    schro_pack_encode_uint (frame->pack, params->yblen_luma);
    schro_pack_encode_uint (frame->pack, params->xbsep_luma);
    schro_pack_encode_uint (frame->pack, params->xbsep_luma);
  }

  MARKER (frame->pack);

  /* mv precision */
  schro_pack_encode_uint (frame->pack, params->mv_precision);

  MARKER (frame->pack);

  /* global motion flag */
  schro_pack_encode_bit (frame->pack, params->have_global_motion);
  if (params->have_global_motion) {
    int i;
    for (i = 0; i < params->num_refs; i++) {
      SchroGlobalMotion *gm = params->global_motion + i;

      /* pan tilt */
      if (gm->b0 == 0 && gm->b1 == 0) {
        schro_pack_encode_bit (frame->pack, 0);
      } else {
        schro_pack_encode_bit (frame->pack, 1);
        schro_pack_encode_sint (frame->pack, gm->b0);
        schro_pack_encode_sint (frame->pack, gm->b1);
      }

      /* zoom rotate shear */
      if (gm->a_exp == 0 && gm->a00 == 1 && gm->a01 == 0 && gm->a10 == 0 &&
          gm->a11 == 1) {
        schro_pack_encode_bit (frame->pack, 0);
      } else {
        schro_pack_encode_bit (frame->pack, 1);
        schro_pack_encode_uint (frame->pack, gm->a_exp);
        schro_pack_encode_sint (frame->pack, gm->a00);
        schro_pack_encode_sint (frame->pack, gm->a01);
        schro_pack_encode_sint (frame->pack, gm->a10);
        schro_pack_encode_sint (frame->pack, gm->a11);
      }

      /* perspective */
      if (gm->c_exp == 0 && gm->c0 == 0 && gm->c1 == 0) {
        schro_pack_encode_bit (frame->pack, 0);
      } else {
        schro_pack_encode_bit (frame->pack, 1);
        schro_pack_encode_uint (frame->pack, gm->c_exp);
        schro_pack_encode_sint (frame->pack, gm->c0);
        schro_pack_encode_sint (frame->pack, gm->c1);
      }
    }
  }

  MARKER (frame->pack);

  /* picture prediction mode flag */
  schro_pack_encode_uint (frame->pack, params->picture_pred_mode);

  /* non-default weights flag */
  SCHRO_ASSERT (params->picture_weight_2 == 1 || params->num_refs == 2);
  if (params->picture_weight_bits == 1 &&
      params->picture_weight_1 == 1 && params->picture_weight_2 == 1) {
    schro_pack_encode_bit (frame->pack, FALSE);
  } else {
    schro_pack_encode_bit (frame->pack, TRUE);
    schro_pack_encode_uint (frame->pack, params->picture_weight_bits);
    schro_pack_encode_sint (frame->pack, params->picture_weight_1);
    if (params->num_refs > 1) {
      schro_pack_encode_sint (frame->pack, params->picture_weight_2);
    }
  }

  MARKER (frame->pack);

}

static void
schro_encoder_encode_superblock_split (SchroEncoderFrame * frame)
{
  SchroParams *params = &frame->params;
  int i, j;
  SchroArith *arith = NULL;
  SchroPack b, *pack = &b;

  if (params->is_noarith) {
    schro_pack_encode_init (pack, frame->subband_buffer);
  } else {
    arith = schro_arith_new ();
    schro_arith_encode_init (arith, frame->subband_buffer);
  }

  for (j = 0; j < params->y_num_blocks; j += 4) {
    for (i = 0; i < params->x_num_blocks; i += 4) {
      int split_prediction;
      int split_residual;
      SchroMotionVector *mv =
          &frame->motion->motion_vectors[j * params->x_num_blocks + i];

      SCHRO_ASSERT (mv->split < 3);

      split_prediction = schro_motion_split_prediction (frame->motion, i, j);
      split_residual = (mv->split - split_prediction + 3) % 3;
      if (params->is_noarith) {
        schro_pack_encode_uint (pack, split_residual);
      } else {
        _schro_arith_encode_uint (arith, SCHRO_CTX_SB_F1,
            SCHRO_CTX_SB_DATA, split_residual);
      }
    }
  }

  schro_pack_sync (frame->pack);
  if (params->is_noarith) {
    schro_pack_flush (pack);
    schro_pack_encode_uint (frame->pack, schro_pack_get_offset (pack));
    schro_pack_sync (frame->pack);
    schro_pack_append (frame->pack, pack->buffer->data,
        schro_pack_get_offset (pack));
  } else {
    schro_arith_flush (arith);
    schro_pack_encode_uint (frame->pack, arith->offset);
    schro_pack_sync (frame->pack);
    schro_pack_append (frame->pack, arith->buffer->data, arith->offset);
    schro_arith_free (arith);
  }
}

static void
schro_encoder_encode_prediction_modes (SchroEncoderFrame * frame)
{
  SchroParams *params = &frame->params;
  int i, j;
  SchroArith *arith = NULL;
  SchroPack b, *pack = &b;

  if (params->is_noarith) {
    schro_pack_encode_init (pack, frame->subband_buffer);
  } else {
    arith = schro_arith_new ();
    schro_arith_encode_init (arith, frame->subband_buffer);
  }

  for (j = 0; j < params->y_num_blocks; j += 4) {
    for (i = 0; i < params->x_num_blocks; i += 4) {
      int k, l;
      SchroMotionVector *mv1 =
          &frame->motion->motion_vectors[j * params->x_num_blocks + i];

      for (l = 0; l < 4; l += (4 >> mv1->split)) {
        for (k = 0; k < 4; k += (4 >> mv1->split)) {
          SchroMotionVector *mv =
              &frame->motion->motion_vectors[(j + l) * params->x_num_blocks +
              i + k];
          int pred_mode;

          pred_mode = mv->pred_mode ^
              schro_motion_get_mode_prediction (frame->motion, i + k, j + l);

          if (params->is_noarith) {
            schro_pack_encode_bit (pack, pred_mode & 1);
          } else {
            _schro_arith_encode_bit (arith, SCHRO_CTX_BLOCK_MODE_REF1,
                pred_mode & 1);
          }
          if (params->num_refs > 1) {
            if (params->is_noarith) {
              schro_pack_encode_bit (pack, pred_mode >> 1);
            } else {
              _schro_arith_encode_bit (arith, SCHRO_CTX_BLOCK_MODE_REF2,
                  pred_mode >> 1);
            }
          }
          if (mv->pred_mode != 0) {
            if (params->have_global_motion) {
              int pred;
              pred = schro_motion_get_global_prediction (frame->motion,
                  i + k, j + l);
              if (params->is_noarith) {
                schro_pack_encode_bit (pack, mv->using_global ^ pred);
              } else {
                _schro_arith_encode_bit (arith, SCHRO_CTX_GLOBAL_BLOCK,
                    mv->using_global ^ pred);
              }
            } else {
              SCHRO_ASSERT (mv->using_global == FALSE);
            }
          }
        }
      }
    }
  }

  schro_pack_sync (frame->pack);
  if (params->is_noarith) {
    schro_pack_flush (pack);
    schro_pack_encode_uint (frame->pack, schro_pack_get_offset (pack));
    schro_pack_sync (frame->pack);
    schro_pack_append (frame->pack, pack->buffer->data,
        schro_pack_get_offset (pack));
  } else {
    schro_arith_flush (arith);
    schro_pack_encode_uint (frame->pack, arith->offset);
    schro_pack_sync (frame->pack);
    schro_pack_append (frame->pack, arith->buffer->data, arith->offset);
    schro_arith_free (arith);
  }
}

static void
schro_encoder_encode_vector_data (SchroEncoderFrame * frame, int ref, int xy)
{
  SchroParams *params = &frame->params;
  int i, j;
  SchroArith *arith = NULL;
  int cont, value, sign;
  SchroPack b, *pack = &b;

  if (params->is_noarith) {
    schro_pack_encode_init (pack, frame->subband_buffer);
  } else {
    arith = schro_arith_new ();
    schro_arith_encode_init (arith, frame->subband_buffer);
  }

  if (xy == 0) {
    if (ref == 0) {
      cont = SCHRO_CTX_MV_REF1_H_CONT_BIN1;
      value = SCHRO_CTX_MV_REF1_H_VALUE;
      sign = SCHRO_CTX_MV_REF1_H_SIGN;
    } else {
      cont = SCHRO_CTX_MV_REF2_H_CONT_BIN1;
      value = SCHRO_CTX_MV_REF2_H_VALUE;
      sign = SCHRO_CTX_MV_REF2_H_SIGN;
    }
  } else {
    if (ref == 0) {
      cont = SCHRO_CTX_MV_REF1_V_CONT_BIN1;
      value = SCHRO_CTX_MV_REF1_V_VALUE;
      sign = SCHRO_CTX_MV_REF1_V_SIGN;
    } else {
      cont = SCHRO_CTX_MV_REF2_V_CONT_BIN1;
      value = SCHRO_CTX_MV_REF2_V_VALUE;
      sign = SCHRO_CTX_MV_REF2_V_SIGN;
    }
  }

  for (j = 0; j < params->y_num_blocks; j += 4) {
    for (i = 0; i < params->x_num_blocks; i += 4) {
      int k, l;
      SchroMotionVector *mv1 =
          &frame->motion->motion_vectors[j * params->x_num_blocks + i];

      for (l = 0; l < 4; l += (4 >> mv1->split)) {
        for (k = 0; k < 4; k += (4 >> mv1->split)) {
          int pred_x, pred_y;
          int x, y;
          SchroMotionVector *mv =
              &frame->motion->motion_vectors[(j + l) * params->x_num_blocks +
              i + k];

          if ((mv->pred_mode & (1 << ref)) && !mv->using_global) {
            schro_motion_vector_prediction (frame->motion,
                i + k, j + l, &pred_x, &pred_y, 1 << ref);
            x = mv->u.vec.dx[ref];
            y = mv->u.vec.dy[ref];

            if (!params->is_noarith) {
              if (xy == 0) {
                _schro_arith_encode_sint (arith, cont, value, sign, x - pred_x);
              } else {
                _schro_arith_encode_sint (arith, cont, value, sign, y - pred_y);
              }
            } else {
              if (xy == 0) {
                schro_pack_encode_sint (pack, x - pred_x);
              } else {
                schro_pack_encode_sint (pack, y - pred_y);
              }
            }
          }
        }
      }
    }
  }

  schro_pack_sync (frame->pack);
  if (params->is_noarith) {
    schro_pack_flush (pack);
    schro_pack_encode_uint (frame->pack, schro_pack_get_offset (pack));
    schro_pack_sync (frame->pack);
    schro_pack_append (frame->pack, pack->buffer->data,
        schro_pack_get_offset (pack));
  } else {
    schro_arith_flush (arith);
    schro_pack_encode_uint (frame->pack, arith->offset);
    schro_pack_sync (frame->pack);
    schro_pack_append (frame->pack, arith->buffer->data, arith->offset);
    schro_arith_free (arith);
  }
}

static void
schro_encoder_encode_dc_data (SchroEncoderFrame * frame, int comp)
{
  SchroParams *params = &frame->params;
  int i, j;
  SchroArith *arith = NULL;
  SchroPack b, *pack = &b;

  if (params->is_noarith) {
    schro_pack_encode_init (pack, frame->subband_buffer);
  } else {
    arith = schro_arith_new ();
    schro_arith_encode_init (arith, frame->subband_buffer);
  }

  for (j = 0; j < params->y_num_blocks; j += 4) {
    for (i = 0; i < params->x_num_blocks; i += 4) {
      int k, l;
      SchroMotionVector *mv1 =
          &frame->motion->motion_vectors[j * params->x_num_blocks + i];

      for (l = 0; l < 4; l += (4 >> mv1->split)) {
        for (k = 0; k < 4; k += (4 >> mv1->split)) {
          SchroMotionVector *mv =
              &frame->motion->motion_vectors[(j + l) * params->x_num_blocks +
              i + k];

          if (mv->pred_mode == 0) {
            int pred[3];

            schro_motion_dc_prediction (frame->motion, i + k, j + l, pred);

            if (!params->is_noarith) {
              switch (comp) {
                case 0:
                  _schro_arith_encode_sint (arith,
                      SCHRO_CTX_LUMA_DC_CONT_BIN1, SCHRO_CTX_LUMA_DC_VALUE,
                      SCHRO_CTX_LUMA_DC_SIGN, mv->u.dc.dc[0] - pred[0]);
                  break;
                case 1:
                  _schro_arith_encode_sint (arith,
                      SCHRO_CTX_CHROMA1_DC_CONT_BIN1,
                      SCHRO_CTX_CHROMA1_DC_VALUE, SCHRO_CTX_CHROMA1_DC_SIGN,
                      mv->u.dc.dc[1] - pred[1]);
                  break;
                case 2:
                  _schro_arith_encode_sint (arith,
                      SCHRO_CTX_CHROMA2_DC_CONT_BIN1,
                      SCHRO_CTX_CHROMA2_DC_VALUE, SCHRO_CTX_CHROMA2_DC_SIGN,
                      mv->u.dc.dc[2] - pred[2]);
                  break;
                default:
                  SCHRO_ASSERT (0);
              }
            } else {
              schro_pack_encode_sint (pack, mv->u.dc.dc[comp] - pred[comp]);
            }
          }
        }
      }
    }
  }

  schro_pack_sync (frame->pack);
  if (params->is_noarith) {
    schro_pack_flush (pack);
    schro_pack_encode_uint (frame->pack, schro_pack_get_offset (pack));
    schro_pack_sync (frame->pack);
    schro_pack_append (frame->pack, pack->buffer->data,
        schro_pack_get_offset (pack));
  } else {
    schro_arith_flush (arith);
    schro_pack_encode_uint (frame->pack, arith->offset);
    schro_pack_sync (frame->pack);
    schro_pack_append (frame->pack, arith->buffer->data, arith->offset);
    schro_arith_free (arith);
  }
}


void
schro_encoder_encode_sequence_header_header (SchroEncoder * encoder,
    SchroPack * pack)
{
  SchroVideoFormat *format = &encoder->video_format;
  SchroVideoFormat _std_format;
  SchroVideoFormat *std_format = &_std_format;
  int i;

  schro_encoder_encode_parse_info (pack, SCHRO_PARSE_CODE_SEQUENCE_HEADER);

  /* parse parameters */
  schro_pack_encode_uint (pack, encoder->version_major);
  schro_pack_encode_uint (pack, encoder->version_minor);
  schro_pack_encode_uint (pack, encoder->profile);
  schro_pack_encode_uint (pack, encoder->level);

  /* sequence parameters */
  schro_pack_encode_uint (pack, encoder->video_format.index);
  schro_video_format_set_std_video_format (std_format,
      encoder->video_format.index);

  if (std_format->width == format->width &&
      std_format->height == format->height) {
    schro_pack_encode_bit (pack, FALSE);
  } else {
    schro_pack_encode_bit (pack, TRUE);
    schro_pack_encode_uint (pack, format->width);
    schro_pack_encode_uint (pack, format->height);
  }

  if (std_format->chroma_format == format->chroma_format) {
    schro_pack_encode_bit (pack, FALSE);
  } else {
    schro_pack_encode_bit (pack, TRUE);
    schro_pack_encode_uint (pack, format->chroma_format);
  }

  /* scan format */
  if (std_format->interlaced == format->interlaced) {
    schro_pack_encode_bit (pack, FALSE);
  } else {
    schro_pack_encode_bit (pack, TRUE);
    schro_pack_encode_uint (pack, format->interlaced);
  }

  MARKER (pack);

  /* frame rate */
  if (std_format->frame_rate_numerator == format->frame_rate_numerator &&
      std_format->frame_rate_denominator == format->frame_rate_denominator) {
    schro_pack_encode_bit (pack, FALSE);
  } else {
    schro_pack_encode_bit (pack, TRUE);
    i = schro_video_format_get_std_frame_rate (format);
    schro_pack_encode_uint (pack, i);
    if (i == 0) {
      schro_pack_encode_uint (pack, format->frame_rate_numerator);
      schro_pack_encode_uint (pack, format->frame_rate_denominator);
    }
  }

  MARKER (pack);

  /* pixel aspect ratio */
  if (std_format->aspect_ratio_numerator == format->aspect_ratio_numerator &&
      std_format->aspect_ratio_denominator ==
      format->aspect_ratio_denominator) {
    schro_pack_encode_bit (pack, FALSE);
  } else {
    schro_pack_encode_bit (pack, TRUE);
    i = schro_video_format_get_std_aspect_ratio (format);
    schro_pack_encode_uint (pack, i);
    if (i == 0) {
      schro_pack_encode_uint (pack, format->aspect_ratio_numerator);
      schro_pack_encode_uint (pack, format->aspect_ratio_denominator);
    }
  }

  MARKER (pack);

  /* clean area */
  if (std_format->clean_width == format->clean_width &&
      std_format->clean_height == format->clean_height &&
      std_format->left_offset == format->left_offset &&
      std_format->top_offset == format->top_offset) {
    schro_pack_encode_bit (pack, FALSE);
  } else {
    schro_pack_encode_bit (pack, TRUE);
    schro_pack_encode_uint (pack, format->clean_width);
    schro_pack_encode_uint (pack, format->clean_height);
    schro_pack_encode_uint (pack, format->left_offset);
    schro_pack_encode_uint (pack, format->top_offset);
  }

  MARKER (pack);

  /* signal range */
  if (std_format->luma_offset == format->luma_offset &&
      std_format->luma_excursion == format->luma_excursion &&
      std_format->chroma_offset == format->chroma_offset &&
      std_format->chroma_excursion == format->chroma_excursion) {
    schro_pack_encode_bit (pack, FALSE);
  } else {
    schro_pack_encode_bit (pack, TRUE);
    i = schro_video_format_get_std_signal_range (format);
    schro_pack_encode_uint (pack, i);
    if (i == 0) {
      schro_pack_encode_uint (pack, format->luma_offset);
      schro_pack_encode_uint (pack, format->luma_excursion);
      schro_pack_encode_uint (pack, format->chroma_offset);
      schro_pack_encode_uint (pack, format->chroma_excursion);
    }
  }

  MARKER (pack);

  /* colour spec */
  if (std_format->colour_primaries == format->colour_primaries &&
      std_format->colour_matrix == format->colour_matrix &&
      std_format->transfer_function == format->transfer_function) {
    schro_pack_encode_bit (pack, FALSE);
  } else {
    schro_pack_encode_bit (pack, TRUE);
    i = schro_video_format_get_std_colour_spec (format);
    schro_pack_encode_uint (pack, i);
    if (i == 0) {
      schro_pack_encode_bit (pack, TRUE);
      schro_pack_encode_uint (pack, format->colour_primaries);
      schro_pack_encode_bit (pack, TRUE);
      schro_pack_encode_uint (pack, format->colour_matrix);
      schro_pack_encode_bit (pack, TRUE);
      schro_pack_encode_uint (pack, format->transfer_function);
    }
  }

  /* interlaced coding */
  schro_pack_encode_uint (pack, format->interlaced_coding);

  MARKER (pack);

  schro_pack_sync (pack);
}

void
schro_encoder_encode_parse_info (SchroPack * pack, int parse_code)
{
  /* parse parameters */
  schro_pack_encode_bits (pack, 8, 'B');
  schro_pack_encode_bits (pack, 8, 'B');
  schro_pack_encode_bits (pack, 8, 'C');
  schro_pack_encode_bits (pack, 8, 'D');
  schro_pack_encode_bits (pack, 8, parse_code);

  /* offsets */
  schro_pack_encode_bits (pack, 32, 0);
  schro_pack_encode_bits (pack, 32, 0);
}

void
schro_encoder_encode_picture_header (SchroEncoderFrame * frame)
{
  schro_pack_sync (frame->pack);
  schro_pack_encode_bits (frame->pack, 32, frame->frame_number);

  SCHRO_DEBUG ("refs %d ref0 %d ref1 %d", frame->params.num_refs,
      frame->picture_number_ref[0], frame->picture_number_ref[1]);

  if (frame->params.num_refs > 0) {
    schro_pack_encode_sint (frame->pack,
        (int32_t) (frame->picture_number_ref[0] - frame->frame_number));
    if (frame->params.num_refs > 1) {
      schro_pack_encode_sint (frame->pack,
          (int32_t) (frame->picture_number_ref[1] - frame->frame_number));
    }
  }

  if (frame->is_ref) {
    if (frame->retired_picture_number != -1) {
      schro_pack_encode_sint (frame->pack,
          (int32_t) (frame->retired_picture_number - frame->frame_number));
    } else {
      schro_pack_encode_sint (frame->pack, 0);
    }
  }
}


static void
schro_encoder_encode_transform_parameters (SchroEncoderFrame * frame)
{
  SchroParams *params = &frame->params;
  SchroPack *pack = frame->pack;

  if (params->num_refs > 0) {
    /* zero residual */
    schro_pack_encode_bit (pack, FALSE);
  }

  /* transform */
  schro_pack_encode_uint (pack, params->wavelet_filter_index);

  /* transform depth */
  schro_pack_encode_uint (pack, params->transform_depth);

  /* spatial partitioning */
  if (!params->is_lowdelay) {
    if (schro_params_is_default_codeblock (params)) {
      schro_pack_encode_bit (pack, FALSE);
    } else {
      int i;

      schro_pack_encode_bit (pack, TRUE);
      for (i = 0; i < params->transform_depth + 1; i++) {
        schro_pack_encode_uint (pack, params->horiz_codeblocks[i]);
        schro_pack_encode_uint (pack, params->vert_codeblocks[i]);
      }
      schro_pack_encode_uint (pack, params->codeblock_mode_index);
    }
  } else {
    schro_pack_encode_uint (pack, params->n_horiz_slices);
    schro_pack_encode_uint (pack, params->n_vert_slices);
    schro_pack_encode_uint (pack, params->slice_bytes_num);
    schro_pack_encode_uint (pack, params->slice_bytes_denom);

    if (schro_params_is_default_quant_matrix (params)) {
      schro_pack_encode_bit (pack, FALSE);
    } else {
      int i;
      schro_pack_encode_bit (pack, TRUE);
      schro_pack_encode_uint (pack, params->quant_matrix[0]);
      for (i = 0; i < params->transform_depth; i++) {
        schro_pack_encode_uint (pack, params->quant_matrix[1 + 3 * i]);
        schro_pack_encode_uint (pack, params->quant_matrix[2 + 3 * i]);
        schro_pack_encode_uint (pack, params->quant_matrix[3 + 3 * i]);
      }
    }
  }
}



void
schro_encoder_clean_up_transform (SchroEncoderFrame * frame)
{
  int i;
  int component;
  SchroParams *params = &frame->params;

  for (component = 0; component < 3; component++) {
    for (i = 0; i < 1 + 3 * params->transform_depth; i++) {
      schro_encoder_clean_up_transform_subband (frame, component, i);
    }
  }
}

static void
schro_encoder_clean_up_transform_subband (SchroEncoderFrame * frame,
    int component, int index)
{
  static const int wavelet_extent[SCHRO_N_WAVELETS] = { 2, 1, 2, 0, 0, 4, 2 };
  SchroParams *params = &frame->params;
  SchroFrameData fd;
  int w;
  int h;
  int shift;
  int16_t *line;
  int i, j;
  int position;

  position = schro_subband_get_position (index);
  schro_subband_get_frame_data (&fd, frame->iwt_frame, component,
      position, params);

  shift = params->transform_depth - SCHRO_SUBBAND_SHIFT (position);
  if (component == 0) {
    schro_video_format_get_picture_luma_size (params->video_format, &w, &h);
  } else {
    schro_video_format_get_picture_chroma_size (params->video_format, &w, &h);
  }

  h = MIN (h + wavelet_extent[params->wavelet_filter_index], fd.height);
  w = MIN (w + wavelet_extent[params->wavelet_filter_index], fd.width);

  if (w < fd.width) {
    for (j = 0; j < h; j++) {
      line = SCHRO_FRAME_DATA_GET_LINE (&fd, j);
      for (i = w; i < fd.width; i++) {
        line[i] = 0;
      }
    }
  }
  for (j = h; j < fd.height; j++) {
    line = SCHRO_FRAME_DATA_GET_LINE (&fd, j);
    for (i = 0; i < fd.width; i++) {
      line[i] = 0;
    }
  }
}

static void
schro_encoder_encode_transform_data (SchroEncoderFrame * frame)
{
  int i;
  int component;
  SchroParams *params = &frame->params;

  for (component = 0; component < 3; component++) {
    for (i = 0; i < 1 + 3 * params->transform_depth; i++) {
      schro_pack_sync (frame->pack);
      frame->actual_subband_bits[component][i] =
          -schro_pack_get_offset (frame->pack) * 8;
      if (params->is_noarith) {
        schro_encoder_encode_subband_noarith (frame, component, i);
      } else {
        schro_encoder_encode_subband (frame, component, i);
      }
      frame->actual_subband_bits[component][i] +=
          schro_pack_get_offset (frame->pack) * 8;
    }
  }
}

static void
schro_frame_data_quantise (SchroFrameData * quant_fd,
    SchroFrameData * fd, int quant_index, schro_bool is_intra)
{
  int j;
  int16_t *line;
  int16_t *quant_line;
  int quant_factor = schro_table_quant[quant_index];
  int quant_offset;
  int quant_shift = (quant_index >> 2) + 2;
  int real_quant_offset;
  int inv_quant = schro_table_inverse_quant[quant_index];

  if (is_intra) {
    quant_offset = schro_table_offset_1_2[quant_index];
  } else {
    quant_offset = schro_table_offset_3_8[quant_index];
  }
  real_quant_offset = quant_offset;
  quant_offset -= quant_factor >> 1;

  if (quant_index == 0) {
    for (j = 0; j < fd->height; j++) {
      line = SCHRO_FRAME_DATA_GET_LINE (fd, j);
      quant_line = SCHRO_FRAME_DATA_GET_LINE (quant_fd, j);

      orc_memcpy (quant_line, line, fd->width * sizeof (int16_t));
    }
  } else if ((quant_index & 3) == 0) {
    for (j = 0; j < fd->height; j++) {
      line = SCHRO_FRAME_DATA_GET_LINE (fd, j);
      quant_line = SCHRO_FRAME_DATA_GET_LINE (quant_fd, j);

      orc_quantdequant2_s16 (quant_line, line, quant_shift,
          quant_offset, quant_factor, real_quant_offset + 2, fd->width);
    }
  } else if (quant_index == 3) {
    for (j = 0; j < fd->height; j++) {
      line = SCHRO_FRAME_DATA_GET_LINE (fd, j);
      quant_line = SCHRO_FRAME_DATA_GET_LINE (quant_fd, j);

      orc_quantdequant3_s16 (quant_line, line, inv_quant, quant_offset,
          quant_shift + 16, quant_factor, real_quant_offset + 2, 32768,
          fd->width);
    }
  } else {
    if (quant_index > 8)
      quant_offset--;
    for (j = 0; j < fd->height; j++) {
      line = SCHRO_FRAME_DATA_GET_LINE (fd, j);
      quant_line = SCHRO_FRAME_DATA_GET_LINE (quant_fd, j);

      orc_quantdequant1_s16 (quant_line, line, inv_quant, quant_offset,
          quant_shift, quant_factor, real_quant_offset + 2, fd->width);
    }
  }
}

#if 0
static void
schro_frame_data_dequantise (SchroFrameData * fd,
    SchroFrameData * quant_fd, int quant_index, schro_bool is_intra)
{
  int j;
  int16_t *line;
  int16_t *quant_line;
  int quant_factor = schro_table_quant[quant_index];
  int quant_offset;

  if (is_intra) {
    quant_offset = schro_table_offset_1_2[quant_index];
  } else {
    quant_offset = schro_table_offset_3_8[quant_index];
  }

  if (quant_index == 0) {
    for (j = 0; j < fd->height; j++) {
      line = SCHRO_FRAME_DATA_GET_LINE (fd, j);
      quant_line = SCHRO_FRAME_DATA_GET_LINE (quant_fd, j);

      orc_memcpy (line, quant_line, fd->width * sizeof (int16_t));
    }
  } else {
    for (j = 0; j < fd->height; j++) {
      line = SCHRO_FRAME_DATA_GET_LINE (fd, j);
      quant_line = SCHRO_FRAME_DATA_GET_LINE (quant_fd, j);

      orc_dequantise_s16 (line, quant_line, quant_factor, quant_offset + 2,
          fd->width);
    }
  }
}
#endif

static void
schro_frame_data_quantise_dc_predict (SchroFrameData * quant_fd,
    SchroFrameData * fd, int quant_factor, int quant_offset, int x, int y)
{
  int i, j;
  int16_t *line;
  int16_t *prev_line;
  int16_t *quant_line;

  for (j = 0; j < fd->height; j++) {
    line = SCHRO_FRAME_DATA_GET_LINE (fd, j);
    prev_line = SCHRO_FRAME_DATA_GET_LINE (fd, j - 1);
    quant_line = SCHRO_FRAME_DATA_GET_LINE (quant_fd, j);

    for (i = 0; i < fd->width; i++) {
      int q;
      int pred_value;

      if (y + j > 0) {
        if (x + i > 0) {
          pred_value = schro_divide3 (line[i - 1] +
              prev_line[i] + prev_line[i - 1] + 1);
        } else {
          pred_value = prev_line[i];
        }
      } else {
        if (x + i > 0) {
          pred_value = line[i - 1];
        } else {
          pred_value = 0;
        }
      }

      q = schro_quantise (line[i] - pred_value, quant_factor, quant_offset);
      line[i] = schro_dequantise (q, quant_factor, quant_offset) + pred_value;
      quant_line[i] = q;
    }
  }
}

int
schro_encoder_frame_get_quant_index (SchroEncoderFrame * frame, int component,
    int index, int x, int y)
{
  SchroParams *params = &frame->params;
  int position;
  int horiz_codeblocks;
  int *codeblock_quants;

  position = schro_subband_get_position (index);
  horiz_codeblocks =
      params->horiz_codeblocks[SCHRO_SUBBAND_SHIFT (position) + 1];

  codeblock_quants = frame->quant_indices[component][index];

  SCHRO_ASSERT (codeblock_quants);

  return codeblock_quants[y * horiz_codeblocks + x];
}

void
schro_encoder_frame_set_quant_index (SchroEncoderFrame * frame, int component,
    int index, int x, int y, int quant_index)
{
  SchroParams *params = &frame->params;
  int *codeblock_quants;
  int position;
  int horiz_codeblocks;
  int vert_codeblocks;
  int i;


  position = schro_subband_get_position (index);
  horiz_codeblocks =
      params->horiz_codeblocks[SCHRO_SUBBAND_SHIFT (position) + 1];
  vert_codeblocks = params->vert_codeblocks[SCHRO_SUBBAND_SHIFT (position) + 1];

  SCHRO_ASSERT (horiz_codeblocks > 0);
  SCHRO_ASSERT (vert_codeblocks > 0);
  SCHRO_ASSERT (x < horiz_codeblocks);
  SCHRO_ASSERT (y < vert_codeblocks);

  if (frame->quant_indices[component][index] == NULL) {
    frame->quant_indices[component][index] =
        schro_malloc (horiz_codeblocks * vert_codeblocks * sizeof (int));
    x = -1;
    y = -1;
  }

  codeblock_quants = frame->quant_indices[component][index];

  if (x < 0 || y < 0) {
    for (i = 0; i < horiz_codeblocks * vert_codeblocks; i++) {
      codeblock_quants[i] = quant_index;
    }
  } else {
    codeblock_quants[x + y * horiz_codeblocks] = quant_index;
  }
}

static int
schro_encoder_quantise_subband (SchroEncoderFrame * frame, int component,
    int index)
{
  int quant_index;
  int quant_factor;
  int quant_offset;
  SchroFrameData fd;
  SchroFrameData qd;
  int position;
  SchroParams *params = &frame->params;
  int horiz_codeblocks;
  int vert_codeblocks;
  int x, y;

  position = schro_subband_get_position (index);
  schro_subband_get_frame_data (&fd, frame->iwt_frame, component,
      position, params);
  schro_subband_get_frame_data (&qd, frame->quant_frame, component,
      position, params);

  horiz_codeblocks =
      params->horiz_codeblocks[SCHRO_SUBBAND_SHIFT (position) + 1];
  vert_codeblocks = params->vert_codeblocks[SCHRO_SUBBAND_SHIFT (position) + 1];

  for (y = 0; y < vert_codeblocks; y++) {

    for (x = 0; x < horiz_codeblocks; x++) {
      SchroFrameData quant_cb;
      SchroFrameData cb;

      quant_index = schro_encoder_frame_get_quant_index (frame, component,
          index, x, y);
      quant_factor = schro_table_quant[quant_index];
      if (params->num_refs > 0) {
        quant_offset = schro_table_offset_3_8[quant_index];
      } else {
        quant_offset = schro_table_offset_1_2[quant_index];
      }

      schro_frame_data_get_codeblock (&cb, &fd, x, y, horiz_codeblocks,
          vert_codeblocks);
      schro_frame_data_get_codeblock (&quant_cb, &qd, x, y, horiz_codeblocks,
          vert_codeblocks);

      if (params->num_refs == 0 && index == 0) {
        schro_frame_data_quantise_dc_predict (&quant_cb, &cb, quant_factor,
            quant_offset, x, y);
      } else {
        schro_frame_data_quantise (&quant_cb, &cb, quant_index,
            (params->num_refs == 0));
#if 0
        schro_frame_data_dequantise (&cb, &quant_cb, quant_index,
            (params->num_refs == 0));
#endif
      }
    }
  }

  return schro_frame_data_is_zero (&qd);
}

static void
schro_frame_data_clear (SchroFrameData * fd)
{
  int i;

  for (i = 0; i < fd->height; i++) {
    orc_splat_s16_ns (SCHRO_FRAME_DATA_GET_LINE (fd, i), 0, fd->width);
  }
}

void
schro_encoder_encode_subband (SchroEncoderFrame * frame, int component,
    int index)
{
  SchroParams *params = &frame->params;
  SchroArith *arith;
  int i, j;
  int subband_zero_flag;
  int x, y;
  int horiz_codeblocks;
  int vert_codeblocks;
  int have_zero_flags;
  int have_quant_offset;
  int position;
  SchroFrameData fd;
  SchroFrameData qd;
  SchroFrameData parent_fd;
  int quant_index;
  int n_subbands_left;
  int first_quant_index;

  position = schro_subband_get_position (index);
  schro_subband_get_frame_data (&fd, frame->iwt_frame, component,
      position, params);
  schro_subband_get_frame_data (&qd, frame->quant_frame, component,
      position, params);

  if (position >= 4) {
    schro_subband_get_frame_data (&parent_fd, frame->iwt_frame, component,
        position - 4, params);
  } else {
    parent_fd.data = NULL;
    parent_fd.stride = 0;
  }

  arith = schro_arith_new ();
  schro_arith_encode_init (arith, frame->subband_buffer);

  subband_zero_flag = schro_encoder_quantise_subband (frame, component, index);

  if (subband_zero_flag) {
    SCHRO_DEBUG ("subband is zero");
    schro_pack_encode_uint (frame->pack, 0);
    schro_arith_free (arith);
    return;
  }

  if (index == 0) {
    horiz_codeblocks = params->horiz_codeblocks[0];
    vert_codeblocks = params->vert_codeblocks[0];
  } else {
    horiz_codeblocks =
        params->horiz_codeblocks[SCHRO_SUBBAND_SHIFT (position) + 1];
    vert_codeblocks =
        params->vert_codeblocks[SCHRO_SUBBAND_SHIFT (position) + 1];
  }
  if (horiz_codeblocks > 1 || vert_codeblocks > 1) {
    have_zero_flags = TRUE;
  } else {
    have_zero_flags = FALSE;
  }
  if (params->codeblock_mode_index == 1) {
    have_quant_offset = TRUE;
  } else {
    have_quant_offset = FALSE;
  }

  first_quant_index = -1;
  quant_index = 0;
  for (y = 0; y < vert_codeblocks; y++) {
    int ymin = (fd.height * y) / vert_codeblocks;
    int ymax = (fd.height * (y + 1)) / vert_codeblocks;

    for (x = 0; x < horiz_codeblocks; x++) {
      int xmin = (fd.width * x) / horiz_codeblocks;
      int xmax = (fd.width * (x + 1)) / horiz_codeblocks;
      SchroFrameData quant_cb;
      SchroFrameData cb;

      schro_frame_data_get_codeblock (&cb, &fd, x, y, horiz_codeblocks,
          vert_codeblocks);
      schro_frame_data_get_codeblock (&quant_cb, &qd, x, y, horiz_codeblocks,
          vert_codeblocks);

      if (have_zero_flags) {
        int zero_codeblock = schro_frame_data_is_zero (&quant_cb);

        _schro_arith_encode_bit (arith, SCHRO_CTX_ZERO_CODEBLOCK,
            zero_codeblock);
        if (zero_codeblock) {
          continue;
        }
      }

      if (have_quant_offset) {
        int new_quant_index;

        new_quant_index = schro_encoder_frame_get_quant_index (frame,
            component, index, x, y);
        if (first_quant_index == -1) {
          quant_index = new_quant_index;
          first_quant_index = new_quant_index;
        }

        _schro_arith_encode_sint (arith,
            SCHRO_CTX_QUANTISER_CONT, SCHRO_CTX_QUANTISER_VALUE,
            SCHRO_CTX_QUANTISER_SIGN, new_quant_index - quant_index);

        quant_index = new_quant_index;
      }

      for (j = ymin; j < ymax; j++) {
        int16_t *prev_quant_line = SCHRO_FRAME_DATA_GET_LINE (&qd, j - 1);
        int16_t *quant_line = SCHRO_FRAME_DATA_GET_LINE (&qd, j);
        int16_t *parent_line = SCHRO_FRAME_DATA_GET_LINE (&parent_fd, (j >> 1));

#define STUFF(have_parent,is_horiz,is_vert) \
        for(i=xmin;i<xmax;i++){ \
          int parent; \
          int cont_context; \
          int value_context; \
          int nhood_or; \
          int previous_value; \
          int sign_context; \
 \
          if (have_parent) { \
            parent = parent_line[(i>>1)]; \
          } else { \
            parent = 0; \
          } \
\
          nhood_or = 0; \
          if (j>0) { \
            nhood_or |= prev_quant_line[i]; \
          } \
          if (i>0) { \
            nhood_or |= quant_line[i - 1]; \
          } \
          if (i>0 && j>0) { \
            nhood_or |= prev_quant_line[i - 1]; \
          } \
 \
          previous_value = 0; \
          if (is_horiz) { \
            if (i > 0) { \
              previous_value = quant_line[i - 1]; \
            } \
          } else if (is_vert) { \
            if (j > 0) { \
              previous_value = prev_quant_line[i]; \
            } \
          } \
 \
          if (previous_value < 0) { \
            sign_context = SCHRO_CTX_SIGN_NEG; \
          } else { \
            if (previous_value > 0) { \
              sign_context = SCHRO_CTX_SIGN_POS; \
            } else { \
              sign_context = SCHRO_CTX_SIGN_ZERO; \
            } \
          } \
 \
          if (parent == 0) { \
            if (nhood_or == 0) { \
              cont_context = SCHRO_CTX_ZPZN_F1; \
            } else { \
              cont_context = SCHRO_CTX_ZPNN_F1; \
            } \
          } else { \
            if (nhood_or == 0) { \
              cont_context = SCHRO_CTX_NPZN_F1; \
            } else { \
              cont_context = SCHRO_CTX_NPNN_F1; \
            } \
          } \
 \
          value_context = SCHRO_CTX_COEFF_DATA; \
 \
          _schro_arith_encode_sint (arith, cont_context, value_context, \
              sign_context, quant_line[i]); \
        }

        if (position >= 4) {
          if (SCHRO_SUBBAND_IS_HORIZONTALLY_ORIENTED (position)) {
            STUFF (TRUE, TRUE, FALSE);
          } else if (SCHRO_SUBBAND_IS_VERTICALLY_ORIENTED (position)) {
            STUFF (TRUE, FALSE, TRUE);
          } else {
            STUFF (TRUE, FALSE, FALSE);
          }
        } else {
          if (SCHRO_SUBBAND_IS_HORIZONTALLY_ORIENTED (position)) {
            STUFF (FALSE, TRUE, FALSE);
          } else if (SCHRO_SUBBAND_IS_VERTICALLY_ORIENTED (position)) {
            STUFF (FALSE, FALSE, TRUE);
          } else {
            STUFF (FALSE, FALSE, FALSE);
          }
        }
      }
    }
  }

  schro_arith_flush (arith);

  SCHRO_ASSERT (arith->offset < frame->subband_buffer->length);

  schro_dump (SCHRO_DUMP_SUBBAND_EST, "%d %d %d %g %d\n",
      frame->frame_number, component, index,
      frame->
      est_entropy[component][index][frame->quant_indices[component][index][0]],
      arith->offset * 8);

  n_subbands_left = (3 - component) * (1 + 3 * params->transform_depth) - index;
  if ((schro_pack_get_offset (frame->pack) + arith->offset +
          n_subbands_left + 4) * 8 > frame->hard_limit_bits) {
    SCHRO_ERROR ("skipping comp=%d subband=%d, too big (%d+%d+%d+32 > %d)",
        component, index,
        schro_pack_get_offset (frame->pack) * 8, arith->offset * 8,
        n_subbands_left * 8, frame->hard_limit_bits);

    schro_pack_encode_uint (frame->pack, 0);

    schro_frame_data_clear (&fd);
  } else {
    SCHRO_DEBUG ("appending comp=%d subband=%d, (%d+%d+%d+32 <= %d)",
        component, index,
        schro_pack_get_offset (frame->pack) * 8, arith->offset * 8,
        n_subbands_left * 8, frame->hard_limit_bits);
    schro_pack_encode_uint (frame->pack, arith->offset);
    if (first_quant_index == -1) {
      first_quant_index =
          schro_encoder_frame_get_quant_index (frame, component, index, 0, 0);
    }
    if (arith->offset > 0) {
      schro_pack_encode_uint (frame->pack, first_quant_index);

      schro_pack_sync (frame->pack);

      schro_pack_append (frame->pack, arith->buffer->data, arith->offset);
    }
  }
  schro_arith_free (arith);
}

static schro_bool
schro_frame_data_is_zero (SchroFrameData * fd)
{
  int j;
  int16_t *line;
  int acc;

  for (j = 0; j < fd->height; j++) {
    line = SCHRO_FRAME_DATA_GET_LINE (fd, j);
    /* FIXME this could theoretically cause false positives.  Fix when
     * Orc gets an accorw opcode. */
    orc_accw (&acc, line, fd->width);
    if (acc != 0)
      return FALSE;
  }

  return TRUE;
}

void
schro_encoder_encode_subband_noarith (SchroEncoderFrame * frame,
    int component, int index)
{
  SchroParams *params = &frame->params;
  SchroPack b;
  SchroPack *pack = &b;
  int i, j;
  int subband_zero_flag;
  int x, y;
  int horiz_codeblocks;
  int vert_codeblocks;
  int have_zero_flags;
  int have_quant_offset;
  int position;
  SchroFrameData fd;
  SchroFrameData qd;

  position = schro_subband_get_position (index);
  schro_subband_get_frame_data (&fd, frame->iwt_frame, component,
      position, params);
  schro_subband_get_frame_data (&qd, frame->quant_frame, component,
      position, params);

  subband_zero_flag = schro_encoder_quantise_subband (frame, component, index);

  if (subband_zero_flag) {
    SCHRO_DEBUG ("subband is zero");
    schro_pack_encode_uint (frame->pack, 0);
    return;
  }

  schro_pack_encode_init (pack, frame->subband_buffer);

  if (index == 0) {
    horiz_codeblocks = params->horiz_codeblocks[0];
    vert_codeblocks = params->vert_codeblocks[0];
  } else {
    horiz_codeblocks =
        params->horiz_codeblocks[SCHRO_SUBBAND_SHIFT (position) + 1];
    vert_codeblocks =
        params->vert_codeblocks[SCHRO_SUBBAND_SHIFT (position) + 1];
  }
  if ((horiz_codeblocks > 1 || vert_codeblocks > 1) && index > 0) {
    have_zero_flags = TRUE;
  } else {
    have_zero_flags = FALSE;
  }
  if (horiz_codeblocks > 1 || vert_codeblocks > 1) {
    if (params->codeblock_mode_index == 1) {
      have_quant_offset = TRUE;
    } else {
      have_quant_offset = FALSE;
    }
  } else {
    have_quant_offset = FALSE;
  }

  for (y = 0; y < vert_codeblocks; y++) {
    for (x = 0; x < horiz_codeblocks; x++) {
      SchroFrameData cb;

      schro_frame_data_get_codeblock (&cb, &qd, x, y, horiz_codeblocks,
          vert_codeblocks);

      if (have_zero_flags) {
        int zero_codeblock = schro_frame_data_is_zero (&cb);
        schro_pack_encode_bit (pack, zero_codeblock);
        if (zero_codeblock) {
          continue;
        }
      }

      if (have_quant_offset) {
        schro_pack_encode_sint (pack, 0);
      }

      for (j = 0; j < cb.height; j++) {
        int16_t *quant_line = SCHRO_FRAME_DATA_GET_LINE (&cb, j);
        for (i = 0; i < cb.width; i++) {
          schro_pack_encode_sint (pack, quant_line[i]);
        }
      }
    }
  }
  schro_pack_flush (pack);

  SCHRO_ASSERT (schro_pack_get_offset (pack) < frame->subband_buffer->length);

  schro_dump (SCHRO_DUMP_SUBBAND_EST, "%d %d %d %d %d\n",
      frame->frame_number, component, index,
      frame->estimated_residual_bits, schro_pack_get_offset (pack) * 8);

  schro_pack_encode_uint (frame->pack, schro_pack_get_offset (pack));
  if (schro_pack_get_offset (pack) > 0) {
    schro_pack_encode_uint (frame->pack,
        schro_encoder_frame_get_quant_index (frame, component, index, 0, 0));

    schro_pack_sync (frame->pack);
    schro_pack_append (frame->pack, pack->buffer->data,
        schro_pack_get_offset (pack));
  }

}

/* frame queue */

SchroEncoderFrame *
schro_encoder_frame_new (SchroEncoder * encoder)
{
  SchroEncoderFrame *encoder_frame;
  SchroFrameFormat frame_format;
  int iwt_width, iwt_height;
  int picture_width;
  int picture_height;
  int i;

  encoder_frame = schro_malloc0 (sizeof (SchroEncoderFrame));
  for (i = 0; i < SCHRO_ENCODER_FRAME_STAGE_LAST; i++) {
    encoder_frame->stages[i].is_needed = TRUE;
  }
  encoder_frame->refcount = 1;

  encoder_frame->sc_mad = -1;
  encoder_frame->sc_threshold = -1.;
  encoder_frame->scene_change_score = -1.;

  frame_format = schro_params_get_frame_format (16,
      encoder->video_format.chroma_format);

  schro_video_format_get_iwt_alloc_size (&encoder->video_format,
      &iwt_width, &iwt_height, encoder->transform_depth);
  encoder_frame->iwt_frame = schro_frame_new_and_alloc (NULL, frame_format,
      iwt_width, iwt_height);
  encoder_frame->quant_frame = schro_frame_new_and_alloc (NULL, frame_format,
      iwt_width, iwt_height);

  schro_video_format_get_picture_luma_size (&encoder->video_format,
      &picture_width, &picture_height);
  encoder_frame->prediction_frame =
      schro_frame_new_and_alloc (NULL, frame_format, picture_width,
      picture_height);

  encoder_frame->inserted_buffers =
      schro_list_new_full ((SchroListFreeFunc) schro_buffer_unref, NULL);

  encoder_frame->retired_picture_number = -1;

  return encoder_frame;
}

void
schro_encoder_frame_ref (SchroEncoderFrame * frame)
{
  SCHRO_ASSERT (frame && frame->refcount > 0);
  frame->refcount++;
}

void
schro_encoder_frame_unref (SchroEncoderFrame * frame)
{
  int i;

  frame->refcount--;
  if (frame->refcount == 0) {
    if (frame->previous_frame) {
      schro_encoder_frame_unref (frame->previous_frame);
    }
    if (frame->original_frame) {
      schro_frame_unref (frame->original_frame);
    }
    if (frame->filtered_frame) {
      schro_frame_unref (frame->filtered_frame);
    }
    if (frame->reconstructed_frame) {
      schro_upsampled_frame_free (frame->reconstructed_frame);
    }

    if (frame->upsampled_original_frame) {
      schro_upsampled_frame_free (frame->upsampled_original_frame);
    }
#if 0
    for (i = 0; i < 2; i++) {
      if (frame->mf[i]) {
        schro_motion_field_free (frame->mf[i]);
      }
    }
#endif

    for (i = 0; i < frame->encoder->downsample_levels; i++) {
      if (frame->downsampled_frames[i]) {
        schro_frame_unref (frame->downsampled_frames[i]);
      }
    }
    if (frame->iwt_frame) {
      schro_frame_unref (frame->iwt_frame);
    }
    if (frame->prediction_frame) {
      schro_frame_unref (frame->prediction_frame);
    }
    if (frame->motion) {
      schro_motion_free (frame->motion);
    }

    schro_list_free (frame->inserted_buffers);
    if (frame->output_buffer) {
      schro_buffer_unref (frame->output_buffer);
    }
    if (frame->sequence_header_buffer) {
      schro_buffer_unref (frame->sequence_header_buffer);
    }

    if (frame->me) {
      schro_motionest_free (frame->me);
    }
    if (frame->rme[0])
      schro_rough_me_free (frame->rme[0]);
    if (frame->rme[1])
      schro_rough_me_free (frame->rme[1]);
    if (frame->hier_bm[0])
      schro_hbm_unref (frame->hier_bm[0]);
    frame->hier_bm[0] = NULL;
    if (frame->hier_bm[1])
      schro_hbm_unref (frame->hier_bm[1]);
    frame->hier_bm[1] = NULL;
    if (frame->deep_me)
      schro_me_free (frame->deep_me);
    frame->deep_me = NULL;
    if (frame->phasecorr[0])
      schro_phasecorr_free (frame->phasecorr[0]);
    if (frame->phasecorr[1])
      schro_phasecorr_free (frame->phasecorr[1]);

    for (i = 0; i < SCHRO_LIMIT_SUBBANDS; i++) {
      if (frame->quant_indices[0][i])
        schro_free (frame->quant_indices[0][i]);
      if (frame->quant_indices[1][i])
        schro_free (frame->quant_indices[1][i]);
      if (frame->quant_indices[2][i])
        schro_free (frame->quant_indices[2][i]);
    }


    schro_free (frame);
  }
}

#if 0
/* reference pool */

void
schro_encoder_reference_add (SchroEncoder * encoder, SchroEncoderFrame * frame)
{
  if (schro_queue_is_full (encoder->reference_queue)) {
    schro_queue_pop (encoder->reference_queue);
  }

  SCHRO_DEBUG ("adding reference %p %d", frame, frame->frame_number);

  schro_encoder_frame_ref (frame);
  schro_queue_add (encoder->reference_queue, frame, frame->frame_number);
}
#endif

SchroEncoderFrame *
schro_encoder_reference_get (SchroEncoder * encoder,
    SchroPictureNumber frame_number)
{
  int i;
  for (i = 0; i < SCHRO_LIMIT_REFERENCE_FRAMES; i++) {
    if (encoder->reference_pictures[i] &&
        encoder->reference_pictures[i]->frame_number == frame_number) {
      return encoder->reference_pictures[i];
    }
  }
  return NULL;
}

/* settings */

#define ENUM(name,list,def) \
  {{#name, SCHRO_ENCODER_SETTING_TYPE_ENUM, 0, ARRAY_SIZE(list)-1, def, list}, offsetof(SchroEncoder, name)}
#define INT(name,min,max,def) \
  {{#name, SCHRO_ENCODER_SETTING_TYPE_INT, min, max, def}, offsetof(SchroEncoder, name)}
#define BOOL(name,def) \
  {{#name, SCHRO_ENCODER_SETTING_TYPE_BOOLEAN, 0, 1, def}, offsetof(SchroEncoder, name)}
#define DOUB(name,min,max,def) \
  {{#name, SCHRO_ENCODER_SETTING_TYPE_DOUBLE, min, max, def}, offsetof(SchroEncoder, name)}

static const char *rate_control_list[] = {
  "constant_noise_threshold",
  "constant_bitrate",
  "low_delay",
  "lossless",
  "constant_lambda",
  "constant_error",
  "constant_quality"
};

static const char *profile_list[] = {
  "auto",
  "vc2_low_delay",
  "vc2_simple",
  "vc2_main",
  "main"
};

static const char *gop_structure_list[] = {
  "adaptive",
  "intra_only",
  "backref",
  "chained_backref",
  "biref",
  "chained_biref"
};

static const char *perceptual_weighting_list[] = {
  "none",
  "ccir959",
  "moo",
  "manos_sakrison"
};

static const char *filtering_list[] = {
  "none",
  "center_weighted_median",
  "gaussian",
  "add_noise",
  "adaptive_gaussian",
  "lowpass"
};

static const char *wavelet_list[] = {
  "desl_dubuc_9_7",
  "le_gall_5_3",
  "desl_dubuc_13_7",
  "haar_0",
  "haar_1",
  "fidelity",
  "daub_9_7"
};

static const char *block_size_list[] = {
  "automatic",
  "small",
  "medium",
  "large"
};

static const char *codeblock_size_list[] = {
  "automatic",
  "small",
  "medium",
  "large",
  "full"
};

static const char *block_overlap_list[] = {
  "automatic",
  "none",
  "partial",
  "full"
};

#ifndef INT_MAX
#define INT_MAX 2147483647
#endif

struct SchroEncoderSettings
{
  SchroEncoderSetting s;
  int offset;
} static const encoder_settings[] = {
  ENUM (rate_control, rate_control_list, 6),
  INT (bitrate, 0, INT_MAX, 0),
  INT (max_bitrate, 0, INT_MAX, 13824000),
  INT (min_bitrate, 0, INT_MAX, 13824000),
  INT (buffer_size, 0, INT_MAX, 0),
  INT (buffer_level, 0, INT_MAX, 0),
  DOUB (quality, 0, 10.0, 5.0),
  DOUB (noise_threshold, 0, 100.0, 25.0),
  ENUM (gop_structure, gop_structure_list, 0),
  INT (queue_depth, 1, SCHRO_LIMIT_FRAME_QUEUE_LENGTH, 20),
  ENUM (perceptual_weighting, perceptual_weighting_list, 1),
  DOUB (perceptual_distance, 0, 100.0, 4.0),
  ENUM (filtering, filtering_list, 0),
  DOUB (filter_value, 0, 100.0, 5.0),
  INT (profile, 0, 0, 0),
  ENUM (force_profile, profile_list, 0),
  INT (level, 0, 0, 0),
  INT (max_refs, 1, 4, 3),
  BOOL (open_gop, TRUE),
  INT (au_distance, 1, INT_MAX, 120),
  BOOL (enable_psnr, FALSE),
  BOOL (enable_ssim, FALSE),

  INT (transform_depth, 0, SCHRO_LIMIT_ENCODER_TRANSFORM_DEPTH, 3),
  ENUM (intra_wavelet, wavelet_list, SCHRO_WAVELET_DESLAURIERS_DUBUC_9_7),
  ENUM (inter_wavelet, wavelet_list, SCHRO_WAVELET_DESLAURIERS_DUBUC_9_7),
  INT (mv_precision, 0, 3, 0),
  INT (downsample_levels, 2, 8, 5),
  ENUM (motion_block_size, block_size_list, 0),
  ENUM (motion_block_overlap, block_overlap_list, 0),
  BOOL (interlaced_coding, FALSE),
  BOOL (enable_internal_testing, FALSE),
  BOOL (enable_noarith, FALSE),
  BOOL (enable_md5, FALSE),
  BOOL (enable_fullscan_estimation, FALSE),
  BOOL (enable_hierarchical_estimation, TRUE),
  BOOL (enable_zero_estimation, FALSE),
  BOOL (enable_phasecorr_estimation, FALSE),
  BOOL (enable_bigblock_estimation, TRUE),
  BOOL (enable_multiquant, FALSE),
  BOOL (enable_dc_multiquant, FALSE),
  BOOL (enable_global_motion, FALSE),
  BOOL (enable_scene_change_detection, TRUE),
  BOOL (enable_deep_estimation, TRUE),
  BOOL (enable_rdo_cbr, TRUE),
  BOOL (enable_chroma_me, FALSE),
  INT (horiz_slices, 0, INT_MAX, 0),
  INT (vert_slices, 0, INT_MAX, 0),
  ENUM (codeblock_size, codeblock_size_list, 0),

  DOUB (magic_dc_metric_offset, 0.0, 1000.0, 1.0),
  DOUB (magic_subband0_lambda_scale, 0.0, 1000.0, 10.0),
  DOUB (magic_chroma_lambda_scale, 0.0, 1000.0, 0.1),
  DOUB (magic_nonref_lambda_scale, 0.0, 1000.0, 0.01),
  DOUB (magic_me_lambda_scale, 0.0, 100.0, 1.0),
  DOUB (magic_I_lambda_scale, 0.0, 100.0, 1.0),
  DOUB (magic_P_lambda_scale, 0.0, 10.0, 0.25),
  DOUB (magic_B_lambda_scale, 0.0, 10.0, 0.01),
  DOUB (magic_allocation_scale, 0.0, 1000.0, 1.1),
  DOUB (magic_inter_cpd_scale, 0.0, 1.0, 1.0),
  DOUB (magic_keyframe_weight, 0.0, 1000.0, 7.5),
  DOUB (magic_scene_change_threshold, 0.0, 1000.0, 3.0),
  DOUB (magic_inter_p_weight, 0.0, 1000.0, 1.5),
  DOUB (magic_inter_b_weight, 0.0, 1000.0, 0.2),
  DOUB (magic_me_bailout_limit, 0.0, 1000.0, 0.33),
  DOUB (magic_bailout_weight, 0.0, 1000.0, 4.0),
  DOUB (magic_error_power, 0.0, 1000.0, 4.0),
  DOUB (magic_subgroup_length, 1.0, 10.0, 4.0),
  DOUB (magic_badblock_multiplier_nonref, 0.0, 1000.0, 4.0),
  DOUB (magic_badblock_multiplier_ref, 0.0, 1000.0, 8.0),
  DOUB (magic_block_search_threshold, 0.0, 1000.0, 15.0),
  DOUB (magic_scan_distance, 0.0, 1000.0, 4.0),
  DOUB (magic_diagonal_lambda_scale, 0.0, 1000.0, 1.0),
};

int
schro_encoder_get_n_settings (void)
{
  return ARRAY_SIZE (encoder_settings);
}

const SchroEncoderSetting *
schro_encoder_get_setting_info (int i)
{
  if (i >= 0 && i < ARRAY_SIZE (encoder_settings)) {
    return &encoder_settings[i].s;
  }
  return NULL;
}

/**
 * schro_encoder_setting_set_defaults:
 * @encoder: an encoder structure
 *
 * set the encoder options to the defaults advertised through
 * schro_encoder_get_setting_info.  old settings are lost.
 */
static void
schro_encoder_setting_set_defaults (SchroEncoder * encoder)
{
  int i;
  for (i = 0; i < ARRAY_SIZE (encoder_settings); i++) {
    switch (encoder_settings[i].s.type) {
      case SCHRO_ENCODER_SETTING_TYPE_BOOLEAN:
      case SCHRO_ENCODER_SETTING_TYPE_INT:
      case SCHRO_ENCODER_SETTING_TYPE_ENUM:
        *(int *) SCHRO_OFFSET (encoder, encoder_settings[i].offset) =
            encoder_settings[i].s.default_value;
        break;
      case SCHRO_ENCODER_SETTING_TYPE_DOUBLE:
        *(double *) SCHRO_OFFSET (encoder, encoder_settings[i].offset) =
            encoder_settings[i].s.default_value;
        break;
      default:
        break;
    }
  }
}

/**
 * schro_encoder_setting_set_double:
 * @encoder: an encoder object
 *
 * set the encoder option given by @name to @value.
 */
void
schro_encoder_setting_set_double (SchroEncoder * encoder, const char *name,
    double value)
{
  int i;
  for (i = 0; i < ARRAY_SIZE (encoder_settings); i++) {
    if (strcmp (name, encoder_settings[i].s.name)) {
      continue;
    }
    switch (encoder_settings[i].s.type) {
      case SCHRO_ENCODER_SETTING_TYPE_BOOLEAN:
      case SCHRO_ENCODER_SETTING_TYPE_INT:
      case SCHRO_ENCODER_SETTING_TYPE_ENUM:
        *(int *) SCHRO_OFFSET (encoder, encoder_settings[i].offset) = value;
        return;
      case SCHRO_ENCODER_SETTING_TYPE_DOUBLE:
        *(double *) SCHRO_OFFSET (encoder, encoder_settings[i].offset) = value;
        return;
      default:
        return;
    }
  }
}

/**
 * schro_encoder_setting_get_double:
 * @encoder: an encoder object
 *
 * Returns: the current value of an encoder option given by @name
 */
double
schro_encoder_setting_get_double (SchroEncoder * encoder, const char *name)
{
  int i;
  for (i = 0; i < ARRAY_SIZE (encoder_settings); i++) {
    if (strcmp (name, encoder_settings[i].s.name)) {
      continue;
    }
    switch (encoder_settings[i].s.type) {
      case SCHRO_ENCODER_SETTING_TYPE_BOOLEAN:
      case SCHRO_ENCODER_SETTING_TYPE_INT:
      case SCHRO_ENCODER_SETTING_TYPE_ENUM:
        return *(int *) SCHRO_OFFSET (encoder, encoder_settings[i].offset);
      case SCHRO_ENCODER_SETTING_TYPE_DOUBLE:
        return *(double *) SCHRO_OFFSET (encoder, encoder_settings[i].offset);
      default:
        return 0;
    }
  }

  return 0;
}
