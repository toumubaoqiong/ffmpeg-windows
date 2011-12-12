
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#define SCHRO_ARITH_DEFINE_INLINE
#include <schroedinger/schro.h>
#include <schroedinger/schrocuda.h>
#include <schroedinger/schrogpuframe.h>
#include <schroedinger/opengl/schroopengl.h>
#include <schroedinger/opengl/schroopenglframe.h>
#include <schroedinger/opengl/schroopenglmotion.h>
#include <schroedinger/schroorc.h>
#include <string.h>
#include <stdio.h>

#if 0
/* Used for testing bitstream */
#define MARKER() do{ \
  SCHRO_ASSERT(schro_unpack_decode_uint(unpack) == 1234567); \
} while(0)
#else
#define MARKER()
#endif

#define SCHRO_SKIP_TIME_CONSTANT 0.1

typedef struct _SchroPictureSubbandContext SchroPictureSubbandContext;

struct _SchroPictureSubbandContext
{
  int component;
  int index;
  int position;
  int broken;

  SchroFrameData *frame_data;
  SchroFrameData *parent_frame_data;

  int quant_index;
  int is_intra;
  int subband_length;
  SchroArith *arith;
  SchroUnpack unpack;
  int vert_codeblocks;
  int horiz_codeblocks;
  int have_zero_flags;
  int have_quant_offset;

  int ymin;
  int ymax;
  int xmin;
  int xmax;

  int quant_factor;
  int quant_offset;
  int16_t *line;
};

enum
{
  SCHRO_DECODER_STAGE_INIT = 0,
  SCHRO_DECODER_STAGE_REFERENCES,
  SCHRO_DECODER_STAGE_MOTION_DECODE,
  SCHRO_DECODER_STAGE_MOTION_RENDER,
  SCHRO_DECODER_STAGE_RESIDUAL_DECODE,
  SCHRO_DECODER_STAGE_WAVELET_TRANSFORM,
  SCHRO_DECODER_STAGE_COMBINE,
  SCHRO_DECODER_STAGE_UPSAMPLE,
  SCHRO_DECODER_STAGE_DONE
};


int _schro_decode_prediction_only;
int _schro_telemetry;

static void schro_decoder_x_decode_motion (SchroAsyncStage * stage);
static void schro_decoder_x_render_motion (SchroAsyncStage * stage);
static void schro_decoder_x_decode_residual (SchroAsyncStage * stage);
static void schro_decoder_x_wavelet_transform (SchroAsyncStage * stage);
static void schro_decoder_x_combine (SchroAsyncStage * stage);
static void schro_decoder_x_upsample (SchroAsyncStage * stage);

static void schro_decoder_reference_add (SchroDecoderInstance * instance,
    SchroPicture * picture);
static SchroPicture *schro_decoder_reference_get (SchroDecoderInstance *
    instance, SchroPictureNumber frame_number);
static void schro_decoder_reference_retire (SchroDecoderInstance * instance,
    SchroPictureNumber frame_number);
static void schro_decoder_decode_subband (SchroPicture * picture,
    SchroPictureSubbandContext * ctx);
static int schro_decoder_async_schedule (SchroDecoder * decoder,
    SchroExecDomain domain);
static void schro_decoder_picture_complete (SchroAsyncStage * stage);

static void schro_decoder_error (SchroDecoder * decoder, const char *s);

static void schro_decoder_set_rob_size (SchroDecoderInstance * instance);

static int schro_decoder_frame_is_twofield (SchroDecoderInstance * instance,
    SchroFrame * frame);

static void schro_picturequeue_rob_insert (SchroQueue * queue,
    SchroPicture * picture, unsigned windowsize);
static int schro_picture_n_before_m (SchroPictureNumber n,
    SchroPictureNumber m);

static void schro_decoder_telemetry (SchroPicture * picture,
    SchroFrame * output_picture);

/* API */

static SchroDecoderInstance *
schro_decoder_instance_new (SchroDecoder * decoder)
{
  SchroDecoderInstance *instance;

  instance = schro_malloc0 (sizeof (SchroDecoderInstance));

  instance->decoder = decoder;

  instance->reference_queue = schro_queue_new (SCHRO_LIMIT_REFERENCE_FRAMES,
      (SchroQueueFreeFunc) schro_picture_unref);
  /* the output queue shouldn't be too large: the output queue may be
   * a precious resource, no point holding on to loads without reason */
  instance->output_queue =
      schro_queue_new (4, (SchroQueueFreeFunc) schro_frame_unref);

  instance->reorder_queue =
      schro_queue_new (5, (SchroQueueFreeFunc) schro_picture_unref);

  schro_decoder_set_rob_size (instance);

  return instance;
}

static void
schro_decoder_instance_free (SchroDecoderInstance * instance)
{
  schro_queue_free (instance->output_queue);
  schro_queue_free (instance->reference_queue);
  schro_queue_free (instance->reorder_queue);

  if (instance->sequence_header_buffer) {
    schro_buffer_unref (instance->sequence_header_buffer);
    instance->sequence_header_buffer = NULL;
  }

  /* do not call instance_free(instance->next).
   * It must be handled by the caller if required */

  schro_free (instance);
}

/**
 * schro_decoder_new:
 *
 * Creates a new decoder object.  The decoder object should be freed
 * using @schro_decoder_free() when it is no longer needed.
 *
 * Returns: a new decoder object
 */
SchroDecoder *
schro_decoder_new (void)
{
  SchroDecoder *decoder;

  decoder = schro_malloc0 (sizeof (SchroDecoder));

  schro_tables_init ();

  decoder->skip_value = 1.0;
  decoder->skip_ratio = 1.0;

  decoder->input_buflist = schro_buflist_new ();
  decoder->sps = schro_parse_sync_new ();

  decoder->cpu_domain = schro_memory_domain_new_local ();
#ifdef HAVE_CUDA
  decoder->cuda_domain = schro_memory_domain_new_cuda ();
#endif
#ifdef HAVE_OPENGL
  decoder->opengl_domain = schro_memory_domain_new_opengl ();
#endif

  decoder->async = schro_async_new (0,
      (SchroAsyncScheduleFunc) schro_decoder_async_schedule,
      (SchroAsyncCompleteFunc) schro_decoder_picture_complete, decoder);

#ifdef HAVE_CUDA
  schro_async_add_exec_domain (decoder->async, SCHRO_EXEC_DOMAIN_CUDA);
  decoder->use_cuda = TRUE;
#endif

#ifdef HAVE_OPENGL
  decoder->opengl = schro_opengl_new ();

  if (schro_opengl_is_usable (decoder->opengl)) {
    schro_async_add_exec_domain (decoder->async, SCHRO_EXEC_DOMAIN_OPENGL);

    decoder->use_opengl = TRUE;
  } else {
    schro_opengl_free (decoder->opengl);

    decoder->opengl = NULL;
    decoder->use_opengl = FALSE;
  }
#endif

  decoder->instance = schro_decoder_instance_new (decoder);

  return decoder;
}

/**
 * schro_decoder_free:
 * @decoder: decoder object
 *
 * Frees a decoder object.
 */
void
schro_decoder_free (SchroDecoder * decoder)
{
  if (decoder->async) {
    schro_async_free (decoder->async);
  }

  do {
    SchroDecoderInstance *next = decoder->instance->next;
    schro_decoder_instance_free (decoder->instance);
    decoder->instance = next;
  } while (decoder->instance);

  schro_buflist_free (decoder->input_buflist);
  schro_parse_sync_free (decoder->sps);

  if (decoder->error_message)
    schro_free (decoder->error_message);

#ifdef HAVE_OPENGL
  if (decoder->opengl)
    schro_opengl_free (decoder->opengl);
#endif

  if (decoder->cpu_domain)
    schro_memory_domain_free (decoder->cpu_domain);
  if (decoder->cuda_domain)
    schro_memory_domain_free (decoder->cuda_domain);
  if (decoder->opengl_domain)
    schro_memory_domain_free (decoder->opengl_domain);


  schro_free (decoder);
}

/**
 * schro_decoder_begin_sequence:
 *
 * Prepare the @decoder to accept a new sequence.
 *
 * Returns SCHRO_DECODER_OK
 *         SCHRO_DECODER_ERROR: if oldest sequence hasn't been marked with
 *                              end of sequence
 */
int
schro_decoder_begin_sequence (SchroDecoder * decoder)
{
  SchroDecoderInstance *instance = decoder->instance;
  /* find newest sequence */
  while (instance->next)
    instance = instance->next;

  if (!instance->flushing || !instance->end_of_stream) {
    return SCHRO_DECODER_ERROR;
  }

  schro_async_lock (decoder->async);
  instance->next = schro_decoder_instance_new (decoder);
  schro_async_unlock (decoder->async);

  return SCHRO_DECODER_OK;
}

/**
 * schro_decoder_end_sequence:
 *
 * When decoder_wait() returns SCHRO_DECODER_EOS, the sequence
 * that terminated must be cleaned up before the state machine
 * will advance.
 *
 * Returns SCHRO_DECODER_OK:
 *         SCHRO_DECODER_ERROR: if oldest sequence has not completed.
 *         SCHRO_DECODER_ERROR: if the decoder has not been prepared
 *                              for a new sequence.
 */
int
schro_decoder_end_sequence (SchroDecoder * decoder)
{
  /* oldest sequence */
  SchroDecoderInstance *instance = decoder->instance;
  SchroDecoderInstance *old;

  if (!instance->flushing || !instance->end_of_stream ||
      instance->reorder_queue->n > 0) {
    /* can not guarantee no threads are using the instance */
    return SCHRO_DECODER_ERROR;
  }

  if (!instance->next) {
    /* decoder has no new instance waiting */
    return SCHRO_DECODER_ERROR;
  }

  schro_async_lock (decoder->async);
  old = instance->next;
  schro_decoder_instance_free (instance);
  decoder->instance = old;
  schro_async_unlock (decoder->async);

  return SCHRO_DECODER_OK;
}

/**
 * schro_picture_new:
 * @decoder: a decoder object
 *
 * Creates a new picture for @decoder.
 *
 * Internal API.
 *
 * Returns: a new picture
 */
SchroPicture *
schro_picture_new (SchroDecoderInstance * instance)
{
  SchroDecoder *decoder = instance->decoder;
  SchroPicture *picture;
  SchroFrameFormat frame_format;
  SchroVideoFormat *video_format = &instance->video_format;
  int picture_width, picture_height;
  int iwt_width, iwt_height;
  int picture_chroma_width, picture_chroma_height;

  picture = schro_malloc0 (sizeof (SchroPicture));
  picture->refcount = 1;

  picture->decoder_instance = instance;

  picture->params.video_format = video_format;

  frame_format = schro_params_get_frame_format (16,
      video_format->chroma_format);
  schro_video_format_get_picture_chroma_size (video_format,
      &picture_chroma_width, &picture_chroma_height);

  picture_width = video_format->width;
  picture_height = schro_video_format_get_picture_height (video_format);

  schro_video_format_get_iwt_alloc_size (video_format, &iwt_width,
      &iwt_height, SCHRO_LIMIT_TRANSFORM_DEPTH);

  if (decoder->use_cuda) {
    picture->transform_frame = schro_frame_new_and_alloc (decoder->cpu_domain,
        frame_format, iwt_width, iwt_height);
#if 0
    /* These get allocated later, while in the CUDA thread */
    picture->mc_tmp_frame = schro_frame_new_and_alloc (decoder->cuda_domain,
        frame_format, frame_width, frame_height);
    picture->frame = schro_frame_new_and_alloc (decoder->cuda_domain,
        frame_format, frame_width, frame_height);
    picture->planar_output_frame =
        schro_frame_new_and_alloc (decoder->cuda_domain, frame_format,
        video_format->width, video_format->height);
#endif
  } else if (decoder->use_opengl) {
    picture->transform_frame = schro_frame_new_and_alloc (decoder->cpu_domain,
        frame_format, iwt_width, iwt_height);
    picture->planar_output_frame =
        schro_frame_new_and_alloc (decoder->cpu_domain,
        schro_params_get_frame_format (8, video_format->chroma_format),
        video_format->width, video_format->height);
  } else {
    picture->mc_tmp_frame = schro_frame_new_and_alloc (decoder->cpu_domain,
        frame_format, picture_width, picture_height);
    picture->frame = schro_frame_new_and_alloc (decoder->cpu_domain,
        frame_format, iwt_width, iwt_height);
    picture->transform_frame = schro_frame_ref (picture->frame);
  }

  SCHRO_DEBUG ("planar output frame %dx%d",
      video_format->width, video_format->height);

  return picture;
}

SchroPicture *
schro_picture_ref (SchroPicture * picture)
{
  picture->refcount++;
  return picture;
}

void
schro_picture_unref (SchroPicture * picture)
{
  SCHRO_ASSERT (picture->refcount > 0);
  picture->refcount--;
  if (picture->refcount == 0) {
    int i;
    int component;

    SCHRO_DEBUG ("freeing picture %p", picture);
    for (component = 0; component < 3; component++) {
      for (i = 0; i < SCHRO_LIMIT_SUBBANDS; i++) {
        if (picture->subband_buffer[component][i]) {
          schro_buffer_unref (picture->subband_buffer[component][i]);
          picture->subband_buffer[component][i] = NULL;
        }
      }
    }
    for (i = 0; i < 9; i++) {
      if (picture->motion_buffers[i]) {
        schro_buffer_unref (picture->motion_buffers[i]);
        picture->motion_buffers[i] = NULL;
      }
    }
    if (picture->lowdelay_buffer)
      schro_buffer_unref (picture->lowdelay_buffer);

    if (picture->transform_frame)
      schro_frame_unref (picture->transform_frame);
    if (picture->frame)
      schro_frame_unref (picture->frame);
    if (picture->mc_tmp_frame)
      schro_frame_unref (picture->mc_tmp_frame);
    if (picture->planar_output_frame)
      schro_frame_unref (picture->planar_output_frame);
    if (picture->output_picture)
      schro_frame_unref (picture->output_picture);
    if (picture->motion)
      schro_motion_free (picture->motion);
    if (picture->input_buffer)
      schro_buffer_unref (picture->input_buffer);
    if (picture->upsampled_frame)
      schro_upsampled_frame_free (picture->upsampled_frame);
    if (picture->ref0)
      schro_picture_unref (picture->ref0);
    if (picture->ref1)
      schro_picture_unref (picture->ref1);
    if (picture->ref_output_frame)
      schro_frame_unref (picture->ref_output_frame);

    if (picture->tag)
      schro_tag_free (picture->tag);

    schro_free (picture);
  }
}

/**
 * schro_decoder_reset:
 * @decoder: a decoder object
 *
 * Resets the internal state of the decoder.  This function should be
 * called after a discontinuity of the stream, for example, as the
 * result of a seek.
 */
void
schro_decoder_reset (SchroDecoder * decoder)
{
  schro_async_stop (decoder->async);

  schro_buflist_free (decoder->input_buflist);
  decoder->input_buflist = schro_buflist_new ();

  schro_parse_sync_free (decoder->sps);
  decoder->sps = schro_parse_sync_new ();

  schro_decoder_instance_free (decoder->instance);
  decoder->instance = schro_decoder_instance_new (decoder);

  decoder->error = FALSE;

  schro_async_start (decoder->async);
}

/**
 * schro_decoder_get_video_format:
 * @decoder: a decoder object
 *
 * Returns a structure containing information on the video format being
 * decoded by the decoder.  This structure should be freed using free()
 * when it is no longer needed.
 *
 * Returns: a video format structure
 */
/* xxx: this is nolonger api/client thread safe with respect to reset */
SchroVideoFormat *
schro_decoder_get_video_format (SchroDecoder * decoder)
{
  SchroVideoFormat *format;

  /* FIXME check that decoder is in the right state */

  format = malloc (sizeof (SchroVideoFormat));
  memcpy (format, &decoder->instance->video_format, sizeof (SchroVideoFormat));

  return format;
}

/**
 * schro_decoder_get_picture_number:
 * @decoder: a decoder object
 *
 * Returns the picture number of the next picture that will be returned
 * by @schro_decoder_pull().
 *
 * Returns: a picture number
 */
SchroPictureNumber
schro_decoder_get_picture_number (SchroDecoder * decoder)
{
  SchroDecoderInstance *instance = decoder->instance;
  SchroPicture *picture = NULL;

  if (instance->reorder_queue->n >= instance->reorder_queue_size ||
      instance->flushing) {
    picture = schro_queue_peek (instance->reorder_queue);
  }
  if (picture)
    return picture->picture_number;
  return SCHRO_PICTURE_NUMBER_INVALID;
}

/**
 * schro_decoder_get_picture_tag:
 * @decoder: a decoder object
 *
 * Returns any tag associated with the next picture to be returned
 * by @schro_decoder_pull().  Ownership of the tag is transfered to
 * the caller.
 *
 * Returns: a tag represented by void* or NULL
 */
SchroTag *
schro_decoder_get_picture_tag (SchroDecoder * decoder)
{
  SchroDecoderInstance *instance = decoder->instance;
  SchroPicture *picture = NULL;

  if (instance->reorder_queue->n >= instance->reorder_queue_size ||
      instance->flushing) {
    picture = schro_queue_peek (instance->reorder_queue);
  }
  if (picture) {
    SchroTag *tag = picture->tag;
    picture->tag = NULL;
    return tag;
  }
  return NULL;
}

/**
 * schro_decoder_add_output_picture:
 * @decoder: a decoder object
 * @frame: the frame to add to the picture queue
 *
 * Adds a frame provided by the application to the picture queue.
 * Frames in the picture queue will be used for decoding images, and
 * are eventually returned to the application by schro_decoder_pull().
 *
 * The caller loses its reference to @frame after calling this
 * function.
 */
void
schro_decoder_add_output_picture (SchroDecoder * decoder, SchroFrame * frame)
{
  schro_async_lock (decoder->async);
  schro_queue_add (decoder->instance->output_queue, frame, 0);
  schro_async_signal_scheduler (decoder->async);
  schro_async_unlock (decoder->async);
}

/**
 * schro_decoder_set_earliest_frame:
 * @decoder: a decoder object
 * @earliest_frame: the earliest frame that the application is interested in
 *
 * The application can tell the decoder the earliest frame it is
 * interested in by calling this function.  Subsequent calls to
 * schro_decoder_pull() will only return pictures with picture
 * numbers greater than or equal to this number.  The decoder will
 * avoid decoding pictures that will not be displayed or used as
 * reference pictures.
 *
 * This feature can be used for frame-accurate seeking.
 *
 * This function can be called at any time during decoding.  Calling
 * this function with a picture number less than the current earliest
 * frame setting is invalid.
 */
void
schro_decoder_set_earliest_frame (SchroDecoder * decoder,
    SchroPictureNumber earliest_frame)
{
  decoder->earliest_frame = earliest_frame;
}

/**
 * schro_decoder_set_skip_ratio:
 * @decoder: a decoder object
 * @ratio: skip ratio.
 *
 * Sets the skip ratio of the decoder.  The skip ratio is used by the
 * decoder to skip decoding of some pictures.  Reference pictures are
 * always decoded.
 *
 * A picture is skipped when the running average of the proportion of
 * pictures skipped is less than the skip ratio.  Reference frames are
 * always decoded and contribute to the running average.  Thus, the
 * actual ratio of skipped pictures may be larger than the requested
 * skip ratio.
 *
 * The decoder indicates a skipped picture in the pictures returned
 * by @schro_decoder_pull() by a frame that has a width and height of
 * 0.
 *
 * The default skip ratio is 1.0, indicating that all pictures should
 * be decoded.  A skip ratio of 0.0 indicates that no pictures should
 * be decoded, although as mentioned above, some pictures will be
 * decoded anyway.  Values outside the range of 0.0 to 1.0 are quietly
 * clamped to that range.
 *
 * This function may be called at any time during decoding.
 */
void
schro_decoder_set_skip_ratio (SchroDecoder * decoder, double ratio)
{
  if (ratio > 1.0)
    ratio = 1.0;
  if (ratio < 0.0)
    ratio = 0.0;
  decoder->skip_ratio = ratio;
}

void
schro_decoder_set_picture_order (SchroDecoder * decoder, int order)
{
  SchroDecoderInstance *instance;

  decoder->coded_order = order == SCHRO_DECODER_PICTURE_ORDER_CODED;

  /* propagate to all instances */
  for (instance = decoder->instance; instance; instance = instance->next) {
    if (instance->have_sequence_header) {
      SCHRO_ERROR ("Don't call this function after decoding has commenced");
    }
    schro_decoder_set_rob_size (instance);
  }
}

static int
schro_decoder_pull_is_ready_locked (SchroDecoder * decoder)
{
  SchroDecoderInstance *instance = decoder->instance;
  SchroPicture *picture = NULL;

  /* Not possible to pull from the RoB if not full */
  /* NB, schro's RoB implementation can be larger than the spec */
  if (instance->reorder_queue->n >= instance->reorder_queue_size ||
      instance->flushing) {
    picture = schro_queue_peek (instance->reorder_queue);
  }

  if (!picture || !picture->stages[SCHRO_DECODER_STAGE_DONE].is_done) {
    /* nothing avaliable, give up */
    return FALSE;
  }
  if (!schro_decoder_frame_is_twofield (instance, picture->output_picture)) {
    /* one picture is avaliable, sufficient for:
     *  - frame_coding
     *  - field_coding and user supplies fields */
    return TRUE;
  }
  /* interlaced_coding with twofiled output: must be two pictures avaliable */
  if (instance->flushing) {
    /* except with broken streams and flushing.  there may only be 1,
     * in which case, we are as ready as we'll ever be, and leave it
     * up to _pull to fix missing field issue */
    if (instance->reorder_queue->n == 1)
      return TRUE;
  }
  SCHRO_ASSERT (instance->reorder_queue->n >= 2);

  /* don't check if the second field is the pair to the first, since the
   * RoB is full, it would cause a deadlock to block on there being a
   * valid pair at the front of the RoB. */
  picture = instance->reorder_queue->elements[1].data;
  /* is second field ready ? */
  return picture->stages[SCHRO_DECODER_STAGE_DONE].is_done;
}

/**
 * schro_decoder_pull:
 * @decoder: a decoder object
 *
 * Removes the next picture from the picture queue and returns a frame
 * containing the image.
 *
 * The application provides the frames that pictures are decoded into,
 * and the same frames are returned from this function.  However, the
 * order of frames returned may be different than the order that the
 * application provides the frames to the decoder.
 *
 * An exception to this is that skipped frames are indicated by a
 * frame having a height and width equal to 0.  This frame is created
 * using @schro_frame_new(), and is not one of the frames provided by
 * the application.
 *
 * Frames should be freed using @schro_frame_unref() when no longer
 * needed.  The frame must not be reused by the application, since it
 * may contain a reference frame still in use by the decoder.
 *
 * Returns: the next picture
 */
SchroFrame *
schro_decoder_pull (SchroDecoder * decoder)
{
  SchroDecoderInstance *instance = decoder->instance;
  SchroPicture *picture = NULL;
  SchroPictureNumber picture_number;
  SchroFrame *frame;

  schro_async_lock (decoder->async);

  if (schro_decoder_pull_is_ready_locked (decoder)) {
    picture = schro_queue_pull (instance->reorder_queue);
  }

  if (!picture) {
    return NULL;
  }

  /* XXX would be nice to warn if expected picture not present */
  frame = schro_frame_ref (picture->output_picture);
  picture_number = picture->picture_number;
  schro_picture_unref (picture);
  picture = NULL;

  if (schro_decoder_frame_is_twofield (instance, frame))
    do {
      /* only consider the 2nd field if it can reference
       * picture->output_picture, ie frame is twofields */
      if (picture_number & 1) {
        /* The following is voilated:
         * - 10.4p3 earliest field in each frame shall have an even picture number
         * Then the head of the reorder_queue can't be the pair to picture */
        break;
      }

      picture = schro_queue_peek (decoder->instance->reorder_queue);
      if (!picture) {
        if (instance->flushing) {
          /* when flushing, a broken stream might not have a pair of
           * pictures: discard the frame, (otherwise at a broken sequence
           * boundry, could cause two frames to be emitted rather than one) */
          schro_frame_unref (frame);
          frame = NULL;
          break;
        }
        SCHRO_ASSERT (picture);
      }
      if (picture_number + 1 != picture->picture_number) {
        /* The second field in the frame can only be a pair to the first if
         * they have consecutive picture numbers */
        break;
      }

      /* second field is the pair to the first: discard it */
      picture = schro_queue_pull (decoder->instance->reorder_queue);
      picture_number = picture->picture_number;
      schro_picture_unref (picture);
      picture = NULL;
    } while (0);

  instance->last_picture_number = picture_number;
  instance->last_picture_number_valid = TRUE;

  schro_async_unlock (decoder->async);

  return frame;
}

/**
 * schro_decoder_push_ready:
 * @decoder: a decoder object
 *
 * This function is used by the application to determine if it should push
 * more data to the decoder.
 *
 * Returns: TRUE if the decoder is ready for more data
 */
int
schro_decoder_push_ready (SchroDecoder * decoder)
{
  int ret;
  SchroDecoderInstance *instance = decoder->instance;
  /* by definition, all instances prior to the last are flushing, just
   * check if the final sequence not full */
  while (instance->next)
    instance = instance->next;

  schro_async_lock (decoder->async);
  /* may not push more data if the decoder is flushing:
   *   This occurs when an EOS is pushed and no new sequence is prepared */
  ret = !instance->flushing
      && !schro_queue_is_full (decoder->instance->reorder_queue);
  schro_async_unlock (decoder->async);

  return ret;
}

static int
schro_decoder_need_output_frame_locked (SchroDecoder * decoder)
{
  /* output frames are only required to satisfy completion of the
   * oldest sequence. Do not iterate over instances */
  SchroDecoderInstance *instance = decoder->instance;
  int num_frames_in_hand = instance->output_queue->n;
  int i;
  if (schro_queue_is_full (instance->output_queue)) {
    return 0;
  }
  if (instance->video_format.interlaced_coding) {
    for (i = 0; i < instance->output_queue->n; i++) {
      SchroFrame *output_frame = instance->output_queue->elements[i].data;
      if (schro_decoder_frame_is_twofield (instance, output_frame)) {
        /* this frame counts for two pictures */
        num_frames_in_hand++;
      }
    }
  }
  for (i = 0; i < instance->reorder_queue->n; i++) {
    SchroPicture *picture = instance->reorder_queue->elements[i].data;
    if (!picture->output_picture)
      num_frames_in_hand--;
  }
  return num_frames_in_hand < 0;
}

int
schro_decoder_need_output_frame (SchroDecoder * decoder)
{
  int ret;

  schro_async_lock (decoder->async);
  ret = schro_decoder_need_output_frame_locked (decoder);
  schro_async_unlock (decoder->async);

  return ret;
}

static int
schro_decoder_get_status_locked (SchroDecoder * decoder)
{
  SchroDecoderInstance *instance = decoder->instance;
  /* NB, this function should be `pure' (no sideeffects),
   * since it is called in a context which should not update state */
  if (!instance) {
    /* Under normal operation, this shouldn't be possible:
     *  - only if the user uses the raw api in the wrong order */
    schro_decoder_error (decoder, "Missing decoder instance");
    return SCHRO_DECODER_ERROR;
  }
  if (instance->first_sequence_header) {
    return SCHRO_DECODER_FIRST_ACCESS_UNIT;
  }
  if (schro_decoder_pull_is_ready_locked (decoder)) {
    return SCHRO_DECODER_OK;
  }
  if (decoder->error) {
    return SCHRO_DECODER_ERROR;
  }
  if (instance->have_sequence_header &&
      schro_decoder_need_output_frame_locked (decoder)) {
    return SCHRO_DECODER_NEED_FRAME;
  }
  if (!schro_queue_is_full (instance->reorder_queue) && !instance->flushing) {
    return SCHRO_DECODER_NEED_BITS;
  }

  if (instance->flushing && !instance->reorder_queue->n) {
    if (instance->end_of_stream) {
      return SCHRO_DECODER_EOS;
    } else {
      return SCHRO_DECODER_STALLED;
    }
  }

  return SCHRO_DECODER_WAIT;
}

#if 0
static int
schro_decoder_get_status (SchroDecoder * decoder)
{
  int ret;

  schro_async_lock (decoder->async);
  ret = schro_decoder_get_status_locked (decoder);
  schro_async_unlock (decoder->async);

  return ret;
}
#endif

static void
schro_decoder_dump (SchroDecoder * decoder)
{
  SchroDecoderInstance *instance = decoder->instance;
  int i;

  SCHRO_ERROR ("index, picture_number, busy, state, needed_state, working");
  for (i = 0; i < instance->reorder_queue->n; i++) {
    SchroPicture *picture = instance->reorder_queue->elements[i].data;
    int done = 0, needed = 0, j;
    for (j = 0; j <= SCHRO_DECODER_STAGE_DONE; j++) {
      done |= picture->stages[j].is_done << j;
      needed |= picture->stages[j].is_needed << j;
    }

    SCHRO_ERROR ("%d: %d %d %04x %04x -",
        i, picture->picture_number, picture->busy, done, needed);
  }
  if (instance->reorder_queue->n >= instance->reorder_queue_size ||
      instance->flushing) {
    SCHRO_ERROR ("next_picture_number %d",
        schro_decoder_get_picture_number (decoder));
  } else {
    SCHRO_ERROR ("reorder_queue too empty to determine next_picture_number: "
        "needs: %d pictures",
        instance->reorder_queue_size - instance->reorder_queue->n);
  }
}

/**
 * schro_decoder_wait:
 * @decoder: a decoder object
 *
 * Waits until the decoder requires the application to do something,
 * e.g., push more data or remove a frame from the picture queue,
 * and then returns the decoder status.
 *
 * Returns: decoder status
 */
int
schro_decoder_wait (SchroDecoder * decoder)
{
  int ret;

  schro_async_lock (decoder->async);
  while (1) {
    ret = schro_decoder_get_status_locked (decoder);
    switch (ret) {
      default:
        goto cleanup;

      case SCHRO_DECODER_FIRST_ACCESS_UNIT:
        /* clean up state for next iteration */
        decoder->instance->first_sequence_header = FALSE;
        goto cleanup;

      case SCHRO_DECODER_WAIT:
        break;
    }

    ret = schro_async_wait_locked (decoder->async);
    if (!ret) {
      SCHRO_ERROR ("doh!");
      schro_decoder_dump (decoder);
      /* hack */
      schro_async_signal_scheduler (decoder->async);
      //SCHRO_ASSERT(0);
    }
  }
cleanup:
  schro_async_unlock (decoder->async);

  return ret;
}

/**
 * schro_decoder_autoparse_wait:
 * @decoder: a decoder object
 *
 * Waits until the decoder requires the application to do something,
 * e.g., push more data or remove a frame from the picture queue,
 * and then returns the decoder status.
 *
 * When using @schro_decoder_autoparse_push, this function should
 * be used instead of @schro_decoder_wait.
 *
 * Returns: decoder status
 */
int
schro_decoder_autoparse_wait (SchroDecoder * decoder)
{
  while (1) {
    int ret = schro_decoder_wait (decoder);
    switch (ret) {
      default:
        return ret;
      case SCHRO_DECODER_NEED_BITS:
        /* see if there is anything sitting in the input_buflist
         * before requesting more data */
        /* xxx, it would be good if this were moved to an idle thread */
        ret = schro_decoder_autoparse_push (decoder, NULL);
        if (ret == SCHRO_DECODER_NEED_BITS) {
          return ret;
        }
        break;
      case SCHRO_DECODER_EOS:
        /* delete the decoder instance -- it is finished with
         * This is safe since for the status to be EOS
         * there must be:
         *  - no work left in the decoder queue
         *  - no pictures left to retrieve
         * Therefore there are no threads running in this instance.
         */
        ret = schro_decoder_end_sequence (decoder);
        if (ret == SCHRO_DECODER_ERROR) {
          return SCHRO_DECODER_EOS;
        }
        /* try again to discover the next state: EOS never gets passed
         * to the user */
        continue;
    }
  }
}

int
schro_decoder_push_end_of_stream (SchroDecoder * decoder)
{
  SchroDecoderInstance *instance = decoder->instance;
  /* find newest sequence */
  while (instance->next)
    instance = instance->next;

  instance->flushing = TRUE;
  instance->end_of_stream = TRUE;

  return SCHRO_DECODER_EOS;
}

/**
 * schro_decoder_autoparse_push_end_of_sequence:
 *
 * Signals that the most recent sequence is to terminates: decoding
 * of all pictures still in progress will continue.
 *
 * The decoder state is prepared to immediately handle data for
 * a new sequence.
 *
 * Returns: SCHRO_DECODER_EOS
 */
int
schro_decoder_autoparse_push_end_of_sequence (SchroDecoder * decoder)
{
  schro_decoder_push_end_of_stream (decoder);
  schro_decoder_begin_sequence (decoder);
  return SCHRO_DECODER_EOS;
}

/**
 * schro_decoder_set_flushing:
 *
 * This function is depricated and has no effect.
 * If control over the picture order is required, consult
 * schro_decoder_set_picture_order()
 */
int
schro_decoder_set_flushing (SchroDecoder * decoder, int value)
{
  return SCHRO_DECODER_OK;
}

int
schro_decoder_push (SchroDecoder * decoder, SchroBuffer * buffer)
{
  SchroDecoderInstance *instance = decoder->instance;
  SchroUnpack unpack;
  int parse_code;

  if (!instance) {
    /* this should not be possible with the autoparse api:
     *  - decoder is created with an instance
     *  - instances are created eachtime an EOS is inserted
     *  - instances are deleted when EOS is pulled.
     * => Next instance is created before previous one is destroyed
     * Assumptions:
     *  - Only examined case where wait() and push() are called
     *    from same thread
     */
    return SCHRO_DECODER_ERROR;
  }

  /* find newest sequence */
  while (instance->next)
    instance = instance->next;
  /* instance is now the next instance to send data to */

  instance->flushing = FALSE;

  /* Buffers may have associated (tagged) private data.
   * This data is to be associated with the next(/this) picture data unit */
  if (buffer->tag) {
    if (decoder->next_picture_tag) {
      schro_tag_free (decoder->next_picture_tag);
    }
    decoder->next_picture_tag = buffer->tag;
  }
  buffer->tag = NULL;

  schro_unpack_init_with_data (&unpack, buffer->data, buffer->length, 1);
  parse_code = schro_decoder_decode_parse_header (&unpack);

  if (parse_code == -1) {
    schro_buffer_unref (buffer);
    return SCHRO_DECODER_ERROR;
  }

  if (parse_code == SCHRO_PARSE_CODE_SEQUENCE_HEADER) {
    int ret;

    SCHRO_INFO ("decoding sequence header");
    if (!instance->have_sequence_header) {
      schro_decoder_parse_sequence_header (instance, &unpack);
      /* safe to resize RoB, since nothing can be inflight in this instance */
      schro_decoder_set_rob_size (instance);
      instance->have_sequence_header = TRUE;
      instance->first_sequence_header = TRUE;
      instance->sequence_header_buffer = schro_buffer_dup (buffer);

      ret = SCHRO_DECODER_FIRST_ACCESS_UNIT;
    } else {
      if (schro_decoder_compare_sequence_header_buffer (buffer,
              instance->sequence_header_buffer)) {
        ret = SCHRO_DECODER_OK;
      } else {
        schro_decoder_error (decoder, "sequence header changed");
        ret = SCHRO_DECODER_ERROR;
      }
    }

    schro_buffer_unref (buffer);
    return ret;
  }

  if (parse_code == SCHRO_PARSE_CODE_AUXILIARY_DATA) {
    int code;

    code = schro_unpack_decode_bits (&unpack, 8);

    if (code == SCHRO_AUX_DATA_MD5_CHECKSUM) {
      int i;
      for (i = 0; i < 16; i++) {
        instance->md5_checksum[i] = schro_unpack_decode_bits (&unpack, 8);
      }
      instance->has_md5 = TRUE;
    }

    if (code == SCHRO_AUX_DATA_ENCODER_STRING) {
      char s[20];
      int i;

      for (i = 0; i < 20; i++) {
        s[i] = schro_unpack_decode_bits (&unpack, 8);
      }
      if (memcmp ("Schroedinger 1.0.", s, 17) == 0) {
        if (s[17] >= '0' && s[17] <= '7' && (s[18] == '.'
                || s[18] == (char) 0xff)) {
          SCHRO_WARNING ("turning on codeblock quantiser compatibility mode");
          instance->compat_quant_offset = TRUE;
        }
      }
    }

    schro_buffer_unref (buffer);
    return SCHRO_DECODER_OK;
  }

  if (SCHRO_PARSE_CODE_IS_PADDING (parse_code)) {
    schro_buffer_unref (buffer);
    return SCHRO_DECODER_OK;
  }

  if (SCHRO_PARSE_CODE_IS_END_OF_SEQUENCE (parse_code)) {
    SCHRO_DEBUG ("decoding end sequence");
    schro_buffer_unref (buffer);
    instance->end_of_stream = TRUE;
    instance->flushing = TRUE;
    return SCHRO_DECODER_EOS;
  }

  if (SCHRO_PARSE_CODE_IS_PICTURE (parse_code)) {
    if (!instance->have_sequence_header) {
      SCHRO_INFO ("no sequence header -- dropping picture");
      if (decoder->next_picture_tag) {
        schro_tag_free (decoder->next_picture_tag);
      }
      decoder->next_picture_tag = NULL;
      schro_buffer_unref (buffer);
      return SCHRO_DECODER_OK;
    }

    return schro_decoder_iterate_picture (instance, buffer, &unpack,
        parse_code);
  }

  schro_buffer_unref (buffer);
  return SCHRO_DECODER_ERROR;
}

/**
 * schro_decoder_autoparse_push:
 *
 * Appends @buffer@ to the internal input queue, extracting work
 * units as necessary.  There are no alignment constraints on the
 * contents of @buffer@; it is not required to be data unit aligned,
 * nor contain a single whole data unit.
 *
 * Returns: if decoder ready for more data, but insufficient input buffer:
 *          SCHRO_DECODER_NEED_BITS
 *          otherwise SCHRO_DECODER_OK
 */
int
schro_decoder_autoparse_push (SchroDecoder * decoder, SchroBuffer * buffer)
{
  /* 1. buffer is added to the decoder input queue.
   * 2. The synchronizer extracts (if possible) a data unit for parsing */
  if (buffer)
    schro_buflist_append (decoder->input_buflist, buffer);

  while (schro_decoder_push_ready (decoder)) {
    buffer = schro_parse_sync (decoder->sps, decoder->input_buflist);
    if (!buffer)
      return SCHRO_DECODER_NEED_BITS;

    switch (schro_decoder_push (decoder, buffer)) {
      default:
        break;
      case SCHRO_DECODER_EOS:
        /* Prepare decoder for immediately handling the next sequence */
        schro_decoder_begin_sequence (decoder);
        break;
    }
  }

  return SCHRO_DECODER_OK;
}

int
schro_decoder_iterate_picture (SchroDecoderInstance * instance,
    SchroBuffer * buffer, SchroUnpack * unpack, int parse_code)
{
  SchroDecoder *decoder = instance->decoder;
  SchroPicture *picture;
  SchroParams *params;

  picture = schro_picture_new (instance);
  params = &picture->params;

  picture->input_buffer = buffer;

  picture->tag = decoder->next_picture_tag;
  decoder->next_picture_tag = NULL;

  params->num_refs = SCHRO_PARSE_CODE_NUM_REFS (parse_code);
  params->is_lowdelay = SCHRO_PARSE_CODE_IS_LOW_DELAY (parse_code);
  params->is_noarith = !SCHRO_PARSE_CODE_USING_AC (parse_code);
  picture->is_ref = SCHRO_PARSE_CODE_IS_REFERENCE (parse_code);

  if (instance->has_md5) {
    picture->has_md5 = TRUE;
    memcpy (picture->md5_checksum, instance->md5_checksum, 16);
    instance->has_md5 = FALSE;
  }

  schro_decoder_parse_picture_header (picture, unpack);

  SCHRO_DEBUG ("picturenumber: %u", picture->picture_number);

  if (picture->is_ref) {
    /* xxx: why is this locked? -- no decoding should actually consult the
     * reference buffer surely? */
    schro_async_lock (instance->decoder->async);
    schro_decoder_reference_retire (instance, picture->retired_picture_number);
    schro_decoder_reference_add (instance, picture);
    schro_async_unlock (instance->decoder->async);
  }
  schro_decoder_parse_picture (picture, unpack);

  if (picture->error) {
    SCHRO_WARNING ("skipping because of error");
    picture->skip = TRUE;
  }

  if (instance->last_picture_number_valid) {
    if (schro_picture_n_before_m (picture->picture_number,
            instance->last_picture_number)) {
      /* picture logically occurs before a picture that has already been
       * emitted. The stream must have jumped backwards. It is important
       * to detect this, otherwise pictures can get stuck in the reorder
       * buffer.  NB, it isn't required to detect a forwards jump, since
       * this will cause the reorder buffer to flush naturally */
      /* xxx: future improvements would be to also note if a sequence header
       * occured just before this picture, and to inject that into the new
       * instance */
      SCHRO_WARNING ("stream jumped backwards, %u before %u, treating as EOS",
          picture->picture_number, instance->last_picture_number);
      schro_picture_unref (picture);
      schro_decoder_push_end_of_stream (decoder);
      return SCHRO_DECODER_EOS;
    }
  }

  /* todo: prune pictures that are inaccessible in the first gop */

  if (!instance->video_format.interlaced_coding &&
      !picture->is_ref && picture->picture_number < decoder->earliest_frame) {
    picture->skip = TRUE;
    SCHRO_INFO ("skipping frame %d (early)", picture->picture_number);
  }

  if (!instance->video_format.interlaced_coding &&
      !picture->is_ref && decoder->skip_value > decoder->skip_ratio) {
    decoder->skip_value = (1 - SCHRO_SKIP_TIME_CONSTANT) * decoder->skip_value;
    SCHRO_INFO ("skipping frame %d", picture->picture_number);
    SCHRO_DEBUG ("skip value %g ratio %g", decoder->skip_value,
        decoder->skip_ratio);

    picture->skip = TRUE;
  } else {
    decoder->skip_value = (1 - SCHRO_SKIP_TIME_CONSTANT) * decoder->skip_value +
        SCHRO_SKIP_TIME_CONSTANT;
  }
  SCHRO_DEBUG ("skip value %g ratio %g", decoder->skip_value,
      decoder->skip_ratio);

  if (picture->skip) {
    picture->output_picture = schro_frame_new ();
    if (picture->is_ref) {
      SchroFrameFormat frame_format;
      SchroFrame *ref;

      frame_format = schro_params_get_frame_format (8,
          params->video_format->chroma_format);
      ref =
          schro_frame_new_and_alloc_extended (decoder->cpu_domain, frame_format,
          instance->video_format.width,
          schro_video_format_get_picture_height (&instance->video_format), 32);
      schro_frame_clear (ref);
      picture->upsampled_frame = schro_upsampled_frame_new (ref);
    }

    SCHRO_DEBUG ("adding %d to queue (skipped)", picture->picture_number);

    picture->stages[SCHRO_DECODER_STAGE_DONE].is_done = TRUE;
    picture->stages[SCHRO_DECODER_STAGE_DONE].is_needed = TRUE;
  }

  schro_async_lock (decoder->async);
  SCHRO_DEBUG ("adding %d to queue", picture->picture_number);
  schro_picturequeue_rob_insert (instance->reorder_queue, picture,
      instance->reorder_queue_size);
  schro_async_signal_scheduler (decoder->async);
  schro_async_unlock (decoder->async);

  return SCHRO_DECODER_OK;
}

void
schro_decoder_parse_picture (SchroPicture * picture, SchroUnpack * unpack)
{
  SchroParams *params = &picture->params;

  if (params->num_refs > 0) {
    SCHRO_DEBUG ("inter");

    schro_async_lock (picture->decoder_instance->decoder->async);
    picture->ref0 =
        schro_decoder_reference_get (picture->decoder_instance,
        picture->reference1);
    if (picture->ref0 == NULL) {
      SCHRO_WARNING ("ref0 not found");
      picture->error = TRUE;
      schro_async_unlock (picture->decoder_instance->decoder->async);
      return;
    }
    schro_picture_ref (picture->ref0);

    picture->ref1 = NULL;
    if (params->num_refs > 1) {
      picture->ref1 =
          schro_decoder_reference_get (picture->decoder_instance,
          picture->reference2);
      if (picture->ref1 == NULL) {
        SCHRO_WARNING ("ref1 not found");
        picture->error = TRUE;
        schro_async_unlock (picture->decoder_instance->decoder->async);
        return;
      }
      schro_picture_ref (picture->ref1);
    }
    schro_async_unlock (picture->decoder_instance->decoder->async);

    schro_unpack_byte_sync (unpack);
    schro_decoder_parse_picture_prediction_parameters (picture, unpack);

    if (!picture->error) {
      schro_params_calculate_mc_sizes (params);
    }

    schro_unpack_byte_sync (unpack);
    schro_decoder_parse_block_data (picture, unpack);
  }

  schro_unpack_byte_sync (unpack);
  picture->zero_residual = FALSE;
  if (params->num_refs > 0) {
    picture->zero_residual = schro_unpack_decode_bit (unpack);

    SCHRO_DEBUG ("zero residual %d", picture->zero_residual);
  }

  if (!picture->zero_residual) {
    schro_decoder_parse_transform_parameters (picture, unpack);
    schro_params_calculate_iwt_sizes (params);

    schro_unpack_byte_sync (unpack);
    if (params->is_lowdelay) {
      schro_decoder_parse_lowdelay_transform_data (picture, unpack);
    } else {
      schro_decoder_parse_transform_data (picture, unpack);

      if (picture->decoder_instance->decoder->use_opengl) {
#ifdef HAVE_OPENGL
        schro_decoder_init_subband_frame_data (picture);
#else
        SCHRO_ASSERT (0);
#endif
      } else {
        schro_decoder_init_subband_frame_data_interleaved (picture);
      }
    }
  }

  if (picture->error)
    return;

  picture->stages[SCHRO_DECODER_STAGE_REFERENCES].is_needed = TRUE;
  picture->stages[SCHRO_DECODER_STAGE_MOTION_DECODE].is_needed = TRUE;
  picture->stages[SCHRO_DECODER_STAGE_MOTION_RENDER].is_needed = TRUE;
  picture->stages[SCHRO_DECODER_STAGE_RESIDUAL_DECODE].is_needed = TRUE;
  picture->stages[SCHRO_DECODER_STAGE_WAVELET_TRANSFORM].is_needed = TRUE;
  picture->stages[SCHRO_DECODER_STAGE_COMBINE].is_needed = TRUE;
}

/**
 * schro_decoder_assign_output_picture:
 * @picture that needs consideration of its output_picture
 * @robpos position of picture in reorder buffer
 *
 * Assign an output picture to @picture, taking into account field_coding
 * and the sharing of a frame between fields.
 */
static void
schro_decoder_assign_output_picture (SchroPicture * picture, int robpos)
{
  SchroDecoderInstance *instance = picture->decoder_instance;

  if (picture->output_picture)
    return;

  if (instance->video_format.interlaced_coding)
    do {
      SchroPicture *pair_picture;
      int pair_robpos;
      /* field coding: one frame is shared between two field pictures.
       * has the pair to this field already got a frame allocated?
       *  - If so, take a reference to the already allocated frame.
       *  - 10.4p3 earliest field in each frame shall have an even picture number
       *  => for even pictures, pair picture_number+1, for odd pictures, -1.
       * The action of the RoB guarantees that the pair will eventually be
       * sitting next to each other in the RoB. */
      /* find which RoB position the pair to this picture should reside: */
      pair_robpos = robpos + (picture->picture_number & 1 ? -1 : 1);
      if (pair_robpos < 0 || pair_robpos >= instance->reorder_queue->n)
        break;
      pair_picture = instance->reorder_queue->elements[pair_robpos].data;
      if (pair_picture->picture_number != (picture->picture_number ^ 1))
        break;
      /* picture_pair and picture are not only next to each other
       * in the RoB, but also belong to the same frame. */
      if (!pair_picture->output_picture)
        break;
      /* only reference the pairs output picture if its frame sized,
       * ie, the user supplied a frame and not a field */
      if (!schro_decoder_frame_is_twofield (instance,
              pair_picture->output_picture))
        break;

      /* pair_picture->output_picture is full frame */
      picture->output_picture = schro_frame_ref (pair_picture->output_picture);
    } while (0);

  if (!picture->output_picture) {
    /* either frame coding, or first field to be decoded of a frame */
    picture->output_picture = schro_queue_pull (instance->output_queue);
    /* NB, there may not be anything in the output_queue */
  }
}

void
schro_decoder_picture_complete (SchroAsyncStage * stage)
{
  SchroPicture *picture = (SchroPicture *) stage->priv;

  SCHRO_DEBUG ("picture complete");

  stage->is_done = TRUE;
  if (stage == picture->stages + SCHRO_DECODER_STAGE_COMBINE) {
    picture->stages[SCHRO_DECODER_STAGE_DONE].is_done = TRUE;
  }
  picture->busy = FALSE;

  schro_picture_unref (picture);
}

static int
schro_decoder_async_schedule (SchroDecoder * decoder,
    SchroExecDomain exec_domain)
{
  int i;
  int render_ok;
  int decode_ok;

  SCHRO_DEBUG ("schedule");

  if (decoder->use_cuda || decoder->use_opengl) {
    if (exec_domain == SCHRO_EXEC_DOMAIN_CUDA ||
        exec_domain == SCHRO_EXEC_DOMAIN_OPENGL) {
      decode_ok = FALSE;
      render_ok = TRUE;
    } else {
      decode_ok = TRUE;
      render_ok = FALSE;
    }
  } else {
    decode_ok = TRUE;
    render_ok = TRUE;
  }

  for (i = 0; i < decoder->instance->reorder_queue->n; i++) {
    SchroPicture *picture = decoder->instance->reorder_queue->elements[i].data;
    void *func = NULL;
    int stage = 0;

    if (picture->busy)
      continue;

    SCHRO_DEBUG ("picture %d", picture->picture_number);

#define TODO(stage) \
    (picture->stages[(stage)].is_needed && \
     !picture->stages[(stage)].is_done)
    if (TODO (SCHRO_DECODER_STAGE_REFERENCES)) {
      int j;
      int refs_ready = TRUE;

      for (j = 0; j < picture->params.num_refs; j++) {
        SchroPicture *refpic;

        refpic = (j == 0) ? picture->ref0 : picture->ref1;

        if (refpic->busy || !(refpic->stages[SCHRO_DECODER_STAGE_DONE].is_done)) {
          refs_ready = FALSE;
          continue;
        }

        if (
#ifdef HAVE_CUDA
            1 &&
#else
            picture->params.mv_precision > 0 &&
#endif
            !(refpic->stages[SCHRO_DECODER_STAGE_UPSAMPLE].is_done)) {
          if (!render_ok) {
            refs_ready = FALSE;
            continue;
          }
          refpic->busy = TRUE;

          func = schro_decoder_x_upsample;
          schro_picture_ref (refpic);
          refpic->stages[SCHRO_DECODER_STAGE_UPSAMPLE].task_func = func;
          refpic->stages[SCHRO_DECODER_STAGE_UPSAMPLE].priv = refpic;
          schro_async_run_stage_locked (decoder->async,
              refpic->stages + SCHRO_DECODER_STAGE_UPSAMPLE);

          return TRUE;
        }
      }
      if (refs_ready) {
        picture->stages[SCHRO_DECODER_STAGE_REFERENCES].is_done = TRUE;
      }
    }


    if (TODO (SCHRO_DECODER_STAGE_RESIDUAL_DECODE) && decode_ok) {
      func = schro_decoder_x_decode_residual;
      stage = SCHRO_DECODER_STAGE_RESIDUAL_DECODE;
    } else if (TODO (SCHRO_DECODER_STAGE_WAVELET_TRANSFORM) &&
        picture->stages[SCHRO_DECODER_STAGE_RESIDUAL_DECODE].is_done
        && render_ok) {
      func = schro_decoder_x_wavelet_transform;
      stage = SCHRO_DECODER_STAGE_WAVELET_TRANSFORM;
    } else if (TODO (SCHRO_DECODER_STAGE_MOTION_DECODE) &&
        picture->stages[SCHRO_DECODER_STAGE_REFERENCES].is_done && decode_ok) {
      func = schro_decoder_x_decode_motion;
      stage = SCHRO_DECODER_STAGE_MOTION_DECODE;
    } else if (TODO (SCHRO_DECODER_STAGE_MOTION_RENDER) &&
        picture->stages[SCHRO_DECODER_STAGE_MOTION_DECODE].is_done &&
        picture->stages[SCHRO_DECODER_STAGE_WAVELET_TRANSFORM].is_done &&
        render_ok) {
      func = schro_decoder_x_render_motion;
      stage = SCHRO_DECODER_STAGE_MOTION_RENDER;
    } else if (TODO (SCHRO_DECODER_STAGE_COMBINE) &&
        picture->stages[SCHRO_DECODER_STAGE_WAVELET_TRANSFORM].is_done &&
        picture->stages[SCHRO_DECODER_STAGE_MOTION_RENDER].is_done
        && render_ok) {
      /* Eagerly unreference the ref picures.  Otherwise they are kept
       * until the picture dependency chain terminates (worst case, Ponly
       * coding = infinite dependency chain = -ENOMEM) */
      if (picture->ref0) {
        schro_picture_unref (picture->ref0);
        picture->ref0 = NULL;
      }
      if (picture->ref1) {
        schro_picture_unref (picture->ref1);
        picture->ref1 = NULL;
      }
      if (!picture->output_picture) {
        schro_decoder_assign_output_picture (picture, i);
      }
      if (picture->output_picture) {
        func = schro_decoder_x_combine;
        stage = SCHRO_DECODER_STAGE_COMBINE;
      }
    }

    if (func) {
      picture->busy = TRUE;

      schro_picture_ref (picture);

      picture->stages[stage].task_func = func;
      picture->stages[stage].priv = picture;
      schro_async_run_stage_locked (decoder->async, picture->stages + stage);

      return TRUE;
    }
  }

  return FALSE;
}

void
schro_decoder_x_decode_motion (SchroAsyncStage * stage)
{
  SchroPicture *picture = (SchroPicture *) stage->priv;
  SchroParams *params = &picture->params;

  if (params->num_refs > 0) {
    picture->motion = schro_motion_new (params, picture->ref0->upsampled_frame,
        picture->ref1 ? picture->ref1->upsampled_frame : NULL);
    schro_decoder_decode_block_data (picture);
  }
}

void
schro_decoder_x_render_motion (SchroAsyncStage * stage)
{
  SchroPicture *picture = (SchroPicture *) stage->priv;
  SchroParams *params = &picture->params;
  SchroDecoder *decoder = picture->decoder_instance->decoder;

  if (1) {
    SchroFrameFormat frame_format;

    frame_format = schro_params_get_frame_format (8,
        params->video_format->chroma_format);

    if (decoder->use_cuda) {
#ifdef HAVE_CUDA
      picture->ref_output_frame =
          schro_frame_new_and_alloc (decoder->cuda_domain, frame_format,
          picture->decoder_instance->video_format.width,
          schro_video_format_get_picture_height (&picture->
              decoder_instance->video_format));
#else
      SCHRO_ASSERT (0);
#endif
    } else if (decoder->use_opengl) {
#ifdef HAVE_OPENGL
      picture->ref_output_frame =
          schro_opengl_frame_new (decoder->opengl, decoder->opengl_domain,
          frame_format, picture->decoder_instance->video_format.width,
          schro_video_format_get_picture_height (&decoder->video_format));
#else
      SCHRO_ASSERT (0);
#endif
    } else {
      if (params->num_refs > 0 || picture->is_ref || picture->has_md5) {
        picture->ref_output_frame =
            schro_frame_new_and_alloc_extended (decoder->cpu_domain,
            frame_format, picture->decoder_instance->video_format.width,
            schro_video_format_get_picture_height (&picture->
                decoder_instance->video_format), 32);
      }
    }
  }

  if (params->num_refs > 0) {
    SCHRO_DEBUG ("motion render with %p and %p", picture->ref0, picture->ref1);
    if (decoder->use_cuda) {
#ifdef HAVE_CUDA
      int frame_width;
      int frame_height;
      int frame_format;
      SchroVideoFormat *video_format = params->video_format;

      frame_format = schro_params_get_frame_format (16,
          video_format->chroma_format);
      frame_width = ROUND_UP_POW2 (video_format->width,
          SCHRO_LIMIT_TRANSFORM_DEPTH +
          SCHRO_CHROMA_FORMAT_H_SHIFT (video_format->chroma_format));
      frame_height = ROUND_UP_POW2 (video_format->height,
          SCHRO_LIMIT_TRANSFORM_DEPTH +
          SCHRO_CHROMA_FORMAT_V_SHIFT (video_format->chroma_format));

      picture->mc_tmp_frame = schro_frame_new_and_alloc (decoder->cuda_domain,
          frame_format, frame_width, frame_height);
      schro_motion_render_cuda (picture->motion, picture->mc_tmp_frame);
#else
      SCHRO_ASSERT (0);
#endif
    } else if (decoder->use_opengl) {
#ifdef HAVE_OPENGL
      SchroFrameFormat frame_format;
      int picture_width;
      int picture_height;

      frame_format = schro_params_get_frame_format (16,
          params->video_format->chroma_format);
      picture_width = params->video_format->width;
      picture_height =
          schro_video_format_get_picture_height (params->video_format);

      picture->mc_tmp_frame = schro_opengl_frame_new (decoder->opengl,
          decoder->opengl_domain, frame_format, picture_width, picture_height);

      schro_opengl_motion_render (picture->motion, picture->mc_tmp_frame);
#else
      SCHRO_ASSERT (0);
#endif
    } else {
      schro_motion_render (picture->motion, picture->mc_tmp_frame,
          picture->frame, TRUE, picture->ref_output_frame);
    }
  } else {
    if (params->num_refs > 0 || picture->is_ref || picture->has_md5) {
      schro_frame_convert (picture->ref_output_frame, picture->frame);
    }
  }
}

void
schro_decoder_x_decode_residual (SchroAsyncStage * stage)
{
  SchroPicture *picture = (SchroPicture *) stage->priv;
  SchroParams *params = &picture->params;

  if (!picture->zero_residual) {
    if (params->is_lowdelay) {
      schro_decoder_decode_lowdelay_transform_data (picture);
    } else {
      schro_decoder_decode_transform_data (picture);
    }
  }
}

void
schro_decoder_x_wavelet_transform (SchroAsyncStage * stage)
{
  SchroPicture *picture = (SchroPicture *) stage->priv;
  SchroDecoder *decoder = picture->decoder_instance->decoder;

  if (!picture->zero_residual) {
    if (decoder->use_cuda) {
      picture->frame = schro_frame_clone (decoder->cuda_domain,
          picture->transform_frame);
#ifdef HAVE_CUDA
      schro_frame_inverse_iwt_transform_cuda (picture->frame,
          picture->transform_frame, &picture->params);
#else
      SCHRO_ASSERT (0);
#endif
    } else if (decoder->use_opengl) {
#ifdef HAVE_OPENGL
      picture->frame
          = schro_opengl_frame_clone_and_push (decoder->opengl,
          decoder->opengl_domain, picture->transform_frame);

      schro_opengl_frame_inverse_iwt_transform (picture->frame,
          &picture->params);
#else
      SCHRO_ASSERT (0);
#endif
    } else {
      schro_frame_inverse_iwt_transform (picture->frame, &picture->params);
    }
  }
}

void
schro_decoder_x_combine (SchroAsyncStage * stage)
{
  SchroPicture *picture = (SchroPicture *) stage->priv;
  SchroParams *params = &picture->params;
  SchroDecoder *decoder = picture->decoder_instance->decoder;
#if defined(HAVE_CUDA) || defined(HAVE_OPENGL)
  SchroFrame *combined_frame;
  SchroFrame *output_frame;
#endif
  /*> output_picture: a view of picture->output_picture, that is modified
   * when field coding.  Be very careful referencing this, no reference to
   * output_picture may leave this function */
  SchroFrame output_picture = *picture->output_picture;

#if defined(HAVE_CUDA) || defined(HAVE_OPENGL)
  if (picture->zero_residual) {
    combined_frame = picture->mc_tmp_frame;
  } else {
    if (params->num_refs > 0) {
      if (decoder->use_cuda) {
#ifdef HAVE_CUDA
        schro_gpuframe_add (picture->frame, picture->mc_tmp_frame);
#else
        SCHRO_ASSERT (0);
#endif
      } else if (decoder->use_opengl) {
#ifdef HAVE_OPENGL
        schro_opengl_frame_add (picture->frame, picture->mc_tmp_frame);
#else
        SCHRO_ASSERT (0);
#endif
      } else {
        //schro_frame_add (picture->frame, picture->mc_tmp_frame);
      }
    }
    combined_frame = picture->frame;
  }

  if (_schro_decode_prediction_only) {
    if (params->num_refs > 0 && !picture->is_ref) {
      output_frame = picture->mc_tmp_frame;
    } else {
      output_frame = combined_frame;
    }
  } else {
    output_frame = combined_frame;
  }
#endif

  if (picture->decoder_instance->video_format.interlaced_coding &&
      schro_decoder_frame_is_twofield (picture->decoder_instance,
          &output_picture)) {
    /* output_picture is a frame, however the [planar_]output_frame only
     * contains a field.  Create a view of the output_picture that corresponds
     * to the correct field */
    /* 10.4p3 earliest field in each frame shall have an even picture number:
     *   PicNum&1 | top_field_first=1 | top_field_first=0
     *   ---------+-------------------+------------------
     *        0   |       top         |       bot
     *        1   |       bot         |       top*/
    if ((picture->picture_number & 1) ==
        picture->decoder_instance->video_format.top_field_first) {
      /* to access bot_field, data pointers for first line need moving */
      output_picture.components[0].data =
          SCHRO_FRAME_DATA_GET_LINE (&output_picture.components[0], 1);
      output_picture.components[1].data =
          SCHRO_FRAME_DATA_GET_LINE (&output_picture.components[1], 1);
      output_picture.components[2].data =
          SCHRO_FRAME_DATA_GET_LINE (&output_picture.components[2], 1);
    }
    /* skip every other line and half heights */
    output_picture.components[0].stride *= 2;
    output_picture.components[1].stride *= 2;
    output_picture.components[2].stride *= 2;
    output_picture.components[0].height /= 2;
    output_picture.components[1].height /= 2;
    output_picture.components[2].height /= 2;
    output_picture.height /= 2;
  }

  if (SCHRO_FRAME_IS_PACKED (picture->output_picture->format)) {
    if (decoder->use_cuda) {
#ifdef HAVE_CUDA
      SchroFrame *cuda_output_frame;
      SchroFrame *planar_output_frame;
      planar_output_frame = schro_frame_new_and_alloc (decoder->cpu_domain,
          schro_params_get_frame_format (8,
              picture->decoder_instance->video_format.chroma_format),
          picture->decoder_instance->video_format.width,
          picture->decoder_instance->video_format.height);
      cuda_output_frame =
          schro_frame_clone (decoder->cuda_domain, &output_picture);
      schro_gpuframe_convert (planar_output_frame, output_frame);
      schro_gpuframe_convert (cuda_output_frame, planar_output_frame);
      schro_gpuframe_to_cpu (&output_picture, cuda_output_frame);
      schro_frame_unref (cuda_output_frame);
      cuda_output_frame = NULL;
      schro_frame_unref (planar_output_frame);
#else
      SCHRO_ASSERT (0);
#endif
    } else if (decoder->use_opengl) {
#ifdef HAVE_OPENGL
      SchroFrame *tmp_opengl_output_frame;

      tmp_opengl_output_frame = schro_opengl_frame_new (decoder->opengl,
          decoder->opengl_domain, picture->planar_output_frame->format,
          picture->planar_output_frame->width,
          picture->planar_output_frame->height);

      schro_opengl_frame_convert (tmp_opengl_output_frame, output_frame);
      schro_opengl_frame_pull (picture->planar_output_frame,
          tmp_opengl_output_frame);
      schro_frame_unref (tmp_opengl_output_frame);
      tmp_opengl_output_frame = NULL;

      schro_frame_convert (&output_picture, picture->planar_output_frame);
#else
      SCHRO_ASSERT (0);
#endif
    } else {
      if (params->num_refs > 0 || picture->is_ref) {
        schro_frame_convert (&output_picture, picture->ref_output_frame);
      } else {
        int depth;
        depth =
            schro_video_format_get_bit_depth (&picture->
            decoder_instance->video_format);
        if (depth > 8) {
          /* hack to handle 10-bit to 8-bit output */
          schro_frame_shift_right (picture->frame, depth - 8);
        }
        schro_frame_convert (&output_picture, picture->frame);
      }
    }
  } else {
    if (decoder->use_cuda) {
#ifdef HAVE_CUDA
      SchroFrame *cuda_output_frame;
      cuda_output_frame = schro_frame_clone (decoder->cuda_domain,
          &output_picture);
      schro_gpuframe_convert (cuda_output_frame, output_frame);
      schro_gpuframe_to_cpu (&output_picture, cuda_output_frame);
      schro_frame_unref (cuda_output_frame);
      cuda_output_frame = NULL;
#else
      SCHRO_ASSERT (0);
#endif
    } else if (decoder->use_opengl) {
#ifdef HAVE_OPENGL
      SchroFrame *tmp_opengl_output_frame;

      tmp_opengl_output_frame = schro_opengl_frame_new (decoder->opengl,
          decoder->opengl_domain, output_picture.format,
          output_picture.width, output_picture.height);

      schro_opengl_frame_convert (tmp_opengl_output_frame, output_frame);
      schro_opengl_frame_pull (&output_picture, tmp_opengl_output_frame);
      schro_frame_unref (tmp_opengl_output_frame);
      tmp_opengl_output_frame = NULL;
#else
      SCHRO_ASSERT (0);
#endif
    } else {
      if (params->num_refs > 0 || picture->is_ref) {
        schro_frame_convert (&output_picture, picture->ref_output_frame);
      } else {
        int depth;
        depth =
            schro_video_format_get_bit_depth (&picture->
            decoder_instance->video_format);
        if (depth > 8) {
          /* hack to handle 10-bit to 8-bit output */
          schro_frame_shift_right (picture->frame, depth - 8);
        }
        schro_frame_convert (&output_picture, picture->frame);
      }
      if (_schro_telemetry) {
        schro_decoder_telemetry (picture, &output_picture);
      }
    }
  }

  if (picture->is_ref) {
    SchroFrame *ref;
    SchroFrameFormat frame_format;

    frame_format = schro_params_get_frame_format (8,
        params->video_format->chroma_format);

    ref = schro_frame_ref (picture->ref_output_frame);
    if (decoder->use_cuda) {
#ifdef HAVE_CUDA
      schro_gpuframe_convert (ref, combined_frame);
#else
      SCHRO_ASSERT (0);
#endif
    } else if (decoder->use_opengl) {
#ifdef HAVE_OPENGL
      schro_opengl_frame_convert (ref, combined_frame);
#else
      SCHRO_ASSERT (0);
#endif
    } else {
      schro_frame_mc_edgeextend (ref);
    }
    picture->upsampled_frame = schro_upsampled_frame_new (ref);
  }

  if (picture->has_md5) {
    uint32_t state[4];

    schro_frame_md5 (picture->ref_output_frame, state);
    if (memcmp (state, picture->md5_checksum, 16) != 0) {
      char a[33];
      char b[33];
      int i;

      for (i = 0; i < 16; i++) {
        sprintf (a + 2 * i, "%02x", ((uint8_t *) state)[i]);
        sprintf (b + 2 * i, "%02x", picture->md5_checksum[i]);
      }
      a[32] = 0;
      b[32] = 0;
      SCHRO_ERROR ("MD5 checksum mismatch (%s should be %s)", a, b);
    }
  }

  /* eagerly unreference any storage that is nolonger required */
  if (picture->mc_tmp_frame) {
    schro_frame_unref (picture->mc_tmp_frame);
    picture->mc_tmp_frame = NULL;
  }
  schro_frame_unref (picture->transform_frame);
  picture->transform_frame = NULL;
  schro_frame_unref (picture->frame);
  picture->frame = NULL;
}

void
schro_decoder_x_upsample (SchroAsyncStage * stage)
{
  SchroPicture *picture = (SchroPicture *) stage->priv;
  SchroDecoder *decoder = picture->decoder_instance->decoder;

  if (decoder->use_cuda) {
#ifdef HAVE_CUDA
    schro_upsampled_gpuframe_upsample (picture->upsampled_frame);
#else
    SCHRO_ASSERT (0);
#endif
  } else if (decoder->use_opengl) {
#ifdef HAVE_OPENGL
    schro_opengl_upsampled_frame_upsample (picture->upsampled_frame);
#else
    SCHRO_ASSERT (0);
#endif
  } else {
    schro_upsampled_frame_upsample (picture->upsampled_frame);
  }
}

int
schro_decoder_decode_parse_header (SchroUnpack * unpack)
{
  int v1, v2, v3, v4;
  int parse_code;

  v1 = schro_unpack_decode_bits (unpack, 8);
  v2 = schro_unpack_decode_bits (unpack, 8);
  v3 = schro_unpack_decode_bits (unpack, 8);
  v4 = schro_unpack_decode_bits (unpack, 8);
  SCHRO_DEBUG ("parse header %02x %02x %02x %02x", v1, v2, v3, v4);
  if (v1 != 'B' || v2 != 'B' || v3 != 'C' || v4 != 'D') {
    SCHRO_ERROR ("expected parse header");
    return -1;
  }

  parse_code = schro_unpack_decode_bits (unpack, 8);
  SCHRO_DEBUG ("parse code %02x", parse_code);

  v1 = schro_unpack_decode_bits (unpack, 32);
  SCHRO_DEBUG ("next_parse_offset %d", v1);
  v1 = schro_unpack_decode_bits (unpack, 32);
  SCHRO_DEBUG ("prev_parse_offset %d", v1);

  return parse_code;
}

static int
schro_decoder_check_version (int major, int minor)
{
  if (major == 0 && minor == 20071203)
    return TRUE;
  if (major == 1 && minor == 0)
    return TRUE;
  if (major == 2 && minor == 0)
    return TRUE;
  if (major == 2 && minor == 1)
    return TRUE;
  if (major == 2 && minor == 2)
    return TRUE;

  return FALSE;
}

int
schro_decoder_compare_sequence_header_buffer (SchroBuffer * a, SchroBuffer * b)
{
  return TRUE;

  /* FIXME */
  if (a->length != b->length) {
    SCHRO_ERROR ("length different %d vs %d", a->length, b->length);
    return FALSE;
  }
  if (a->length < 13)
    return FALSE;

  {
    int i;
    for (i = 13; i < a->length; i++) {
      SCHRO_ERROR ("seq headers: %02x %02x", a->data[i], b->data[i]);
    }
  }

  if (memcmp (a->data + 13, b->data + 13, a->length - 13) != 0)
    return FALSE;

  return TRUE;
}

void
schro_decoder_parse_sequence_header (SchroDecoderInstance * instance,
    SchroUnpack * unpack)
{
  int bit;
  int index;
  SchroVideoFormat *format = &instance->video_format;

  SCHRO_DEBUG ("decoding sequence header");

  /* parse parameters */
  instance->major_version = schro_unpack_decode_uint (unpack);
  SCHRO_DEBUG ("major_version = %d", instance->major_version);
  instance->minor_version = schro_unpack_decode_uint (unpack);
  SCHRO_DEBUG ("minor_version = %d", instance->minor_version);
  instance->profile = schro_unpack_decode_uint (unpack);
  SCHRO_DEBUG ("profile = %d", instance->profile);
  instance->level = schro_unpack_decode_uint (unpack);
  SCHRO_DEBUG ("level = %d", instance->level);

  if (!schro_decoder_check_version (instance->major_version,
          instance->minor_version)) {
    SCHRO_WARNING
        ("Stream version number %d:%d not handled, expecting 0:20071203, 1:0, 2:0, 2:1, or 2:2",
        instance->major_version, instance->minor_version);
  }

  /* base video format */
  index = schro_unpack_decode_uint (unpack);
  schro_video_format_set_std_video_format (format, index);

  /* source parameters */
  /* frame dimensions */
  bit = schro_unpack_decode_bit (unpack);
  if (bit) {
    format->width = schro_unpack_decode_uint (unpack);
    format->height = schro_unpack_decode_uint (unpack);
  }
  SCHRO_DEBUG ("size = %d x %d", format->width, format->height);

  /* chroma format */
  bit = schro_unpack_decode_bit (unpack);
  if (bit) {
    format->chroma_format = schro_unpack_decode_uint (unpack);
  }
  SCHRO_DEBUG ("chroma_format %d", format->chroma_format);

  /* scan format */
  bit = schro_unpack_decode_bit (unpack);
  if (bit) {
    format->interlaced = schro_unpack_decode_uint (unpack);
  }
  SCHRO_DEBUG ("interlaced %d top_field_first %d",
      format->interlaced, format->top_field_first);

  MARKER ();

  /* frame rate */
  bit = schro_unpack_decode_bit (unpack);
  if (bit) {
    index = schro_unpack_decode_uint (unpack);
    if (index == 0) {
      format->frame_rate_numerator = schro_unpack_decode_uint (unpack);
      format->frame_rate_denominator = schro_unpack_decode_uint (unpack);
    } else {
      schro_video_format_set_std_frame_rate (format, index);
    }
  }
  SCHRO_DEBUG ("frame rate %d/%d", format->frame_rate_numerator,
      format->frame_rate_denominator);

  MARKER ();

  /* aspect ratio */
  bit = schro_unpack_decode_bit (unpack);
  if (bit) {
    index = schro_unpack_decode_uint (unpack);
    if (index == 0) {
      format->aspect_ratio_numerator = schro_unpack_decode_uint (unpack);
      format->aspect_ratio_denominator = schro_unpack_decode_uint (unpack);
    } else {
      schro_video_format_set_std_aspect_ratio (format, index);
    }
  }
  SCHRO_DEBUG ("aspect ratio %d/%d", format->aspect_ratio_numerator,
      format->aspect_ratio_denominator);

  MARKER ();

  /* clean area */
  bit = schro_unpack_decode_bit (unpack);
  if (bit) {
    format->clean_width = schro_unpack_decode_uint (unpack);
    format->clean_height = schro_unpack_decode_uint (unpack);
    format->left_offset = schro_unpack_decode_uint (unpack);
    format->top_offset = schro_unpack_decode_uint (unpack);
  }
  SCHRO_DEBUG ("clean offset %d %d", format->left_offset, format->top_offset);
  SCHRO_DEBUG ("clean size %d %d", format->clean_width, format->clean_height);

  MARKER ();

  /* signal range */
  bit = schro_unpack_decode_bit (unpack);
  if (bit) {
    index = schro_unpack_decode_uint (unpack);
    if (index == 0) {
      format->luma_offset = schro_unpack_decode_uint (unpack);
      format->luma_excursion = schro_unpack_decode_uint (unpack);
      format->chroma_offset = schro_unpack_decode_uint (unpack);
      format->chroma_excursion = schro_unpack_decode_uint (unpack);
    } else {
      if (index <= SCHRO_SIGNAL_RANGE_12BIT_VIDEO) {
        schro_video_format_set_std_signal_range (format, index);
      } else {
        schro_decoder_error (instance->decoder,
            "signal range index out of range");
      }
    }
  }
  SCHRO_DEBUG ("luma offset %d excursion %d", format->luma_offset,
      format->luma_excursion);
  SCHRO_DEBUG ("chroma offset %d excursion %d", format->chroma_offset,
      format->chroma_excursion);

  MARKER ();

  /* colour spec */
  bit = schro_unpack_decode_bit (unpack);
  if (bit) {
    index = schro_unpack_decode_uint (unpack);
    if (index <= SCHRO_COLOUR_SPEC_CINEMA) {
      schro_video_format_set_std_colour_spec (format, index);
    } else {
      schro_decoder_error (instance->decoder, "colour spec index out of range");
    }
    if (index == 0) {
      /* colour primaries */
      bit = schro_unpack_decode_bit (unpack);
      if (bit) {
        format->colour_primaries = schro_unpack_decode_uint (unpack);
      }
      /* colour matrix */
      bit = schro_unpack_decode_bit (unpack);
      if (bit) {
        format->colour_matrix = schro_unpack_decode_uint (unpack);
      }
      /* transfer function */
      bit = schro_unpack_decode_bit (unpack);
      if (bit) {
        format->transfer_function = schro_unpack_decode_uint (unpack);
      }
    }
  }

  format->interlaced_coding = schro_unpack_decode_uint (unpack);

  MARKER ();

  schro_video_format_validate (format);
}

void
schro_decoder_parse_picture_header (SchroPicture * picture,
    SchroUnpack * unpack)
{
  SchroParams *params = &picture->params;

  schro_unpack_byte_sync (unpack);

  picture->picture_number = schro_unpack_decode_bits (unpack, 32);
  SCHRO_DEBUG ("picture number %d", picture->picture_number);

  if (params->num_refs > 0) {
    picture->reference1 = picture->picture_number +
        schro_unpack_decode_sint (unpack);
    SCHRO_DEBUG ("ref1 %d", picture->reference1);
  }

  if (params->num_refs > 1) {
    picture->reference2 = picture->picture_number +
        schro_unpack_decode_sint (unpack);
    SCHRO_DEBUG ("ref2 %d", picture->reference2);
  }

  if (picture->is_ref) {
    picture->retired_picture_number = picture->picture_number +
        schro_unpack_decode_sint (unpack);
  }
}

void
schro_decoder_parse_picture_prediction_parameters (SchroPicture * picture,
    SchroUnpack * unpack)
{
  SchroParams *params = &picture->params;
  int bit;
  int index;
  int ret;

  /* block parameters */
  index = schro_unpack_decode_uint (unpack);
  if (index == 0) {
    params->xblen_luma = schro_unpack_decode_uint (unpack);
    params->yblen_luma = schro_unpack_decode_uint (unpack);
    params->xbsep_luma = schro_unpack_decode_uint (unpack);
    params->ybsep_luma = schro_unpack_decode_uint (unpack);
    if (!schro_params_verify_block_params (params))
      picture->error = TRUE;
  } else {
    ret = schro_params_set_block_params (params, index);
    if (!ret)
      picture->error = TRUE;
  }
  SCHRO_DEBUG ("blen_luma %d %d bsep_luma %d %d",
      params->xblen_luma, params->yblen_luma,
      params->xbsep_luma, params->ybsep_luma);

  MARKER ();

  /* mv precision */
  params->mv_precision = schro_unpack_decode_uint (unpack);
  SCHRO_DEBUG ("mv_precision %d", params->mv_precision);
  if (params->mv_precision > 3) {
    picture->error = TRUE;
  }

  MARKER ();

  /* global motion */
  params->have_global_motion = schro_unpack_decode_bit (unpack);
  if (params->have_global_motion) {
    int i;

    for (i = 0; i < params->num_refs; i++) {
      SchroGlobalMotion *gm = params->global_motion + i;

      /* pan/tilt */
      bit = schro_unpack_decode_bit (unpack);
      if (bit) {
        gm->b0 = schro_unpack_decode_sint (unpack);
        gm->b1 = schro_unpack_decode_sint (unpack);
      } else {
        gm->b0 = 0;
        gm->b1 = 0;
      }

      /* zoom/rotate/shear */
      bit = schro_unpack_decode_bit (unpack);
      if (bit) {
        gm->a_exp = schro_unpack_decode_uint (unpack);
        gm->a00 = schro_unpack_decode_sint (unpack);
        gm->a01 = schro_unpack_decode_sint (unpack);
        gm->a10 = schro_unpack_decode_sint (unpack);
        gm->a11 = schro_unpack_decode_sint (unpack);
      } else {
        gm->a_exp = 0;
        gm->a00 = 1;
        gm->a01 = 0;
        gm->a10 = 0;
        gm->a11 = 1;
      }

      /* perspective */
      bit = schro_unpack_decode_bit (unpack);
      if (bit) {
        gm->c_exp = schro_unpack_decode_uint (unpack);
        gm->c0 = schro_unpack_decode_sint (unpack);
        gm->c1 = schro_unpack_decode_sint (unpack);
      } else {
        gm->c_exp = 0;
        gm->c0 = 0;
        gm->c1 = 0;
      }

      SCHRO_DEBUG ("ref %d pan %d %d matrix %d %d %d %d perspective %d %d",
          i, gm->b0, gm->b1, gm->a00, gm->a01, gm->a10, gm->a11,
          gm->c0, gm->c1);
    }
  }

  MARKER ();

  /* picture prediction mode */
  params->picture_pred_mode = schro_unpack_decode_uint (unpack);
  if (params->picture_pred_mode != 0) {
    picture->error = TRUE;
  }

  /* reference picture weights */
  params->picture_weight_bits = 1;
  params->picture_weight_1 = 1;
  params->picture_weight_2 = 1;
  bit = schro_unpack_decode_bit (unpack);
  if (bit) {
    params->picture_weight_bits = schro_unpack_decode_uint (unpack);
    params->picture_weight_1 = schro_unpack_decode_sint (unpack);
    if (params->num_refs > 1) {
      params->picture_weight_2 = schro_unpack_decode_sint (unpack);
    }
  }

  MARKER ();
}

enum
{
  SCHRO_DECODER_ARITH_SUPERBLOCK,
  SCHRO_DECODER_ARITH_PRED_MODE,
  SCHRO_DECODER_ARITH_VECTOR_REF1_X,
  SCHRO_DECODER_ARITH_VECTOR_REF1_Y,
  SCHRO_DECODER_ARITH_VECTOR_REF2_X,
  SCHRO_DECODER_ARITH_VECTOR_REF2_Y,
  SCHRO_DECODER_ARITH_DC_0,
  SCHRO_DECODER_ARITH_DC_1,
  SCHRO_DECODER_ARITH_DC_2
};

void
schro_decoder_parse_block_data (SchroPicture * picture, SchroUnpack * unpack)
{
  SchroParams *params = &picture->params;
  int i;

  for (i = 0; i < 9; i++) {
    int length;

    if (params->num_refs < 2 && (i == SCHRO_DECODER_ARITH_VECTOR_REF2_X ||
            i == SCHRO_DECODER_ARITH_VECTOR_REF2_Y)) {
      picture->motion_buffers[i] = NULL;
      continue;
    }

    length = schro_unpack_decode_uint (unpack);
    schro_unpack_byte_sync (unpack);
    picture->motion_buffers[i] =
        schro_buffer_new_subbuffer (picture->input_buffer,
        schro_unpack_get_bits_read (unpack) / 8, length);
    schro_unpack_skip_bits (unpack, length * 8);
  }
}

void
schro_decoder_decode_block_data (SchroPicture * picture)
{
  SchroParams *params = &picture->params;
  SchroArith *arith[9];
  SchroUnpack unpack[9];
  int i, j;

  orc_splat_u8_ns ((uint8_t *) picture->motion->motion_vectors, 0,
      sizeof (SchroMotionVector) * params->y_num_blocks * params->x_num_blocks);

  for (i = 0; i < 9; i++) {
    if (params->num_refs < 2 && (i == SCHRO_DECODER_ARITH_VECTOR_REF2_X ||
            i == SCHRO_DECODER_ARITH_VECTOR_REF2_Y)) {
      arith[i] = NULL;
      continue;
    }

    if (!params->is_noarith) {
      arith[i] = schro_arith_new ();
      schro_arith_decode_init (arith[i], picture->motion_buffers[i]);
    } else {
      schro_unpack_init_with_data (unpack + i,
          picture->motion_buffers[i]->data,
          picture->motion_buffers[i]->length, 1);
    }
  }

  for (j = 0; j < params->y_num_blocks; j += 4) {
    for (i = 0; i < params->x_num_blocks; i += 4) {
      schro_decoder_decode_macroblock (picture, arith, unpack, i, j);
    }
  }

  for (i = 0; i < 9; i++) {
    if (!params->is_noarith) {
      if (arith[i] == NULL)
        continue;

      if (arith[i]->offset < arith[i]->buffer->length) {
        SCHRO_DEBUG ("arith decoding %d didn't consume buffer (%d < %d)", i,
            arith[i]->offset, arith[i]->buffer->length);
      }
      if (arith[i]->offset > arith[i]->buffer->length + 6) {
        SCHRO_WARNING ("arith decoding %d overran buffer (%d > %d)", i,
            arith[i]->offset, arith[i]->buffer->length);
      }
      schro_arith_free (arith[i]);
    } else {
      /* FIXME complain about buffer over/underrun */
    }
  }

  schro_motion_verify (picture->motion);
}

void
schro_decoder_decode_macroblock (SchroPicture * picture, SchroArith ** arith,
    SchroUnpack * unpack, int i, int j)
{
  SchroParams *params = &picture->params;
  SchroMotion *motion = picture->motion;
  SchroMotionVector *mv = &motion->motion_vectors[j * params->x_num_blocks + i];
  int k, l;
  int split_prediction;

  split_prediction = schro_motion_split_prediction (motion, i, j);
  if (!params->is_noarith) {
    mv->split = (split_prediction +
        _schro_arith_decode_uint (arith[SCHRO_DECODER_ARITH_SUPERBLOCK],
            SCHRO_CTX_SB_F1, SCHRO_CTX_SB_DATA)) % 3;
  } else {
    mv->split = (split_prediction +
        schro_unpack_decode_uint (unpack + SCHRO_DECODER_ARITH_SUPERBLOCK)) % 3;
  }

  switch (mv->split) {
    default:
      /* this can happen if _schro_arith_decode_uint() decodes a value
       * that overflows.  FIXME should cause a picture error.  For now,
       * just pretend it didn't happen.  */
      SCHRO_ERROR ("mv->split == %d, split_prediction %d", mv->split,
          split_prediction);
      /* fall through */
    case 0:
      schro_decoder_decode_prediction_unit (picture, arith, unpack,
          motion->motion_vectors, i, j);
      mv[1] = mv[0];
      mv[2] = mv[0];
      mv[3] = mv[0];
      memcpy (mv + params->x_num_blocks, mv, 4 * sizeof (*mv));
      memcpy (mv + 2 * params->x_num_blocks, mv, 4 * sizeof (*mv));
      memcpy (mv + 3 * params->x_num_blocks, mv, 4 * sizeof (*mv));
      break;
    case 1:
      schro_decoder_decode_prediction_unit (picture, arith, unpack,
          motion->motion_vectors, i, j);
      mv[1] = mv[0];
      schro_decoder_decode_prediction_unit (picture, arith, unpack,
          motion->motion_vectors, i + 2, j);
      mv[2].split = 1;
      mv[3] = mv[2];
      memcpy (mv + params->x_num_blocks, mv, 4 * sizeof (*mv));

      mv += 2 * params->x_num_blocks;
      schro_decoder_decode_prediction_unit (picture, arith, unpack,
          motion->motion_vectors, i, j + 2);
      mv[0].split = 1;
      mv[1] = mv[0];
      schro_decoder_decode_prediction_unit (picture, arith, unpack,
          motion->motion_vectors, i + 2, j + 2);
      mv[2].split = 1;
      mv[3] = mv[2];
      memcpy (mv + params->x_num_blocks, mv, 4 * sizeof (*mv));
      break;
    case 2:
      for (l = 0; l < 4; l++) {
        for (k = 0; k < 4; k++) {
          mv[l * params->x_num_blocks + k].split = 2;
          schro_decoder_decode_prediction_unit (picture, arith, unpack,
              motion->motion_vectors, i + k, j + l);
        }
      }
      break;
  }
}

void
schro_decoder_decode_prediction_unit (SchroPicture * picture,
    SchroArith ** arith, SchroUnpack * unpack,
    SchroMotionVector * motion_vectors, int x, int y)
{
  SchroParams *params = &picture->params;
  SchroMotion *motion = picture->motion;
  SchroMotionVector *mv = &motion_vectors[y * params->x_num_blocks + x];

  mv->pred_mode = schro_motion_get_mode_prediction (motion, x, y);
  if (!params->is_noarith) {
    mv->pred_mode ^=
        _schro_arith_decode_bit (arith[SCHRO_DECODER_ARITH_PRED_MODE],
        SCHRO_CTX_BLOCK_MODE_REF1);
  } else {
    mv->pred_mode ^=
        schro_unpack_decode_bit (unpack + SCHRO_DECODER_ARITH_PRED_MODE);
  }
  if (params->num_refs > 1) {
    if (!params->is_noarith) {
      mv->pred_mode ^=
          _schro_arith_decode_bit (arith[SCHRO_DECODER_ARITH_PRED_MODE],
          SCHRO_CTX_BLOCK_MODE_REF2) << 1;
    } else {
      mv->pred_mode ^=
          schro_unpack_decode_bit (unpack + SCHRO_DECODER_ARITH_PRED_MODE) << 1;
    }
  }

  if (mv->pred_mode == 0) {
    int pred[3];

    schro_motion_dc_prediction (motion, x, y, pred);

    if (!params->is_noarith) {
      mv->u.dc.dc[0] =
          pred[0] + _schro_arith_decode_sint (arith[SCHRO_DECODER_ARITH_DC_0],
          SCHRO_CTX_LUMA_DC_CONT_BIN1, SCHRO_CTX_LUMA_DC_VALUE,
          SCHRO_CTX_LUMA_DC_SIGN);
      mv->u.dc.dc[1] =
          pred[1] + _schro_arith_decode_sint (arith[SCHRO_DECODER_ARITH_DC_1],
          SCHRO_CTX_CHROMA1_DC_CONT_BIN1, SCHRO_CTX_CHROMA1_DC_VALUE,
          SCHRO_CTX_CHROMA1_DC_SIGN);
      mv->u.dc.dc[2] =
          pred[2] + _schro_arith_decode_sint (arith[SCHRO_DECODER_ARITH_DC_2],
          SCHRO_CTX_CHROMA2_DC_CONT_BIN1, SCHRO_CTX_CHROMA2_DC_VALUE,
          SCHRO_CTX_CHROMA2_DC_SIGN);
    } else {
      mv->u.dc.dc[0] = pred[0] +
          schro_unpack_decode_sint (unpack + SCHRO_DECODER_ARITH_DC_0);
      mv->u.dc.dc[1] = pred[1] +
          schro_unpack_decode_sint (unpack + SCHRO_DECODER_ARITH_DC_1);
      mv->u.dc.dc[2] = pred[2] +
          schro_unpack_decode_sint (unpack + SCHRO_DECODER_ARITH_DC_2);
    }
  } else {
    int pred_x, pred_y;

    if (params->have_global_motion) {
      int pred;
      pred = schro_motion_get_global_prediction (motion, x, y);
      if (!params->is_noarith) {
        mv->using_global =
            pred ^
            _schro_arith_decode_bit (arith[SCHRO_DECODER_ARITH_PRED_MODE],
            SCHRO_CTX_GLOBAL_BLOCK);
      } else {
        mv->using_global =
            pred ^ schro_unpack_decode_bit (unpack +
            SCHRO_DECODER_ARITH_PRED_MODE);
      }
    } else {
      mv->using_global = FALSE;
    }
    if (!mv->using_global) {
      if (mv->pred_mode & 1) {
        schro_motion_vector_prediction (motion, x, y, &pred_x, &pred_y, 1);

        if (!params->is_noarith) {
          mv->u.vec.dx[0] =
              pred_x +
              _schro_arith_decode_sint (arith
              [SCHRO_DECODER_ARITH_VECTOR_REF1_X],
              SCHRO_CTX_MV_REF1_H_CONT_BIN1, SCHRO_CTX_MV_REF1_H_VALUE,
              SCHRO_CTX_MV_REF1_H_SIGN);
          mv->u.vec.dy[0] =
              pred_y +
              _schro_arith_decode_sint (arith
              [SCHRO_DECODER_ARITH_VECTOR_REF1_Y],
              SCHRO_CTX_MV_REF1_V_CONT_BIN1, SCHRO_CTX_MV_REF1_V_VALUE,
              SCHRO_CTX_MV_REF1_V_SIGN);
        } else {
          mv->u.vec.dx[0] =
              pred_x + schro_unpack_decode_sint (unpack +
              SCHRO_DECODER_ARITH_VECTOR_REF1_X);
          mv->u.vec.dy[0] =
              pred_y + schro_unpack_decode_sint (unpack +
              SCHRO_DECODER_ARITH_VECTOR_REF1_Y);
        }
      }
      if (mv->pred_mode & 2) {
        schro_motion_vector_prediction (motion, x, y, &pred_x, &pred_y, 2);

        if (!params->is_noarith) {
          mv->u.vec.dx[1] =
              pred_x +
              _schro_arith_decode_sint (arith
              [SCHRO_DECODER_ARITH_VECTOR_REF2_X],
              SCHRO_CTX_MV_REF2_H_CONT_BIN1, SCHRO_CTX_MV_REF2_H_VALUE,
              SCHRO_CTX_MV_REF2_H_SIGN);
          mv->u.vec.dy[1] =
              pred_y +
              _schro_arith_decode_sint (arith
              [SCHRO_DECODER_ARITH_VECTOR_REF2_Y],
              SCHRO_CTX_MV_REF2_V_CONT_BIN1, SCHRO_CTX_MV_REF2_V_VALUE,
              SCHRO_CTX_MV_REF2_V_SIGN);
        } else {
          mv->u.vec.dx[1] =
              pred_x + schro_unpack_decode_sint (unpack +
              SCHRO_DECODER_ARITH_VECTOR_REF2_X);
          mv->u.vec.dy[1] =
              pred_y + schro_unpack_decode_sint (unpack +
              SCHRO_DECODER_ARITH_VECTOR_REF2_Y);
        }
      }
    } else {
      mv->u.vec.dx[0] = 0;
      mv->u.vec.dy[0] = 0;
      mv->u.vec.dx[1] = 0;
      mv->u.vec.dy[1] = 0;
    }
  }
}

void
schro_decoder_parse_transform_parameters (SchroPicture * picture,
    SchroUnpack * unpack)
{
  int bit;
  int i;
  SchroParams *params = &picture->params;

  /* transform */
  params->wavelet_filter_index = schro_unpack_decode_uint (unpack);
  SCHRO_DEBUG ("wavelet filter index %d", params->wavelet_filter_index);

  /* transform depth */
  params->transform_depth = schro_unpack_decode_uint (unpack);
  SCHRO_DEBUG ("transform depth %d", params->transform_depth);
  if (params->transform_depth > SCHRO_LIMIT_TRANSFORM_DEPTH) {
    picture->error = TRUE;
    return;
  }

  if (!params->is_lowdelay) {
    /* codeblock parameters */
    params->codeblock_mode_index = 0;
    for (i = 0; i < params->transform_depth + 1; i++) {
      params->horiz_codeblocks[i] = 1;
      params->vert_codeblocks[i] = 1;
    }

    bit = schro_unpack_decode_bit (unpack);
    if (bit) {
      for (i = 0; i < params->transform_depth + 1; i++) {
        params->horiz_codeblocks[i] = schro_unpack_decode_uint (unpack);
        params->vert_codeblocks[i] = schro_unpack_decode_uint (unpack);
      }
      params->codeblock_mode_index = schro_unpack_decode_uint (unpack);
    }
  } else {
    /* slice parameters */
    params->n_horiz_slices = schro_unpack_decode_uint (unpack);
    params->n_vert_slices = schro_unpack_decode_uint (unpack);

    params->slice_bytes_num = schro_unpack_decode_uint (unpack);
    params->slice_bytes_denom = schro_unpack_decode_uint (unpack);

    /* quant matrix */
    bit = schro_unpack_decode_bit (unpack);
    if (bit) {
      params->quant_matrix[0] = schro_unpack_decode_uint (unpack);
      for (i = 0; i < params->transform_depth; i++) {
        params->quant_matrix[1 + 3 * i] = schro_unpack_decode_uint (unpack);
        params->quant_matrix[2 + 3 * i] = schro_unpack_decode_uint (unpack);
        params->quant_matrix[3 + 3 * i] = schro_unpack_decode_uint (unpack);
      }
    } else {
      schro_params_set_default_quant_matrix (params);
    }
  }
}

void
schro_decoder_init_subband_frame_data_interleaved (SchroPicture * picture)
{
  int i;
  int component;
  SchroFrameData *comp;
  SchroFrameData *fd;
  SchroParams *params = &picture->params;
  int position;

  if (picture->error)
    return;

  for (component = 0; component < 3; component++) {
    comp = &picture->transform_frame->components[component];
    for (i = 0; i < 1 + 3 * params->transform_depth; i++) {
      position = schro_subband_get_position (i);

      fd = &picture->subband_data[component][i];

      schro_subband_get_frame_data (fd, picture->transform_frame,
          component, position, params);
    }
  }
}

#ifdef HAVE_OPENGL
void
schro_decoder_init_subband_frame_data (SchroPicture * picture)
{
  int i;
  int component;
  SchroFrameData *fd;
  SchroParams *params = &picture->params;
  int position;

  if (picture->error)
    return;

  for (component = 0; component < 3; ++component) {
    for (i = 0; i < 1 + 3 * params->transform_depth; ++i) {
      position = schro_subband_get_position (i);
      fd = &picture->subband_data[component][i];

      schro_subband_get_frame_data (fd, picture->transform_frame,
          component, position, params);
    }
  }
}
#endif

void
schro_decoder_parse_lowdelay_transform_data (SchroPicture * picture,
    SchroUnpack * unpack)
{
  SchroParams *params = &picture->params;
  int length;

  length = (params->slice_bytes_num * params->n_horiz_slices *
      params->n_vert_slices) / params->slice_bytes_denom;
  picture->lowdelay_buffer = schro_buffer_new_subbuffer (picture->input_buffer,
      schro_unpack_get_bits_read (unpack) / 8, length);
  schro_unpack_skip_bits (unpack, length * 8);
}

void
schro_decoder_parse_transform_data (SchroPicture * picture,
    SchroUnpack * unpack)
{
  int i;
  int component;
  SchroParams *params = &picture->params;
  int subband_length;

  if (picture->error)
    return;

  for (component = 0; component < 3; component++) {
    for (i = 0; i < 1 + 3 * params->transform_depth; i++) {
      schro_unpack_byte_sync (unpack);

      subband_length = schro_unpack_decode_uint (unpack);

      SCHRO_DEBUG ("subband %d %d length %d", component, i, subband_length);

      if (subband_length == 0) {
        SCHRO_DEBUG ("subband is zero");
        schro_unpack_byte_sync (unpack);

        picture->subband_quant_index[component][i] = 0;
        picture->subband_length[component][i] = 0;
        picture->subband_buffer[component][i] = NULL;
      } else {
        int quant_index;

        quant_index = schro_unpack_decode_uint (unpack);
        SCHRO_DEBUG ("quant index %d", quant_index);

        if (quant_index < 0 || quant_index > 60) {
          picture->error = TRUE;
          return;
        }

        schro_unpack_byte_sync (unpack);

        picture->subband_quant_index[component][i] = quant_index;
        picture->subband_length[component][i] = subband_length;
        picture->subband_buffer[component][i] =
            schro_buffer_new_subbuffer (picture->input_buffer,
            schro_unpack_get_bits_read (unpack) / 8, subband_length);
        schro_unpack_skip_bits (unpack, subband_length * 8);
      }
    }
  }
}

void
schro_decoder_decode_transform_data (SchroPicture * picture)
{
  int i;
  int component;
  SchroParams *params = &picture->params;
  SchroPictureSubbandContext context = { 0 }, *ctx = &context;
  int skip_subbands;

  if (_schro_decode_prediction_only && params->num_refs > 0 && !picture->is_ref) {
    schro_frame_clear (picture->frame);
    return;
  }

  /* FIXME some day, hook this up into automatic degraded decoding */
  skip_subbands = 0;

  for (component = 0; component < 3; component++) {
    for (i = 0; i < 1 + 3 * params->transform_depth - skip_subbands; i++) {
      ctx->component = component;
      ctx->index = i;
      ctx->position = schro_subband_get_position (i);

      schro_decoder_decode_subband (picture, ctx);
    }
  }
}

static void
codeblock_line_decode_generic (SchroPictureSubbandContext * ctx,
    int16_t * line, int j, const int16_t * parent_data, const int16_t * prev)
{
  int i;

  for (i = ctx->xmin; i < ctx->xmax; i++) {
    int v;
    int parent;
    int nhood_or;
    int previous_value;

    if (parent_data) {
      parent = parent_data[(i >> 1)];
    } else {
      parent = 0;
    }

    nhood_or = 0;
    if (j > 0)
      nhood_or |= prev[i];
    if (i > 0)
      nhood_or |= line[i - 1];
    if (i > 0 && j > 0)
      nhood_or |= prev[i - 1];

    previous_value = 0;
    if (SCHRO_SUBBAND_IS_HORIZONTALLY_ORIENTED (ctx->position)) {
      if (i > 0)
        previous_value = line[i - 1];
    } else if (SCHRO_SUBBAND_IS_VERTICALLY_ORIENTED (ctx->position)) {
      if (j > 0)
        previous_value = prev[i];
    }
#define STUFF \
  do { \
    int cont_context, sign_context, value_context; \
    \
    if (parent == 0) { \
      cont_context = nhood_or ? SCHRO_CTX_ZPNN_F1 : SCHRO_CTX_ZPZN_F1; \
    } else { \
      cont_context = nhood_or ? SCHRO_CTX_NPNN_F1 : SCHRO_CTX_NPZN_F1; \
    } \
     \
    if (previous_value < 0) { \
      sign_context = SCHRO_CTX_SIGN_NEG; \
    } else { \
      sign_context = (previous_value > 0) ? SCHRO_CTX_SIGN_POS : \
        SCHRO_CTX_SIGN_ZERO; \
    } \
 \
    value_context = SCHRO_CTX_COEFF_DATA; \
 \
    v = _schro_arith_decode_uint (ctx->arith, cont_context, \
        value_context); \
    if (v) { \
      v = (ctx->quant_offset + ctx->quant_factor * v + 2)>>2; \
      if (_schro_arith_decode_bit (ctx->arith, sign_context)) { \
        v = -v; \
      } \
      line[i] = v; \
    } else { \
      line[i] = 0; \
    } \
  } while(0)

    STUFF;
  }
}

#if 0
static void
codeblock_line_decode_deep (SchroPictureSubbandContext * ctx,
    int32_t * line, int j, const int32_t * parent_data, const int32_t * prev)
{
  int i;

  for (i = ctx->xmin; i < ctx->xmax; i++) {
    int v;
    int parent;
    int nhood_or;
    int previous_value;

    if (parent_data) {
      parent = parent_data[(i >> 1)];
    } else {
      parent = 0;
    }

    nhood_or = 0;
    if (j > 0)
      nhood_or |= prev[i];
    if (i > 0)
      nhood_or |= line[i - 1];
    if (i > 0 && j > 0)
      nhood_or |= prev[i - 1];

    previous_value = 0;
    if (SCHRO_SUBBAND_IS_HORIZONTALLY_ORIENTED (ctx->position)) {
      if (i > 0)
        previous_value = line[i - 1];
    } else if (SCHRO_SUBBAND_IS_VERTICALLY_ORIENTED (ctx->position)) {
      if (j > 0)
        previous_value = prev[i];
    }

    STUFF;
  }
}

static void
codeblock_line_decode_deep_parent (SchroPictureSubbandContext * ctx,
    int16_t * line, int j, const int32_t * parent_data, const int16_t * prev)
{
  int i;

  for (i = ctx->xmin; i < ctx->xmax; i++) {
    int v;
    int parent;
    int nhood_or;
    int previous_value;

    if (parent_data) {
      parent = parent_data[(i >> 1)];
    } else {
      parent = 0;
    }

    nhood_or = 0;
    if (j > 0)
      nhood_or |= prev[i];
    if (i > 0)
      nhood_or |= line[i - 1];
    if (i > 0 && j > 0)
      nhood_or |= prev[i - 1];

    previous_value = 0;
    if (SCHRO_SUBBAND_IS_HORIZONTALLY_ORIENTED (ctx->position)) {
      if (i > 0)
        previous_value = line[i - 1];
    } else if (SCHRO_SUBBAND_IS_VERTICALLY_ORIENTED (ctx->position)) {
      if (j > 0)
        previous_value = prev[i];
    }

    STUFF;
  }
}
#endif


static void
codeblock_line_decode_p_horiz (SchroPictureSubbandContext * ctx,
    int16_t * line, int j, const int16_t * parent_data, const int16_t * prev)
{
  int i = ctx->xmin;
  int v;
  int parent;
  int nhood_or;
  int previous_value;

  if (i == 0) {
    parent = parent_data[(i >> 1)];
    nhood_or = prev[i];
    previous_value = 0;

    STUFF;
    i++;
  }
  for (; i < ctx->xmax; i++) {
    parent = parent_data[(i >> 1)];

    nhood_or = prev[i];
    nhood_or |= line[i - 1];
    nhood_or |= prev[i - 1];

    previous_value = line[i - 1];

    STUFF;
  }
}

static void
codeblock_line_decode_p_vert (SchroPictureSubbandContext * ctx,
    int16_t * line, int j, const int16_t * parent_data, const int16_t * prev)
{
  int i = ctx->xmin;
  int v;
  int parent;
  int nhood_or;
  int previous_value;

  if (i == 0) {
    parent = parent_data[(i >> 1)];
    nhood_or = prev[i];
    previous_value = prev[i];

    STUFF;
    i++;
  }
  for (; i < ctx->xmax; i++) {
    parent = parent_data[(i >> 1)];

    nhood_or = prev[i];
    nhood_or |= line[i - 1];
    nhood_or |= prev[i - 1];

    previous_value = prev[i];

    STUFF;
  }
}

static void
codeblock_line_decode_p_diag (SchroPictureSubbandContext * ctx,
    int16_t * line, int j, const int16_t * parent_data, const int16_t * prev)
{
  int i;
  int v;
  int parent;
  int nhood_or;
  int previous_value;

  i = ctx->xmin;
  if (i == 0) {
    parent = parent_data[(i >> 1)];
    nhood_or = prev[i];
    previous_value = 0;

    STUFF;
    i++;
  }
  for (; i < ctx->xmax; i++) {
    parent = parent_data[(i >> 1)];

    nhood_or = prev[i];
    nhood_or |= line[i - 1];
    nhood_or |= prev[i - 1];
    previous_value = 0;

    STUFF;
  }
}

void
schro_decoder_subband_dc_predict (SchroFrameData * fd)
{
  int16_t *prev_line;
  int16_t *line;
  int i, j;
  int pred_value;

  line = SCHRO_FRAME_DATA_GET_LINE (fd, 0);
  for (i = 1; i < fd->width; i++) {
    pred_value = line[i - 1];
    line[i] += pred_value;
  }

  for (j = 1; j < fd->height; j++) {
    line = SCHRO_FRAME_DATA_GET_LINE (fd, j);
    prev_line = SCHRO_FRAME_DATA_GET_LINE (fd, j - 1);

    pred_value = prev_line[0];
    line[0] += pred_value;

    for (i = 1; i < fd->width; i++) {
      pred_value = schro_divide3 (line[i - 1] + prev_line[i] +
          prev_line[i - 1] + 1);
      line[i] += pred_value;
    }
  }

}

static void
schro_decoder_setup_codeblocks (SchroPicture * picture,
    SchroPictureSubbandContext * ctx)
{
  SchroParams *params = &picture->params;

  if (ctx->position == 0) {
    ctx->vert_codeblocks = params->vert_codeblocks[0];
    ctx->horiz_codeblocks = params->horiz_codeblocks[0];
  } else {
    ctx->vert_codeblocks =
        params->vert_codeblocks[SCHRO_SUBBAND_SHIFT (ctx->position) + 1];
    ctx->horiz_codeblocks =
        params->horiz_codeblocks[SCHRO_SUBBAND_SHIFT (ctx->position) + 1];
  }
  if (ctx->horiz_codeblocks > 1 || ctx->vert_codeblocks > 1) {
    ctx->have_zero_flags = TRUE;
  } else {
    ctx->have_zero_flags = FALSE;
  }
  if (params->codeblock_mode_index == 1) {
    ctx->have_quant_offset = TRUE;
    if (picture->decoder_instance->compat_quant_offset &&
        ctx->horiz_codeblocks == 1 && ctx->vert_codeblocks == 1) {
      ctx->have_quant_offset = FALSE;
    }
  } else {
    ctx->have_quant_offset = FALSE;
  }
}

static void
schro_decoder_zero_block (SchroPictureSubbandContext * ctx,
    int x1, int y1, int x2, int y2)
{
  //SCHRO_DEBUG("subband is zero");
  orc_splat_s16_2d (SCHRO_FRAME_DATA_GET_PIXEL_S16 (ctx->frame_data, x1, y1),
      ctx->frame_data->stride, 0, x2 - x1, y2 - y1);
}

static void
schro_decoder_test_quant_offset_compat (SchroPicture * picture,
    SchroPictureSubbandContext * ctx)
{
  SchroParams *params = &picture->params;
  int quant_index = ctx->quant_index;

  if (ctx->have_quant_offset && ctx->horiz_codeblocks == 1 &&
      ctx->vert_codeblocks == 1 && ctx->index == 0 &&
      ctx->ymin == 0 && ctx->xmin == 0) {
    if (params->is_noarith) {
      SchroUnpack unpack_copy;
      schro_unpack_copy (&unpack_copy, &ctx->unpack);
      quant_index += schro_unpack_decode_sint (&unpack_copy);
    } else {
      SchroArith arith_copy;
      memcpy (&arith_copy, ctx->arith, sizeof (SchroArith));
      quant_index += _schro_arith_decode_sint (&arith_copy,
          SCHRO_CTX_QUANTISER_CONT, SCHRO_CTX_QUANTISER_VALUE,
          SCHRO_CTX_QUANTISER_SIGN);
    }

    if (quant_index < 0 || quant_index > 60) {
      SCHRO_WARNING ("turning on codeblock quantiser compatibility mode");
      picture->decoder_instance->compat_quant_offset = TRUE;
      ctx->have_quant_offset = FALSE;
    }
  }
}

static void
schro_decoder_decode_codeblock_noarith (SchroPicture * picture,
    SchroPictureSubbandContext * ctx)
{
  SchroParams *params = &picture->params;
  int n = ctx->xmax - ctx->xmin;
  int j;

  if (ctx->have_zero_flags) {
    int bit;

    /* zero codeblock */
    bit = schro_unpack_decode_bit (&ctx->unpack);
    if (bit) {
      if (ctx->xmax - ctx->xmin == 8) {
        orc_splat_s16_2d_8xn (SCHRO_FRAME_DATA_GET_PIXEL_S16 (ctx->frame_data,
                ctx->xmin, ctx->ymin),
            ctx->frame_data->stride, 0, ctx->ymax - ctx->ymin);
      } else if (ctx->xmax - ctx->xmin == 4) {
        orc_splat_s16_2d_4xn (SCHRO_FRAME_DATA_GET_PIXEL_S16 (ctx->frame_data,
                ctx->xmin, ctx->ymin),
            ctx->frame_data->stride, 0, ctx->ymax - ctx->ymin);
      } else {
        schro_decoder_zero_block (ctx, ctx->xmin, ctx->ymin,
            ctx->xmax, ctx->ymax);
      }
      return;
    }
  }

  schro_decoder_test_quant_offset_compat (picture, ctx);

  if (ctx->have_quant_offset) {
    ctx->quant_index += schro_unpack_decode_sint (&ctx->unpack);

    if (ctx->quant_index < 0 || ctx->quant_index > 60) {
      /* FIXME decode error */
    }
    ctx->quant_index = CLAMP (ctx->quant_index, 0, 60);
  }

  ctx->quant_factor = schro_table_quant[ctx->quant_index];
  if (params->num_refs > 0) {
    ctx->quant_offset = schro_table_offset_3_8[ctx->quant_index];
  } else {
    ctx->quant_offset = schro_table_offset_1_2[ctx->quant_index];
  }


  if (n == 8 && (ctx->ymax - ctx->ymin) <= 9) {
    int16_t tmp[72];

    schro_unpack_decode_sint_s16 (tmp, &ctx->unpack,
        n * (ctx->ymax - ctx->ymin));
    orc_dequantise_s16_2d_8xn (SCHRO_FRAME_DATA_GET_PIXEL_S16 (ctx->frame_data,
            ctx->xmin, ctx->ymin), ctx->frame_data->stride, tmp,
        n * sizeof (int16_t), ctx->quant_factor, ctx->quant_offset + 2,
        ctx->ymax - ctx->ymin);
  } else if (n == 4 && (ctx->ymax - ctx->ymin) <= 9) {
    int16_t tmp[36];

    schro_unpack_decode_sint_s16 (tmp, &ctx->unpack,
        n * (ctx->ymax - ctx->ymin));
    orc_dequantise_s16_2d_4xn (SCHRO_FRAME_DATA_GET_PIXEL_S16 (ctx->frame_data,
            ctx->xmin, ctx->ymin), ctx->frame_data->stride, tmp,
        n * sizeof (int16_t), ctx->quant_factor, ctx->quant_offset + 2,
        ctx->ymax - ctx->ymin);
  } else {
    for (j = ctx->ymin; j < ctx->ymax; j++) {
      int16_t *p = SCHRO_FRAME_DATA_GET_LINE (ctx->frame_data, j);

      schro_unpack_decode_sint_s16 (p + ctx->xmin, &ctx->unpack, n);
    }

    orc_dequantise_s16_ip_2d (SCHRO_FRAME_DATA_GET_PIXEL_S16 (ctx->frame_data,
            ctx->xmin, ctx->ymin), ctx->frame_data->stride, ctx->quant_factor,
        ctx->quant_offset + 2, n, ctx->ymax - ctx->ymin);
  }
}

static void
schro_decoder_decode_codeblock (SchroPicture * picture,
    SchroPictureSubbandContext * ctx)
{
  SchroParams *params = &picture->params;
  int j;

  if (ctx->have_zero_flags) {
    int bit;

    /* zero codeblock */
    bit = _schro_arith_decode_bit (ctx->arith, SCHRO_CTX_ZERO_CODEBLOCK);
    if (bit) {
      schro_decoder_zero_block (ctx, ctx->xmin, ctx->ymin,
          ctx->xmax, ctx->ymax);
      return;
    }
  }

  schro_decoder_test_quant_offset_compat (picture, ctx);

  if (ctx->have_quant_offset) {
    ctx->quant_index += _schro_arith_decode_sint (ctx->arith,
        SCHRO_CTX_QUANTISER_CONT, SCHRO_CTX_QUANTISER_VALUE,
        SCHRO_CTX_QUANTISER_SIGN);

    if (ctx->quant_index < 0 || ctx->quant_index > 60) {
      /* FIXME decode error */
    }
    ctx->quant_index = CLAMP (ctx->quant_index, 0, 60);
  }

  ctx->quant_factor = schro_table_quant[ctx->quant_index];
  if (params->num_refs > 0) {
    ctx->quant_offset = schro_table_offset_3_8[ctx->quant_index];
  } else {
    ctx->quant_offset = schro_table_offset_1_2[ctx->quant_index];
  }

  for (j = ctx->ymin; j < ctx->ymax; j++) {
    int16_t *p = SCHRO_FRAME_DATA_GET_LINE (ctx->frame_data, j);
    const int16_t *parent_line;
    const int16_t *prev_line;

    if (ctx->position >= 4) {
      parent_line =
          SCHRO_FRAME_DATA_GET_LINE (ctx->parent_frame_data, (j >> 1));
    } else {
      parent_line = NULL;
    }
    if (j == 0) {
      prev_line = NULL;
    } else {
      prev_line = SCHRO_FRAME_DATA_GET_LINE (ctx->frame_data, (j - 1));
    }
    if (ctx->position >= 4 && j > 0) {
      if (SCHRO_SUBBAND_IS_HORIZONTALLY_ORIENTED (ctx->position)) {
        codeblock_line_decode_p_horiz (ctx, p, j, parent_line, prev_line);
      } else if (SCHRO_SUBBAND_IS_VERTICALLY_ORIENTED (ctx->position)) {
        codeblock_line_decode_p_vert (ctx, p, j, parent_line, prev_line);
      } else {
        codeblock_line_decode_p_diag (ctx, p, j, parent_line, prev_line);
      }
    } else {
      codeblock_line_decode_generic (ctx, p, j, parent_line, prev_line);
    }
  }
}

void
schro_decoder_decode_subband (SchroPicture * picture,
    SchroPictureSubbandContext * ctx)
{
  SchroParams *params = &picture->params;
  int x, y;

  ctx->subband_length = picture->subband_length[ctx->component][ctx->index];
  ctx->quant_index = picture->subband_quant_index[ctx->component][ctx->index];
  ctx->is_intra = (params->num_refs == 0);

  ctx->frame_data = &picture->subband_data[ctx->component][ctx->index];
  if (ctx->position >= 4) {
    ctx->parent_frame_data =
        &picture->subband_data[ctx->component][ctx->index - 3];
  }

  if (picture->subband_length[ctx->component][ctx->index] == 0) {
    schro_decoder_zero_block (ctx, 0, 0,
        ctx->frame_data->width, ctx->frame_data->height);
    return;
  }

  if (!params->is_noarith) {
    ctx->arith = schro_arith_new ();
    schro_arith_decode_init (ctx->arith,
        picture->subband_buffer[ctx->component][ctx->index]);
  } else {
    schro_unpack_init_with_data (&ctx->unpack,
        picture->subband_buffer[ctx->component][ctx->index]->data,
        picture->subband_buffer[ctx->component][ctx->index]->length, 1);
  }

  schro_decoder_setup_codeblocks (picture, ctx);

  for (y = 0; y < ctx->vert_codeblocks; y++) {
    int xmin;
    int acc;
    int codeblock_width;
    int inc;

    ctx->ymin = (ctx->frame_data->height * y) / ctx->vert_codeblocks;
    ctx->ymax = (ctx->frame_data->height * (y + 1)) / ctx->vert_codeblocks;

    xmin = 0;
    codeblock_width = ctx->frame_data->width / ctx->horiz_codeblocks;
    inc = ctx->frame_data->width - (ctx->horiz_codeblocks * codeblock_width);
    acc = 0;
    for (x = 0; x < ctx->horiz_codeblocks; x++) {
      ctx->xmin = xmin;
      xmin += codeblock_width;
      acc += inc;
      if (acc >= ctx->horiz_codeblocks) {
        acc -= ctx->horiz_codeblocks;
        xmin++;
      }
      ctx->xmax = xmin;

      if (params->is_noarith) {
        schro_decoder_decode_codeblock_noarith (picture, ctx);
      } else {
        schro_decoder_decode_codeblock (picture, ctx);
      }
    }
  }
  if (!params->is_noarith) {
    schro_arith_decode_flush (ctx->arith);
    if (ctx->arith->offset < ctx->subband_length - 1) {
      SCHRO_WARNING ("arith decoding didn't consume buffer (%d < %d)",
          ctx->arith->offset, ctx->subband_length);
#ifdef DONT_DO_THIS
      ctx->broken = TRUE;
#endif
    }
    if (ctx->arith->offset > ctx->subband_length + 4) {
      SCHRO_WARNING ("arith decoding overran buffer (%d > %d)",
          ctx->arith->offset, ctx->subband_length);
#ifdef DONT_DO_THIS
      ctx->broken = TRUE;
#endif
    }
    schro_arith_free (ctx->arith);
  } else {
    int offset = (ctx->unpack.n_bits_read + 0) / 8;
    if (offset < ctx->subband_length - 1) {
      SCHRO_WARNING
          ("vlc decoding didn't consume buffer (%d < %d) component=%d subband=%d",
          offset, ctx->subband_length, ctx->component, ctx->index);
#ifdef DONT_DO_THIS
      ctx->broken = TRUE;
#endif
    }
    if (offset > ctx->subband_length + 0) {
      SCHRO_WARNING ("vlc decoding overran buffer (%d > %d)",
          offset, ctx->subband_length);
#ifdef DONT_DO_THIS
      ctx->broken = TRUE;
#endif
    }
  }

  if (ctx->broken && ctx->position != 0) {
    schro_decoder_zero_block (ctx, 0, 0,
        ctx->frame_data->width, ctx->frame_data->height);
  }

  if (ctx->position == 0 && picture->params.num_refs == 0) {
    schro_decoder_subband_dc_predict (ctx->frame_data);
  }
}

/* reference pool */

static void
schro_decoder_reference_add (SchroDecoderInstance * instance,
    SchroPicture * picture)
{
  SCHRO_DEBUG ("adding %d", picture->picture_number);

  if (schro_queue_is_full (instance->reference_queue)) {
    SCHRO_ERROR ("auto-retiring reference picture");
    schro_queue_pop (instance->reference_queue);
  }
  schro_queue_add (instance->reference_queue, schro_picture_ref (picture),
      picture->picture_number);
}

static SchroPicture *
schro_decoder_reference_get (SchroDecoderInstance * instance,
    SchroPictureNumber picture_number)
{
  SCHRO_DEBUG ("getting %d", picture_number);
  return schro_queue_find (instance->reference_queue, picture_number);
}

static void
schro_decoder_reference_retire (SchroDecoderInstance * instance,
    SchroPictureNumber picture_number)
{
  SCHRO_DEBUG ("retiring %d", picture_number);
  schro_queue_delete (instance->reference_queue, picture_number);
}

static void
schro_decoder_error (SchroDecoder * decoder, const char *s)
{
  SCHRO_ERROR ("decoder error: %s", s);
  decoder->error = TRUE;
  if (!decoder->error_message) {
    decoder->error_message = strdup (s);
  }
}

/* compare picture numbers in a way that handles wrap around */
static int
schro_picture_n_before_m (SchroPictureNumber n, SchroPictureNumber m)
{
  /* specified as: n occurs before m if:
   *   (m - n) mod (1<<32) < D */
  return (uint32_t) (m - n) < (1u << 31);
}

/* model the reorder buffer */
static void
schro_picturequeue_rob_insert (SchroQueue * queue, SchroPicture * picture,
    unsigned windowsize)
{
  /* the implemented reorder buffer may be larger than signalled in the stream
   * to fix this, window the insertion sort to work on upto the last @windowsize@
   * elements. */
  /* NB, a window size of 1 implies coded_order */
  int i = queue->n + 1 - windowsize;

  SCHRO_ASSERT (queue->n < queue->size);

  if (i < 0)
    i = 0;

  /* find the position to insert before. */
  for (; i < queue->n; i++) {
    if (schro_picture_n_before_m (picture->picture_number,
            queue->elements[i].picture_number)) {
      break;
    }
  }

  /* make space and insert at i */
  memmove (queue->elements + i + 1, queue->elements + i,
      sizeof (SchroQueueElement) * (queue->n - i));
  queue->n++;
  queue->elements[i].data = picture;
  queue->elements[i].picture_number = picture->picture_number;
}

/**
 * schro_decoder_set_rob_size:
 * @instance a decoder instance to set the rob size
 *
 * precondition: either be using coded_order, or the instance must have
 * seen a sequence_header for guaranteed operation.
 *
 * caller must guarantee that modifying the RoB size is safe
 */
static void
schro_decoder_set_rob_size (SchroDecoderInstance * instance)
{
  if (instance->decoder->coded_order) {
    /* when using coded_order, the RoB size is 1 */
    instance->reorder_queue_size = 1;
    return;
  }

  /* set default RoB sizes */
  if (!instance->video_format.interlaced_coding) {
    instance->reorder_queue_size = 2;
  } else {
    instance->reorder_queue_size = 4;
  }

  /* todo: override rob size with value specified in sequence header */

  /* Under normal operation, the value of reorder_queue_size needs to be one
   * greater than any spec derived value, since schro is unable to bypass the
   * reorder buffer in the same way a realtime h/w system would. */
  instance->reorder_queue_size++;

  SCHRO_ASSERT (instance->reorder_queue_size <= instance->reorder_queue->size);
}

/**
 * schro_decoder_frame_is_twofield:
 *
 * Returns: TRUE if @frame can store two field picture from @instance in a
 *          pseudo-progressive manner
 */
static int
schro_decoder_frame_is_twofield (SchroDecoderInstance * instance,
    SchroFrame * frame)
{
  int picture_height =
      schro_video_format_get_picture_height (&instance->video_format);

  if (frame->height == 0) {
    /* Indicates a skipped picture */
    return FALSE;
  }

  if (frame->height == picture_height)
    return FALSE;

  if (!instance->video_format.interlaced_coding) {
    SCHRO_ERROR
        ("supplying non frame-sized pictures when frame_coding is not supported (%d should be %d)",
        frame->height, picture_height);
  }

  return TRUE;
}

static void
schro_frame_data_draw_box (SchroFrameData * fd,
    int x, int y, int size_x, int size_y, SchroMotionVector * mv);

static void
schro_decoder_telemetry (SchroPicture * picture, SchroFrame * output_picture)
{
  int i, j;
  SchroFrameData *fd = output_picture->components + 0;
  SchroParams *params = &picture->params;
  SchroMotionVector *mv;
  int skip;
  int ii, jj;

  for (j = 0; j < params->y_num_blocks; j += 4) {
    for (i = 0; i < params->x_num_blocks; i += 4) {
      mv = &picture->motion->motion_vectors[j * params->x_num_blocks + i];
      skip = 4 >> mv->split;

      for (jj = 0; jj < 4; jj += skip) {
        for (ii = 0; ii < 4; ii += skip) {
          schro_frame_data_draw_box (fd,
              (i + ii) * params->xbsep_luma,
              (j + jj) * params->ybsep_luma,
              skip * params->xbsep_luma,
              skip * params->ybsep_luma,
              &picture->motion->motion_vectors[(j + jj) * params->x_num_blocks +
                  (i + ii)]);
        }
      }
    }
  }

  if (picture->params.num_refs > 0) {
    uint8_t *data;
    int n1;
    int n2;

    data = SCHRO_FRAME_DATA_GET_LINE (fd, 0);
    n1 = picture->reference1 - picture->picture_number;
    if (picture->params.num_refs > 1) {
      n2 = picture->reference2 - picture->picture_number;
    } else {
      n2 = 0;
    }
    if (n2 < n1) {
      int x = n2;
      n2 = n1;
      n1 = x;
    }

    for (j = -4; j < -2; j++) {
      if (fd->width / 2 + j >= 0)
        data[fd->width / 2 + j] = 255;
    }
    if (n2 < 0) {
      for (i = n1; i < n2 - 1; i++) {
        for (j = i * 16; j < i * 16 + 10; j++) {
          if (fd->width / 2 + j >= 0)
            data[fd->width / 2 + j] = 255;
        }
      }
      for (i = n2; i < 0; i++) {
        for (j = i * 16; j < i * 16 + 10; j++) {
          if (fd->width / 2 + j >= 0)
            data[fd->width / 2 + j] = 255;
        }
      }
    } else {
      for (i = n1; i < 0; i++) {
        for (j = i * 16; j < i * 16 + 10; j++) {
          if (fd->width / 2 + j >= 0)
            data[fd->width / 2 + j] = 255;
        }
      }
      for (i = 0; i < n2; i++) {
        for (j = i * 16; j < i * 16 + 10; j++) {
          if (fd->width / 2 + j >= 0)
            data[fd->width / 2 + j] = 255;
        }
      }
    }

  }
}

static void
schro_frame_data_draw_line (SchroFrameData * fd, int x1, int y1, int x2,
    int y2);

static void
schro_frame_data_draw_box (SchroFrameData * fd,
    int x, int y, int size_x, int size_y, SchroMotionVector * mv)
{
  int i;
  uint8_t *data;

  if (y < fd->height) {
    data = SCHRO_FRAME_DATA_GET_LINE (fd, y);
    for (i = x; i < x + size_x; i++) {
      data[i] = 0;
    }
  }
  for (i = y; i < y + size_y && i < fd->height; i++) {
    data = SCHRO_FRAME_DATA_GET_LINE (fd, i);
    if (x < fd->width)
      data[x] = 0;
  }
  if (y + 1 < fd->height) {
    data = SCHRO_FRAME_DATA_GET_LINE (fd, y + 1);
    if (mv->pred_mode & 1) {
      if (x + size_x / 2 - 1 < fd->width)
        data[x + size_x / 2 - 1] = 0;
    }
    if (mv->pred_mode & 2) {
      if (x + size_x / 2 + 1 < fd->width)
        data[x + size_x / 2 + 1] = 0;
    }
  }
#if 0
  if (mv->pred_mode & 1) {
    data = SCHRO_FRAME_DATA_GET_LINE (fd, y + size_y / 2 + mv->u.vec.dy[0]);
    data[x + size_x / 2 + mv->u.vec.dx[0]] = 0;
  }
  if (mv->pred_mode & 2) {
    data = SCHRO_FRAME_DATA_GET_LINE (fd, y + size_y / 2 + mv->u.vec.dy[1]);
    data[x + size_x / 2 + mv->u.vec.dx[1]] = 0;
  }
#endif
  if (mv->pred_mode & 1) {
    schro_frame_data_draw_line (fd, x + size_x / 2, y + size_y / 2,
        x + size_x / 2 + mv->u.vec.dx[0], y + size_y / 2 + mv->u.vec.dy[0]);
  }
  if (mv->pred_mode & 2) {
    schro_frame_data_draw_line (fd, x + size_x / 2, y + size_y / 2,
        x + size_x / 2 + mv->u.vec.dx[1], y + size_y / 2 + mv->u.vec.dy[1]);
  }

}

static void
schro_frame_data_draw_point (SchroFrameData * fd, int x, int y)
{
  uint8_t *data;

  if (y >= 0 && y < fd->height && x >= 0 && x < fd->width) {
    data = SCHRO_FRAME_DATA_GET_LINE (fd, y);
    data[x] = 0;
  }
}

static void
schro_frame_data_draw_line (SchroFrameData * fd, int x1, int y1, int x2, int y2)
{
  int dx, dy;
  int a = 0;
  int i, j;
  int b;

  dx = abs (x1 - x2);
  dy = abs (y1 - y2);
  if (dx > dy) {
    if (x1 < x2) {
      j = y1;
      b = (y1 < y2) ? 1 : -1;
      a = dy / 2;
      for (i = x1; i <= x2; i++) {
        schro_frame_data_draw_point (fd, i, j);
        a += dy;
        if (a >= dx) {
          a -= dx;
          j += b;
        }
      }
    } else {
      j = y2;
      b = (y2 < y1) ? 1 : -1;
      a = dy / 2;
      for (i = x2; i <= x1; i++) {
        schro_frame_data_draw_point (fd, i, j);
        a += dy;
        if (a > dx) {
          a -= dx;
          j += b;
        }
      }
    }
  } else {
    if (y1 < y2) {
      i = x1;
      b = (x1 < x2) ? 1 : -1;
      a = dx / 2;
      for (j = y1; j <= y2; j++) {
        schro_frame_data_draw_point (fd, i, j);
        a += dx;
        if (a >= dy) {
          a -= dy;
          i += b;
        }
      }
    } else {
      i = x2;
      b = (x2 < x1) ? 1 : -1;
      a = dx / 2;
      for (j = y2; j <= y1; j++) {
        schro_frame_data_draw_point (fd, i, j);
        a += dx;
        if (a > dy) {
          a -= dy;
          i += b;
        }
      }
    }
  }
}
