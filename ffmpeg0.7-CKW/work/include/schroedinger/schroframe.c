

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>
#include <schroedinger/schroframe.h>
#include <schroedinger/schrogpuframe.h>
#include <schroedinger/opengl/schroopenglframe.h>
#include <schroedinger/schrovirtframe.h>
#include <schroedinger/schroorc.h>

#include <orc/orc.h>

#include <stdlib.h>
#include <string.h>


static SchroMutex *frame_mutex;

/**
 * schro_frame_new:
 *
 * Creates a new SchroFrame object.  The created frame is uninitialized
 * and has no data storage associated with it.  The caller must fill
 * in the required information.
 *
 * Returns: a new SchroFrame object
 */
SchroFrame *
schro_frame_new (void)
{
  SchroFrame *frame;

  if (frame_mutex == NULL) {
    frame_mutex = schro_mutex_new ();
  }

  frame = schro_malloc0 (sizeof (*frame));
  frame->refcount = 1;

  return frame;
}

/**
 * schro_frame_new_and_alloc:
 *
 * Creates a new SchroFrame object with the requested size and format.
 *
 * Returns: a new SchroFrame object
 */
SchroFrame *
schro_frame_new_and_alloc (SchroMemoryDomain * domain, SchroFrameFormat format,
    int width, int height)
{
  return schro_frame_new_and_alloc_extended (domain, format, width, height, 0);
}

SchroFrame *
schro_frame_new_and_alloc_extended (SchroMemoryDomain * domain,
    SchroFrameFormat format, int width, int height, int extension)
{
  SchroFrame *frame = schro_frame_new ();
  int bytes_pp;
  int h_shift, v_shift;
  int chroma_width;
  int chroma_height;
  int ext_width;
  int ext_height;

  SCHRO_ASSERT (width > 0);
  SCHRO_ASSERT (height > 0);

  frame->format = format;
  frame->width = width;
  frame->height = height;
  frame->domain = domain;
  frame->extension = extension;

  ext_width = width + extension * 2;
  ext_height = height + extension * 2;

  if (SCHRO_FRAME_IS_PACKED (format)) {
    SCHRO_ASSERT (extension == 0);

    frame->components[0].format = format;
    frame->components[0].width = width;
    frame->components[0].height = height;
    if (format == SCHRO_FRAME_FORMAT_AYUV) {
      frame->components[0].stride = width * 4;
    } else {
      frame->components[0].stride = ROUND_UP_POW2 (width, 1) * 2;
    }
    frame->components[0].length = frame->components[0].stride * height;

    if (domain) {
      frame->regions[0] = schro_memory_domain_alloc (domain,
          frame->components[0].length);
    } else {
      frame->regions[0] = schro_malloc (frame->components[0].length);
    }

    frame->components[0].data = frame->regions[0];
    frame->components[0].v_shift = 0;
    frame->components[0].h_shift = 0;

    return frame;
  }

  switch (SCHRO_FRAME_FORMAT_DEPTH (format)) {
    case SCHRO_FRAME_FORMAT_DEPTH_U8:
      bytes_pp = 1;
      break;
    case SCHRO_FRAME_FORMAT_DEPTH_S16:
      bytes_pp = 2;
      break;
    case SCHRO_FRAME_FORMAT_DEPTH_S32:
      bytes_pp = 4;
      break;
    default:
      SCHRO_ASSERT (0);
      bytes_pp = 0;
      break;
  }

  h_shift = SCHRO_FRAME_FORMAT_H_SHIFT (format);
  v_shift = SCHRO_FRAME_FORMAT_V_SHIFT (format);
  chroma_width = ROUND_UP_SHIFT (width, h_shift);
  chroma_height = ROUND_UP_SHIFT (height, v_shift);

  frame->components[0].format = format;
  frame->components[0].width = width;
  frame->components[0].height = height;
  frame->components[0].stride =
      ROUND_UP_16 ((width + extension * 2) * bytes_pp);
  frame->components[0].length =
      frame->components[0].stride * (frame->components[0].height +
      extension * 2);
  frame->components[0].v_shift = 0;
  frame->components[0].h_shift = 0;

  frame->components[1].format = format;
  frame->components[1].width = chroma_width;
  frame->components[1].height = chroma_height;
  frame->components[1].stride =
      ROUND_UP_16 ((chroma_width + extension * 2) * bytes_pp);
  frame->components[1].length =
      frame->components[1].stride * (frame->components[1].height +
      extension * 2);
  frame->components[1].v_shift = v_shift;
  frame->components[1].h_shift = h_shift;

  frame->components[2].format = format;
  frame->components[2].width = chroma_width;
  frame->components[2].height = chroma_height;
  frame->components[2].stride =
      ROUND_UP_16 ((chroma_width + extension * 2) * bytes_pp);
  frame->components[2].length =
      frame->components[2].stride * (frame->components[2].height +
      extension * 2);
  frame->components[2].v_shift = v_shift;
  frame->components[2].h_shift = h_shift;

  if (domain) {
    frame->regions[0] = schro_memory_domain_alloc (domain,
        frame->components[0].length +
        frame->components[1].length + frame->components[2].length);
  } else {
    frame->regions[0] = malloc (frame->components[0].length +
        frame->components[1].length + frame->components[2].length);
  }

  frame->components[0].data = SCHRO_OFFSET (frame->regions[0],
      frame->components[0].stride * extension + bytes_pp * extension);
  frame->components[1].data = SCHRO_OFFSET (frame->regions[0],
      frame->components[0].length +
      frame->components[1].stride * extension + bytes_pp * extension);
  frame->components[2].data = SCHRO_OFFSET (frame->regions[0],
      frame->components[0].length + frame->components[1].length +
      frame->components[2].stride * extension + bytes_pp * extension);

  return frame;
}

/**
 * schro_frame_new_from_data_YUY2:
 *
 * Creates a new SchroFrame object with the requested size using
 * the data pointed to by @data.  The data must be in YUY2 format.
 * The data must remain for the lifetime of the SchroFrame object.
 * It is recommended to use schro_frame_set_free_callback() for
 * notification when the data is no longer needed.
 *
 * Returns: a new SchroFrame object
 */
SchroFrame *
schro_frame_new_from_data_YUY2 (void *data, int width, int height)
{
  SchroFrame *frame = schro_frame_new ();

  frame->format = SCHRO_FRAME_FORMAT_YUYV;

  frame->width = width;
  frame->height = height;

  frame->components[0].format = frame->format;
  frame->components[0].width = width;
  frame->components[0].height = height;
  frame->components[0].stride = ROUND_UP_POW2 (width, 1) * 2;
  frame->components[0].data = data;
  frame->components[0].length = frame->components[0].stride * height;
  frame->components[0].v_shift = 0;
  frame->components[0].h_shift = 0;

  return frame;
}

/**
 * schro_frame_new_from_data_UYVY:
 *
 * Creates a new SchroFrame object with the requested size using
 * the data pointed to by @data.  The data must be in UYVY format.
 * The data must remain for the lifetime of the SchroFrame object.
 * It is recommended to use schro_frame_set_free_callback() for
 * notification when the data is no longer needed.
 *
 * Returns: a new SchroFrame object
 */
SchroFrame *
schro_frame_new_from_data_UYVY (void *data, int width, int height)
{
  SchroFrame *frame = schro_frame_new ();

  frame->format = SCHRO_FRAME_FORMAT_UYVY;

  frame->width = width;
  frame->height = height;

  frame->components[0].format = frame->format;
  frame->components[0].width = width;
  frame->components[0].height = height;
  frame->components[0].stride = ROUND_UP_POW2 (width, 1) * 2;
  frame->components[0].data = data;
  frame->components[0].length = frame->components[0].stride * height;
  frame->components[0].v_shift = 0;
  frame->components[0].h_shift = 0;

  return frame;
}

/**
 * schro_frame_new_from_data_UYVY_full:
 *
 * Creates a new SchroFrame object with the requested size using
 * the data pointed to by @data.  The data must be in UYVY format,
 * although the row stride is allowed to be different than what
 * would normally be calculated from @width.
 * The data must remain for the lifetime of the SchroFrame object.
 * It is recommended to use schro_frame_set_free_callback() for
 * notification when the data is no longer needed.
 *
 * Returns: a new SchroFrame object
 */
SchroFrame *
schro_frame_new_from_data_UYVY_full (void *data, int width, int height,
    int stride)
{
  SchroFrame *frame = schro_frame_new ();

  frame->format = SCHRO_FRAME_FORMAT_UYVY;

  frame->width = width;
  frame->height = height;

  frame->components[0].width = width;
  frame->components[0].height = height;
  frame->components[0].stride = stride;
  frame->components[0].data = data;
  frame->components[0].length = frame->components[0].stride * height;
  frame->components[0].v_shift = 0;
  frame->components[0].h_shift = 0;

  return frame;
}

/**
 * schro_frame_new_from_data_AYUV:
 *
 * Creates a new SchroFrame object with the requested size using
 * the data pointed to by @data.  The data must be in AYUV format.
 * The data must remain for the lifetime of the SchroFrame object.
 * It is recommended to use schro_frame_set_free_callback() for
 * notification when the data is no longer needed.
 *
 * Returns: a new SchroFrame object
 */
SchroFrame *
schro_frame_new_from_data_AYUV (void *data, int width, int height)
{
  SchroFrame *frame = schro_frame_new ();

  frame->format = SCHRO_FRAME_FORMAT_AYUV;

  frame->width = width;
  frame->height = height;

  frame->components[0].format = frame->format;
  frame->components[0].width = width;
  frame->components[0].height = height;
  frame->components[0].stride = width * 4;
  frame->components[0].data = data;
  frame->components[0].length = frame->components[0].stride * height;
  frame->components[0].v_shift = 0;
  frame->components[0].h_shift = 0;

  return frame;
}

/**
 * schro_frame_new_from_data_v216:
 *
 * Creates a new SchroFrame object with the requested size using
 * the data pointed to by @data.  The data must be in v216 format.
 * The data must remain for the lifetime of the SchroFrame object.
 * It is recommended to use schro_frame_set_free_callback() for
 * notification when the data is no longer needed.
 *
 * Returns: a new SchroFrame object
 */
SchroFrame *
schro_frame_new_from_data_v216 (void *data, int width, int height)
{
  SchroFrame *frame = schro_frame_new ();

  frame->format = SCHRO_FRAME_FORMAT_v216;

  frame->width = width;
  frame->height = height;

  frame->components[0].format = frame->format;
  frame->components[0].width = width;
  frame->components[0].height = height;
  frame->components[0].stride = ROUND_UP_POW2 (width, 1) * 4;
  frame->components[0].data = data;
  frame->components[0].length = frame->components[0].stride * height;
  frame->components[0].v_shift = 0;
  frame->components[0].h_shift = 0;

  return frame;
}

/**
 * schro_frame_new_from_data_v210:
 *
 * Creates a new SchroFrame object with the requested size using
 * the data pointed to by @data.  The data must be in v210 format.
 * The data must remain for the lifetime of the SchroFrame object.
 * It is recommended to use schro_frame_set_free_callback() for
 * notification when the data is no longer needed.
 *
 * Returns: a new SchroFrame object
 */
SchroFrame *
schro_frame_new_from_data_v210 (void *data, int width, int height)
{
  SchroFrame *frame = schro_frame_new ();

  frame->format = SCHRO_FRAME_FORMAT_v210;

  frame->width = width;
  frame->height = height;

  frame->components[0].format = frame->format;
  frame->components[0].width = width;
  frame->components[0].height = height;
  frame->components[0].stride = ((width + 5) / 6) * 16;
  frame->components[0].data = data;
  frame->components[0].length = frame->components[0].stride * height;
  frame->components[0].v_shift = 0;
  frame->components[0].h_shift = 0;

  return frame;
}

/**
 * schro_frame_new_from_data_I420:
 *
 * Creates a new SchroFrame object with the requested size using
 * the data pointed to by @data.  The data must be in I420 format.
 * The data must remain for the lifetime of the SchroFrame object.
 * It is recommended to use schro_frame_set_free_callback() for
 * notification when the data is no longer needed.
 *
 * Returns: a new SchroFrame object
 */
SchroFrame *
schro_frame_new_from_data_I420 (void *data, int width, int height)
{
  SchroFrame *frame = schro_frame_new ();

  frame->format = SCHRO_FRAME_FORMAT_U8_420;

  frame->width = width;
  frame->height = height;

  frame->components[0].format = frame->format;
  frame->components[0].width = width;
  frame->components[0].height = height;
  frame->components[0].stride = ROUND_UP_POW2 (width, 2);
  frame->components[0].data = data;
  frame->components[0].length = frame->components[0].stride *
      ROUND_UP_POW2 (frame->components[0].height, 1);
  frame->components[0].v_shift = 0;
  frame->components[0].h_shift = 0;

  frame->components[1].format = frame->format;
  frame->components[1].width = ROUND_UP_SHIFT (width, 1);
  frame->components[1].height = ROUND_UP_SHIFT (height, 1);
  frame->components[1].stride = ROUND_UP_POW2 (frame->components[1].width, 2);
  frame->components[1].length =
      frame->components[1].stride * frame->components[1].height;
  frame->components[1].data = SCHRO_OFFSET (frame->components[0].data,
      frame->components[0].length);
  frame->components[1].v_shift = 1;
  frame->components[1].h_shift = 1;

  frame->components[2].format = frame->format;
  frame->components[2].width = ROUND_UP_SHIFT (width, 1);
  frame->components[2].height = ROUND_UP_SHIFT (height, 1);
  frame->components[2].stride = ROUND_UP_POW2 (frame->components[2].width, 2);
  frame->components[2].length =
      frame->components[2].stride * frame->components[2].height;
  frame->components[2].data = SCHRO_OFFSET (frame->components[1].data,
      frame->components[1].length);
  frame->components[2].v_shift = 1;
  frame->components[2].h_shift = 1;

  return frame;
}

/**
 * schro_frame_new_from_data_Y42B:
 *
 * Creates a new SchroFrame object with the requested size using
 * the data pointed to by @data.  The data must be in Y42B format.
 * The data must remain for the lifetime of the SchroFrame object.
 * It is recommended to use schro_frame_set_free_callback() for
 * notification when the data is no longer needed.
 *
 * Returns: a new SchroFrame object
 */
SchroFrame *
schro_frame_new_from_data_Y42B (void *data, int width, int height)
{
  SchroFrame *frame = schro_frame_new ();

  frame->format = SCHRO_FRAME_FORMAT_U8_422;

  frame->width = width;
  frame->height = height;

  frame->components[0].format = frame->format;
  frame->components[0].width = width;
  frame->components[0].height = height;
  frame->components[0].stride = ROUND_UP_POW2 (width, 2);
  frame->components[0].data = data;
  frame->components[0].length = frame->components[0].stride *
      ROUND_UP_POW2 (frame->components[0].height, 1);
  frame->components[0].v_shift = 0;
  frame->components[0].h_shift = 0;

  frame->components[1].format = frame->format;
  frame->components[1].width = ROUND_UP_SHIFT (width, 1);
  frame->components[1].height = height;
  frame->components[1].stride = ROUND_UP_POW2 (frame->components[1].width, 2);
  frame->components[1].length =
      frame->components[1].stride * frame->components[1].height;
  frame->components[1].data = SCHRO_OFFSET (frame->components[0].data,
      frame->components[0].length);
  frame->components[1].v_shift = 0;
  frame->components[1].h_shift = 1;

  frame->components[2].format = frame->format;
  frame->components[2].width = ROUND_UP_SHIFT (width, 1);
  frame->components[2].height = height;
  frame->components[2].stride = ROUND_UP_POW2 (frame->components[2].width, 2);
  frame->components[2].length =
      frame->components[2].stride * frame->components[2].height;
  frame->components[2].data = SCHRO_OFFSET (frame->components[1].data,
      frame->components[1].length);
  frame->components[2].v_shift = 0;
  frame->components[2].h_shift = 1;

  return frame;
}

/**
 * schro_frame_new_from_data_Y444:
 *
 * Creates a new SchroFrame object with the requested size using
 * the data pointed to by @data.  The data must be in Y444 format.
 * The data must remain for the lifetime of the SchroFrame object.
 * It is recommended to use schro_frame_set_free_callback() for
 * notification when the data is no longer needed.
 *
 * Returns: a new SchroFrame object
 */
SchroFrame *
schro_frame_new_from_data_Y444 (void *data, int width, int height)
{
  SchroFrame *frame = schro_frame_new ();

  frame->format = SCHRO_FRAME_FORMAT_U8_420;

  frame->width = width;
  frame->height = height;

  frame->components[0].format = frame->format;
  frame->components[0].width = width;
  frame->components[0].height = height;
  frame->components[0].stride = ROUND_UP_POW2 (width, 2);
  frame->components[0].data = data;
  frame->components[0].length = frame->components[0].stride *
      ROUND_UP_POW2 (frame->components[0].height, 1);
  frame->components[0].v_shift = 0;
  frame->components[0].h_shift = 0;

  frame->components[1].format = frame->format;
  frame->components[1].width = width;
  frame->components[1].height = height;
  frame->components[1].stride = ROUND_UP_POW2 (frame->components[1].width, 2);
  frame->components[1].length =
      frame->components[1].stride * frame->components[1].height;
  frame->components[1].data = SCHRO_OFFSET (frame->components[0].data,
      frame->components[0].length);
  frame->components[1].v_shift = 0;
  frame->components[1].h_shift = 0;

  frame->components[2].format = frame->format;
  frame->components[2].width = width;
  frame->components[2].height = height;
  frame->components[2].stride = ROUND_UP_POW2 (frame->components[2].width, 2);
  frame->components[2].length =
      frame->components[2].stride * frame->components[2].height;
  frame->components[2].data = SCHRO_OFFSET (frame->components[1].data,
      frame->components[1].length);
  frame->components[2].v_shift = 0;
  frame->components[2].h_shift = 0;

  return frame;
}

/**
 * schro_frame_new_from_data_YV12:
 *
 * Creates a new SchroFrame object with the requested size using
 * the data pointed to by @data.  The data must be in YV12 format.
 * The data must remain for the lifetime of the SchroFrame object.
 * It is recommended to use schro_frame_set_free_callback() for
 * notification when the data is no longer needed.
 *
 * Returns: a new SchroFrame object
 */
SchroFrame *
schro_frame_new_from_data_YV12 (void *data, int width, int height)
{
  SchroFrame *frame = schro_frame_new ();

  frame->format = SCHRO_FRAME_FORMAT_U8_420;

  frame->width = width;
  frame->height = height;

  frame->components[0].format = frame->format;
  frame->components[0].width = width;
  frame->components[0].height = height;
  frame->components[0].stride = ROUND_UP_POW2 (width, 2);
  frame->components[0].data = data;
  frame->components[0].length = frame->components[0].stride *
      ROUND_UP_POW2 (frame->components[0].height, 1);
  frame->components[0].v_shift = 0;
  frame->components[0].h_shift = 0;

  frame->components[2].format = frame->format;
  frame->components[2].width = ROUND_UP_SHIFT (width, 1);
  frame->components[2].height = ROUND_UP_SHIFT (height, 1);
  frame->components[2].stride = ROUND_UP_POW2 (frame->components[2].width, 2);
  frame->components[2].length =
      frame->components[2].stride * frame->components[2].height;
  frame->components[2].data = SCHRO_OFFSET (frame->components[0].data,
      frame->components[0].length);
  frame->components[2].v_shift = 1;
  frame->components[2].h_shift = 1;

  frame->components[1].format = frame->format;
  frame->components[1].width = ROUND_UP_SHIFT (width, 1);
  frame->components[1].height = ROUND_UP_SHIFT (height, 1);
  frame->components[1].stride = ROUND_UP_POW2 (frame->components[1].width, 2);
  frame->components[1].length =
      frame->components[1].stride * frame->components[1].height;
  frame->components[1].data = SCHRO_OFFSET (frame->components[2].data,
      frame->components[2].length);
  frame->components[1].v_shift = 1;
  frame->components[1].h_shift = 1;

  return frame;
}

/**
 * schro_frame_dup:
 *
 * Creates a new SchroFrame object with the same dimensions and format
 * as @frame, and copies the data from the @frame to the new object.
 *
 * Returns: a new SchroFrame object
 */
SchroFrame *
schro_frame_dup (SchroFrame * frame)
{
  return schro_frame_dup_extended (frame, 0);
}

SchroFrame *
schro_frame_dup_extended (SchroFrame * frame, int extension)
{
  SchroFrame *dup_frame;

  dup_frame = schro_frame_new_and_alloc_extended (frame->domain,
      frame->format, frame->width, frame->height, extension);
  schro_frame_convert (dup_frame, frame);

  return dup_frame;
}

/**
 * schro_frame_clone:
 *
 * Creates a new SchroFrame object with the same dimensions and format
 * as @frame.  This function leaves the data in the new object
 * uninitialized.
 *
 * Returns: a new SchroFrame object
 */
SchroFrame *
schro_frame_clone (SchroMemoryDomain * domain, SchroFrame * frame)
{
  return schro_frame_new_and_alloc (domain,
      frame->format, frame->width, frame->height);
}


/**
 * schro_frame_ref:
 * @frame: a frame object
 *
 * Increases the reference count of @frame.
 *
 * Returns: the value of @frame
 */
SchroFrame *
schro_frame_ref (SchroFrame * frame)
{
  SCHRO_ASSERT (frame && frame->refcount > 0);
  schro_mutex_lock (frame_mutex);
  frame->refcount++;
  schro_mutex_unlock (frame_mutex);
  return frame;
}


/**
 * schro_frame_unref:
 * @frame: a frame object
 *
 * Decreases the reference count of @frame.  If the new reference
 * count is 0, the frame is freed.  If a frame free callback was
 * set, this function is called.
 *
 * Returns: the value of @frame
 */
void
schro_frame_unref (SchroFrame * frame)
{
  int i;

  SCHRO_ASSERT (frame->refcount > 0);

  schro_mutex_lock (frame_mutex);
  frame->refcount--;
  if (frame->refcount == 0) {
    schro_mutex_unlock (frame_mutex);
    if (frame->free) {
      frame->free (frame, frame->priv);
    }
#ifdef HAVE_OPENGL
    if (SCHRO_FRAME_IS_OPENGL (frame)) {
      schro_opengl_frame_cleanup (frame);
    }
#endif

    for (i = 0; i < 3; i++) {
      if (frame->regions[i]) {
        if (frame->domain) {
          schro_memory_domain_memfree (frame->domain, frame->regions[i]);
        } else {
          free (frame->regions[i]);
        }
      }
    }

    if (frame->virt_frame1) {
      schro_frame_unref (frame->virt_frame1);
    }
    if (frame->virt_frame2) {
      schro_frame_unref (frame->virt_frame2);
    }
    if (frame->virt_priv) {
      schro_free (frame->virt_priv);
    }

    schro_free (frame);
  } else {
    schro_mutex_unlock (frame_mutex);
  }
}

/**
 * schro_frame_set_free_callback:
 * @frame: a frame object
 * @free_func: the function to call when the frame is freed
 * @priv: callback key
 *
 * Sets a function that will be called when the object reference
 * count drops to zero and the object is freed.
 */
void
schro_frame_set_free_callback (SchroFrame * frame,
    SchroFrameFreeFunc free_func, void *priv)
{
  frame->free = free_func;
  frame->priv = priv;
}

static void
schro_frame_component_clear (SchroFrameData * fd)
{
  //int j;

  if (SCHRO_FRAME_FORMAT_DEPTH (fd->format) == SCHRO_FRAME_FORMAT_DEPTH_U8) {
    orc_splat_u8_2d (fd->data, fd->stride, 0, fd->width, fd->height);
  } else {
    orc_splat_s16_2d (fd->data, fd->stride, 0, fd->width, fd->height);
  }
}

void
schro_frame_clear (SchroFrame * frame)
{
  schro_frame_component_clear (frame->components + 0);
  schro_frame_component_clear (frame->components + 1);
  schro_frame_component_clear (frame->components + 2);
}

typedef void (*SchroFrameBinaryFunc) (SchroFrame * dest, SchroFrame * src);

struct binary_struct
{
  SchroFrameFormat from;
  SchroFrameFormat to;
  SchroFrameBinaryFunc func;
};

/**
 * schro_frame_convert:
 * @dest: destination frame
 * @src: source frame
 *
 * Copies data from the source frame to the destination frame, converting
 * formats if necessary.  Only a few conversions are supported.
 */
void
schro_frame_convert (SchroFrame * dest, SchroFrame * src)
{
  SchroFrame *frame;
  SchroFrameFormat dest_format;

  SCHRO_ASSERT (dest != NULL);
  SCHRO_ASSERT (src != NULL);

  switch (dest->format) {
    case SCHRO_FRAME_FORMAT_YUYV:
    case SCHRO_FRAME_FORMAT_UYVY:
      dest_format = SCHRO_FRAME_FORMAT_U8_422;
      break;
    case SCHRO_FRAME_FORMAT_AYUV:
    case SCHRO_FRAME_FORMAT_ARGB:
      dest_format = SCHRO_FRAME_FORMAT_U8_444;
      break;
    case SCHRO_FRAME_FORMAT_v210:
    case SCHRO_FRAME_FORMAT_v216:
      dest_format = SCHRO_FRAME_FORMAT_S16_422;
      break;
    default:
      dest_format = dest->format;
      break;
  }
  schro_frame_ref (src);

  frame = schro_virt_frame_new_unpack (src);
  SCHRO_DEBUG ("unpack %p", frame);

  if (SCHRO_FRAME_FORMAT_DEPTH (dest_format) !=
      SCHRO_FRAME_FORMAT_DEPTH (frame->format)) {
    if (SCHRO_FRAME_FORMAT_DEPTH (dest_format) == SCHRO_FRAME_FORMAT_DEPTH_U8) {
      frame = schro_virt_frame_new_convert_u8 (frame);
      SCHRO_DEBUG ("convert_u8 %p", frame);
    } else if (SCHRO_FRAME_FORMAT_DEPTH (dest_format) ==
        SCHRO_FRAME_FORMAT_DEPTH_S16) {
      frame = schro_virt_frame_new_convert_s16 (frame);
      SCHRO_DEBUG ("convert_s16 %p", frame);
    }
  }

  if ((dest_format & 3) != (frame->format & 3)) {
    frame = schro_virt_frame_new_subsample (frame, dest_format);
    SCHRO_DEBUG ("subsample %p", frame);
  }

  if (dest->width < frame->width || dest->height < frame->height) {
    SCHRO_DEBUG ("crop %d %d to %d %d",
        frame->width, frame->height, dest->width, dest->height);

    frame = schro_virt_frame_new_crop (frame, dest->width, dest->height);
    SCHRO_DEBUG ("crop %p", frame);
  }
  if (dest->width > src->width || dest->height > src->height) {
    frame = schro_virt_frame_new_edgeextend (frame, dest->width, dest->height);
    SCHRO_DEBUG ("edgeextend %p", frame);
  }

  switch (dest->format) {
    case SCHRO_FRAME_FORMAT_YUYV:
      frame = schro_virt_frame_new_pack_YUY2 (frame);
      SCHRO_DEBUG ("pack_YUY2 %p", frame);
      break;
    case SCHRO_FRAME_FORMAT_UYVY:
      frame = schro_virt_frame_new_pack_UYVY (frame);
      SCHRO_DEBUG ("pack_UYVY %p", frame);
      break;
    case SCHRO_FRAME_FORMAT_AYUV:
      frame = schro_virt_frame_new_pack_AYUV (frame);
      SCHRO_DEBUG ("pack_AYUV %p", frame);
      break;
    case SCHRO_FRAME_FORMAT_v210:
      frame = schro_virt_frame_new_pack_v210 (frame);
      SCHRO_DEBUG ("pack_v210 %p", frame);
      break;
    case SCHRO_FRAME_FORMAT_v216:
      frame = schro_virt_frame_new_pack_v216 (frame);
      SCHRO_DEBUG ("pack_v216 %p", frame);
      break;
    default:
      break;
  }

  schro_virt_frame_render (frame, dest);
  schro_frame_unref (frame);

}

static void schro_frame_add_s16_s16 (SchroFrame * dest, SchroFrame * src);
static void schro_frame_add_s16_u8 (SchroFrame * dest, SchroFrame * src);

static struct binary_struct schro_frame_add_func_list[] = {
  {SCHRO_FRAME_FORMAT_S16_444, SCHRO_FRAME_FORMAT_S16_444,
      schro_frame_add_s16_s16},
  {SCHRO_FRAME_FORMAT_S16_422, SCHRO_FRAME_FORMAT_S16_422,
      schro_frame_add_s16_s16},
  {SCHRO_FRAME_FORMAT_S16_420, SCHRO_FRAME_FORMAT_S16_420,
      schro_frame_add_s16_s16},

  {SCHRO_FRAME_FORMAT_U8_444, SCHRO_FRAME_FORMAT_S16_444,
      schro_frame_add_s16_u8},
  {SCHRO_FRAME_FORMAT_U8_422, SCHRO_FRAME_FORMAT_S16_422,
      schro_frame_add_s16_u8},
  {SCHRO_FRAME_FORMAT_U8_420, SCHRO_FRAME_FORMAT_S16_420,
      schro_frame_add_s16_u8},

  {0}
};

/**
 * schro_frame_add:
 * @dest: destination frame
 * @src: source frame
 *
 * Adds data from the source frame to the destination frame.  The
 * frames must have the same chroma subsampling, and only a few
 * combinations of bit depths are supported.
 */
void
schro_frame_add (SchroFrame * dest, SchroFrame * src)
{
  int i;

  SCHRO_ASSERT (dest != NULL);
  SCHRO_ASSERT (src != NULL);

  for (i = 0; schro_frame_add_func_list[i].func; i++) {
    if (schro_frame_add_func_list[i].from == src->format &&
        schro_frame_add_func_list[i].to == dest->format) {
      schro_frame_add_func_list[i].func (dest, src);
      return;
    }
  }

  SCHRO_ERROR ("add function unimplemented");
  SCHRO_ASSERT (0);
}

static void schro_frame_subtract_s16_s16 (SchroFrame * dest, SchroFrame * src);
static void schro_frame_subtract_s16_u8 (SchroFrame * dest, SchroFrame * src);

static struct binary_struct schro_frame_subtract_func_list[] = {
  {SCHRO_FRAME_FORMAT_S16_444, SCHRO_FRAME_FORMAT_S16_444,
      schro_frame_subtract_s16_s16},
  {SCHRO_FRAME_FORMAT_S16_422, SCHRO_FRAME_FORMAT_S16_422,
      schro_frame_subtract_s16_s16},
  {SCHRO_FRAME_FORMAT_S16_420, SCHRO_FRAME_FORMAT_S16_420,
      schro_frame_subtract_s16_s16},

  {SCHRO_FRAME_FORMAT_U8_444, SCHRO_FRAME_FORMAT_S16_444,
      schro_frame_subtract_s16_u8},
  {SCHRO_FRAME_FORMAT_U8_422, SCHRO_FRAME_FORMAT_S16_422,
      schro_frame_subtract_s16_u8},
  {SCHRO_FRAME_FORMAT_U8_420, SCHRO_FRAME_FORMAT_S16_420,
      schro_frame_subtract_s16_u8},

  {0}
};

/**
 * schro_frame_subtract:
 * @dest: destination frame
 * @src: source frame
 *
 * Subtracts data from the source frame to the destination frame.  The
 * frames must have the same chroma subsampling, and only a few
 * combinations of bit depths are supported.
 */
void
schro_frame_subtract (SchroFrame * dest, SchroFrame * src)
{
  int i;

  SCHRO_ASSERT (dest != NULL);
  SCHRO_ASSERT (src != NULL);

  for (i = 0; schro_frame_subtract_func_list[i].func; i++) {
    if (schro_frame_subtract_func_list[i].from == src->format &&
        schro_frame_subtract_func_list[i].to == dest->format) {
      schro_frame_subtract_func_list[i].func (dest, src);
      return;
    }
  }

  SCHRO_ERROR (0);
  SCHRO_ASSERT ("subtract function unimplemented");
}


static void
schro_frame_add_s16_s16 (SchroFrame * dest, SchroFrame * src)
{
  SchroFrameData *dcomp;
  SchroFrameData *scomp;
  int i;
  int width, height;

  for (i = 0; i < 3; i++) {
    dcomp = &dest->components[i];
    scomp = &src->components[i];

    width = MIN (dcomp->width, scomp->width);
    height = MIN (dcomp->height, scomp->height);

    orc_add_s16_2d (dcomp->data, dcomp->stride,
        scomp->data, scomp->stride, width, height);
  }
}

static void
schro_frame_add_s16_u8 (SchroFrame * dest, SchroFrame * src)
{
  SchroFrameData *dcomp;
  SchroFrameData *scomp;
  //int16_t *ddata;
  //uint8_t *sdata;
  int i;
  //int y;
  int width, height;

  for (i = 0; i < 3; i++) {
    dcomp = &dest->components[i];
    scomp = &src->components[i];

    width = MIN (dcomp->width, scomp->width);
    height = MIN (dcomp->height, scomp->height);

    orc_add_s16_u8_2d (dcomp->data, dcomp->stride,
        scomp->data, scomp->stride, width, height);
#if 0
    for (y = 0; y < height; y++) {
      sdata = SCHRO_FRAME_DATA_GET_LINE (scomp, y);
      ddata = SCHRO_FRAME_DATA_GET_LINE (dcomp, y);
      orc_add_s16_u8 (ddata, ddata, sdata, width);
    }
#endif
  }
}

static void
schro_frame_subtract_s16_s16 (SchroFrame * dest, SchroFrame * src)
{
  SchroFrameData *dcomp;
  SchroFrameData *scomp;
  int16_t *ddata;
  int16_t *sdata;
  int i;
  int y;
  int width, height;

  for (i = 0; i < 3; i++) {
    dcomp = &dest->components[i];
    scomp = &src->components[i];

    width = MIN (dcomp->width, scomp->width);
    height = MIN (dcomp->height, scomp->height);

    for (y = 0; y < height; y++) {
      sdata = SCHRO_FRAME_DATA_GET_LINE (scomp, y);
      ddata = SCHRO_FRAME_DATA_GET_LINE (dcomp, y);
      orc_subtract_s16 (ddata, ddata, sdata, width);
    }
  }
}

static void
schro_frame_subtract_s16_u8 (SchroFrame * dest, SchroFrame * src)
{
  SchroFrameData *dcomp;
  SchroFrameData *scomp;
  int16_t *ddata;
  uint8_t *sdata;
  int i;
  int y;
  int width, height;

  for (i = 0; i < 3; i++) {
    dcomp = &dest->components[i];
    scomp = &src->components[i];

    width = MIN (dcomp->width, scomp->width);
    height = MIN (dcomp->height, scomp->height);

    for (y = 0; y < height; y++) {
      sdata = SCHRO_FRAME_DATA_GET_LINE (scomp, y);
      ddata = SCHRO_FRAME_DATA_GET_LINE (dcomp, y);
      orc_subtract_s16_u8 (ddata, ddata, sdata, width);
    }
  }
}

/**
 * schro_frame_iwt_transform:
 * @frame: frame
 * @params: transform parameters
 *
 * Performs an in-place integer wavelet transform on @frame.  The
 * frame must have a bit depth of 16.
 */
void
schro_frame_iwt_transform (SchroFrame * frame, SchroParams * params)
{
  int component;
  int width;
  int height;
  int level;
  int16_t *tmp;

  tmp = schro_malloc (sizeof (int16_t) * (params->iwt_luma_width + 16));

  for (component = 0; component < 3; component++) {
    SchroFrameData *comp = &frame->components[component];

    if (component == 0) {
      width = params->iwt_luma_width;
      height = params->iwt_luma_height;
    } else {
      width = params->iwt_chroma_width;
      height = params->iwt_chroma_height;
    }

    for (level = 0; level < params->transform_depth; level++) {
      SchroFrameData fd;

      fd.format = frame->format;
      fd.data = comp->data;
      fd.width = width >> level;
      fd.height = height >> level;
      fd.stride = comp->stride << level;

      schro_wavelet_transform_2d (&fd, params->wavelet_filter_index, tmp);
    }
  }

  schro_free (tmp);
}

/**
 * schro_frame_inverse_iwt_transform:
 * @frame: frame
 * @params: transform parameters
 *
 * Performs an in-place inverse integer wavelet transform on @frame.  The
 * frame must have a bit depth of 16.
 */
void
schro_frame_inverse_iwt_transform (SchroFrame * frame, SchroParams * params)
{
  int width;
  int height;
  int level;
  int component;
  int16_t *tmp;

  tmp = schro_malloc (sizeof (int16_t) * (params->iwt_luma_width + 16));

  for (component = 0; component < 3; component++) {
    SchroFrameData *comp = &frame->components[component];

    if (component == 0) {
      width = params->iwt_luma_width;
      height = params->iwt_luma_height;
    } else {
      width = params->iwt_chroma_width;
      height = params->iwt_chroma_height;
    }

    for (level = params->transform_depth - 1; level >= 0; level--) {
      SchroFrameData fd;

      fd.format = frame->format;
      fd.data = comp->data;
      fd.width = width >> level;
      fd.height = height >> level;
      fd.stride = comp->stride << level;

      schro_wavelet_inverse_transform_2d (&fd, params->wavelet_filter_index,
          tmp);
    }
  }

  schro_free (tmp);
}

/**
 * schro_frame_shift_left:
 * @frame: frame
 * @shift: number of bits to shift
 *
 * Shifts each value in @frame to the left by @shift bits.  This
 * operation happens in-place.
 */
void
schro_frame_shift_left (SchroFrame * frame, int shift)
{
  SchroFrameData *comp;
  int16_t *data;
  int i;
  int y;

  for (i = 0; i < 3; i++) {
    comp = &frame->components[i];

    for (y = 0; y < comp->height; y++) {
      data = SCHRO_FRAME_DATA_GET_LINE (comp, y);
      orc_lshift_s16_ip (data, shift, comp->width);
    }
  }
}

/**
 * schro_frame_shift_right:
 * @frame: frame
 * @shift: number of bits to shift
 *
 * Shifts each value in @frame to the right by @shift bits.  This
 * operation happens in-place.
 */
void
schro_frame_shift_right (SchroFrame * frame, int shift)
{
  SchroFrameData *comp;
  int16_t *data;
  int i;
  int y;

  for (i = 0; i < 3; i++) {
    comp = &frame->components[i];

    for (y = 0; y < comp->height; y++) {
      data = SCHRO_FRAME_DATA_GET_LINE (comp, y);
      orc_add_const_rshift_s16 (data, (1 << shift) >> 1, shift, comp->width);
    }
  }
}

#ifdef unused
/**
 * schro_frame_edge_extend:
 * @frame: frame
 * @width: width of subpicture
 * @height: height of subpicture
 *
 * Extends the edges of the subpicture defined from 0,0 to @width,@height
 * to the size of @frame.
 */
void
schro_frame_edge_extend (SchroFrame * frame, int width, int height)
{
  SchroFrameData *comp;
  int i;
  int y;
  int chroma_width;
  int chroma_height;

  SCHRO_DEBUG ("extending %d %d -> %d %d", width, height,
      frame->width, frame->height);

  chroma_width = ROUND_UP_SHIFT (width,
      SCHRO_FRAME_FORMAT_H_SHIFT (frame->format));
  chroma_height = ROUND_UP_SHIFT (height,
      SCHRO_FRAME_FORMAT_V_SHIFT (frame->format));

  SCHRO_DEBUG ("chroma %d %d -> %d %d", chroma_width, chroma_height,
      frame->components[1].width, frame->components[1].height);

  switch (SCHRO_FRAME_FORMAT_DEPTH (frame->format)) {
    case SCHRO_FRAME_FORMAT_DEPTH_U8:
      for (i = 0; i < 3; i++) {
        uint8_t *data;
        int w, h;

        comp = &frame->components[i];
        data = comp->data;

        w = (i > 0) ? chroma_width : width;
        h = (i > 0) ? chroma_height : height;

        if (w < comp->width) {
          for (y = 0; y < MIN (h, comp->height); y++) {
            data = SCHRO_FRAME_DATA_GET_LINE (comp, y);
            orc_splat_u8_ns (data + w, data[w - 1], comp->width - w);
          }
        }
        for (y = h; y < comp->height; y++) {
          orc_memcpy (SCHRO_FRAME_DATA_GET_LINE (comp, y),
              SCHRO_FRAME_DATA_GET_LINE (comp, h - 1), comp->width);
        }
      }
      break;
    case SCHRO_FRAME_FORMAT_DEPTH_S16:
      for (i = 0; i < 3; i++) {
        int16_t *data;
        int w, h;

        comp = &frame->components[i];
        data = comp->data;

        w = (i > 0) ? chroma_width : width;
        h = (i > 0) ? chroma_height : height;

        if (w < comp->width) {
          for (y = 0; y < MIN (h, comp->height); y++) {
            data = SCHRO_FRAME_DATA_GET_LINE (comp, y);
            orc_splat_s16_ns (data + w, data[w - 1], comp->width - w);
          }
        }
        for (y = h; y < comp->height; y++) {
          orc_memcpy (SCHRO_FRAME_DATA_GET_LINE (comp, y),
              SCHRO_FRAME_DATA_GET_LINE (comp, h - 1), comp->width * 2);
        }
      }
      break;
    default:
      SCHRO_ERROR ("unimplemented case");
      SCHRO_ASSERT (0);
      break;
  }
}
#endif

void
schro_frame_zero_extend (SchroFrame * frame, int width, int height)
{
  SchroFrameData *comp;
  int i;
  int y;
  int chroma_width;
  int chroma_height;

  SCHRO_DEBUG ("extending %d %d -> %d %d", width, height,
      frame->width, frame->height);

  chroma_width = ROUND_UP_SHIFT (width,
      SCHRO_FRAME_FORMAT_H_SHIFT (frame->format));
  chroma_height = ROUND_UP_SHIFT (height,
      SCHRO_FRAME_FORMAT_V_SHIFT (frame->format));

  switch (SCHRO_FRAME_FORMAT_DEPTH (frame->format)) {
    case SCHRO_FRAME_FORMAT_DEPTH_U8:
      for (i = 0; i < 3; i++) {
        uint8_t *data;
        int w, h;

        comp = &frame->components[i];
        data = comp->data;

        w = (i > 0) ? chroma_width : width;
        h = (i > 0) ? chroma_height : height;

        if (w < comp->width) {
          for (y = 0; y < h; y++) {
            data = SCHRO_FRAME_DATA_GET_LINE (comp, y);
            orc_splat_u8_ns (data + w, 0, comp->width - w);
          }
        }
        for (y = h; y < comp->height; y++) {
          orc_splat_u8_ns (SCHRO_FRAME_DATA_GET_LINE (comp, y), 0, comp->width);
        }
      }
      break;
    case SCHRO_FRAME_FORMAT_DEPTH_S16:
      for (i = 0; i < 3; i++) {
        int16_t *data;
        int w, h;

        comp = &frame->components[i];
        data = comp->data;

        w = (i > 0) ? chroma_width : width;
        h = (i > 0) ? chroma_height : height;

        if (w < comp->width) {
          for (y = 0; y < h; y++) {
            data = SCHRO_FRAME_DATA_GET_LINE (comp, y);
            orc_splat_s16_ns (data + w, 0, comp->width - w);
          }
        }
        for (y = h; y < comp->height; y++) {
          orc_splat_s16_ns (SCHRO_FRAME_DATA_GET_LINE (comp, y), 0,
              comp->width);
        }
      }
      break;
    default:
      SCHRO_ERROR ("unimplemented case");
      break;
  }
}

static void
downsample_horiz_u8 (uint8_t * dest, int n_dest, uint8_t * src, int n_src)
{
  int i;

  if (n_dest < 4) {
    for (i = 0; i < n_dest; i++) {
      int x = 0;
      x += 6 * src[CLAMP (i * 2 - 1, 0, n_src - 1)];
      x += 26 * src[CLAMP (i * 2 + 0, 0, n_src - 1)];
      x += 26 * src[CLAMP (i * 2 + 1, 0, n_src - 1)];
      x += 6 * src[CLAMP (i * 2 + 2, 0, n_src - 1)];
      dest[i] = CLAMP ((x + 32) >> 6, 0, 255);
    }
  } else {
    for (i = 0; i < 1; i++) {
      int x = 0;
      x += 6 * src[CLAMP (i * 2 - 1, 0, n_src - 1)];
      x += 26 * src[CLAMP (i * 2 + 0, 0, n_src - 1)];
      x += 26 * src[CLAMP (i * 2 + 1, 0, n_src - 1)];
      x += 6 * src[CLAMP (i * 2 + 2, 0, n_src - 1)];
      dest[i] = CLAMP ((x + 32) >> 6, 0, 255);
    }
    orc_downsample_horiz_u8 (dest + 1, src, n_src / 2 - 2);
    for (i = n_src / 2 - 2; i < n_dest; i++) {
      int x = 0;
      x += 6 * src[CLAMP (i * 2 - 1, 0, n_src - 1)];
      x += 26 * src[CLAMP (i * 2 + 0, 0, n_src - 1)];
      x += 26 * src[CLAMP (i * 2 + 1, 0, n_src - 1)];
      x += 6 * src[CLAMP (i * 2 + 2, 0, n_src - 1)];
      dest[i] = CLAMP ((x + 32) >> 6, 0, 255);
    }

  }
}

static void
schro_frame_component_downsample (SchroFrameData * dest, SchroFrameData * src)
{
  int i;
  uint8_t *tmp;

  tmp = schro_malloc (src->width);

  for (i = 0; i < dest->height; i++) {
    orc_downsample_vert_u8 (tmp,
        SCHRO_FRAME_DATA_GET_LINE (src, CLAMP (i * 2 - 1, 0, src->height - 1)),
        SCHRO_FRAME_DATA_GET_LINE (src, CLAMP (i * 2 + 0, 0, src->height - 1)),
        SCHRO_FRAME_DATA_GET_LINE (src, CLAMP (i * 2 + 1, 0, src->height - 1)),
        SCHRO_FRAME_DATA_GET_LINE (src, CLAMP (i * 2 + 2, 0, src->height - 1)),
        src->width);
    downsample_horiz_u8 (SCHRO_FRAME_DATA_GET_LINE (dest, i), dest->width,
        tmp, src->width);
  }

  schro_free (tmp);
}

void
schro_frame_downsample (SchroFrame * dest, SchroFrame * src)
{
  schro_frame_component_downsample (&dest->components[0], &src->components[0]);
  schro_frame_component_downsample (&dest->components[1], &src->components[1]);
  schro_frame_component_downsample (&dest->components[2], &src->components[2]);
}

static void
mas8_u8_edgeextend (uint8_t * d, const uint8_t * s,
    const int16_t * taps, int offset, int shift, int index_offset, int n)
{
  int i, j;
  int x;

  if (n <= 8) {
    for (i = 0; i < n; i++) {
      x = 0;
      for (j = 0; j < 8; j++) {
        x += s[CLAMP (i + j - index_offset, 0, n - 1)] * taps[j];
      }
      d[i] = CLAMP ((x + offset) >> shift, 0, 255);
    }
  } else {
    for (i = 0; i < index_offset; i++) {
      x = 0;
      for (j = 0; j < 8; j++) {
        x += s[CLAMP (i + j - index_offset, 0, n - 1)] * taps[j];
      }
      d[i] = CLAMP ((x + offset) >> shift, 0, 255);
    }
    for (i = index_offset; i < n - 8 + index_offset; i++) {
      x = 0;
      for (j = 0; j < 8; j++) {
        x += s[i + j - index_offset] * taps[j];
      }
      d[i] = CLAMP ((x + offset) >> shift, 0, 255);
    }
    for (i = n - 8 + index_offset; i < n; i++) {
      x = 0;
      for (j = 0; j < 8; j++) {
        x += s[CLAMP (i + j - index_offset, 0, n - 1)] * taps[j];
      }
      d[i] = CLAMP ((x + offset) >> shift, 0, 255);
    }
    i = n - 1;
    d[i] = s[i];
  }
}

void
schro_frame_upsample_horiz (SchroFrame * dest, SchroFrame * src)
{
  int j, k;
  SchroFrameData *dcomp;
  SchroFrameData *scomp;

  if (SCHRO_FRAME_FORMAT_DEPTH (dest->format) != SCHRO_FRAME_FORMAT_DEPTH_U8 ||
      SCHRO_FRAME_FORMAT_DEPTH (src->format) != SCHRO_FRAME_FORMAT_DEPTH_U8 ||
      src->format != dest->format) {
    SCHRO_ERROR ("unimplemented");
    return;
  }

  for (k = 0; k < 3; k++) {
    static const int16_t taps[8] = { -1, 3, -7, 21, 21, -7, 3, -1 };

    dcomp = &dest->components[k];
    scomp = &src->components[k];

    for (j = 0; j < dcomp->height; j++) {
      mas8_u8_edgeextend (SCHRO_FRAME_DATA_GET_LINE (dcomp, j),
          SCHRO_FRAME_DATA_GET_LINE (scomp, j), taps, 16, 5, 3, scomp->width);
    }
  }
}

static void
mas8_across_u8 (uint8_t * dest, const uint8_t * src, int stride,
    const int16_t * weights, int offset, int shift, int n)
{
  int i;
  for (i = 0; i < n; i++) {
    int x = offset;
    x += ((uint8_t *) SCHRO_OFFSET (src, stride * 0))[i] * weights[0];
    x += ((uint8_t *) SCHRO_OFFSET (src, stride * 1))[i] * weights[1];
    x += ((uint8_t *) SCHRO_OFFSET (src, stride * 2))[i] * weights[2];
    x += ((uint8_t *) SCHRO_OFFSET (src, stride * 3))[i] * weights[3];
    x += ((uint8_t *) SCHRO_OFFSET (src, stride * 4))[i] * weights[4];
    x += ((uint8_t *) SCHRO_OFFSET (src, stride * 5))[i] * weights[5];
    x += ((uint8_t *) SCHRO_OFFSET (src, stride * 6))[i] * weights[6];
    x += ((uint8_t *) SCHRO_OFFSET (src, stride * 7))[i] * weights[7];
    dest[i] = CLAMP (x >> shift, 0, 255);
  }
}

static void
mas8_across_u8_slow (uint8_t * d, uint8_t ** s1_a8,
    const int16_t * s2_8, int offset, int shift, int n)
{
  int i;
  int j;
  int x;

  for (i = 0; i < n; i++) {
    x = 0;
    for (j = 0; j < 8; j++) {
      x += s1_a8[j][i] * s2_8[j];
    }
    d[i] = CLAMP ((x + offset) >> shift, 0, 255);
  }
}

void
schro_frame_upsample_vert (SchroFrame * dest, SchroFrame * src)
{
  int i, j, k;
  SchroFrameData *dcomp;
  SchroFrameData *scomp;

  if (SCHRO_FRAME_FORMAT_DEPTH (dest->format) != SCHRO_FRAME_FORMAT_DEPTH_U8 ||
      SCHRO_FRAME_FORMAT_DEPTH (src->format) != SCHRO_FRAME_FORMAT_DEPTH_U8 ||
      src->format != dest->format) {
    SCHRO_ERROR ("unimplemented");
    return;
  }

  for (k = 0; k < 3; k++) {
    static const int16_t taps[8] = { -1, 3, -7, 21, 21, -7, 3, -1 };
    uint8_t *list[8];

    dcomp = &dest->components[k];
    scomp = &src->components[k];

    for (j = 0; j < dcomp->height - 1; j++) {
      if (j < 3 || j >= scomp->height - 4) {
        for (i = 0; i < 8; i++) {
          list[i] = SCHRO_FRAME_DATA_GET_LINE (scomp,
              CLAMP (i + j - 3, 0, scomp->height - 1));
        }
        mas8_across_u8_slow (SCHRO_FRAME_DATA_GET_LINE (dcomp, j), list,
            taps, 16, 5, scomp->width);
      } else {
        SCHRO_ASSERT (j - 3 >= 0);
        SCHRO_ASSERT (j - 3 + 7 < scomp->height);
        mas8_across_u8 (SCHRO_FRAME_DATA_GET_LINE (dcomp, j),
            SCHRO_FRAME_DATA_GET_LINE (scomp, j - 3), scomp->stride,
            taps, 16, 5, scomp->width);
      }
    }
    j = dcomp->height - 1;
    orc_memcpy (SCHRO_FRAME_DATA_GET_LINE (dcomp, j),
        SCHRO_FRAME_DATA_GET_LINE (scomp, j), dcomp->width);
  }
}

double
schro_frame_calculate_average_luma (SchroFrame * frame)
{
  SchroFrameData *comp;
  int j;
  int sum = 0;
  int n;

  comp = &frame->components[0];

  switch (SCHRO_FRAME_FORMAT_DEPTH (frame->format)) {
    case SCHRO_FRAME_FORMAT_DEPTH_U8:
      for (j = 0; j < comp->height; j++) {
        int32_t linesum;
        orc_sum_u8 (&linesum, SCHRO_FRAME_DATA_GET_LINE (comp, j), comp->width);
        sum += linesum;
      }
      break;
    case SCHRO_FRAME_FORMAT_DEPTH_S16:
      for (j = 0; j < comp->height; j++) {
        int32_t linesum;
        orc_sum_s16 (&linesum, SCHRO_FRAME_DATA_GET_LINE (comp, j),
            comp->width);
        sum += linesum;
      }
      break;
    default:
      SCHRO_ERROR ("unimplemented");
      break;
  }

  n = comp->height * comp->width;
  return (double) sum / n;
}

SchroFrame *
schro_frame_convert_to_444 (SchroFrame * frame)
{
  SchroFrame *dest;

  SCHRO_ASSERT (frame->format == SCHRO_FRAME_FORMAT_U8_420);

  dest = schro_frame_new_and_alloc (frame->domain, SCHRO_FRAME_FORMAT_U8_444,
      frame->width, frame->height);

  schro_frame_convert (dest, frame);

  return dest;
}


/* Copied from elsewhere */
/*
 * This code implements the MD5 message-digest algorithm.
 * The algorithm is due to Ron Rivest.  This code was
 * written by Colin Plumb in 1993, no copyright is claimed.
 * This code is in the public domain; do with it what you wish.
 *
 * Equivalent code is available from RSA Data Security, Inc.
 * This code has been tested against that, and is equivalent,
 * except that you don't need to include two pages of legalese
 * with every copy.
 *
 * To compute the message digest of a chunk of bytes, declare an
 * MD5Context structure, pass it to MD5Init, call MD5Update as
 * needed on buffers full of bytes, and then call MD5Final, which
 * will fill a supplied 16-byte array with the digest.
 */

#ifdef WORDS_BIGENDIAN
#define uint32_to_host(a) \
  ((((a)&0xff)<<24)|(((a)&0xff00)<<8)|(((a)&0xff0000)>>8)|(((a)>>24)&0xff))
#else
#define uint32_to_host(a) (a)
#endif

#define F1(x, y, z) (z ^ (x & (y ^ z)))
#define F2(x, y, z) F1(z, x, y)
#define F3(x, y, z) (x ^ y ^ z)
#define F4(x, y, z) (y ^ (x | ~z))

#define MD5STEP(f,w,x,y,z,in,offset,s) \
  (w += f(x,y,z) + uint32_to_host(in) + offset, w = (w<<s | w>>(32-s)) + x)

static void
schro_md5 (uint32_t * state, uint32_t * src)
{
  uint32_t a, b, c, d;

  a = state[0];
  b = state[1];
  c = state[2];
  d = state[3];

  MD5STEP (F1, a, b, c, d, src[0], 0xd76aa478, 7);
  MD5STEP (F1, d, a, b, c, src[1], 0xe8c7b756, 12);
  MD5STEP (F1, c, d, a, b, src[2], 0x242070db, 17);
  MD5STEP (F1, b, c, d, a, src[3], 0xc1bdceee, 22);
  MD5STEP (F1, a, b, c, d, src[4], 0xf57c0faf, 7);
  MD5STEP (F1, d, a, b, c, src[5], 0x4787c62a, 12);
  MD5STEP (F1, c, d, a, b, src[6], 0xa8304613, 17);
  MD5STEP (F1, b, c, d, a, src[7], 0xfd469501, 22);
  MD5STEP (F1, a, b, c, d, src[8], 0x698098d8, 7);
  MD5STEP (F1, d, a, b, c, src[9], 0x8b44f7af, 12);
  MD5STEP (F1, c, d, a, b, src[10], 0xffff5bb1, 17);
  MD5STEP (F1, b, c, d, a, src[11], 0x895cd7be, 22);
  MD5STEP (F1, a, b, c, d, src[12], 0x6b901122, 7);
  MD5STEP (F1, d, a, b, c, src[13], 0xfd987193, 12);
  MD5STEP (F1, c, d, a, b, src[14], 0xa679438e, 17);
  MD5STEP (F1, b, c, d, a, src[15], 0x49b40821, 22);

  MD5STEP (F2, a, b, c, d, src[1], 0xf61e2562, 5);
  MD5STEP (F2, d, a, b, c, src[6], 0xc040b340, 9);
  MD5STEP (F2, c, d, a, b, src[11], 0x265e5a51, 14);
  MD5STEP (F2, b, c, d, a, src[0], 0xe9b6c7aa, 20);
  MD5STEP (F2, a, b, c, d, src[5], 0xd62f105d, 5);
  MD5STEP (F2, d, a, b, c, src[10], 0x02441453, 9);
  MD5STEP (F2, c, d, a, b, src[15], 0xd8a1e681, 14);
  MD5STEP (F2, b, c, d, a, src[4], 0xe7d3fbc8, 20);
  MD5STEP (F2, a, b, c, d, src[9], 0x21e1cde6, 5);
  MD5STEP (F2, d, a, b, c, src[14], 0xc33707d6, 9);
  MD5STEP (F2, c, d, a, b, src[3], 0xf4d50d87, 14);
  MD5STEP (F2, b, c, d, a, src[8], 0x455a14ed, 20);
  MD5STEP (F2, a, b, c, d, src[13], 0xa9e3e905, 5);
  MD5STEP (F2, d, a, b, c, src[2], 0xfcefa3f8, 9);
  MD5STEP (F2, c, d, a, b, src[7], 0x676f02d9, 14);
  MD5STEP (F2, b, c, d, a, src[12], 0x8d2a4c8a, 20);

  MD5STEP (F3, a, b, c, d, src[5], 0xfffa3942, 4);
  MD5STEP (F3, d, a, b, c, src[8], 0x8771f681, 11);
  MD5STEP (F3, c, d, a, b, src[11], 0x6d9d6122, 16);
  MD5STEP (F3, b, c, d, a, src[14], 0xfde5380c, 23);
  MD5STEP (F3, a, b, c, d, src[1], 0xa4beea44, 4);
  MD5STEP (F3, d, a, b, c, src[4], 0x4bdecfa9, 11);
  MD5STEP (F3, c, d, a, b, src[7], 0xf6bb4b60, 16);
  MD5STEP (F3, b, c, d, a, src[10], 0xbebfbc70, 23);
  MD5STEP (F3, a, b, c, d, src[13], 0x289b7ec6, 4);
  MD5STEP (F3, d, a, b, c, src[0], 0xeaa127fa, 11);
  MD5STEP (F3, c, d, a, b, src[3], 0xd4ef3085, 16);
  MD5STEP (F3, b, c, d, a, src[6], 0x04881d05, 23);
  MD5STEP (F3, a, b, c, d, src[9], 0xd9d4d039, 4);
  MD5STEP (F3, d, a, b, c, src[12], 0xe6db99e5, 11);
  MD5STEP (F3, c, d, a, b, src[15], 0x1fa27cf8, 16);
  MD5STEP (F3, b, c, d, a, src[2], 0xc4ac5665, 23);

  MD5STEP (F4, a, b, c, d, src[0], 0xf4292244, 6);
  MD5STEP (F4, d, a, b, c, src[7], 0x432aff97, 10);
  MD5STEP (F4, c, d, a, b, src[14], 0xab9423a7, 15);
  MD5STEP (F4, b, c, d, a, src[5], 0xfc93a039, 21);
  MD5STEP (F4, a, b, c, d, src[12], 0x655b59c3, 6);
  MD5STEP (F4, d, a, b, c, src[3], 0x8f0ccc92, 10);
  MD5STEP (F4, c, d, a, b, src[10], 0xffeff47d, 15);
  MD5STEP (F4, b, c, d, a, src[1], 0x85845dd1, 21);
  MD5STEP (F4, a, b, c, d, src[8], 0x6fa87e4f, 6);
  MD5STEP (F4, d, a, b, c, src[15], 0xfe2ce6e0, 10);
  MD5STEP (F4, c, d, a, b, src[6], 0xa3014314, 15);
  MD5STEP (F4, b, c, d, a, src[13], 0x4e0811a1, 21);
  MD5STEP (F4, a, b, c, d, src[4], 0xf7537e82, 6);
  MD5STEP (F4, d, a, b, c, src[11], 0xbd3af235, 10);
  MD5STEP (F4, c, d, a, b, src[2], 0x2ad7d2bb, 15);
  MD5STEP (F4, b, c, d, a, src[9], 0xeb86d391, 21);

  state[0] += a;
  state[1] += b;
  state[2] += c;
  state[3] += d;
}

/* END Copied from elsewhere */


void
schro_frame_md5 (SchroFrame * frame, uint32_t * state)
{
  uint8_t *line;
  int x, y, k;

  state[0] = 0x67452301;
  state[1] = 0xefcdab89;
  state[2] = 0x98badcfe;
  state[3] = 0x10325476;

  x = 0;
  y = 0;
  k = 0;
  for (k = 0; k < 3; k++) {
    for (y = 0; y < frame->components[k].height; y++) {
      line = SCHRO_FRAME_DATA_GET_LINE (&frame->components[k], y);
      for (x = 0; x + 63 < frame->components[k].width; x += 64) {
        schro_md5 (state, (uint32_t *) (line + x));
      }
      if (x < frame->components[k].width) {
        uint8_t tmp[64];
        int left;
        left = frame->components[k].width - x;
        memcpy (tmp, line + x, left);
        memset (tmp + left, 0, 64 - left);
        schro_md5 (state, (uint32_t *) tmp);
      }
    }
  }

  SCHRO_DEBUG
      ("md5 %02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x",
      state[0] & 0xff, (state[0] >> 8) & 0xff, (state[0] >> 16) & 0xff,
      (state[0] >> 24) & 0xff, state[1] & 0xff, (state[1] >> 8) & 0xff,
      (state[1] >> 16) & 0xff, (state[1] >> 24) & 0xff, state[2] & 0xff,
      (state[2] >> 8) & 0xff, (state[2] >> 16) & 0xff, (state[2] >> 24) & 0xff,
      state[3] & 0xff, (state[3] >> 8) & 0xff, (state[3] >> 16) & 0xff,
      (state[3] >> 24) & 0xff);
}

void
schro_frame_data_get_codeblock (SchroFrameData * dest, SchroFrameData * src,
    int x, int y, int horiz_codeblocks, int vert_codeblocks)
{
  int xmin = (src->width * x) / horiz_codeblocks;
  int xmax = (src->width * (x + 1)) / horiz_codeblocks;
  int ymin = (src->height * y) / vert_codeblocks;
  int ymax = (src->height * (y + 1)) / vert_codeblocks;

  dest->format = src->format;
  dest->data = SCHRO_FRAME_DATA_GET_PIXEL_S16 (src, xmin, ymin);
  dest->stride = src->stride;
  dest->width = xmax - xmin;
  dest->height = ymax - ymin;
  dest->length = 0;
  dest->h_shift = src->h_shift;
  dest->v_shift = src->v_shift;
}

void
schro_frame_split_fields (SchroFrame * dest1, SchroFrame * dest2,
    SchroFrame * src)
{
  SchroFrame src_tmp;

  SCHRO_ASSERT ((src->height & 1) == 0);

  memcpy (&src_tmp, src, sizeof (src_tmp));

  src_tmp.height = src->height / 2;
  src_tmp.components[0].stride *= 2;
  src_tmp.components[1].stride *= 2;
  src_tmp.components[2].stride *= 2;

  schro_frame_convert (dest1, &src_tmp);

  src_tmp.components[0].data =
      SCHRO_FRAME_DATA_GET_LINE (&src->components[0], 1);
  src_tmp.components[1].data =
      SCHRO_FRAME_DATA_GET_LINE (&src->components[1], 1);
  src_tmp.components[2].data =
      SCHRO_FRAME_DATA_GET_LINE (&src->components[2], 1);

  schro_frame_convert (dest2, &src_tmp);
}

/* upsampled frame */

SchroUpsampledFrame *
schro_upsampled_frame_new (SchroFrame * frame)
{
  SchroUpsampledFrame *df;

  df = schro_malloc0 (sizeof (SchroUpsampledFrame));

  df->frames[0] = frame;

  return df;
}

void
schro_upsampled_frame_free (SchroUpsampledFrame * df)
{
  int i;
  for (i = 0; i < 4; i++) {
    if (df->frames[i]) {
      schro_frame_unref (df->frames[i]);
    }
  }
  schro_free (df);
}

static void
schro_frame_mc_edgeextend_horiz (SchroFrame * frame, SchroFrame * src)
{
  int k;
  int j;

  for (k = 0; k < 3; k++) {
    int width = frame->components[k].width;

    for (j = 0; j < frame->components[k].height; j++) {
      uint8_t *line = SCHRO_FRAME_DATA_GET_LINE (frame->components + k, j);
      uint8_t *src_line = SCHRO_FRAME_DATA_GET_LINE (src->components + k, j);

      memset (line - frame->extension, src_line[0], frame->extension);
      /* A picture of size (w,h) is upconverted to (2*w-1,2*h-1)
       * However, schroedinger's effective upconverted size is (2*w,2*h)
       * Remember to overwrite the last horizontal pel */
      memset (line + width - 1, src_line[width - 1], frame->extension + 1);
    }
  }
}

static void
schro_frame_mc_edgeextend_vert (SchroFrame * frame, SchroFrame * src)
{
  int k;
  int j;

  for (k = 0; k < 3; k++) {
    int height = frame->components[k].height;
    int width = frame->components[k].width;

    for (j = 0; j < frame->extension; j++) {
      orc_memcpy (SCHRO_OFFSET (SCHRO_FRAME_DATA_GET_LINE (frame->components +
                  k, -j - 1), -frame->extension),
          SCHRO_OFFSET (SCHRO_FRAME_DATA_GET_LINE (src->components + k, 0),
              -frame->extension), width + frame->extension * 2);
      orc_memcpy (SCHRO_OFFSET (SCHRO_FRAME_DATA_GET_LINE (frame->components +
                  k, height + j), -frame->extension),
          SCHRO_OFFSET (SCHRO_FRAME_DATA_GET_LINE (src->components + k,
                  height - 1), -frame->extension),
          width + frame->extension * 2);
    }
    /* A picture of size (w,h) is upconverted to (2*w-1,2*h-1)
     * However, schroedinger's effective upconverted size is (2*w,2*h)
     * Copy the src into the bottom line of frame.
     * NB, this assumes that orc_memcpy is safe when src == dest */
    orc_memcpy (SCHRO_OFFSET (SCHRO_FRAME_DATA_GET_LINE (frame->components + k,
                height - 1), -frame->extension),
        SCHRO_OFFSET (SCHRO_FRAME_DATA_GET_LINE (src->components + k,
                height - 1), -frame->extension), width + frame->extension * 2);
  }
}

void
schro_frame_mc_edgeextend (SchroFrame * frame)
{
  schro_frame_mc_edgeextend_horiz (frame, frame);
  schro_frame_mc_edgeextend_vert (frame, frame);
}


void
schro_upsampled_frame_upsample (SchroUpsampledFrame * df)
{
  if (df->frames[1])
    return;

  df->frames[1] = schro_frame_new_and_alloc_extended (df->frames[0]->domain,
      df->frames[0]->format, df->frames[0]->width, df->frames[0]->height,
      df->frames[0]->extension);
  df->frames[2] = schro_frame_new_and_alloc_extended (df->frames[0]->domain,
      df->frames[0]->format, df->frames[0]->width, df->frames[0]->height,
      df->frames[0]->extension);
  df->frames[3] = schro_frame_new_and_alloc_extended (df->frames[0]->domain,
      df->frames[0]->format, df->frames[0]->width, df->frames[0]->height,
      df->frames[0]->extension);

  schro_frame_upsample_vert (df->frames[2], df->frames[0]);
  schro_frame_mc_edgeextend_horiz (df->frames[2], df->frames[2]);
  schro_frame_mc_edgeextend_vert (df->frames[2], df->frames[0]);

  schro_frame_upsample_horiz (df->frames[1], df->frames[0]);
  schro_frame_mc_edgeextend_horiz (df->frames[1], df->frames[0]);
  schro_frame_mc_edgeextend_vert (df->frames[1], df->frames[1]);

  schro_frame_upsample_horiz (df->frames[3], df->frames[2]);
  schro_frame_mc_edgeextend_horiz (df->frames[3], df->frames[2]);
  schro_frame_mc_edgeextend_vert (df->frames[3], df->frames[1]);
}

#ifdef ENABLE_MOTION_REF
int
schro_upsampled_frame_get_pixel_prec0 (SchroUpsampledFrame * upframe, int k,
    int x, int y)
{
  SchroFrameData *comp;
  uint8_t *line;

  comp = upframe->frames[0]->components + k;
  x = CLAMP (x, 0, comp->width - 1);
  y = CLAMP (y, 0, comp->height - 1);

  line = SCHRO_FRAME_DATA_GET_LINE (comp, y);

  return line[x];
}

int
schro_frame_get_data (SchroFrame * frame, SchroFrameData * fd, int k, int x,
    int y)
{
  SchroFrameData *comp = frame->components + k;

  SCHRO_ASSERT (frame && fd && !(0 > x) && !(0 > y));
  /* check whether the required block lies completely outside the frame */
  if (!(frame->width > x) || !(frame->height > y)) {
    return FALSE;
  }
  SCHRO_ASSERT (SCHRO_FRAME_FORMAT_DEPTH (comp->format) ==
      SCHRO_FRAME_FORMAT_DEPTH_U8);

  fd->format = comp->format;
  fd->data = SCHRO_FRAME_DATA_GET_PIXEL_U8 (comp, x, y);
  fd->stride = comp->stride;
  fd->width = comp->width - x;
  fd->height = comp->height - y;
  fd->h_shift = comp->h_shift;
  fd->v_shift = comp->v_shift;

  return TRUE;
}

#ifdef unused
void
schro_upsampled_frame_get_block_prec0 (SchroUpsampledFrame * upframe, int k,
    int x, int y, SchroFrameData * fd)
{
  int i, j;
  uint8_t *data;

  for (j = 0; j < fd->height; j++) {
    data = SCHRO_FRAME_DATA_GET_LINE (fd, j);
    for (i = 0; i < fd->width; i++) {
      data[i] = schro_upsampled_frame_get_pixel_prec0 (upframe, k,
          x + i, y + j);
    }
  }
}
#endif
#endif

#ifdef unused
void
schro_upsampled_frame_get_block_fast_prec0 (SchroUpsampledFrame * upframe,
    int k, int x, int y, SchroFrameData * fd)
{
  SchroFrameData *comp;
  int j;

  comp = upframe->frames[0]->components + k;

  for (j = 0; j < fd->height; j++) {
    uint8_t *dest = SCHRO_FRAME_DATA_GET_LINE (fd, j);
    uint8_t *src = SCHRO_FRAME_DATA_GET_LINE (comp, y + j);
    memcpy (dest, src + x, fd->width);
  }
}
#endif

void
schro_upsampled_frame_get_subdata_prec0 (SchroUpsampledFrame * upframe,
    int component, int x, int y, SchroFrameData * fd)
{
  SchroFrameData *comp = upframe->frames[0]->components + component;

  fd->data = SCHRO_FRAME_DATA_GET_PIXEL_U8 (comp, x, y);
  fd->stride = comp->stride;
}

#ifdef ENABLE_MOTION_REF
int
schro_upsampled_frame_get_pixel_prec1 (SchroUpsampledFrame * upframe, int k,
    int x, int y)
{
  SchroFrameData *comp;
  uint8_t *line;
  int i;

  comp = upframe->frames[0]->components + k;
  x = CLAMP (x, 0, comp->width * 2 - 2);
  y = CLAMP (y, 0, comp->height * 2 - 2);

  i = ((y & 1) << 1) | (x & 1);
  x >>= 1;
  y >>= 1;

  comp = upframe->frames[i]->components + k;
  line = SCHRO_FRAME_DATA_GET_LINE (comp, y);

  return line[x];
}

#ifdef unused
void
schro_upsampled_frame_get_block_prec1 (SchroUpsampledFrame * upframe, int k,
    int x, int y, SchroFrameData * fd)
{
  int i, j;
  uint8_t *data;

  for (j = 0; j < fd->height; j++) {
    data = SCHRO_FRAME_DATA_GET_LINE (fd, j);
    for (i = 0; i < fd->width; i++) {
      data[i] = schro_upsampled_frame_get_pixel_prec1 (upframe, k,
          x + (i << 1), y + (j << 1));
    }
  }
}
#endif
#endif

static void
schro_upsampled_frame_get_block_fast_prec1 (SchroUpsampledFrame * upframe,
    int k, int x, int y, SchroFrameData * fd)
{
  SchroFrameData *comp;
  int i;
  int j;

  i = ((y & 1) << 1) | (x & 1);
  x >>= 1;
  y >>= 1;

  comp = upframe->frames[i]->components + k;
  for (j = 0; j < fd->height; j++) {
    uint8_t *dest = SCHRO_FRAME_DATA_GET_LINE (fd, j);
    uint8_t *src = SCHRO_FRAME_DATA_GET_LINE (comp, y + j);
    orc_memcpy (dest, src + x, fd->width);
  }
}

static void
__schro_upsampled_frame_get_subdata_prec1 (SchroUpsampledFrame * upframe,
    int k, int x, int y, SchroFrameData * fd)
{
  SchroFrameData *comp;
  int i;

  i = ((y & 1) << 1) | (x & 1);
  x >>= 1;
  y >>= 1;

  comp = upframe->frames[i]->components + k;
  fd->data = SCHRO_FRAME_DATA_GET_PIXEL_U8 (comp, x, y);
  fd->stride = comp->stride;
}

void
schro_upsampled_frame_get_subdata_prec1 (SchroUpsampledFrame * upframe,
    int k, int x, int y, SchroFrameData * fd)
{
  __schro_upsampled_frame_get_subdata_prec1 (upframe, k, x, y, fd);
}

#ifdef ENABLE_MOTION_REF
int
schro_upsampled_frame_get_pixel_prec3 (SchroUpsampledFrame * upframe, int k,
    int x, int y)
{
  int hx, hy;
  int rx, ry;
  int w00, w01, w10, w11;
  int value;

  hx = x >> 2;
  hy = y >> 2;

  rx = x & 0x3;
  ry = y & 0x3;

  w00 = (4 - ry) * (4 - rx);
  w01 = (4 - ry) * rx;
  w10 = ry * (4 - rx);
  w11 = ry * rx;

  if (hx >= 0 && hx < 2 * upframe->frames[0]->components[k].width - 2 &&
      hy >= 0 && hy < 2 * upframe->frames[0]->components[k].height - 2) {
    SchroFrameData *comp;
    int p;
    int i;

    i = ((hy & 1) << 1) | (hx & 1);

    comp = upframe->frames[i]->components + k;
    p = *SCHRO_FRAME_DATA_GET_PIXEL_U8 (comp, hx >> 1, hy >> 1);
    value = w00 * p;

    comp = upframe->frames[i ^ 1]->components + k;
    p = *SCHRO_FRAME_DATA_GET_PIXEL_U8 (comp, (hx + 1) >> 1, hy >> 1);
    value += w01 * p;

    comp = upframe->frames[i ^ 2]->components + k;
    p = *SCHRO_FRAME_DATA_GET_PIXEL_U8 (comp, hx >> 1, (hy + 1) >> 1);
    value += w10 * p;

    comp = upframe->frames[i ^ 3]->components + k;
    p = *SCHRO_FRAME_DATA_GET_PIXEL_U8 (comp, (hx + 1) >> 1, (hy + 1) >> 1);
    value += w11 * p;
  } else {
    value = w00 * schro_upsampled_frame_get_pixel_prec1 (upframe, k, hx, hy);
    value +=
        w01 * schro_upsampled_frame_get_pixel_prec1 (upframe, k, hx + 1, hy);
    value +=
        w10 * schro_upsampled_frame_get_pixel_prec1 (upframe, k, hx, hy + 1);
    value +=
        w11 * schro_upsampled_frame_get_pixel_prec1 (upframe, k, hx + 1,
        hy + 1);
  }

  return ROUND_SHIFT (value, 4);
}

#ifdef unused
void
schro_upsampled_frame_get_block_prec3 (SchroUpsampledFrame * upframe, int k,
    int x, int y, SchroFrameData * fd)
{
  int i, j;
  uint8_t *data;

  for (j = 0; j < fd->height; j++) {
    data = SCHRO_FRAME_DATA_GET_LINE (fd, j);
    for (i = 0; i < fd->width; i++) {
      data[i] = schro_upsampled_frame_get_pixel_prec3 (upframe, k,
          x + (i << 3), y + (j << 3));
    }
  }
}
#endif
#endif

static void
schro_upsampled_frame_get_block_fast_prec3 (SchroUpsampledFrame * upframe,
    int k, int x, int y, SchroFrameData * fd)
{
  int hx, hy;
  int rx, ry;
  int w00, w01, w10, w11;
  SchroFrameData fd00;
  SchroFrameData fd01;
  SchroFrameData fd10;
  SchroFrameData fd11;
  int16_t p[6];

  hx = x >> 2;
  hy = y >> 2;

  rx = x & 0x3;
  ry = y & 0x3;

  switch ((ry << 2) | rx) {
    case 0:
      schro_upsampled_frame_get_block_fast_prec1 (upframe, k, hx, hy, fd);
      break;
    case 2:
    case 8:
      __schro_upsampled_frame_get_subdata_prec1 (upframe, k, hx, hy, &fd00);
      if (rx == 0) {
        __schro_upsampled_frame_get_subdata_prec1 (upframe, k, hx, hy + 1,
            &fd10);
      } else {
        __schro_upsampled_frame_get_subdata_prec1 (upframe, k, hx + 1, hy,
            &fd10);
      }

      switch (fd->width) {
        case 8:
          orc_avg2_8xn_u8 (fd->data, fd->stride,
              fd00.data, fd00.stride, fd10.data, fd10.stride, fd->height);
          break;
        case 12:
          orc_avg2_12xn_u8 (fd->data, fd->stride,
              fd00.data, fd00.stride, fd10.data, fd10.stride, fd->height);
          break;
        case 16:
          orc_avg2_16xn_u8 (fd->data, fd->stride,
              fd00.data, fd00.stride, fd10.data, fd10.stride, fd->height);
          break;
        case 24:
          orc_avg2_16xn_u8 (fd->data, fd->stride,
              fd00.data, fd00.stride, fd10.data, fd10.stride, fd->height);
          orc_avg2_8xn_u8 (SCHRO_OFFSET (fd->data, 16), fd->stride,
              SCHRO_OFFSET (fd00.data, 16), fd00.stride,
              SCHRO_OFFSET (fd10.data, 16), fd10.stride, fd->height);
          break;
        case 32:
          orc_avg2_32xn_u8 (fd->data, fd->stride,
              fd00.data, fd00.stride, fd10.data, fd10.stride, fd->height);
          break;
        default:
          orc_avg2_nxm_u8 (fd->data, fd->stride,
              fd00.data, fd00.stride, fd10.data, fd10.stride,
              fd->width, fd->height);
          break;
      }
      break;
    default:
      w00 = (4 - ry) * (4 - rx);
      w01 = (4 - ry) * rx;
      w10 = ry * (4 - rx);
      w11 = ry * rx;

      __schro_upsampled_frame_get_subdata_prec1 (upframe, k, hx, hy, &fd00);
      __schro_upsampled_frame_get_subdata_prec1 (upframe, k, hx + 1, hy, &fd01);
      __schro_upsampled_frame_get_subdata_prec1 (upframe, k, hx, hy + 1, &fd10);
      __schro_upsampled_frame_get_subdata_prec1 (upframe, k, hx + 1, hy + 1,
          &fd11);

      p[0] = w00;
      p[1] = w01;
      p[2] = w10;
      p[3] = w11;
      p[4] = 8;
      p[5] = 4;

      switch (fd->width) {
#if 0
        case 8:
          orc_combine4_8xn_u8 (fd->data, fd->stride,
              fd00.data, fd00.stride,
              fd01.data, fd01.stride,
              fd10.data, fd10.stride,
              fd11.data, fd11.stride, w00, w01, w10, w11, fd->height);
          break;
        case 12:
          orc_combine4_12xn_u8 (fd->data, fd->stride,
              fd00.data, fd00.stride,
              fd01.data, fd01.stride,
              fd10.data, fd10.stride,
              fd11.data, fd11.stride, w00, w01, w10, w11, fd->height);
          break;
        case 16:
          orc_combine4_16xn_u8 (fd->data, fd->stride,
              fd00.data, fd00.stride,
              fd01.data, fd01.stride,
              fd10.data, fd10.stride,
              fd11.data, fd11.stride, w00, w01, w10, w11, fd->height);
          break;
        case 24:
          orc_combine4_24xn_u8 (fd->data, fd->stride,
              fd00.data, fd00.stride,
              fd01.data, fd01.stride,
              fd10.data, fd10.stride,
              fd11.data, fd11.stride, w00, w01, w10, w11, fd->height);
          break;
        case 32:
          orc_combine4_32xn_u8 (fd->data, fd->stride,
              fd00.data, fd00.stride,
              fd01.data, fd01.stride,
              fd10.data, fd10.stride,
              fd11.data, fd11.stride, w00, w01, w10, w11, fd->height);
          break;
#endif
        default:
          orc_combine4_nxm_u8 (fd->data, fd->stride,
              fd00.data, fd00.stride,
              fd01.data, fd01.stride,
              fd10.data, fd10.stride,
              fd11.data, fd11.stride,
              w00, w01, w10, w11, fd->width, fd->height);
          break;
      }
      break;
  }
}

#ifdef ENABLE_MOTION_REF
int
schro_upsampled_frame_get_pixel_precN (SchroUpsampledFrame * upframe, int k,
    int x, int y, int prec)
{
  switch (prec) {
    case 0:
      return schro_upsampled_frame_get_pixel_prec0 (upframe, k, x, y);
    case 1:
      return schro_upsampled_frame_get_pixel_prec1 (upframe, k, x, y);
    case 2:
      return schro_upsampled_frame_get_pixel_prec3 (upframe, k, x << 1, y << 1);
    case 3:
      return schro_upsampled_frame_get_pixel_prec3 (upframe, k, x, y);
    default:
      SCHRO_ASSERT (0);
  }
}

#ifdef unused
void
schro_upsampled_frame_get_block_precN (SchroUpsampledFrame * upframe, int k,
    int x, int y, int prec, SchroFrameData * fd)
{
  switch (prec) {
    case 0:
      schro_upsampled_frame_get_block_prec0 (upframe, k, x, y, fd);
      return;
    case 1:
      schro_upsampled_frame_get_block_prec1 (upframe, k, x, y, fd);
      return;
    case 2:
      schro_upsampled_frame_get_block_prec3 (upframe, k, x << 1, y << 1, fd);
      return;
    case 3:
      schro_upsampled_frame_get_block_prec3 (upframe, k, x, y, fd);
      return;
    default:
      SCHRO_ASSERT (0);
  }
}
#endif
#endif

void
schro_upsampled_frame_get_block_fast_precN (SchroUpsampledFrame * upframe,
    int k, int x, int y, int prec, SchroFrameData * dest, SchroFrameData * fd)
{
  switch (prec) {
    case 0:
      schro_upsampled_frame_get_subdata_prec0 (upframe, k, x, y, dest);
      return;
    case 1:
      schro_upsampled_frame_get_subdata_prec1 (upframe, k, x, y, dest);
      return;
    case 2:
      memcpy (dest, fd, sizeof (SchroFrameData));
      schro_upsampled_frame_get_block_fast_prec3 (upframe, k, x << 1, y << 1,
          dest);
      return;
    case 3:
      memcpy (dest, fd, sizeof (SchroFrameData));
      schro_upsampled_frame_get_block_fast_prec3 (upframe, k, x, y, dest);
      return;
    default:
      SCHRO_ASSERT (0);
  }
}

void
schro_frame_get_subdata (SchroFrame * frame, SchroFrameData * fd,
    int component, int x, int y)
{
  SchroFrameData *comp = frame->components + component;

  SCHRO_ASSERT (SCHRO_FRAME_FORMAT_DEPTH (comp->format) ==
      SCHRO_FRAME_FORMAT_DEPTH_U8);

  fd->format = comp->format;
  fd->data = SCHRO_FRAME_DATA_GET_PIXEL_U8 (comp, x, y);
  fd->stride = comp->stride;
  fd->width = MAX (0, comp->width - x);
  fd->height = MAX (0, comp->height - y);
  fd->h_shift = comp->h_shift;
  fd->v_shift = comp->v_shift;
}

void
schro_frame_get_reference_subdata (SchroFrame * frame, SchroFrameData * fd,
    int component, int x, int y)
{
  SchroFrameData *comp = frame->components + component;
  int extension = frame->extension;

  schro_frame_get_subdata (frame, fd, component, x, y);
  /* modify width and height to account for a reference block
   * that can be completely outside of a frame */
  fd->width = MAX (0, comp->width + extension - x);
  fd->height = MAX (0, comp->height + extension - y);
}
