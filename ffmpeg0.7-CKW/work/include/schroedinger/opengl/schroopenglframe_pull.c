
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schrodebug.h>
#include <schroedinger/opengl/schroopengl.h>
#include <schroedinger/opengl/schroopenglcanvas.h>
#include <schroedinger/opengl/schroopenglframe.h>
#include <liboil/liboil.h>

static void
schro_opengl_canvas_pull_convert (SchroFrameData *dest, SchroOpenGLCanvas *src,
    void *texture_data, int y_offset, int height)
{
  int x, y;
  int width;
  int frame_stride, texture_stride, texture_channels;
  uint8_t *texture_data_u8 = NULL;
  uint16_t *texture_data_u16 = NULL;
  int16_t *texture_data_s16 = NULL;
  float *texture_data_f32 = NULL;
  uint8_t *frame_data_u8 = NULL;
  int16_t *frame_data_s16 = NULL;

  SCHRO_ASSERT (dest != NULL);
  SCHRO_ASSERT (src != NULL);
  SCHRO_ASSERT (dest->format == src->format);
  SCHRO_ASSERT (dest->width == src->width);
  SCHRO_ASSERT (dest->height == src->height);
  SCHRO_ASSERT (texture_data != NULL);

  width = dest->width;
  frame_stride = dest->stride;
  texture_stride = src->pull.stride;
  texture_channels = SCHRO_FRAME_IS_PACKED (dest->format)
      ? 1 : src->texture.channels;

  switch (SCHRO_FRAME_FORMAT_DEPTH (dest->format)) {
    case SCHRO_FRAME_FORMAT_DEPTH_U8:
      frame_data_u8 = SCHRO_FRAME_DATA_GET_LINE (dest, y_offset);

      if (SCHRO_OPENGL_CANVAS_IS_FLAG_SET (PULL_U8_AS_F32)) {
        texture_data_f32 = (float *) texture_data;

        for (y = 0; y < height; ++y) {
          for (x = 0; x < width; ++x) {
            frame_data_u8[x]
                = (uint8_t) (texture_data_f32[x * texture_channels] * 255.0);
          }

          frame_data_u8 = OFFSET (frame_data_u8, frame_stride);
          texture_data_f32 = OFFSET (texture_data_f32, texture_stride);
        }
      } else {
        texture_data_u8 = (uint8_t *) texture_data;

        if (texture_channels > 1) {
          for (y = 0; y < height; ++y) {
            for (x = 0; x < width; ++x) {
              frame_data_u8[x] = texture_data_u8[x * texture_channels];
            }

            frame_data_u8 = OFFSET (frame_data_u8, frame_stride);
            texture_data_u8 = OFFSET (texture_data_u8, texture_stride);
          }
        } else {
          for (y = 0; y < height; ++y) {
            oil_memcpy (frame_data_u8, texture_data_u8, width);

            frame_data_u8 = OFFSET (frame_data_u8, frame_stride);
            texture_data_u8 = OFFSET (texture_data_u8, texture_stride);
          }
        }
      }

      break;
    case SCHRO_FRAME_FORMAT_DEPTH_S16:
      frame_data_s16 = SCHRO_FRAME_DATA_GET_LINE (dest, y_offset);

      if (SCHRO_OPENGL_CANVAS_IS_FLAG_SET (PULL_S16_AS_F32)) {
        texture_data_f32 = (float *) texture_data;

        for (y = 0; y < height; ++y) {
          for (x = 0; x < width; ++x) {
            // FIXME: for some unknown reason I need to scale with 65536.0
            // instead of 65535.0 to get correct S16 value back. I also get
            // correct S16 values with rounding: round (x) := floor (x + 0.5)
            // but thats way to expensive
            frame_data_s16[x]
                = (int16_t) ((int32_t) (texture_data_f32[x * texture_channels]
                * 65536.0) - 32768);
          }

          frame_data_s16 = OFFSET (frame_data_s16, frame_stride);
          texture_data_f32 = OFFSET (texture_data_f32, texture_stride);
        }
      } else if (SCHRO_OPENGL_CANVAS_IS_FLAG_SET (PULL_S16_AS_U16)) {
        texture_data_u16 = (uint16_t *) texture_data;

        for (y = 0; y < height; ++y) {
          for (x = 0; x < width; ++x) {
            frame_data_s16[x]
                = (int16_t) ((int32_t) texture_data_u16[x * texture_channels]
                - 32768);
          }

          frame_data_s16 = OFFSET (frame_data_s16, frame_stride);
          texture_data_u16 = OFFSET (texture_data_u16, texture_stride);
        }
      } else {
        texture_data_s16 = (int16_t *) texture_data;

        if (texture_channels > 1) {
          for (y = 0; y < height; ++y) {
            for (x = 0; x < width; ++x) {
              frame_data_s16[x] = texture_data_s16[x * texture_channels];
            }

            frame_data_s16 = OFFSET (frame_data_s16, frame_stride);
            texture_data_s16 = OFFSET (texture_data_s16, texture_stride);
          }
        } else {
          for (y = 0; y < height; ++y) {
            oil_memcpy (frame_data_s16, texture_data_s16,
                width * sizeof (int16_t));

            frame_data_s16 = OFFSET (frame_data_s16, frame_stride);
            texture_data_s16 = OFFSET (texture_data_s16, texture_stride);
          }
        }
      }
      break;
    default:
      SCHRO_ASSERT (0);
      break;
  }
}

void
schro_opengl_frame_pull (SchroFrame *dest, SchroFrame *src)
{
  int i;
  int components;
  SchroOpenGLCanvas *src_canvas;
  SchroOpenGL *opengl;

  SCHRO_ASSERT (dest != NULL);
  SCHRO_ASSERT (src != NULL);
  SCHRO_ASSERT (!SCHRO_FRAME_IS_OPENGL (dest));
  SCHRO_ASSERT (SCHRO_FRAME_IS_OPENGL (src));
  SCHRO_ASSERT (dest->format == src->format);

  components = SCHRO_FRAME_IS_PACKED (src->format) ? 1 : 3;
  // FIXME: hack to store custom data per frame component
  src_canvas = *((SchroOpenGLCanvas **) src->components[0].data);

  SCHRO_ASSERT (src_canvas != NULL);

  opengl = src_canvas->opengl;

  schro_opengl_lock (opengl);

  for (i = 0; i < components; ++i) {
    // FIXME: hack to store custom data per frame component
    src_canvas = *((SchroOpenGLCanvas **) src->components[i].data);

    SCHRO_ASSERT (src_canvas != NULL);
    SCHRO_ASSERT (src_canvas->opengl == opengl);

    schro_opengl_canvas_pull (dest->components + i, src_canvas);
  }

  schro_opengl_unlock (opengl);
}

void
schro_opengl_canvas_pull (SchroFrameData *dest, SchroOpenGLCanvas *src)
{
  int i;
  int width, height;
  int pixelbuffer_y_offset, pixelbuffer_height;
  void *mapped_data = NULL;
  void *tmp_data = NULL;

  SCHRO_ASSERT (dest->format == src->format);
  SCHRO_ASSERT (dest->width == src->width);
  SCHRO_ASSERT (dest->height == src->height);
  SCHRO_ASSERT (src->texture.handles[0] != 0);
  SCHRO_ASSERT (src->texture.handles[1] != 0);
  SCHRO_ASSERT (src->framebuffers[0] != 0);
  SCHRO_ASSERT (src->framebuffers[1] != 0);

  width = dest->width;
  height = dest->height;

  schro_opengl_lock (src->opengl);

  if (SCHRO_OPENGL_CANVAS_IS_FLAG_SET (PULL_PIXELBUFFER)) {
    glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, src->framebuffers[0]);

    pixelbuffer_y_offset = 0;

    for (i = 0; i < SCHRO_OPENGL_TRANSFER_PIXELBUFFERS; ++i) {
      pixelbuffer_height = src->pull.heights[i];

      glBindBufferARB (GL_PIXEL_PACK_BUFFER_ARB, src->pull.pixelbuffers[i]);
      glReadPixels (0, pixelbuffer_y_offset, width, pixelbuffer_height,
          src->texture.pixel_format, src->pull.type, NULL);

      pixelbuffer_y_offset += pixelbuffer_height;

      SCHRO_OPENGL_CHECK_ERROR
    }

    pixelbuffer_y_offset = 0;

    for (i = 0; i < SCHRO_OPENGL_TRANSFER_PIXELBUFFERS; ++i) {
      pixelbuffer_height = src->pull.heights[i];

      glBindBufferARB (GL_PIXEL_PACK_BUFFER_ARB, src->pull.pixelbuffers[i]);

      mapped_data = glMapBufferARB (GL_PIXEL_PACK_BUFFER_ARB, GL_READ_ONLY_ARB);

      schro_opengl_canvas_pull_convert (dest, src, mapped_data,
          pixelbuffer_y_offset, pixelbuffer_height);

      glUnmapBufferARB (GL_PIXEL_PACK_BUFFER_ARB);

      pixelbuffer_y_offset += pixelbuffer_height;

      SCHRO_OPENGL_CHECK_ERROR
    }

    glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, 0);
  } else {
    tmp_data = schro_opengl_get_tmp (src->opengl, src->pull.stride * height);

    glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, src->framebuffers[0]);
    glReadPixels (0, 0, width, height, src->texture.pixel_format,
        src->pull.type, tmp_data);
    glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, 0);

    schro_opengl_canvas_pull_convert (dest, src, tmp_data, 0, height);
  }

  schro_opengl_unlock (src->opengl);
}

