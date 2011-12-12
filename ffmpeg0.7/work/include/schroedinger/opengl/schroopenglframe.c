
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schrodebug.h>
#include <schroedinger/opengl/schroopengl.h>
#include <schroedinger/opengl/schroopenglcanvas.h>
#include <schroedinger/opengl/schroopenglframe.h>
#include <schroedinger/opengl/schroopenglshader.h>
#include <schroedinger/opengl/schroopenglwavelet.h>
#include <stdio.h>

void
schro_opengl_frame_setup (SchroOpenGL *opengl, SchroFrame *frame)
{
  int i;
  int components;
  int width, height;
  SchroFrameFormat format;
  SchroOpenGLCanvasPool *canvas_pool;
  SchroOpenGLCanvas *canvas;

  SCHRO_ASSERT (frame != NULL);
  SCHRO_ASSERT (SCHRO_FRAME_IS_OPENGL (frame));

  components = SCHRO_FRAME_IS_PACKED (frame->format) ? 1 : 3;

  schro_opengl_lock (opengl);

  canvas_pool = schro_opengl_get_canvas_pool (opengl);

  for (i = 0; i < components; ++i) {
    format = frame->components[i].format;
    width = frame->components[i].width;
    height = frame->components[i].height;

    SCHRO_ASSERT (frame->format == format);

    canvas = schro_opengl_canvas_pool_pull_or_new (canvas_pool, opengl, format,
        width, height);

    // FIXME: hack to store custom data per frame component
    *((SchroOpenGLCanvas **) frame->components[i].data) = canvas;
  }

  schro_opengl_unlock (opengl);
}

void
schro_opengl_frame_cleanup (SchroFrame *frame)
{
  int i;
  int components;
  SchroOpenGL *opengl;
  SchroOpenGLCanvasPool *canvas_pool;
  SchroOpenGLCanvas *canvas;

  SCHRO_ASSERT (frame != NULL);
  SCHRO_ASSERT (SCHRO_FRAME_IS_OPENGL (frame));

  components = SCHRO_FRAME_IS_PACKED (frame->format) ? 1 : 3;
  // FIXME: hack to store custom data per frame component
  canvas = *((SchroOpenGLCanvas **) frame->components[0].data);

  SCHRO_ASSERT (canvas != NULL);

  opengl = canvas->opengl;

  schro_opengl_lock (opengl);

  canvas_pool = schro_opengl_get_canvas_pool (opengl);

  for (i = 0; i < components; ++i) {
    // FIXME: hack to store custom data per frame component
    canvas = *((SchroOpenGLCanvas **) frame->components[i].data);

    SCHRO_ASSERT (canvas != NULL);
    SCHRO_ASSERT (canvas->opengl == opengl);

    schro_opengl_canvas_pool_push_or_free (canvas_pool, canvas);
  }

  schro_opengl_unlock (opengl);
}

SchroFrame *
schro_opengl_frame_new (SchroOpenGL *opengl,
    SchroMemoryDomain *opengl_domain, SchroFrameFormat format, int width,
    int height)
{
  SchroFrame *opengl_frame;

  SCHRO_ASSERT (opengl_domain->flags & SCHRO_MEMORY_DOMAIN_OPENGL);

  opengl_frame = schro_frame_new_and_alloc (opengl_domain, format, width,
      height);

  schro_opengl_frame_setup (opengl, opengl_frame);

  return opengl_frame;
}

SchroFrame *
schro_opengl_frame_clone (SchroFrame *opengl_frame)
{
  SchroOpenGLCanvas *canvas;

  SCHRO_ASSERT (opengl_frame != NULL);
  SCHRO_ASSERT (SCHRO_FRAME_IS_OPENGL (opengl_frame));

  // FIXME: hack to store custom data per frame component
  canvas = *((SchroOpenGLCanvas **) opengl_frame->components[0].data);

  SCHRO_ASSERT (canvas != NULL);

  return schro_opengl_frame_new (canvas->opengl, opengl_frame->domain,
      opengl_frame->format, opengl_frame->width, opengl_frame->height);
}

SchroFrame *
schro_opengl_frame_clone_and_push (SchroOpenGL *opengl,
    SchroMemoryDomain *opengl_domain, SchroFrame *cpu_frame)
{
  SchroFrame *opengl_frame;

  SCHRO_ASSERT (opengl_domain->flags & SCHRO_MEMORY_DOMAIN_OPENGL);
  SCHRO_ASSERT (!SCHRO_FRAME_IS_OPENGL (cpu_frame));

  opengl_frame = schro_frame_clone (opengl_domain, cpu_frame);

  schro_opengl_frame_setup (opengl, opengl_frame);
  schro_opengl_frame_push (opengl_frame, cpu_frame);

  return opengl_frame;
}

void
schro_opengl_frame_inverse_iwt_transform (SchroFrame *frame,
    SchroParams *params)
{
  int i;
  int width, height;
  int level;
  SchroOpenGL *opengl;
  SchroOpenGLCanvas *canvas;

  // FIXME: hack to store custom data per frame component
  canvas = *((SchroOpenGLCanvas **) frame->components[0].data);

  SCHRO_ASSERT (canvas != NULL);

  opengl = canvas->opengl;

  schro_opengl_lock (opengl);

  for (i = 0; i < 3; ++i) {
    // FIXME: hack to store custom data per frame component
    canvas = *((SchroOpenGLCanvas **) frame->components[i].data);

    SCHRO_ASSERT (canvas != NULL);
    SCHRO_ASSERT (canvas->opengl == opengl);

    if (i == 0) {
      width = params->iwt_luma_width;
      height = params->iwt_luma_height;
    } else {
      width = params->iwt_chroma_width;
      height = params->iwt_chroma_height;
    }

    for (level = params->transform_depth - 1; level >= 0; --level) {
      SchroFrameData frame_data;

      frame_data.format = frame->format;
      frame_data.data = frame->components[i].data;
      frame_data.width = width >> level;
      frame_data.height = height >> level;
      frame_data.stride = frame->components[i].stride << level;

      schro_opengl_wavelet_inverse_transform (&frame_data,
          params->wavelet_filter_index);
    }
  }

  schro_opengl_unlock (opengl);
}

static void
schro_opengl_upsampled_frame_render_quad (SchroOpenGLShader *shader, int x,
    int y, int quad_width, int quad_height, int total_width, int total_height)
{
  int x_inverse, y_inverse;
  int four_x = 0, four_y = 0, three_x = 0, three_y = 0, two_x = 0, two_y = 0,
      one_x = 0, one_y = 0;

  x_inverse = total_width - x - quad_width;
  y_inverse = total_height - y - quad_height;

  if (quad_width == total_width && quad_height < total_height) {
    four_y = 4;
    three_y = 3;
    two_y = 2;
    one_y = 1;
  } else if (quad_width < total_width && quad_height == total_height) {
    four_x = 4;
    three_x = 3;
    two_x = 2;
    one_x = 1;
  } else {
    SCHRO_ERROR ("invalid quad to total relation");
    SCHRO_ASSERT (0);
  }

  SCHRO_ASSERT (x_inverse >= 0);
  SCHRO_ASSERT (y_inverse >= 0);

  #define UNIFORM(_number, _operation, __x, __y) \
      do { \
        if (shader->_number##_##_operation != -1) { \
          glUniform2fARB (shader->_number##_##_operation, \
              __x < _number##_x ? __x : _number##_x, \
              __y < _number##_y ? __y : _number##_y); \
        } \
      } while (0)

  UNIFORM (four, decrease, x, y);
  UNIFORM (three, decrease, x, y);
  UNIFORM (two, decrease, x, y);
  UNIFORM (one, decrease, x, y);
  UNIFORM (one, increase, x_inverse, y_inverse);
  UNIFORM (two, increase, x_inverse, y_inverse);
  UNIFORM (three, increase, x_inverse, y_inverse);
  UNIFORM (four, increase, x_inverse, y_inverse);

  #undef UNIFORM

  schro_opengl_render_quad (x, y, quad_width, quad_height);
}

void
schro_opengl_upsampled_frame_upsample (SchroUpsampledFrame *upsampled_frame)
{
  int i;
  int width, height;
  SchroOpenGLCanvas *canvases[4];
  SchroOpenGL *opengl;
  SchroOpenGLShader *shader = NULL;

  SCHRO_ASSERT (upsampled_frame->frames[0] != NULL);
  SCHRO_ASSERT (upsampled_frame->frames[1] == NULL);
  SCHRO_ASSERT (upsampled_frame->frames[2] == NULL);
  SCHRO_ASSERT (upsampled_frame->frames[3] == NULL);
  SCHRO_ASSERT (SCHRO_FRAME_IS_OPENGL (upsampled_frame->frames[0]));
  SCHRO_ASSERT (!SCHRO_FRAME_IS_PACKED (upsampled_frame->frames[0]->format));

  // FIXME: hack to store custom data per frame component
  canvases[0] = *((SchroOpenGLCanvas **) upsampled_frame->frames[0]->components[0].data);

  SCHRO_ASSERT (canvases[0] != NULL);

  opengl = canvases[0]->opengl;

  schro_opengl_lock (opengl);

  upsampled_frame->frames[1] = schro_opengl_frame_clone (upsampled_frame->frames[0]);
  upsampled_frame->frames[2] = schro_opengl_frame_clone (upsampled_frame->frames[0]);
  upsampled_frame->frames[3] = schro_opengl_frame_clone (upsampled_frame->frames[0]);

  shader = schro_opengl_shader_get (opengl, SCHRO_OPENGL_SHADER_UPSAMPLE_U8);

  SCHRO_ASSERT (shader != NULL);

  glUseProgramObjectARB (shader->program);
  glUniform1iARB (shader->textures[0], 0);

  SCHRO_OPENGL_CHECK_ERROR

  for (i = 0; i < 3; ++i) {
    // FIXME: hack to store custom data per frame component
    canvases[0] = *((SchroOpenGLCanvas **) upsampled_frame->frames[0]->components[i].data);
    canvases[1] = *((SchroOpenGLCanvas **) upsampled_frame->frames[1]->components[i].data);
    canvases[2] = *((SchroOpenGLCanvas **) upsampled_frame->frames[2]->components[i].data);
    canvases[3] = *((SchroOpenGLCanvas **) upsampled_frame->frames[3]->components[i].data);

    SCHRO_ASSERT (canvases[0] != NULL);
    SCHRO_ASSERT (canvases[1] != NULL);
    SCHRO_ASSERT (canvases[2] != NULL);
    SCHRO_ASSERT (canvases[3] != NULL);
    SCHRO_ASSERT (canvases[0]->opengl == opengl);
    SCHRO_ASSERT (canvases[1]->opengl == opengl);
    SCHRO_ASSERT (canvases[2]->opengl == opengl);
    SCHRO_ASSERT (canvases[3]->opengl == opengl);

    width = upsampled_frame->frames[0]->components[i].width;
    height = upsampled_frame->frames[0]->components[i].height;

    SCHRO_ASSERT (width >= 2);
    SCHRO_ASSERT (height >= 2);
    SCHRO_ASSERT (width % 2 == 0);
    SCHRO_ASSERT (height % 2 == 0);

    schro_opengl_setup_viewport (width, height);

    /* horizontal filter 0 -> 1 */
    glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, canvases[1]->framebuffers[0]);
    glBindTexture (GL_TEXTURE_RECTANGLE_ARB, canvases[0]->texture.handles[0]);

    SCHRO_OPENGL_CHECK_ERROR

    #define RENDER_QUAD_HORIZONTAL(_x, _quad_width) \
        schro_opengl_upsampled_frame_render_quad (shader, _x, 0,  _quad_width,\
            height, width, height)

    RENDER_QUAD_HORIZONTAL (0, 1);

    if (width > 2) {
      RENDER_QUAD_HORIZONTAL (1, 1);

      if (width > 4) {
        RENDER_QUAD_HORIZONTAL (2, 1);

        if (width > 6) {
          RENDER_QUAD_HORIZONTAL (3, 1);

           if (width > 8) {
             RENDER_QUAD_HORIZONTAL (4, width - 8);
           }

           RENDER_QUAD_HORIZONTAL (width - 4, 1);
        }

        RENDER_QUAD_HORIZONTAL (width - 3, 1);
      }

      RENDER_QUAD_HORIZONTAL (width - 2, 1);
    }

    RENDER_QUAD_HORIZONTAL (width - 1, 1);

    #undef RENDER_QUAD_HORIZONTAL

    /* vertical filter 0 -> 2 */
    glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, canvases[2]->framebuffers[0]);
    glBindTexture (GL_TEXTURE_RECTANGLE_ARB, canvases[0]->texture.handles[0]);

    SCHRO_OPENGL_CHECK_ERROR

    #define RENDER_QUAD_VERTICAL(_y, _quad_height) \
        schro_opengl_upsampled_frame_render_quad (shader, 0, _y,  width,\
            _quad_height, width, height)

    RENDER_QUAD_VERTICAL (0, 1);

    if (height > 2) {
      RENDER_QUAD_VERTICAL (1, 1);

      if (height > 4) {
        RENDER_QUAD_VERTICAL (2, 1);

        if (height > 6) {
          RENDER_QUAD_VERTICAL (3, 1);

           if (height > 8) {
             RENDER_QUAD_VERTICAL (4, height - 8);
           }

           RENDER_QUAD_VERTICAL (height - 4, 1);
        }

        RENDER_QUAD_VERTICAL (height - 3, 1);
      }

      RENDER_QUAD_VERTICAL (height - 2, 1);
    }

    RENDER_QUAD_VERTICAL (height - 1, 1);

    #undef RENDER_QUAD_VERTICAL

    /* horizontal filter 2 -> 3 */
    glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, canvases[3]->framebuffers[0]);
    glBindTexture (GL_TEXTURE_RECTANGLE_ARB, canvases[2]->texture.handles[0]);

    SCHRO_OPENGL_CHECK_ERROR

    #define RENDER_QUAD_HORIZONTAL(_x, _quad_width) \
        schro_opengl_upsampled_frame_render_quad (shader, _x, 0,  _quad_width,\
            height, width, height)

    RENDER_QUAD_HORIZONTAL (0, 1);

    if (width > 2) {
      RENDER_QUAD_HORIZONTAL (1, 1);

      if (width > 4) {
        RENDER_QUAD_HORIZONTAL (2, 1);

        if (width > 6) {
          RENDER_QUAD_HORIZONTAL (3, 1);

           if (width > 8) {
             RENDER_QUAD_HORIZONTAL (4, width - 8);
           }

           RENDER_QUAD_HORIZONTAL (width - 4, 1);
        }

        RENDER_QUAD_HORIZONTAL (width - 3, 1);
      }

      RENDER_QUAD_HORIZONTAL (width - 2, 1);
    }

    RENDER_QUAD_HORIZONTAL (width - 1, 1);

    #undef RENDER_QUAD_HORIZONTAL
  }

  glUseProgramObjectARB (0);
#if SCHRO_OPENGL_UNBIND_TEXTURES
  glBindTexture (GL_TEXTURE_RECTANGLE_ARB, 0);
#endif
  glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, 0);

  schro_opengl_unlock (opengl);
}

void
schro_frame_print (SchroFrame *frame, const char* name)
{
  printf ("schro_frame_print: %s\n", name);

  switch (SCHRO_FRAME_FORMAT_DEPTH (frame->format)) {
    case SCHRO_FRAME_FORMAT_DEPTH_U8:
      printf ("  depth:  U8\n");
      break;
    case SCHRO_FRAME_FORMAT_DEPTH_S16:
      printf ("  depth:  S16\n");
      break;
    case SCHRO_FRAME_FORMAT_DEPTH_S32:
      printf ("  depth:  S32\n");
      break;
    default:
      printf ("  depth:  unknown\n");
      break;
  }

  printf ("  packed: %s\n", SCHRO_FRAME_IS_PACKED (frame->format) ? "yes": "no");
  printf ("  width:  %i\n", frame->width);
  printf ("  height: %i\n", frame->height);
  printf ("  opengl: %s\n", SCHRO_FRAME_IS_OPENGL (frame) ? "yes": "no");
}

