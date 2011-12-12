
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schrodebug.h>
#include <schroedinger/opengl/schroopengl.h>
#include <schroedinger/opengl/schroopenglcanvas.h>
#include <schroedinger/opengl/schroopenglframe.h>
#include <schroedinger/opengl/schroopenglshader.h>

void
schro_opengl_wavelet_transform (SchroFrameData *frame_data, int filter)
{
  SCHRO_ERROR ("unimplemented");
  SCHRO_ASSERT (0);
}

void
schro_opengl_wavelet_vertical_deinterleave (SchroFrameData *frame_data)
{
  int width, height;
  int framebuffer_index, texture_index;
  SchroOpenGLCanvas *canvas = NULL;
  SchroOpenGL *opengl = NULL;
  SchroOpenGLShader *shader_copy = NULL;
  SchroOpenGLShader *shader_vertical_deinterleave_l = NULL;
  SchroOpenGLShader *shader_vertical_deinterleave_h = NULL;

  SCHRO_ASSERT (SCHRO_FRAME_FORMAT_DEPTH (frame_data->format)
      == SCHRO_FRAME_FORMAT_DEPTH_S16);
  SCHRO_ASSERT (frame_data->width % 2 == 0);
  SCHRO_ASSERT (frame_data->height % 2 == 0);

  width = frame_data->width;
  height = frame_data->height;
  // FIXME: hack to store custom data per frame component
  canvas = *((SchroOpenGLCanvas **) frame_data->data);

  SCHRO_ASSERT (canvas != NULL);

  opengl = canvas->opengl;

  schro_opengl_lock (opengl);

  shader_copy = schro_opengl_shader_get (opengl,
      SCHRO_OPENGL_SHADER_COPY_S16);
  shader_vertical_deinterleave_l = schro_opengl_shader_get (opengl,
      SCHRO_OPENGL_SHADER_IIWT_S16_VERTICAL_DEINTERLEAVE_L);
  shader_vertical_deinterleave_h = schro_opengl_shader_get (opengl,
      SCHRO_OPENGL_SHADER_IIWT_S16_VERTICAL_DEINTERLEAVE_H);

  SCHRO_ASSERT (shader_copy != NULL);
  SCHRO_ASSERT (shader_vertical_deinterleave_l != NULL);
  SCHRO_ASSERT (shader_vertical_deinterleave_h != NULL);

  schro_opengl_setup_viewport (width, height);

  SCHRO_OPENGL_CHECK_ERROR

  #define SWITCH_FRAMEBUFFER_AND_TEXTURE_INDICES \
      framebuffer_index = 1 - framebuffer_index; \
      texture_index = 1 - texture_index; \
      SCHRO_ASSERT (framebuffer_index != texture_index);

  #define BIND_FRAMEBUFFER_AND_TEXTURE \
      glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, \
          canvas->framebuffers[framebuffer_index]); \
      glBindTexture (GL_TEXTURE_RECTANGLE_ARB, \
          canvas->texture.handles[texture_index]); \
      SCHRO_OPENGL_CHECK_ERROR

  framebuffer_index = 1;
  texture_index = 0;

  /* pass 1: vertical deinterleave */
  BIND_FRAMEBUFFER_AND_TEXTURE

  glUseProgramObjectARB (shader_vertical_deinterleave_l->program);
  glUniform1iARB (shader_vertical_deinterleave_l->textures[0], 0);

  schro_opengl_render_quad (0, 0, width, height / 2);

  glUseProgramObjectARB (shader_vertical_deinterleave_h->program);
  glUniform1iARB (shader_vertical_deinterleave_h->textures[0], 0);
  glUniform2fARB (shader_vertical_deinterleave_h->offset, 0, height / 2);

  schro_opengl_render_quad (0, height / 2, width, height / 2);

  SCHRO_OPENGL_CHECK_ERROR

  glFlush ();

  /* pass 2: transfer data from secondary to primary framebuffer */
  SWITCH_FRAMEBUFFER_AND_TEXTURE_INDICES
  BIND_FRAMEBUFFER_AND_TEXTURE

  glUseProgramObjectARB (shader_copy->program);
  glUniform1iARB (shader_copy->textures[0], 0);

  schro_opengl_render_quad (0, 0, width, height);

  SCHRO_OPENGL_CHECK_ERROR

  glFlush ();

  #undef SWITCH_FRAMEBUFFER_AND_TEXTURE_INDICES
  #undef BIND_FRAMEBUFFER_AND_TEXTURE

#if SCHRO_OPENGL_UNBIND_TEXTURES
  glBindTexture (GL_TEXTURE_RECTANGLE_ARB, 0);
#endif
  glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, 0);

  schro_opengl_unlock (opengl);
}

static void
schro_opengl_wavelet_render_quad (SchroOpenGLShader *shader, int x, int y,
    int quad_width, int quad_height, int total_width, int total_height)
{
  int x_inverse, y_inverse;
  int two_x = 0, two_y = 0, one_x = 0, one_y = 0;

  x_inverse = total_width - x - quad_width;
  y_inverse = total_height - y - quad_height;

  if (quad_width == total_width && quad_height < total_height) {
    two_y = 2;
    one_y = 1;
  } else if (quad_width < total_width && quad_height == total_height) {
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

  UNIFORM (two, decrease, x, y);
  UNIFORM (one, decrease, x, y);
  UNIFORM (one, increase, x_inverse, y_inverse);
  UNIFORM (two, increase, x_inverse, y_inverse);

  #undef UNIFORM

  schro_opengl_render_quad (x, y, quad_width, quad_height);
}

void
schro_opengl_wavelet_inverse_transform (SchroFrameData *frame_data,
    int filter)
{
  int width, height, subband_width, subband_height;
  int framebuffer_index, texture_index;
  int filter_shift = FALSE;
  SchroOpenGLCanvas *canvas = NULL;
  SchroOpenGL *opengl = NULL;
  SchroOpenGLShader *shader_copy = NULL;
  SchroOpenGLShader *shader_filter_lp = NULL;
  SchroOpenGLShader *shader_filter_hp = NULL;
  SchroOpenGLShader *shader_vertical_interleave = NULL;
  SchroOpenGLShader *shader_horizontal_interleave = NULL;
  SchroOpenGLShader *shader_filter_shift = NULL;

  SCHRO_ASSERT (SCHRO_FRAME_FORMAT_DEPTH (frame_data->format)
      == SCHRO_FRAME_FORMAT_DEPTH_S16);
  SCHRO_ASSERT (frame_data->width >= 2);
  SCHRO_ASSERT (frame_data->height >= 2);
  SCHRO_ASSERT (frame_data->width % 2 == 0);
  SCHRO_ASSERT (frame_data->height % 2 == 0);

  width = frame_data->width;
  height = frame_data->height;
  subband_width = width / 2;
  subband_height = height / 2;
  // FIXME: hack to store custom data per frame component
  canvas = *((SchroOpenGLCanvas **) frame_data->data);

  SCHRO_ASSERT (canvas != NULL);

  opengl = canvas->opengl;

  schro_opengl_lock (opengl);

  shader_copy = schro_opengl_shader_get (opengl,
      SCHRO_OPENGL_SHADER_COPY_S16);

  SCHRO_ASSERT (shader_copy != NULL);

  switch (filter) {
    case SCHRO_WAVELET_DESLAURIES_DUBUC_9_7:
      shader_filter_lp = schro_opengl_shader_get (opengl,
          SCHRO_OPENGL_SHADER_IIWT_S16_FILTER_DESLAURIERS_DUBUC_9_7_Lp);
      shader_filter_hp = schro_opengl_shader_get (opengl,
          SCHRO_OPENGL_SHADER_IIWT_S16_FILTER_DESLAURIERS_DUBUC_9_7_Hp);

      filter_shift = TRUE;
      break;
    case SCHRO_WAVELET_LE_GALL_5_3:
      shader_filter_lp = schro_opengl_shader_get (opengl,
          SCHRO_OPENGL_SHADER_IIWT_S16_FILTER_LE_GALL_5_3_Lp);
      shader_filter_hp = schro_opengl_shader_get (opengl,
          SCHRO_OPENGL_SHADER_IIWT_S16_FILTER_LE_GALL_5_3_Hp);

      filter_shift = TRUE;
      break;
    case SCHRO_WAVELET_DESLAURIES_DUBUC_13_7:
      shader_filter_lp = schro_opengl_shader_get (opengl,
          SCHRO_OPENGL_SHADER_IIWT_S16_FILTER_DESLAURIERS_DUBUC_13_7_Lp);
      shader_filter_hp = schro_opengl_shader_get (opengl,
          SCHRO_OPENGL_SHADER_IIWT_S16_FILTER_DESLAURIERS_DUBUC_13_7_Hp);

      filter_shift = TRUE;
      break;
    case SCHRO_WAVELET_HAAR_0:
      shader_filter_lp = schro_opengl_shader_get (opengl,
          SCHRO_OPENGL_SHADER_IIWT_S16_FILTER_HAAR_Lp);
      shader_filter_hp = schro_opengl_shader_get (opengl,
          SCHRO_OPENGL_SHADER_IIWT_S16_FILTER_HAAR_Hp);

      filter_shift = FALSE;
      break;
    case SCHRO_WAVELET_HAAR_1:
      shader_filter_lp = schro_opengl_shader_get (opengl,
          SCHRO_OPENGL_SHADER_IIWT_S16_FILTER_HAAR_Lp);
      shader_filter_hp = schro_opengl_shader_get (opengl,
          SCHRO_OPENGL_SHADER_IIWT_S16_FILTER_HAAR_Hp);

      filter_shift = TRUE;
      break;
    case SCHRO_WAVELET_FIDELITY:
      SCHRO_ERROR ("fidelity filter is not implemented yet");
      SCHRO_ASSERT (0);

      filter_shift = FALSE;
      break;
    case SCHRO_WAVELET_DAUBECHIES_9_7:
      SCHRO_ERROR ("daubechies 9,7 filter is not implemented yet");
      SCHRO_ASSERT (0);

      filter_shift = TRUE;
      break;
    default:
      SCHRO_ERROR ("unknown filter %i", filter);
      SCHRO_ASSERT (0);
      break;
  }

  SCHRO_ASSERT (shader_filter_lp != NULL);
  SCHRO_ASSERT (shader_filter_hp != NULL);

  shader_vertical_interleave = schro_opengl_shader_get (opengl,
      SCHRO_OPENGL_SHADER_IIWT_S16_VERTICAL_INTERLEAVE);
  shader_horizontal_interleave = schro_opengl_shader_get (opengl,
      SCHRO_OPENGL_SHADER_IIWT_S16_HORIZONTAL_INTERLEAVE);

  SCHRO_ASSERT (shader_vertical_interleave != NULL);
  SCHRO_ASSERT (shader_horizontal_interleave != NULL);

  if (filter_shift) {
    shader_filter_shift = schro_opengl_shader_get (opengl,
        SCHRO_OPENGL_SHADER_IIWT_S16_FILTER_SHIFT);

    SCHRO_ASSERT (shader_filter_shift);
  } else {
    shader_filter_shift = NULL;
  }

  schro_opengl_setup_viewport (width, height);

  SCHRO_OPENGL_CHECK_ERROR

  #define SWITCH_FRAMEBUFFER_AND_TEXTURE_INDICES \
      framebuffer_index = 1 - framebuffer_index; \
      texture_index = 1 - texture_index; \
      SCHRO_ASSERT (framebuffer_index != texture_index);

  #define BIND_FRAMEBUFFER_AND_TEXTURE \
      glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, \
          canvas->framebuffers[framebuffer_index]); \
      glBindTexture (GL_TEXTURE_RECTANGLE_ARB, \
          canvas->texture.handles[texture_index]); \
      SCHRO_OPENGL_CHECK_ERROR

  framebuffer_index = 1;
  texture_index = 0;

  /* pass 1: vertical filtering => XL + f(XH) = XL' */
  BIND_FRAMEBUFFER_AND_TEXTURE

  glUseProgramObjectARB (shader_filter_lp->program);
  glUniform1iARB (shader_filter_lp->textures[0], 0);
  glUniform2fARB (shader_filter_lp->offset, 0, subband_height);

  #define RENDER_QUAD_VERTICAL_Lp(_y, _quad_height) \
      schro_opengl_wavelet_render_quad (shader_filter_lp, 0, _y, width, \
          _quad_height, width, height)

  RENDER_QUAD_VERTICAL_Lp (0, 1);

  if (subband_height > 2) {
    RENDER_QUAD_VERTICAL_Lp (1, 1);

    if (subband_height > 4) {
      RENDER_QUAD_VERTICAL_Lp (2, subband_height - 4);
    }

    RENDER_QUAD_VERTICAL_Lp (subband_height - 2, 1);
  }

  RENDER_QUAD_VERTICAL_Lp (subband_height - 1, 1);

  #undef RENDER_QUAD_VERTICAL_Lp

  /* copy XH */
  glUseProgramObjectARB (shader_copy->program);
  glUniform1iARB (shader_copy->textures[0], 0);

  schro_opengl_render_quad (0, subband_height, width, subband_height);

  SCHRO_OPENGL_CHECK_ERROR

  glFlush ();

  /* pass 2: vertical filtering => f(XL') + XH = XH' */
  SWITCH_FRAMEBUFFER_AND_TEXTURE_INDICES
  BIND_FRAMEBUFFER_AND_TEXTURE

  glUseProgramObjectARB (shader_filter_hp->program);
  glUniform1iARB (shader_filter_hp->textures[0], 0);
  glUniform2fARB (shader_filter_hp->offset, 0, subband_height);

  #define RENDER_QUAD_VERTICAL_Hp(_y_offset, _quad_height) \
      schro_opengl_wavelet_render_quad (shader_filter_hp, 0, \
          subband_height + (_y_offset), width, _quad_height, width, height)

  RENDER_QUAD_VERTICAL_Hp (0, 1);

  if (subband_height > 2) {
    RENDER_QUAD_VERTICAL_Hp (1, 1);

    if (subband_height > 4) {
      RENDER_QUAD_VERTICAL_Hp (2, subband_height - 4);
    }

    RENDER_QUAD_VERTICAL_Hp (subband_height - 2, 1);
  }

  RENDER_QUAD_VERTICAL_Hp (subband_height - 1, 1);

  #undef RENDER_QUAD_VERTICAL_Hp

  /* copy XL' */
  glUseProgramObjectARB (shader_copy->program);
  glUniform1iARB (shader_copy->textures[0], 0);

  schro_opengl_render_quad (0, 0, width, subband_height);

  SCHRO_OPENGL_CHECK_ERROR

  glFlush ();

  /* pass 3: vertical interleave => i(LL', LH') = L, i(HL', HH') = H */
  SWITCH_FRAMEBUFFER_AND_TEXTURE_INDICES
  BIND_FRAMEBUFFER_AND_TEXTURE

  glUseProgramObjectARB (shader_vertical_interleave->program);
  glUniform1iARB (shader_vertical_interleave->textures[0], 0);
  glUniform2fARB (shader_vertical_interleave->offset, 0, subband_height);

  schro_opengl_render_quad (0, 0, width, height);

  SCHRO_OPENGL_CHECK_ERROR

  glFlush ();

  /* pass 4: horizontal filtering => L + f(H) = L' */
  SWITCH_FRAMEBUFFER_AND_TEXTURE_INDICES
  BIND_FRAMEBUFFER_AND_TEXTURE

  glUseProgramObjectARB (shader_filter_lp->program);
  glUniform1iARB (shader_filter_lp->textures[0], 0);
  glUniform2fARB (shader_filter_lp->offset, subband_width, 0);

  #define RENDER_QUAD_HORIZONTAL_Lp(_x, _quad_width) \
      schro_opengl_wavelet_render_quad (shader_filter_lp, _x, 0, _quad_width, \
          height, width, height)

  RENDER_QUAD_HORIZONTAL_Lp (0, 1);

  if (subband_width > 2) {
    RENDER_QUAD_HORIZONTAL_Lp (1, 1);

    if (subband_width > 4) {
      RENDER_QUAD_HORIZONTAL_Lp (2, subband_width - 4);
    }

    RENDER_QUAD_HORIZONTAL_Lp (subband_width - 2, 1);
  }

  RENDER_QUAD_HORIZONTAL_Lp (subband_width - 1, 1);

  #undef RENDER_QUAD_HORIZONTAL_Lp

  /* copy H */
  glUseProgramObjectARB (shader_copy->program);
  glUniform1iARB (shader_copy->textures[0], 0);

  schro_opengl_render_quad (subband_width, 0, subband_width, height);

  SCHRO_OPENGL_CHECK_ERROR

  glFlush ();

  /* pass 5: horizontal filtering => f(L') + H = H' */
  SWITCH_FRAMEBUFFER_AND_TEXTURE_INDICES
  BIND_FRAMEBUFFER_AND_TEXTURE

  glUseProgramObjectARB (shader_filter_hp->program);
  glUniform1iARB (shader_filter_hp->textures[0], 0);
  glUniform2fARB (shader_filter_hp->offset, subband_width, 0);

  #define RENDER_QUAD_HORIZONTAL_Hp(_x_offset, _quad_width) \
      schro_opengl_wavelet_render_quad (shader_filter_hp, \
          subband_width + (_x_offset), 0, _quad_width, height, width, height);

  RENDER_QUAD_HORIZONTAL_Hp (0, 1);

  if (subband_width > 2) {
    RENDER_QUAD_HORIZONTAL_Hp (1, 1);

    if (subband_width > 4) {
      RENDER_QUAD_HORIZONTAL_Hp (2, subband_width - 4);
    }

    RENDER_QUAD_HORIZONTAL_Hp (subband_width - 2, 1);
  }

  RENDER_QUAD_HORIZONTAL_Hp (subband_width - 1, 1);

  #undef RENDER_QUAD_HORIZONTAL_Hp

  /* copy L' */
  glUseProgramObjectARB (shader_copy->program);
  glUniform1iARB (shader_copy->textures[0], 0);

  schro_opengl_render_quad (0, 0, subband_width, height);

  SCHRO_OPENGL_CHECK_ERROR

  glFlush ();

  /* pass 6: horizontal interleave => i(L', H') = LL */
  SWITCH_FRAMEBUFFER_AND_TEXTURE_INDICES
  BIND_FRAMEBUFFER_AND_TEXTURE

  glUseProgramObjectARB (shader_horizontal_interleave->program);
  glUniform1iARB (shader_horizontal_interleave->textures[0], 0);
  glUniform2fARB (shader_horizontal_interleave->offset, width / 2, 0);

  schro_opengl_render_quad (0, 0, width, height);

  SCHRO_OPENGL_CHECK_ERROR

  glFlush ();

  /* pass 7: filter shift */
  if (filter_shift) {
    SWITCH_FRAMEBUFFER_AND_TEXTURE_INDICES
    BIND_FRAMEBUFFER_AND_TEXTURE

    glUseProgramObjectARB (shader_filter_shift->program);
    glUniform1iARB (shader_filter_shift->textures[0], 0);

    schro_opengl_render_quad (0, 0, width, height);

    SCHRO_OPENGL_CHECK_ERROR

    glFlush ();
  }

  /* pass 8: transfer data from secondary to primary framebuffer if previous
             pass result wasn't rendered into the primary framebuffer */
  if (framebuffer_index != 0) {
    SWITCH_FRAMEBUFFER_AND_TEXTURE_INDICES
    BIND_FRAMEBUFFER_AND_TEXTURE

    glUseProgramObjectARB (shader_copy->program);
    glUniform1iARB (shader_copy->textures[0], 0);

    schro_opengl_render_quad (0, 0, width, height);

    SCHRO_OPENGL_CHECK_ERROR

    glFlush ();
  }

  #undef SWITCH_FRAMEBUFFER_AND_TEXTURE_INDICES
  #undef BIND_FRAMEBUFFER_AND_TEXTURE

  glUseProgramObjectARB (0);
#if SCHRO_OPENGL_UNBIND_TEXTURES
  glBindTexture (GL_TEXTURE_RECTANGLE_ARB, 0);
#endif
  glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, 0);

  schro_opengl_unlock (opengl);
}

