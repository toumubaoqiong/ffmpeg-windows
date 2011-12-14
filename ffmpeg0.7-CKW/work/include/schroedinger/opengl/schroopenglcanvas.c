
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schrodebug.h>
#include <schroedinger/opengl/schroopengl.h>
#include <schroedinger/opengl/schroopenglcanvas.h>
#include <stdio.h>

unsigned int _schro_opengl_canvas_flags
    = 0
    //| SCHRO_OPENGL_CANVAS_STORE_BGRA /* FIXME: currently broken with packed formats in convert */
    | SCHRO_OPENGL_CANVAS_STORE_U8_AS_UI8
    //| SCHRO_OPENGL_CANVAS_STORE_U8_AS_F16
    //| SCHRO_OPENGL_CANVAS_STORE_U8_AS_F32
    | SCHRO_OPENGL_CANVAS_STORE_S16_AS_UI16
    //| SCHRO_OPENGL_CANVAS_STORE_S16_AS_I16 /* FIXME: doesn't yield a useable mapping in shader */
    //| SCHRO_OPENGL_CANVAS_STORE_S16_AS_U16
    //| SCHRO_OPENGL_CANVAS_STORE_S16_AS_F16 /* FIXME: currently broken in shader */
    //| SCHRO_OPENGL_CANVAS_STORE_S16_AS_F32 /* FIXME: currently broken in shader */

    //| SCHRO_OPENGL_CANVAS_PUSH_RENDER_QUAD
    //| SCHRO_OPENGL_CANVAS_PUSH_SHADER
    //| SCHRO_OPENGL_CANVAS_PUSH_DRAWPIXELS /* FIXME: currently broken */
    | SCHRO_OPENGL_CANVAS_PUSH_U8_PIXELBUFFER
    //| SCHRO_OPENGL_CANVAS_PUSH_U8_AS_F32
    //| SCHRO_OPENGL_CANVAS_PUSH_S16_PIXELBUFFER
    | SCHRO_OPENGL_CANVAS_PUSH_S16_AS_U16
    //| SCHRO_OPENGL_CANVAS_PUSH_S16_AS_F32

    //| SCHRO_OPENGL_CANVAS_PULL_PIXELBUFFER
    //| SCHRO_OPENGL_CANVAS_PULL_U8_AS_F32
    | SCHRO_OPENGL_CANVAS_PULL_S16_AS_U16
    //| SCHRO_OPENGL_CANVAS_PULL_S16_AS_F32
    ;

/* results on a NVIDIA 8800 GT with nvidia-glx-new drivers on Ubuntu Hardy */

/* U8: 259.028421/502.960679 mbyte/sec *//*
unsigned int _schro_opengl_canvas_flags
    = 0;*/

/* U8: 382.692291/447.573619 mbyte/sec *//*
unsigned int _schro_opengl_canvas_flags
    = SCHRO_OPENGL_CANVAS_PUSH_RENDER_QUAD
    | SCHRO_OPENGL_CANVAS_PUSH_U8_PIXELBUFFER;*/

/* U8: 972.809028/962.217704 mbyte/sec *//*
unsigned int _schro_opengl_canvas_flags
    = SCHRO_OPENGL_CANVAS_STORE_U8_AS_UI8;*/

/* U8: 1890.699986/848.954058 mbyte/sec *//*
unsigned int _schro_opengl_canvas_flags
    = SCHRO_OPENGL_CANVAS_STORE_U8_AS_UI8
    | SCHRO_OPENGL_CANVAS_PUSH_U8_PIXELBUFFER;*/

/* U8: 2003.478261/462.976159 mbyte/sec *//*
unsigned int _schro_opengl_canvas_flags
    = SCHRO_OPENGL_CANVAS_PUSH_U8_PIXELBUFFER;*/

/* S16: 22.265474/492.245509 mbyte/sec *//*
unsigned int _schro_opengl_canvas_flags
    = SCHRO_OPENGL_CANVAS_PUSH_S16_AS_U16
    | SCHRO_OPENGL_CANVAS_PUSH_S16_PIXELBUFFER
    | SCHRO_OPENGL_CANVAS_PULL_S16_AS_U16;*/

/* S16: 85.136173/499.591624 mbyte/sec *//*
unsigned int _schro_opengl_canvas_flags
    = SCHRO_OPENGL_CANVAS_PULL_S16_AS_U16;*/

/* S16: 266.568537/490.034023 mbyte/sec *//*
unsigned int _schro_opengl_canvas_flags
    = SCHRO_OPENGL_CANVAS_PUSH_S16_AS_U16
    | SCHRO_OPENGL_CANVAS_PULL_S16_AS_U16;*/

/* S16: 601.249413/914.319981 mbyte/sec *//*
unsigned int _schro_opengl_canvas_flags
    = SCHRO_OPENGL_CANVAS_STORE_S16_AS_UI16
    | SCHRO_OPENGL_CANVAS_PUSH_S16_AS_U16
    | SCHRO_OPENGL_CANVAS_PULL_S16_AS_U16;*/

void
schro_opengl_canvas_check_flags (void)
{
  int count;

  /* store */
  if (SCHRO_OPENGL_CANVAS_IS_FLAG_SET (STORE_BGRA)
      && !GLEW_EXT_bgra) {
    SCHRO_ERROR ("missing extension GL_EXT_bgra, disabling BGRA storing");
    SCHRO_OPENGL_CANVAS_CLEAR_FLAG (STORE_BGRA);
  }

  count = 0;

  count += SCHRO_OPENGL_CANVAS_IS_FLAG_SET (STORE_U8_AS_UI8)   ? 1 : 0;
  count += SCHRO_OPENGL_CANVAS_IS_FLAG_SET (STORE_S16_AS_UI16) ? 1 : 0;
  count += SCHRO_OPENGL_CANVAS_IS_FLAG_SET (STORE_S16_AS_I16)  ? 1 : 0;

  if (count > 0 && (!GLEW_EXT_texture_integer || !GLEW_EXT_gpu_shader4)) {
    SCHRO_ERROR ("missing extension GL_EXT_texture_integer or "
        "GLEW_EXT_gpu_shader4, can't store U8/S16 as UI8/UI16/I16, disabling "
        "U8/S16 as UI8/UI16/I16 storing");
    SCHRO_OPENGL_CANVAS_CLEAR_FLAG (STORE_U8_AS_UI8);
    SCHRO_OPENGL_CANVAS_CLEAR_FLAG (STORE_S16_AS_UI16);
    SCHRO_OPENGL_CANVAS_CLEAR_FLAG (STORE_S16_AS_I16);
  }

  count = 0;

  count += SCHRO_OPENGL_CANVAS_IS_FLAG_SET (STORE_U8_AS_F16)  ? 1 : 0;
  count += SCHRO_OPENGL_CANVAS_IS_FLAG_SET (STORE_U8_AS_F32)  ? 1 : 0;
  count += SCHRO_OPENGL_CANVAS_IS_FLAG_SET (STORE_S16_AS_F16) ? 1 : 0;
  count += SCHRO_OPENGL_CANVAS_IS_FLAG_SET (STORE_S16_AS_F32) ? 1 : 0;

  if (count > 0 && !GLEW_ARB_texture_float && !GLEW_ATI_texture_float) {
    SCHRO_ERROR ("missing extension GL_{ARB|ATI}_texture_float, can't store "
        "U8/S16 as F16/F32, disabling U8/S16 as F16/F32 storing");
    SCHRO_OPENGL_CANVAS_CLEAR_FLAG (STORE_U8_AS_F16);
    SCHRO_OPENGL_CANVAS_CLEAR_FLAG (STORE_U8_AS_F32);
    SCHRO_OPENGL_CANVAS_CLEAR_FLAG (STORE_S16_AS_F16);
    SCHRO_OPENGL_CANVAS_CLEAR_FLAG (STORE_S16_AS_F32);
  }

  /* store U8 */
  count = 0;

  count += SCHRO_OPENGL_CANVAS_IS_FLAG_SET (STORE_U8_AS_UI8) ? 1 : 0;
  count += SCHRO_OPENGL_CANVAS_IS_FLAG_SET (STORE_U8_AS_F16) ? 1 : 0;
  count += SCHRO_OPENGL_CANVAS_IS_FLAG_SET (STORE_U8_AS_F32) ? 1 : 0;

  if (count > 1) {
    SCHRO_ERROR ("multiple flags for U8 storage type are set, disabling all");
    SCHRO_OPENGL_CANVAS_CLEAR_FLAG (STORE_U8_AS_UI8);
    SCHRO_OPENGL_CANVAS_CLEAR_FLAG (STORE_U8_AS_F16);
    SCHRO_OPENGL_CANVAS_CLEAR_FLAG (STORE_U8_AS_F32);
  }

  /* store S16 */
  count = 0;

  count += SCHRO_OPENGL_CANVAS_IS_FLAG_SET (STORE_S16_AS_UI16) ? 1 : 0;
  count += SCHRO_OPENGL_CANVAS_IS_FLAG_SET (STORE_S16_AS_I16)  ? 1 : 0;
  count += SCHRO_OPENGL_CANVAS_IS_FLAG_SET (STORE_S16_AS_U16)  ? 1 : 0;
  count += SCHRO_OPENGL_CANVAS_IS_FLAG_SET (STORE_S16_AS_F16)  ? 1 : 0;
  count += SCHRO_OPENGL_CANVAS_IS_FLAG_SET (STORE_S16_AS_F32)  ? 1 : 0;

  if (count > 1) {
    SCHRO_ERROR ("multiple flags for S16 storage type are set, disabling all");
    SCHRO_OPENGL_CANVAS_CLEAR_FLAG (STORE_S16_AS_UI16);
    SCHRO_OPENGL_CANVAS_CLEAR_FLAG (STORE_S16_AS_I16);
    SCHRO_OPENGL_CANVAS_CLEAR_FLAG (STORE_S16_AS_U16);
    SCHRO_OPENGL_CANVAS_CLEAR_FLAG (STORE_S16_AS_F16);
    SCHRO_OPENGL_CANVAS_CLEAR_FLAG (STORE_S16_AS_F32);
  }

  /* push */
  if (!SCHRO_OPENGL_CANVAS_IS_FLAG_SET (PUSH_RENDER_QUAD) &&
      SCHRO_OPENGL_CANVAS_IS_FLAG_SET (PUSH_SHADER)) {
    SCHRO_ERROR ("can't use shader to push without render quad, disabling "
        "shader");
    SCHRO_OPENGL_CANVAS_CLEAR_FLAG (PUSH_SHADER);
  }

  count = 0;

  count += SCHRO_OPENGL_CANVAS_IS_FLAG_SET (STORE_U8_AS_UI8) ? 1 : 0;
  count += SCHRO_OPENGL_CANVAS_IS_FLAG_SET (STORE_S16_AS_UI16) ? 1 : 0;
  count += SCHRO_OPENGL_CANVAS_IS_FLAG_SET (STORE_S16_AS_I16)  ? 1 : 0;

  if (count > 0 &&
      SCHRO_OPENGL_CANVAS_IS_FLAG_SET (PUSH_RENDER_QUAD) &&
      !SCHRO_OPENGL_CANVAS_IS_FLAG_SET (PUSH_SHADER)) {
    SCHRO_ERROR ("can't push U8/S16 as UI8/UI16/I16 shader, disabling "
        "pushing U8/S16 as UI8/UI16/I16");
    SCHRO_OPENGL_CANVAS_CLEAR_FLAG (STORE_U8_AS_UI8);
    SCHRO_OPENGL_CANVAS_CLEAR_FLAG (STORE_S16_AS_UI16);
    SCHRO_OPENGL_CANVAS_CLEAR_FLAG (STORE_S16_AS_I16);
  }

  if (SCHRO_OPENGL_CANVAS_IS_FLAG_SET (PUSH_RENDER_QUAD) &&
      SCHRO_OPENGL_CANVAS_IS_FLAG_SET (PUSH_DRAWPIXELS)) {
    SCHRO_ERROR ("can't render quad and drawpixels to push, disabling "
        "drawpixels push");
    SCHRO_OPENGL_CANVAS_CLEAR_FLAG (PUSH_DRAWPIXELS);
  }

  if (SCHRO_OPENGL_CANVAS_IS_FLAG_SET (PUSH_DRAWPIXELS) &&
      !GLEW_ARB_window_pos) {
    SCHRO_ERROR ("missing extension GL_ARB_window_pos, disabling drawpixels "
        "push");
    SCHRO_OPENGL_CANVAS_CLEAR_FLAG (PUSH_DRAWPIXELS);
  }

  if (SCHRO_OPENGL_CANVAS_IS_FLAG_SET (PUSH_U8_PIXELBUFFER) &&
      (!GLEW_ARB_vertex_buffer_object || !GLEW_ARB_pixel_buffer_object)) {
    SCHRO_ERROR ("missing extensions GL_ARB_vertex_buffer_object and/or "
        "GL_ARB_pixel_buffer_object, disabling U8 pixelbuffer push");
    SCHRO_OPENGL_CANVAS_CLEAR_FLAG (PUSH_U8_PIXELBUFFER);
  }

  if (SCHRO_OPENGL_CANVAS_IS_FLAG_SET (PUSH_S16_PIXELBUFFER) &&
      (!GLEW_ARB_vertex_buffer_object || !GLEW_ARB_pixel_buffer_object)) {
    SCHRO_ERROR ("missing extensions GL_ARB_vertex_buffer_object and/or "
        "GL_ARB_pixel_buffer_object, disabling S16 pixelbuffer push");
    SCHRO_OPENGL_CANVAS_CLEAR_FLAG (PUSH_S16_PIXELBUFFER);
  }

  if (SCHRO_OPENGL_CANVAS_IS_FLAG_SET (PUSH_S16_AS_U16) &&
      SCHRO_OPENGL_CANVAS_IS_FLAG_SET (PUSH_S16_AS_F32)) {
    SCHRO_ERROR ("can't push S16 as U16 and F32, disabling U16 push");
    SCHRO_OPENGL_CANVAS_CLEAR_FLAG (PUSH_S16_AS_U16);
  }

  /* pull */
  if (SCHRO_OPENGL_CANVAS_IS_FLAG_SET (PULL_PIXELBUFFER) &&
      (!GLEW_ARB_vertex_buffer_object || !GLEW_ARB_pixel_buffer_object)) {
    SCHRO_ERROR ("missing extensions GL_ARB_vertex_buffer_object and/or "
        "GL_ARB_pixel_buffer_object, disabling S16 pixelbuffer pull");
    SCHRO_OPENGL_CANVAS_CLEAR_FLAG (PULL_PIXELBUFFER);
  }

  if (SCHRO_OPENGL_CANVAS_IS_FLAG_SET (PULL_S16_AS_U16) &&
      SCHRO_OPENGL_CANVAS_IS_FLAG_SET (PULL_S16_AS_F32)) {
    SCHRO_ERROR ("can't pull S16 as U16 and F32, disabling U16 pull");
    SCHRO_OPENGL_CANVAS_CLEAR_FLAG (PULL_S16_AS_U16);
  }
}

void
schro_opengl_canvas_print_flags (const char* indent)
{
  schro_opengl_canvas_check_flags ();

  #define PRINT_FLAG(_text, _flag) \
      printf ("%s  "_text"%s\n", indent, \
          SCHRO_OPENGL_CANVAS_IS_FLAG_SET (_flag) ? "on" : "off")

  printf ("%sstore flags\n", indent);

  PRINT_FLAG ("BGRA:            ", STORE_BGRA);
  PRINT_FLAG ("U8 as UI8:       ", STORE_U8_AS_UI8);
  PRINT_FLAG ("U8 as F16:       ", STORE_U8_AS_F16);
  PRINT_FLAG ("U8 as F32:       ", STORE_U8_AS_F32);
  PRINT_FLAG ("S16 as UI16:     ", STORE_S16_AS_UI16);
  PRINT_FLAG ("S16 as I16:      ", STORE_S16_AS_I16);
  PRINT_FLAG ("S16 as U16:      ", STORE_S16_AS_U16);
  PRINT_FLAG ("S16 as F16:      ", STORE_S16_AS_F16);
  PRINT_FLAG ("S16 as F32:      ", STORE_S16_AS_F32);

  printf ("%spush flags\n", indent);

  PRINT_FLAG ("render quad:     ", PUSH_RENDER_QUAD);
  PRINT_FLAG ("shader:          ", PUSH_SHADER);
  PRINT_FLAG ("drawpixels:      ", PUSH_DRAWPIXELS);
  PRINT_FLAG ("U8 pixelbuffer:  ", PUSH_U8_PIXELBUFFER);
  PRINT_FLAG ("U8 as F32:       ", PUSH_U8_AS_F32);
  PRINT_FLAG ("S16 pixelbuffer: ", PUSH_S16_PIXELBUFFER);
  PRINT_FLAG ("S16 as U16:      ", PUSH_S16_AS_U16);
  PRINT_FLAG ("S16 as F32:      ", PUSH_S16_AS_F32);

  printf ("%spull flags\n", indent);

  PRINT_FLAG ("pixelbuffer:     ", PULL_PIXELBUFFER);
  PRINT_FLAG ("U8 as F32:       ", PULL_U8_AS_F32);
  PRINT_FLAG ("S16 as U16:      ", PULL_S16_AS_U16);
  PRINT_FLAG ("S16 as F32:      ", PULL_S16_AS_F32);

  #undef PRINT_FLAG
}

SchroOpenGLCanvas *
schro_opengl_canvas_new (SchroOpenGL *opengl, SchroFrameFormat format,
    int width, int height)
{
  int i;
  int create_push_pixelbuffers = FALSE;
  SchroOpenGLCanvas *canvas = schro_malloc0 (sizeof (SchroOpenGLCanvas));

  schro_opengl_canvas_check_flags (); // FIXME

  schro_opengl_lock (opengl);

  canvas->opengl = opengl;
  canvas->format = format;
  canvas->width = width;
  canvas->height = height;

  switch (SCHRO_FRAME_FORMAT_DEPTH (format)) {
    case SCHRO_FRAME_FORMAT_DEPTH_U8:
      if (SCHRO_OPENGL_CANVAS_IS_FLAG_SET (STORE_U8_AS_F16)) {
        if (!SCHRO_FRAME_IS_PACKED (format) && GLEW_NV_float_buffer) {
          canvas->texture.internal_format = GL_FLOAT_R16_NV;
        } else {
          canvas->texture.internal_format = GL_RGBA16F_ARB;
        }

        canvas->texture.type = GL_FLOAT;
      } else if (SCHRO_OPENGL_CANVAS_IS_FLAG_SET (STORE_U8_AS_F32)) {
        if (!SCHRO_FRAME_IS_PACKED (format) && GLEW_NV_float_buffer) {
          canvas->texture.internal_format = GL_FLOAT_R32_NV;
        } else {
          canvas->texture.internal_format = GL_RGBA32F_ARB;
        }

        canvas->texture.type = GL_FLOAT;
      } else if (SCHRO_OPENGL_CANVAS_IS_FLAG_SET (STORE_U8_AS_UI8)) {
        if (SCHRO_FRAME_IS_PACKED (format)) {
          canvas->texture.internal_format = GL_RGBA8UI_EXT;
        } else {
          canvas->texture.internal_format = GL_ALPHA8UI_EXT;
        }

        canvas->texture.type = GL_UNSIGNED_BYTE;
      } else {
        /* must use RGBA format here, because other formats are in general
           not supported by framebuffers */
        canvas->texture.internal_format = GL_RGBA8;
        canvas->texture.type = GL_UNSIGNED_BYTE;
      }

      if (SCHRO_FRAME_IS_PACKED (format)) {
        if (SCHRO_OPENGL_CANVAS_IS_FLAG_SET (STORE_BGRA)) {
          if (SCHRO_OPENGL_CANVAS_IS_FLAG_SET (STORE_U8_AS_UI8)) {
            canvas->texture.pixel_format = GL_BGRA_INTEGER_EXT;
          } else {
            canvas->texture.pixel_format = GL_BGRA;
          }

          canvas->texture.channels = 4;
        } else {
          if (SCHRO_OPENGL_CANVAS_IS_FLAG_SET (STORE_U8_AS_UI8)) {
            canvas->texture.pixel_format = GL_RGBA_INTEGER_EXT;
          } else {
            canvas->texture.pixel_format = GL_RGBA;
          }

          canvas->texture.channels = 4;
        }
      } else {
        if (SCHRO_OPENGL_CANVAS_IS_FLAG_SET (STORE_U8_AS_UI8)) {
          canvas->texture.pixel_format = GL_ALPHA_INTEGER_EXT;
        } else {
          canvas->texture.pixel_format = GL_RED;
        }

        canvas->texture.channels = 1;
      }

      if (SCHRO_OPENGL_CANVAS_IS_FLAG_SET (PUSH_U8_PIXELBUFFER)) {
        create_push_pixelbuffers = TRUE;
      }

      if (SCHRO_OPENGL_CANVAS_IS_FLAG_SET (PUSH_U8_AS_F32)) {
        canvas->push.type = GL_FLOAT;
        canvas->push.stride = ROUND_UP_4 (width * canvas->texture.channels
            * sizeof (float));
      } else {
        canvas->push.type = GL_UNSIGNED_BYTE;
        canvas->push.stride = ROUND_UP_4 (width * canvas->texture.channels
            * sizeof (uint8_t));
      }

      if (SCHRO_OPENGL_CANVAS_IS_FLAG_SET (PULL_U8_AS_F32)) {
        canvas->pull.type = GL_FLOAT;
        canvas->pull.stride = ROUND_UP_4 (width * canvas->texture.channels
            * sizeof (float));
      } else {
        canvas->pull.type = GL_UNSIGNED_BYTE;
        canvas->pull.stride = ROUND_UP_4 (width * canvas->texture.channels
            * sizeof (uint8_t));
      }

      break;
    case SCHRO_FRAME_FORMAT_DEPTH_S16:
      if (SCHRO_OPENGL_CANVAS_IS_FLAG_SET (STORE_S16_AS_F16)) {
        if (!SCHRO_FRAME_IS_PACKED (format) && GLEW_NV_float_buffer) {
          canvas->texture.internal_format = GL_FLOAT_R16_NV;
        } else {
          canvas->texture.internal_format = GL_RGBA16F_ARB;
        }

        canvas->texture.type = GL_FLOAT;
      } else if (SCHRO_OPENGL_CANVAS_IS_FLAG_SET (STORE_S16_AS_F32)) {
        if (!SCHRO_FRAME_IS_PACKED (format) && GLEW_NV_float_buffer) {
          canvas->texture.internal_format = GL_FLOAT_R32_NV;
        } else {
          canvas->texture.internal_format = GL_RGBA32F_ARB;
        }

        canvas->texture.type = GL_FLOAT;
      } else if (SCHRO_OPENGL_CANVAS_IS_FLAG_SET (STORE_S16_AS_UI16)) {
        if (SCHRO_FRAME_IS_PACKED (format)) {
          canvas->texture.internal_format = GL_RGBA16UI_EXT;
        } else {
          canvas->texture.internal_format = GL_ALPHA16UI_EXT;
        }

        canvas->texture.type = GL_UNSIGNED_SHORT;
      } else if (SCHRO_OPENGL_CANVAS_IS_FLAG_SET (STORE_S16_AS_I16)) {
        if (SCHRO_FRAME_IS_PACKED (format)) {
          canvas->texture.internal_format = GL_RGBA16I_EXT;
        } else {
          canvas->texture.internal_format = GL_ALPHA16I_EXT;
        }

        canvas->texture.type = GL_SHORT;
      } else if (SCHRO_OPENGL_CANVAS_IS_FLAG_SET (STORE_S16_AS_U16)) {
        /* must use RGBA format here, because other formats are in general
           not supported by framebuffers */
        canvas->texture.internal_format = GL_RGBA16;
        canvas->texture.type = GL_UNSIGNED_SHORT;
      } else {
        /* must use RGBA format here, because other formats are in general
           not supported by framebuffers */
        canvas->texture.internal_format = GL_RGBA16;
        canvas->texture.type = GL_SHORT;
      }

      if (SCHRO_FRAME_IS_PACKED (format)) {
        if (SCHRO_OPENGL_CANVAS_IS_FLAG_SET (STORE_BGRA)) {
          if (SCHRO_OPENGL_CANVAS_IS_FLAG_SET (STORE_S16_AS_UI16) ||
              SCHRO_OPENGL_CANVAS_IS_FLAG_SET (STORE_S16_AS_I16)) {
            canvas->texture.pixel_format = GL_BGRA_INTEGER_EXT;
          } else {
            canvas->texture.pixel_format = GL_BGRA;
          }

          canvas->texture.channels = 4;
        } else {
          if (SCHRO_OPENGL_CANVAS_IS_FLAG_SET (STORE_S16_AS_UI16) ||
              SCHRO_OPENGL_CANVAS_IS_FLAG_SET (STORE_S16_AS_I16)) {
            canvas->texture.pixel_format = GL_RGBA_INTEGER_EXT;
          } else {
            canvas->texture.pixel_format = GL_RGBA;
          }

          canvas->texture.channels = 4;
        }
      } else {
        if (SCHRO_OPENGL_CANVAS_IS_FLAG_SET (STORE_S16_AS_UI16) ||
            SCHRO_OPENGL_CANVAS_IS_FLAG_SET (STORE_S16_AS_I16)) {
          canvas->texture.pixel_format = GL_ALPHA_INTEGER_EXT;
        } else {
          canvas->texture.pixel_format = GL_RED;
        }

        canvas->texture.channels = 1;
      }

      if (SCHRO_OPENGL_CANVAS_IS_FLAG_SET (PUSH_S16_AS_U16)) {
        canvas->push.type = GL_UNSIGNED_SHORT;
        canvas->push.stride = ROUND_UP_4 (width * canvas->texture.channels
            * sizeof (uint16_t));
      } else if (SCHRO_OPENGL_CANVAS_IS_FLAG_SET (PUSH_S16_AS_F32)) {
        canvas->push.type = GL_FLOAT;
        canvas->push.stride = ROUND_UP_4 (width * canvas->texture.channels
            * sizeof (float));
      } else {
        canvas->push.type = GL_SHORT;
        canvas->push.stride = ROUND_UP_4 (width * canvas->texture.channels
            * sizeof (int16_t));
      }

      if (SCHRO_OPENGL_CANVAS_IS_FLAG_SET (PUSH_S16_PIXELBUFFER)) {
        create_push_pixelbuffers = TRUE;
      }

      if (SCHRO_OPENGL_CANVAS_IS_FLAG_SET (PULL_S16_AS_U16)) {
        /* must pull S16 as GL_UNSIGNED_SHORT instead of GL_SHORT because
           the OpenGL mapping form internal float represenation into S16
           values with GL_SHORT maps 0.0 to 0 and 1.0 to 32767 clamping all
           negative values to 0, see glReadPixel documentation. so the pull
           is done with GL_UNSIGNED_SHORT and the resulting U16 values are
           manually shifted to S16 */
        canvas->pull.type = GL_UNSIGNED_SHORT;
        canvas->pull.stride = ROUND_UP_4 (width * canvas->texture.channels
            * sizeof (uint16_t));
      } else if (SCHRO_OPENGL_CANVAS_IS_FLAG_SET (PULL_S16_AS_F32)) {
        canvas->pull.type = GL_FLOAT;
        canvas->pull.stride = ROUND_UP_4 (width * canvas->texture.channels
            * sizeof (float));
      } else {
        // FIXME: pulling S16 as GL_SHORT doesn't work in general, maybe
        // it's the right mode if the internal format is an integer format
        // but for some reason storing as I16 doesn't work either and only
        // gives garbage pull results
        canvas->pull.type = GL_SHORT;
        canvas->pull.stride = ROUND_UP_4 (width * canvas->texture.channels
            * sizeof (int16_t));
      }

      break;
    default:
      SCHRO_ASSERT (0);
      break;
  }

  /* textures */
  for (i = 0; i < 2; ++i) {
    glGenTextures (1, &canvas->texture.handles[i]);
    glBindTexture (GL_TEXTURE_RECTANGLE_ARB, canvas->texture.handles[i]);
    glTexImage2D (GL_TEXTURE_RECTANGLE_ARB, 0,
        canvas->texture.internal_format, width, height, 0,
        canvas->texture.pixel_format, canvas->texture.type, NULL);
    glTexParameteri (GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER,
        GL_NEAREST);
    glTexParameteri (GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER,
        GL_NEAREST);
    glTexParameteri (GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri (GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexEnvi (GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    SCHRO_OPENGL_CHECK_ERROR
  }

#if SCHRO_OPENGL_UNBIND_TEXTURES
  glBindTexture (GL_TEXTURE_RECTANGLE_ARB, 0);
#endif

  /* framebuffers */
  for (i = 0; i < 2; ++i) {
    glGenFramebuffersEXT (1, &canvas->framebuffers[i]);
    glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, canvas->framebuffers[i]);
    glFramebufferTexture2DEXT (GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT,
        GL_TEXTURE_RECTANGLE_ARB, canvas->texture.handles[i], 0);
    glDrawBuffer (GL_COLOR_ATTACHMENT0_EXT);
    glReadBuffer (GL_COLOR_ATTACHMENT0_EXT);

    SCHRO_OPENGL_CHECK_ERROR
    // FIXME: checking framebuffer status is an expensive operation
    SCHRO_OPENGL_CHECK_FRAMEBUFFER
  }

  glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, 0);

  SCHRO_ASSERT (height >= SCHRO_OPENGL_TRANSFER_PIXELBUFFERS);

  /* push pixelbuffers */
  if (create_push_pixelbuffers) {
    for (i = 0; i < SCHRO_OPENGL_TRANSFER_PIXELBUFFERS; ++i) {
      SCHRO_ASSERT (canvas->push.pixelbuffers[i] == 0);

      if (i == SCHRO_OPENGL_TRANSFER_PIXELBUFFERS - 1) {
        canvas->push.heights[i]
            = height - (height / SCHRO_OPENGL_TRANSFER_PIXELBUFFERS) * i;
      } else {
        canvas->push.heights[i] = height / SCHRO_OPENGL_TRANSFER_PIXELBUFFERS;
      }

      glGenBuffersARB (1, &canvas->push.pixelbuffers[i]);
      glBindBufferARB (GL_PIXEL_UNPACK_BUFFER_ARB,
          canvas->push.pixelbuffers[i]);
      glBufferDataARB (GL_PIXEL_UNPACK_BUFFER_ARB,
          canvas->push.stride * canvas->push.heights[i], NULL,
          GL_STREAM_DRAW_ARB);

      SCHRO_OPENGL_CHECK_ERROR
    }

    glBindBufferARB (GL_PIXEL_UNPACK_BUFFER_ARB, 0);
  }

  /* pull pixelbuffers */
  if (SCHRO_OPENGL_CANVAS_IS_FLAG_SET (PULL_PIXELBUFFER)) {
    for (i = 0; i < SCHRO_OPENGL_TRANSFER_PIXELBUFFERS; ++i) {
      SCHRO_ASSERT (canvas->pull.pixelbuffers[i] == 0);

      if (i == SCHRO_OPENGL_TRANSFER_PIXELBUFFERS - 1) {
        canvas->pull.heights[i]
            = height - (height / SCHRO_OPENGL_TRANSFER_PIXELBUFFERS) * i;
      } else {
        canvas->pull.heights[i] = height / SCHRO_OPENGL_TRANSFER_PIXELBUFFERS;
      }

      glGenBuffersARB (1, &canvas->pull.pixelbuffers[i]);
      glBindBufferARB (GL_PIXEL_PACK_BUFFER_ARB, canvas->pull.pixelbuffers[i]);
      glBufferDataARB (GL_PIXEL_PACK_BUFFER_ARB,
          canvas->pull.stride * canvas->pull.heights[i], NULL,
          GL_STATIC_READ_ARB);

      SCHRO_OPENGL_CHECK_ERROR
    }

    glBindBufferARB (GL_PIXEL_PACK_BUFFER_ARB, 0);
  }

  schro_opengl_unlock (opengl);

  return canvas;
}

void
schro_opengl_canvas_free (SchroOpenGLCanvas *canvas)
{
  int i;
  SchroOpenGL *opengl;

  SCHRO_ASSERT (canvas != NULL);

  opengl = canvas->opengl;
  canvas->opengl = NULL;

  schro_opengl_lock (opengl);

  /* textures */
  for (i = 0; i < 2; ++i) {
    if (glIsTexture (canvas->texture.handles[i])) {
      glDeleteTextures (1, &canvas->texture.handles[i]);

      canvas->texture.handles[i] = 0;

      SCHRO_OPENGL_CHECK_ERROR
    }
  }

  /* framebuffers */
  for (i = 0; i < 2; ++i) {
    if (glIsFramebufferEXT (canvas->framebuffers[i])) {
      glDeleteFramebuffersEXT (1, &canvas->framebuffers[i]);

      canvas->framebuffers[i] = 0;

      SCHRO_OPENGL_CHECK_ERROR
    }
  }

  /* pixelbuffers */
  for (i = 0; i < SCHRO_OPENGL_TRANSFER_PIXELBUFFERS; ++i) {
    if (glIsBufferARB (canvas->push.pixelbuffers[i])) {
      glDeleteBuffersARB (1, &canvas->push.pixelbuffers[i]);

      canvas->push.pixelbuffers[i] = 0;

      SCHRO_OPENGL_CHECK_ERROR
    }

    if (glIsBufferARB (canvas->pull.pixelbuffers[i])) {
      glDeleteBuffersARB (1, &canvas->pull.pixelbuffers[i]);

      canvas->pull.pixelbuffers[i] = 0;

      SCHRO_OPENGL_CHECK_ERROR
    }
  }

  schro_opengl_unlock (opengl);

  schro_free (canvas);
}

SchroOpenGLCanvasPool *schro_opengl_canvas_pool_new (SchroOpenGL *opengl)
{
  SchroOpenGLCanvasPool *canvas_pool;

  canvas_pool = schro_malloc0 (sizeof (SchroOpenGLCanvasPool));

  canvas_pool->opengl = opengl;
  canvas_pool->size = 0;

  return canvas_pool;
}

void schro_opengl_canvas_pool_free (SchroOpenGLCanvasPool* canvas_pool)
{
  int i;

  SCHRO_ASSERT (canvas_pool->size >= 0);
  SCHRO_ASSERT (canvas_pool->size <= SCHRO_OPENGL_CANVAS_POOL_SIZE);

  schro_opengl_lock (canvas_pool->opengl);

  for (i = 0; i < canvas_pool->size; ++i) {
    schro_opengl_canvas_free (canvas_pool->canvases[i]);
  }

  schro_opengl_unlock (canvas_pool->opengl);

  schro_free (canvas_pool);
}

static int
schro_opengl_canvas_pool_is_empty (SchroOpenGLCanvasPool* canvas_pool)
{
  SCHRO_ASSERT (canvas_pool->size >= 0);
  SCHRO_ASSERT (canvas_pool->size <= SCHRO_OPENGL_CANVAS_POOL_SIZE);

  return canvas_pool->size == 0;
}

static int
schro_opengl_canvas_pool_is_full (SchroOpenGLCanvasPool* canvas_pool)
{
  SCHRO_ASSERT (canvas_pool->size >= 0);
  SCHRO_ASSERT (canvas_pool->size <= SCHRO_OPENGL_CANVAS_POOL_SIZE);

  return canvas_pool->size == SCHRO_OPENGL_CANVAS_POOL_SIZE;
}

static SchroOpenGLCanvas *
schro_opengl_canvas_pool_pull (SchroOpenGLCanvasPool* canvas_pool,
    SchroFrameFormat format, int width, int height)
{
  int i;
  SchroOpenGLCanvas *canvas;

  SCHRO_ASSERT (canvas_pool->size >= 1);
  SCHRO_ASSERT (canvas_pool->size <= SCHRO_OPENGL_CANVAS_POOL_SIZE);

  for (i = 0; i < canvas_pool->size; ++i) {
    canvas = canvas_pool->canvases[i];

    if (canvas->format == format && canvas->width == width &&
        canvas->height == height) {
      --canvas_pool->size;

      /* move the last canvas in the pool to the slot of the pulled one to
         maintain the pool continuous in memory */
      canvas_pool->canvases[i] = canvas_pool->canvases[canvas_pool->size];

      return canvas;
    }
  }

  return NULL;
}

static void
schro_opengl_canvas_pool_push (SchroOpenGLCanvasPool* canvas_pool,
    SchroOpenGLCanvas *canvas)
{
  SCHRO_ASSERT (canvas_pool->size >= 0);
  SCHRO_ASSERT (canvas_pool->size <= SCHRO_OPENGL_CANVAS_POOL_SIZE - 1);

  canvas_pool->canvases[canvas_pool->size] = canvas;

  ++canvas_pool->size;
}

SchroOpenGLCanvas *
schro_opengl_canvas_pool_pull_or_new (SchroOpenGLCanvasPool* canvas_pool,
    SchroOpenGL *opengl, SchroFrameFormat format, int width, int height)
{
  SchroOpenGLCanvas *canvas = NULL;

  schro_opengl_lock (canvas_pool->opengl);

  if (!schro_opengl_canvas_pool_is_empty (canvas_pool)) {
    canvas = schro_opengl_canvas_pool_pull (canvas_pool, format, width,
        height);
  }

  if (!canvas) {
    canvas = schro_opengl_canvas_new (opengl, format, width, height);
  }

  schro_opengl_unlock (canvas_pool->opengl);

  return canvas;
}

void
schro_opengl_canvas_pool_push_or_free (SchroOpenGLCanvasPool* canvas_pool,
    SchroOpenGLCanvas *canvas)
{
  schro_opengl_lock (canvas_pool->opengl);

  if (!schro_opengl_canvas_pool_is_full (canvas_pool)) {
    schro_opengl_canvas_pool_push (canvas_pool, canvas);
  } else {
    schro_opengl_canvas_free (canvas);
  }

  schro_opengl_unlock (canvas_pool->opengl);
}

