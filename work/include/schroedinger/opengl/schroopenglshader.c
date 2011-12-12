
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schrodebug.h>
#include <schroedinger/opengl/schroopengl.h>
#include <schroedinger/opengl/schroopenglcanvas.h>
#include <schroedinger/opengl/schroopenglshader.h>
#include <stdio.h>
#include <string.h>

static char*
schro_opengl_shader_add_linenumbers (const char* code)
{
  const char *src = code;
  char *dest;
  char *linenumbered_code;
  char number[16];
  int lines = 1;
  int size;

  while (*src) {
    if (*src == '\n') {
      ++lines;
    }

    ++src;
  }

  snprintf (number, sizeof (number) - 1, "%3i: ", lines);

  size = strlen (code) + 1 + lines * strlen (number);
  linenumbered_code = schro_malloc0 (size);
  src = code;
  dest = linenumbered_code;

  strcpy (dest, "  1: ");

  dest += strlen ("  1: ");
  lines = 2;

  while (*src) {
    *dest++ = *src;

    if (*src == '\n') {
      snprintf (number, sizeof (number) - 1, "%3i: ", lines);
      strcpy (dest, number);

      dest += strlen (number);
      ++lines;
    }

    ++src;
  }

  return linenumbered_code;
}

static int
schro_opengl_shader_check_status (GLhandleARB handle, GLenum status,
    const char* code, const char* name)
{
  GLint result;
  GLint length;
  char* infolog;
  char* linenumbered_code;

  glGetObjectParameterivARB (handle, status, &result);
  glGetObjectParameterivARB (handle, GL_OBJECT_INFO_LOG_LENGTH_ARB, &length);

  if (length < 1) {
    SCHRO_ERROR ("invalid infolog length %i", length);
    return FALSE;
  }

  infolog = schro_malloc0 (length * sizeof (char));

  glGetInfoLogARB (handle, length, &length, infolog);

  if (length > 0) {
    linenumbered_code = schro_opengl_shader_add_linenumbers (code);

    SCHRO_ERROR ("\nshadername:\n%s\nshadercode:\n%s\ninfolog:\n%s", name,
        linenumbered_code, infolog);

    schro_free (linenumbered_code);
  }

  schro_free (infolog);

  return result != 0;
}

static SchroOpenGLShader *
schro_opengl_shader_new (const char* code, const char* name)
{
  SchroOpenGLShader *shader;
  GLhandleARB handle;
  int ok;

  shader = schro_malloc0 (sizeof (SchroOpenGLShader));
  handle = glCreateShaderObjectARB (GL_FRAGMENT_SHADER_ARB);

  glShaderSourceARB (handle, 1, (const char**)&code, 0);
  glCompileShaderARB (handle);

  ok = schro_opengl_shader_check_status (handle, GL_OBJECT_COMPILE_STATUS_ARB,
      code, name);

  SCHRO_ASSERT (ok);

  shader->program = glCreateProgramObjectARB ();

  glAttachObjectARB (shader->program, handle);
  glDeleteObjectARB (handle);
  glLinkProgramARB (shader->program);

  ok = schro_opengl_shader_check_status (shader->program,
      GL_OBJECT_LINK_STATUS_ARB, code, name);

  SCHRO_ASSERT (ok);

  glValidateProgramARB (shader->program);

  ok = schro_opengl_shader_check_status (shader->program,
      GL_OBJECT_VALIDATE_STATUS_ARB, code, name);

  SCHRO_ASSERT (ok);

  #define UNIFORM_LOCATION_SAMPLER(_name, _member) \
      do { \
        if (strstr (code, "uniform sampler2DRect "#_name";") || \
            strstr (code, "uniform usampler2DRect "#_name";") || \
            strstr (code, "uniform isampler2DRect "#_name";")) { \
          shader->_member = glGetUniformLocationARB (shader->program, #_name); \
        } else { \
          shader->_member = -1; \
        } \
      } while (0)

  #define UNIFORM_LOCATION(_type, _name, _member) \
      do { \
        if (strstr (code, "uniform "#_type" "#_name";")) { \
          shader->_member = glGetUniformLocationARB (shader->program, #_name); \
        } else { \
          shader->_member = -1; \
        } \
      } while (0)

  UNIFORM_LOCATION_SAMPLER (texture1, textures[0]);
  UNIFORM_LOCATION_SAMPLER (texture2, textures[1]);
  UNIFORM_LOCATION_SAMPLER (texture3, textures[2]);
  UNIFORM_LOCATION (vec2, offset, offset);
  UNIFORM_LOCATION (vec2, origin, origin);
  UNIFORM_LOCATION (vec2, size, size);
  UNIFORM_LOCATION (vec2, four_decrease, four_decrease);
  UNIFORM_LOCATION (vec2, three_decrease, three_decrease);
  UNIFORM_LOCATION (vec2, two_decrease, two_decrease);
  UNIFORM_LOCATION (vec2, one_decrease, one_decrease);
  UNIFORM_LOCATION (vec2, one_increase, one_increase);
  UNIFORM_LOCATION (vec2, two_increase, two_increase);
  UNIFORM_LOCATION (vec2, three_increase, three_increase);
  UNIFORM_LOCATION (vec2, four_increase, four_increase);
  UNIFORM_LOCATION (float, dc, dc);
  UNIFORM_LOCATION (float, weight, weight);
  UNIFORM_LOCATION (float, addend, addend);
  UNIFORM_LOCATION (float, divisor, divisor);

  #undef UNIFORM_LOCATION_SAMPLER
  #undef UNIFORM_LOCATION

  if (GLEW_EXT_gpu_shader4) {
    if (strstr (code, "varying out uvec4 fragcolor_u8;")) {
      glBindFragDataLocationEXT (shader->program, 0, "fragcolor_u8");
    } else if (strstr (code, "varying out ivec4 fragcolor_s16;")) {
      glBindFragDataLocationEXT (shader->program, 0, "fragcolor_s16");
    }
  }

  return shader;
}

static void
schro_opengl_shader_free (SchroOpenGLShader *shader)
{
  SCHRO_ASSERT (shader != NULL);

  glDeleteObjectARB (shader->program);

  schro_free (shader);
}

struct ShaderCode {
  int index;
  const char *name;
  unsigned int flags;
  const char *code;
  const char *code_integer;
};

#define SHADER_FLAG_USE_U8  (1 << 0)
#define SHADER_FLAG_USE_S16 (1 << 1)

#define SHADER_HEADER \
    "#version 110\n" \
    "#extension GL_ARB_draw_buffers : require\n" \
    "#extension GL_ARB_texture_rectangle : require\n"

#define SHADER_HEADER_INTEGER \
    "#version 120\n" \
    "#extension GL_ARB_draw_buffers : require\n" \
    "#extension GL_ARB_texture_rectangle : require\n" \
    "#extension GL_EXT_gpu_shader4 : require\n" \
    "#define uint unsigned int\n"

#define SHADER_READ_U8(_texture, _postfix) \
    "uniform sampler2DRect "_texture";\n" \
    "float read"_postfix"_u8 (vec2 offset = vec2 (0.0)) {\n" \
    "  float fp = texture2DRect ("_texture", gl_TexCoord[0].xy + offset).r;\n" \
    /* scale from FP [0..1] to real U8 [0..255] and apply proper rounding */ \
    "  return floor (fp * 255.0 + 0.5);\n" \
    "}\n"

#define SHADER_WRITE_U8 \
    "void write_u8 (float u8) {\n" \
    /* scale from real U8 [0..255] to FP [0..1] */ \
    "  gl_FragColor = vec4 (u8 / 255.0);\n" \
    "}\n"

#define SHADER_READ_U8_INTEGER(_texture, _postfix) \
    "uniform usampler2DRect "_texture";\n" \
    "uint read"_postfix"_u8 (vec2 offset = vec2 (0.0)) {\n" \
    "  return texture2DRect ("_texture", gl_TexCoord[0].xy + offset).a;\n" \
    "}\n"

#define SHADER_WRITE_U8_INTEGER \
    "varying out uvec4 fragcolor_u8;\n" \
    "void write_u8 (uint u8) {\n" \
    "  fragcolor_u8 = uvec4 (u8);\n" \
    "}\n"

#define SHADER_WRITE_U8_VEC4_INTEGER \
    "varying out uvec4 fragcolor_u8;\n" \
    "void write_u8_vec4 (uvec4 u8) {\n" \
    "  fragcolor_u8 = u8;\n" \
    "}\n"

#define SHADER_WRITE_U16_INTEGER \
    "varying out ivec4 fragcolor_s16;\n" \
    "void write_u16 (int u16) {\n" \
    "  fragcolor_s16 = ivec4 (u16);\n" \
    "}\n"

#define SHADER_READ_S16(_texture, _postfix) \
    "uniform sampler2DRect "_texture";\n" \
    "float read"_postfix"_s16 (vec2 offset = vec2 (0.0)) {\n" \
    /* scale and bias from FP range [0..1] to S16 range [-32768..32767] */ \
    "  return floor (texture2DRect ("_texture", gl_TexCoord[0].xy + offset).r\n" \
    "      * 65535.0 + 0.5) - 32768.0;\n" \
    "}\n"

#define SHADER_WRITE_S16 \
    "void write_s16 (float s16) {\n" \
    /* scale and bias from S16 range [-32768..32767] to FP range [0..1] */ \
    "  gl_FragColor = vec4 ((s16 + 32768.0) / 65535.0);\n" \
    "}\n"

#define SHADER_READ_S16_INTEGER(_texture, _postfix) \
    "uniform isampler2DRect "_texture";\n" \
    "int read"_postfix"_s16 (vec2 offset = vec2 (0.0)) {\n" \
    "  return texture2DRect ("_texture", gl_TexCoord[0].xy + offset).a - 32768;\n" \
    "}\n"

#define SHADER_WRITE_S16_INTEGER \
    "varying out ivec4 fragcolor_s16;\n" \
    "void write_s16 (int s16) {\n" \
    "  fragcolor_s16 = ivec4 (s16 + 32768);\n" \
    "}\n"

#define SHADER_CAST_S16_U8_INTEGER \
    "int cast_s16_u8 (uint u8) {\n" \
    "  return int (u8);\n" \
    "}\n"

#define SHADER_CAST_U8_S16_INTEGER \
    "uint cast_u8_s16 (int s16) {\n" \
    "  return uint (clamp (s16, 0, 255));\n" \
    "}\n"

#define SHADER_DIVIDE_S16 \
    "float divide_s16 (float value, float divisor) {\n" \
    "  return floor (value / divisor);\n" \
    "}\n"

#if 0

#define SHADER_DIVIDE_S16_INTEGER \
    "int divide_s16 (int value, int divisor) {\n" \
    "  return value < 0 ? (value - (divisor - ((-value) % divisor))) / divisor\n" \
    "      : value / divisor;\n" \
    "}\n"

#else

#define SHADER_DIVIDE_S16_INTEGER \
    "int divide_s16 (int value, int divisor) {\n" \
    "  return int (floor (float (value) / float (divisor)));\n" \
    "}\n"

#endif

#define SHADER_RSHIFT(_a, _b) \
    "float rshift (float value) {\n" \
    "  return floor ((value + ("#_a")) / ("#_b"));\n" \
    "}\n"

#define SHADER_RSHIFT_U8_INTEGER(_a, _b) \
    "uint rshift_u8 (uint value) {\n" \
    "  return (value + ("#_a") / ("#_b");\n" \
    "}\n"

#define SHADER_RSHIFT_S16_INTEGER(_a, _b) \
    SHADER_DIVIDE_S16_INTEGER \
    "int rshift_s16 (int value) {\n" \
    "  return divide_s16 (value + ("#_a"), "#_b");\n" \
    "}\n"

static struct ShaderCode schro_opengl_shader_code_list[] = {
  { SCHRO_OPENGL_SHADER_COPY_U8,
      "copy_u8", SHADER_FLAG_USE_U8,
      SHADER_HEADER
      "uniform sampler2DRect texture1;\n" /* U8 */
      "void main (void) {\n"
      "  gl_FragColor = texture2DRect (texture1, gl_TexCoord[0].xy);\n"
      "}\n",
      SHADER_HEADER_INTEGER
      SHADER_READ_U8_INTEGER ("texture1", "") /* U8 */
      SHADER_WRITE_U8_INTEGER
      "void main (void) {\n"
      "  write_u8 (read_u8 ());\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_COPY_S16,
      "copy_s16", SHADER_FLAG_USE_S16,
      SHADER_HEADER
      "uniform sampler2DRect texture1;\n" /* S16 */
      "void main (void) {\n"
      "  gl_FragColor = texture2DRect (texture1, gl_TexCoord[0].xy);\n"
      "}\n",
      SHADER_HEADER_INTEGER
      SHADER_READ_S16_INTEGER ("texture1", "") /* S16 */
      SHADER_WRITE_S16_INTEGER
      "void main (void) {\n"
      "  write_s16 (read_s16 ());\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_S16,
      "convert_u8_s16", SHADER_FLAG_USE_U8 | SHADER_FLAG_USE_S16,
      SHADER_HEADER
      "uniform sampler2DRect texture1;\n" /* S16 */
      "const float scale = 255.0 / 65535.0;\n"
      "const float bias = (32767.0 - 127.0) / 65535.0;\n"
      "void main (void) {\n"
      "  gl_FragColor\n"
      "      = (texture2DRect (texture1, gl_TexCoord[0].xy) - bias) / scale;\n"
      "}\n",
      SHADER_HEADER_INTEGER
      SHADER_READ_S16_INTEGER ("texture1", "") /* S16 */
      SHADER_WRITE_U8_INTEGER
      SHADER_CAST_U8_S16_INTEGER
      "void main (void) {\n"
      "  write_u8 (cast_u8_s16 (read_s16 () + 128));\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_S16_U8,
      "convert_s16_u8", SHADER_FLAG_USE_U8 | SHADER_FLAG_USE_S16,
      SHADER_HEADER
      "uniform sampler2DRect texture1;\n" /* U8 */
      "const float scale = 255.0 / 65535.0;\n"
      "const float bias = (32767.0 - 127.0) / 65535.0;\n"
      "void main (void) {\n"
      "  gl_FragColor\n"
      "      = texture2DRect (texture1, gl_TexCoord[0].xy) * scale + bias;\n"
      "}\n",
      SHADER_HEADER_INTEGER
      SHADER_READ_U8_INTEGER ("texture1", "") /* U8 */
      SHADER_WRITE_S16_INTEGER
      SHADER_CAST_S16_U8_INTEGER
      "void main (void) {\n"
      "  write_s16 (cast_s16_u8 (read_u8 ()) - 128);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_U8,
      "convert_u8_u8", SHADER_FLAG_USE_U8,
      SHADER_HEADER
      "uniform sampler2DRect texture1;\n" /* U8 */
      "void main (void) {\n"
      "  gl_FragColor = texture2DRect (texture1, gl_TexCoord[0].xy);\n"
      "}\n",
      SHADER_HEADER_INTEGER
      SHADER_READ_U8_INTEGER ("texture1", "") /* U8 */
      SHADER_WRITE_U8_INTEGER
      "void main (void) {\n"
      "  write_u8 (read_u8 ());\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_S16_S16,
      "convert_s16_s16", SHADER_FLAG_USE_S16,
      SHADER_HEADER
      "uniform sampler2DRect texture1;\n" /* S16 */
      "void main (void) {\n"
      "  gl_FragColor = texture2DRect (texture1, gl_TexCoord[0].xy);\n"
      "}\n",
      SHADER_HEADER_INTEGER
      SHADER_READ_S16_INTEGER ("texture1", "") /* S16 */
      SHADER_WRITE_S16_INTEGER
      "void main (void) {\n"
      "  write_s16 (read_s16 ());\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_Y4_YUYV,
      "convert_u8_y4_yuyv", SHADER_FLAG_USE_U8,
      SHADER_HEADER
      "uniform sampler2DRect texture1;\n" /* YUYV */
      "void main (void) {\n"
      /* round x coordinate down from texel center n.5 to texel edge n.0 */
      "  float x = floor (gl_TexCoord[0].x) / 2.0;\n"
      "  float y = gl_TexCoord[0].y;\n"
      /* shift x coordinate from texel edge n.0 to texel center n.5 */
      "  vec2 coordinate = vec2 (floor (x) + 0.5, y);\n"
      "  vec4 yuyv = texture2DRect (texture1, coordinate);\n"
      "  if (fract (x) < 0.25) {\n"
      "    gl_FragColor = vec4 (yuyv.r);\n"
      "  } else {\n"
      "    gl_FragColor = vec4 (yuyv.b);\n"
      "  }\n"
      "}\n",
      SHADER_HEADER_INTEGER
      SHADER_WRITE_U8_INTEGER
      "uniform usampler2DRect texture1;\n" /* YUYV */
      "void main (void) {\n"
      /* round x coordinate down from texel center n.5 to texel edge n.0 */
      "  float x = floor (gl_TexCoord[0].x) / 2.0;\n"
      "  float y = gl_TexCoord[0].y;\n"
      /* shift x coordinate from texel edge n.0 to texel center n.5 */
      "  vec2 coordinate = vec2 (floor (x) + 0.5, y);\n"
      "  uvec4 yuyv = texture2DRect (texture1, coordinate);\n"
      "  if (fract (x) < 0.25) {\n"
      "    write_u8 (yuyv.r);\n"
      "  } else {\n"
      "    write_u8 (yuyv.b);\n"
      "  }\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_U2_YUYV,
      "convert_u8_u2_yuyv", SHADER_FLAG_USE_U8,
      SHADER_HEADER
      "uniform sampler2DRect texture1;\n" /* YUYV */
      "void main (void) {\n"
      "  vec4 yuyv = texture2DRect (texture1, gl_TexCoord[0].xy);\n"
      "  gl_FragColor = vec4 (yuyv.g);\n"
      "}\n",
      SHADER_HEADER_INTEGER
      SHADER_WRITE_U8_INTEGER
      "uniform usampler2DRect texture1;\n" /* YUYV */
      "void main (void) {\n"
      "  uvec4 yuyv = texture2DRect (texture1, gl_TexCoord[0].xy);\n"
      "  write_u8 (yuyv.g);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_V2_YUYV,
      "convert_u8_v2_yuyv", SHADER_FLAG_USE_U8,
      SHADER_HEADER
      "uniform sampler2DRect texture1;\n" /* YUYV */
      "void main (void) {\n"
      "  vec4 yuyv = texture2DRect (texture1, gl_TexCoord[0].xy);\n"
      "  gl_FragColor = vec4 (yuyv.a);\n"
      "}\n",
      SHADER_HEADER_INTEGER
      SHADER_WRITE_U8_INTEGER
      "uniform usampler2DRect texture1;\n" /* YUYV */
      "void main (void) {\n"
      "  uvec4 yuyv = texture2DRect (texture1, gl_TexCoord[0].xy);\n"
      "  write_u8 (yuyv.a);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_Y4_UYVY,
      "convert_u8_y4_uyvy", SHADER_FLAG_USE_U8,
      SHADER_HEADER
      "uniform sampler2DRect texture1;\n" /* UYVY */
      "void main (void) {\n"
      /* round x coordinate down from texel center n.5 to texel edge n.0 */
      "  float x = floor (gl_TexCoord[0].x) / 2.0;\n"
      "  float y = gl_TexCoord[0].y;\n"
      /* shift x coordinate from texel edge n.0 to texel center n.5 */
      "  vec2 coordinate = vec2 (floor (x) + 0.5, y);\n"
      "  vec4 uyvy = texture2DRect (texture1, coordinate);\n"
      "  if (fract (x) < 0.25) {\n"
      "    gl_FragColor = vec4 (uyvy.g);\n"
      "  } else {\n"
      "    gl_FragColor = vec4 (uyvy.a);\n"
      "  }\n"
      "}\n",
      SHADER_HEADER_INTEGER
      SHADER_WRITE_U8_INTEGER
      "uniform usampler2DRect texture1;\n" /* UYVY */
      "void main (void) {\n"
      /* round x coordinate down from texel center n.5 to texel edge n.0 */
      "  float x = floor (gl_TexCoord[0].x) / 2.0;\n"
      "  float y = gl_TexCoord[0].y;\n"
      /* shift x coordinate from texel edge n.0 to texel center n.5 */
      "  vec2 coordinate = vec2 (floor (x) + 0.5, y);\n"
      "  uvec4 uyvy = texture2DRect (texture1, coordinate);\n"
      "  if (fract (x) < 0.25) {\n"
      "    write_u8 (uyvy.g);\n"
      "  } else {\n"
      "    write_u8 (uyvy.a);\n"
      "  }\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_U2_UYVY,
      "convert_u8_u2_uyvy", SHADER_FLAG_USE_U8,
      SHADER_HEADER
      "uniform sampler2DRect texture1;\n" /* UYVY */
      "void main (void) {\n"
      "  vec4 uyvy = texture2DRect (texture1, gl_TexCoord[0].xy);\n"
      "  gl_FragColor = vec4 (uyvy.r);\n"
      "}\n",
      SHADER_HEADER_INTEGER
      SHADER_WRITE_U8_INTEGER
      "uniform usampler2DRect texture1;\n" /* UYVY */
      "void main (void) {\n"
      "  uvec4 uyvy = texture2DRect (texture1, gl_TexCoord[0].xy);\n"
      "  write_u8 (uyvy.r);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_V2_UYVY,
      "convert_u8_v2_uyvy", SHADER_FLAG_USE_U8,
      SHADER_HEADER
      "uniform sampler2DRect texture1;\n" /* UYVY */
      "void main (void) {\n"
      "  vec4 uyvy = texture2DRect (texture1, gl_TexCoord[0].xy);\n"
      "  gl_FragColor = vec4 (uyvy.b);\n"
      "}\n",
      SHADER_HEADER_INTEGER
      SHADER_WRITE_U8_INTEGER
      "uniform usampler2DRect texture1;\n" /* UYVY */
      "void main (void) {\n"
      "  uvec4 uyvy = texture2DRect (texture1, gl_TexCoord[0].xy);\n"
      "  write_u8 (uyvy.b);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_Y4_AYUV,
      "convert_u8_y4_ayuv", SHADER_FLAG_USE_U8,
      SHADER_HEADER
      "uniform sampler2DRect texture1;\n" /* AYUV */
      "void main (void) {\n"
      "  vec4 ayuv = texture2DRect (texture1, gl_TexCoord[0].xy);\n"
      "  gl_FragColor = vec4 (ayuv.g);\n"
      "}\n",
      SHADER_HEADER_INTEGER
      SHADER_WRITE_U8_INTEGER
      "uniform usampler2DRect texture1;\n" /* AYUV */
      "void main (void) {\n"
      "  uvec4 ayuv = texture2DRect (texture1, gl_TexCoord[0].xy);\n"
      "  write_u8 (ayuv.g);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_U4_AYUV,
      "convert_u8_u4_ayuv", SHADER_FLAG_USE_U8,
      SHADER_HEADER
      "uniform sampler2DRect texture1;\n" /* AYUV */
      "void main (void) {\n"
      "  vec4 ayuv = texture2DRect (texture1, gl_TexCoord[0].xy);\n"
      "  gl_FragColor = vec4 (ayuv.b);\n"
      "}\n",
      SHADER_HEADER_INTEGER
      SHADER_WRITE_U8_INTEGER
      "uniform usampler2DRect texture1;\n" /* AYUV */
      "void main (void) {\n"
      "  uvec4 ayuv = texture2DRect (texture1, gl_TexCoord[0].xy);\n"
      "  write_u8 (ayuv.b);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_V4_AYUV,
      "convert_u8_v4_ayuv", SHADER_FLAG_USE_U8,
      SHADER_HEADER
      "uniform sampler2DRect texture1;\n" /* AYUV */
      "void main (void) {\n"
      "  vec4 ayuv = texture2DRect (texture1, gl_TexCoord[0].xy);\n"
      "  gl_FragColor = vec4 (ayuv.a);\n"
      "}\n",
      SHADER_HEADER_INTEGER
      SHADER_WRITE_U8_INTEGER
      "uniform usampler2DRect texture1;\n" /* AYUV */
      "void main (void) {\n"
      "  uvec4 ayuv = texture2DRect (texture1, gl_TexCoord[0].xy);\n"
      "  write_u8 (ayuv.a);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_YUYV_U8_422,
      "convert_yuyv_u8_422", SHADER_FLAG_USE_U8,
      SHADER_HEADER
      "uniform sampler2DRect texture1;\n" /* Y4 */
      "uniform sampler2DRect texture2;\n" /* U2 */
      "uniform sampler2DRect texture3;\n" /* V2 */
      "void main (void) {\n"
      "  vec4 yuyv;\n"
      /* round x coordinate down from texel center n.5 to texel edge n.0 */
      "  float x = floor (gl_TexCoord[0].x) * 2.0;\n"
      "  float y = gl_TexCoord[0].y;\n"
      /* shift x coordinate from texel edge n.0 to texel center n.5 */
      "  vec2 coordinate1 = vec2 (x + 0.5, y);\n"
      "  vec2 coordinate2 = vec2 (x + 1.5, y);\n"
      "  yuyv.r = texture2DRect (texture1, coordinate1).r;\n"
      "  yuyv.g = texture2DRect (texture2, gl_TexCoord[0].xy).r;\n"
      "  yuyv.b = texture2DRect (texture1, coordinate2).r;\n"
      "  yuyv.a = texture2DRect (texture3, gl_TexCoord[0].xy).r;\n"
      "  gl_FragColor = yuyv;\n"
      "}\n",
      SHADER_HEADER_INTEGER
      SHADER_WRITE_U8_VEC4_INTEGER
      "uniform usampler2DRect texture1;\n" /* Y4 */
      "uniform usampler2DRect texture2;\n" /* U2 */
      "uniform usampler2DRect texture3;\n" /* V2 */
      "void main (void) {\n"
      "  uvec4 yuyv;\n"
      /* round x coordinate down from texel center n.5 to texel edge n.0 */
      "  float x = floor (gl_TexCoord[0].x) * 2.0;\n"
      "  float y = gl_TexCoord[0].y;\n"
      /* shift x coordinate from texel edge n.0 to texel center n.5 */
      "  vec2 coordinate1 = vec2 (x + 0.5, y);\n"
      "  vec2 coordinate2 = vec2 (x + 1.5, y);\n"
      "  yuyv.r = texture2DRect (texture1, coordinate1).a;\n"
      "  yuyv.g = texture2DRect (texture2, gl_TexCoord[0].xy).a;\n"
      "  yuyv.b = texture2DRect (texture1, coordinate2).a;\n"
      "  yuyv.a = texture2DRect (texture3, gl_TexCoord[0].xy).a;\n"
      "  write_u8_vec4 (yuyv);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_UYVY_U8_422,
      "convert_uyvy_u8_422", SHADER_FLAG_USE_U8,
      SHADER_HEADER
      "uniform sampler2DRect texture1;\n" /* Y4 */
      "uniform sampler2DRect texture2;\n" /* U2 */
      "uniform sampler2DRect texture3;\n" /* V2 */
      "void main (void) {\n"
      "  vec4 uyvy;\n"
      /* round x coordinate down from texel center n.5 to n.0 and scale up to
         double width */
      "  float x = floor (gl_TexCoord[0].x) * 2.0;\n"
      "  float y = gl_TexCoord[0].y;\n"
      /* shift x coordinate from texel edge n.0 to texel center n.5 */
      "  vec2 coordinate1 = vec2 (x + 0.5, y);\n"
      "  vec2 coordinate2 = vec2 (x + 1.5, y);\n"
      "  uyvy.r = texture2DRect (texture2, gl_TexCoord[0].xy).r;\n"
      "  uyvy.g = texture2DRect (texture1, coordinate1).r;\n"
      "  uyvy.b = texture2DRect (texture3, gl_TexCoord[0].xy).r;\n"
      "  uyvy.a = texture2DRect (texture1, coordinate2).r;\n"
      "  gl_FragColor = uyvy;\n"
      "}\n",
      SHADER_HEADER_INTEGER
      SHADER_WRITE_U8_VEC4_INTEGER
      "uniform usampler2DRect texture1;\n" /* Y4 */
      "uniform usampler2DRect texture2;\n" /* U2 */
      "uniform usampler2DRect texture3;\n" /* V2 */
      "void main (void) {\n"
      "  uvec4 uyvy;\n"
      /* round x coordinate down from texel center n.5 to n.0 and scale up to
         double width */
      "  float x = floor (gl_TexCoord[0].x) * 2.0;\n"
      "  float y = gl_TexCoord[0].y;\n"
      /* shift x coordinate from texel edge n.0 to texel center n.5 */
      "  vec2 coordinate1 = vec2 (x + 0.5, y);\n"
      "  vec2 coordinate2 = vec2 (x + 1.5, y);\n"
      "  uyvy.r = texture2DRect (texture2, gl_TexCoord[0].xy).a;\n"
      "  uyvy.g = texture2DRect (texture1, coordinate1).a;\n"
      "  uyvy.b = texture2DRect (texture3, gl_TexCoord[0].xy).a;\n"
      "  uyvy.a = texture2DRect (texture1, coordinate2).a;\n"
      "  write_u8_vec4 (uyvy);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_AYUV_U8_444,
      "convert_ayuv_u8_444", SHADER_FLAG_USE_U8,
      SHADER_HEADER
      "uniform sampler2DRect texture1;\n" /* Y4 */
      "uniform sampler2DRect texture2;\n" /* U4 */
      "uniform sampler2DRect texture3;\n" /* V4 */
      "void main (void) {\n"
      "  vec4 ayuv;\n"
      "  ayuv.r = 1.0;\n"
      "  ayuv.g = texture2DRect (texture1, gl_TexCoord[0].xy).r;\n"
      "  ayuv.b = texture2DRect (texture2, gl_TexCoord[0].xy).r;\n"
      "  ayuv.a = texture2DRect (texture3, gl_TexCoord[0].xy).r;\n"
      "  gl_FragColor = ayuv;\n"
      "}\n",
      SHADER_HEADER_INTEGER
      SHADER_WRITE_U8_VEC4_INTEGER
      "uniform usampler2DRect texture1;\n" /* Y4 */
      "uniform usampler2DRect texture2;\n" /* U4 */
      "uniform usampler2DRect texture3;\n" /* V4 */
      "void main (void) {\n"
      "  uvec4 ayuv;\n"
      "  ayuv.r = 255u;\n"
      "  ayuv.g = texture2DRect (texture1, gl_TexCoord[0].xy).a;\n"
      "  ayuv.b = texture2DRect (texture2, gl_TexCoord[0].xy).a;\n"
      "  ayuv.a = texture2DRect (texture3, gl_TexCoord[0].xy).a;\n"
      "  write_u8_vec4 (ayuv);\n"
      "}\n" },
  /* FIXME: CPU overflows, GPU clamps, is this a problem? */
  { SCHRO_OPENGL_SHADER_ADD_S16_U8,
      "add_s16_u8", SHADER_FLAG_USE_U8 | SHADER_FLAG_USE_S16,
      SHADER_HEADER
      "uniform sampler2DRect texture1;\n" /* S16 */
      "uniform sampler2DRect texture2;\n" /* U8 */
      "const float scale = 255.0 / 65535.0;\n"
      "const float bias = -32768.0 / 65535.0;\n"
      "void main (void) {\n"
      /* bias from [-32768..32767] = [0..1] to [-32768..32767] ~= [-0.5..0.5]
         so that S16 zero maps to FP zero, otherwise S16 zero maps to FP ~0.5
         leading to S16 zero - S16 zero != S16 zero if calculation is done in
         FP space */
      "  vec4 a = texture2DRect (texture1, gl_TexCoord[0].xy) + bias;\n"
      /* scale from U8 [0..255] == [0..1] to S16 [..0..255..] ~= [0..0.004]
         so that both inputs from S16 and U8 are mapped equivalent to FP and
         U8 zero == S16 zero == FP zero holds */
      "  vec4 b = texture2DRect (texture2, gl_TexCoord[0].xy) * scale;\n"
      /* bias from [-32768..32767] ~= [-0.5..0.5] to [-32768..32767] = [0..1]
         to undo the initial bias */
      "  gl_FragColor = (a + b) - bias;\n"
      "}\n",
      SHADER_HEADER_INTEGER
      SHADER_READ_S16_INTEGER ("texture1", "_op1") /* S16 */
      SHADER_READ_U8_INTEGER ("texture2", "_op2") /* U8 */
      SHADER_WRITE_S16_INTEGER
      SHADER_CAST_S16_U8_INTEGER
      "void main (void) {\n"
      "  write_s16 (read_op1_s16 () + cast_s16_u8 (read_op2_u8 ()));\n" \
      "}\n" },
  /* FIXME: CPU overflows, GPU clamps, is this a problem? */
  { SCHRO_OPENGL_SHADER_ADD_S16_S16,
      "add_s16_s16", SHADER_FLAG_USE_S16,
      SHADER_HEADER
      "uniform sampler2DRect texture1;\n" /* S16 */
      "uniform sampler2DRect texture2;\n" /* S16 */
      "const float bias = -32768.0 / 65535.0;\n"
      "void main (void) {\n"
      /* bias from [-32768..32767] = [0..1] to [-32768..32767] ~= [-0.5..0.5]
         so that S16 zero maps to FP zero, otherwise S16 zero maps to FP ~0.5
         leading to S16 zero - S16 zero != S16 zero if calculation is done in
         FP space */
      "  vec4 a = texture2DRect (texture1, gl_TexCoord[0].xy) + bias;\n"
      "  vec4 b = texture2DRect (texture2, gl_TexCoord[0].xy) + bias;\n"
      /* bias from [-32768..32767] ~= [-0.5..0.5] to [-32768..32767] = [0..1]
         to undo the initial bias */
      "  gl_FragColor = (a + b) - bias;\n"
      "}\n",
      SHADER_HEADER_INTEGER
      SHADER_READ_S16_INTEGER ("texture1", "_op1") /* S16 */
      SHADER_READ_S16_INTEGER ("texture2", "_op2") /* S16 */
      SHADER_WRITE_S16_INTEGER
      "void main (void) {\n"
      "  write_s16 (read_op1_s16 () + read_op2_s16 ());\n"
      "}\n" },
  /* FIXME: CPU overflows, GPU clamps, is this a problem? */
  { SCHRO_OPENGL_SHADER_SUBTRACT_S16_U8,
      "subtract_s16_u8", SHADER_FLAG_USE_U8 | SHADER_FLAG_USE_S16,
      SHADER_HEADER
      "uniform sampler2DRect texture1;\n" /* S16 */
      "uniform sampler2DRect texture2;\n" /* U8 */
      "const float scale = 255.0 / 65535.0;\n"
      "const float bias = -32768.0 / 65535.0;\n"
      "void main (void) {\n"
      /* bias from [-32768..32767] == [0..1] to [-32768..32767] ~= [-0.5..0.5]
         so that S16 zero maps to FP zero, otherwise S16 zero maps to FP ~0.5
         leading to S16 zero - S16 zero != S16 zero if calculation is done in
         FP space */
      "  vec4 a = texture2DRect (texture1, gl_TexCoord[0].xy) + bias;\n"
      /* scale from U8 [0..255] == [0..1] to S16 [..0..255..] ~= [0..0.004]
         so that both inputs from S16 and U8 are mapped equivalent to FP and
         U8 zero == S16 zero == FP zero holds */
      "  vec4 b = texture2DRect (texture2, gl_TexCoord[0].xy) * scale;\n"
      /* bias from [-32768..32767] ~= [-0.5..0.5] to [-32768..32767] = [0..1]
         to undo the initial bias */
      "  gl_FragColor = (a - b) - bias;\n"
      "}\n",
      SHADER_HEADER_INTEGER
      SHADER_READ_S16_INTEGER ("texture1", "_op1") /* S16 */
      SHADER_READ_U8_INTEGER ("texture2", "_op2") /* U8 */
      SHADER_WRITE_S16_INTEGER
      SHADER_CAST_S16_U8_INTEGER
      "void main (void) {\n"
      "  write_s16 (read_op1_s16 () - cast_s16_u8 (read_op2_u8 ()));\n" \
      "}\n" },
  /* FIXME: CPU overflows, GPU clamps, is this a problem? */
  { SCHRO_OPENGL_SHADER_SUBTRACT_S16_S16,
      "subtract_s16_s16", SHADER_FLAG_USE_S16,
      SHADER_HEADER
      "uniform sampler2DRect texture1;\n" /* S16 */
      "uniform sampler2DRect texture2;\n" /* S16 */
      "const float bias = -32768.0 / 65535.0;\n"
      "void main (void) {\n"
      /* bias from [-32768..32767] == [0..1] to [-32768..32767] ~= [-0.5..0.5]
         so that S16 zero maps to FP zero, otherwise S16 zero maps to FP ~0.5
         leading to S16 zero - S16 zero != S16 zero if calculation is done in
         FP space */
      "  vec4 a = texture2DRect (texture1, gl_TexCoord[0].xy) + bias;\n"
      "  vec4 b = texture2DRect (texture2, gl_TexCoord[0].xy) + bias;\n"
      /* bias from [-32768..32767] ~= [-0.5..0.5] to [-32768..32767] == [0..1]
         to undo the initial bias */
      "  gl_FragColor = (a - b) - bias;\n"
      "}\n",
      SHADER_HEADER_INTEGER
      SHADER_READ_S16_INTEGER ("texture1", "_op1") /* S16 */
      SHADER_READ_S16_INTEGER ("texture2", "_op2") /* S16 */
      SHADER_WRITE_S16_INTEGER
      "void main (void) {\n"
      "  write_s16 (read_op1_s16 () - read_op2_s16 ());\n" \
      "}\n" },

  /* 1 = Deslauriers-Debuc (9,7)
     2 = LeGall (5,3)
     3 = Deslauriers-Debuc (13,7)
     4 = Haar 0/1
     5 = Fidelity
     6 = Daubechies (9,7)

     offset = height / 2

     +---------------+                read for...
     |               |
     |       L       |                L'            H'
     |               |
     |             o | A[2 * n - 6]   - - - - ? ?   - - - - ? ?
     |             o | A[2 * n - 4]   - - - - ? ?   - - - - ? ?
     |             o | A[2 * n - 2]   - - - - ? ?   o - o - ? ?
     |          /> X | A[2 * n    ]   = = = = ? ?   X X X X ? ?
     |         /   o | A[2 * n + 2]   - - - - ? ?   o o o - ? ?
     |         |   o | A[2 * n + 4]   - - - - ? ?   o - o - ? ?
     |         |   o | A[2 * n + 6]   - - - - ? ?   - - - - ? ?
     |  offset |   o | A[2 * n + 8]   - - - - ? ?   - - - - ? ?
     |         |     |
     +---------|-----+                1 2 3 4 5 6   1 2 3 4 5 6
     |         |     |
     |         |   o | A[2 * n - 7]   - - - - ? ?   - - - - ? ?
     |         |   o | A[2 * n - 5]   - - - - ? ?   - - - - ? ?
     |         |   o | A[2 * n - 3]   - - o - ? ?   - - - - ? ?
     |         \   o | A[2 * n - 1]   o o o - ? ?   - - - - ? ?
     |          \> X | A[2 * n + 1]   X X X X ? ?   = = = = ? ?
     |             o | A[2 * n + 3]   - - o - ? ?   - - - - ? ?
     |             o | A[2 * n + 5]   - - - - ? ?   - - - - ? ?
     |             o | A[2 * n + 7]   - - - - ? ?   - - - - ? ?
     |               |
     |       H       |
     |               |
     +---------------+ */

  { SCHRO_OPENGL_SHADER_IIWT_S16_FILTER_DESLAURIERS_DUBUC_9_7_Lp,
      "iiwt_s16_filter_deslauriers_dubuc_9_7_lp", SHADER_FLAG_USE_S16,
      SHADER_HEADER
      SHADER_READ_S16 ("texture1", "")
      SHADER_WRITE_S16
      SHADER_DIVIDE_S16
      /* distance between two corresponding texels from subbands L and H in
         texels = vec2 (width / 2.0, 0.0) or vec2 (0.0, height / 2.0) */
      "uniform vec2 offset;\n"
      "uniform vec2 one_decrease;\n"
      "float filter (float h1m, float h0) {\n"
      "  return divide_s16 (h1m + h0 + 2.0, 4.0);\n"
      "}\n"
      "void main (void) {\n"
      "  float l0 = read_s16 ();\n"                       /* A[2 ∗ n] */
      "  float h1m = read_s16 (offset - one_decrease);\n" /* A[2 ∗ n - 1] */
      "  float h0 = read_s16 (offset);\n"                 /* A[2 ∗ n + 1] */
      "  write_s16 (l0 - filter (h1m, h0));\n"
      "}\n",
      SHADER_HEADER_INTEGER
      SHADER_READ_S16_INTEGER ("texture1", "")
      SHADER_WRITE_S16_INTEGER
      SHADER_DIVIDE_S16_INTEGER
      /* distance between two corresponding texels from subbands L and H in
         texels = vec2 (width / 2.0, 0.0) or vec2 (0.0, height / 2.0) */
      "uniform vec2 offset;\n"
      "uniform vec2 one_decrease;\n"
      "int filter (int h1m, int h0) {\n"
      "  return divide_s16 (h1m + h0 + 2, 4);\n"
      "}\n"
      "void main (void) {\n"
      "  int l0 = read_s16 ();\n"                       /* A[2 ∗ n] */
      "  int h1m = read_s16 (offset - one_decrease);\n" /* A[2 ∗ n - 1] */
      "  int h0 = read_s16 (offset);\n"                 /* A[2 ∗ n + 1] */
      "  write_s16 (l0 - filter (h1m, h0));\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_IIWT_S16_FILTER_DESLAURIERS_DUBUC_9_7_Hp,
      "iiwt_s16_filter_deslauriers_dubuc_9_7_hp", SHADER_FLAG_USE_S16,
      SHADER_HEADER
      SHADER_READ_S16 ("texture1", "")
      SHADER_WRITE_S16
      SHADER_DIVIDE_S16
      /* distance between two corresponding texels from subbands L' and H in
         texels = vec2 (width / 2.0, 0.0) or vec2 (0.0, height / 2.0) */
      "uniform vec2 offset;\n"
      "uniform vec2 one_decrease;\n"
      "uniform vec2 one_increase;\n"
      "uniform vec2 two_increase;\n"
      "float filter (float l1m, float l0, float l1p, float l2p) {\n"
      "  return divide_s16 (-l1m + 9.0 * (l0 + l1p) - l2p + 8.0, 16.0);\n"
      "}\n"
      "void main (void) {\n"
      "  float l1m = read_s16 (-offset - one_decrease);\n" /* A[2 ∗ n - 2] */
      "  float l0 = read_s16 (-offset);\n"                 /* A[2 ∗ n] */
      "  float l1p = read_s16 (-offset + one_increase);\n" /* A[2 ∗ n + 2] */
      "  float l2p = read_s16 (-offset + two_increase);\n" /* A[2 ∗ n + 4] */
      "  float h0 = read_s16 ();\n"                        /* A[2 ∗ n + 1] */
      "  write_s16 (h0 + filter (l1m, l0, l1p, l2p));\n"
      "}\n",
      SHADER_HEADER_INTEGER
      SHADER_READ_S16_INTEGER ("texture1", "")
      SHADER_WRITE_S16_INTEGER
      SHADER_DIVIDE_S16_INTEGER
      /* distance between two corresponding texels from subbands L' and H in
         texels = vec2 (width / 2.0, 0.0) or vec2 (0.0, height / 2.0) */
      "uniform vec2 offset;\n"
      "uniform vec2 one_decrease;\n"
      "uniform vec2 one_increase;\n"
      "uniform vec2 two_increase;\n"
      "int filter (int l1m, int l0, int l1p, int l2p) {\n"
      "  return divide_s16 (-l1m + 9 * (l0 + l1p) - l2p + 8, 16);\n"
      "}\n"
      "void main (void) {\n"
      "  int l1m = read_s16 (-offset - one_decrease);\n" /* A[2 ∗ n - 2] */
      "  int l0 = read_s16 (-offset);\n"                 /* A[2 ∗ n] */
      "  int l1p = read_s16 (-offset + one_increase);\n" /* A[2 ∗ n + 2] */
      "  int l2p = read_s16 (-offset + two_increase);\n" /* A[2 ∗ n + 4] */
      "  int h0 = read_s16 ();\n"                        /* A[2 ∗ n + 1] */
      "  write_s16 (h0 + filter (l1m, l0, l1p, l2p));\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_IIWT_S16_FILTER_LE_GALL_5_3_Lp,
      "iiwt_s16_filter_le_gall_5_3_lp", SHADER_FLAG_USE_S16,
      SHADER_HEADER
      SHADER_READ_S16 ("texture1", "")
      SHADER_WRITE_S16
      SHADER_DIVIDE_S16
      /* distance between two corresponding texels from subbands L and H in
         texels = vec2 (width / 2.0, 0.0) or vec2 (0.0, height / 2.0) */
      "uniform vec2 offset;\n"
      "uniform vec2 one_decrease;\n"
      "float filter (float h1m, float h0) {\n"
      "  return divide_s16 (h1m + h0 + 2.0, 4.0);\n"
      "}\n"
      "void main (void) {\n"
      "  float l0 = read_s16 ();\n"                       /* A[2 ∗ n] */
      "  float h1m = read_s16 (offset - one_decrease);\n" /* A[2 ∗ n - 1] */
      "  float h0 = read_s16 (offset);\n"                 /* A[2 ∗ n + 1] */
      "  write_s16 (l0 - filter (h1m, h0));\n"
      "}\n",
      SHADER_HEADER_INTEGER
      SHADER_READ_S16_INTEGER ("texture1", "")
      SHADER_WRITE_S16_INTEGER
      SHADER_DIVIDE_S16_INTEGER
      /* distance between two corresponding texels from subbands L and H in
         texels = vec2 (width / 2.0, 0.0) or vec2 (0.0, height / 2.0) */
      "uniform vec2 offset;\n"
      "uniform vec2 one_decrease;\n"
      "int filter (int h1m, int h0) {\n"
      "  return divide_s16 (h1m + h0 + 2, 4);\n"
      "}\n"
      "void main (void) {\n"
      "  int l0 = read_s16 ();\n"                       /* A[2 ∗ n] */
      "  int h1m = read_s16 (offset - one_decrease);\n" /* A[2 ∗ n - 1] */
      "  int h0 = read_s16 (offset);\n"                 /* A[2 ∗ n + 1] */
      "  write_s16 (l0 - filter (h1m, h0));\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_IIWT_S16_FILTER_LE_GALL_5_3_Hp,
      "iiwt_s16_filter_le_gall_5_3_hp", SHADER_FLAG_USE_S16,
      SHADER_HEADER
      SHADER_READ_S16 ("texture1", "")
      SHADER_WRITE_S16
      SHADER_DIVIDE_S16
      /* distance between two corresponding texels from subbands L' and H in
         texels = vec2 (width / 2.0, 0.0) or vec2 (0.0, height / 2.0) */
      "uniform vec2 offset;\n"
      "uniform vec2 one_increase;\n"
      "float filter (float l0, float l1p) {\n"
      "  return divide_s16 (l0 + l1p + 1.0, 2.0);\n"
      "}\n"
      "void main (void) {\n"
      "  float l0 = read_s16 (-offset);\n"                 /* A[2 ∗ n] */
      "  float l1p = read_s16 (-offset + one_increase);\n" /* A[2 ∗ n + 2] */
      "  float h0 = read_s16 ();\n"                        /* A[2 ∗ n + 1] */
      "  write_s16 (h0 + filter (l0, l1p));\n"
      "}\n",
      SHADER_HEADER_INTEGER
      SHADER_READ_S16_INTEGER ("texture1", "")
      SHADER_WRITE_S16_INTEGER
      SHADER_DIVIDE_S16_INTEGER
      /* distance between two corresponding texels from subbands L' and H in
         texels = vec2 (width / 2.0, 0.0) or vec2 (0.0, height / 2.0) */
      "uniform vec2 offset;\n"
      "uniform vec2 one_increase;\n"
      "int filter (int l0, int l1p) {\n"
      "  return divide_s16 (l0 + l1p + 1, 2);\n"
      "}\n"
      "void main (void) {\n"
      "  int l0 = read_s16 (-offset);\n"                 /* A[2 ∗ n] */
      "  int l1p = read_s16 (-offset + one_increase);\n" /* A[2 ∗ n + 2] */
      "  int h0 = read_s16 ();\n"                        /* A[2 ∗ n + 1] */
      "  write_s16 (h0 + filter (l0, l1p));\n"
      "}\n" },
  /* FIXME: more than 1 level leads to errors */
  { SCHRO_OPENGL_SHADER_IIWT_S16_FILTER_DESLAURIERS_DUBUC_13_7_Lp,
      "iiwt_s16_filter_deslauriers_dubuc_13_7_lp", SHADER_FLAG_USE_S16,
      SHADER_HEADER
      SHADER_READ_S16 ("texture1", "")
      SHADER_WRITE_S16
      SHADER_DIVIDE_S16
      /* distance between two corresponding texels from subbands L and H in
         texels = vec2 (width / 2.0, 0.0) or vec2 (0.0, height / 2.0) */
      "uniform vec2 offset;\n"
      "uniform vec2 two_decrease;\n"
      "uniform vec2 one_decrease;\n"
      "uniform vec2 one_increase;\n"
      "float filter (float h2m, float h1m, float h0, float h1p) {\n"
      "  return divide_s16 (-h2m + 9.0 * (h1m + h0) - h1p + 16.0, 32.0);\n"
      "}\n"
      "void main (void) {\n"
      "  float l0 = read_s16 ();\n"                       /* A[2 ∗ n] */
      "  float h2m = read_s16 (offset - two_decrease);\n" /* A[2 ∗ n - 3] */
      "  float h1m = read_s16 (offset - one_decrease);\n" /* A[2 ∗ n - 1] */
      "  float h0 = read_s16 (offset);\n"                 /* A[2 ∗ n + 1] */
      "  float h1p = read_s16 (offset + one_increase);\n" /* A[2 ∗ n + 3] */
      "  write_s16 (l0 - filter (h2m, h1m, h0, h1p));\n"
      "}\n",
      SHADER_HEADER_INTEGER
      SHADER_READ_S16_INTEGER ("texture1", "")
      SHADER_WRITE_S16_INTEGER
      SHADER_DIVIDE_S16_INTEGER
      /* distance between two corresponding texels from subbands L and H in
         texels = vec2 (width / 2.0, 0.0) or vec2 (0.0, height / 2.0) */
      "uniform vec2 offset;\n"
      "uniform vec2 two_decrease;\n"
      "uniform vec2 one_decrease;\n"
      "uniform vec2 one_increase;\n"
      "int filter (int h2m, int h1m, int h0, int h1p) {\n"
      "  return divide_s16 (-h2m + 9 * (h1m + h0) - h1p + 16, 32);\n"
      "}\n"
      "void main (void) {\n"
      "  int l0 = read_s16 ();\n"                       /* A[2 ∗ n] */
      "  int h2m = read_s16 (offset - two_decrease);\n" /* A[2 ∗ n - 3] */
      "  int h1m = read_s16 (offset - one_decrease);\n" /* A[2 ∗ n - 1] */
      "  int h0 = read_s16 (offset);\n"                 /* A[2 ∗ n + 1] */
      "  int h1p = read_s16 (offset + one_increase);\n" /* A[2 ∗ n + 3] */
      "  write_s16 (l0 - filter (h2m, h1m, h0, h1p));\n"
      "}\n" },
  /* FIXME: more than 1 level leads to errors */
  { SCHRO_OPENGL_SHADER_IIWT_S16_FILTER_DESLAURIERS_DUBUC_13_7_Hp,
      "iiwt_s16_filter_deslauriers_dubuc_13_7_hp", SHADER_FLAG_USE_S16,
      SHADER_HEADER
      SHADER_READ_S16 ("texture1", "")
      SHADER_WRITE_S16
      SHADER_DIVIDE_S16
      /* distance between two corresponding texels from subbands L' and H in
         texels = vec2 (width / 2.0, 0.0) or vec2 (0.0, height / 2.0) */
      "uniform vec2 offset;\n"
      "uniform vec2 one_decrease;\n"
      "uniform vec2 one_increase;\n"
      "uniform vec2 two_increase;\n"
      "float filter (float l1m, float l0, float l1p, float l2p) {\n"
      "  return divide_s16 (-l1m + 9.0 * (l0 + l1p) - l2p + 8.0,16.0);\n"
      "}\n"
      "void main (void) {\n"
      "  float l1m = read_s16 (-offset - one_decrease);\n" /* A[2 ∗ n - 2] */
      "  float l0 = read_s16 (-offset);\n"                 /* A[2 ∗ n] */
      "  float l1p = read_s16 (-offset + one_increase);\n" /* A[2 ∗ n + 2] */
      "  float l2p = read_s16 (-offset + two_increase);\n" /* A[2 ∗ n + 4] */
      "  float h0 = read_s16 ();\n"                        /* A[2 ∗ n + 1] */
      "  write_s16 (h0 + filter (l1m, l0, l1p, l2p));\n"
      "}\n",
      SHADER_HEADER_INTEGER
      SHADER_READ_S16_INTEGER ("texture1", "")
      SHADER_WRITE_S16_INTEGER
      SHADER_DIVIDE_S16_INTEGER
      /* distance between two corresponding texels from subbands L' and H in
         texels = vec2 (width / 2.0, 0.0) or vec2 (0.0, height / 2.0) */
      "uniform vec2 offset;\n"
      "uniform vec2 one_decrease;\n"
      "uniform vec2 one_increase;\n"
      "uniform vec2 two_increase;\n"
      "int filter (int l1m, int l0, int l1p, int l2p) {\n"
      "  return divide_s16 (-l1m + 9 * (l0 + l1p) - l2p + 8, 16);\n"
      "}\n"
      "void main (void) {\n"
      "  int l1m = read_s16 (-offset - one_decrease);\n" /* A[2 ∗ n - 2] */
      "  int l0 = read_s16 (-offset);\n"                 /* A[2 ∗ n] */
      "  int l1p = read_s16 (-offset + one_increase);\n" /* A[2 ∗ n + 2] */
      "  int l2p = read_s16 (-offset + two_increase);\n" /* A[2 ∗ n + 4] */
      "  int h0 = read_s16 ();\n"                        /* A[2 ∗ n + 1] */
      "  write_s16 (h0 + filter (l1m, l0, l1p, l2p));\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_IIWT_S16_FILTER_HAAR_Lp,
      "iiwt_s16_filter_haar_lp", SHADER_FLAG_USE_S16,
      SHADER_HEADER
      SHADER_READ_S16 ("texture1", "")
      SHADER_WRITE_S16
      SHADER_DIVIDE_S16
      /* distance between two corresponding texels from subbands L and H in
         texels = vec2 (width / 2.0, 0.0) or vec2 (0.0, height / 2.0) */
      "uniform vec2 offset;\n"
      "float filter (float h0) {\n"
      "  return divide_s16 (h0 + 1.0, 2.0);\n"
      "}\n"
      "void main (void) {\n"
      "  float l0 = read_s16 ();\n"       /* A[2 ∗ n] */
      "  float h0 = read_s16 (offset);\n" /* A[2 ∗ n + 1] */
      "  write_s16 (l0 - filter (h0));\n"
      "}\n",
      SHADER_HEADER_INTEGER
      SHADER_READ_S16_INTEGER ("texture1", "")
      SHADER_WRITE_S16_INTEGER
      SHADER_DIVIDE_S16_INTEGER
      /* distance between two corresponding texels from subbands L and H in
         texels = vec2 (width / 2.0, 0.0) or vec2 (0.0, height / 2.0) */
      "uniform vec2 offset;\n"
      "int filter (int h0) {\n"
      "  return divide_s16 (h0 + 1, 2);\n"
      "}\n"
      "void main (void) {\n"
      "  int l0 = read_s16 ();\n"       /* A[2 ∗ n] */
      "  int h0 = read_s16 (offset);\n" /* A[2 ∗ n + 1] */
      "  write_s16 (l0 - filter (h0));\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_IIWT_S16_FILTER_HAAR_Hp,
      "iiwt_s16_filter_haar_hp", SHADER_FLAG_USE_S16,
      SHADER_HEADER
      SHADER_READ_S16 ("texture1", "")
      SHADER_WRITE_S16
      /* distance between two corresponding texels from subbands L' and H in
         texels = vec2 (width / 2.0, 0.0) or vec2 (0.0, height / 2.0) */
      "uniform vec2 offset;\n"
      "void main (void) {\n"
      "  float l0 = read_s16 (-offset);\n" /* A[2 ∗ n] */
      "  float h0 = read_s16 ();\n"        /* A[2 ∗ n + 1] */
      "  write_s16 (h0 + l0);\n"
      "}\n",
      SHADER_HEADER_INTEGER
      SHADER_READ_S16_INTEGER ("texture1", "")
      SHADER_WRITE_S16_INTEGER
      /* distance between two corresponding texels from subbands L' and H in
         texels = vec2 (width / 2.0, 0.0) or vec2 (0.0, height / 2.0) */
      "uniform vec2 offset;\n"
      "void main (void) {\n"
      "  int l0 = read_s16 (-offset);\n" /* A[2 ∗ n] */
      "  int h0 = read_s16 ();\n"        /* A[2 ∗ n + 1] */
      "  write_s16 (h0 + l0);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_IIWT_S16_VERTICAL_DEINTERLEAVE_L,
      "iiwt_s16_vertical_deinterleave_l", SHADER_FLAG_USE_S16,
      SHADER_HEADER
      "uniform sampler2DRect texture1;\n"
      "void main (void) {\n"
      "  float x = gl_TexCoord[0].x;\n"
      /* round y coordinate down from texel center n.5 to texel edge n.0 */
      "  float y = floor (gl_TexCoord[0].y);\n"
      /* scale y coordinate to the destination coordinate and shift it from
         texel edge n.0 to texel center n.5 */
      "  vec2 coordinate = vec2 (x, y * 2.0 + 0.5);\n"
      "  gl_FragColor = texture2DRect (texture1, coordinate);\n"
      "}\n",
      SHADER_HEADER_INTEGER
      SHADER_WRITE_U16_INTEGER
      "uniform isampler2DRect texture1;\n"
      "void main (void) {\n"
      "  float x = gl_TexCoord[0].x;\n"
      /* round y coordinate down from texel center n.5 to texel edge n.0 */
      "  float y = floor (gl_TexCoord[0].y);\n"
      /* scale y coordinate to the destination coordinate and shift it from
         texel edge n.0 to texel center n.5 */
      "  vec2 coordinate = vec2 (x, y * 2.0 + 0.5);\n"
      "  write_u16 (texture2DRect (texture1, coordinate).a);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_IIWT_S16_VERTICAL_DEINTERLEAVE_H,
      "iiwt_s16_vertical_deinterleave_h", SHADER_FLAG_USE_S16,
      SHADER_HEADER
      "uniform sampler2DRect texture1;\n"
      /* height of subband XL */
      "uniform vec2 offset;\n" /* = vec2 (0.0, height / 2.0) */
      "void main (void) {\n"
      "  float x = gl_TexCoord[0].x;\n"
      /* round y coordinate down from texel center n.5 to texel edge n.0 */
      "  float y = floor (gl_TexCoord[0].y) - offset.y;\n"
      /* scale y coordinate to the destination coordinate and shift it from
         texel edge n.0 to texel center n.5 */
      "  vec2 coordinate = vec2 (x, y * 2.0 + 1.5);\n"
      "  gl_FragColor = texture2DRect (texture1, coordinate);\n"
      "}\n",
      SHADER_HEADER_INTEGER
      SHADER_WRITE_U16_INTEGER
      "uniform isampler2DRect texture1;\n"
      /* height of subband XL */
      "uniform vec2 offset;\n" /* = vec2 (0.0, height / 2.0) */
      "void main (void) {\n"
      "  float x = gl_TexCoord[0].x;\n"
      /* round y coordinate down from texel center n.5 to texel edge n.0 */
      "  float y = floor (gl_TexCoord[0].y) - offset.y;\n"
      /* scale y coordinate to the destination coordinate and shift it from
         texel edge n.0 to texel center n.5 */
      "  vec2 coordinate = vec2 (x, y * 2.0 + 1.5);\n"
      "  write_u16 (texture2DRect (texture1, coordinate).a);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_IIWT_S16_VERTICAL_INTERLEAVE,
      "iiwt_s16_vertical_interleave", SHADER_FLAG_USE_S16,
      SHADER_HEADER
      "uniform sampler2DRect texture1;\n"
      /* vertical distance between two corresponding texels from subband XL
         and XH in texels */
      "uniform vec2 offset;\n" /* = vec2 (0.0, height / 2.0) */
      "void main (void) {\n"
      "  float x = gl_TexCoord[0].x;\n"
      /* round y coordinate down from texel center n.5 to texel edge n.0 */
      "  float y = floor (gl_TexCoord[0].y);\n"
      "  if (mod (y, 2.0) < 0.5) {\n"
      "    y = floor (y / 2.0);\n"
      "  } else {\n"
      "    y = floor (y / 2.0) + offset.y;\n"
      "  }\n"
      /* shift y coordinate from texel edge n.0 to texel center n.5 */
      "  vec2 coordinate = vec2 (x, y + 0.5);\n"
      "  gl_FragColor = texture2DRect (texture1, coordinate);\n"
      "}\n",
      SHADER_HEADER_INTEGER
      SHADER_WRITE_U16_INTEGER
      "uniform isampler2DRect texture1;\n"
      /* vertical distance between two corresponding texels from subband XL
         and XH in texels */
      "uniform vec2 offset;\n" /* = vec2 (0.0, height / 2.0) */
      "void main (void) {\n"
      "  float x = gl_TexCoord[0].x;\n"
      /* round y coordinate down from texel center n.5 to texel edge n.0 */
      "  float y = floor (gl_TexCoord[0].y);\n"
      "  if (mod (y, 2.0) < 0.5) {\n"
      "    y = floor (y / 2.0);\n"
      "  } else {\n"
      "    y = floor (y / 2.0) + offset.y;\n"
      "  }\n"
      /* shift y coordinate from texel edge n.0 to texel center n.5 */
      "  vec2 coordinate = vec2 (x, y + 0.5);\n"
      "  write_u16 (texture2DRect (texture1, coordinate).a);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_IIWT_S16_HORIZONTAL_INTERLEAVE,
      "iiwt_s16_horizontal_interleave", SHADER_FLAG_USE_S16,
      SHADER_HEADER
      "uniform sampler2DRect texture1;\n"
      /* horizontal distance between two corresponding texels from subband L'
         and H' in texels */
      "uniform vec2 offset;\n" /* = vec2 (width / 2.0, 0.0) */
      "void main (void) {\n"
      /* round x coordinate down from texel center n.5 to texel edge n.0 */
      "  float x = floor (gl_TexCoord[0].x);\n"
      "  float y = gl_TexCoord[0].y;\n"
      "  if (mod (x, 2.0) < 0.5) {\n"
      "    x = floor (x / 2.0);\n"
      "  } else {\n"
      "    x = floor (x / 2.0) + offset.x;\n"
      "  }\n"
      /* shift y coordinate from texel edge n.0 to texel center n.5 */
      "  vec2 coordinate = vec2 (x + 0.5, y);\n"
      "  gl_FragColor = texture2DRect (texture1, coordinate);\n"
      "}\n",
      SHADER_HEADER_INTEGER
      SHADER_WRITE_U16_INTEGER
      "uniform isampler2DRect texture1;\n"
      /* horizontal distance between two corresponding texels from subband L'
         and H' in texels */
      "uniform vec2 offset;\n" /* = vec2 (width / 2.0, 0.0) */
      "void main (void) {\n"
      /* round x coordinate down from texel center n.5 to texel edge n.0 */
      "  float x = floor (gl_TexCoord[0].x);\n"
      "  float y = gl_TexCoord[0].y;\n"
      "  if (mod (x, 2.0) < 0.5) {\n"
      "    x = floor (x / 2.0);\n"
      "  } else {\n"
      "    x = floor (x / 2.0) + offset.x;\n"
      "  }\n"
      /* shift y coordinate from texel edge n.0 to texel center n.5 */
      "  vec2 coordinate = vec2 (x + 0.5, y);\n"
      "  write_u16 (texture2DRect (texture1, coordinate).a);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_IIWT_S16_FILTER_SHIFT,
      "iiwt_s16_filter_shift", SHADER_FLAG_USE_S16,
      SHADER_HEADER
      SHADER_READ_S16 ("texture1", "")
      SHADER_WRITE_S16
      SHADER_RSHIFT (1.0, 2.0)
      "void main (void) {\n"
      "  write_s16 (rshift (read_s16 ()));\n"
      "}\n",
      SHADER_HEADER_INTEGER
      SHADER_READ_S16_INTEGER ("texture1", "")
      SHADER_WRITE_S16_INTEGER
      SHADER_RSHIFT_S16_INTEGER (1, 2)
      "void main (void) {\n"
      "  write_s16 (rshift_s16 (read_s16 ()));\n"
      "}\n" },

  { SCHRO_OPENGL_SHADER_UPSAMPLE_U8,
      "upsample_u8", SHADER_FLAG_USE_U8,
      SHADER_HEADER
      SHADER_READ_U8 ("texture1", "")
      SHADER_WRITE_U8
      SHADER_DIVIDE_S16
      "uniform vec2 three_decrease;\n"
      "uniform vec2 two_decrease;\n"
      "uniform vec2 one_decrease;\n"
      "uniform vec2 one_increase;\n"
      "uniform vec2 two_increase;\n"
      "uniform vec2 three_increase;\n"
      "uniform vec2 four_increase;\n"
      "void main (void) {\n"
      "  float s3m = read_u8 (-three_decrease);\n" /* S[n - 3] */
      "  float s2m = read_u8 (-two_decrease);\n"   /* S[n - 2] */
      "  float s1m = read_u8 (-one_decrease);\n"   /* S[n - 1] */
      "  float s0 = read_u8 ();\n"                 /* S[n] */
      "  float s1p = read_u8 (one_increase);\n"    /* S[n + 1] */
      "  float s2p = read_u8 (two_increase);\n"    /* S[n + 2] */
      "  float s3p = read_u8 (three_increase);\n"  /* S[n + 3] */
      "  float s4p = read_u8 (four_increase);\n"   /* S[n + 4] */
      "  float sum = divide_s16 (-s3m + 3.0 * s2m - 7.0 * s1m + 21.0 * s0\n"
      "      + 21.0 * s1p - 7.0 * s2p + 3.0 * s3p - s4p + 16.0, 32.0);\n"
      "  write_u8 (clamp (sum, 0.0, 255.0));\n"
      "}\n",
      SHADER_HEADER_INTEGER
      SHADER_READ_U8_INTEGER ("texture1", "")
      SHADER_WRITE_U8_INTEGER
      SHADER_CAST_U8_S16_INTEGER
      SHADER_CAST_S16_U8_INTEGER
      SHADER_DIVIDE_S16_INTEGER
      "uniform vec2 three_decrease;\n"
      "uniform vec2 two_decrease;\n"
      "uniform vec2 one_decrease;\n"
      "uniform vec2 one_increase;\n"
      "uniform vec2 two_increase;\n"
      "uniform vec2 three_increase;\n"
      "uniform vec2 four_increase;\n"
      "void main (void) {\n"
      "  int s3m = cast_s16_u8 (read_u8 (-three_decrease));\n" /* S[n - 3] */
      "  int s2m = cast_s16_u8 (read_u8 (-two_decrease));\n"   /* S[n - 2] */
      "  int s1m = cast_s16_u8 (read_u8 (-one_decrease));\n"   /* S[n - 1] */
      "  int s0 = cast_s16_u8 (read_u8 ());\n"                 /* S[n] */
      "  int s1p = cast_s16_u8 (read_u8 (one_increase));\n"    /* S[n + 1] */
      "  int s2p = cast_s16_u8 (read_u8 (two_increase));\n"    /* S[n + 2] */
      "  int s3p = cast_s16_u8 (read_u8 (three_increase));\n"  /* S[n + 3] */
      "  int s4p = cast_s16_u8 (read_u8 (four_increase));\n"   /* S[n + 4] */
      "  int sum =  divide_s16 (-s3m + 3 * s2m - 7 * s1m + 21 * s0\n"
      "      + 21 * s1p - 7 * s2p + 3 * s3p - s4p + 16, 32);\n"
      "  write_u8 (cast_u8_s16 (sum));\n"
      "}\n" },

  { SCHRO_OPENGL_SHADER_MC_OBMC_WEIGHT,
      "mc_obmc_weight", SHADER_FLAG_USE_S16,
      SHADER_HEADER
      SHADER_WRITE_S16
      SHADER_DIVIDE_S16
      "uniform vec2 size;\n" /* block size */
      "uniform vec2 offset;\n" /* block offset */
      "float ramp (float coordinate, float offset) {\n"
      "  if (offset == 1.0) {\n"
      "    if (coordinate == 0.0) {\n"
      "      return 3.0;\n"
      "    }\n"
      "    return 5.0;\n"
      "  }\n"
      "  return 1.0 + divide_s16 (6.0 * coordinate + offset - 1.0,\n"
      "      2.0 * offset - 1.0);\n"
      "}\n"
      "float obmc_weight (float coordinate, float size, float offset) {\n"
      "  if (offset == 0.0) {\n"
      "    return 8.0;\n"
      "  } else if (coordinate < 2.0 * offset) {\n"
      "    return ramp (coordinate, offset);\n"
      "  } else if (size - 1.0 - coordinate < 2.0 * offset) {\n"
      "    return ramp (size - 1.0 - coordinate, offset);\n"
      "  }\n"
      "  return 8.0;\n"
      "}\n"
      "void main (void) {\n"
      /* round coordinate down from texel center n.5 to texel edge n.0 */
      "  vec2 coordinate = floor (gl_TexCoord[0].xy);\n"
      "  write_s16 (obmc_weight (coordinate.x, size.x, offset.x)\n"
      "      * obmc_weight (coordinate.y, size.y, offset.y));\n"
      "}\n",
      SHADER_HEADER_INTEGER
      SHADER_WRITE_S16_INTEGER
      SHADER_DIVIDE_S16_INTEGER
      "uniform vec2 size;\n" /* block size */ // FIXME: bind as int
      "uniform vec2 offset;\n" /* block offset */ // FIXME: bind as int
      "int ramp (int coordinate, int offset) {\n"
      "  if (offset == 1) {\n"
      "    if (coordinate == 0) {\n"
      "      return 3;\n"
      "    }\n"
      "    return 5;\n"
      "  }\n"
      "  return 1 + divide_s16 (6 * coordinate + offset - 1, 2 * offset - 1);\n"
      "}\n"
      "int obmc_weight (int coordinate, int size, int offset) {\n"
      "  if (offset == 0) {\n"
      "    return 8;\n"
      "  } else if (coordinate < 2 * offset) {\n"
      "    return ramp (coordinate, offset);\n"
      "  } else if (size - 1 - coordinate < 2 * offset) {\n"
      "    return ramp (size - 1 - coordinate, offset);\n"
      "  }\n"
      "  return 8;\n"
      "}\n"
      "void main (void) {\n"
      /* round coordinate down from texel center n.5 to texel edge n.0 */
      "  vec2 coordinate = floor (gl_TexCoord[0].xy);\n"
      "  write_s16 (obmc_weight (int (coordinate.x), int (size.x), int (offset.x))\n"
      "      * obmc_weight (int (coordinate.y), int (size.y), int (offset.y)));\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_MC_CLEAR,
      "mc_clear", SHADER_FLAG_USE_S16,
      SHADER_HEADER
      SHADER_WRITE_S16
      "void main (void) {\n"
      "  write_s16 (0.0);\n"
      "}\n",
      SHADER_HEADER_INTEGER
      SHADER_WRITE_S16_INTEGER
      "void main (void) {\n"
      "  write_s16 (0);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_MC_RENDER_DC,
      "mc_render_dc", SHADER_FLAG_USE_U8 | SHADER_FLAG_USE_S16,
      SHADER_HEADER
      SHADER_READ_S16 ("texture1", "_previous") /* previous to blend with */
      SHADER_READ_S16 ("texture2", "_obmc_weight") /* obmc weight */
      SHADER_WRITE_S16
      "uniform vec2 origin;\n" /* block origin */
      "uniform float dc;\n"
      "void main (void) {\n"
      "  float previous = read_previous_s16 ();\n"
      "  float obmc_weight = read_obmc_weight_s16 (-origin);\n"
      "  write_s16 (previous + dc * obmc_weight);\n"
      "}\n",
      SHADER_HEADER_INTEGER
      SHADER_READ_S16_INTEGER ("texture1", "_previous") /* previous to blend with */
      SHADER_READ_S16_INTEGER ("texture2", "_obmc_weight") /* obmc weight */
      SHADER_WRITE_S16_INTEGER
      "uniform vec2 origin;\n" /* block origin */
      "uniform float dc;\n" // FIXME: bind as int
      "void main (void) {\n"
      "  int previous = read_previous_s16 ();\n"
      "  int obmc_weight = read_obmc_weight_s16 (-origin);\n"
      "  write_s16 (previous + int (dc) * obmc_weight);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_MC_RENDER_REF_PREC_0,
      "mc_render_ref_prec_0", SHADER_FLAG_USE_U8 | SHADER_FLAG_USE_S16,
      SHADER_HEADER
      SHADER_READ_S16 ("texture1", "_previous") /* previous to blend with */
      SHADER_READ_S16 ("texture2", "_obmc_weight") /* obmc weight */
      SHADER_READ_U8 ("texture3", "_sub0") /* upsampled sub frame 0 */
      SHADER_WRITE_S16
      "uniform vec2 offset;\n"
      "uniform vec2 origin;\n" /* block origin */
      "void main (void) {\n"
      "  float previous = read_previous_s16 ();\n"
      "  float obmc_weight = read_obmc_weight_s16 (-origin);\n"
      "  float sub0 = read_sub0_u8 (offset);\n"
      "  write_s16 (previous + sub0 * obmc_weight);\n"
      "}\n",
      SHADER_HEADER_INTEGER
      SHADER_READ_S16_INTEGER ("texture1", "_previous") /* previous to blend with */
      SHADER_READ_S16_INTEGER ("texture2", "_obmc_weight") /* obmc weight */
      SHADER_READ_U8_INTEGER ("texture3", "_sub0") /* upsampled sub frame 0 */
      SHADER_WRITE_S16_INTEGER
      SHADER_CAST_S16_U8_INTEGER
      "uniform vec2 offset;\n"
      "uniform vec2 origin;\n" /* block origin */
      "void main (void) {\n"
      "  int previous = read_previous_s16 ();\n"
      "  int obmc_weight = read_obmc_weight_s16 (-origin);\n"
      "  int sub0 = cast_s16_u8 (read_sub0_u8 (offset));\n"
      "  write_s16 (previous + sub0 * obmc_weight);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_MC_RENDER_REF_PREC_0_WEIGHT,
      "mc_render_ref_prec_0_weight", SHADER_FLAG_USE_U8 | SHADER_FLAG_USE_S16,
      SHADER_HEADER
      SHADER_READ_S16 ("texture1", "_previous") /* previous to blend with */
      SHADER_READ_S16 ("texture2", "_obmc_weight") /* obmc weight */
      SHADER_READ_U8 ("texture3", "_sub0") /* upsampled sub frame 0 */
      SHADER_WRITE_S16
      SHADER_DIVIDE_S16
      "uniform vec2 offset;\n"
      "uniform vec2 origin;\n" /* block origin */
      "uniform float weight;\n"
      "uniform float addend;\n" /* 1 << (shift - 1) */
      "uniform float divisor;\n" /* 1 << shift */
      "void main (void) {\n"
      "  float previous = read_previous_s16 ();\n"
      "  float obmc_weight = read_obmc_weight_s16 (-origin);\n"
      "  float sub0 = read_sub0_u8 (offset);\n"
      "  write_s16 (previous + divide_s16 (sub0 * weight + addend, divisor)\n"
      "      * obmc_weight);\n"
      "}\n",
      SHADER_HEADER_INTEGER
      SHADER_READ_S16_INTEGER ("texture1", "_previous") /* previous to blend with */
      SHADER_READ_S16_INTEGER ("texture2", "_obmc_weight") /* obmc weight */
      SHADER_READ_U8_INTEGER ("texture3", "_sub0") /* upsampled sub frame 0 */
      SHADER_WRITE_S16_INTEGER
      SHADER_CAST_S16_U8_INTEGER
      SHADER_DIVIDE_S16_INTEGER
      "uniform vec2 offset;\n"
      "uniform vec2 origin;\n" /* block origin */
      "uniform float weight;\n" // FIXME: bind as int
      "uniform float addend;\n" /* 1 << (shift - 1) */ // FIXME: bind as int
      "uniform float divisor;\n" /* 1 << shift */ // FIXME: bind as int
      "void main (void) {\n"
      "  int previous = read_previous_s16 ();\n"
      "  int obmc_weight = read_obmc_weight_s16 (-origin);\n"
      "  int sub0 = cast_s16_u8 (read_sub0_u8 (offset));\n"
      "  write_s16 (previous + divide_s16 ((sub0 * int (weight) + int (addend)),\n"
      "      int (divisor)) * obmc_weight);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_MC_RENDER_REF_PREC_1,
      "mc_render_ref_prec_1", SHADER_FLAG_USE_U8 | SHADER_FLAG_USE_S16,
      SHADER_HEADER
      SHADER_READ_S16 ("texture1", "_obmc_weight") /* obmc weights */
      SHADER_READ_U8 ("texture2", "_subx") /* upsampled sub frame x */
      SHADER_WRITE_S16
      "uniform vec2 offset;\n"
      "uniform vec2 origin;\n" /* block origin */
      "void main (void) {\n"
      "  float obmc_weight = read_obmc_weight_s16 (-origin);\n"
      "  float subx = read_subx_u8 (offset);\n"
      "  write_s16 (subx * obmc_weight);\n"
      /* FIXME: ref weighting */
      "}\n",
      "" },
  { SCHRO_OPENGL_SHADER_MC_RENDER_REF_PREC_3,
      "mc_render_ref_prec_3", SHADER_FLAG_USE_U8 | SHADER_FLAG_USE_S16,
      SHADER_HEADER
      "void main (void) {\n"
      "}\n",
      "" },
  { SCHRO_OPENGL_SHADER_MC_RENDER_BIREF,
      "mc_render_biref", SHADER_FLAG_USE_U8 | SHADER_FLAG_USE_S16,
      SHADER_HEADER
      "void main (void) {\n"
      "}\n",
      "" },

  { SCHRO_OPENGL_SHADER_MC_SHIFT,
      "mc_shift", SHADER_FLAG_USE_S16,
      SHADER_HEADER
      SHADER_READ_S16 ("texture1", "")
      SHADER_WRITE_S16
      SHADER_RSHIFT (-8160.0, 64.0)
      "void main (void) {\n"
      "  write_s16 (rshift (read_s16 ()));\n"
      "}\n",
      SHADER_HEADER_INTEGER
      SHADER_READ_S16_INTEGER ("texture1", "")
      SHADER_WRITE_S16_INTEGER
      SHADER_RSHIFT_S16_INTEGER (-8160, 64)
      "void main (void) {\n"
      "  write_s16 (rshift_s16 (read_s16 ()));\n"
      "}\n" },

  { -1, NULL }
};

struct _SchroOpenGLShaderLibrary {
  SchroOpenGL *opengl;
  SchroOpenGLShader *shaders[SCHRO_OPENGL_SHADER_COUNT];
};

SchroOpenGLShaderLibrary *
schro_opengl_shader_library_new (SchroOpenGL *opengl)
{
  SchroOpenGLShaderLibrary* library
      = schro_malloc0 (sizeof (SchroOpenGLShaderLibrary));

  library->opengl = opengl;

  return library;
}

void
schro_opengl_shader_library_free (SchroOpenGLShaderLibrary *library)
{
  int i;

  SCHRO_ASSERT (library != NULL);

  schro_opengl_lock (library->opengl);

  for (i = 0; i < ARRAY_SIZE (library->shaders); ++i) {
    if (library->shaders[i]) {
      schro_opengl_shader_free (library->shaders[i]);
    }
  }

  schro_opengl_unlock (library->opengl);

  schro_free (library);
}

SchroOpenGLShader *
schro_opengl_shader_get (SchroOpenGL *opengl, int index)
{
  SchroOpenGLShaderLibrary* shader_library;

  SCHRO_ASSERT (index >= 0);
  SCHRO_ASSERT (index < SCHRO_OPENGL_SHADER_COUNT);

  shader_library = schro_opengl_get_shader_library (opengl);

  if (!shader_library->shaders[index]) {
    schro_opengl_lock (opengl);

    if ((SCHRO_OPENGL_CANVAS_IS_FLAG_SET (STORE_U8_AS_UI8) &&
        schro_opengl_shader_code_list[index].flags & SHADER_FLAG_USE_U8) ||
        (SCHRO_OPENGL_CANVAS_IS_FLAG_SET (STORE_S16_AS_UI16) &&
        schro_opengl_shader_code_list[index].flags & SHADER_FLAG_USE_S16) ||
        (SCHRO_OPENGL_CANVAS_IS_FLAG_SET (STORE_S16_AS_U16) &&
        schro_opengl_shader_code_list[index].flags & SHADER_FLAG_USE_S16)) {
      shader_library->shaders[index]
          = schro_opengl_shader_new (schro_opengl_shader_code_list[index].code_integer,
          schro_opengl_shader_code_list[index].name);
    } else {
      shader_library->shaders[index]
          = schro_opengl_shader_new (schro_opengl_shader_code_list[index].code,
          schro_opengl_shader_code_list[index].name);
    }

    schro_opengl_unlock (opengl);
  }

  return shader_library->shaders[index];
}

