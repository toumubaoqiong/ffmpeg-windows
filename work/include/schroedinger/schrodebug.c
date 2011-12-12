
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <stdarg.h>
#include <stdio.h>

static const char *schro_debug_level_names[] = {
  "NONE",
  "ERROR",
  "WARNING",
  "INFO",
  "DEBUG",
  "LOG"
};

static int schro_debug_level = SCHRO_LEVEL_ERROR;
int _schro_dump_enable;

static void schro_debug_log_valist (int level, const char *file,
    const char *func, int line, const char *format, va_list varargs);

static SchroDebugLogFunc _schro_debug_log_func = schro_debug_log_valist;

static void
schro_debug_log_valist (int level, const char *file, const char *func,
    int line, const char *format, va_list varargs)
{
#ifdef HAVE_GLIB
  char *s;

  if (level > schro_debug_level)
    return;

  s = g_strdup_vprintf (format, varargs);

  fprintf (stderr, "SCHRO: %s: %s(%d): %s: %s\n",
      schro_debug_level_names[level], file, line, func, s);
  g_free (s);
#else
  char s[4096];

  if (level > schro_debug_level)
    return;

  vsnprintf (s, sizeof (s) - 1, format, varargs);

  fprintf (stderr, "SCHRO: %s: %s(%d): %s: %s\n",
      schro_debug_level_names[level], file, line, func, s);
#endif
}

void
schro_debug_log (int level, const char *file, const char *func,
    int line, const char *format, ...)
{
  va_list var_args;

  va_start (var_args, format);
  _schro_debug_log_func (level, file, func, line, format, var_args);
  va_end (var_args);
}

void
schro_debug_set_level (int level)
{
  schro_debug_level = level;
}

int
schro_debug_get_level (void)
{
  return schro_debug_level;
}


static FILE *dump_files[SCHRO_DUMP_LAST];

static const char *dump_file_names[SCHRO_DUMP_LAST] = {
  "schro_dump.subband_curve",
  "schro_dump.subband_est",
  "schro_dump.picture",
  "schro_dump.psnr",
  "schro_dump.ssim",
  "schro_dump.lambda_curve",
  "schro_dump.hist_test",
  "schro_dump.scene_change",
  "schro_dump.phase_correlation",
  "schro_dump.motionest"
};

void
schro_dump (int type, const char *format, ...)
{
  va_list varargs;

  if (!_schro_dump_enable)
    return;

  if (dump_files[type] == NULL) {
    dump_files[type] = fopen (dump_file_names[type], "w");
  }

  va_start (varargs, format);
  vfprintf (dump_files[type], format, varargs);
  va_end (varargs);

  fflush (dump_files[type]);
}

void
schro_debug_set_log_function (SchroDebugLogFunc func)
{
  if (func) {
    _schro_debug_log_func = func;
  } else {
    _schro_debug_log_func = schro_debug_log_valist;
  }
}
