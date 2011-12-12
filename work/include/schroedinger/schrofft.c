
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <string.h>
#include <math.h>
#include <schroedinger/schrofft.h>

void
schro_fft_generate_tables_f32 (float *costable, float *sintable, int shift)
{
  int i;
  int n = 1 << shift;
  double x = 2 * M_PI / n;

  for (i = 0; i < n; i++) {
    costable[i] = cos (x * i);
    sintable[i] = sin (x * i);
  }
}

#define COMPLEX_MULT_R(a,b,c,d) ((a)*(c) - (b)*(d))
#define COMPLEX_MULT_I(a,b,c,d) ((a)*(d) + (b)*(c))

static void
fft_stage (float *d1, float *d2, const float *s1, const float *s2,
    const float *costable, const float *sintable, int i, int shift)
{
  int j;
  int k;
  float x, y;
  int skip;
  int half_n;
  int offset;

  half_n = 1 << i;
  skip = 1 << (shift - i - 1);
  for (j = 0; j < skip; j++) {
    for (k = 0; k < half_n; k++) {
      offset = 2 * k * skip;
      x = COMPLEX_MULT_R (s1[offset + skip + j], s2[offset + skip + j],
          costable[k * skip], sintable[k * skip]);
      y = COMPLEX_MULT_I (s1[offset + skip + j], s2[offset + skip + j],
          costable[k * skip], sintable[k * skip]);

      d1[k * skip + j] = s1[offset + j] + x;
      d2[k * skip + j] = s2[offset + j] + y;
      d1[k * skip + half_n * skip + j] = s1[offset + j] - x;
      d2[k * skip + half_n * skip + j] = s2[offset + j] - y;
    }
  }
}

void
schro_fft_fwd_f32 (float *d_real, float *d_imag, const float *s_real,
    const float *s_imag, const float *costable, const float *sintable,
    int shift)
{
  int i;
  int n = 1 << shift;
  float *tmp;
  float *tmp1_1, *tmp1_2, *tmp2_1, *tmp2_2;

  tmp = schro_malloc (4 * sizeof (float) * n);
  tmp1_1 = tmp;
  tmp1_2 = tmp + n;
  tmp2_1 = tmp + 2 * n;
  tmp2_2 = tmp + 3 * n;

  i = 0;
  fft_stage (tmp1_1, tmp1_2, s_real, s_imag, costable, sintable, i, shift);
  for (i = 1; i < shift - 2; i += 2) {
    fft_stage (tmp2_1, tmp2_2, tmp1_1, tmp1_2, costable, sintable, i, shift);
    fft_stage (tmp1_1, tmp1_2, tmp2_1, tmp2_2, costable, sintable, i + 1,
        shift);
  }
  if (i < shift - 1) {
    fft_stage (tmp2_1, tmp2_2, tmp1_1, tmp1_2, costable, sintable, i, shift);
    fft_stage (d_real, d_imag, tmp2_1, tmp2_2, costable, sintable, i + 1,
        shift);
  } else {
    fft_stage (d_real, d_imag, tmp1_1, tmp1_2, costable, sintable, i, shift);
  }

  schro_free (tmp);
}

void
schro_fft_rev_f32 (float *d_real, float *d_imag, const float *s_real,
    const float *s_imag, const float *costable, const float *sintable,
    int shift)
{
  schro_fft_fwd_f32 (d_imag, d_real, s_imag, s_real, costable, sintable, shift);
}
