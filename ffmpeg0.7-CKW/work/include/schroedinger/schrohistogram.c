
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <string.h>
#include <math.h>
#include <schroedinger/schrohistogram.h>


static int
ilogx (int x)
{
  int i = 0;
  if (x < 0)
    x = -x;
  while (x >= 2 << SCHRO_HISTOGRAM_SHIFT) {
    x >>= 1;
    i++;
  }
  return x + (i << SCHRO_HISTOGRAM_SHIFT);
}

static int
iexpx (int x)
{
  if (x < (1 << SCHRO_HISTOGRAM_SHIFT))
    return x;

  return ((1 << SCHRO_HISTOGRAM_SHIFT) | (x & ((1 << SCHRO_HISTOGRAM_SHIFT) -
              1))) << ((x >> SCHRO_HISTOGRAM_SHIFT) - 1);
}

static int
ilogx_size (int i)
{
  if (i < (1 << SCHRO_HISTOGRAM_SHIFT))
    return 1;
  return 1 << ((i >> SCHRO_HISTOGRAM_SHIFT) - 1);
}

double
schro_histogram_get_range (SchroHistogram * hist, int start, int end)
{
  int i;
  int iend;
  int size;
  double x;

  if (start >= end)
    return 0;

  i = ilogx (start);
  size = ilogx_size (i);
  x = (double) (iexpx (i + 1) - start) / size * hist->bins[i];

  i++;
  iend = ilogx (end);
  while (i <= iend) {
    x += hist->bins[i];
    i++;
  }

  size = ilogx_size (iend);
  x -= (double) (iexpx (iend + 1) - end) / size * hist->bins[iend];

  return x;
}

void
schro_histogram_table_generate (SchroHistogramTable * table,
    double (*func) (int value, void *priv), void *priv)
{
  int i;
  int j;

  for (i = 0; i < SCHRO_HISTOGRAM_SIZE; i++) {
    int jmin, jmax;
    double sum;

    jmin = iexpx (i);
    jmax = iexpx (i + 1);

    sum = 0;
    for (j = jmin; j < jmax; j++) {
      sum += func (j, priv);
    }
    table->weights[i] = sum / ilogx_size (i);
  }
}

double
schro_histogram_apply_table (SchroHistogram * hist, SchroHistogramTable * table)
{
  int i;
  double sum;

  sum = 0;
  for (i = 0; i < SCHRO_HISTOGRAM_SIZE; i++) {
    sum += hist->bins[i] * table->weights[i];
  }

  return sum;
}

#ifdef unused
double
schro_histogram_apply_table_range (SchroHistogram * hist,
    SchroHistogramTable * table, int start, int end)
{
  int i;
  int iend;
  int size;
  double sum;

  if (start >= end)
    return 0;

  i = ilogx (start);
  size = ilogx_size (i);
  sum =
      (double) (iexpx (i + 1) -
      start) / size * hist->bins[i] * table->weights[i];

  i++;
  iend = ilogx (end);
  while (i <= iend) {
    sum += hist->bins[i] * table->weights[i];
    i++;
  }

  size = ilogx_size (iend);
  sum -=
      (double) (iexpx (iend + 1) -
      end) / size * hist->bins[iend] * table->weights[iend];

  return sum;
}
#endif


void
schro_histogram_init (SchroHistogram * hist)
{
  memset (hist, 0, sizeof (*hist));
}

void
schro_histogram_add (SchroHistogram * hist, int value)
{
  hist->bins[ilogx (value)]++;
  hist->n++;
}

void
schro_histogram_add_array_s16 (SchroHistogram * hist, int16_t * src, int n)
{
  int i;
  for (i = 0; i < n; i++) {
    hist->bins[ilogx (src[i])]++;
  }
  hist->n += n;
}

void
schro_histogram_scale (SchroHistogram * hist, double scale)
{
  int i;
  for (i = 0; i < SCHRO_HISTOGRAM_SIZE; i++) {
    hist->bins[i] *= scale;
  }
  hist->n *= scale;
}

#ifdef unused
static double
pow2 (int i, void *priv)
{
  return i * i;
}

double
schro_histogram_estimate_noise_level (SchroHistogram * hist)
{
  static SchroHistogramTable table;
  static int table_inited;
  int i;
  int j;
  int n;
  double sigma;

  if (!table_inited) {
    schro_histogram_table_generate (&table, pow2, NULL);
    table_inited = TRUE;
  }

  sigma = sqrt (schro_histogram_apply_table (hist, &table) / hist->n);
  SCHRO_DEBUG ("sigma %g", sigma);
  for (i = 0; i < 5; i++) {
    j = ceil (sigma * 2.0);
    n = schro_histogram_get_range (hist, 0, j);
    sigma = 1.14 *
        sqrt (schro_histogram_apply_table_range (hist, &table, 0, j) / n);
    SCHRO_DEBUG ("sigma %g (%d)", sigma, j);
  }
  SCHRO_DEBUG ("sigma %g n %d", sigma, n);

  return sigma;
}
#endif

double
schro_histogram_estimate_slope (SchroHistogram * hist)
{
  int i;
  int n;
  double x, y;
  double m_x;
  double m_y;
  double m_xx;
  double m_xy;
  double ave_x, ave_y;
  double slope, y0;

  m_x = 0;
  m_y = 0;
  m_xx = 0;
  m_xy = 0;
  n = 0;
  for (i = 0; i < SCHRO_HISTOGRAM_SIZE; i++) {
    if (i > 0 && hist->bins[i] > 0) {
      x = sqrt (iexpx (i));
      y = log (hist->bins[i] / ilogx_size (i));
      m_x += x;
      m_y += y;
      m_xy += x * y;
      m_xx += x * x;
      n++;
    }
  }

  ave_x = m_x / n;
  ave_y = m_y / n;

  slope = (n * m_xy - m_x * m_y) / (n * m_xx - m_x * m_x);
  y0 = ave_y - slope * ave_x;

  SCHRO_DEBUG ("n %d slope %g y0 %g", n, slope, y0);

#if 0
  for (i = 0; i < SCHRO_HISTOGRAM_SIZE; i++) {
    //if (hist->bins[i]/ilogx_size(i) >= hist->n*0.0003 &&
    //    hist->bins[i]/ilogx_size(i) < hist->n*0.03) {
    if (i > 0 && hist->bins[i] > 0) {
      x = sqrt (iexpx (i));
      y = y0 + slope * x;
      hist->bins[i] = exp (y) * ilogx_size (i);
    }
  }
#endif

  return slope;
}


double
schro_histogram_estimate_entropy (SchroHistogram * hist, int quant_index,
    int noarith)
{
#define N 12
  double estimated_entropy = 0;
  double bin[N];
  int quant_factor;
  int i;
  double post5;

  quant_factor = schro_table_quant[quant_index];

  bin[0] = schro_histogram_get_range (hist, 0, 32000);
  for (i = 0; i < N; i++) {
    bin[i] =
        schro_histogram_get_range (hist,
        (quant_factor * ((1 << i) - 1) + 3) / 4, 32000);
  }

  if (!noarith) {
    double ones, zeros;

    /* entropy of sign bit */
    estimated_entropy += bin[1];

    /* entropy of continue bits */
    estimated_entropy += schro_utils_entropy (bin[1], bin[0]);
    estimated_entropy += schro_utils_entropy (bin[2], bin[1]);
    estimated_entropy += schro_utils_entropy (bin[3], bin[2]);
    estimated_entropy += schro_utils_entropy (bin[4], bin[3]);
    estimated_entropy += schro_utils_entropy (bin[5], bin[4]);

    post5 = 0;
    for (i = 6; i < N; i++) {
      post5 += bin[i];
    }
    estimated_entropy += schro_utils_entropy (post5, post5 + bin[5]);

    /* data entropy */
    ones = schro_histogram_apply_table (hist,
        (SchroHistogramTable
            *) (schro_table_onebits_hist_shift3_1_2[quant_index]));
    zeros =
        schro_histogram_apply_table (hist,
        (SchroHistogramTable
            *) (schro_table_zerobits_hist_shift3_1_2[quant_index]));

    estimated_entropy += schro_utils_entropy (ones, zeros + ones);
  } else {
    double x;

    /* When the proportion of 0's gets very large, codeblocks are
     * skipped, dropping the contribution from bin[0].  This is a
     * gross hack estimate. */

    /* proportion of non-zero coefficients */
    x = (double) bin[1] / bin[0];

    /* probability that a codeblock is entirely zero.  25 (5x5) is the
     * size of the codeblocks created by init_small_codeblocks, and 0.5
     * is a magic factor */
    x = 1.0 - exp (-0.5 * 25 * x);

    /* entropy of first continue bit */
    estimated_entropy += x * bin[0] + (1 - x) * bin[1];

    /* entropy of sign bit */
    estimated_entropy += bin[1];

    /* entropy of continue and data bits */
    for (i = 1; i < N; i++) {
      estimated_entropy += 2 * bin[i];
    }
  }

  return estimated_entropy;
}

void
schro_frame_data_generate_histogram (SchroFrameData * fd,
    SchroHistogram * hist, int skip)
{
  int j;

  schro_histogram_init (hist);
  for (j = 0; j < fd->height; j += skip) {
    schro_histogram_add_array_s16 (hist,
        SCHRO_FRAME_DATA_GET_LINE (fd, j), fd->width);
  }
  schro_histogram_scale (hist, skip);
}

void
schro_frame_data_generate_histogram_dc_predict (SchroFrameData * fd,
    SchroHistogram * hist, int skip, int x, int y)
{
  int i, j;
  int16_t *prev_line;
  int16_t *line;

  schro_histogram_init (hist);
  for (j = 0; j < fd->height; j += skip) {
    prev_line = SCHRO_FRAME_DATA_GET_LINE (fd, j - 1);
    line = SCHRO_FRAME_DATA_GET_LINE (fd, j);
    for (i = 0; i < fd->width; i++) {
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
      schro_histogram_add (hist, line[i] - pred_value);
    }
  }
  schro_histogram_scale (hist, skip);
}
