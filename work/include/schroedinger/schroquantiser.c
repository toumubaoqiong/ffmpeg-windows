
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <schroedinger/schrohistogram.h>
#include <string.h>
#include <math.h>

//#define DUMP_SUBBAND_CURVES

void schro_encoder_choose_quantisers_simple (SchroEncoderFrame * frame);
void schro_encoder_choose_quantisers_rdo_bit_allocation (SchroEncoderFrame *
    frame);
void schro_encoder_choose_quantisers_lossless (SchroEncoderFrame * frame);
void schro_encoder_choose_quantisers_lowdelay (SchroEncoderFrame * frame);
void schro_encoder_choose_quantisers_rdo_lambda (SchroEncoderFrame * frame);
void schro_encoder_choose_quantisers_rdo_cbr (SchroEncoderFrame * frame);
void schro_encoder_choose_quantisers_constant_error (SchroEncoderFrame * frame);

double schro_encoder_entropy_to_lambda (SchroEncoderFrame * frame,
    double entropy);
static double schro_encoder_lambda_to_entropy (SchroEncoderFrame * frame,
    double lambda);
static void schro_encoder_generate_subband_histograms (SchroEncoderFrame *
    frame);


static int schro_subband_pick_quant (SchroEncoderFrame * frame,
    int component, int i, double lambda);

#define CURVE_SIZE 128

double
schro_encoder_perceptual_weight_moo (double cpd)
{
  /* I pretty much pulled this out of my ass (ppd = pixels per degree) */
  if (cpd < 4)
    return 1;
  return 0.68 * cpd * exp (-0.25 * cpd);
}

double
schro_encoder_perceptual_weight_constant (double cpd)
{
  return 1;
}

double
schro_encoder_perceptual_weight_ccir959 (double cpd)
{
  double w;
  w = 0.255 * pow (1 + 0.2561 * cpd * cpd, -0.75);

  /* return normalized value */
  return w / 0.255;
}

double
schro_encoder_perceptual_weight_manos_sakrison (double cpd)
{
  double w;

  if (cpd < 4)
    return 1;
  w = 2.6 * (0.0192 + 0.114 * cpd) * exp (-pow (0.114 * cpd, 1.1));

  /* return normalized value */
  return w / 0.980779694777866;
}

static double
weighted_sum (const float *h1, const float *v1, double *weight)
{
  int i, j;
  double sum;
  double rowsum;

  sum = 0;
  for (j = 0; j < CURVE_SIZE; j++) {
    rowsum = 0;
    for (i = 0; i < CURVE_SIZE; i++) {
      rowsum += h1[i] * v1[j] * weight[CURVE_SIZE * j + i];
    }
    sum += rowsum;
  }
  return sum;
}

static double
dot_product (const float *h1, const float *v1, const float *h2,
    const float *v2, double *weight)
{
  int i, j;
  double sum;
  double rowsum;

  sum = 0;
  for (j = 0; j < CURVE_SIZE; j++) {
    rowsum = 0;
    for (i = 0; i < CURVE_SIZE; i++) {
      rowsum +=
          h1[i] * v1[j] * h2[i] * v2[j] * weight[CURVE_SIZE * j +
          i] * weight[CURVE_SIZE * j + i];
    }
    sum += rowsum;
  }
  return sum;
}

static void
solve (double *matrix, double *column, int n)
{
  int i;
  int j;
  int k;
  double x;

  for (i = 0; i < n; i++) {
    x = 1 / matrix[i * n + i];
    for (k = i; k < n; k++) {
      matrix[i * n + k] *= x;
    }
    column[i] *= x;

    for (j = i + 1; j < n; j++) {
      x = matrix[j * n + i];
      for (k = i; k < n; k++) {
        matrix[j * n + k] -= matrix[i * n + k] * x;
      }
      column[j] -= column[i] * x;
    }
  }

  for (i = n - 1; i > 0; i--) {
    for (j = i - 1; j >= 0; j--) {
      column[j] -= matrix[j * n + i] * column[i];
      matrix[j * n + i] = 0;
    }
  }
}

#ifdef unused
void
schro_encoder_set_default_subband_weights (SchroEncoder * encoder)
{
  schro_encoder_calculate_subband_weights (encoder,
      schro_encoder_perceptual_weight_constant);
}
#endif

void
schro_encoder_calculate_subband_weights (SchroEncoder * encoder,
    double (*perceptual_weight) (double))
{
  int wavelet;
  int n_levels;
  double *matrix_intra;
  double *matrix_inter;
  int n;
  int i, j;
  double column_intra[SCHRO_LIMIT_SUBBANDS];
  double column_inter[SCHRO_LIMIT_SUBBANDS];
  double *weight_intra;
  double *weight_inter;

  matrix_intra =
      schro_malloc (sizeof (double) * SCHRO_LIMIT_SUBBANDS *
      SCHRO_LIMIT_SUBBANDS);
  matrix_inter =
      schro_malloc (sizeof (double) * SCHRO_LIMIT_SUBBANDS *
      SCHRO_LIMIT_SUBBANDS);
  weight_intra = schro_malloc (sizeof (double) * CURVE_SIZE * CURVE_SIZE);
  weight_inter = schro_malloc (sizeof (double) * CURVE_SIZE * CURVE_SIZE);

  for (j = 0; j < CURVE_SIZE; j++) {
    for (i = 0; i < CURVE_SIZE; i++) {
      double fv_intra =
          j * encoder->cycles_per_degree_vert * (1.0 / CURVE_SIZE);
      double fh_intra =
          i * encoder->cycles_per_degree_horiz * (1.0 / CURVE_SIZE);

      double fv_inter =
          j * encoder->cycles_per_degree_vert * (1.0 / CURVE_SIZE) *
          encoder->magic_inter_cpd_scale;
      double fh_inter =
          i * encoder->cycles_per_degree_horiz * (1.0 / CURVE_SIZE) *
          encoder->magic_inter_cpd_scale;

      weight_intra[j * CURVE_SIZE + i] =
          perceptual_weight (sqrt (fv_intra * fv_intra + fh_intra * fh_intra));
      weight_inter[j * CURVE_SIZE + i] =
          perceptual_weight (sqrt (fv_intra * fv_inter + fh_inter * fh_inter));
    }
  }

  for (wavelet = 0; wavelet < SCHRO_N_WAVELETS; wavelet++) {
    for (n_levels = 1; n_levels <= SCHRO_LIMIT_ENCODER_TRANSFORM_DEPTH;
        n_levels++) {
      const float *h_curve[SCHRO_LIMIT_SUBBANDS];
      const float *v_curve[SCHRO_LIMIT_SUBBANDS];
      int hi[SCHRO_LIMIT_SUBBANDS];
      int vi[SCHRO_LIMIT_SUBBANDS];

      n = 3 * n_levels + 1;

      for (i = 0; i < n; i++) {
        int position = schro_subband_get_position (i);
        int n_transforms;

        n_transforms = n_levels - SCHRO_SUBBAND_SHIFT (position);
        if (position & 1) {
          hi[i] = (n_transforms - 1) * 2;
        } else {
          hi[i] = (n_transforms - 1) * 2 + 1;
        }
        if (position & 2) {
          vi[i] = (n_transforms - 1) * 2;
        } else {
          vi[i] = (n_transforms - 1) * 2 + 1;
        }
        h_curve[i] = schro_tables_wavelet_noise_curve[wavelet][hi[i]];
        v_curve[i] = schro_tables_wavelet_noise_curve[wavelet][vi[i]];
      }

      if (0) {
        for (i = 0; i < n; i++) {
          column_intra[i] = weighted_sum (h_curve[i], v_curve[i], weight_intra);
          column_inter[i] = weighted_sum (h_curve[i], v_curve[i], weight_inter);
          matrix_intra[i * n + i] = dot_product (h_curve[i], v_curve[i],
              h_curve[i], v_curve[i], weight_intra);
          matrix_inter[i * n + i] = dot_product (h_curve[i], v_curve[i],
              h_curve[i], v_curve[i], weight_inter);
          for (j = i + 1; j < n; j++) {
            matrix_intra[i * n + j] = dot_product (h_curve[i], v_curve[i],
                h_curve[j], v_curve[j], weight_intra);
            matrix_intra[j * n + i] = matrix_intra[i * n + j];
            matrix_inter[i * n + j] = dot_product (h_curve[i], v_curve[i],
                h_curve[j], v_curve[j], weight_inter);
            matrix_inter[j * n + i] = matrix_inter[i * n + j];
          }
        }

        solve (matrix_intra, column_intra, n);
        solve (matrix_inter, column_inter, n);

        for (i = 0; i < n; i++) {
          if (column_intra[i] < 0 || column_inter[i] < 0) {
            SCHRO_ERROR ("BROKEN wavelet %d n_levels %d", wavelet, n_levels);
            break;
          }
        }

        SCHRO_DEBUG ("wavelet %d n_levels %d", wavelet, n_levels);
        for (i = 0; i < n; i++) {
          SCHRO_DEBUG ("%g", 1.0 / sqrt (column_intra[i]));
          SCHRO_DEBUG ("%g", 1.0 / sqrt (column_inter[i]));
          encoder->intra_subband_weights[wavelet][n_levels - 1][i] =
              sqrt (column_intra[i]);
          encoder->inter_subband_weights[wavelet][n_levels - 1][i] =
              sqrt (column_inter[i]);
        }
      } else {
        for (i = 0; i < n; i++) {
          int position = schro_subband_get_position (i);
          int n_transforms;
          double size;

          n_transforms = n_levels - SCHRO_SUBBAND_SHIFT (position);
          size = (1.0 / CURVE_SIZE) * (1 << n_transforms);
          encoder->intra_subband_weights[wavelet][n_levels - 1][i] =
              1.0 / (size * sqrt (weighted_sum (h_curve[i], v_curve[i],
                      weight_intra)));
          encoder->inter_subband_weights[wavelet][n_levels - 1][i] =
              1.0 / (size * sqrt (weighted_sum (h_curve[i], v_curve[i],
                      weight_inter)));
        }
      }
    }
  }

  schro_free (weight_intra);
  schro_free (matrix_intra);
  schro_free (weight_inter);
  schro_free (matrix_inter);
}

void
schro_encoder_choose_quantisers (SchroEncoderFrame * frame)
{
  switch (frame->encoder->quantiser_engine) {
    case SCHRO_QUANTISER_ENGINE_SIMPLE:
      schro_encoder_choose_quantisers_simple (frame);
      break;
    case SCHRO_QUANTISER_ENGINE_RDO_BIT_ALLOCATION:
      schro_encoder_choose_quantisers_rdo_bit_allocation (frame);
      break;
    case SCHRO_QUANTISER_ENGINE_CBR:
      schro_encoder_choose_quantisers_rdo_cbr (frame);
      break;
    case SCHRO_QUANTISER_ENGINE_LOSSLESS:
      schro_encoder_choose_quantisers_lossless (frame);
      break;
    case SCHRO_QUANTISER_ENGINE_LOWDELAY:
      schro_encoder_choose_quantisers_lowdelay (frame);
      break;
    case SCHRO_QUANTISER_ENGINE_RDO_LAMBDA:
      schro_encoder_choose_quantisers_rdo_lambda (frame);
      break;
    case SCHRO_QUANTISER_ENGINE_CONSTANT_ERROR:
      schro_encoder_choose_quantisers_constant_error (frame);
      break;
    default:
      SCHRO_ASSERT (0);
  }
}

void
schro_encoder_choose_quantisers_lossless (SchroEncoderFrame * frame)
{
  int i;
  int component;

  for (component = 0; component < 3; component++) {
    for (i = 0; i < 1 + 3 * frame->params.transform_depth; i++) {
      schro_encoder_frame_set_quant_index (frame, component, i, -1, -1, 0);
    }
  }
}

void
schro_encoder_choose_quantisers_simple (SchroEncoderFrame * frame)
{
  SchroParams *params = &frame->params;
  int i;
  int component;
  double noise_amplitude;
  double a;
  double max;
  double *table;

  noise_amplitude = 255.0 * pow (0.1, frame->encoder->noise_threshold * 0.05);
  SCHRO_DEBUG ("noise %g", noise_amplitude);

  if (frame->num_refs == 0) {
    table = frame->encoder->intra_subband_weights[params->wavelet_filter_index]
        [MAX (0, params->transform_depth - 1)];
  } else {
    table = frame->encoder->inter_subband_weights[params->wavelet_filter_index]
        [MAX (0, params->transform_depth - 1)];
  }

  for (component = 0; component < 3; component++) {
    for (i = 0; i < 1 + 3 * params->transform_depth; i++) {
      a = noise_amplitude * table[i];

      schro_encoder_frame_set_quant_index (frame, component, i, -1, -1,
          schro_utils_multiplier_to_quant_index (a));
    }
  }

  max = 1.0;

  for (i = 0; i < 1 + 3 * params->transform_depth; i++) {
    params->quant_matrix[i] =
        schro_utils_multiplier_to_quant_index (max / table[i]);
    SCHRO_DEBUG ("%g %g %d", table[i], max / table[i], params->quant_matrix[i]);
  }
}

void
schro_encoder_choose_quantisers_lowdelay (SchroEncoderFrame * frame)
{
  SchroParams *params = &frame->params;
  int i;
  int component;
  int base;
  const int *table;

  /* completely made up */
  base = 12 + (30 - frame->encoder->noise_threshold) / 2;

  table = schro_tables_lowdelay_quants[params->wavelet_filter_index]
      [MAX (0, params->transform_depth - 1)];

  for (component = 0; component < 3; component++) {
    schro_encoder_frame_set_quant_index (frame, component, 0, -1, -1,
        base - table[0]);

    for (i = 0; i < params->transform_depth; i++) {
      schro_encoder_frame_set_quant_index (frame, component, 1 + 3 * i + 0, -1,
          -1, base - table[1 + 2 * i + 0]);
      schro_encoder_frame_set_quant_index (frame, component, 1 + 3 * i + 1, -1,
          -1, base - table[1 + 2 * i + 0]);
      schro_encoder_frame_set_quant_index (frame, component, 1 + 3 * i + 2, -1,
          -1, base - table[1 + 2 * i + 1]);
    }
  }

}

#ifdef DUMP_SUBBAND_CURVES
static double
pow2 (double x)
{
  return x * x;
}
#endif

#ifdef DUMP_SUBBAND_CURVES
static double
measure_error_subband (SchroEncoderFrame * frame, int component, int index,
    int quant_index)
{
  int i;
  int j;
  int16_t *line;
  int skip = 1;
  double error = 0;
  int q;
  int quant_factor;
  int quant_offset;
  int value;
  int position;
  SchroFrameData fd;

  position = schro_subband_get_position (index);
  schro_subband_get_frame_data (&fd, frame->iwt_frame, component, position,
      &frame->params);

  quant_factor = schro_table_quant[quant_index];
  if (frame->params.num_refs > 0) {
    quant_offset = schro_table_offset_3_8[quant_index];
  } else {
    quant_offset = schro_table_offset_1_2[quant_index];
  }

  error = 0;
  if (index == 0) {
    for (j = 0; j < fd.height; j += skip) {
      line = SCHRO_FRAME_DATA_GET_LINE (&fd, j);
      for (i = 1; i < fd.width; i += skip) {
        q = schro_quantise (abs (line[i] - line[i - 1]), quant_factor,
            quant_offset);
        value = schro_dequantise (q, quant_factor, quant_offset);
        error += pow2 (value - abs (line[i] - line[i - 1]));
      }
    }
  } else {
    for (j = 0; j < fd.height; j += skip) {
      line = SCHRO_FRAME_DATA_GET_LINE (&fd, j);
      for (i = 0; i < fd.width; i += skip) {
        q = schro_quantise (line[i], quant_factor, quant_offset);
        value = schro_dequantise (q, quant_factor, quant_offset);
        error += pow2 (value - line[i]);
      }
    }
  }
  error *= skip * skip;

  return error;
}
#endif

typedef struct _ErrorFuncInfo ErrorFuncInfo;
struct _ErrorFuncInfo
{
  int quant_factor;
  int quant_offset;
  double power;
};

static double
error_pow (int x, void *priv)
{
  ErrorFuncInfo *efi = priv;
  int q;
  int value;
  int y;

  q = schro_quantise (x, efi->quant_factor, efi->quant_offset);
  value = schro_dequantise (q, efi->quant_factor, efi->quant_offset);

  y = abs (value - x);

  return pow (y, efi->power);
}

void
schro_encoder_init_error_tables (SchroEncoder * encoder)
{
  int i;

  for (i = 0; i < 60; i++) {
    ErrorFuncInfo efi;

    efi.quant_factor = schro_table_quant[i];
    efi.quant_offset = schro_table_offset_1_2[i];
    efi.power = encoder->magic_error_power;

    schro_histogram_table_generate (encoder->intra_hist_tables + i,
        error_pow, &efi);
  }

}

#ifdef unused
static double
schro_histogram_estimate_error (SchroHistogram * hist, int quant_index,
    int num_refs)
{
  SchroHistogramTable *table;

  if (num_refs == 0) {
    table =
        (SchroHistogramTable
        *) (schro_table_error_hist_shift3_1_2[quant_index]);
  } else {
    /* FIXME the 3/8 table doesn't exist yet */
    //table = (SchroHistogramTable *)(schro_table_error_hist_shift3_3_8[quant_index]);
    table =
        (SchroHistogramTable
        *) (schro_table_error_hist_shift3_1_2[quant_index]);
  }
  return schro_histogram_apply_table (hist, table);
}
#endif

#ifdef DUMP_SUBBAND_CURVES
static double
schro_encoder_estimate_subband_arith (SchroEncoderFrame * frame, int component,
    int index, int quant_index)
{
  int i;
  int j;
  int16_t *line;
  int q;
  int quant_factor;
  int quant_offset;
  int estimated_entropy;
  SchroArith *arith;
  int position;
  SchroFrameData fd;

  arith = schro_arith_new ();
  schro_arith_estimate_init (arith);

  position = schro_subband_get_position (index);
  schro_subband_get_frame_data (&fd, frame->iwt_frame, component, position,
      &frame->params);

  quant_factor = schro_table_quant[quant_index];
  quant_offset = schro_table_offset_1_2[quant_index];

  if (index == 0) {
    for (j = 0; j < fd.height; j++) {
      line = SCHRO_FRAME_DATA_GET_LINE (&fd, j);
      schro_arith_estimate_sint (arith,
          SCHRO_CTX_ZPZN_F1, SCHRO_CTX_COEFF_DATA, SCHRO_CTX_SIGN_ZERO, 0);
      for (i = 1; i < fd.width; i++) {
        q = schro_quantise (line[i] - line[i - 1], quant_factor, quant_offset);
        schro_arith_estimate_sint (arith,
            SCHRO_CTX_ZPZN_F1, SCHRO_CTX_COEFF_DATA, SCHRO_CTX_SIGN_ZERO, q);
      }
    }
  } else {
    for (j = 0; j < fd.height; j++) {
      line = SCHRO_FRAME_DATA_GET_LINE (&fd, j);
      for (i = 0; i < fd.width; i++) {
        q = schro_quantise (line[i], quant_factor, quant_offset);
        schro_arith_estimate_sint (arith,
            SCHRO_CTX_ZPZN_F1, SCHRO_CTX_COEFF_DATA, SCHRO_CTX_SIGN_ZERO, q);
      }
    }
  }

  estimated_entropy = 0;

  estimated_entropy += arith->contexts[SCHRO_CTX_ZPZN_F1].n_bits;
  estimated_entropy += arith->contexts[SCHRO_CTX_ZP_F2].n_bits;
  estimated_entropy += arith->contexts[SCHRO_CTX_ZP_F3].n_bits;
  estimated_entropy += arith->contexts[SCHRO_CTX_ZP_F4].n_bits;
  estimated_entropy += arith->contexts[SCHRO_CTX_ZP_F5].n_bits;
  estimated_entropy += arith->contexts[SCHRO_CTX_ZP_F6p].n_bits;

  estimated_entropy += arith->contexts[SCHRO_CTX_COEFF_DATA].n_bits;

  estimated_entropy += arith->contexts[SCHRO_CTX_SIGN_ZERO].n_bits;

  schro_arith_free (arith);

  return estimated_entropy;
}
#endif

static void
schro_encoder_generate_subband_histogram (SchroEncoderFrame * frame,
    int component, int index, SchroHistogram * hist, int skip)
{
  int position;
  SchroFrameData fd;

  position = schro_subband_get_position (index);
  schro_subband_get_frame_data (&fd, frame->iwt_frame, component, position,
      &frame->params);

  if (index == 0 && frame->num_refs == 0) {
    schro_frame_data_generate_histogram_dc_predict (&fd, hist, skip, 0, 0);
  } else {
    schro_frame_data_generate_histogram (&fd, hist, skip);
  }
}

static void
schro_encoder_generate_subband_histograms (SchroEncoderFrame * frame)
{
  SchroParams *params = &frame->params;
  int i;
  int component;
  int pos;
  int skip;

  for (component = 0; component < 3; component++) {
    for (i = 0; i < 1 + 3 * params->transform_depth; i++) {
      pos = schro_subband_get_position (i);
      skip = 1 << MAX (0, SCHRO_SUBBAND_SHIFT (pos) - 1);
      schro_encoder_generate_subband_histogram (frame, component, i,
          &frame->subband_hists[component][i], skip);
    }
  }

  frame->have_histograms = TRUE;
}

#ifdef DUMP_SUBBAND_CURVES
static void
schro_encoder_dump_subband_curves (SchroEncoderFrame * frame)
{
  SchroParams *params = &frame->params;
  int i;
  int component;
  int position;

  SCHRO_ASSERT (frame->have_histograms);

  for (component = 0; component < 3; component++) {
    for (i = 0; i < 1 + 3 * params->transform_depth; i++) {
      int vol;
      SchroFrameData fd;
      int j;
      SchroHistogram *hist;

      position = schro_subband_get_position (i);
      schro_subband_get_frame_data (&fd, frame->iwt_frame, component,
          position, &frame->params);
      vol = fd.width * fd.height;
      hist = &frame->subband_hists[component][i];

      for (j = 0; j < 60; j++) {
        double est_entropy;
        double error;
        double est_error;
        double arith_entropy;

        error = measure_error_subband (frame, component, i, j);
        est_entropy =
            schro_histogram_estimate_entropy (&frame->subband_hists[component]
            [i], j, params->is_noarith);
        est_error =
            schro_histogram_apply_table (hist,
            &frame->encoder->intra_hist_tables[j]);
        arith_entropy =
            schro_encoder_estimate_subband_arith (frame, component, i, j);

        schro_dump (SCHRO_DUMP_SUBBAND_CURVE, "%d %d %d %g %g %g %g\n",
            component, i, j, est_entropy / vol, arith_entropy / vol,
            est_error / vol, error / vol);
      }
    }
  }
}
#endif

static void
schro_encoder_calc_estimates (SchroEncoderFrame * frame)
{
  SchroParams *params = &frame->params;
  int i;
  int j;
  int component;
  double *arith_context_ratios;

  SCHRO_ASSERT (frame->have_histograms);

#ifdef DUMP_SUBBAND_CURVES
  schro_encoder_dump_subband_curves (frame);
#endif

  for (component = 0; component < 3; component++) {
    if (frame->num_refs == 0) {
      arith_context_ratios =
          frame->encoder->average_arith_context_ratios_intra[component];
    } else {
      arith_context_ratios =
          frame->encoder->average_arith_context_ratios_inter[component];
    }

    for (i = 0; i < 1 + 3 * params->transform_depth; i++) {
      for (j = 0; j < 60; j++) {
        int vol;
        int position;
        SchroHistogram *hist;
        SchroFrameData fd;

        position = schro_subband_get_position (i);
        schro_subband_get_frame_data (&fd, frame->iwt_frame, component,
            position, &frame->params);
        vol = fd.width * fd.height;

        hist = &frame->subband_hists[component][i];
        frame->est_entropy[component][i][j] =
            schro_histogram_estimate_entropy (hist, j, params->is_noarith);
        frame->est_entropy[component][i][j] *= arith_context_ratios[i];
        frame->est_error[component][i][j] = schro_histogram_apply_table (hist,
            &frame->encoder->intra_hist_tables[j]);
      }
    }
  }
  frame->have_estimate_tables = TRUE;
}

/*
 * Quantiser engine which picks the best RDO quantisers to fit in
 * a frame bit allocation.
 */
void
schro_encoder_choose_quantisers_rdo_bit_allocation (SchroEncoderFrame * frame)
{
  double frame_lambda;
  int bits;

  schro_encoder_generate_subband_histograms (frame);
  schro_encoder_calc_estimates (frame);

  SCHRO_ASSERT (frame->have_estimate_tables);

  bits = frame->allocated_residual_bits;

  frame_lambda = schro_encoder_entropy_to_lambda (frame, bits);
  frame->frame_lambda = frame_lambda;
  SCHRO_DEBUG ("LAMBDA: %d %g %d", frame->frame_number, frame_lambda, bits);

  schro_encoder_lambda_to_entropy (frame, frame_lambda);
}

void
schro_encoder_choose_quantisers_rdo_lambda (SchroEncoderFrame * frame)
{
  SCHRO_DEBUG ("Using rdo_lambda quant selection on frame %d with lambda %g",
      frame->frame_number, frame->frame_lambda);

  schro_encoder_generate_subband_histograms (frame);
  schro_encoder_calc_estimates (frame);

  SCHRO_ASSERT (frame->have_estimate_tables);

  schro_encoder_lambda_to_entropy (frame, frame->frame_lambda);
}


void
schro_encoder_choose_quantisers_rdo_cbr (SchroEncoderFrame * frame)
{
  int est_bits, alloc_bits;
  SchroEncoder *encoder = frame->encoder;

  schro_encoder_generate_subband_histograms (frame);
  schro_encoder_calc_estimates (frame);

  SCHRO_ASSERT (frame->have_estimate_tables);

  est_bits =
      (int) (schro_encoder_lambda_to_entropy (frame, frame->frame_lambda));

  // SCHRO_ERROR("Estimated bits for frame %d residual : %d", frame>frame_number, est_bits);

  // Check that we're within reasonable bounds of the allocation

  if (frame->num_refs == 0) {
    alloc_bits = encoder->I_frame_alloc;
  } else if (schro_encoder_frame_is_B_frame (frame) == TRUE) {
    alloc_bits = encoder->B_frame_alloc;
  } else {
    alloc_bits = encoder->P_frame_alloc;
  }

}

void
schro_encoder_estimate_entropy (SchroEncoderFrame * frame)
{
  SchroParams *params = &frame->params;
  int i;
  int component;
  int n = 0;

  for (component = 0; component < 3; component++) {
    for (i = 0; i < 1 + 3 * params->transform_depth; i++) {
      n += frame->
          est_entropy[component][i][frame->quant_indices[component][i][0]];
    }
  }
  frame->estimated_residual_bits = n;

  if (frame->allocated_residual_bits > 0 &&
      frame->estimated_residual_bits >
      2 * frame->encoder->bits_per_picture + frame->allocated_residual_bits) {
    SCHRO_WARNING ("%d: estimated entropy too big (%d vs %d)",
        frame->frame_number,
        frame->estimated_residual_bits, frame->allocated_residual_bits);
  }
}

static int
schro_subband_pick_quant (SchroEncoderFrame * frame, int component, int i,
    double lambda)
{
  double x;
  double min;
  int j;
  int j_min;
  double entropy;
  double error;

  SCHRO_ASSERT (frame->have_estimate_tables);

  j_min = -1;
  min = 0;
  for (j = 0; j < 60; j++) {
    entropy = frame->est_entropy[component][i][j];
    error = frame->est_error[component][i][j];

    x = entropy + lambda * error;
    if (j == 0 || x < min) {
      j_min = j;
      min = x;
    }
  }

  return j_min;
}

static double
schro_encoder_lambda_to_entropy (SchroEncoderFrame * frame, double frame_lambda)
{
  SchroParams *params = &frame->params;
  int i;
  int component;
  double entropy = 0;
  double *table;

  if (frame->num_refs == 0) {
    table = frame->encoder->intra_subband_weights[params->wavelet_filter_index]
        [MAX (0, params->transform_depth - 1)];
  } else {
    table = frame->encoder->inter_subband_weights[params->wavelet_filter_index]
        [MAX (0, params->transform_depth - 1)];
  }

  for (component = 0; component < 3; component++) {
    for (i = 0; i < 1 + 3 * params->transform_depth; i++) {
      double lambda;
      double weight;
      int quant_index;
      int position = schro_subband_get_position (i);

      lambda = frame_lambda;

      if (i == 0) {
        lambda *= frame->encoder->magic_subband0_lambda_scale;
      }
      if (component > 0) {
        lambda *= frame->encoder->magic_chroma_lambda_scale;
      }
      if (SCHRO_SUBBAND_IS_DIAGONALLY_ORIENTED (position)) {
        lambda *= frame->encoder->magic_diagonal_lambda_scale;
      }

      weight = table[i];
      lambda /= weight * weight;

      quant_index = schro_subband_pick_quant (frame, component, i, lambda);
      entropy += frame->est_entropy[component][i][quant_index];
      schro_encoder_frame_set_quant_index (frame, component, i, -1, -1,
          quant_index);
    }
  }

  return entropy;
}

double
schro_encoder_entropy_to_lambda (SchroEncoderFrame * frame, double entropy)
{
  int j;
  double lambda_hi, lambda_lo, lambda_mid;
  double entropy_hi, entropy_lo, entropy_mid;

  lambda_hi = 1;
  entropy_hi = schro_encoder_lambda_to_entropy (frame, lambda_hi);
  SCHRO_DEBUG ("start target=%g lambda=%g entropy=%g",
      entropy, lambda_hi, entropy_hi);

  if (entropy_hi < entropy) {
    entropy_lo = entropy_hi;
    lambda_lo = lambda_hi;

    for (j = 0; j < 5; j++) {
      lambda_hi = lambda_lo * 100;
      entropy_hi = schro_encoder_lambda_to_entropy (frame, lambda_hi);

      SCHRO_DEBUG ("have: lambda=[%g,%g] entropy=[%g,%g] target=%g",
          lambda_lo, lambda_hi, entropy_lo, entropy_hi, entropy);
      if (entropy_hi > entropy)
        break;

      SCHRO_DEBUG ("--> step up");

      entropy_lo = entropy_hi;
      lambda_lo = lambda_hi;
    }
    SCHRO_DEBUG ("--> stopping");
  } else {
    for (j = 0; j < 5; j++) {
      lambda_lo = lambda_hi * 0.01;
      entropy_lo = schro_encoder_lambda_to_entropy (frame, lambda_lo);

      SCHRO_DEBUG ("have: lambda=[%g,%g] entropy=[%g,%g] target=%g",
          lambda_lo, lambda_hi, entropy_lo, entropy_hi, entropy);

      SCHRO_DEBUG ("--> step down");
      if (entropy_lo < entropy)
        break;

      entropy_hi = entropy_lo;
      lambda_hi = lambda_lo;
    }
    SCHRO_DEBUG ("--> stopping");
  }
  if (entropy_lo == entropy_hi) {
    return sqrt (lambda_lo * lambda_hi);
  }

  if (entropy_lo > entropy || entropy_hi < entropy) {
    SCHRO_ERROR ("entropy not bracketed");
  }

  for (j = 0; j < 7; j++) {
    if (entropy_hi == entropy_lo)
      break;

    SCHRO_DEBUG ("have: lambda=[%g,%g] entropy=[%g,%g] target=%g",
        lambda_lo, lambda_hi, entropy_lo, entropy_hi, entropy);

    lambda_mid = sqrt (lambda_lo * lambda_hi);
    entropy_mid = schro_encoder_lambda_to_entropy (frame, lambda_mid);

    SCHRO_DEBUG ("picking lambda_mid=%g entropy=%g", lambda_mid, entropy_mid);

    if (entropy_mid > entropy) {
      lambda_hi = lambda_mid;
      entropy_hi = entropy_mid;
      SCHRO_DEBUG ("--> focus up");
    } else {
      lambda_lo = lambda_mid;
      entropy_lo = entropy_mid;
      SCHRO_DEBUG ("--> focus down");
    }
  }

  lambda_mid = sqrt (lambda_hi * lambda_lo);
  SCHRO_DEBUG ("done %g", lambda_mid);
  return lambda_mid;
}

static double
schro_encoder_lambda_to_error (SchroEncoderFrame * frame, double frame_lambda)
{
  SchroParams *params = &frame->params;
  int i;
  int component;
  double error = 0;
  double *table;

  if (frame->num_refs == 0) {
    table = frame->encoder->intra_subband_weights[params->wavelet_filter_index]
        [MAX (0, params->transform_depth - 1)];
  } else {
    table = frame->encoder->inter_subband_weights[params->wavelet_filter_index]
        [MAX (0, params->transform_depth - 1)];
  }

  for (component = 0; component < 3; component++) {
    for (i = 0; i < 1 + 3 * params->transform_depth; i++) {
      double lambda;
      double weight;
      int quant_index;

      lambda = frame_lambda;

      if (i == 0) {
        lambda *= frame->encoder->magic_subband0_lambda_scale;
      }
      if (component > 0) {
        lambda *= frame->encoder->magic_chroma_lambda_scale;
      }

      weight = table[i];
      lambda /= weight * weight;

      quant_index = schro_subband_pick_quant (frame, component, i, lambda);
      error += frame->est_error[component][i][quant_index];
      schro_encoder_frame_set_quant_index (frame, component, i, -1, -1,
          quant_index);
    }
  }

  return error;
}

static double
schro_encoder_error_to_lambda (SchroEncoderFrame * frame, double error)
{
  int j;
  double lambda_hi, lambda_lo, lambda_mid;
  double error_hi, error_lo, error_mid;

  lambda_lo = 1;
  error_lo = schro_encoder_lambda_to_error (frame, lambda_lo);
  SCHRO_DEBUG ("start target=%g lambda=%g error=%g",
      error, lambda_lo, error_lo, lambda_lo, error);

  if (error < error_lo) {
    error_hi = error_lo;
    lambda_hi = lambda_lo;

    for (j = 0; j < 5; j++) {
      lambda_lo = lambda_hi * 100;
      error_lo = schro_encoder_lambda_to_error (frame, lambda_lo);

      SCHRO_DEBUG ("have: lambda=[%g,%g] error=[%g,%g] target=%g",
          lambda_lo, lambda_hi, error_lo, error_hi, error);
      if (error > error_lo)
        break;

      SCHRO_DEBUG ("--> step up");

      error_hi = error_lo;
      lambda_hi = lambda_lo;
    }
    SCHRO_DEBUG ("--> stopping");
  } else {
    for (j = 0; j < 5; j++) {
      lambda_hi = lambda_lo * 0.01;
      error_hi = schro_encoder_lambda_to_error (frame, lambda_hi);

      SCHRO_DEBUG ("have: lambda=[%g,%g] error=[%g,%g] target=%g",
          lambda_lo, lambda_hi, error_lo, error_hi, error);

      SCHRO_DEBUG ("--> step down");
      if (error < error_hi)
        break;

      error_lo = error_hi;
      lambda_lo = lambda_hi;
    }
    SCHRO_DEBUG ("--> stopping");
  }
  if (error_lo == error_hi) {
    return sqrt (lambda_lo * lambda_hi);
  }

  if (error_lo > error || error_hi < error) {
    SCHRO_DEBUG ("error not bracketed");
  }

  for (j = 0; j < 14; j++) {
    if (error_hi == error_lo)
      break;

    SCHRO_DEBUG ("have: lambda=[%g,%g] error=[%g,%g] target=%g",
        lambda_lo, lambda_hi, error_lo, error_hi, error);

    lambda_mid = sqrt (lambda_lo * lambda_hi);
    error_mid = schro_encoder_lambda_to_error (frame, lambda_mid);

    SCHRO_DEBUG ("picking lambda_mid=%g error=%g", lambda_mid, error_mid);

    if (error_mid > error) {
      lambda_hi = lambda_mid;
      error_hi = error_mid;
      SCHRO_DEBUG ("--> focus up");
    } else {
      lambda_lo = lambda_mid;
      error_lo = error_mid;
      SCHRO_DEBUG ("--> focus down");
    }
  }

  lambda_mid = sqrt (lambda_hi * lambda_lo);
  SCHRO_DEBUG ("done %g", lambda_mid);
  return lambda_mid;
}

void
schro_encoder_choose_quantisers_constant_error (SchroEncoderFrame * frame)
{
  double frame_lambda;
  double error;

  schro_encoder_generate_subband_histograms (frame);
  schro_encoder_calc_estimates (frame);

  SCHRO_ASSERT (frame->have_estimate_tables);

  error = 255.0 * pow (0.1, frame->encoder->noise_threshold * 0.05);
  error *= frame->params.video_format->width *
      frame->params.video_format->height;

  frame_lambda = schro_encoder_error_to_lambda (frame, error);

  frame->frame_lambda = frame_lambda;
  SCHRO_DEBUG ("LAMBDA: %d %g", frame->frame_number, frame_lambda);
}
