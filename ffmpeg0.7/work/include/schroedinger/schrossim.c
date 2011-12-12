
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>


static SchroFrame *
schro_frame_dup16 (SchroFrame * frame)
{
  SchroFrame *newframe;
  SchroFrameFormat format;

  /* FIXME hack */
  format = SCHRO_FRAME_FORMAT_S16_444 | frame->format;
  newframe = schro_frame_new_and_alloc (NULL, format,
      frame->width, frame->height);
  schro_frame_convert (newframe, frame);

  return newframe;
}

static void
schro_frame_multiply_s16 (SchroFrame * dest, SchroFrame * src)
{
  SchroFrameData *dcomp;
  SchroFrameData *scomp;
  int i;
  int16_t *ddata;
  int16_t *sdata;
  int width, height;
  int x, y;

  for (i = 0; i < 3; i++) {
    dcomp = &dest->components[i];
    scomp = &src->components[i];
    ddata = dcomp->data;
    sdata = scomp->data;

    width = (dcomp->width < scomp->width) ? dcomp->width : scomp->width;
    height = (dcomp->height < scomp->height) ? dcomp->height : scomp->height;

    for (y = 0; y < height; y++) {
      ddata = SCHRO_FRAME_DATA_GET_LINE (dcomp, y);
      sdata = SCHRO_FRAME_DATA_GET_LINE (scomp, y);
      for (x = 0; x < width; x++) {
        int z;
        z = ddata[x] * sdata[x];
        ddata[x] = CLAMP (z, -32768, 32767);
      }
    }
  }
}

static void
schro_frame_multiply (SchroFrame * a, SchroFrame * b)
{
  schro_frame_multiply_s16 (a, b);
}

#define SSIM_SIGMA 1.5

double
schro_frame_ssim (SchroFrame * a, SchroFrame * b)
{
  SchroFrame *a_lowpass;
  SchroFrame *b_lowpass;
  SchroFrame *a_hipass;
  SchroFrame *b_hipass;
  SchroFrame *ab;
  double ssim_sum;
  double mssim;
  double sum, diff;
  double ave;
  int i, j;

  a_lowpass = schro_frame_dup (a);
  schro_frame_filter_lowpass2 (a_lowpass, (a->width / 256.0) * SSIM_SIGMA);

  b_lowpass = schro_frame_dup (b);
  schro_frame_filter_lowpass2 (b_lowpass, (b->width / 256.0) * SSIM_SIGMA);

  a_hipass = schro_frame_dup16 (a);
  schro_frame_subtract (a_hipass, a_lowpass);

  b_hipass = schro_frame_dup16 (b);
  schro_frame_subtract (b_hipass, b_lowpass);

  ab = schro_frame_dup (a_hipass);
  schro_frame_multiply (ab, b_hipass);
  schro_frame_multiply (a_hipass, a_hipass);
  schro_frame_multiply (b_hipass, b_hipass);

  schro_frame_filter_lowpass2 (a_hipass, (a->width / 256.0) * SSIM_SIGMA);
  schro_frame_filter_lowpass2 (b_hipass, (a->width / 256.0) * SSIM_SIGMA);
  schro_frame_filter_lowpass2 (ab, (a->width / 256.0) * SSIM_SIGMA);

  ssim_sum = 0;
  for (j = 0; j < a->height; j++) {
    for (i = 0; i < a->width; i++) {
      double ssim;
      double c1 = (0.01 * 255) * (0.01 * 255);
      double c2 = (0.03 * 255) * (0.03 * 255);
      int mu_x, mu_y, sigma_x2, sigma_y2, sigma_xy;

#define GET(f, x, y) SCHRO_GET(f->components[0].data, f->components[0].stride * j + i * sizeof(uint8_t), uint8_t)
#define GET16(f, x, y) SCHRO_GET(f->components[0].data, f->components[0].stride * j + i * sizeof(int16_t), int16_t)

      mu_x = GET (a_lowpass, i, j);
      mu_y = GET (b_lowpass, i, j);
      sigma_x2 = GET16 (a_hipass, i, j);
      sigma_y2 = GET16 (b_hipass, i, j);
      sigma_xy = GET16 (ab, i, j);

      ssim = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2) /
          ((mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x2 + sigma_y2 + c2));
      //SCHRO_ERROR("%d %d, mu %d %d sigma %d %d %d ssim %g",
      //    i,j, mu_x, mu_y, sigma_x2, sigma_y2, sigma_xy, ssim);
      //SCHRO_ASSERT(ssim <= 1.0);
      ssim_sum += ssim;
    }
  }
  mssim = ssim_sum / (a->width * a->height);

  sum = 0;
  for (j = 0; j < a->height; j++) {
    for (i = 0; i < a->width; i++) {
      int x_a, x_b;

      x_a = GET (a, i, j);
      x_b = GET (b, i, j);

      sum += abs (x_a - x_b);
    }
  }
  diff = sum / (a->width * a->height * 255.0);
  ave = schro_frame_calculate_average_luma (a) / 255.0;

  SCHRO_DEBUG ("mssim,diff,ave %g %g %g", mssim, diff, ave);

  schro_frame_unref (a_lowpass);
  schro_frame_unref (b_lowpass);
  schro_frame_unref (a_hipass);
  schro_frame_unref (b_hipass);
  schro_frame_unref (ab);

  return mssim;
}
