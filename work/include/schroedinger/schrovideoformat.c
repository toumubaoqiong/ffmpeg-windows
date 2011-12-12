
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <string.h>
#include <stdlib.h>


/*
 * schro_video_format_validate:
 * @format: pointer to a SchroVideoFormat structure
 *
 * Checks the video format structure pointed to by @format for
 * inconsistencies.
 *
 * Returns: TRUE if the contents of @format is valid
 */
int
schro_video_format_validate (SchroVideoFormat * format)
{
  int fix_clean_area = 0;

  if (format->aspect_ratio_numerator == 0) {
    SCHRO_ERROR ("aspect_ratio_numerator is 0");
    format->aspect_ratio_numerator = 1;
  }
  if (format->aspect_ratio_denominator == 0) {
    SCHRO_ERROR ("aspect_ratio_denominator is 0");
    format->aspect_ratio_denominator = 1;
  }

  if (format->clean_width + format->left_offset > format->width) {
    SCHRO_ERROR
        ("10.3.7: horizontal clean area is not legal (clean_width + left_offset > width)");
    fix_clean_area = 1;
  }
  if (format->clean_height + format->top_offset > format->height) {
    SCHRO_ERROR
        ("10.3.7: vertical clean area is not legal (clean_height + top_offset > height)");
    fix_clean_area = 1;
  }
  if (fix_clean_area) {
    SCHRO_ERROR ("resetting clean area to frame size");
    format->clean_width = format->width;
    format->clean_height = format->height;
    format->left_offset = format->top_offset = 0;
  }

  if (schro_video_format_get_bit_depth (format) != 8) {
    SCHRO_WARNING ("video bit depth != 8");
    return 0;
  }

  return 1;
}

#ifdef unused
int
schro_video_format_compare_new_sequence (SchroVideoFormat * a,
    SchroVideoFormat * b)
{
  if (a->index != b->index ||
      a->width != b->width ||
      a->height != b->height ||
      a->chroma_format != b->chroma_format ||
      a->interlaced != b->interlaced ||
      a->top_field_first != b->top_field_first ||
      a->frame_rate_numerator != b->frame_rate_numerator ||
      a->frame_rate_denominator != b->frame_rate_denominator ||
      a->aspect_ratio_numerator != b->aspect_ratio_numerator ||
      a->aspect_ratio_denominator != b->aspect_ratio_denominator ||
      a->clean_width != b->clean_width ||
      a->clean_height != b->clean_height ||
      a->left_offset != b->left_offset ||
      a->top_offset != b->top_offset ||
      a->luma_offset != b->luma_offset ||
      a->luma_excursion != b->luma_excursion ||
      a->chroma_offset != b->chroma_offset ||
      a->chroma_excursion != b->chroma_excursion ||
      a->colour_primaries != b->colour_primaries ||
      a->colour_matrix != b->colour_matrix ||
      a->transfer_function != b->transfer_function) {
    return FALSE;
  }
  return TRUE;
}
#endif

#ifdef unused
int
schro_video_format_compare (SchroVideoFormat * a, SchroVideoFormat * b)
{
  if (!schro_video_format_compare_new_sequence (a, b) ||
      a->interlaced_coding != b->interlaced_coding) {
    return FALSE;
  }
  return TRUE;
}
#endif

int
schro_video_format_get_bit_depth (SchroVideoFormat * format)
{
  int max;
  int i;

  max = MAX (format->chroma_excursion, format->luma_excursion);

  for (i = 0; i < 32; i++) {
    if (max < (1 << i))
      return i;
  }
  return 0;
}

static SchroVideoFormat schro_video_formats[] = {
  {0,                           /* custom */
        640, 480, SCHRO_CHROMA_420,
        FALSE, FALSE,
        24000, 1001, 1, 1,
        640, 480, 0, 0,
        0, 255, 128, 255,
      0, 0, 0},
  {1,                           /* QSIF525 */
        176, 120, SCHRO_CHROMA_420,
        FALSE, FALSE,
        15000, 1001, 10, 11,
        176, 120, 0, 0,
        0, 255, 128, 255,
      1, 1, 0},
  {2,                           /* QCIF */
        176, 144, SCHRO_CHROMA_420,
        FALSE, TRUE,
        25, 2, 12, 11,
        176, 144, 0, 0,
        0, 255, 128, 255,
      2, 1, 0},
  {3,                           /* SIF525 */
        352, 240, SCHRO_CHROMA_420,
        FALSE, FALSE,
        15000, 1001, 10, 11,
        352, 240, 0, 0,
        0, 255, 128, 255,
      1, 1, 0},
  {4,                           /* CIF */
        352, 288, SCHRO_CHROMA_420,
        FALSE, TRUE,
        25, 2, 12, 11,
        352, 288, 0, 0,
        0, 255, 128, 255,
      2, 1, 0},
  {5,                           /* 4SIF525 */
        704, 480, SCHRO_CHROMA_420,
        FALSE, FALSE,
        15000, 1001, 10, 11,
        704, 480, 0, 0,
        0, 255, 128, 255,
      1, 1, 0},
  {6,                           /* 4CIF */
        704, 576, SCHRO_CHROMA_420,
        FALSE, TRUE,
        25, 2, 12, 11,
        704, 576, 0, 0,
        0, 255, 128, 255,
      2, 1, 0},
  {7,                           /* SD480I-60 */
        720, 480, SCHRO_CHROMA_422,
        TRUE, FALSE,
        30000, 1001, 10, 11,
        704, 480, 8, 0,
        64, 876, 512, 896,
      1, 1, 0},
  {8,                           /* SD576I-50 */
        720, 576, SCHRO_CHROMA_422,
        TRUE, TRUE,
        25, 1, 12, 11,
        704, 576, 8, 0,
        64, 876, 512, 896,
      2, 1, 0},
  {9,                           /* HD720P-60 */
        1280, 720, SCHRO_CHROMA_422,
        FALSE, TRUE,
        60000, 1001, 1, 1,
        1280, 720, 0, 0,
        64, 876, 512, 896,
      0, 0, 0},
  {10,                          /* HD720P-50 */
        1280, 720, SCHRO_CHROMA_422,
        FALSE, TRUE,
        50, 1, 1, 1,
        1280, 720, 0, 0,
        64, 876, 512, 896,
      0, 0, 0},
  {11,                          /* HD1080I-60 */
        1920, 1080, SCHRO_CHROMA_422,
        TRUE, TRUE,
        30000, 1001, 1, 1,
        1920, 1080, 0, 0,
        64, 876, 512, 896,
      0, 0, 0},
  {12,                          /* HD1080I-50 */
        1920, 1080, SCHRO_CHROMA_422,
        TRUE, TRUE,
        25, 1, 1, 1,
        1920, 1080, 0, 0,
        64, 876, 512, 896,
      0, 0, 0},
  {13,                          /* HD1080P-60 */
        1920, 1080, SCHRO_CHROMA_422,
        FALSE, TRUE,
        60000, 1001, 1, 1,
        1920, 1080, 0, 0,
        64, 876, 512, 896,
      0, 0, 0},
  {14,                          /* HD1080P-50 */
        1920, 1080, SCHRO_CHROMA_422,
        FALSE, TRUE,
        50, 1, 1, 1,
        1920, 1080, 0, 0,
        64, 876, 512, 896,
      0, 0, 0},
  {15,                          /* DC2K */
        2048, 1080, SCHRO_CHROMA_444,
        FALSE, TRUE,
        24, 1, 1, 1,
        2048, 1080, 0, 0,
        256, 3504, 2048, 3584,
      3, 0, 0},
  {16,                          /* DC4K */
        4096, 2160, SCHRO_CHROMA_444,
        FALSE, TRUE,
        24, 1, 1, 1,
        2048, 1536, 0, 0,
        256, 3504, 2048, 3584,
      3, 0, 0},
  {17,                          /* UHDTV 4K-60  */
        3840, 2160, SCHRO_CHROMA_422,
        FALSE, TRUE,
        60000, 1001, 1, 1,
        3840, 2160, 0, 0,
        64, 876, 512, 896,
      0, 0, 0},
  {18,                          /* UHDTV 4K-50 */
        3840, 2160, SCHRO_CHROMA_422,
        FALSE, TRUE,
        50, 1, 1, 1,
        3840, 2160, 0, 0,
        64, 876, 512, 896,
      0, 0, 0},
  {19,                          /* UHDTV 8K-60 */
        7680, 4320, SCHRO_CHROMA_422,
        FALSE, TRUE,
        60000, 1001, 1, 1,
        7680, 4320, 0, 0,
        64, 876, 512, 896,
      0, 0, 0},
  {20,                          /* UHDTV 8K-50 */
        7680, 4320, SCHRO_CHROMA_422,
        FALSE, TRUE,
        50, 1, 1, 1,
        7680, 4320, 0, 0,
        64, 876, 512, 896,
      0, 0, 0},
};

/**
 * schro_video_format_set_std_video_format:
 * @format:
 * @index:
 *
 * Initializes the video format structure pointed to by @format to
 * the standard Dirac video formats specified by @index.
 */
void
schro_video_format_set_std_video_format (SchroVideoFormat * format,
    SchroVideoFormatEnum index)
{
  if (index < 0 || index >= ARRAY_SIZE (schro_video_formats)) {
    SCHRO_ERROR ("illegal video format index");
    return;
  }

  memcpy (format, schro_video_formats + index, sizeof (SchroVideoFormat));
}

static int
schro_video_format_get_video_format_metric (SchroVideoFormat * format, int i)
{
  SchroVideoFormat *std_format;
  int metric = 0;

  std_format = schro_video_formats + i;

  /* this is really important because it can't be overrided */
  if (format->interlaced &&
      format->top_field_first == std_format->top_field_first) {
    metric |= 0x8000;
  }

  metric += schro_pack_estimate_uint (i);

  if (std_format->width == format->width &&
      std_format->height == format->height) {
    metric++;
  } else {
    metric++;
    metric += schro_pack_estimate_uint (format->width);
    metric += schro_pack_estimate_uint (format->height);
  }

  if (std_format->chroma_format == format->chroma_format) {
    metric++;
  } else {
    metric++;
    metric += schro_pack_estimate_uint (format->chroma_format);
  }

  /* scan format */
  if (std_format->interlaced == format->interlaced) {
    metric++;
  } else {
    metric++;
    metric += schro_pack_estimate_uint (format->interlaced);
  }

  /* frame rate */
  if (std_format->frame_rate_numerator == format->frame_rate_numerator &&
      std_format->frame_rate_denominator == format->frame_rate_denominator) {
    metric++;
  } else {
    metric++;
    i = schro_video_format_get_std_frame_rate (format);
    metric += schro_pack_estimate_uint (i);
    if (i == 0) {
      metric += schro_pack_estimate_uint (format->frame_rate_numerator);
      metric += schro_pack_estimate_uint (format->frame_rate_denominator);
    }
  }

  /* pixel aspect ratio */
  if (std_format->aspect_ratio_numerator == format->aspect_ratio_numerator &&
      std_format->aspect_ratio_denominator ==
      format->aspect_ratio_denominator) {
    metric++;
  } else {
    metric++;
    i = schro_video_format_get_std_aspect_ratio (format);
    metric += schro_pack_estimate_uint (i);
    if (i == 0) {
      metric += schro_pack_estimate_uint (format->aspect_ratio_numerator);
      metric += schro_pack_estimate_uint (format->aspect_ratio_denominator);
    }
  }

  /* clean area */
  if (std_format->clean_width == format->clean_width &&
      std_format->clean_height == format->clean_height &&
      std_format->left_offset == format->left_offset &&
      std_format->top_offset == format->top_offset) {
    metric++;
  } else {
    metric++;
    metric += schro_pack_estimate_uint (format->clean_width);
    metric += schro_pack_estimate_uint (format->clean_height);
    metric += schro_pack_estimate_uint (format->left_offset);
    metric += schro_pack_estimate_uint (format->top_offset);
  }


#if 0
  if (format->left_offset == std_format->left_offset &&
      format->top_offset == std_format->top_offset &&
      format->clean_width == std_format->clean_width &&
      format->clean_height == std_format->clean_height) {
    metric |= 0x40;
  }
#endif

  return metric;
}

/**
 * schro_video_format_get_std_video_format:
 * @format: pointer to SchroVideoFormat structure
 *
 * In Dirac streams, video formats are encoded by specifying a standard
 * format, and then modifying that to get the desired video format.  This
 * function guesses a standard format to use as a starting point for
 * encoding the video format pointed to by @format.
 *
 * Returns: an index to the optimal standard format
 */
SchroVideoFormatEnum
schro_video_format_get_std_video_format (SchroVideoFormat * format)
{
  int metric;
  int max_index;
  int max_metric;
  int i;

  max_index = 0;
  max_metric = schro_video_format_get_video_format_metric (format, 1);
  for (i = 1; i < ARRAY_SIZE (schro_video_formats); i++) {
    metric = schro_video_format_get_video_format_metric (format, i);
    if (metric > max_metric) {
      max_index = i;
      max_metric = metric;
    }
  }
  return max_index;
}

typedef struct _SchroFrameRate SchroFrameRate;
struct _SchroFrameRate
{
  int numerator;
  int denominator;
};

static SchroFrameRate schro_frame_rates[] = {
  {0, 0},
  {24000, 1001},
  {24, 1},
  {25, 1},
  {30000, 1001},
  {30, 1},
  {50, 1},
  {60000, 1001},
  {60, 1},
  {15000, 1001},
  {25, 2}
};

/**
 * schro_video_format_set_std_frame_rate:
 * @format:
 * @index:
 *
 * Sets the frame rate of the video format structure pointed to by
 * @format to the Dirac standard frame specified by @index.
 */
void
schro_video_format_set_std_frame_rate (SchroVideoFormat * format, int index)
{
  if (index < 1 || index >= ARRAY_SIZE (schro_frame_rates)) {
    SCHRO_ERROR ("illegal frame rate index");
    return;
  }

  format->frame_rate_numerator = schro_frame_rates[index].numerator;
  format->frame_rate_denominator = schro_frame_rates[index].denominator;
}

/**
 * schro_video_format_get_std_frame_rate:
 * @format:
 *
 * In Dirac bitstreams, frame rates can be one of several standard
 * frame rates, encoded as an index, or the numerator and denominator
 * of the framerate can be encoded directly.  This function looks up
 * the frame rate contained in the video format structure @format in
 * the list of standard frame rates.  If the frame rate is a standard
 * frame rate, the corresponding index is returned, otherwise 0 is
 * returned.
 *
 * Returns: index to a standard Dirac frame rate, or 0 if the frame rate
 * is custom.
 */
int
schro_video_format_get_std_frame_rate (SchroVideoFormat * format)
{
  int i;

  for (i = 1; i < ARRAY_SIZE (schro_frame_rates); i++) {
    if (format->frame_rate_numerator == schro_frame_rates[i].numerator &&
        format->frame_rate_denominator == schro_frame_rates[i].denominator) {
      return i;
    }
  }

  return 0;
}

typedef struct _SchroPixelAspectRatio SchroPixelAspectRatio;
struct _SchroPixelAspectRatio
{
  int numerator;
  int denominator;
};

static const SchroPixelAspectRatio schro_aspect_ratios[] = {
  {0, 0},
  {1, 1},
  {10, 11},
  {12, 11},
  {40, 33},
  {16, 11},
  {4, 3}
};

/*
 * schro_video_format_set_std_aspect_ratio:
 * @format: pointer to a SchroVideoFormat structure
 * @index: index to a standard aspect ratio
 *
 * Sets the pixel aspect ratio of the video format structure pointed to
 * by @format to the standard pixel aspect ratio indicated by @index.
 */
void
schro_video_format_set_std_aspect_ratio (SchroVideoFormat * format, int index)
{
  if (index < 1 || index >= ARRAY_SIZE (schro_aspect_ratios)) {
    SCHRO_ERROR ("illegal pixel aspect ratio index");
    return;
  }

  format->aspect_ratio_numerator = schro_aspect_ratios[index].numerator;
  format->aspect_ratio_denominator = schro_aspect_ratios[index].denominator;

}

/*
 * schro_video_format_get_std_aspect_ratio:
 * @format: pointer to a SchroVideoFormat structure
 *
 * In Dirac bitstreams, pixel aspect ratios can be one of several standard
 * pixel aspect ratios, encoded as an index, or the numerator and denominator
 * of the pixel aspect ratio can be encoded directly.  This function looks up
 * the pixel aspect ratio contained in the video format structure @format in
 * the list of standard pixel aspect ratios.  If the pixel aspect ratio is
 * a standard pixel aspect ratio, the corresponding index is returned,
 * otherwise 0 is returned.
 *
 * Returns: index to standard pixel aspect ratio, or 0 if there is no
 * corresponding standard pixel aspect ratio.
 */
int
schro_video_format_get_std_aspect_ratio (SchroVideoFormat * format)
{
  int i;

  for (i = 1; i < ARRAY_SIZE (schro_aspect_ratios); i++) {
    if (format->aspect_ratio_numerator ==
        schro_aspect_ratios[i].numerator &&
        format->aspect_ratio_denominator ==
        schro_aspect_ratios[i].denominator) {
      return i;
    }
  }

  return 0;
}

typedef struct _SchroSignalRangeStruct SchroSignalRangeStruct;
struct _SchroSignalRangeStruct
{
  int luma_offset;
  int luma_excursion;
  int chroma_offset;
  int chroma_excursion;
};

static const SchroSignalRangeStruct schro_signal_ranges[] = {
  {0, 0, 0, 0},
  {0, 255, 128, 255},
  {16, 219, 128, 224},
  {64, 876, 512, 896},
  {256, 3504, 2048, 3584}
};

/**
 * schro_video_format_set_std_signal_range:
 * @format:
 * @index:
 *
 * Sets the signal range of the video format structure to one of the
 * standard values indicated by @index.
 */
void
schro_video_format_set_std_signal_range (SchroVideoFormat * format,
    SchroSignalRange i)
{
  if (i < 1 || i >= ARRAY_SIZE (schro_signal_ranges)) {
    SCHRO_ERROR ("illegal signal range index");
    return;
  }

  format->luma_offset = schro_signal_ranges[i].luma_offset;
  format->luma_excursion = schro_signal_ranges[i].luma_excursion;
  format->chroma_offset = schro_signal_ranges[i].chroma_offset;
  format->chroma_excursion = schro_signal_ranges[i].chroma_excursion;
}

/**
 * schro_video_format_get_std_signal_range:
 * @format: pointer to SchroVideoFormat structure
 *
 * In Dirac bitstreams, signal ranges can be one of several standard
 * signal ranges, encoded as an index, or the extents of the signal
 * range can be encoded directly.  This function looks up
 * the signal range contained in the video format structure @format in
 * the list of standard signal ranges.  If the signal range is
 * a standard signal range, the corresponding index is returned,
 * otherwise 0 is returned.
 *
 * Returns: index to standard signal range, or 0 if there is no
 * corresponding standard signal range.
 */
SchroSignalRange
schro_video_format_get_std_signal_range (SchroVideoFormat * format)
{
  int i;

  for (i = 1; i < ARRAY_SIZE (schro_signal_ranges); i++) {
    if (format->luma_offset == schro_signal_ranges[i].luma_offset &&
        format->luma_excursion == schro_signal_ranges[i].luma_excursion &&
        format->chroma_offset == schro_signal_ranges[i].chroma_offset &&
        format->chroma_excursion == schro_signal_ranges[i].chroma_excursion) {
      return i;
    }
  }

  return 0;

}

typedef struct _SchroColourSpecStruct SchroColourSpecStruct;
struct _SchroColourSpecStruct
{
  int colour_primaries;
  int colour_matrix;
  int transfer_function;
};

static const SchroColourSpecStruct schro_colour_specs[] = {
  {                             /* Custom */
        SCHRO_COLOUR_PRIMARY_HDTV,
        SCHRO_COLOUR_MATRIX_HDTV,
      SCHRO_TRANSFER_CHAR_TV_GAMMA},
  {                             /* SDTV 525 */
        SCHRO_COLOUR_PRIMARY_SDTV_525,
        SCHRO_COLOUR_MATRIX_SDTV,
      SCHRO_TRANSFER_CHAR_TV_GAMMA},
  {                             /* SDTV 625 */
        SCHRO_COLOUR_PRIMARY_SDTV_625,
        SCHRO_COLOUR_MATRIX_SDTV,
      SCHRO_TRANSFER_CHAR_TV_GAMMA},
  {                             /* HDTV */
        SCHRO_COLOUR_PRIMARY_HDTV,
        SCHRO_COLOUR_MATRIX_HDTV,
      SCHRO_TRANSFER_CHAR_TV_GAMMA},
  {                             /* Cinema */
        SCHRO_COLOUR_PRIMARY_CINEMA,
        SCHRO_COLOUR_MATRIX_HDTV,
      SCHRO_TRANSFER_CHAR_TV_GAMMA}
};

/**
 * schro_video_format_set_std_colour_spec:
 * @format: pointer to SchroVideoFormat structure
 * @index: index to standard colour specification
 *
 * Sets the colour specification of the video format structure to one of the
 * standard values indicated by @index.
 */
void
schro_video_format_set_std_colour_spec (SchroVideoFormat * format,
    SchroColourSpec i)
{
  if (i < 0 || i >= ARRAY_SIZE (schro_colour_specs)) {
    SCHRO_ERROR ("illegal signal range index");
    return;
  }

  format->colour_primaries = schro_colour_specs[i].colour_primaries;
  format->colour_matrix = schro_colour_specs[i].colour_matrix;
  format->transfer_function = schro_colour_specs[i].transfer_function;
}

/**
 * schro_video_format_get_std_colour_spec:
 * @format: pointer to SchroVideoFormat structure
 *
 * In Dirac bitstreams, colour specifications can be one of several standard
 * colour specifications, encoded as an index, or the individual parts of
 * the colour specication can be encoded.  This function looks up
 * the colour specification contained in the video format structure @format in
 * the list of standard colour specifications.  If the colour specification is
 * a standard colour specification, the corresponding index is returned,
 * otherwise 0 is returned.
 *
 * Returns: index to standard colour specification, or 0 if there is no
 * corresponding standard colour specification.
 */
SchroColourSpec
schro_video_format_get_std_colour_spec (SchroVideoFormat * format)
{
  int i;

  for (i = 1; i < ARRAY_SIZE (schro_colour_specs); i++) {
    if (format->colour_primaries == schro_colour_specs[i].colour_primaries &&
        format->colour_matrix == schro_colour_specs[i].colour_matrix &&
        format->transfer_function == schro_colour_specs[i].transfer_function) {
      return i;
    }
  }

  return 0;
}

/**
 * schro_video_format_get_picture_height:
 * @format: pointer to SchroVideoFormat structure
 *
 * Returns the height of coded pictures in the Dirac stream.  For
 * streams encoded with interlaced_coding enabled, this will be the
 * field height, or half of the video height.
 */
int
schro_video_format_get_picture_height (SchroVideoFormat * format)
{
  if (format->interlaced_coding) {
    return ROUND_UP_SHIFT (format->height, 1);
  }
  return format->height;
}

#ifdef unused
int
schro_video_format_get_chroma_width (SchroVideoFormat * format)
{
  return ROUND_UP_SHIFT (format->width,
      SCHRO_CHROMA_FORMAT_H_SHIFT (format->chroma_format));
}
#endif

#ifdef unused
int
schro_video_format_get_chroma_height (SchroVideoFormat * format)
{
  return ROUND_UP_SHIFT (format->height,
      SCHRO_CHROMA_FORMAT_V_SHIFT (format->chroma_format));
}
#endif

void
schro_video_format_get_picture_luma_size (SchroVideoFormat * format,
    int *width, int *height)
{
  *width = format->width;
  *height = ROUND_UP_SHIFT (format->height, format->interlaced_coding);
}

void
schro_video_format_get_picture_chroma_size (SchroVideoFormat * format,
    int *width, int *height)
{
  *width = ROUND_UP_SHIFT (format->width,
      SCHRO_CHROMA_FORMAT_H_SHIFT (format->chroma_format));
  *height = ROUND_UP_SHIFT (format->height,
      SCHRO_CHROMA_FORMAT_V_SHIFT (format->chroma_format) +
      format->interlaced_coding);
}

void
schro_video_format_get_iwt_alloc_size (SchroVideoFormat * format,
    int *width, int *height, int transform_depth)
{
  int picture_chroma_width;
  int picture_chroma_height;

  schro_video_format_get_picture_chroma_size (format, &picture_chroma_width,
      &picture_chroma_height);

  picture_chroma_width = ROUND_UP_POW2 (picture_chroma_width, transform_depth);
  picture_chroma_height = ROUND_UP_POW2 (picture_chroma_height,
      transform_depth);

  *width = picture_chroma_width <<
      SCHRO_CHROMA_FORMAT_H_SHIFT (format->chroma_format);
  *height = picture_chroma_height <<
      SCHRO_CHROMA_FORMAT_V_SHIFT (format->chroma_format);
}

schro_bool
schro_video_format_check_MP_DL (SchroVideoFormat * format)
{
  SchroVideoFormat base_format;

  if (format->index < 1 || format->index > 20) {
    return FALSE;
  }

  schro_video_format_set_std_video_format (&base_format, format->index);

  if (format->width > base_format.width || format->height > base_format.height) {
    return FALSE;
  }

  if (format->frame_rate_numerator != base_format.frame_rate_numerator ||
      format->frame_rate_denominator != base_format.frame_rate_denominator) {
    return FALSE;
  }

  if (format->clean_width != base_format.clean_width ||
      format->clean_height != base_format.clean_height ||
      format->left_offset != base_format.left_offset ||
      format->top_offset != base_format.top_offset) {
    return FALSE;
  }

  if (schro_video_format_get_std_signal_range (format) != 2) {
    return FALSE;
  }

  if (format->colour_primaries != base_format.colour_primaries ||
      format->colour_matrix != base_format.colour_matrix ||
      format->transfer_function != base_format.transfer_function) {
    return FALSE;
  }

  return TRUE;
}

schro_bool
schro_video_format_check_VC2_DL (SchroVideoFormat * format)
{
  SchroVideoFormat base_format;

  if (format->index < 1 || format->index > 20) {
    return FALSE;
  }

  schro_video_format_set_std_video_format (&base_format, format->index);

  if (memcmp (&base_format, format, sizeof (SchroVideoFormat)) != 0) {
    return FALSE;
  }

  return TRUE;
}
