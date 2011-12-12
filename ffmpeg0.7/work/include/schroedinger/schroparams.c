
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>


void
schro_params_init (SchroParams * params, int video_format)
{
  int i;

  params->transform_depth = 4;

  if (params->num_refs == 0) {
    if (video_format < 11) {
      params->wavelet_filter_index = SCHRO_WAVELET_DESLAURIERS_DUBUC_9_7;
    } else {
      params->wavelet_filter_index = SCHRO_WAVELET_FIDELITY;
    }
  } else {
    if (video_format < 11) {
      params->wavelet_filter_index = SCHRO_WAVELET_LE_GALL_5_3;
    } else {
      params->wavelet_filter_index = SCHRO_WAVELET_DESLAURIERS_DUBUC_9_7;
    }
  }

  switch (video_format) {
    case SCHRO_VIDEO_FORMAT_QCIF:
    case SCHRO_VIDEO_FORMAT_QSIF:
      params->xblen_luma = 8;
      params->yblen_luma = 8;
      params->xbsep_luma = 4;
      params->ybsep_luma = 4;
      break;

    case SCHRO_VIDEO_FORMAT_CUSTOM:
    case SCHRO_VIDEO_FORMAT_SIF:
    case SCHRO_VIDEO_FORMAT_CIF:
    case SCHRO_VIDEO_FORMAT_4SIF:
    case SCHRO_VIDEO_FORMAT_4CIF:
    case SCHRO_VIDEO_FORMAT_SD480I_60:
    case SCHRO_VIDEO_FORMAT_SD576I_50:
      params->xblen_luma = 12;
      params->yblen_luma = 12;
      params->xbsep_luma = 8;
      params->ybsep_luma = 8;
      break;

    case SCHRO_VIDEO_FORMAT_HD720P_60:
    case SCHRO_VIDEO_FORMAT_HD720P_50:
      params->xblen_luma = 16;
      params->yblen_luma = 16;
      params->xbsep_luma = 12;
      params->ybsep_luma = 12;
      break;

    case SCHRO_VIDEO_FORMAT_HD1080I_60:
    case SCHRO_VIDEO_FORMAT_HD1080I_50:
    case SCHRO_VIDEO_FORMAT_HD1080P_60:
    case SCHRO_VIDEO_FORMAT_HD1080P_50:
    case SCHRO_VIDEO_FORMAT_DC2K_24:
    case SCHRO_VIDEO_FORMAT_DC4K_24:
      params->xblen_luma = 24;
      params->yblen_luma = 24;
      params->xbsep_luma = 16;
      params->ybsep_luma = 16;
      break;

    default:
      SCHRO_ERROR ("schro_params_init called with video_format_index %d",
          video_format);
      SCHRO_ASSERT (0);
  }
  SCHRO_DEBUG ("schro_params_init %i %i %i %i", params->xblen_luma,
      params->yblen_luma, params->xbsep_luma, params->ybsep_luma);

  params->mv_precision = 2;
  params->picture_weight_1 = 1;
  params->picture_weight_2 = 1;
  params->picture_weight_bits = 1;

  if (params->num_refs == 0) {
    for (i = 0; i < 3; i++) {
      params->horiz_codeblocks[i] = 1;
      params->vert_codeblocks[i] = 1;
    }
    for (i = 3; i < SCHRO_LIMIT_TRANSFORM_DEPTH + 1; i++) {
      params->horiz_codeblocks[i] = 4;
      params->vert_codeblocks[i] = 3;
    }
  } else {
    for (i = 0; i < 2; i++) {
      params->horiz_codeblocks[i] = 1;
      params->vert_codeblocks[i] = 1;
    }
    params->horiz_codeblocks[2] = 8;
    params->vert_codeblocks[2] = 6;
    for (i = 3; i < SCHRO_LIMIT_TRANSFORM_DEPTH + 1; i++) {
      params->horiz_codeblocks[i] = 12;
      params->vert_codeblocks[i] = 8;
    }
  }

  /* other initializations */

  params->codeblock_mode_index = 1;
  params->have_global_motion = FALSE;
  params->picture_pred_mode = 0;
}

/**
 * schro_params_calculate_iwt_sizes:
 * @params: pointer to @SchroParams structure
 *
 * Calculates the size of the array used for wavelet transformation
 * using the current video format and transformation depth in the
 * @params structure.  The @params structure is updated with the new
 * values.
 *
 * The structure fields changed are: iwt_chroma_width, iwt_chroma_height,
 * iwt_luma_width, iwt_luma_height.
 */
void
schro_params_calculate_iwt_sizes (SchroParams * params)
{
  SchroVideoFormat *video_format = params->video_format;
  int picture_luma_width, picture_luma_height;
  int picture_chroma_width, picture_chroma_height;

  schro_video_format_get_picture_luma_size (video_format,
      &picture_luma_width, &picture_luma_height);
  params->iwt_luma_width =
      ROUND_UP_POW2 (picture_luma_width, params->transform_depth);
  params->iwt_luma_height =
      ROUND_UP_POW2 (picture_luma_height, params->transform_depth);

  schro_video_format_get_picture_chroma_size (video_format,
      &picture_chroma_width, &picture_chroma_height);
  params->iwt_chroma_width =
      ROUND_UP_POW2 (picture_chroma_width, params->transform_depth);
  params->iwt_chroma_height =
      ROUND_UP_POW2 (picture_chroma_height, params->transform_depth);

  SCHRO_DEBUG ("iwt chroma size %d x %d", params->iwt_chroma_width,
      params->iwt_chroma_height);
  SCHRO_DEBUG ("iwt luma size %d x %d", params->iwt_luma_width,
      params->iwt_luma_height);
}

/**
 * schro_params_calculate_mc_sizes:
 * @params: pointer to @SchroParams structure
 *
 * Calculates the size of the array used for motion compensation
 * using the current video format and motion compensation paramters
 * in the @params structure.  The @params structure is updated with
 * the new values.
 *
 * The structure fields changed are: x_num_blocks, y_num_blocks,
 * mc_luma_width, mc_luma_height, mc_chroma_width, mc_chroma_height,
 * x_offset, y_offset.
 */
void
schro_params_calculate_mc_sizes (SchroParams * params)
{
  SchroVideoFormat *video_format = params->video_format;
  int width, height;

  schro_video_format_get_picture_luma_size (video_format, &width, &height);

  params->x_num_blocks = 4 * DIVIDE_ROUND_UP (width, 4 * params->xbsep_luma);
  params->y_num_blocks = 4 * DIVIDE_ROUND_UP (height, 4 * params->ybsep_luma);

  SCHRO_DEBUG ("picture %dx%d, num_blocks %dx%d", width, height,
      params->x_num_blocks, params->y_num_blocks);

  params->x_offset = (params->xblen_luma - params->xbsep_luma) / 2;
  params->y_offset = (params->yblen_luma - params->ybsep_luma) / 2;
}

typedef struct _SchroBlockParams SchroBlockParams;
struct _SchroBlockParams
{
  int xblen_luma;
  int yblen_luma;
  int xbsep_luma;
  int ybsep_luma;
};

static SchroBlockParams schro_block_params[] = {
  {0, 0, 0, 0},
  {8, 8, 4, 4},
  {12, 12, 8, 8},
  {16, 16, 12, 12},
  {24, 24, 16, 16}
};

/**
 * schro_params_set_block_params:
 * @params: pointer to SchroParams structure
 * @index: index to standard block parameters
 *
 * Sets the block parameters for motion compensation in the parameters
 * structure pointed to by @params to the
 * standard block parameters given by @index.
 */
int
schro_params_set_block_params (SchroParams * params, int index)
{
  if (index < 1 || index >= ARRAY_SIZE (schro_block_params)) {
    SCHRO_ERROR ("illegal block params index");
    return FALSE;
  }

  params->xblen_luma = schro_block_params[index].xblen_luma;
  params->yblen_luma = schro_block_params[index].yblen_luma;
  params->xbsep_luma = schro_block_params[index].xbsep_luma;
  params->ybsep_luma = schro_block_params[index].ybsep_luma;

  return TRUE;
}

int
schro_params_get_block_params (SchroParams * params)
{
  int i;
  for (i = 1; i < ARRAY_SIZE (schro_block_params); i++) {
    if (schro_block_params[i].xblen_luma == params->xblen_luma &&
        schro_block_params[i].xbsep_luma == params->xbsep_luma &&
        schro_block_params[i].yblen_luma == params->yblen_luma &&
        schro_block_params[i].ybsep_luma == params->ybsep_luma) {
      return i;
    }
  }
  return 0;
}

int
schro_params_verify_block_params (SchroParams * params)
{
  if (params->xblen_luma < 0)
    return FALSE;
  if (params->yblen_luma < 0)
    return FALSE;
  if (params->xbsep_luma < 0)
    return FALSE;
  if (params->ybsep_luma < 0)
    return FALSE;
  if (params->xblen_luma > SCHRO_LIMIT_BLOCK_SIZE)
    return FALSE;
  if (params->yblen_luma > SCHRO_LIMIT_BLOCK_SIZE)
    return FALSE;
  if ((params->xblen_luma & 3) != 0)
    return FALSE;
  if ((params->xbsep_luma & 3) != 0)
    return FALSE;
  if ((params->yblen_luma & 3) != 0)
    return FALSE;
  if ((params->ybsep_luma & 3) != 0)
    return FALSE;
  if (params->xblen_luma < params->xbsep_luma)
    return FALSE;
  if (params->yblen_luma < params->ybsep_luma)
    return FALSE;
  if (params->xblen_luma > 2 * params->xbsep_luma)
    return FALSE;
  if (params->yblen_luma > 2 * params->ybsep_luma)
    return FALSE;

  return TRUE;
}

#ifdef unused
/**
 * schro_params_set_default_codeblock:
 * @params: pointer to SchroParams structure
 *
 * Sets the codeblock parameters in the parameters structure pointed to
 * by @params to the defaults.
 */
void
schro_params_set_default_codeblock (SchroParams * params)
{
  int i;

  for (i = 0; i < SCHRO_LIMIT_TRANSFORM_DEPTH + 1; i++) {
    params->horiz_codeblocks[i] = 1;
    params->vert_codeblocks[i] = 1;
  }
  /* FIXME default in Dirac is 0 */
  params->codeblock_mode_index = 1;
}
#endif

/**
 * schro_params_is_default_codeblock:
 * @params: pointer to SchroParams structure
 *
 * Returns: True if the codeblocks in @params are the default codeblocks.
 */
schro_bool
schro_params_is_default_codeblock (SchroParams * params)
{
  int i;

  for (i = 0; i < params->transform_depth + 1; i++) {
    if (params->horiz_codeblocks[i] != 1 || params->vert_codeblocks[i] != 1) {
      return FALSE;
    }
  }
  if (params->codeblock_mode_index != 0)
    return FALSE;

  return TRUE;
}

void
schro_subband_get_frame_data (SchroFrameData * fd,
    SchroFrame * frame, int component, int position, SchroParams * params)
{
  int shift;
  SchroFrameData *comp = &frame->components[component];

  shift = params->transform_depth - SCHRO_SUBBAND_SHIFT (position);

  fd->format = frame->format;
  fd->h_shift = comp->h_shift + shift;
  fd->v_shift = comp->v_shift + shift;
  fd->stride = comp->stride << shift;
  if (component == 0) {
    fd->width = params->iwt_luma_width >> shift;
    fd->height = params->iwt_luma_height >> shift;
  } else {
    fd->width = params->iwt_chroma_width >> shift;
    fd->height = params->iwt_chroma_height >> shift;
  }

  fd->data = comp->data;
  if (position & 2) {
    fd->data = OFFSET (fd->data, fd->stride >> 1);
  }
  if (position & 1) {
    fd->data = OFFSET (fd->data, fd->width * sizeof (int16_t));
  }
}

int
schro_subband_get_position (int index)
{
  const int subband_position[] = {
    0, 1, 2, 3,
    5, 6, 7,
    9, 10, 11,
    13, 14, 15,
    17, 18, 19,
    21, 22, 23,
    25, 26, 27
  };

  return subband_position[index];
}

int
schro_params_get_frame_format (int depth, SchroChromaFormat chroma_format)
{
  if (depth == 8) {
    switch (chroma_format) {
      case SCHRO_CHROMA_444:
        return SCHRO_FRAME_FORMAT_U8_444;
      case SCHRO_CHROMA_422:
        return SCHRO_FRAME_FORMAT_U8_422;
      case SCHRO_CHROMA_420:
        return SCHRO_FRAME_FORMAT_U8_420;
      default:
        SCHRO_ASSERT (0);
    }
  } else if (depth == 16) {
    switch (chroma_format) {
      case SCHRO_CHROMA_444:
        return SCHRO_FRAME_FORMAT_S16_444;
      case SCHRO_CHROMA_422:
        return SCHRO_FRAME_FORMAT_S16_422;
      case SCHRO_CHROMA_420:
        return SCHRO_FRAME_FORMAT_S16_420;
      default:
        SCHRO_ASSERT (0);
    }
  }

  SCHRO_ASSERT (0);
}


const int schro_tables_lowdelay_quants[7][4][9] = {
  {                             /* wavelet 0 */
        {5, 3, 0},
        {5, 3, 0, 4, 1},
        {5, 3, 0, 4, 1, 5, 2},
        {5, 3, 0, 4, 1, 5, 2, 6, 3},
      },
  {                             /* wavelet 1 */
        {4, 2, 0},
        {4, 2, 0, 4, 2},
        {4, 2, 0, 4, 2, 5, 3},
        {4, 2, 0, 4, 2, 5, 3, 7, 5},
      },
  {                             /* wavelet 2 */
        {5, 3, 0},
        {5, 3, 0, 4, 1},
        {5, 3, 0, 4, 1, 5, 2},
        {5, 3, 0, 4, 1, 5, 2, 6, 3},
      },
  {                             /* wavelet 3 */
        {8, 4, 0},
        {12, 8, 4, 4, 0},
        {16, 12, 8, 8, 4, 4, 0},
        {20, 16, 12, 12, 8, 8, 4, 4, 0},
      },
  {                             /* wavelet 4 */
        {8, 4, 0},
        {8, 4, 0, 4, 0},
        {8, 4, 0, 4, 0, 4, 0},
        {8, 4, 0, 4, 0, 4, 0, 4, 0},
      },
  {                             /* wavelet 5 */
        {0, 4, 8},
        {0, 4, 8, 8, 10},
        {0, 4, 8, 8, 12, 13, 17},
        {0, 4, 8, 8, 12, 13, 17, 17, 21},
      },
  {                             /* wavelet 6 */
        {3, 1, 0},
        {3, 1, 0, 4, 2},
        {3, 1, 0, 4, 2, 6, 5},
        {3, 1, 0, 4, 2, 6, 5, 9, 7},
      },
};

void
schro_params_set_default_quant_matrix (SchroParams * params)
{
  int i;
  const int *table;

  table = schro_tables_lowdelay_quants[params->wavelet_filter_index]
      [MAX (0, params->transform_depth - 1)];

  params->quant_matrix[0] = table[0];
  for (i = 0; i < params->transform_depth; i++) {
    params->quant_matrix[1 + 3 * i + 0] = table[1 + 2 * i + 0];
    params->quant_matrix[1 + 3 * i + 1] = table[1 + 2 * i + 0];
    params->quant_matrix[1 + 3 * i + 2] = table[1 + 2 * i + 1];
  }
}

schro_bool
schro_params_is_default_quant_matrix (SchroParams * params)
{
  int i;
  const int *table;

  if (params->transform_depth < 1 || params->transform_depth > 4)
    return FALSE;

  table = schro_tables_lowdelay_quants[params->wavelet_filter_index]
      [params->transform_depth - 1];

  if (params->quant_matrix[0] != table[0])
    return FALSE;
  for (i = 0; i < params->transform_depth; i++) {
    if (params->quant_matrix[1 + 3 * i + 0] != table[1 + 2 * i + 0] ||
        params->quant_matrix[1 + 3 * i + 1] != table[1 + 2 * i + 0] ||
        params->quant_matrix[1 + 3 * i + 2] != table[1 + 2 * i + 1]) {
      return FALSE;
    }
  }
  return TRUE;
}
