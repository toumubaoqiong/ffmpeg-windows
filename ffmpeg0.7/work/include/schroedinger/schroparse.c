
#include "config.h"

#include <schroedinger/schroparse.h>
#include <schroedinger/schrodebug.h>
#include <schroedinger/schrounpack.h>
#include <schroedinger/schrodecoder.h>

int
schro_parse_decode_sequence_header (uint8_t * data, int length,
    SchroVideoFormat * format)
{
  int bit;
  int index;
  SchroUnpack u;
  SchroUnpack *unpack = &u;
  int major_version, minor_version;
  int profile, level;

  SCHRO_DEBUG ("decoding sequence header");

  schro_unpack_init_with_data (unpack, data, length, 1);

  /* parse parameters */
  major_version = schro_unpack_decode_uint (unpack);
  SCHRO_DEBUG ("major_version = %d", major_version);
  minor_version = schro_unpack_decode_uint (unpack);
  SCHRO_DEBUG ("minor_version = %d", minor_version);
  profile = schro_unpack_decode_uint (unpack);
  SCHRO_DEBUG ("profile = %d", profile);
  level = schro_unpack_decode_uint (unpack);
  SCHRO_DEBUG ("level = %d", level);

#if 0
  if (!schro_decoder_check_version (major_version, minor_version)) {
    SCHRO_WARNING
        ("Stream version number %d:%d not handled, expecting 0:20071203, 1:0, 2:0, or 2:1",
        major_version, minor_version);
  }
#endif
  if (profile != 0 || level != 0) {
    SCHRO_WARNING ("Expecting profile/level 0:0, got %d:%d", profile, level);
  }

  /* base video format */
  index = schro_unpack_decode_uint (unpack);
  schro_video_format_set_std_video_format (format, index);

  /* source parameters */
  /* frame dimensions */
  bit = schro_unpack_decode_bit (unpack);
  if (bit) {
    format->width = schro_unpack_decode_uint (unpack);
    format->height = schro_unpack_decode_uint (unpack);
  }
  SCHRO_DEBUG ("size = %d x %d", format->width, format->height);

  /* chroma format */
  bit = schro_unpack_decode_bit (unpack);
  if (bit) {
    format->chroma_format = schro_unpack_decode_uint (unpack);
  }
  SCHRO_DEBUG ("chroma_format %d", format->chroma_format);

  /* scan format */
  bit = schro_unpack_decode_bit (unpack);
  if (bit) {
    format->interlaced = schro_unpack_decode_uint (unpack);
  }
  SCHRO_DEBUG ("interlaced %d top_field_first %d",
      format->interlaced, format->top_field_first);

  /* frame rate */
  bit = schro_unpack_decode_bit (unpack);
  if (bit) {
    index = schro_unpack_decode_uint (unpack);
    if (index == 0) {
      format->frame_rate_numerator = schro_unpack_decode_uint (unpack);
      format->frame_rate_denominator = schro_unpack_decode_uint (unpack);
    } else {
      schro_video_format_set_std_frame_rate (format, index);
    }
  }
  SCHRO_DEBUG ("frame rate %d/%d", format->frame_rate_numerator,
      format->frame_rate_denominator);

  /* aspect ratio */
  bit = schro_unpack_decode_bit (unpack);
  if (bit) {
    index = schro_unpack_decode_uint (unpack);
    if (index == 0) {
      format->aspect_ratio_numerator = schro_unpack_decode_uint (unpack);
      format->aspect_ratio_denominator = schro_unpack_decode_uint (unpack);
    } else {
      schro_video_format_set_std_aspect_ratio (format, index);
    }
  }
  SCHRO_DEBUG ("aspect ratio %d/%d", format->aspect_ratio_numerator,
      format->aspect_ratio_denominator);

  /* clean area */
  bit = schro_unpack_decode_bit (unpack);
  if (bit) {
    format->clean_width = schro_unpack_decode_uint (unpack);
    format->clean_height = schro_unpack_decode_uint (unpack);
    format->left_offset = schro_unpack_decode_uint (unpack);
    format->top_offset = schro_unpack_decode_uint (unpack);
  }
  SCHRO_DEBUG ("clean offset %d %d", format->left_offset, format->top_offset);
  SCHRO_DEBUG ("clean size %d %d", format->clean_width, format->clean_height);

  /* signal range */
  bit = schro_unpack_decode_bit (unpack);
  if (bit) {
    index = schro_unpack_decode_uint (unpack);
    if (index == 0) {
      format->luma_offset = schro_unpack_decode_uint (unpack);
      format->luma_excursion = schro_unpack_decode_uint (unpack);
      format->chroma_offset = schro_unpack_decode_uint (unpack);
      format->chroma_excursion = schro_unpack_decode_uint (unpack);
    } else {
      if (index <= SCHRO_SIGNAL_RANGE_12BIT_VIDEO) {
        schro_video_format_set_std_signal_range (format, index);
      } else {
        SCHRO_DEBUG ("signal range index %d out of range", index);
        return FALSE;
      }
    }
  }
  SCHRO_DEBUG ("luma offset %d excursion %d", format->luma_offset,
      format->luma_excursion);
  SCHRO_DEBUG ("chroma offset %d excursion %d", format->chroma_offset,
      format->chroma_excursion);

  /* colour spec */
  bit = schro_unpack_decode_bit (unpack);
  if (bit) {
    index = schro_unpack_decode_uint (unpack);
    if (index <= SCHRO_COLOUR_SPEC_CINEMA) {
      schro_video_format_set_std_colour_spec (format, index);
    } else {
      SCHRO_DEBUG ("colour spec index %d out of range", index);
      return FALSE;
    }
    if (index == 0) {
      /* colour primaries */
      bit = schro_unpack_decode_bit (unpack);
      if (bit) {
        format->colour_primaries = schro_unpack_decode_uint (unpack);
      }
      /* colour matrix */
      bit = schro_unpack_decode_bit (unpack);
      if (bit) {
        format->colour_matrix = schro_unpack_decode_uint (unpack);
      }
      /* transfer function */
      bit = schro_unpack_decode_bit (unpack);
      if (bit) {
        format->transfer_function = schro_unpack_decode_uint (unpack);
      }
    }
  }

  format->interlaced_coding = schro_unpack_decode_uint (unpack);

  schro_video_format_validate (format);

  return TRUE;
}

typedef struct parse_info
{
  uint32_t next_parse_offset;
  uint32_t prev_parse_offset;
  uint_least8_t parse_code;
} parse_info_t;

/**
 * schro_parse_decode_parseinfo:
 *
 * decodes a parse info structure aligned at the start of *@data@,
 * of maximal length @length@, storing the results in *@pi@.
 *
 * Returns 0 if decoding is unsuccessful; 1 on success
 */
static int
schro_parse_decode_parseinfo (uint8_t * data, unsigned length,
    parse_info_t * pi)
{
  if (length < 13) {
    return 0;
  }

  if (data[0] != 'B' || data[1] != 'B' || data[2] != 'C' || data[3] != 'D') {
    return 0;
  }

  pi->parse_code = data[4];
  pi->next_parse_offset =
      data[5] << 24 | data[6] << 16 | data[7] << 8 | data[8];
  pi->prev_parse_offset =
      data[9] << 24 | data[10] << 16 | data[11] << 8 | data[12];
  return 1;
}

struct _SchroParseSyncState
{
  /* <private> */
  int sync_state;
  unsigned offset;
  uint32_t last_npo;
  int done_special_startup;
};

enum
{
  NOT_SYNCED = 0,
  TRY_SYNC,
  SYNCED,
  SYNCED_INCOMPLETEDU,
};

/**
 * schro_parse_sync_new:
 *
 * returns: NULL or pointer to allocated SchroParseSyncState
 */
SchroParseSyncState *
schro_parse_sync_new (void)
{
  return schro_malloc0 (sizeof (SchroParseSyncState));
}

/**
 * schro_parse_sync_free:
 *
 * release storage for an allocated SchroParseSyncState
 */
void
schro_parse_sync_free (SchroParseSyncState * sps)
{
  schro_free (sps);
}

/**
 * schro_parse_sync:
 *
 * Synchronises to and extracts single data units from @buflist@.
 * Synchronisation state is updated through @sps@.
 *
 * Returns: NULL if no data unit may be extracted; or pointer to
 *          SchroBuffer containing a single data unit.
 */
SchroBuffer *
schro_parse_sync (SchroParseSyncState * sps, SchroBufferList * buflist)
{
  uint8_t tmp[13];
  const uint8_t *parse_code_prefix = (const uint8_t *) "BBCD";
  parse_info_t pu = { 0 };
  SchroBuffer *du;

  do {
    switch (sps->sync_state) {
      case NOT_SYNCED:{        /* -> TRY_SYNC | NOT_SYNCED */
        /* find start code (offset), stop so as to include a whole PI */
        int found =
            schro_buflist_findbytes (buflist, &sps->offset, parse_code_prefix,
            4);
        /* xxx, worth flushing upto this point, although that is
         * quite complicated. */
        if (!found) {
          return NULL;
        }
        /* protect the case where there isn't a whole parse_info avaliable.
         * eagerly read parse_info into tmp, it'll be required shortly */
        if (!schro_buflist_peekbytes (tmp, 13, buflist, sps->offset)) {
          return NULL;
        }
        while (!sps->done_special_startup && !sps->offset) {
          uint8_t c;
          sps->done_special_startup = TRUE;
          /* special startup case: the very first buffer may consist of a
           * single data unit, (usually a seqhdr), to aleviate waiting for
           * two parse_infos to arrive, assume that we are synced IFF the
           * next_parse_offset <= length(buflist) */
          if (!schro_parse_decode_parseinfo (tmp, 13, &pu)) {
            break;
          }
          if (!schro_buflist_peekbytes (&c, 1, buflist,
                  sps->offset + pu.next_parse_offset - 1)) {
            break;
          }
          sps->last_npo = pu.next_parse_offset;
          sps->sync_state = SYNCED;
          goto extract;
        }
        /* found, fall through */
      }
      case TRY_SYNC:{          /* -> SYNCED | NOT_SYNCED */
        parse_info_t pu1;
        /* tmp is still valid from NOT_SYNCED case */
        if (!schro_parse_decode_parseinfo (tmp, 13, &pu1)) {
          goto try_sync_fail;
        }
        /* Check that prev_parse_offset doesn't reference something not yet seen */
        if (sps->offset < pu1.prev_parse_offset) {
          goto try_sync_fail;
        }
        /* NB, guaranteed that there are 13 bytes avaliable */
        schro_buflist_peekbytes (tmp, 13, buflist,
            sps->offset - pu1.prev_parse_offset);
        if (!schro_parse_decode_parseinfo (tmp, 13, &pu)) {
          goto try_sync_fail;
        }
        if (pu1.prev_parse_offset != pu.next_parse_offset) {
        try_sync_fail:
          sps->sync_state = NOT_SYNCED;
          sps->offset++;
          /* find somewhere else to try again */
          break;
        }
        sps->last_npo = pu.next_parse_offset;
        /* offset was pointing at pu1, rewind to point at pu */
        sps->offset -= pu.next_parse_offset;
        sps->sync_state = SYNCED;
        break;
      }
      case SYNCED:{            /* -> SYNCED | SYNCED_INCOMPLETEDU | NOT_SYNCED */
        int a;
        if (schro_buflist_peekbytes (tmp, 13, buflist, sps->offset) < 13)
          return NULL;
        a = schro_parse_decode_parseinfo (tmp, 13, &pu);
        if (!a || (sps->last_npo != pu.prev_parse_offset)) {
          sps->sync_state = NOT_SYNCED;
          break;
        }
        sps->last_npo = pu.next_parse_offset;
        sps->sync_state = SYNCED;
        break;
      }
      case SYNCED_INCOMPLETEDU:{
        /* -> SYNCED */
        /* NB, this is safe -- to get here we must've already read pu
         * previously, so no need to check that it is ok again */
        schro_buflist_peekbytes (tmp, 13, buflist, sps->offset);
        schro_parse_decode_parseinfo (tmp, 13, &pu);
        sps->sync_state = SYNCED;
        /* assume that the DU is complete this time */
        break;
      default:
        SCHRO_ASSERT (0);
      }
    }
  } while (NOT_SYNCED == sps->sync_state);
extract:
  /*
   * synced, attempt to extract a data unit
   */

  /* fixup for case where pu.next_parse_offset = 0 (eg, EOS) */
  if (!pu.next_parse_offset) {
    pu.next_parse_offset = 13;
  }

  /* flush everything upto the DU */
  schro_buflist_flush (buflist, sps->offset);
  sps->offset = 0;

  /* try to extract the complete DU */
  du = schro_buflist_extract (buflist, 0, pu.next_parse_offset);
  if (!du) {
    /* the whole DU isn't in the buffer, try again */
    sps->sync_state = SYNCED_INCOMPLETEDU;
    return NULL;
  }

  /* flush everything upto the end of DU */
  schro_buflist_flush (buflist, pu.next_parse_offset);

  return du;
}
