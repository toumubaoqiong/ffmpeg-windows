
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schroutils.h>
#include <schroedinger/schrobufferlist.h>
#include <schroedinger/schrodebug.h>
#include <string.h>

/**
 * schro_buflist_new:
 *
 * Creates a new (empty) buffer list; this should be freed
 * using @schro_buflist_free() when nolonger required.
 *
 * A buffer list allows:
 *  - extraction of a range of bytes
 *  - searching for a sequence of bytes
 *  From a set of non-contiguous buffers.
 *
 * Returns: a new (empty) buffer list.
 */
SchroBufferList *
schro_buflist_new (void)
{
  SchroBufferList *buflist = schro_malloc0 (sizeof (*buflist));
  buflist->list =
      schro_list_new_full ((SchroListFreeFunc) schro_buffer_unref, NULL);
  buflist->offset = 0;

  return buflist;
}

/**
 * schro_buflist_free:
 *
 * Frees resources associated with @buflist@.
 */
void
schro_buflist_free (SchroBufferList * buflist)
{
  if (!buflist) {
    return;
  }
  schro_list_free (buflist->list);
  if (buflist->tag)
    schro_tag_free (buflist->tag);
  schro_free (buflist);
}

/**
 * schro_buflist_append:
 *
 * Append @buf@ to the end of the buffer list.
 */
void
schro_buflist_append (SchroBufferList * buflist, SchroBuffer * buf)
{
  schro_list_append (buflist->list, buf);
}

/**
 * schro_buflist_internal_seek:
 *
 * xxx
 */
static int
schro_buflist_internal_seek (SchroBufferList * buflist, unsigned *offset)
{
  int bufidx;
  /* skip over any discarded (consumed) data at start of buflist */
  *offset += buflist->offset;

  /* find buffer @offset@ starts in */
  for (bufidx = 0; bufidx < buflist->list->n; bufidx++) {
    SchroBuffer *buf = buflist->list->members[bufidx];
    if (*offset < buf->length) {
      /* found */
      break;
    }
    *offset -= buf->length;
  }

  return bufidx;
}

/**
 * schro_buflist_peekbytes:
 *
 * Extracts @len@ bytes (without marking them as consumed) to be stored
 * in @dst@ commencing at @offset@ bytes into the buflist.
 *
 * NB, @*dst@ must have sufficient storage for @len@ bytes.
 *
 * Returns: Number of bytes copied
 */
int
schro_buflist_peekbytes (uint8_t * dst, unsigned len, SchroBufferList * buflist,
    unsigned offset)
{
  int bufidx;
  int coppied = 0;

  if (!dst || !len) {
    return 0;
  }

  bufidx = schro_buflist_internal_seek (buflist, &offset);

  /* copy out upto len worth of bytes. NB, this may cross
   * buffer boundries */
  for (; bufidx < buflist->list->n; bufidx++) {
    SchroBuffer *buf = buflist->list->members[bufidx];
    unsigned size = MIN (len, buf->length - offset);
    memcpy (dst + coppied, buf->data + offset, size);
    offset = 0;
    coppied += size;
    len -= size;
    if (!len) {
      break;
    }
  }

  return coppied;
}

/**
 * schro_buflist_findbytes:
 *
 * Searches @buflist@ for @needle_len@ bytes of @needle[]@, commencing at
 * @*start@.
 *
 * if needle is found, @*start@ is updated to contain the offset to the start of the found
 * needle.
 *
 * Returns: non-zero if found
 */
int
schro_buflist_findbytes (SchroBufferList * buflist, unsigned *start,
    const uint8_t * needle, unsigned needle_len)
{
  SchroBuffer *buf;
  unsigned bufidx = 0;
  unsigned where = *start;
  unsigned offset = *start;
  unsigned n = 0;
  unsigned backtrack_bufidx = 0, backtrack_where = 0, backtrack_i = 0;

  if (!needle || !needle_len) {
    return 0;
  }

  bufidx = schro_buflist_internal_seek (buflist, &offset);

  /* search for what[], NB this may span multiple buffers */
  /* NB, if a search fails, it must be backtracked */
  for (; bufidx < buflist->list->n; bufidx++) {
    /* todo, maybe add memmem call ? */
    unsigned i = offset;
    buf = buflist->list->members[bufidx];
    for (; i < buf->length; i++) {
      if (needle[n] == buf->data[i]) {
        if (!n) {
          /* save back tracking point */
          backtrack_where = where;
          backtrack_i = i;
          backtrack_bufidx = bufidx;
        }
        n++;
        if (n == needle_len) {
          *start = backtrack_where;
          return 1;
        }
      } else if (n) {
        n = 0;
        /* restore backtracking point */
        i = backtrack_i;
        where = backtrack_where;
        bufidx = backtrack_bufidx;
      }
    }
    where += buf->length - offset;
    offset = 0;
  }

  /* not found */
  /* NB, to make it possible to resume a search where this failed,
   * must move where back by (needle_len-1), but don't go past start */
  /* but avoid unsigned wraparound */
  if (needle_len <= where)
    *start = MAX (where - needle_len + 1, *start);
  return 0;
}

/**
 * schro_buflist_flush:
 *
 * flushes (consumes) @ammount@ bytes.
 */
void
schro_buflist_flush (SchroBufferList * buflist, unsigned amount)
{
  SchroBuffer *buf;
  int bufidx = 0;

  /* include any discarded data at start of buflist */
  buflist->offset += amount;

  /* pop and unref all buffers that end before ammount */
  for (bufidx = 0; bufidx < buflist->list->n; bufidx++) {
    buf = buflist->list->members[bufidx];
    if (buflist->offset < buf->length) {
      /* found */
      break;
    }
    buflist->offset -= buf->length;
    schro_list_delete (buflist->list, 0);
    bufidx--;
  }
}

/**
 * schro_buflist_extract:
 *
 * extract @len@ bytes from @buflist@, commencing at @start@; stored
 * in a SchroBuffer.
 *
 * This operation does not flush (consume) any of the buffer list,
 * call @schro_buflist_flush to do so.
 *
 * NB, the returned buffer may be a subbuffer.
 *
 * Returns: SchroBuffer* or NULL on failure.
 */
SchroBuffer *
schro_buflist_extract (SchroBufferList * buflist, unsigned start, unsigned len)
{
  SchroBuffer *buf, *dst;
  SchroTag *tag = NULL;
  unsigned pos = start;
  int bufidx;
  uint8_t tmp;

  SCHRO_ASSERT (buflist);

  if (!len) {
    return NULL;
  }

  /* first check that the (start + len)th byte is avaliable */
  if (!schro_buflist_peekbytes (&tmp, 1, buflist, start + len - 1)) {
    return NULL;
  }
  /* guaranteed that the range is wholly contained within buflist */

  bufidx = schro_buflist_internal_seek (buflist, &pos);
  SCHRO_ASSERT (bufidx < buflist->list->n);

  buf = buflist->list->members[bufidx];

  /* Semantics for private void* in buffers in a bufferlist:
   *  - Shall be associated with the next dataunit to commence after (or at)
   *    the start of the buffer in the bufferist.
   *  - May only be extracted once.
   *  - If multiple void* could apply to the next dataunit, behaviour is undefined
   *    NB: this implementation will discard all but the first.
   *  - Ownership of the tag is transfered to the output buffer
   */
  if (buflist->tag) {
    /* take a previously discovered tag */
    tag = buflist->tag;
    buflist->tag = NULL;
  } else {
    /* fall back and try to take from the buffer */
    tag = buf->tag;
    buf->tag = NULL;
  }

  if (pos + len <= buf->length) {
    /* Special case, if the requested range is contained within a single
     * buffer, then use a subbuffer */
    dst = schro_buffer_new_subbuffer (buf, pos, len);
    dst->tag = tag;
    return dst;
  }

  /* dataunit spans multiple buffers */
  dst = schro_buffer_new_and_alloc (len);
  dst->tag = tag;
  schro_buflist_peekbytes (dst->data, len, buflist, start);

  /* sort out rescuing the first tag that was in the extracted
   * region and saving it ready for extraction at start+len. */
  len += pos;
  for (pos = 0; pos < len;) {
    buf = buflist->list->members[bufidx];
    if (!tag) {
      /* find the first non null tag we come across, take it,
       * and store in internal state */
      buflist->tag = buf->tag;
      buf->tag = NULL;
    }
    pos += buf->length;
    bufidx++;
  }

  return dst;
}
