
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <string.h>

#include <schroedinger/schropack.h>
#include <schroedinger/schro.h>
#include <schroedinger/schroorc.h>
#include <orc/orc.h>


SchroPack *
schro_pack_new (void)
{
  SchroPack *pack;

  pack = schro_malloc0 (sizeof (*pack));

  return pack;
}

void
schro_pack_free (SchroPack * pack)
{
  schro_free (pack);
}

#ifdef unused
void
schro_pack_copy (SchroPack * dest, SchroPack * src)
{
  memcpy (dest, src, sizeof (SchroPack));
}
#endif

static void
schro_pack_shift_out (SchroPack * pack)
{
  if (pack->n < pack->buffer->length) {
    pack->buffer->data[pack->n] = pack->value;
    pack->n++;
    pack->shift = 7;
    pack->value = 0;
    return;
  }
  if (pack->error == FALSE) {
    SCHRO_ERROR ("buffer overrun");
  }
  pack->error = TRUE;
  pack->shift = 7;
  pack->value = 0;
}

void
schro_pack_encode_init (SchroPack * pack, SchroBuffer * buffer)
{
  pack->buffer = buffer;
  pack->n = 0;

  pack->value = 0;
  pack->shift = 7;
}

int
schro_pack_get_offset (SchroPack * pack)
{
  return pack->n;
}

int
schro_pack_get_bit_offset (SchroPack * pack)
{
  return pack->n * 8 + (7 - pack->shift);
}

void
schro_pack_flush (SchroPack * pack)
{
  schro_pack_sync (pack);
}

void
schro_pack_sync (SchroPack * pack)
{
  if (pack->shift != 7) {
    schro_pack_shift_out (pack);
  }
}

void
schro_pack_append (SchroPack * pack, const uint8_t * data, int len)
{
  if (pack->shift != 7) {
    SCHRO_ERROR ("appending to unsyncronized pack");
  }

  SCHRO_ASSERT (pack->n + len <= pack->buffer->length);

  orc_memcpy (pack->buffer->data + pack->n, (void *) data, len);
  pack->n += len;
}

void
schro_pack_append_zero (SchroPack * pack, int len)
{
  if (pack->shift != 7) {
    SCHRO_ERROR ("appending to unsyncronized pack");
  }

  SCHRO_ASSERT (pack->n + len <= pack->buffer->length);

  memset (pack->buffer->data + pack->n, 0, len);
  pack->n += len;
}

void
schro_pack_encode_bit (SchroPack * pack, int value)
{
  value &= 1;
  pack->value |= (value << pack->shift);
  pack->shift--;
  if (pack->shift < 0) {
    schro_pack_shift_out (pack);
  }
}

void
schro_pack_encode_bits (SchroPack * pack, int n, unsigned int value)
{
  int i;
  for (i = 0; i < n; i++) {
    schro_pack_encode_bit (pack, (value >> (n - 1 - i)) & 1);
  }
}

static int
maxbit (unsigned int x)
{
  int i;
  for (i = 0; x; i++) {
    x >>= 1;
  }
  return i;
}

void
schro_pack_encode_uint (SchroPack * pack, int value)
{
  int i;
  int n_bits;

  value++;
  n_bits = maxbit (value);
  for (i = 0; i < n_bits - 1; i++) {
    schro_pack_encode_bit (pack, 0);
    schro_pack_encode_bit (pack, (value >> (n_bits - 2 - i)) & 1);
  }
  schro_pack_encode_bit (pack, 1);
}

void
schro_pack_encode_sint (SchroPack * pack, int value)
{
  int sign;

  if (value < 0) {
    sign = 1;
    value = -value;
  } else {
    sign = 0;
  }
  schro_pack_encode_uint (pack, value);
  if (value) {
    schro_pack_encode_bit (pack, sign);
  }
}

int
schro_pack_estimate_uint (int value)
{
  int n_bits;

  value++;
  n_bits = maxbit (value);
  return n_bits + n_bits - 1;
}

int
schro_pack_estimate_sint (int value)
{
  int n_bits;

  if (value < 0) {
    value = -value;
  }
  n_bits = schro_pack_estimate_uint (value);
  if (value)
    n_bits++;
  return n_bits;
}
