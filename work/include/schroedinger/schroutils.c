
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schroutils.h>
#include <schroedinger/schrotables.h>
#include <schroedinger/schrodebug.h>
#include <schroedinger/schro-stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#ifndef SCHRO_MALLOC_USE_MMAP
/* enable this for mmap based buffer overrun checks */
//# define SCHRO_MALLOC_USE_MMAP
/* enable this if there are alignment issues */
//# define ALIGN_16
#endif

#ifdef _WIN32
#include <windows.h>
#undef SCHRO_MALLOC_USE_MMAP
#endif

#ifdef SCHRO_MALLOC_USE_MMAP
#include <sys/mman.h>
#endif

#ifndef SCHRO_MALLOC_USE_MMAP
void *
schro_malloc (int size)
{
  void *ptr;

  ptr = malloc (size);
  SCHRO_DEBUG ("alloc %p %d", ptr, size);

  return ptr;
}

void *
schro_malloc0 (int size)
{
  void *ptr;

  ptr = malloc (size);
  memset (ptr, 0, size);
  SCHRO_DEBUG ("alloc %p %d", ptr, size);

  return ptr;
}

void *
schro_realloc (void *ptr, int size)
{
  ptr = realloc (ptr, size);
  SCHRO_DEBUG ("realloc %p %d", ptr, size);

  return ptr;
}

void
schro_free (void *ptr)
{
  SCHRO_DEBUG ("free %p", ptr);
  free (ptr);
}
#else /* SCHRO_MALLOC_USE_MMAP */

static const char sentinel[] = "This came from schro";

void *
schro_malloc (int size)
{
  void *ptr;
  int rsize;

#ifdef ALIGN_16
  size = ROUND_UP_POW2 (size, 4);
#endif

  rsize = ROUND_UP_POW2 (size + sizeof (int) + sizeof (sentinel), 12);
  ptr =
      mmap (NULL, rsize + 8192, PROT_READ | PROT_WRITE,
      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  SCHRO_ASSERT (ptr != MAP_FAILED);

  mprotect (ptr, 4096, PROT_NONE);
  mprotect (OFFSET (ptr, 4096 + rsize), 4096, PROT_NONE);

  SCHRO_DEBUG ("alloc %p %d", ptr, size);

  *(int *) OFFSET (ptr, 4096) = rsize;
  memcpy (OFFSET (ptr, 4096 + sizeof (int)), sentinel, sizeof (sentinel));

  return OFFSET (ptr, 4096 + rsize - size);
}

void *
schro_malloc0 (int size)
{
  return schro_malloc (size);
}

void *
schro_realloc (void *ptr, int size)
{
  unsigned long page = ((unsigned long) ptr) & ~(4095);
  int rsize, old_size;

  if (!ptr)
    return schro_malloc (size);

  /* find original size */
  if ((unsigned long) ptr - page <= sizeof (int) + sizeof (sentinel)) {
    /* if ptr is too close to start of page, then the base pointer is
     * the previous page */
    page -= 4096;
  }
  rsize = *(int *) page;
  old_size = page + rsize - (unsigned long) ptr;;

  void *new = schro_malloc (size);
  if (size < old_size)
    memcpy (new, ptr, size);
  else
    memcpy (new, ptr, old_size);

  schro_free (ptr);

  return new;
}

void
schro_free (void *ptr)
{
  unsigned long page = ((unsigned long) ptr) & ~(4095);
  int rsize;

  if ((unsigned long) ptr - page <= sizeof (int) + sizeof (sentinel)) {
    /* if ptr is too close to start of page, then the base pointer is
     * the previous page */
    page -= 4096;
  }

  rsize = *(int *) page;

  SCHRO_ASSERT (!memcmp ((void *) page + sizeof (int), sentinel,
          sizeof (sentinel)));

  munmap ((void *) (page - 4096), rsize + 8192);
}
#endif /* SCHRO_MALLOC_USE_MMAP */

int
muldiv64 (int a, int b, int c)
{
  int64_t x;

  x = a;
  x *= b;
  x /= c;

  return (int) x;
}

int
schro_utils_multiplier_to_quant_index (double x)
{
  return CLAMP (rint (log (x) / log (2) * 4.0), 0, 60);
}


static int
__schro_dequantise (int q, int quant_factor, int quant_offset)
{
  if (q == 0)
    return 0;
  if (q < 0) {
    return -((-q * quant_factor + quant_offset + 2) >> 2);
  } else {
    return (q * quant_factor + quant_offset + 2) >> 2;
  }
}

int
schro_dequantise (int q, int quant_factor, int quant_offset)
{
  return __schro_dequantise (q, quant_factor, quant_offset);
}

static int
__schro_quantise (int value, int quant_factor, int quant_offset)
{
  int x;
  int dead_zone = quant_offset;
  int offset = quant_offset - quant_factor / 2;
  /*
   * offset = quant_offset  always undershoots
   * offset = quant_offset - quant_factor  always overshoots
   * offset = quant_offset - quant_factor/2  gives an error that averages 0
   */

  if (value == 0)
    return 0;
  if (value < 0) {
    x = (-value) << 2;
    if (x < dead_zone) {
      x = 0;
    } else {
      x = (x - offset) / quant_factor;
    }
    value = -x;
  } else {
    x = value << 2;
    if (x < dead_zone) {
      x = 0;
    } else {
      x = (x - offset) / quant_factor;
    }
    value = x;
  }
  return value;
}

int
schro_quantise (int value, int quant_factor, int quant_offset)
{
  return __schro_quantise (value, quant_factor, quant_offset);
}

void
schro_quantise_s16 (int16_t * dest, int16_t * src, int quant_factor,
    int quant_offset, int n)
{
  int i;
  for (i = 0; i < n; i++) {
    dest[i] = __schro_quantise (src[i], quant_factor, quant_offset);
    src[i] = __schro_dequantise (dest[i], quant_factor, quant_offset);
  }
}

#ifdef unused
void
schro_quantise_s16_table (int16_t * dest, int16_t * src, int quant_index,
    schro_bool is_intra, int n)
{
  int i;
  int16_t *table;

  table = schro_tables_get_quantise_table (quant_index, is_intra);

  table += 32768;

  for (i = 0; i < n; i++) {
    dest[i] = table[src[i]];
  }
}

void
schro_dequantise_s16_table (int16_t * dest, int16_t * src, int quant_index,
    schro_bool is_intra, int n)
{
  int i;
  int16_t *table;

  table = schro_tables_get_dequantise_table (quant_index, is_intra);

  table += 32768;

  for (i = 0; i < n; i++) {
    dest[i] = table[src[i]];
  }
}
#endif

void
schro_dequantise_s16 (int16_t * dest, int16_t * src, int quant_factor,
    int quant_offset, int n)
{
  int i;
  for (i = 0; i < n; i++) {
    dest[i] = __schro_dequantise (src[i], quant_factor, quant_offset);
  }
}

/* log(2.0) */
#define LOG_2 0.69314718055994528623
/* 1.0/log(2.0) */
#define INV_LOG_2 1.44269504088896338700

double
schro_utils_probability_to_entropy (double x)
{
  if (x <= 0 || x >= 1.0)
    return 0;

  return -(x * log (x) + (1 - x) * log (1 - x)) * INV_LOG_2;
}

double
schro_utils_entropy (double a, double total)
{
  double x;

  if (total == 0)
    return 0;

  x = a / total;
  return schro_utils_probability_to_entropy (x) * total;
}

void
schro_utils_reduce_fraction (int *n, int *d)
{
  static const int primes[] = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37,
    91
  };
  int i;
  int p;

  SCHRO_DEBUG ("reduce %d/%d", *n, *d);
  for (i = 0; i < sizeof (primes) / sizeof (primes[0]); i++) {
    p = primes[i];
    while (*n % p == 0 && *d % p == 0) {
      *n /= p;
      *d /= p;
    }
    if (*d == 1)
      break;
  }
  SCHRO_DEBUG ("to %d/%d", *n, *d);
}

double
schro_utils_get_time (void)
{
#ifndef _WIN32
  struct timeval tv;

  gettimeofday (&tv, NULL);

  return tv.tv_sec + 1e-6 * tv.tv_usec;
#else
  return (double) GetTickCount () / 1000.;
#endif
}
