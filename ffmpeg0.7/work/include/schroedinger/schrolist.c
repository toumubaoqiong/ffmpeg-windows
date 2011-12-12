
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schrolist.h>

#include <string.h>
#include <stdlib.h>

SchroList *
schro_list_new (void)
{
  SchroList *list;

  list = schro_malloc0 (sizeof (*list));

  return list;
}

SchroList *
schro_list_new_full (SchroListFreeFunc freefunc, void *priv)
{
  SchroList *list = schro_list_new ();

  list->free = freefunc;
  list->priv = priv;

  return list;
}

void
schro_list_free (SchroList * list)
{
  int i;

  if (list->free) {
    for (i = 0; i < list->n; i++) {
      list->free (list->members[i], list->priv);
    }
  }
  if (list->members) {
    schro_free (list->members);
  }
  schro_free (list);
}

void *
schro_list_get (SchroList * list, int i)
{
  if (i < 0 || i >= list->n)
    return NULL;
  return list->members[i];
}

int
schro_list_get_size (SchroList * list)
{
  return list->n;
}

static void
_schro_list_expand (SchroList * list, int n)
{
  if (n <= list->n_alloc)
    return;

  list->members = schro_realloc (list->members, n * sizeof (void *));
  list->n_alloc = n;
}

static void
_schro_list_free_member (SchroList * list, void *value)
{
  if (list->free) {
    list->free (value, list->priv);
  }
}

void
schro_list_append (SchroList * list, void *value)
{
  _schro_list_expand (list, list->n + 1);
  list->members[list->n] = value;
  list->n++;
}

#ifdef unused
void
schro_list_insert (SchroList * list, int i, void *value)
{
  if (i < 0 || i >= list->n)
    return;

  _schro_list_expand (list, list->n + 1);

  memmove (list->members + i + 1, list->members + i,
      (list->n - i - 1) * sizeof (void *));
  list->members[i] = value;
  list->n++;
}
#endif

void *
schro_list_remove (SchroList * list, int i)
{
  void *value;

  if (i < 0 || i >= list->n)
    return NULL;

  value = list->members[i];

  memmove (list->members + i, list->members + i + 1,
      (list->n - i - 1) * sizeof (void *));
  list->n--;

  return value;
}

void
schro_list_delete (SchroList * list, int i)
{
  _schro_list_free_member (list, schro_list_remove (list, i));
}

#ifdef unused
void *
schro_list_replace (SchroList * list, int i, void *value)
{
  if (i < 0 || i >= list->n)
    return NULL;

  value = list->members[i];
  list->members[i] = value;

  return value;
}
#endif

#ifdef unused
void
schro_list_prepend (SchroList * list, void *value)
{
  schro_list_insert (list, 0, value);
}
#endif
