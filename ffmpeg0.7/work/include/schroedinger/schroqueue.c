
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <schroedinger/schroutils.h>
#include <schroedinger/schroqueue.h>
#include <schroedinger/schrodebug.h>
#include <stdlib.h>
#include <string.h>

SchroQueue *
schro_queue_new (int size, SchroQueueFreeFunc free_func)
{
  SchroQueue *queue;

  queue = schro_malloc0 (sizeof (*queue));

  queue->size = size;
  queue->free = free_func;

  queue->elements = schro_malloc0 (size * sizeof (SchroQueueElement));

  return queue;
}

void
schro_queue_free (SchroQueue * queue)
{
  int i;

  if (queue->free) {
    for (i = 0; i < queue->n; i++) {
      queue->free (queue->elements[i].data, queue->elements[i].picture_number);
    }
  }

  schro_free (queue->elements);
  schro_free (queue);
}

void
schro_queue_add (SchroQueue * queue, void *data,
    SchroPictureNumber picture_number)
{
  SCHRO_ASSERT (queue->n < queue->size);

  queue->elements[queue->n].data = data;
  queue->elements[queue->n].picture_number = picture_number;
  queue->n++;
}

void *
schro_queue_find (SchroQueue * queue, SchroPictureNumber picture_number)
{
  int i;

  for (i = 0; i < queue->n; i++) {
    if (queue->elements[i].picture_number == picture_number) {
      return queue->elements[i].data;
    }
  }

  return NULL;
}

void
schro_queue_delete (SchroQueue * queue, SchroPictureNumber picture_number)
{
  int i;

  for (i = 0; i < queue->n; i++) {
    if (queue->elements[i].picture_number == picture_number) {
      if (queue->free) {
        queue->free (queue->elements[i].data,
            queue->elements[i].picture_number);
      }
      memmove (queue->elements + i, queue->elements + i + 1,
          sizeof (SchroQueueElement) * (queue->n - i - 1));
      queue->n--;
      return;
    }
  }
}

#ifdef unused
void *
schro_queue_remove (SchroQueue * queue, SchroPictureNumber picture_number)
{
  int i;
  void *ret;

  for (i = 0; i < queue->n; i++) {
    if (queue->elements[i].picture_number == picture_number) {
      ret = queue->elements[i].data;
      memmove (queue->elements + i, queue->elements + i + 1,
          sizeof (SchroQueueElement) * (queue->n - i - 1));
      queue->n--;
      return ret;
    }
  }

  return NULL;
}
#endif

#ifdef unused
void
schro_queue_clear (SchroQueue * queue)
{
  int i;

  for (i = 0; i < queue->n; i++) {
    if (queue->free) {
      queue->free (queue->elements[i].data, queue->elements[i].picture_number);
    }
  }
  queue->n = 0;
}
#endif

void
schro_queue_pop (SchroQueue * queue)
{
  if (queue->n == 0)
    return;

  if (queue->free) {
    queue->free (queue->elements[0].data, queue->elements[0].picture_number);
  }
  memmove (queue->elements, queue->elements + 1,
      sizeof (SchroQueueElement) * (queue->n - 1));
  queue->n--;
}

void *
schro_queue_peek (SchroQueue * queue)
{
  if (queue->n == 0)
    return NULL;

  return queue->elements[0].data;
}

void *
schro_queue_pull (SchroQueue * queue)
{
  void *ret;

  if (queue->n == 0)
    return NULL;

  ret = queue->elements[0].data;
  memmove (queue->elements, queue->elements + 1,
      sizeof (SchroQueueElement) * (queue->n - 1));
  queue->n--;

  return ret;
}

int
schro_queue_is_full (SchroQueue * queue)
{
  return (queue->n == queue->size);
}

int
schro_queue_slots_available (SchroQueue * queue)
{
  return queue->size - queue->n;
}

int
schro_queue_is_empty (SchroQueue * queue)
{
  return (queue->n == 0);
}
