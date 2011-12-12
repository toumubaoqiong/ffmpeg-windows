
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <schroedinger/schroutils.h>
#include <schroedinger/schrodomain.h>
#include <schroedinger/schrodebug.h>
#include <stdlib.h>

//#define MEM_DOMAIN_ALWAYS_FREE 1

/* SchroMemoryDomain */

SchroMemoryDomain *
schro_memory_domain_new (void)
{
  SchroMemoryDomain *domain;

  domain = schro_malloc0 (sizeof (SchroMemoryDomain));

  domain->mutex = schro_mutex_new ();

  return domain;
}

SchroMemoryDomain *
schro_memory_domain_new_local (void)
{
  SchroMemoryDomain *domain;

  domain = schro_memory_domain_new ();

  domain->alloc = (void *) malloc;
  domain->free = (void *) free;
  domain->flags = SCHRO_MEMORY_DOMAIN_CPU;

  return domain;
}

void
schro_memory_domain_free (SchroMemoryDomain * domain)
{
  int i;

  SCHRO_ASSERT (domain != NULL);

  for (i = 0; i < SCHRO_MEMORY_DOMAIN_SLOTS; i++) {
    if (domain->slots[i].flags & SCHRO_MEMORY_DOMAIN_SLOT_ALLOCATED) {
      domain->free (domain->slots[i].ptr, domain->slots[i].size);
    }
  }

  schro_mutex_free (domain->mutex);

  schro_free (domain);
}

void *
schro_memory_domain_alloc (SchroMemoryDomain * domain, int size)
{
  int i;
  void *ptr;

  SCHRO_ASSERT (domain != NULL);

  SCHRO_DEBUG ("alloc %d", size);

  schro_mutex_lock (domain->mutex);
  for (i = 0; i < SCHRO_MEMORY_DOMAIN_SLOTS; i++) {
    if (!(domain->slots[i].flags & SCHRO_MEMORY_DOMAIN_SLOT_ALLOCATED)) {
      continue;
    }
    if (domain->slots[i].flags & SCHRO_MEMORY_DOMAIN_SLOT_IN_USE) {
      continue;
    }
    if (domain->slots[i].size == size) {
      domain->slots[i].flags |= SCHRO_MEMORY_DOMAIN_SLOT_IN_USE;
      SCHRO_DEBUG ("got %p", domain->slots[i].ptr);
      ptr = domain->slots[i].ptr;
      goto done;
    }
  }

  for (i = 0; i < SCHRO_MEMORY_DOMAIN_SLOTS; i++) {
    if (domain->slots[i].flags & SCHRO_MEMORY_DOMAIN_SLOT_ALLOCATED) {
      continue;
    }

    domain->slots[i].flags |= SCHRO_MEMORY_DOMAIN_SLOT_ALLOCATED;
    domain->slots[i].flags |= SCHRO_MEMORY_DOMAIN_SLOT_IN_USE;
    domain->slots[i].size = size;
    domain->slots[i].ptr = domain->alloc (size);

    SCHRO_DEBUG ("created %p", domain->slots[i].ptr);
    ptr = domain->slots[i].ptr;
    goto done;
  }

  SCHRO_ASSERT (0);
done:
  schro_mutex_unlock (domain->mutex);
  return ptr;
}

void
schro_memory_domain_memfree (SchroMemoryDomain * domain, void *ptr)
{
  int i;

  SCHRO_ASSERT (domain != NULL);

  SCHRO_DEBUG ("free %p", ptr);

  schro_mutex_lock (domain->mutex);
  for (i = 0; i < SCHRO_MEMORY_DOMAIN_SLOTS; i++) {
    if (!(domain->slots[i].flags & SCHRO_MEMORY_DOMAIN_SLOT_ALLOCATED)) {
      continue;
    }
    if (!(domain->slots[i].flags & SCHRO_MEMORY_DOMAIN_SLOT_IN_USE)) {
      continue;
    }
    if (domain->slots[i].ptr == ptr) {
#ifdef MEM_DOMAIN_ALWAYS_FREE
      domain->free (domain->slots[i].ptr, domain->slots[i].size);
      domain->slots[i].flags = 0;
#else
      domain->slots[i].flags &= (~SCHRO_MEMORY_DOMAIN_SLOT_IN_USE);
#endif
      schro_mutex_unlock (domain->mutex);
      return;
    }
  }
  schro_mutex_unlock (domain->mutex);

  SCHRO_ASSERT (0);
}
