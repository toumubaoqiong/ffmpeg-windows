
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>
#include <schroedinger/schroasync.h>
#include <schroedinger/schrodebug.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

struct _SchroAsync
{
  int n_idle;

  volatile int n_completed;

  SchroAsyncTask task;

  SchroAsyncScheduleFunc schedule;
  void *schedule_closure;

  SchroAsyncCompleteFunc complete;
};

struct _SchroThread
{
  SchroAsync *async;
  int busy;
  int index;
};

struct _SchroMutex
{
  int ignore;
};

void
schro_async_init (void)
{

}

SchroAsync *
schro_async_new (int n_threads,
    SchroAsyncScheduleFunc schedule,
    SchroAsyncCompleteFunc complete, void *closure)
{
  SchroAsync *async;

  async = schro_malloc0 (sizeof (SchroAsync));

  async->schedule = schedule;
  async->schedule_closure = closure;
  async->complete = complete;

  return async;
}

void
schro_async_free (SchroAsync * async)
{
  schro_free (async);
}

void
schro_async_start (SchroAsync * async)
{
}

void
schro_async_stop (SchroAsync * async)
{

}

void
schro_async_run_stage_locked (SchroAsync * async, SchroAsyncStage * stage)
{
  SCHRO_ASSERT (async->task.task_func == NULL);

  async->task.task_func = stage->task_func;
  async->task.priv = stage;

}

int
schro_async_wait_locked (SchroAsync * async)
{
  async->schedule (async->schedule_closure, SCHRO_EXEC_DOMAIN_CPU);
  if (async->task.task_func) {
    async->task.task_func (async->task.priv);
    async->task.task_func = NULL;
    async->complete (async->task.priv);
  }

  return TRUE;
}

void
schro_async_lock (SchroAsync * async)
{
}

void
schro_async_unlock (SchroAsync * async)
{
}

void
schro_async_signal_scheduler (SchroAsync * async)
{
}

void
schro_async_add_exec_domain (SchroAsync * async, SchroExecDomain exec_domain)
{
}

SchroExecDomain
schro_async_get_exec_domain (void)
{
  return 0;
}

SchroMutex *
schro_mutex_new (void)
{
  SchroMutex *mutex;

  mutex = schro_malloc (sizeof (SchroMutex));

  return mutex;
}

void
schro_mutex_lock (SchroMutex * mutex)
{
}

void
schro_mutex_unlock (SchroMutex * mutex)
{
}

void
schro_mutex_free (SchroMutex * mutex)
{
  schro_free (mutex);
}
