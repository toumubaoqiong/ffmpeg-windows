
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>
#include <schroedinger/schroasync.h>
#include <schroedinger/schrodebug.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>

#include <windows.h>

#define COND_LIST_SIZE 10
#define STACK_SIZE 65536

struct _SchroMutex
{
  CRITICAL_SECTION mutex;
};

struct _SchroAsync
{
  int n_threads;
  int n_threads_running;
  int n_idle;

  enum
  { RUNNING = 0, STOP, DIE } stop;

  volatile int n_completed;

  CRITICAL_SECTION mutex;

  HANDLE app_event;

  SchroThread *threads;

  SchroAsyncTask task;

  SchroAsyncScheduleFunc schedule;
  void *schedule_closure;

  SchroAsyncCompleteFunc complete;
};

struct _SchroThread
{
  HANDLE thread;
  HANDLE event;
  SchroExecDomain exec_domain;
  SchroAsync *async;
  int busy;
  int index;
};

static DWORD domain_key;

static unsigned int __stdcall schro_thread_main (void *ptr);

void
schro_async_init (void)
{
  domain_key = TlsAlloc ();
  if (domain_key == TLS_OUT_OF_INDEXES) {
    /* FIXME */
  }
}

SchroAsync *
schro_async_new (int n_threads,
    SchroAsyncScheduleFunc schedule,
    SchroAsyncCompleteFunc complete, void *closure)
{
  SchroAsync *async;
  int i;

  if (n_threads == 0) {
    const char *s;

    s = getenv ("SCHRO_THREADS");
    if (s && s[0]) {
      char *end;
      int n;
      n = strtoul (s, &end, 0);
      if (end[0] == 0) {
        n_threads = n;
      }
    }
    if (n_threads == 0) {
      s = getenv ("NUMBER_OF_PROCESSORS");
      if (s) {
        n_threads = atoi (s);
      }
    }
    if (n_threads == 0) {
      n_threads = 1;
    }
  }
  async = schro_malloc0 (sizeof (SchroAsync));

  SCHRO_DEBUG ("%d", n_threads);
  async->n_threads = n_threads;
  async->threads = schro_malloc0 (sizeof (SchroThread) * (n_threads + 1));

  async->stop = RUNNING;
  async->schedule = schedule;
  async->schedule_closure = closure;
  async->complete = complete;

  InitializeCriticalSection (&async->mutex);

  async->app_event = CreateEvent (0, FALSE, FALSE, NULL);

  EnterCriticalSection (&async->mutex);

  for (i = 0; i < n_threads; i++) {
    SchroThread *thread = async->threads + i;
    unsigned int ignore;

    thread->event = CreateEvent (0, FALSE, FALSE, NULL);
    thread->async = async;
    thread->index = i;
    thread->exec_domain = SCHRO_EXEC_DOMAIN_CPU;
    async->threads[i].thread = (HANDLE) _beginthreadex (NULL, STACK_SIZE,
        schro_thread_main, async->threads + i, 0, &ignore);
    EnterCriticalSection (&async->mutex);
  }
  LeaveCriticalSection (&async->mutex);

  return async;
}

void
schro_async_free (SchroAsync * async)
{
  int i;
  HANDLE *handles;

  EnterCriticalSection (&async->mutex);
  async->stop = DIE;
  LeaveCriticalSection (&async->mutex);

  for (i = 0; i < async->n_threads; i++) {
    SetEvent (async->threads[i].event);
  }

  handles = schro_malloc (sizeof (HANDLE) * async->n_threads);
  for (i = 0; i < async->n_threads; i++) {
    handles[i] = async->threads[i].thread;
  }
  WaitForMultipleObjects (async->n_threads, handles, TRUE, INFINITE);
  schro_free (handles);

  for (i = 0; i < async->n_threads; i++) {
    CloseHandle (async->threads[i].event);
    CloseHandle (async->threads[i].thread);
  }
  CloseHandle (async->app_event);
  DeleteCriticalSection (&async->mutex);
  schro_free (async->threads);
  schro_free (async);
}

void
schro_async_start (SchroAsync * async)
{
  int i;
  async->stop = RUNNING;
  for (i = 0; i < async->n_threads; i++) {
    SetEvent (async->threads[i].event);
  }
}

void
schro_async_stop (SchroAsync * async)
{
  int i;
  HANDLE *handles;

  for (i = 0; i < async->n_threads; i++) {
    SetEvent (async->threads[i].event);
  }

  handles = schro_malloc (sizeof (HANDLE) * async->n_threads);
  for (i = 0; i < async->n_threads; i++) {
    handles[i] = async->threads[i].thread;
  }
  WaitForMultipleObjects (async->n_threads, handles, TRUE, INFINITE);
  schro_free (handles);
}

void
schro_async_run_stage_locked (SchroAsync * async, SchroAsyncStage * stage)
{
  SCHRO_ASSERT (async->task.task_func == NULL);

  async->task.task_func = stage->task_func;
  async->task.priv = stage;

  schro_async_signal_scheduler (async);
}

static void
schro_async_dump (SchroAsync * async)
{
  int i;
  SCHRO_WARNING ("stop = %d", async->stop);
  for (i = 0; i < async->n_threads; i++) {
    SchroThread *thread = async->threads + i;

    SCHRO_WARNING ("thread %d: busy=%d", i, thread->busy);
  }
}

int
schro_async_wait_locked (SchroAsync * async)
{
  DWORD ret;
  LeaveCriticalSection (&async->mutex);
  ret = WaitForSingleObject (async->app_event, 1000);
  EnterCriticalSection (&async->mutex);
  if (ret == WAIT_TIMEOUT) {
    int i;
    for (i = 0; i < async->n_threads; i++) {
      if (async->threads[i].busy != 0)
        break;
    }
    if (i == async->n_threads) {
      SCHRO_WARNING ("timeout.  deadlock?");
      schro_async_dump (async);
      return FALSE;
    }
  }
  return TRUE;
}

static unsigned int __stdcall
schro_thread_main (void *ptr)
{
  void (*func) (void *);
  void *priv;
  SchroThread *thread = ptr;
  SchroAsync *async = thread->async;
  int ret;

  /* thread starts with async->mutex locked */

  TlsSetValue (domain_key, (void *) (unsigned long) thread->exec_domain);

  async->n_threads_running++;
  thread->busy = FALSE;
  while (1) {
    /* check for deaths each time */
    if (async->stop != RUNNING) {
      async->n_idle++;
      thread->busy = FALSE;
      SetEvent (async->app_event);
      if (async->stop == DIE) {
        async->n_threads_running--;
        LeaveCriticalSection (&async->mutex);
        SCHRO_DEBUG ("thread %d: dying", thread->index);
        return 0;
      }
      SCHRO_DEBUG ("thread %d: stopping (until restarted)", thread->index);
      WaitForSingleObject (thread->event, INFINITE);
      SCHRO_DEBUG ("thread %d: resuming", thread->index);
      async->n_idle--;
      continue;
    }
    if (thread->busy == 0) {
      async->n_idle++;
      LeaveCriticalSection (&async->mutex);
      SCHRO_DEBUG ("thread %d: idle", thread->index);
      WaitForSingleObject (thread->event, INFINITE);
      SCHRO_DEBUG ("thread %d: got signal", thread->index);
      EnterCriticalSection (&async->mutex);
      async->n_idle--;
      thread->busy = TRUE;
      /* check for stop requests before doing work */
      continue;
    }
    if (1) {                    /* avoiding indent change */
      ret = async->schedule (async->schedule_closure, thread->exec_domain);
      /* FIXME ignoring ret */
      if (!async->task.task_func) {
        thread->busy = FALSE;
        continue;
      }
      //thread->busy = TRUE;
      func = async->task.task_func;
      priv = async->task.priv;
      async->task.task_func = NULL;

      //if (async->n_idle > 0) {
      //  SetEvent (thread->event);
      //}
      LeaveCriticalSection (&async->mutex);

      SCHRO_DEBUG ("thread %d: running", thread->index);
      func (priv);
      SCHRO_DEBUG ("thread %d: done", thread->index);

      EnterCriticalSection (&async->mutex);

      async->complete (priv);

      SetEvent (async->app_event);
#if defined HAVE_CUDA || defined HAVE_OPENGL
      /* FIXME */
      /* This is required because we don't have a better mechanism
       * for indicating to threads in other exec domains that it is
       * their turn to run.  It's mostly harmless, although causes
       * a lot of unnecessary wakeups in some cases. */
      {
        int i;
        for (i = 0; i < async->n_threads; i++) {
          SetEvent (async->thread_event);
        }
      }
#endif

      //thread->busy = FALSE;
    }
  }
}

void
schro_async_lock (SchroAsync * async)
{
  EnterCriticalSection (&async->mutex);
}

void
schro_async_unlock (SchroAsync * async)
{
  LeaveCriticalSection (&async->mutex);
}

void
schro_async_signal_scheduler (SchroAsync * async)
{
  int i;

  SCHRO_DEBUG ("signal scheduler");
  for (i = 0; i < async->n_threads; i++) {
    SetEvent (async->threads[i].event);
  }
}

void
schro_async_add_exec_domain (SchroAsync * async, SchroExecDomain exec_domain)
{
  SchroThread *thread;
  int i;
  unsigned int ignore;

  EnterCriticalSection (&async->mutex);

  /* We allocated a spare thread structure just for this case. */
  async->n_threads++;
  i = async->n_threads - 1;

  thread = async->threads + i;
  memset (thread, 0, sizeof (SchroThread));

  thread->async = async;
  thread->index = i;
  thread->exec_domain = exec_domain;

  thread->thread = (HANDLE) _beginthreadex (NULL, STACK_SIZE,
      schro_thread_main, thread, 0, &ignore);
  EnterCriticalSection (&async->mutex);
  LeaveCriticalSection (&async->mutex);
}

SchroExecDomain
schro_async_get_exec_domain (void)
{
  void *domain;
  domain = TlsGetValue (domain_key);
  return (int) (unsigned long) domain;
}

SchroMutex *
schro_mutex_new (void)
{
  SchroMutex *mutex;

  mutex = malloc (sizeof (SchroMutex));
  InitializeCriticalSection (&mutex->mutex);

  return mutex;
}

void
schro_mutex_lock (SchroMutex * mutex)
{
  EnterCriticalSection (&mutex->mutex);
}

void
schro_mutex_unlock (SchroMutex * mutex)
{
  LeaveCriticalSection (&mutex->mutex);
}

void
schro_mutex_free (SchroMutex * mutex)
{
  DeleteCriticalSection (&mutex->mutex);
  schro_free (mutex);
}
