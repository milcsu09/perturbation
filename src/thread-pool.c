#include "thread-pool.h"
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>


struct thread_pool_work
{
  void (*function) (void *);
  void  *argument;
};


struct thread_pool
{
  pthread_mutex_t           mutex;
  pthread_cond_t            cond;

  pthread_t                *threads;
  int                       threads_amount;
  atomic_int                threads_active;

  struct thread_pool_work  *queue;
  int                       queue_capacity;
  int                       queue_size;
  int                       queue_head;
  int                       queue_tail;

  int                       stop;
};


void *thread_pool_thread_work (void *);


struct thread_pool *
thread_pool_create (int threads_amount, int queue_capacity)
{
  struct thread_pool *pool;

  pool = malloc (sizeof (struct thread_pool));

  pthread_mutex_init (&pool->mutex, NULL);
  pthread_cond_init (&pool->cond, NULL);

  pool->threads = calloc (threads_amount, sizeof (pthread_t));

  pool->threads_amount = threads_amount;
  atomic_init (&pool->threads_active, 0);

  pool->queue = calloc (queue_capacity, sizeof (struct thread_pool_work));
  pool->queue_capacity = queue_capacity;
  pool->queue_size = 0;
  pool->queue_head = 0;
  pool->queue_tail = 0;

  pool->stop = 0;

  for (int i = 0; i < threads_amount; ++i)
    pthread_create (&pool->threads[i], NULL, thread_pool_thread_work, pool);

  return pool;
}


void
thread_pool_destroy (struct thread_pool *pool)
{
  thread_pool_stop (pool);

  pthread_mutex_destroy (&pool->mutex);
  pthread_cond_destroy (&pool->cond);

  free (pool->threads);
  free (pool->queue);

  free (pool);
}


void
thread_pool_enqueue (struct thread_pool *pool, void (*function) (void *),
                     void *argument)
{
  pthread_mutex_lock (&pool->mutex);

  if (pool->queue_size == pool->queue_capacity || pool->stop)
    {
      pthread_mutex_unlock (&pool->mutex);
      return;
    }

  struct thread_pool_work work;

  work.function = function;
  work.argument = argument;

  pool->queue[pool->queue_tail] = work;
  pool->queue_tail = (pool->queue_tail + 1) % pool->queue_capacity;
  pool->queue_size++;

  pthread_cond_signal (&pool->cond);
  pthread_mutex_unlock (&pool->mutex);
}


void
thread_pool_clear (struct thread_pool *pool)
{
  pthread_mutex_lock (&pool->mutex);

  pool->queue_size = 0;
  pool->queue_head = 0;
  pool->queue_tail = 0;

  pthread_mutex_unlock (&pool->mutex);
}


void
thread_pool_stop (struct thread_pool *pool)
{
  pthread_mutex_lock (&pool->mutex);

  if (pool->stop)
    {
      pthread_mutex_unlock (&pool->mutex);
      return;
    }

  pool->stop = 1;
  pthread_cond_broadcast (&pool->cond);

  pthread_mutex_unlock (&pool->mutex);

  for (int i = 0; i < pool->threads_amount; ++i)
    pthread_join (pool->threads[i], NULL);
}


int
thread_pool_get_threads_active (struct thread_pool *pool)
{
  return atomic_load (&pool->threads_active);
}


void *
thread_pool_thread_work (void *argument)
{
  struct thread_pool *pool = argument;

  while (1)
    {
      pthread_mutex_lock (&pool->mutex);

      while (!pool->stop && pool->queue_size == 0)
        pthread_cond_wait (&pool->cond, &pool->mutex);

      if (pool->stop && pool->queue_size == 0)
        {
          pthread_mutex_unlock (&pool->mutex);
          break;
        }

      struct thread_pool_work work;

      work = pool->queue[pool->queue_head];

      pool->queue_head = (pool->queue_head + 1) % pool->queue_capacity;
      pool->queue_size--;

      pthread_mutex_unlock (&pool->mutex);

      atomic_fetch_add (&pool->threads_active, 1);

      if (work.function)
        work.function (work.argument);

      atomic_fetch_sub (&pool->threads_active, 1);
    }

  return NULL;
}

