#ifndef THREAD_POOL_H
#define THREAD_POOL_H

struct thread_pool_work;
struct thread_pool;

struct thread_pool *thread_pool_create (int, int);

void thread_pool_destroy (struct thread_pool *);

void thread_pool_enqueue (struct thread_pool *, void (*) (void *), void *);

void thread_pool_clear (struct thread_pool *);

void thread_pool_stop (struct thread_pool *);

int thread_pool_get_threads_active (struct thread_pool *);

#endif // THREAD_POOL_H

