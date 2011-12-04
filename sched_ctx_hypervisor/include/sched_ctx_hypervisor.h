#include <starpu.h>
#include <../common/config.h>
#include <../common/htable32.h>
#include <pthread.h>

/* ioctl properties*/
#define HYPERVISOR_MAX_IDLE (1<<1)
#define HYPERVISOR_MIN_WORKING (1<<2)
#define HYPERVISOR_PRIORITY (1<<3)
#define HYPERVISOR_MIN_PROCS (1<<4)
#define HYPERVISOR_MAX_PROCS (1<<5)
#define HYPERVISOR_GRANULARITY (1<<6)
#define HYPERVISOR_FIXED_PROCS (1<<7)
#define HYPERVISOR_MIN_TASKS (1<<8)
#define HYPERVISOR_NEW_WORKERS_MAX_IDLE (1<<9)
#define HYPERVISOR_TIME_TO_APPLY (1<<10)

struct sched_ctx_hypervisor_reply{
	int procs[STARPU_NMAXWORKERS];
	int nprocs;
};
pthread_mutex_t act_hypervisor_mutex;

struct starpu_sched_ctx_hypervisor_criteria* sched_ctx_hypervisor_init(int type);

void sched_ctx_hypervisor_shutdown(void);

void sched_ctx_hypervisor_handle_ctx(unsigned sched_ctx);

void sched_ctx_hypervisor_ignore_ctx(unsigned sched_ctx);

void sched_ctx_hypervisor_resize(unsigned sender_sched_ctx, unsigned receier_sched_ctx, int *workers_to_move, unsigned nworkers_to_movex);

void sched_ctx_hypervisor_stop_resize(unsigned sched_ctx);

void sched_ctx_hypervisor_start_resize(unsigned sched_ctx);

void sched_ctx_hypervisor_set_config(unsigned sched_ctx, void *config);

void* sched_ctx_hypervisor_get_config(unsigned sched_ctx);

void sched_ctx_hypervisor_ioctl(unsigned sched_ctx, ...);

void sched_ctx_hypervisor_advise(unsigned sched_ctx, int *workers, int nworkers, int task_tag);

void sched_ctx_hypervisor_request(unsigned sched_ctx, int *workers, int nworkers, int task_tag);

/* hypervisor policies */
#define SIMPLE_POLICY 1

struct hypervisor_policy {
	void (*init)(void);
	void (*deinit)(void);
	void (*add_sched_ctx)(unsigned sched_ctx);
	void(*remove_sched_ctx)(unsigned sched_ctx);
	void* (*ioctl)(unsigned sched_ctx, va_list varg_list, unsigned later);
	unsigned (*manage_idle_time)(unsigned req_sched_ctx, int *sched_ctxs, unsigned nsched_ctxs, int worker, double idle_time);
	void (*update_config)(void* old_config, void* new_config);
};
