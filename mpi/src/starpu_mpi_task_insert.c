/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011, 2012, 2013, 2014  Centre National de la Recherche Scientifique
 * Copyright (C) 2011-2014  Université de Bordeaux 1
 * Copyright (C) 2014 INRIA
 *
 * StarPU is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * StarPU is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <stdarg.h>
#include <mpi.h>

#include <starpu.h>
#include <starpu_data.h>
#include <common/utils.h>
#include <util/starpu_task_insert_utils.h>
#include <datawizard/coherency.h>
#include <core/task.h>

#include <starpu_mpi_private.h>
#include <starpu_mpi_task_insert.h>
#include <starpu_mpi_cache.h>

static
int _starpu_mpi_find_executee_node(starpu_data_handle_t data, enum starpu_data_access_mode mode, int me, int *do_execute, int *inconsistent_execute, int *dest, size_t *size_on_nodes)
{
	if (data && mode & STARPU_R)
	{
		struct starpu_data_interface_ops *ops;
		int rank = starpu_data_get_rank(data);

		ops = data->ops;
		size_on_nodes[rank] += ops->get_size(data);
	}

	if (mode & STARPU_W)
	{
		if (!data)
		{
			/* We don't have anything allocated for this.
			 * The application knows we won't do anything
			 * about this task */
			/* Yes, the app could actually not call
			 * task_insert at all itself, this is just a
			 * safeguard. */
			_STARPU_MPI_DEBUG(3, "oh oh\n");
			_STARPU_MPI_LOG_OUT();
			return -EINVAL;
		}
		int mpi_rank = starpu_data_get_rank(data);
		if (mpi_rank == me)
		{
			if (*do_execute == 0)
			{
				*inconsistent_execute = 1;
			}
			else
			{
				*do_execute = 1;
			}
		}
		else if (mpi_rank != -1)
		{
			if (*do_execute == 1)
			{
				*inconsistent_execute = 1;
			}
			else
			{
				*do_execute = 0;
				*dest = mpi_rank;
				/* That's the rank which needs the data to be sent to */
			}
		}
		else
		{
			_STARPU_ERROR("rank %d invalid\n", mpi_rank);
		}
	}
	return 0;
}

static
void _starpu_mpi_exchange_data_before_execution(starpu_data_handle_t data, enum starpu_data_access_mode mode, int me, int dest, int do_execute, MPI_Comm comm)
{
	if (data && mode & STARPU_R)
	{
		int mpi_rank = starpu_data_get_rank(data);
		int mpi_tag = starpu_data_get_tag(data);
		if(mpi_rank == -1)
		{
			fprintf(stderr,"StarPU needs to be told the MPI rank of this data, using starpu_data_set_rank\n");
			STARPU_ABORT();
		}
		if(mpi_tag == -1)
		{
			fprintf(stderr,"StarPU needs to be told the MPI tag of this data, using starpu_data_set_tag\n");
			STARPU_ABORT();
		}
		/* The task needs to read this data */
		if (do_execute && mpi_rank != me && mpi_rank != -1)
		{
			/* I will have to execute but I don't have the data, receive */
			void *already_received = _starpu_mpi_already_received(me, data, mpi_rank);
			if (already_received == NULL)
			{
				_STARPU_MPI_DEBUG(1, "Receive data %p from %d\n", data, mpi_rank);
				starpu_mpi_irecv_detached(data, mpi_rank, mpi_tag, comm, NULL, NULL);
			}
		}
		if (!do_execute && mpi_rank == me)
		{
			/* Somebody else will execute it, and I have the data, send it. */
			void *already_sent = _starpu_mpi_already_sent(data, dest);
			if (already_sent == NULL)
			{
				_STARPU_MPI_DEBUG(1, "Send data %p to %d\n", data, dest);
				starpu_mpi_isend_detached(data, dest, mpi_tag, comm, NULL, NULL);
			}
		}
	}
}

static
void _starpu_mpi_exchange_data_after_execution(starpu_data_handle_t data, enum starpu_data_access_mode mode, int me, int xrank, int dest, int do_execute, MPI_Comm comm)
{
	if (mode & STARPU_W)
	{
		int mpi_rank = starpu_data_get_rank(data);
		int mpi_tag = starpu_data_get_tag(data);
		if(mpi_rank == -1)
		{
			fprintf(stderr,"StarPU needs to be told the MPI rank of this data, using starpu_data_set_rank\n");
			STARPU_ABORT();
		}
		if(mpi_tag == -1)
		{
			fprintf(stderr,"StarPU needs to be told the MPI tag of this data, using starpu_data_set_tag\n");
			STARPU_ABORT();
		}
		if (mpi_rank == me)
		{
			if (xrank != -1 && me != xrank)
			{
				_STARPU_MPI_DEBUG(1, "Receive data %p back from the task %d which executed the codelet ...\n", data, dest);
				starpu_mpi_irecv_detached(data, dest, mpi_tag, comm, NULL, NULL);
			}
		}
		else if (do_execute)
		{
			_STARPU_MPI_DEBUG(1, "Send data %p back to its owner %d...\n", data, mpi_rank);
			starpu_mpi_isend_detached(data, mpi_rank, mpi_tag, comm, NULL, NULL);
		}
	}
}

static
void _starpu_mpi_clear_data_after_execution(starpu_data_handle_t data, enum starpu_data_access_mode mode, int me, int do_execute, MPI_Comm comm)
{
	if (_cache_enabled)
	{
		if (mode & STARPU_W || mode & STARPU_REDUX)
		{
			if (do_execute)
			{
				/* Note that all copies I've sent to neighbours are now invalid */
				_starpu_mpi_cache_flush_sent(comm, data);
			}
			else
			{
				_starpu_mpi_cache_flush_recv(data, me);
			}
		}
	}
	else
	{
		/* We allocated a temporary buffer for the received data, now drop it */
		if ((mode & STARPU_R) && do_execute)
		{
			int mpi_rank = starpu_data_get_rank(data);
			if (mpi_rank != me && mpi_rank != -1)
			{
				starpu_data_invalidate_submit(data);
			}
		}
	}
}

static
int _starpu_mpi_task_decode_v(struct starpu_codelet *codelet, int me, int nb_nodes, int *xrank, int *dest, int *do_execute, va_list varg_list)
{
	va_list varg_list_copy;
	int inconsistent_execute = 0;
	int arg_type, arg_type_nocommute;
	size_t *size_on_nodes;
	int current_data = 0;

	size_on_nodes = (size_t *)calloc(1, nb_nodes * sizeof(size_t));
	*do_execute = -1;
	*xrank = -1;
	va_copy(varg_list_copy, varg_list);
	while ((arg_type = va_arg(varg_list_copy, int)) != 0)
	{
		arg_type_nocommute = arg_type & ~STARPU_COMMUTE;
		if (arg_type==STARPU_EXECUTE_ON_NODE)
		{
			*xrank = va_arg(varg_list_copy, int);
			_STARPU_MPI_DEBUG(1, "Executing on node %d\n", *xrank);
			*do_execute = 1;
		}
		else if (arg_type==STARPU_EXECUTE_ON_DATA)
		{
			starpu_data_handle_t data = va_arg(varg_list_copy, starpu_data_handle_t);
			*xrank = starpu_data_get_rank(data);
			STARPU_ASSERT_MSG(*xrank != -1, "Rank of the data must be set using starpu_mpi_data_register() or starpu_data_set_rank()");
			_STARPU_MPI_DEBUG(1, "Executing on data node %d\n", *xrank);
			STARPU_ASSERT_MSG(*xrank <= nb_nodes, "Node %d to execute codelet is not a valid node (%d)", *xrank, nb_nodes);
			*do_execute = 1;
		}
		else if (arg_type==STARPU_EXECUTE_ON_WORKER)
		{
			// the flag is decoded and set later when
			// calling function _starpu_task_insert_create()
			va_arg(varg_list_copy, int);
		}
		else if (arg_type_nocommute==STARPU_R || arg_type_nocommute==STARPU_W || arg_type_nocommute==STARPU_RW || arg_type==STARPU_SCRATCH || arg_type==STARPU_REDUX)
		{
			starpu_data_handle_t data = va_arg(varg_list_copy, starpu_data_handle_t);
			enum starpu_data_access_mode mode = (enum starpu_data_access_mode) arg_type;
			int ret = _starpu_mpi_find_executee_node(data, mode, me, do_execute, &inconsistent_execute, dest, size_on_nodes);
			if (ret == -EINVAL)
			{
				free(size_on_nodes);
				return ret;
			}
			current_data ++;
		}
		else if (arg_type == STARPU_DATA_ARRAY)
		{
			starpu_data_handle_t *datas = va_arg(varg_list_copy, starpu_data_handle_t *);
			int nb_handles = va_arg(varg_list_copy, int);
			int i;
			for(i=0 ; i<nb_handles ; i++)
			{
				enum starpu_data_access_mode mode = STARPU_CODELET_GET_MODE(codelet, current_data);
				int ret = _starpu_mpi_find_executee_node(datas[i], mode, me, do_execute, &inconsistent_execute, dest, size_on_nodes);
				if (ret == -EINVAL)
				{
					free(size_on_nodes);
					return ret;
				}
				current_data ++;
			}
		}
		else if (arg_type==STARPU_VALUE)
		{
			va_arg(varg_list_copy, void *);
			va_arg(varg_list_copy, size_t);
		}
		else if (arg_type==STARPU_CALLBACK)
		{
			va_arg(varg_list_copy, void (*)(void *));
		}
		else if (arg_type==STARPU_CALLBACK_WITH_ARG)
		{
			va_arg(varg_list_copy, void (*)(void *));
			va_arg(varg_list_copy, void *);
		}
		else if (arg_type==STARPU_CALLBACK_ARG)
		{
			va_arg(varg_list_copy, void *);
		}
		else if (arg_type==STARPU_PROLOGUE_CALLBACK)
                {
                        (void)va_arg(varg_list, _starpu_callback_func_t);
		}
                else if (arg_type==STARPU_PROLOGUE_CALLBACK_ARG)
                {
                        (void)va_arg(varg_list, void *);
                }
                else if (arg_type==STARPU_PROLOGUE_CALLBACK_POP)
                {
			(void)va_arg(varg_list, _starpu_callback_func_t);
                }
                else if (arg_type==STARPU_PROLOGUE_CALLBACK_POP_ARG)
                {
                        (void)va_arg(varg_list, void *);
		}
		else if (arg_type==STARPU_PRIORITY)
		{
			va_arg(varg_list_copy, int);
		}
		else if (arg_type==STARPU_HYPERVISOR_TAG)
		{
			(void)va_arg(varg_list_copy, int);
		}
		else if (arg_type==STARPU_FLOPS)
		{
			(void)va_arg(varg_list_copy, double);
		}
		else if (arg_type==STARPU_TAG)
		{
			STARPU_ASSERT_MSG(0, "STARPU_TAG is not supported in MPI mode\n");
		}
		else
		{
			STARPU_ABORT_MSG("Unrecognized argument %d\n", arg_type);
		}

	}
	va_end(varg_list_copy);

	if (*do_execute == -1)
	{
		int i;
		size_t max_size = 0;
		for(i=0 ; i<nb_nodes ; i++)
		{
			if (size_on_nodes[i] > max_size)
			{
				max_size = size_on_nodes[i];
				*xrank = i;
			}
		}
		if (*xrank != -1)
		{
			_STARPU_MPI_DEBUG(1, "Node %d is having the most R data\n", *xrank);
			*do_execute = 1;
		}
	}
	free(size_on_nodes);

	STARPU_ASSERT_MSG(*do_execute != -1, "StarPU needs to see a W or a REDUX data which will tell it where to execute the task");

	if (inconsistent_execute == 1)
	{
		if (*xrank == -1)
		{
			_STARPU_MPI_DEBUG(1, "Different tasks are owning W data. Needs to specify which one is to execute the codelet, using STARPU_EXECUTE_ON_NODE or STARPU_EXECUTE_ON_DATA\n");
			return -EINVAL;
		}
		else
		{
			*do_execute = (me == *xrank);
			*dest = *xrank;
		}
	}
	else if (*xrank != -1)
	{
		*do_execute = (me == *xrank);
		*dest = *xrank;
	}
	return 0;
}

static
int _starpu_mpi_task_build_v(MPI_Comm comm, struct starpu_codelet *codelet, struct starpu_task **task, int *xrank_p, int *dest_p, va_list varg_list)
{
	int arg_type, arg_type_nocommute;
	va_list varg_list_copy;
	int me, do_execute, xrank, nb_nodes;
	size_t arg_buffer_size = 0;
	void *arg_buffer = NULL;
	int ret, dest=0;
	int current_data;

	_STARPU_MPI_LOG_IN();

	MPI_Comm_rank(comm, &me);
	MPI_Comm_size(comm, &nb_nodes);

	/* Find out whether we are to execute the data because we own the data to be written to. */
	ret = _starpu_mpi_task_decode_v(codelet, me, nb_nodes, &xrank, &dest, &do_execute, varg_list);
	if (ret < 0) return ret;

	/* Send and receive data as requested */
	va_copy(varg_list_copy, varg_list);
	current_data = 0;
	while ((arg_type = va_arg(varg_list_copy, int)) != 0)
	{
		arg_type_nocommute = arg_type & ~STARPU_COMMUTE;
		if (arg_type_nocommute==STARPU_R || arg_type_nocommute==STARPU_W || arg_type_nocommute==STARPU_RW || arg_type==STARPU_SCRATCH || arg_type==STARPU_REDUX)
		{
			starpu_data_handle_t data = va_arg(varg_list_copy, starpu_data_handle_t);
			enum starpu_data_access_mode mode = (enum starpu_data_access_mode) arg_type;

			_starpu_mpi_exchange_data_before_execution(data, mode, me, dest, do_execute, comm);
			current_data ++;

		}
		else if (arg_type == STARPU_DATA_ARRAY)
		{
			starpu_data_handle_t *datas = va_arg(varg_list_copy, starpu_data_handle_t *);
			int nb_handles = va_arg(varg_list_copy, int);
			int i;

			for(i=0 ; i<nb_handles ; i++)
			{
				_starpu_mpi_exchange_data_before_execution(datas[i], STARPU_CODELET_GET_MODE(codelet, current_data), me, dest, do_execute, comm);
				current_data++;
			}
		}
		else if (arg_type==STARPU_VALUE)
		{
			va_arg(varg_list_copy, void *);
			va_arg(varg_list_copy, size_t);
		}
		else if (arg_type==STARPU_CALLBACK)
		{
			va_arg(varg_list_copy, void (*)(void *));
		}
		else if (arg_type==STARPU_CALLBACK_WITH_ARG)
		{
			va_arg(varg_list_copy, void (*)(void *));
			va_arg(varg_list_copy, void *);
		}
		else if (arg_type==STARPU_CALLBACK_ARG)
		{
			va_arg(varg_list_copy, void *);
		}
		else if (arg_type==STARPU_PROLOGUE_CALLBACK)
                {
                        (void)va_arg(varg_list, _starpu_callback_func_t);
		}
                else if (arg_type==STARPU_PROLOGUE_CALLBACK_ARG)
                {
                        (void)va_arg(varg_list, void *);
                }
                else if (arg_type==STARPU_PROLOGUE_CALLBACK_POP)
                {
			(void)va_arg(varg_list, _starpu_callback_func_t);
                }
                else if (arg_type==STARPU_PROLOGUE_CALLBACK_POP_ARG)
                {
                        (void)va_arg(varg_list, void *);
		}
		else if (arg_type==STARPU_PRIORITY)
		{
			va_arg(varg_list_copy, int);
		}
		else if (arg_type==STARPU_EXECUTE_ON_NODE)
		{
			va_arg(varg_list_copy, int);
		}
		else if (arg_type==STARPU_EXECUTE_ON_DATA)
		{
			va_arg(varg_list_copy, starpu_data_handle_t);
		}
		else if (arg_type==STARPU_EXECUTE_ON_WORKER)
		{
			// the flag is decoded and set later when
			// calling function _starpu_task_insert_create()
			va_arg(varg_list_copy, int);
		}
		else if (arg_type==STARPU_HYPERVISOR_TAG)
		{
			(void)va_arg(varg_list_copy, int);
		}
		else if (arg_type==STARPU_FLOPS)
		{
			(void)va_arg(varg_list_copy, double);
		}
		else if (arg_type==STARPU_TAG)
		{
			STARPU_ASSERT_MSG(0, "STARPU_TAG is not supported in MPI mode\n");
		}
	}
	va_end(varg_list_copy);

	if (xrank_p) *xrank_p = xrank;
	if (dest_p) *dest_p = dest;

	if (do_execute == 0) return 1;
	else
	{
		/* Get the number of buffers and the size of the arguments */
		va_copy(varg_list_copy, varg_list);
		arg_buffer_size = _starpu_task_insert_get_arg_size(varg_list_copy);
		va_end(varg_list_copy);

		/* Pack arguments if needed */
		if (arg_buffer_size)
		{
			va_copy(varg_list_copy, varg_list);
			_starpu_codelet_pack_args(&arg_buffer, arg_buffer_size, varg_list_copy);
			va_end(varg_list_copy);
		}

		_STARPU_MPI_DEBUG(1, "Execution of the codelet %p (%s)\n", codelet, codelet->name);

		*task = starpu_task_create();
		(*task)->cl_arg_free = 1;

		if (codelet->nbuffers > STARPU_NMAXBUFS)
		{
			(*task)->dyn_handles = malloc(codelet->nbuffers * sizeof(starpu_data_handle_t));
		}

		va_copy(varg_list_copy, varg_list);
		_starpu_task_insert_create(arg_buffer, arg_buffer_size, codelet, task, varg_list_copy);
		va_end(varg_list_copy);
		return 0;
	}
}

static
int _starpu_mpi_task_postbuild_v(MPI_Comm comm, struct starpu_codelet *codelet, va_list varg_list, int xrank, int dest, int do_execute)
{
	int arg_type, arg_type_nocommute;
	va_list varg_list_copy;
	int current_data;
	int me;

	MPI_Comm_rank(comm, &me);

	va_copy(varg_list_copy, varg_list);
	current_data = 0;
	while ((arg_type = va_arg(varg_list_copy, int)) != 0)
	{
		arg_type_nocommute = arg_type & ~STARPU_COMMUTE;
		if (arg_type_nocommute==STARPU_R || arg_type_nocommute==STARPU_W || arg_type_nocommute==STARPU_RW || arg_type==STARPU_SCRATCH || arg_type==STARPU_REDUX)
		{
			starpu_data_handle_t data = va_arg(varg_list_copy, starpu_data_handle_t);
			enum starpu_data_access_mode mode = (enum starpu_data_access_mode) arg_type;

			_starpu_mpi_exchange_data_after_execution(data, mode, me, xrank, dest, do_execute, comm);
			_starpu_mpi_clear_data_after_execution(data, mode, me, do_execute, comm);
			current_data++;
		}
		else if (arg_type == STARPU_DATA_ARRAY)
		{
			starpu_data_handle_t *datas = va_arg(varg_list_copy, starpu_data_handle_t *);
			int nb_handles = va_arg(varg_list_copy, int);
			int i;

			for(i=0 ; i<nb_handles ; i++)
			{
				_starpu_mpi_exchange_data_after_execution(datas[i], STARPU_CODELET_GET_MODE(codelet, current_data), me, xrank, dest, do_execute, comm);
				_starpu_mpi_clear_data_after_execution(datas[i], STARPU_CODELET_GET_MODE(codelet, current_data), me, do_execute, comm);
				current_data++;
			}
		}
		else if (arg_type==STARPU_VALUE)
		{
			va_arg(varg_list_copy, void *);
			va_arg(varg_list_copy, size_t);
		}
		else if (arg_type==STARPU_CALLBACK)
		{
			va_arg(varg_list_copy, void (*)(void *));
		}
		else if (arg_type==STARPU_CALLBACK_WITH_ARG)
		{
			va_arg(varg_list_copy, void (*)(void *));
			va_arg(varg_list_copy, void *);
		}
		else if (arg_type==STARPU_CALLBACK_ARG)
		{
			va_arg(varg_list_copy, void *);
		}
		else if (arg_type==STARPU_PROLOGUE_CALLBACK)
                {
                        (void)va_arg(varg_list, _starpu_callback_func_t);
		}
                else if (arg_type==STARPU_PROLOGUE_CALLBACK_ARG)
                {
                        (void)va_arg(varg_list, void *);
                }
                else if (arg_type==STARPU_PROLOGUE_CALLBACK_POP)
                {
			(void)va_arg(varg_list, _starpu_callback_func_t);
                }
                else if (arg_type==STARPU_PROLOGUE_CALLBACK_POP_ARG)
                {
                        (void)va_arg(varg_list, void *);
		}
		else if (arg_type==STARPU_PRIORITY)
		{
			va_arg(varg_list_copy, int);
		}
		else if (arg_type==STARPU_EXECUTE_ON_NODE)
		{
			va_arg(varg_list_copy, int);
		}
		else if (arg_type==STARPU_EXECUTE_ON_DATA)
		{
			va_arg(varg_list_copy, starpu_data_handle_t);
		}
		else if (arg_type==STARPU_EXECUTE_ON_WORKER)
		{
			va_arg(varg_list_copy, int);
		}
		else if (arg_type==STARPU_HYPERVISOR_TAG)
		{
			(void)va_arg(varg_list_copy, int);
		}
		else if (arg_type==STARPU_FLOPS)
		{
			(void)va_arg(varg_list_copy, double);
		}
		else if (arg_type==STARPU_TAG)
		{
			STARPU_ASSERT_MSG(0, "STARPU_TAG is not supported in MPI mode\n");
		}
	}

	va_end(varg_list_copy);
	_STARPU_MPI_LOG_OUT();
	return 0;
}

static
int _starpu_mpi_task_insert_v(MPI_Comm comm, struct starpu_codelet *codelet, va_list varg_list)
{
	struct starpu_task *task;
	int ret;
	int xrank;
	int dest;
	int do_execute = 0;

	ret = _starpu_mpi_task_build_v(comm, codelet, &task, &xrank, &dest, varg_list);
	if (ret < 0) return ret;

	if (ret == 0)
	{
		do_execute = 1;
		ret = starpu_task_submit(task);

		if (STARPU_UNLIKELY(ret == -ENODEV))
		{
			fprintf(stderr, "submission of task %p wih codelet %p failed (symbol `%s') (err: ENODEV)\n",
				task, task->cl,
				(codelet == NULL) ? "none" :
				task->cl->name ? task->cl->name :
				(task->cl->model && task->cl->model->symbol)?task->cl->model->symbol:"none");

			task->destroy = 0;
			starpu_task_destroy(task);
		}
	}
	return _starpu_mpi_task_postbuild_v(comm, codelet, varg_list, xrank, dest, do_execute);
}

int starpu_mpi_task_insert(MPI_Comm comm, struct starpu_codelet *codelet, ...)
{
	va_list varg_list;
	int ret;

	va_start(varg_list, codelet);
	ret = _starpu_mpi_task_insert_v(comm, codelet, varg_list);
	va_end(varg_list);
	return ret;
}

int starpu_mpi_insert_task(MPI_Comm comm, struct starpu_codelet *codelet, ...)
{
	va_list varg_list;
	int ret;

	va_start(varg_list, codelet);
	ret = _starpu_mpi_task_insert_v(comm, codelet, varg_list);
	va_end(varg_list);
	return ret;
}

struct starpu_task *starpu_mpi_task_build(MPI_Comm comm, struct starpu_codelet *codelet, ...)
{
	va_list varg_list;
	struct starpu_task *task;
	int ret;

	va_start(varg_list, codelet);
	ret = _starpu_mpi_task_build_v(comm, codelet, &task, NULL, NULL, varg_list);
	va_end(varg_list);
	STARPU_ASSERT(ret >= 0);
	if (ret > 0) return NULL; else return task;
}

int starpu_mpi_task_post_build(MPI_Comm comm, struct starpu_codelet *codelet, ...)
{
	int xrank, dest, do_execute;
	int ret, me, nb_nodes;
	va_list varg_list;

	MPI_Comm_rank(comm, &me);
	MPI_Comm_size(comm, &nb_nodes);

	va_start(varg_list, codelet);
	/* Find out whether we are to execute the data because we own the data to be written to. */
	ret = _starpu_mpi_task_decode_v(codelet, me, nb_nodes, &xrank, &dest, &do_execute, varg_list);
	if (ret < 0) return ret;
	va_end(varg_list);

	return _starpu_mpi_task_postbuild_v(comm, codelet, varg_list, xrank, dest, do_execute);
}

void starpu_mpi_get_data_on_node_detached(MPI_Comm comm, starpu_data_handle_t data_handle, int node, void (*callback)(void*), void *arg)
{
	int me, rank, tag;

	rank = starpu_data_get_rank(data_handle);
	tag = starpu_data_get_tag(data_handle);
	if (rank == -1)
	{
		_STARPU_ERROR("StarPU needs to be told the MPI rank of this data, using starpu_mpi_data_register() or starpu_data_set_rank()\n");
	}
	if (tag == -1)
	{
		_STARPU_ERROR("StarPU needs to be told the MPI tag of this data, using starpu_mpi_data_register() or starpu_data_set_tag()\n");
	}
	MPI_Comm_rank(comm, &me);

	if (node == rank) return;

	if (me == node)
	{
		starpu_mpi_irecv_detached(data_handle, rank, tag, comm, callback, arg);
	}
	else if (me == rank)
	{
		starpu_mpi_isend_detached(data_handle, node, tag, comm, NULL, NULL);
	}
}

void starpu_mpi_get_data_on_node(MPI_Comm comm, starpu_data_handle_t data_handle, int node)
{
	int me, rank, tag;

	rank = starpu_data_get_rank(data_handle);
	tag = starpu_data_get_tag(data_handle);
	if (rank == -1)
	{
		fprintf(stderr,"StarPU needs to be told the MPI rank of this data, using starpu_data_set_rank\n");
		STARPU_ABORT();
	}
	if (tag == -1)
	{
		fprintf(stderr,"StarPU needs to be told the MPI tag of this data, using starpu_data_set_tag\n");
		STARPU_ABORT();
	}
	MPI_Comm_rank(comm, &me);

	if (node == rank) return;

	if (me == node)
	{
		MPI_Status status;
		starpu_mpi_recv(data_handle, rank, tag, comm, &status);
	}
	else if (me == rank)
	{
		starpu_mpi_send(data_handle, node, tag, comm);
	}
}

struct _starpu_mpi_redux_data_args
{
	starpu_data_handle_t data_handle;
	starpu_data_handle_t new_handle;
	int tag;
	int node;
	MPI_Comm comm;
	struct starpu_task *taskB;
};

void _starpu_mpi_redux_data_dummy_func(STARPU_ATTRIBUTE_UNUSED void *buffers[], STARPU_ATTRIBUTE_UNUSED void *cl_arg)
{
}

static
struct starpu_codelet _starpu_mpi_redux_data_read_cl =
{
	.cpu_funcs = {_starpu_mpi_redux_data_dummy_func, NULL},
	.cuda_funcs = {_starpu_mpi_redux_data_dummy_func, NULL},
	.opencl_funcs = {_starpu_mpi_redux_data_dummy_func, NULL},
	.nbuffers = 1,
	.modes = {STARPU_R},
	.name = "_starpu_mpi_redux_data_read_cl"
};

struct starpu_codelet _starpu_mpi_redux_data_readwrite_cl =
{
	.cpu_funcs = {_starpu_mpi_redux_data_dummy_func, NULL},
	.cuda_funcs = {_starpu_mpi_redux_data_dummy_func, NULL},
	.opencl_funcs = {_starpu_mpi_redux_data_dummy_func, NULL},
	.nbuffers = 1,
	.modes = {STARPU_RW},
	.name = "_starpu_mpi_redux_data_write_cl"
};

static
void _starpu_mpi_redux_data_detached_callback(void *arg)
{
	struct _starpu_mpi_redux_data_args *args = (struct _starpu_mpi_redux_data_args *) arg;

	STARPU_TASK_SET_HANDLE(args->taskB, args->new_handle, 1);
	int ret = starpu_task_submit(args->taskB);
	STARPU_ASSERT(ret == 0);

	starpu_data_unregister_submit(args->new_handle);
	free(args);
}

static
void _starpu_mpi_redux_data_recv_callback(void *callback_arg)
{
	struct _starpu_mpi_redux_data_args *args = (struct _starpu_mpi_redux_data_args *) callback_arg;
	starpu_data_register_same(&args->new_handle, args->data_handle);

	starpu_mpi_irecv_detached_sequential_consistency(args->new_handle, args->node, args->tag, args->comm, _starpu_mpi_redux_data_detached_callback, args, 0);
}

/* TODO: this should rather be implicitly called by starpu_mpi_task_insert when
 * a data previously accessed in REDUX mode gets accessed in R mode. */
void starpu_mpi_redux_data(MPI_Comm comm, starpu_data_handle_t data_handle)
{
	int me, rank, tag, nb_nodes;

	rank = starpu_data_get_rank(data_handle);
	tag = starpu_data_get_tag(data_handle);
	if (rank == -1)
	{
		fprintf(stderr,"StarPU needs to be told the MPI rank of this data, using starpu_data_set_rank\n");
		STARPU_ABORT();
	}
	if (tag == -1)
	{
		fprintf(stderr,"StarPU needs to be told the MPI tag of this data, using starpu_data_set_tag\n");
		STARPU_ABORT();
	}

	MPI_Comm_rank(comm, &me);
	MPI_Comm_size(comm, &nb_nodes);

	_STARPU_MPI_DEBUG(1, "Doing reduction for data %p on node %d with %d nodes ...\n", data_handle, rank, nb_nodes);

	// need to count how many nodes have the data in redux mode
	if (me == rank)
	{
		int i, j=0;
		struct starpu_task *taskBs[nb_nodes];

		for(i=0 ; i<nb_nodes ; i++)
		{
			if (i != rank)
			{
				/* We need to make sure all is
				 * executed after data_handle finished
				 * its last read access, we hence do
				 * the following:
				 * - submit an empty task A reading
				 * data_handle whose callback submits
				 * the mpi comm with sequential
				 * consistency set to 0, whose
				 * callback submits the redux_cl task
				 * B with sequential consistency set
				 * to 0,
				 * - submit an empty task C reading
				 * and writing data_handle and
				 * depending on task B, just to replug
				 * with implicit data dependencies
				 * with tasks inserted after this
				 * reduction.
				 */

				struct _starpu_mpi_redux_data_args *args = malloc(sizeof(struct _starpu_mpi_redux_data_args));
				args->data_handle = data_handle;
				args->tag = tag;
				args->node = i;
				args->comm = comm;

				// We need to create taskB early as
				// taskC declares a dependancy on it
				args->taskB = starpu_task_create();
				args->taskB->cl = args->data_handle->redux_cl;
				args->taskB->sequential_consistency = 0;
				STARPU_TASK_SET_HANDLE(args->taskB, args->data_handle, 0);
				taskBs[j] = args->taskB; j++;

				// Submit taskA
				starpu_task_insert(&_starpu_mpi_redux_data_read_cl,
						   STARPU_R, data_handle,
						   STARPU_CALLBACK_WITH_ARG, _starpu_mpi_redux_data_recv_callback, args,
						   0);
			}
		}

		// Submit taskC which depends on all taskBs created
		struct starpu_task *taskC = starpu_task_create();
		taskC->cl = &_starpu_mpi_redux_data_readwrite_cl;
		STARPU_TASK_SET_HANDLE(taskC, data_handle, 0);
		starpu_task_declare_deps_array(taskC, j, taskBs);
		int ret = starpu_task_submit(taskC);
		STARPU_ASSERT(ret == 0);
	}
	else
	{
		_STARPU_MPI_DEBUG(1, "Sending redux handle to %d ...\n", rank);
		starpu_mpi_isend_detached(data_handle, rank, tag, comm, NULL, NULL);
		starpu_task_insert(data_handle->init_cl, STARPU_W, data_handle, 0);
	}
	/* FIXME: In order to prevent simultaneous receive submissions
	 * on the same handle, we need to wait that all the starpu_mpi
	 * tasks are done before submitting next tasks. The current
	 * version of the implementation does not support multiple
	 * simultaneous receive requests on the same handle.*/
	starpu_task_wait_for_all();

}
