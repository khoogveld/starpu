/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2024  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013-2013  Thibaut Lambert
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

#include <starpu.h>
#include "../helper.h"

/*
 *	Memset
 */

void cpu_memset_codelet(void *descr[], void *arg)
{
	(void)arg;
	STARPU_SKIP_IF_VALGRIND;

	char *buf = (char *)STARPU_VECTOR_GET_PTR(descr[0]);
	unsigned length = STARPU_VECTOR_GET_NX(descr[0]);

	memset(buf, 42, length * sizeof(*buf));
}

#ifdef STARPU_USE_CUDA
static void cuda_memset_codelet(void *descr[], void *arg)
{
	(void)arg;
	STARPU_SKIP_IF_VALGRIND;

	char *buf = (char *)STARPU_VECTOR_GET_PTR(descr[0]);
	unsigned length = STARPU_VECTOR_GET_NX(descr[0]);

	cudaMemsetAsync(buf, 42, length, starpu_cuda_get_local_stream());
}
#endif

#ifdef STARPU_USE_OPENCL
static void opencl_memset_codelet(void *buffers[], void *args)
{
	(void) args;
	STARPU_SKIP_IF_VALGRIND;

	cl_command_queue queue;
	int id = starpu_worker_get_id_check();
	int devid = starpu_worker_get_devid(id);
	starpu_opencl_get_queue(devid, &queue);

	cl_mem buffer = (cl_mem) STARPU_VECTOR_GET_DEV_HANDLE(buffers[0]);
	unsigned length = STARPU_VECTOR_GET_NX(buffers[0]);
	char *v = malloc(length);
	STARPU_ASSERT(v != NULL);
	memset(v, 42, length);

	cl_int err;
	err = clEnqueueWriteBuffer(queue,
				   buffer,
				   CL_FALSE,
				   0,      /* offset */
				   length, /* sizeof (char) */
				   v,
				   0,      /* num_events_in_wait_list */
				   NULL,   /* event_wait_list */
				   NULL    /* event */);
	if (STARPU_UNLIKELY(err != CL_SUCCESS)) STARPU_OPENCL_REPORT_ERROR(err);
}
#endif /* !STARPU_USE_OPENCL */

struct starpu_codelet memset_cl =
{
	.cpu_funcs = {cpu_memset_codelet},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {cuda_memset_codelet},
	.cuda_flags = {STARPU_CUDA_ASYNC},
#endif
#ifdef STARPU_USE_OPENCL
	.opencl_funcs = {opencl_memset_codelet},
	.opencl_flags = {STARPU_OPENCL_ASYNC},
#endif
	.cpu_funcs_name = {"cpu_memset_codelet"},
	.nbuffers = 1,
	.modes = {STARPU_W}
};

/*
 *	Check content
 */

void cpu_check_content_codelet(void *descr[], void *arg)
{
	(void)arg;
	STARPU_SKIP_IF_VALGRIND;

	char *buf = (char *)STARPU_VECTOR_GET_PTR(descr[0]);
	unsigned length = STARPU_VECTOR_GET_NX(descr[0]);

	unsigned i;
	for (i = 0; i < length; i++)
	{
		if (buf[i] != 42)
		{
			FPRINTF(stderr, "buf[%u] is '%c' while it should be '%c'\n", i, buf[i], 42);
			exit(-1);
		}
	}
}

#ifdef STARPU_USE_CUDA
static void cuda_check_content_codelet(void *descr[], void *arg)
{
	(void)arg;
	STARPU_SKIP_IF_VALGRIND;

	char *buf = (char *)STARPU_VECTOR_GET_PTR(descr[0]);
	unsigned length = STARPU_VECTOR_GET_NX(descr[0]);

	unsigned i;
	for (i = 0; i < length; i++)
	{
		char dst;
		cudaMemcpyAsync(&dst, &buf[i], sizeof(char), cudaMemcpyDeviceToHost, starpu_cuda_get_local_stream());
		cudaStreamSynchronize(starpu_cuda_get_local_stream());
		if (dst != 42)
		{
			FPRINTF(stderr, "buf[%u] is %c while it should be %c\n", i, dst, 42);
			exit(-1);
		}
	}
}
#endif
#ifdef STARPU_USE_OPENCL
static void opencl_check_content_codelet(void *buffers[], void *args)
{
	(void)args;

	STARPU_SKIP_IF_VALGRIND;

	cl_command_queue queue;
	int id = starpu_worker_get_id_check();
	int devid = starpu_worker_get_devid(id);
	starpu_opencl_get_queue(devid, &queue);

	cl_mem buf = (cl_mem) STARPU_VECTOR_GET_DEV_HANDLE(buffers[0]);
	unsigned length = STARPU_VECTOR_GET_NX(buffers[0]);

	unsigned i;
	for (i = 0; i < length; i++)
	{
		char dst;
		cl_int err;

		err = clEnqueueReadBuffer(queue,
					  buf,
					  CL_FALSE,
					  i * sizeof(dst),
					  sizeof(dst),
					  &dst,
					  0,      /* num_events_in_wait_list */
					  NULL,   /* event_wait_list */
					  NULL    /* event */);
		if (STARPU_UNLIKELY(err != CL_SUCCESS)) STARPU_OPENCL_REPORT_ERROR(err);

		clFinish(queue);
		if (dst != 42)
		{
			FPRINTF(stderr, "buf[%u] is '%c' while it should be '%c'\n", i, dst, 42);
			exit(-1);
		}
	}
}
#endif

struct starpu_codelet memset_check_content_cl =
{
	.cpu_funcs = {cpu_check_content_codelet},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {cuda_check_content_codelet},
#endif
#ifdef STARPU_USE_OPENCL
	.opencl_funcs = {opencl_check_content_codelet},
#endif
	.cpu_funcs_name = {"cpu_check_content_codelet"},
	.nbuffers = 1,
	.modes = {STARPU_R}
};


