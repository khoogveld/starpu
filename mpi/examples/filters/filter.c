/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2019-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/*
 * This examplifies how to declare a new filter function.
 */

#include <starpu_mpi.h>

#define NX    20

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

void cpu_func(void *buffers[], void *cl_arg)
{
	size_t i;
	int factor;
	int rank;

	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	fprintf(stderr, "computing on rank %d\n", rank);
	size_t n = STARPU_VECTOR_GET_NX(buffers[0]);
	int *val = (int *)STARPU_VECTOR_GET_PTR(buffers[0]);
	starpu_codelet_unpack_args(cl_arg, &factor);

	for (i = 0; i < n; i++)
		val[i] *= factor;
}

struct starpu_codelet cl =
{
	.cpu_funcs = {cpu_func},
	.cpu_funcs_name = {"cpu_func"},
	.nbuffers = 1,
	.modes = {STARPU_RW},
	.name = "vector_scal"
};

void vector_filter(void *parent_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nchunks)
{
	struct starpu_vector_interface *vector_parent = (struct starpu_vector_interface *) parent_interface;
	struct starpu_vector_interface *vector_child = (struct starpu_vector_interface *) child_interface;

	size_t nx = vector_parent->nx;
	size_t elemsize = vector_parent->elemsize;

	STARPU_ASSERT_MSG(nchunks <= nx, "%u parts for %zu elements", nchunks, nx);
	STARPU_ASSERT(nchunks == 2);
	STARPU_ASSERT_MSG((nx % nchunks) == 0, "nx=%zu is not a multiple of nchunks %u\n", nx, nchunks);

	vector_child->id = vector_parent->id;
	vector_child->nx = nx/2;
	vector_child->elemsize = elemsize;
	vector_child->allocsize = vector_child->nx * elemsize;

	if (vector_parent->dev_handle)
	{
		size_t offset = (id *(nx/nchunks)) * elemsize;
		if (vector_parent->ptr) vector_child->ptr = vector_parent->ptr + offset;
		vector_child->dev_handle = vector_parent->dev_handle;
		vector_child->offset = vector_parent->offset + offset;
	}
}

int main(int argc, char **argv)
{
	return 77;
	int i, rank, nodes;
	int vector[NX];
	int vector_check[NX];
	starpu_data_handle_t vhandle;
	starpu_data_handle_t handles[2];
	int factor[2] = {2, 3};
	int ret;

	ret = starpu_mpi_init_conf(&argc, &argv, 1, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &nodes);

	if (nodes < 2 || (starpu_cpu_worker_get_count() == 0))
	{
		if (rank == 0)
		{
			if (nodes < 2)
				fprintf(stderr, "We need at least 2 processes.\n");
			else
				fprintf(stderr, "We need at least 1 CPU.\n");
		}
		starpu_mpi_shutdown();
		return 77;
	}

	for(i=0 ; i<NX ; i++)
	{
		vector[i] = i+1;
		if (i < NX/2)
			vector_check[i] = vector[i]*factor[0];
		else
			vector_check[i] = vector[i]*factor[1];
	}
	FPRINTF(stderr,"IN  Vector: ");
	for(i=0 ; i<NX ; i++) FPRINTF(stderr, "%5d ", vector[i]);
	FPRINTF(stderr,"\n");

	/* Declare data to StarPU */
	if (rank == 0)
		starpu_vector_data_register(&vhandle, STARPU_MAIN_RAM, (uintptr_t)vector, NX, sizeof(vector[0]));
	else
		starpu_vector_data_register(&vhandle, -1, (uintptr_t)NULL, NX, sizeof(vector[0]));

	/* Partition the vector in PARTS sub-vectors */
	struct starpu_data_filter f =
	{
		.filter_func = vector_filter,
		.nchildren = 2
	};
	starpu_data_partition_plan(vhandle, &f, handles);
	starpu_data_partition_submit(vhandle, 2, handles);

	/* Submit a task on each sub-vector */
	for (i=0; i<2; i++)
	{
		starpu_mpi_data_register(handles[i], 42+i, 0);
		ret = starpu_mpi_task_insert(MPI_COMM_WORLD,
					     &cl,
					     STARPU_RW, handles[i],
					     STARPU_VALUE, &factor[i], sizeof(factor[i]),
					     STARPU_EXECUTE_ON_NODE, 1,
					     0);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	starpu_data_unpartition_submit(vhandle, 2, handles, -1);
	starpu_data_partition_clean(vhandle, 2, handles);
	int ok=0;
	if (rank == 0)
	{
		starpu_data_acquire(vhandle, STARPU_R);
		int *v = starpu_data_get_local_ptr(vhandle);
		FPRINTF(stderr,"OUT Vector: ");
		for(i=0 ; i<NX ; i++)
		{
			FPRINTF(stderr, "%5d ", v[i]);
			if (v[i] != vector_check[i])
			{
				FPRINTF(stderr, "%5d should be %5d\n", v[i], vector_check[i]);
				ok=1;
			}
		}
		FPRINTF(stderr,"\n");
		starpu_data_release(vhandle);
	}

	starpu_data_unregister(vhandle);
	starpu_mpi_shutdown();

	return ok;
}
