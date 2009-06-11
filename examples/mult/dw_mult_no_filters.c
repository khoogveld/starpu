/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include "dw_mult.h"


float *A, *B, *C;
starpu_data_handle A_handle, B_handle, C_handle;

/*
 * That program should compute C = A * B 
 * 
 *   A of size (z,y)
 *   B of size (x,z)
 *   C of size (x,y)

              |---------------|
            z |       B       |
              |---------------|
       z              x
     |----|   |---------------|
     |    |   |               |
     |    |   |               |
     | A  | y |       C       |
     |    |   |               |
     |    |   |               |
     |----|   |---------------|

 */

static void terminate(void)
{
	starpu_delete_data(C_handle);

	gettimeofday(&end, NULL);

	double timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));

	uint64_t total_flop = niter*BLAS3_FLOP(ydim, xdim, zdim);
	uint64_t total_ls = niter*(ls_cublas + ls_atlas);

	fprintf(stderr, "Computation took (ms):\n");
	printf("%2.2f\n", timing/1000);
	fprintf(stderr, "	GFlop : total (%2.2f) cublas (%2.2f) atlas (%2.2f)\n", (double)total_flop/1000000000.0f, (double)flop_cublas/1000000000.0f, (double)flop_atlas/1000000000.0f);
	fprintf(stderr, "	GFlop/s : %2.2f\n", (double)total_flop / (double)timing/1000);
	fprintf(stderr, "	GB : total (%2.2f) cublas (%2.2f) atlas (%2.2f)\n", (double)total_ls/1000000000.0f, (double)ls_cublas/1000000000.0f, (double)ls_atlas/1000000000.0f);
	fprintf(stderr, "	GB/s : %2.2f\n", (double)total_ls / (double)timing/1000);

#ifdef CHECK_OUTPUT
	/* check results */
	/* compute C = C - niter * AB */

	SGEMM("N", "N", ydim, xdim, zdim, -1.0f*niter, A, ydim, B, zdim, 1.0f, C, ydim);
		
	/* make sure C = 0 */
	float err;
	err = SASUM(xdim*ydim, C, 1);	
	
	if (err < xdim*ydim*0.001) {
		fprintf(stderr, "Results are OK\n");
	}
	else {
		fprintf(stderr, "There were errors ... err = %f\n", err);
	}
#endif // CHECK_OUTPUT
}

#define COMMON_CODE			\
	uint32_t nxC, nyC, nyA;		\
	uint32_t ldA, ldB, ldC;		\
					\
	float *subA;			\
	float *subB;			\
	float *subC;			\
					\
	subA = (float *)descr[0].blas.ptr;	\
	subB = (float *)descr[1].blas.ptr;	\
	subC = (float *)descr[2].blas.ptr;	\
					\
	nxC = descr[2].blas.nx;		\
	nyC = descr[2].blas.ny;		\
	nyA = descr[0].blas.ny;		\
					\
	ldA = descr[0].blas.ld;		\
	ldB = descr[1].blas.ld;		\
	ldC = descr[2].blas.ld;



#ifdef USE_CUDA
void cublas_mult(starpu_data_interface_t *descr, __attribute__((unused)) void *arg)
{
	COMMON_CODE

	cublasSgemm('n', 'n', nxC, nyC, nyA, 1.0f, subA, ldA, subB, ldB, 
					     0.0f, subC, ldC);
	cublasStatus st;
	st = cublasGetError();
	if (st != CUBLAS_STATUS_SUCCESS)
		STARPU_ASSERT(0);

	uint64_t flopcnt = BLAS3_FLOP(nyC, nxC, nyA);

	flop_cublas += flopcnt;
	ls_cublas += BLAS3_LS(nyC, nxC, nyA);
}
#endif

void core_mult(starpu_data_interface_t *descr, __attribute__((unused))  void *arg)
{
	COMMON_CODE

	SGEMM("N", "N", nxC, nyC, nyA, 1.0f, subA, ldA, subB, ldB, 0.0f, subC, ldC);

	flop_atlas += BLAS3_FLOP(nxC, nyC, nyA);
	ls_atlas += BLAS3_LS(nxC, nyC, nyA);
}

static void init_problem_data(void)
{
	unsigned i,j;

#ifdef USE_CUDA
	if (pin) {
		starpu_malloc_pinned_if_possible((void **)&A, zdim*ydim*sizeof(float));
		starpu_malloc_pinned_if_possible((void **)&B, xdim*zdim*sizeof(float));
		starpu_malloc_pinned_if_possible((void **)&C, xdim*ydim*sizeof(float));
	} else
#endif
	{
#ifdef HAVE_POSIX_MEMALIGN
		posix_memalign((void **)&A, 4096, zdim*ydim*sizeof(float));
		posix_memalign((void **)&B, 4096, xdim*zdim*sizeof(float));
		posix_memalign((void **)&C, 4096, xdim*ydim*sizeof(float));
#else
		A = malloc(zdim*ydim*sizeof(float));
		B = malloc(xdim*zdim*sizeof(float));
		C = malloc(xdim*ydim*sizeof(float));
#endif
	}

	/* fill the A and B matrices */
	if (norandom) {
		for (j=0; j < ydim; j++) {
			for (i=0; i < zdim; i++) {
				A[j+i*ydim] = (float)(i);
			}
		}
	
		for (j=0; j < zdim; j++) {
			for (i=0; i < xdim; i++) {
				B[j+i*zdim] = (float)(j);
			}
		}
	} 
	else {
#ifdef NORANDOM
		srand(2009);
		STARPU_ASSERT(0);
#endif
		for (j=0; j < ydim; j++) {
			for (i=0; i < zdim; i++) {
				A[j+i*ydim] = (float)(drand48());
			}
		}
	
		for (j=0; j < zdim; j++) {
			for (i=0; i < xdim; i++) {
				B[j+i*zdim] = (float)(drand48());
			}
		}
	}

	for (j=0; j < ydim; j++) {
		for (i=0; i < xdim; i++) {
			C[j+i*ydim] = (float)(0);
		}
	}

	display_memory_consumption();

	starpu_register_blas_data(&A_handle, 0, (uintptr_t)A, 
		ydim, ydim, zdim, sizeof(float));
	starpu_register_blas_data(&B_handle, 0, (uintptr_t)B, 
		zdim, zdim, xdim, sizeof(float));
	starpu_register_blas_data(&C_handle, 0, (uintptr_t)C, 
		ydim, ydim, xdim, sizeof(float));

	gettimeofday(&start, NULL);
}

static void launch_codelets(void)
{
	srand(time(NULL));

	starpu_codelet cl = {
		.where = CORE|CUBLAS,
		.core_func = core_mult,
#ifdef USE_CUDA
		.cublas_func = cublas_mult,
#endif
		.model = &sgemm_model,
		.nbuffers = 3
	};

	unsigned iter;
	for (iter = 0; iter < niter; iter++) 
	{
		struct starpu_task *task = starpu_task_create();

		task->cl = &cl;

		task->buffers[0].handle = A_handle;
		task->buffers[0].mode = STARPU_R;
		task->buffers[1].handle = B_handle;
		task->buffers[1].mode = STARPU_R;
		task->buffers[2].handle = C_handle;
		task->buffers[2].mode = STARPU_RW;

		task->synchronous = 1;
		starpu_submit_task(task);
	}
}

int main(__attribute__ ((unused)) int argc, 
	 __attribute__ ((unused)) char **argv)
{

	parse_args(argc, argv);

	/* start the runtime */
	starpu_init(NULL);

	init_problem_data();

	launch_codelets();

	terminate();

	starpu_shutdown();

	return 0;
}
