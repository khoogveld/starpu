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

#include <starpu.h>

void cpu_codelet(void *descr[], void *_args)
{
	float *block = (float *)STARPU_BLOCK_GET_PTR(descr[0]);
	int nx = (int)STARPU_BLOCK_GET_NX(descr[0]);
	int ny = (int)STARPU_BLOCK_GET_NY(descr[0]);
	int nz = (int)STARPU_BLOCK_GET_NZ(descr[0]);
        float *multiplier = (float *)_args;
        int i;

        for(i=0 ; i<nx*ny*nz ; i++) block[i] *= *multiplier;
}

