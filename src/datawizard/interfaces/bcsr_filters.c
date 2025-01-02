/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2010-2010  Mehdi Juhoor
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
#include <common/config.h>
#include <datawizard/filters.h>

void starpu_bcsr_filter_vertical_block(void *parent_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, unsigned id, STARPU_ATTRIBUTE_UNUSED unsigned nparts)
{
	struct starpu_bcsr_interface *bcsr_parent = (struct starpu_bcsr_interface *) parent_interface;
	struct starpu_bcsr_interface *bcsr_child = (struct starpu_bcsr_interface *) child_interface;

	size_t elemsize = bcsr_parent->elemsize;
	uint32_t firstentry = bcsr_parent->firstentry;
	uint32_t r = bcsr_parent->r;
	uint32_t c = bcsr_parent->c;
	uint32_t *ram_rowptr = bcsr_parent->ram_rowptr;
	uint32_t *rowptr = bcsr_parent->rowptr;

	size_t child_nrow;
	size_t child_rowoffset;

	STARPU_ASSERT_MSG(bcsr_parent->id == STARPU_BCSR_INTERFACE_ID, "%s can only be applied on a bcsr data", __func__);

	bcsr_child->id = bcsr_parent->id;

	starpu_filter_nparts_compute_chunk_size_and_offset(bcsr_parent->nrow, nparts, 1, id, 1, &child_nrow, &child_rowoffset);

	/* child blocks indexes between these (0-based) */
	uint32_t start_block = ram_rowptr[child_rowoffset] - firstentry;
	uint32_t end_block = ram_rowptr[child_rowoffset + child_nrow] - firstentry;

	bcsr_child->nnz = end_block - start_block;
	bcsr_child->nrow = child_nrow;

	bcsr_child->firstentry = firstentry + start_block;
	bcsr_child->r = bcsr_parent->r;
	bcsr_child->c = bcsr_parent->c;
	bcsr_child->elemsize = elemsize;
	bcsr_child->ram_colind = bcsr_parent->ram_colind + start_block;
	bcsr_child->ram_rowptr = ram_rowptr + child_rowoffset;

	if (bcsr_parent->nzval)
	{
		bcsr_child->nzval = bcsr_parent->nzval + start_block * r*c * elemsize;
		bcsr_child->colind = bcsr_parent->colind + start_block;
		bcsr_child->rowptr = rowptr + child_rowoffset;
	}
}

void starpu_bcsr_filter_canonical_block(void *parent_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, unsigned id, STARPU_ATTRIBUTE_UNUSED unsigned nparts)
{
	struct starpu_bcsr_interface *bcsr_parent = (struct starpu_bcsr_interface *) parent_interface;
	/* each chunk becomes a small dense matrix */
	struct starpu_matrix_interface *matrix_child = (struct starpu_matrix_interface *) child_interface;

	size_t elemsize = bcsr_parent->elemsize;
	uint32_t firstentry = bcsr_parent->firstentry;

	/* size of the tiles */
	uint32_t r = bcsr_parent->r;
	uint32_t c = bcsr_parent->c;

	uint32_t ptr_offset = c*r*id*elemsize;

	STARPU_ASSERT_MSG(bcsr_parent->id == STARPU_BCSR_INTERFACE_ID, "%s can only be applied on a bcsr data", __func__);

	matrix_child->id = STARPU_MATRIX_INTERFACE_ID;
	matrix_child->nx = c;
	matrix_child->ny = r;
	matrix_child->ld = c;
	matrix_child->elemsize = elemsize;
	matrix_child->allocsize = c*r*elemsize;

	if (bcsr_parent->nzval)
	{
		uint8_t *nzval = (uint8_t *)(bcsr_parent->nzval);
		matrix_child->dev_handle = matrix_child->ptr = (uintptr_t)&nzval[firstentry + ptr_offset];
		matrix_child->offset = 0;
	}
}

unsigned starpu_bcsr_filter_canonical_block_get_nchildren(STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, starpu_data_handle_t handle)
{
	return (unsigned)starpu_bcsr_get_nnz(handle);
}

struct starpu_data_interface_ops *starpu_bcsr_filter_canonical_block_child_ops(STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, STARPU_ATTRIBUTE_UNUSED unsigned child)
{
	return &starpu_interface_matrix_ops;
}
