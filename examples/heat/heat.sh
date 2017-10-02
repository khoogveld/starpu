#!/bin/bash
#
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2017  Université de Bordeaux
# Copyright (C) 2017  Inria
#
# StarPU is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or (at
# your option) any later version.
#
# StarPU is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Lesser General Public License in COPYING.LGPL for more details.

# Test various LU options

set -e

PREFIX=$(dirname $0)

if [ -n "$STARPU_MIC_SINK_PROGRAM_PATH" ] ; then
	STARPU_MIC_SINK_PROGRAM_NAME=$STARPU_MIC_SINK_PROGRAM_PATH/heat
	# in case libtool got into play
	[ -x "$STARPU_MIC_SINK_PROGRAM_PATH/.libs/heat" ] && STARPU_MIC_SINK_PROGRAM_NAME=$STARPU_MIC_SINK_PROGRAM_PATH/.libs/heat
fi

$PREFIX/heat -shape 0
$PREFIX/heat -shape 1
# sometimes lead to pivot being 0
#$PREFIX/heat -shape 2

$PREFIX/heat -cg

# TODO: FIXME

# segfault
#$PREFIX/heat -v1

# (actually the default...)
$PREFIX/heat -v2

# hang
#$PREFIX/heat -v3

# hang
#$PREFIX/heat -v4
