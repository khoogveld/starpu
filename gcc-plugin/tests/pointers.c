/* GCC-StarPU
   Copyright (C) 2011 Institut National de Recherche en Informatique et Automatique

   GCC-StarPU is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   GCC-StarPU is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with GCC-StarPU.  If not, see <http://www.gnu.org/licenses/>.  */

#undef NDEBUG

#include <lib.h>


/* The task under test.  */

static void my_pointer_task (int *x, long long *y) __attribute__ ((task));

static void my_pointer_task_cpu (int *, long long *)
  __attribute__ ((task_implementation ("cpu", my_pointer_task)));
static void my_pointer_task_opencl (int *, long long *)
  __attribute__ ((task_implementation ("opencl", my_pointer_task)));

static void
my_pointer_task_cpu (int *x, long long *y)
{
  printf ("%s: x = %p, y = %p\n", __func__, x, y);
}

static void
my_pointer_task_opencl (int *x, long long *y)
{
  printf ("%s: x = %p, y = %p\n", __func__, x, y);
}


int
main (int argc, char *argv[])
{
  int x[] = { 42 };
  long long *y;

  y = malloc (sizeof *y);
  *y = 77;

  struct insert_task_argument expected[] =
    {
      { STARPU_RW, x },
      { STARPU_RW, y },
      { 0, 0, 0 }
    };

  expected_insert_task_arguments = expected;

  /* Invoke the task, which should make sure it gets called with
     EXPECTED.  */
  my_pointer_task (x, y);

  assert (tasks_submitted == 1);

  free (y);

  return EXIT_SUCCESS;
}
