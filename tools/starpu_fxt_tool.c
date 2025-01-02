/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2020,2021  Federal University of Rio Grande do Sul (UFRGS)
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
 * This program should be used to parse the log generated by FxT
 */

#include <starpu.h>
#include <common/config.h>
#include <common/fxt.h>

#define PROGNAME "starpu_fxt_tool"

static void usage()
{
	fprintf(stderr, "Generate a trace in the Paje format\n\n");
	fprintf(stderr, "Usage: %s [ options ]\n", PROGNAME);
	fprintf(stderr, "\n");
	fprintf(stderr, "Options:\n");
	fprintf(stderr, "   -i <input file[s]>	specify the input file[s]. Several files can be provided,\n");
	fprintf(stderr, "			or the option specified several times for MPI execution\n");
	fprintf(stderr, "			case\n");
	fprintf(stderr, "   -o <output file>	specify the paje output filename\n");
	fprintf(stderr, "   -d <directory>	specify the directory in which to save files\n");
	fprintf(stderr, "   -c			use a different colour for every type of task\n");
	fprintf(stderr, "   -no-events		do not show events\n");
	fprintf(stderr, "   -no-counter		do not show scheduler counters\n");
	fprintf(stderr, "   -no-bus		do not show PCI bus transfers\n");
	fprintf(stderr, "   -no-flops		do not show flops\n");
	fprintf(stderr, "   -no-smooth		avoid smoothing values for gflops etc.\n");
	fprintf(stderr, "   -no-acquire		do not show application data acquisitions tasks in DAG\n");
	fprintf(stderr, "   -label-deps		add label on dependencies.\n");
	fprintf(stderr, "   -memory-states	show detailed memory states of handles\n");
	fprintf(stderr, "   -internal		show StarPU-internal tasks in DAG\n");
	fprintf(stderr, "   -number-events	generate a file counting FxT events by type\n");
	fprintf(stderr, "   -use-task-color	propagate the specified task color to the contexts\n");
	fprintf(stderr, "   -h, --help		display this help and exit\n");
	fprintf(stderr, "   -v, --version	output version information and exit\n\n");
	fprintf(stderr, "Report bugs to <%s>.", PACKAGE_BUGREPORT);
	fprintf(stderr, "\n");
}

static struct starpu_fxt_options options;

static int parse_args(int argc, char **argv)
{
	/* Default options */
	starpu_fxt_options_init(&options);

	/* We want to support arguments such as "fxt_tool -i trace_*" */
	unsigned reading_input_filenames = 0;

	int i;
	for (i = 1; i < argc; i++)
	{
		int ret = _starpu_generate_paje_trace_read_option(argv[i], &options);
		if (ret == 0)
		{
			reading_input_filenames = 0;
		}
		else if (strcmp(argv[i], "-o") == 0)
		{
			free(options.out_paje_path);
			options.out_paje_path = strdup(argv[++i]);
			reading_input_filenames = 0;
		}
		else if (strcmp(argv[i], "-d") == 0)
		{
			options.dir = argv[++i];
			reading_input_filenames = 0;
		}
		else if (strcmp(argv[i], "-i") == 0)
		{
			if (options.ninputfiles >= STARPU_FXT_MAX_FILES)
			{
				fprintf(stderr, "Error: The number of trace files is superior to STARPU_FXT_MAX_FILES (%d)\nPlease recompile StarPU with a bigger --enable-fxt-max-files\n", STARPU_FXT_MAX_FILES);
				return 7;
			}
			options.filenames[options.ninputfiles++] = argv[++i];
			reading_input_filenames = 1;
		}
		else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
		{
			usage();
			return 77;
		}
		else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--version") == 0)
		{
			fputs(PROGNAME " (" PACKAGE_NAME ") " PACKAGE_VERSION "\n", stderr);
			return 77;
		}

		/* That's pretty dirty: if the reading_input_filenames flag is
		 * set, and that the argument does not match an option, we
		 * assume this may be another filename */
		else if (reading_input_filenames)
		{
			if (options.ninputfiles >= STARPU_FXT_MAX_FILES)
			{
				fprintf(stderr, "Error: The number of trace files is superior to STARPU_FXT_MAX_FILES (%d)\nPlease recompile StarPU with a bigger --enable-fxt-max-files\n", STARPU_FXT_MAX_FILES);
				return 7;
			}
			options.filenames[options.ninputfiles++] = argv[i];
		}
	}

	if (!options.ninputfiles)
	{
		fprintf(stderr, "Incorrect usage, aborting\n");
		usage();
		return 77;
	}

	return 0;
}

int main(int argc, char **argv)
{
	int ret = parse_args(argc, argv);
	if (ret)
	{
		starpu_fxt_options_shutdown(&options);
		return ret;
	}

	starpu_fxt_generate_trace(&options);

	starpu_fxt_options_shutdown(&options);

	return 0;
}
