# Intro

SAR

Synthetic-Aperture Radar

# Implementations

## original
This is a copy of the RITSAR implementation, with fixes to allow it to run on
more recent versions of Python.

### larger datasets

We've added a function to read input data from `.npy` data files.

The Halide SAR app has some data dumps in this format, and a tool to
expand the data to arbitrary sizes for scalability testing.  See
[halide-sar-app/data](https://github.com/ISI-apex/halide-sar-app/tree/master/data)
for this data and tool.

### MPI distributed operation

The driver scripts in `original/examples` named with a `BPmpi_` prefix
run an MPI distributed version of the `backprojection()` function.
You must have the `mpi4py` module installed to run these.

In this mode of operation, all processes read their own copies of the
datasets, and split up the work evenly.  But only rank 0 prints things
and only rank 0 writes the image data at the end.

There is no synchronization between ranks until the end, so there is no
guarantee that all ranks are running at the same speed.  Thus, rank 0's
progress print statements may not line up perfectly with the rest.  But
it should generally be pretty close.

# Basis for comparison

To keep comparisons clean, the problem size, compiler version and CFLAGS should
be kept consistent for all builds.

## Language version

The code is run using Python version 3.6.9.

## Baseline

Our baseline performance metric will be execution of the AutoFocus demo, using
the Sandia example dataset.

# Reference stuff

* [RITSAR website](https://github.com/dm6718/RITSAR)
* [python 3.6.9 fixes](https://github.com/dm6718/RITSAR/pull/2)
