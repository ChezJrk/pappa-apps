##############################################################################
#                                                                            #
#  This is a demonstration of the ritsar toolset using AFRL Gotcha data.     #
#  Algorithms can be switched in and out by commenting/uncommenting          #
#  the lines of code below.                                                  #
#                                                                            #
#  This version uses MPI to calculate the result in a distributed fashion,   #
#  splitting up the output image across the Y axis and gathering image data  #
#  at the end.                                                               #
#                                                                            #
##############################################################################

#Add include directories to default path list
import sys
sys.path.append('../')

#Include standard library dependencies
import getopt
import os
import matplotlib as mpl
mpl.use('Agg') # this allows it to work without $DISPLAY environment variable
import matplotlib.pylab as plt
from time import time

#Include SARIT toolset
from ritsar import phsRead
from ritsar import imgTools

use_mpi = False
use_cuda = False
mpi_barriers = False
main_or_only_node = True
res_factor = None

# directory containing *.npy files
dataset = './data/AFRL_P1024_S1024'
# path of png file to generate
outfn = None
if os.getenv("DATAPATH") is not None:
    dataset = os.getenv("DATAPATH")
if os.getenv("OUTFILE") is not None:
    outfn = os.getenv("OUTFILE")

opts, _ = getopt.getopt(sys.argv[1:], "", [
    "help",
    "dataset=",
    "output=",
    "res_factor=",
    "cuda",
    "mpi",
    "barrier",
])

for opt, value in opts:
    if opt == '--help':
        print("Usage: {} [--mpi [--barrier]] [--cuda] [--dataset=/path/to/data] [--output=/path/to/output.png]".format(sys.argv[0]))
        exit(0)
    if opt == '--mpi':
        use_mpi = True
    if opt == '--barrier':
        mpi_barrier = True
    if opt == '--cuda':
        use_cuda = True
    if opt == '--output':
        outfn = value
    if opt == '--dataset':
        dataset = value
    if opt == '--res_factor':
        res_factor = float(value)

if use_mpi:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    count = comm.Get_size()
    main_or_only_node = rank == 0
    if main_or_only_node:
        print("Using MPI with {} processes".format(count))
        if mpi_barriers:
            print("MPI will barrier between pulses")

if use_cuda:
    if main_or_only_node:
        print("Using CUDA")

#Import phase history and create platform dictionary
[phs, platform] = phsRead.Halide_SAR(dataset)
dataset_size = phs.shape[0]
dataset_k = str(int(dataset_size/1024)) + "K"
if res_factor is None:
    # original AFRL phs array is 352 pulses x 424 samples
    res_factor = 512.0 / dataset_size

if main_or_only_node:
    print("AFRL {} dataset".format(dataset_size))

#Create image plane dictionary
img_plane = imgTools.img_plane_dict(platform, res_factor = res_factor, upsample = True, aspect = 1.0)

before = time()
img_bp = imgTools.backprojection(phs, platform, img_plane, taylor = 20, upsample = 6, prnt=main_or_only_node, use_mpi=use_mpi, use_cuda=use_cuda, mpi_barriers=mpi_barriers)
after = time()

if main_or_only_node:
    print("backprojection took", after-before, "seconds")
    #Output image
    imgTools.imshow(img_bp, dB_scale = [-30,0])
    if outfn is None:
        outfn = "BP_AFRL{}_demo".format(dataset_k)
        if use_mpi:
            outfn += "_MPI_" + str(count)
            if mpi_barriers:
                outfn += "_barrier"
        if use_cuda:
            outfn += "_CUDA"
        outfn += ".png"
    plt.savefig(outfn)
