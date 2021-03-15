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
import numpy as np
sys.path.append('../')

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
count = comm.Get_size()

#Include standard library dependencies
import os
import matplotlib as mpl
mpl.use('Agg') # this allows it to work without $DISPLAY environment variable
import matplotlib.pylab as plt
from time import time

#Include SARIT toolset
from ritsar import phsRead
from ritsar import imgTools

#Define top level directory containing *.mat file
#and choose polarization and starting azimuth
pol = 'HH_npy'
directory = './data/AFRL_P4096_S4096/pass1'
if os.getenv("DATAPATH") is not None:
    directory = os.getenv("DATAPATH")
    pol = '.'
start_az = 1

#Import phase history and create platform dictionary
[phs, platform] = phsRead.Halide_SAR(directory, pol, start_az, n_az = 3)
print("phs.shape:", phs.shape)

#Create image plane dictionary
img_plane = imgTools.img_plane_dict(platform, res_factor = 0.1555, upsample = True, aspect = 1.0)

before = time()
#Apply algorithm of choice to phase history data
img_bp = imgTools.backprojection_mpi(phs, platform, img_plane, taylor = 20, upsample = 6)
#img_pf = imgTools.polar_format(phs, platform, img_plane, taylor = 20)
after = time()

if rank == 0:
    print("backprojection_mpi took", after-before, "seconds")
    #Output image
    imgTools.imshow(img_bp, dB_scale = [-30,0])
    outfn = "BPmpi_AFRL4K_demo_" + str(count) + ".png"
    if os.getenv("OUTFILE") is not None:
        outfn = os.getenv("OUTFILE")
    plt.savefig(outfn)
