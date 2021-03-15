##############################################################################
#                                                                            #
#  This is a demonstration of the ritsar toolset using Sandia data.          #
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

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
count = comm.Get_size()

#Include standard library dependencies
import os
import matplotlib as mpl
mpl.use('Agg') # this allows it to work without $DISPLAY environment variable
import matplotlib.pylab as plt

#Include SARIT toolset
from ritsar import phsRead
from ritsar import phsTools
from ritsar import imgTools

#Define directory containing *.au2 and *.phs files
directory = './data/Sandia/'

#Import phase history and create platform dictionary
[phs, platform] = phsRead.Sandia(directory)

#Correct for residual video phase
phs_corr = phsTools.RVP_correct(phs, platform)

#Import image plane dictionary from './parameters/img_plane'
img_plane = imgTools.img_plane_dict(platform,
                           res_factor = 1.0, n_hat = platform['n_hat'])

img_bp   = imgTools.backprojection_mpi(phs, platform, img_plane, taylor = 30)

#Apply polar format algorithm to phase history data
#(Other options not available since platform position is unknown)
#img_pf = imgTools.polar_format(phs_corr, platform, img_plane, taylor = 30)

if rank == 0:
    #Output image
    imgTools.imshow(img_bp, [-45,0])
    outfn = "BPmpi_Sandia_demo_" + str(count) + ".png"
    if os.getenv("OUTFILE") is not None:
        outfn = os.getenv("OUTFILE")
    plt.savefig(outfn)
