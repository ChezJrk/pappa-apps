##############################################################################
#                                                                            #
#  This is a demonstration of the ritsar toolset using AFRL Gotcha data.     #
#  Algorithms can be switched in and out by commenting/uncommenting          #
#  the lines of code below.                                                  #
#                                                                            #
##############################################################################

#Add include directories to default path list
import sys
sys.path.append('../')
if len(sys.argv) > 1 and sys.argv[1] == '--batch':
    batch = True
else:
    batch = False

#Include standard library dependencies
import os
import matplotlib as mpl
mpl.use('Agg') # this allows it to work without $DISPLAY environment variable
import matplotlib.pylab as plt

#Include SARIT toolset
from ritsar import phsRead
from ritsar import imgTools

#Define top level directory containing *.mat file
#and choose polarization and starting azimuth
pol = 'HH'
directory = './data/AFRL/pass1'
start_az = 1

#Import phase history and create platform dictionary
[phs, platform] = phsRead.AFRL(directory, pol, start_az, n_az = 3)

#Create image plane dictionary
img_plane = imgTools.img_plane_dict(platform, res_factor = 1.4, upsample = True, aspect = 1.0)

#Apply algorithm of choice to phase history data
img_bp = imgTools.backprojection_cuda(phs, platform, img_plane, taylor = 20, upsample = 6)
#img_pf = imgTools.polar_format(phs, platform, img_plane, taylor = 20)

#Output image
imgTools.imshow(img_bp, dB_scale = [-30,0])
plt.title('AFRL Backprojection (CUDA)')
outfn = "CUDA_AFRL_demo.png"
if os.getenv("OUTFILE") is not None:
    outfn = os.getenv("OUTFILE")
plt.savefig(outfn)
