#!/bin/bash
rm -rf media/images/vis media/videos/vis
time manim vis.py
mkdir -p media/images/vis/
ffmpeg -i media/videos/vis/*/Decompose.mp4 -vf scale=800:450 -f image2 media/images/vis/Decompose%07d.png
time img2webp -o Decompose.webp -min_size -d 50 -loop 0 media/images/vis/*.png
