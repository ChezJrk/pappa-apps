## Makefile for SCF-C

CXX=clang++-10
CXXFLAGS=-O2 -g -Wall -fdiagnostics-color=always
CXXLINK=-ldl -lpthread

THREADS=1
TILE_SIZE=15
VECTOR_SIZE=4
DATA_SIZE=120
PERF_FUNCS=4D_ij_low_kl_low_pairs_low
PERF_FUNCS=all

TRACE_DATA_SIZE=15
TRACE_VIDEO_WIDTH=1920
TRACE_VIDEO_HEIGHT=1080
TRACE_FUNCS=all

run: fake
	HL_NUM_THREADS=$(THREADS) ./fake $(DATA_SIZE)

perf: fake
	HL_NUM_THREADS=$(THREADS) perf record ./fake $(DATA_SIZE)
	perf report

twoel.a twoel.h: fakegen.py ../tools/twoel_gen.py ../tools/decompose.py Makefile
	python3 fakegen.py $(PERF_FUNCS) tilesize=$(TILE_SIZE) vectorsize=$(VECTOR_SIZE)

fake: fake.cpp twoel.a twoel.h
	$(CXX) $(CXXFLAGS) fake.cpp twoel.a -o $@ $(CXXLINK)

twoel.trace: faketwoel.py ../tools/twoel_gen.py ../tools/decompose.py Makefile
	rm -f $@
	HL_NUM_THREADS=$(THREADS) HL_TRACE_FILE=$@ python3 faketwoel.py all datasize=$(TRACE_DATA_SIZE) tilesize=$(TILE_SIZE) vectorsize=$(VECTOR_SIZE) itercount=0 tracing=True

video: twoel.trace
	rm -f twoel.mp4
	time cat twoel.trace | HalideTraceViz \
	  --size $(TRACE_VIDEO_WIDTH) $(TRACE_VIDEO_HEIGHT) --zoom 4 --timestep 15 --hold 100 --decay 5 5 --gray --auto_layout \
	| ffmpeg -loglevel warning -f rawvideo -pix_fmt bgr32 -s $(TRACE_VIDEO_WIDTH)x$(TRACE_VIDEO_HEIGHT) -i /dev/stdin -c:v h264 twoel.mp4

clean:
	rm -rf *.trace twoel.* twoel_g.* fake perf.* __pycache__
