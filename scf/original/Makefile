## Makefile for SCF-C

CC  = clang-10
CFLAGS = -O3 -mtune=native -march=native

OBJFILES = input.o integ.o output.o timer.o scf.o diagonalize.o

MTARGET = scf.x

all: $(MTARGET)

scf.x:  $(OBJFILES)
	$(CC) $(OBJFILES) $(CFLAGS) -o $(MTARGET) -lm

run: $(MTARGET)
	time ./$(MTARGET)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f *~ *.o $(MTARGET)
