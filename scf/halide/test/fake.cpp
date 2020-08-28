/* Standalone program that calls twoel() the same way SCF does, and measures how long it takes. */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <sys/times.h>
#include <unistd.h>

#include <algorithm>
#include <vector>
#include <sstream>

#include <HalideBuffer.h>

#include "twoel.h"

using namespace Halide::Runtime;

int N;

double gen0d() {
    return drand48();
}

Buffer<double> gen1d(int I=0) {
    if(I == 0)
        I = N;
    Buffer<double> rv(I);
    for(int i = 0; i < I; i++)
        rv(i) = drand48();
    return rv;
}

Buffer<double> gen2d(int I=0, int J=0) {
    if(I == 0)
        I = N;
    if(J == 0)
        J = N;
    Buffer<double> rv(I, J);
    for(int i = 0; i < I; i++)
        for(int j = 0; j < J; j++)
            rv(i, j) = drand48();
    return rv;
}

double timestamp() {
    double rv;
    struct timeval tv;
    gettimeofday(&tv, NULL);
    rv = tv.tv_usec;
    rv /= 1000000;
    rv += tv.tv_sec;
    return rv;
}

long cputickstamp() {
    struct tms tms;
    times(&tms);
    return tms.tms_utime;
}

int main(int argc, char **argv) {
    if(argc < 2) {
        fprintf(stderr, "Usage: %s <N>\n", argv[0]);
        return 1;
    }
    N = strtol(argv[1], NULL, 0);
    srand(2);
    srand48(rand());

    double delo2 = gen0d();
    double delta = gen0d();
    double rdelta = gen0d();
    Buffer<double> expnt      = gen1d();
    Buffer<double> rnorm      = gen1d();
    Buffer<double> x          = gen1d();
    Buffer<double> y          = gen1d();
    Buffer<double> z          = gen1d();
    Buffer<double> fm         = gen2d(1002, 5);
    Buffer<double> g_fock_in  = gen2d();
    Buffer<double> g_dens     = gen2d();
    Buffer<double> g_fock_out = gen2d();
    Buffer<double> rv         = gen1d();

    // dry run
    int error = twoel(delo2, delta, rdelta, expnt, rnorm, x, y, z, fm, g_fock_in, g_dens, rv, g_fock_out);
    if(error) {
        fprintf(stderr, "twoel failed with code %d\n", error);
        return 1;
    }

    // benchmark it
    std::vector<double> throughputs = {};
    for(int trial = 0; trial < 4; trial++) {
        double start_walltime  = timestamp();
        clock_t start_cputicks = cputickstamp();
        int itercount;
        for(itercount = 0; timestamp() - start_walltime < 5.0; itercount++) {
            twoel(delo2, delta, rdelta, expnt, rnorm, x, y, z, fm, g_fock_in, g_dens, rv, g_fock_out);
        }
        clock_t cputicks = cputickstamp() - start_cputicks;
        double walltime = timestamp() - start_walltime;
        double cputime = (double)cputicks / sysconf(_SC_CLK_TCK);
        double per_walltime = walltime / itercount;
        double per_cputime  = cputime  / itercount;
        double throughput = (double)1.0 * N * N * N * N / per_walltime;
        printf("%d iterations in %.3f seconds, %.3f seconds of cpu time, %.3e seconds per iter, %.3e cpu seconds per iter, %.3e effective iters per second\n", itercount, walltime, cputime, per_walltime, per_cputime, throughput);
        throughputs.push_back(throughput);
    }
    // sort and stringify the throughput values
    std::sort(throughputs.begin(), throughputs.end());
    std::ostringstream stringify;
    for(int i = 0; i < throughputs.size(); i++) {
        if(i)
            stringify << ", ";
        stringify << throughputs[i];
    }
    printf("throughputs: {%s}\n", stringify.str().c_str());

    return 0;
}
