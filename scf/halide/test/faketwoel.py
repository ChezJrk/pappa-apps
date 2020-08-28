#!/usr/bin/env python3

'''JIT-compile twoel with the specified zones and benchmark it, for quick feedback on scheduling changes'''

import sys
import time
sys.path.append('../tools')
import numpy as np
import halide as hl
import twoel_gen

def call_twoel(zone_name, seed=2, datasize=15, itercount=10, target_name="host-disable_llvm_loop_opt", **kwargs):
    N = datasize
    seed = 2

    inputs = [
        { "name": "delo2"  , "d": 0, "value": 0.001 },
        { "name": "delta"  , "d": 0, "value": 0.001 },
        { "name": "rdelta" , "d": 0, "value": 0.001 },
        { "name": "expnt"  , "d": 1, "value": 0.00001 },
        { "name": "rnorm"  , "d": 1 },
        { "name": "x"      , "d": 1 },
        { "name": "y"      , "d": 1 },
        { "name": "z"      , "d": 1 },
        { "name": "fm"     , "d": 2, "shape": [1002, 5] },
        { "name": "g_fock" , "d": 2 },
        { "name": "g_dens" , "d": 2 },
        { "name": "g_trace", "d": 4, "value": 0.0 },
    ]

    outputs = [
        { "name": "rv"    , "d": 1, "shape": [1] },
        { "name": "g_fock", "d": 2 },
    ]

    inputs  = { x["name"]: x for x in inputs }
    outputs = { x["name"]: x for x in outputs }

    # generate input data
    print("input/output size is", N, "^2")
    buffers = []
    buffers_by_name = {}
    np.random.seed(seed)
    for key in inputs:
        param = inputs[key]
        if param['d'] == 0:
            thing = 0.2
        else:
            shape = [N]*param['d']
            if 'shape' in param:
                shape = param['shape']
            thing = hl.Buffer(hl.Float(64), shape, name=key)
            if 'value' in param:
                if param['value'] != 0.0:
                    for pos in np.ndindex(*shape):
                        thing[pos] = param['value']
            else:
                values = np.random.rand(*shape) - 0.5
                for pos in np.ndindex(*shape):
                    thing[pos] = values[pos]
        buffers.append(thing)
        buffers_by_name[key] = thing

    # get JIT pipeline
    zones = twoel_gen.define_original_twoel_zone().split_recursive()
    zone_names = zone_name.split(",")
    myzones = []
    for zone in zones.loops:
        if zone_name == 'all' or zone['name'] in zone_names:
            myzones.append(zone)
    if len(myzones) == 0:
        if zone_name == 'list':
            print([z.name for z in zones])
        else:
            print("no zone %s found"%zone_name)
        exit(1)
    zones.loops = myzones
    gen = twoel_gen.Generate_twoel(loopnests=zones, **kwargs)
    gen.generate_twoel()
    p = gen.pipeline
    target = hl.Target(target_name)
    zone_names = [ z.name for z in myzones ]
    print("compiling zones", zone_names, "for target", target)
    p.compile_jit(target)
    # plug in the parameter values
    for param in gen.inputs.values():
        name = param.name()
        if name in buffers_by_name:
            thing = buffers_by_name[name]
        elif name.endswith("_in") and name[:-3] in buffers_by_name:
            name = name[:-3]
            thing = buffers_by_name[name]
        else:
            raise KeyError(name)
        param.set(thing)

    # dry-run
    p.realize(N, N)
    print(itercount, "timed runs")

    if itercount == 0:
        # when generating trace output, just doing the dry-run is enough.
        return 0.0, 0.0

    # benchmark it
    walltime = 0.0
    cputime = 0.0
    for _ in range(itercount):
        cpu_before  = time.process_time()
        wall_before = time.time()

        rv, g_fock_out = p.realize(N, N)

        cpu_after   = time.process_time()
        wall_after  = time.time()

        walltime += wall_after - wall_before
        cputime  += cpu_after  - cpu_before
    print("walltime: %.3f"%walltime)
    print("cputime: %.3f"%cputime)
    walltime /= itercount
    cputime  /= itercount
    print("walltime per iter: %.3e"%walltime)
    print("cputime per iter: %.3e"%cputime)
    throughput = N * N * N * N / walltime
    print("throughput: %.3e g() calls per second (roughly)"%throughput)
    rv = rv[0]
    g_fock_out = np.array(g_fock_out)
    return walltime, cputime


if __name__ == "__main__":

    zone_name = sys.argv[1]

    kwargs = {}
    for param in sys.argv[2:]:
        k,v = param.split("=")
        try:
            v = int(v)
        except:
            try:
                v = bool(v)
            except:
                pass
        kwargs[k] = v
    call_twoel(zone_name, **kwargs)
