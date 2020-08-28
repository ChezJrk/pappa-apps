#!/usr/bin/env python3

'''generate a twoel.a and twoel.h with only the specified zones, and with any scheduling params (thread count, vector/block sizes) passed through'''

import sys
sys.path.append('../tools')
import halide as hl
import twoel_gen

def gen_twoel(zone_name, **kwargs):

    # get JIT pipeline
    zone_names = zone_name.split(",")
    myzones = []
    for zone in zones.loops:
        if zone_name == 'all' or zone.name in zone_names:
            myzones.append(zone)
    if len(myzones) == 0:
        if zone_name == 'list':
            print([z['name'] for z in zones.loops])
        else:
            print("no zone %s found"%zone_name)
        exit(1)
    if "target_name" in kwargs:
        target_name = kwargs["target_name"]
        del kwargs["target_name"]
    else:
        target_name = "x86-64-linux-avx-avx2-f16c-fma-sse41-profile-disable_llvm_loop_opt"
    zones.loops = myzones
    gen = twoel_gen.Generate_twoel(loopnests=zones, **kwargs)
    gen.generate_twoel()
    p = gen.pipeline
    print("generating for target", target_name)
    target = hl.Target(target_name)
    p.compile_to(
        {
            hl.Output.c_header: "twoel.h",
            hl.Output.c_source: "twoel.cpp",
            hl.Output.static_library: "twoel.a",
            hl.Output.stmt: "twoel.stmt",
            hl.Output.stmt_html: "twoel.html",
            # the following outputs are useful for running it from python
            #hl.Output.object: "twoel.o",
            #hl.Output.python_extension: "twoel.py.cpp",
        }, list(gen.inputs.values()), "twoel", target
    )

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: %s <zonename>"%sys.argv[0])
        exit(1)

    zone_name = sys.argv[1]

    kwargs = {}
    for param in sys.argv[2:]:
        k, v = param.split("=")
        try:
            v = int(v)
        except:
            try:
                v = bool(v)
            except:
                pass
        kwargs[k] = v

    zones = twoel_gen.define_original_twoel_zone().split_recursive()

    gen_twoel(zone_name, **kwargs)
