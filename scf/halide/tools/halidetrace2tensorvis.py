#!/usr/bin/env python3

import sys

if len(sys.argv) < 2:
    raise Exception("Usage: %s <halide_trace_filename>"%sys.argv[0])

infn = sys.argv[1]
xlimit = None
ylimit = None
if len(sys.argv) > 2:
    xlimit = int(sys.argv[2])
if len(sys.argv) > 3:
    ylimit = int(sys.argv[3])

data_aliases = {
    "constant_exterior$8.0": "g",
}

def parse_tag(tag):
    # format: ? ? data_size_in_bits ? num_indices [min max]+
    #         1 2 64                1 4            0   8 0 8 0 8 0 8
    # format: len(func_types) [func_type_code data_size_in_bits func_type_lanes]+ num_indices [min max]+
    #         1                2              64                1                 4            0   8 0 8 0 8 0 8
    fields = tag.split(" ")
    tagname = fields[0]
    if tagname != "func_type_and_dim":
        return {"unknown": True}
    num_func_types = fields[1]
    offset = 1
    func_types = []
    for i in range(num_func_types):
        func_types.append({ "type_code": fields[offset], "data_size_bits": fields[offset+1], "type_lanes": fields[offset+2] })
        offset += 3
    n = int(fields[offset])
    offset += 1
    shape = []
    for i in range(n):
        shape.append({"min": int(fields[offset]), "max": int(fields[offset+1])})
        offset += 2
    return { "bits": bits, "types": func_types, "shape": shape }

with open(infn, "r") as f:
    for line in f:
        line = line.strip()
        type = line.split(" ")[0]
        line = line[len(type)+1:]
        if type == "Begin":
            print("begin", line)
        elif type == "End":
            print("end", line)
            break
        elif type == "Tag":
            func, tag = line.split(" tag = ")
            tag = tag[1:len(tag)-1]
            print("tag input:", tag)
            tag = parse_tag(tag)
            print("tag func", func, "tag", tag)
        elif type == "Load":
            print("load", line)
        elif type == "Store":
            print("store", line)
