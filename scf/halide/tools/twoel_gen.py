#!/usr/bin/env python3

import halide as hl
from math import pi

'''generate halide code to take advantage of the symmetry of g()'''

import logging

from decompose import Function, Condition, Symmetry, Update, LoopNest, DecomposedLoops

def define_original_twoel_zone():
    '''base definition of the twoel() function, to be decomposed/optimized'''
    g = Function("g", ["i","j","k","l"])
    loop = LoopNest(
        name="g_fock",
        symmetries=[ (g, "i", "j"), (g, "k", "l"), (g, ["i","j"], ["k","l"]) ],
        updates=[
            Update(Function("g_fock", ["i","j"]), [g, Function("g_dens", ["k","l"])]),
            Update(Function("g_fock", ["i","k"]), [g, Function("g_dens", ["j","l"])], coeff=-0.5),
        ]
    )
    return loop

class Generate_twoel():
    ''' SCF twoel() function generator '''
    def __init__(self, loopnests=None, tracing=False, tracing_g=False, tilesize=30, vectorsize=8):
        self.tracing = tracing
        self.tracing_g = tracing_g
        self.tilesize = tilesize
        self.vectorsize = vectorsize
        self.vars = {}
        self.funcs = {}
        self.inputs = {}
        self.clamps = {}
        self.outputs = {}
        self.loopnests = loopnests
        self.loopnest_funcs = {}

    def add_funcs_by_name(self, funcs):
        ''' add funcs to list of funcs (for bookkeeping) '''
        for func in funcs:
            self.funcs[func.name()] = func

    def generate_library(self):
        ''' generate a static library that implements twoel '''
        self.generate_twoel()
        self.compile()

    def generate_twoel(self):
        ''' generate a Halide pipeline in memory that implements twoel '''
        self.setup_vars()
        self.setup_inputs()
        self.gen_g()
        self.gen_main_loop()
        self.gen_outputs()
        self.schedule()
        self.gen_pipeline()

    def setup_vars(self):
        for c in "ijkl":
            v = hl.Var(c)
            self.vars[c] = v

    def setup_inputs(self):
        # input scalars
        delo2  = hl.Param(hl.Float(64), "delo2")
        delta  = hl.Param(hl.Float(64), "delta")
        rdelta = hl.Param(hl.Float(64), "rdelta")

        # input vectors
        expnt_in = hl.ImageParam(hl.Float(64), 1, "expnt_in")
        rnorm_in = hl.ImageParam(hl.Float(64), 1, "rnorm_in")
        x_in     = hl.ImageParam(hl.Float(64), 1, "x_in")
        y_in     = hl.ImageParam(hl.Float(64), 1, "y_in")
        z_in     = hl.ImageParam(hl.Float(64), 1, "z_in")

        # input matrices
        fm_in        = hl.ImageParam(hl.Float(64), 2, "fm_in")
        g_fock_in_in = hl.ImageParam(hl.Float(64), 2, "g_fock_in")
        g_dens_in    = hl.ImageParam(hl.Float(64), 2, "g_dens_in")

        self.inputs.update({ x.name(): x for x in [delo2, delta, rdelta, expnt_in, rnorm_in, x_in, y_in, z_in, fm_in, g_fock_in_in, g_dens_in] })

        # clamp all inputs, to prevent out-of-bounds errors from odd tile sizes and such
        expnt     = hl.BoundaryConditions.constant_exterior(expnt_in    , 0)
        rnorm     = hl.BoundaryConditions.constant_exterior(rnorm_in    , 0)
        x         = hl.BoundaryConditions.constant_exterior(x_in        , 0)
        y         = hl.BoundaryConditions.constant_exterior(y_in        , 0)
        z         = hl.BoundaryConditions.constant_exterior(z_in        , 0)
        fm        = hl.BoundaryConditions.constant_exterior(fm_in       , 0)
        g_fock_in = hl.BoundaryConditions.constant_exterior(g_fock_in_in, 0)
        g_dens    = hl.BoundaryConditions.constant_exterior(g_dens_in   , 0)

        self.clamps.update({ "expnt": expnt, "rnorm": rnorm, "x": x, "y": y, "z": z, "fm": fm, "g_fock_in_clamped": g_fock_in, "g_dens": g_dens })

        # nbfn=number of basis functions.  This is our problem size
        self.nbfn = g_fock_in_in.height()


    def gen_g(self):
        ''' define g() function '''
        # vars
        i, j, k, l = [self.vars[c] for c in "ijkl"]
        # clamped inputs
        x, y, z, expnt, fm, rnorm = [self.clamps[c] for c in ["x", "y", "z", "expnt", "fm", "rnorm"]]
        # unclamped input (for sizing)
        fm_in = self.inputs["fm_in"]
        # scalar inputs
        delo2, delta, rdelta = [self.inputs[c] for c in ["delo2", "delta", "rdelta"]]

        dx = hl.Func("dx")
        dy = hl.Func("dy")
        dz = hl.Func("dz")
        r2 = hl.Func("g_r2")
        expnt2 = hl.Func("expnt2")
        expnt_inv = hl.Func("expnt_inv")
        self.add_funcs_by_name([dx, dy, dz, r2, expnt2, expnt_inv])


        dx[i,j] = x[i] - x[j]
        dy[i,j] = y[i] - y[j]
        dz[i,j] = z[i] - z[j]

        r2[i,j] = dx[i,j] * dx[i,j] + dy[i,j] * dy[i,j] + dz[i,j] * dz[i,j]

        expnt2[i,j]     = expnt[i] + expnt[j]
        expnt_inv[i,j] = hl.f64(1.0) / expnt2[i,j]


        fac2   = hl.Func("fac2")
        ex_arg = hl.Func("ex_arg")
        ex     = hl.Func("ex")
        denom  = hl.Func("denom")
        fac4d  = hl.Func("fac4d")
        self.add_funcs_by_name([fac2, ex_arg, ex, denom, fac4d])
        fac2[i,j] = expnt[i] * expnt[j] * expnt_inv[i,j]
        ex_arg[i,j,k,l] = -fac2[i,j] * r2[i,j] - fac2[k,l] * r2[k,l]
        ex[i,j,k,l] = hl.select(ex_arg[i,j,k,l] < hl.f64(-37.0), hl.f64(0.0), hl.exp(ex_arg[i,j,k,l]))
        denom[i,j,k,l]  = expnt2[i,j] * expnt2[k,l] * hl.sqrt(expnt2[i,j] + expnt2[k,l])
        fac4d[i,j,k,l]  = expnt2[i,j] * expnt2[k,l] /        (expnt2[i,j] + expnt2[k,l])


        x2   = hl.Func("g_x2")
        y2   = hl.Func("g_y2")
        z2   = hl.Func("g_z2")
        rpq2 = hl.Func("rpq2")
        self.add_funcs_by_name([x2, y2, z2, rpq2])
        x2[i,j] = (x[i] * expnt[i] + x[j] * expnt[j]) * expnt_inv[i,j]
        y2[i,j] = (y[i] * expnt[i] + y[j] * expnt[j]) * expnt_inv[i,j]
        z2[i,j] = (z[i] * expnt[i] + z[j] * expnt[j]) * expnt_inv[i,j]
        rpq2[i,j,k,l] = (
              (x2[i,j] - x2[k,l]) * (x2[i,j] - x2[k,l])
            + (y2[i,j] - y2[k,l]) * (y2[i,j] - y2[k,l])
            + (z2[i,j] - z2[k,l]) * (z2[i,j] - z2[k,l]))



        f0t   = hl.Func("f0t")
        f0n   = hl.Func("f0n")
        f0x   = hl.Func("f0x")
        f0val = hl.Func("f0val")
        self.add_funcs_by_name([f0t, f0n, f0x, f0val])
        f0t[i,j,k,l] = fac4d[i,j,k,l] * rpq2[i,j,k,l]
        f0n[i,j,k,l] = hl.clamp(hl.i32((f0t[i,j,k,l] + delo2) * rdelta), fm_in.dim(0).min(), fm_in.dim(0).max())
        f0x[i,j,k,l] = delta * f0n[i,j,k,l] - f0t[i,j,k,l]
        f0val[i,j,k,l] = hl.select(f0t[i,j,k,l] >= hl.f64(28.0),
             hl.f64(0.88622692545276) / hl.sqrt(f0t[i,j,k,l]),
                                               fm[f0n[i,j,k,l],0]
             + f0x[i,j,k,l] *                 (fm[f0n[i,j,k,l],1]
             + f0x[i,j,k,l] * hl.f64(0.5) *   (fm[f0n[i,j,k,l],2]
             + f0x[i,j,k,l] * hl.f64(1./3.) * (fm[f0n[i,j,k,l],3]
             + f0x[i,j,k,l] * hl.f64(0.25) *   fm[f0n[i,j,k,l],4]))))

        g = hl.Func("g")
        self.add_funcs_by_name([g])

        if self.tracing and self.tracing_g:
            g_trace_in = hl.ImageParam(hl.Float(64), 4, "g_trace_in")
            g_trace    = hl.BoundaryConditions.constant_exterior(g_trace_in, 0)
            self.inputs["g_trace_in"] = g_trace_in
            self.clamps["g_trace"] = g_trace
            g_trace.compute_root()
            g[i,j,k,l] = (hl.f64(2.00) * hl.f64(pow(pi, 2.50)) / denom[i,j,k,l]) * ex[i,j,k,l] * f0val[i,j,k,l] * rnorm[i] * rnorm[j] * rnorm[k] * rnorm[l] + g_trace[i,j,k,l]
        else:
            g_trace = None
            g[i,j,k,l] = (hl.f64(2.00) * hl.f64(pow(pi, 2.50)) / denom[i,j,k,l]) * ex[i,j,k,l] * f0val[i,j,k,l] * rnorm[i] * rnorm[j] * rnorm[k] * rnorm[l]


    def gen_main_loop(self):
        ''' define the big tensor contraction (with symmetry decomposition) '''
        i, j = [self.vars[c] for c in "ij"]
        g_fock_components = [ self.clamps["g_fock_in_clamped"] ]
        zones = self.loopnests
        nbfn = self.nbfn
        loop_extents = [nbfn, nbfn, nbfn, nbfn]
        if zones is None:
            original_twoel = define_original_twoel_zone()
            zones = original_twoel.split_recursive()
            # uncomment the next line to disable symmetry decomposition
            #zones = DecomposedLoops(original_twoel, [original_twoel])
        zones.generate_halide(self, loop_extents, g_fock_components)


    def gen_outputs(self):
        ''' define the outputs '''
        nbfn = self.nbfn
        i, j = [self.vars[c] for c in "ij"]
        g_fock = self.funcs["g_fock"]
        g_dens = self.clamps["g_dens"]
        # output scalars
        rv = hl.Func("rv")

        # output matrix
        g_fock_out = hl.Func("g_fock_out")
        self.funcs  .update({"rv": rv, "g_fock_out": g_fock_out})
        self.outputs.update({"rv": rv, "g_fock_out": g_fock_out})

        g_fock_out[i, j] = g_fock[i,j]

        rv[i] = hl.f64(0.0)
        r_rv = hl.RDom([(0, nbfn), (0, nbfn)])
        rv[0] += g_fock[r_rv] * g_dens[r_rv]
        rv[0] *= hl.f64(0.5)


    def schedule(self):
        ''' apply a CPU schedule '''
        # scheduling
        i, j, k, l = [self.vars[c] for c in "ijkl"]
        vectorsize = self.vectorsize
        tilesize = self.tilesize
        logging.info("scheduling")
        if_, jf, kf, lf = [ hl.Var(c+"f") for c in "ijkl" ] # rfactor vars
        io, jo, ko, lo = [ hl.Var(c+"o") for c in "ijkl" ] # block outer variables
        outer_vars = [ io, jo, ko, lo ]
        ii, ji, ki, li = [ hl.Var(c+"i") for c in "ijkl" ] # block inner variables
        inner_vars = [ ii, ji, ki, li ]
        gio, gjo, gko, glo = [ hl.RVar("g"+c+"o") for c in "ijkl" ] # block outer reduction variables
        outer_rvars = [ gio, gjo, gko, glo ]
        gii, gji, gki, gli = [ hl.RVar("g"+c+"i") for c in "ijkl" ] # block inner reduction variables
        inner_rvars = [ gii, gji, gki, gli ]
        jii = hl.Var("jii") # fused i + ji
        kolo = hl.Var("kolo") # fused ko + lo
        gkolo = hl.RVar("gkolo") # fused gko + glo
        ic, jc = hl._0, hl._1 # indexes for clamp funcs

        # schedule the pieces of g

        for clamped_1d_input in [ "expnt", "rnorm", "x", "y", "z", "fm", "g_fock_in_clamped", "g_dens" ]:
            clamped_1d_input = self.clamps[clamped_1d_input]
            clamped_1d_input.compute_root().vectorize(ic, vectorsize)
        for g_precomputed_matrix in [ "expnt2", "fac2", "g_r2", "g_x2", "g_y2", "g_z2" ]:
            g_precomputed_matrix = self.funcs[g_precomputed_matrix]
            g_precomputed_matrix.compute_root().vectorize(i, vectorsize)

        self.funcs["ex_arg"].compute_inline()
        self.funcs["expnt_inv"].compute_inline()
        for generic_4d_thing in ["denom", "ex", "fac4d", "rpq2"]:
            generic_4d_thing = self.funcs[generic_4d_thing]
            generic_4d_thing.compute_inline()

        for zone_name, zone_record in self.loopnest_funcs.items():
            logging.info("scheduling zone %s", zone_name)
            func  = zone_record['func']
            ur    = zone_record['unroll']
            iters = zone_record['iters']
            gi    = iters['i']
            r     = zone_record['rdom']
            riter = [r[i] for i in range(1, len(r))] # skip r[0], the unroll factor
            rinner = riter[0]
            router = riter[-1]
            func.compute_root().parallel(j).vectorize(i, vectorsize)
            func.update().dump_argument_list()
            if len(riter) == 4:
                ir, jr, kr, lr = riter
                func.update().reorder(ur, ir, kr, lr, jr).unroll(ur)
                func_intm = func.update().rfactor([[jr, jf], [ir, if_]])
                func_intm.compute_root().update().reorder(ur, if_, kr, lr, jf).unroll(ur).vectorize(if_, vectorsize).parallel(jf)
                self.funcs[func_intm.name()] = func_intm
                func.update().atomic().vectorize(ir, vectorsize)
                #g.in_(func_intm).reorder(i, k, l, j).compute_at(func_intm, if_).store_at(func_intm, lr).vectorize(i, vectorsize)
            else:
                func.update().atomic().unroll(ur).vectorize(rinner, vectorsize).parallel(router, 8)
            if logging.root.level < logging.INFO:
                logging.debug("function %s loop nest:", zone_name)
                func.print_loop_nest()

        self.funcs['g_fock'].compute_root()

        # tracing
        if self.tracing:
            for func_name in self.funcs:
                if func_name != "g":
                    self.funcs[func_name].trace_stores()
                self.funcs[func_name].trace_loads()


    def gen_pipeline(self):
        '''define the Halide pipeline that generates the outputs'''

        logging.info("generating halide pipeline")

        rv, g_fock_out = [self.outputs[f] for f in ["rv", "g_fock_out"]]

        # return the pipeline
        self.pipeline = hl.Pipeline([rv, g_fock_out])
        return self.pipeline, self.outputs, self.inputs


    def compile(self):
        self.pipeline.compile_to(
            {
                hl.Output.c_header: "twoel.h",
                hl.Output.c_source: "twoel.cpp",
                hl.Output.static_library: "twoel.a",
                hl.Output.stmt: "twoel.stmt",
                #hl.Output.stmt_html: "twoel.html",
                # the following outputs are useful for running it from python
                #hl.Output.object: "twoel.o",
                #hl.Output.python_extension: "twoel.py.cpp",
            }, list(self.inputs.values()), "twoel"
        )



if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    gen = Generate_twoel()
    gen.generate_library()
    logging.debug({"outputs": gen.outputs})
    logging.debug({"inputs": gen.inputs})
