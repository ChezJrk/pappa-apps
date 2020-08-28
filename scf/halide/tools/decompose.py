#!/usr/bin/env python3

import logging
import halide as hl

def die(s):
    '''report a fatal error'''
    logging.critical(s)
    raise Exception(s)

class Function:
    '''a specific tensor access in the RHS update, like g(i,j,k,l)'''
    def __init__(self, name, indexes):
        if isinstance(indexes, set):
            indexes = [*indexes]
        if not isinstance(indexes, list):
            die("function indexes should be a list (or set)")
        for index in indexes:
            if not isinstance(index, str):
                die("function indexes should be a list of index names")
        self.name = name
        self.indexes = indexes

    def update_indexes(self, updates):
        '''apply an index-permutation from a symmetry decomposition'''
        for i in range(len(self.indexes)):
            if self.indexes[i] in updates:
                self.indexes[i] = updates[self.indexes[i]]

    def transpose(self, lhs, rhs):
        '''swap the two sets of indexes'''
        updates = {}
        for l, r in zip(lhs, rhs):
            updates[l.name] = r
            updates[r.name] = l
        self.update_indexes(updates)

    def copy(self):
        return Function(self.name, [*self.indexes])

    def __str__(self):
        indexes = [ str(i) for i in self.indexes ]
        indexes = ",".join(indexes)
        return self.name + "(" + indexes + ")"

    def __repr__(self):
        things = [self.name, self.indexes]
        return "Function(" + ", ".join([repr(x) for x in things]) + ")"


class Symmetry:
    '''a symmetric equality statement, like f(i,j) == f(j,i)'''
    def __init__(self, func, lhs, rhs):
        if not isinstance(func, Function):
            die("symmetry function should be a Function")
        if isinstance(lhs, int):
            lhs = [lhs]
        if isinstance(rhs, int):
            rhs = [rhs]
        if not isinstance(lhs, list) or not isinstance(rhs, list):
            die("symmetry lhs and rhs must be lists")
        seen_index_positions = set()
        for index in lhs:
            if not isinstance(index, int):
                die("symmetry lhs must be a list of ints")
            if index < 0 or index >= len(func.indexes):
                die("symmetry lhs is out of bounds")
            if index in seen_index_positions:
                die("index " + index + " mentioned twice in lhs")
            seen_index_positions.add(index)
        for index in rhs:
            if not isinstance(index, int):
                die("symmetry rhs must be a list of ints")
            if index < 0 or index >= len(func.indexes):
                die("symmetry rhs is out of bounds")
            if index in seen_index_positions:
                die("index " + index + " in both lhs and rhs")
        if len(lhs) != len(rhs):
            die("symmetry lhs and rhs must be of same length")
        self.func = func
        self.lhs = lhs
        self.rhs = rhs
        self.totals = set(lhs + rhs)

    def apply(self, func):
        '''swap indexes in the given func'''
        func = func.copy()
        for l, r in zip(self.lhs, self.rhs):
            func.indexes[l], func.indexes[r] = func.indexes[r], func.indexes[l]
        return func

    def copy(self):
        return Symmetry(self.func.copy(), self.lhs.copy(), self.rhs.copy())

    def __str__(self):
        permuted = self.apply(self.func)
        return "<" + str(self.func) + " = " + str(permuted) + ">"

    def __repr__(self):
        things = [self.func, self.lhs, self.rhs]
        return "Symmetry(" + ", ".join([repr(x) for x in things]) + ")"


class Condition:
    '''a boolean expression that constricts the iteration space, like i < j'''
    def __init__(self, lhs, rhs, op="!="):
        if not isinstance(lhs, list):
            lhs = [lhs]
        if not isinstance(rhs, list):
            rhs = [rhs]
        for l in lhs:
            if not isinstance(l, str):
                die("lhs must contain index names")
        for r in rhs:
            if not isinstance(r, str):
                die("rhs must contain index names")
        if len(lhs) != len(rhs):
            die("condition lhs and rhs must be equally sized")
        if op not in [ '==', '!=', '<', '<=', '>', '>=' ]:
            die("simple condition op is not recognized")
        self.lhs = lhs
        self.rhs = rhs
        self.op  = op

    def canonicalize(self):
        '''organize the condition in a standard form'''
        reverses = {
            ">": "<",
            ">=": "<=",
        }
        associatives = set(["==", "!="])
        if self.op in reverses:
            self.op, self.lhs, self.rhs = reverses[self.op], self.rhs, self.lhs
        if self.op in associatives:
            lhs_name = "".join(self.lhs)
            rhs_name = "".join(self.rhs)
            if lhs_name > rhs_name:
                self.lhs, self.rhs = self.rhs, self.lhs

    def simplify(self):
        '''reduce complexity, possibly resulting in multiple (simpler) Conditions'''
        rv = []
        if self.op == "==":
            # get rid of tautologies (i==i)
            new_lhs = []
            new_rhs = []
            for l, r in zip(self.lhs, self.rhs):
                if str(l) != str(r):
                    new_lhs.append(l)
                    new_rhs.append(r)
            if len(new_lhs) > 0:
                self.lhs = new_lhs
                self.rhs = new_rhs
                # sort the operands: j==i becomes i==j
                if self.lhs[0] > self.rhs[0]:
                    self.rhs, self.lhs = self.lhs, self.rhs
                rv.append(self)
        elif self.op == "!=":
            # sort the operands: j!=i becomes i!=j
            if self.lhs[0].name > self.rhs[0].name:
                self.rhs, self.lhs = self.lhs, self.rhs
            rv.append(self)
        else:
            rv.append(self)
        return rv

    def update_indexes(self, updates):
        '''apply an index-permutation from a symmetry decomposition'''
        new_lhs = []
        new_rhs = []
        for l, r in zip(self.lhs, self.rhs):
            if l in updates:
                l = updates[l]
            if r in updates:
                r = updates[r]
            new_lhs.append(l)
            new_rhs.append(r)
        self.lhs = new_lhs
        self.rhs = new_rhs

    def transpose(self, lhs, rhs):
        '''swap the two sets of indexes'''
        self.update_indexes({ lhs.name: rhs, rhs.name: lhs })

    def generate(self, indexes, N=None):
        '''apply condition logic to produce a boolean expression'''
        lhs = indexes[self.lhs[0]]
        rhs = indexes[self.rhs[0]]
        if len(self.lhs) > 1:
            if N is None:
                die("can't generate tuple condition without a size expression")
            for l in self.lhs[1:]:
                lhs *= N
                lhs += indexes[l]
            for r in self.rhs[1:]:
                rhs *= N
                rhs += indexes[r]
        if self.op == "<":
            return lhs < rhs
        elif self.op == "<=":
            return lhs <= rhs
        elif self.op == ">":
            return lhs > rhs
        elif self.op == ">=":
            return lhs >= rhs
        elif self.op == "==":
            return lhs == rhs
        elif self.op == "!=":
            return lhs != rhs

    def symbol_name(self):
        '''generate a string for this condition which can become part of a C symbol name'''
        # stringify in a way that is valid as a C label
        # assumes the index names are valid
        text = {
            "<": "lt",
            ">": "gt",
            "<=": "le",
            ">=": "ge",
            "==": "eq",
            "!=": "ne",
        }
        lhs = "".join(self.lhs)
        rhs = "".join(self.rhs)
        return "_".join([lhs, text[self.op], rhs])

    def copy(self):
        return Condition(self.lhs.copy(), self.rhs.copy(), self.op)

    def __str__(self):
        return "(" + "".join([str(x) for x in self.lhs]) + " " + self.op + " " + "".join([str(x) for x in self.rhs]) + ")"

    def __repr__(self):
        things = [self.lhs, self.rhs, self.op]
        return "Condition(" + ", ".join([repr(x) for x in things]) + ")"


class Update:
    '''a specific statement in the inner-most loop, like f(i,k) += g(i,j,k,l) * d(j,l) * -0.5'''
    # self.out      = output tensor
    # self.elements = array of input tensors
    # self.op       = reduction operator
    #                 ("*" means the input elements are multiplied)
    # self.coeff    = scalar coefficient
    def __init__(self, out, elements, op="+", coeff=1.0):
        if not isinstance(out, Function):
            die("update out should be a Function")
        if isinstance(elements, Function):
            elements = [ elements ]
        if not isinstance(elements, list):
            die("elements should be a list of functions")
        for element in elements:
            if not isinstance(element, Function):
                die("elements should be a list of Function")
        if op not in "+-/*^":
            die("unknown op")
        self.out = out
        self.elements = elements
        self.op = op
        self.coeff = coeff

    def indexes(self):
        '''determine the set of indexes referenced by this update'''
        rv = set()
        for index in self.out.indexes:
            rv.add(index)
        for element in self.elements:
            for index in element.indexes:
                rv.add(index)
        # sort keys by name
        return sorted(list(rv))

    def update_indexes(self, updates):
        '''apply an index-permutation from a symmetry decomposition'''
        self.out.update_indexes(updates)
        for element in self.elements:
            element.update_indexes(updates)

    def transpose(self, lhs, rhs):
        '''swap the two sets of indexes'''
        self.update_indexes({ lhs.name: rhs, rhs.name: lhs })

    def copy(self):
        return Update(self.out.copy(), [e.copy() for e in self.elements], self.op, self.coeff)

    def __str__(self):
        pieces = self.elements
        pieces = [ str(p) for p in pieces ]
        if self.coeff != 1.0:
            pieces.append(str(self.coeff))
        pieces = " * ".join(pieces)
        return str(self.out) + " " + self.op + "= " + pieces + ";"

    def __repr__(self):
        things = [self.out, self.elements]
        things = [repr(x) for x in things]
        if self.op != "+":
            things.append("op=" + repr(self.op))
        if self.coeff != 1.0:
            things.append("coeff=" + repr(self.coeff))
        return "Update(" + ", ".join(things) + ")"



class Testbed():
    '''uses symbolic execution to determine a loopnest's execution pattern.  used for correctness testing'''
    def __init__(self, zones, symmetries=None, N=8):
        if symmetries is None:
            symmetries = []
        symmetries_by_name = {}
        for symmetry in symmetries:
            name = symmetry.func.name
            if name not in symmetries_by_name:
                symmetries_by_name[name] = []
            symmetries_by_name[name].append(symmetry)
        symmetries = symmetries_by_name
        self.N = N
        self.zones = zones
        self.symmetries = symmetries

    def funcstr(self, func, index_values):
        '''generate string represented a permuted function call'''
        values = [index_values[i] for i in func.indexes]
        if func.name in self.symmetries:
            for symmetry in self.symmetries[func.name]:
                lhs = [values[x] for x in symmetry.lhs]
                rhs = [values[x] for x in symmetry.rhs]
                l = lhs[0]
                r = rhs[0]
                for la in lhs[1:]:
                    l *= self.N
                    l += la
                for ra in rhs[1:]:
                    r *= self.N
                    r += ra
                if l > r:
                    for li, ri in zip(symmetry.lhs, symmetry.rhs):
                        values[li], values[ri] = values[ri], values[li]
        rv = ",".join([str(x) for x in values])
        rv = func.name + "(" + rv + ")"
        return rv

    def run(self):
        '''do the symbolic execution'''
        from itertools import product
        outputs = {}
        for zone in self.zones:
            indexes = zone.indexes()
            # enumerate indexes
            position_space = [range(self.N) for x in indexes]
            for position in product(*position_space):
                # apply conditions
                index_values = { i:v for i,v in zip(indexes, position) }
                for a, b in zone.equalities.items():
                    index_values[a] = index_values[b]
                skip = False
                for condition in zone.conditions:
                    if condition.generate(index_values, self.N) is False:
                        skip = True
                        break
                if skip:
                    continue

                # apply updates
                for update in zone.updates:
                    output = self.funcstr(update.out, index_values)
                    if output not in outputs:
                        outputs[output] = {}
                    elements = update.op.join([self.funcstr(x,index_values) for x in update.elements])
                    if elements not in outputs[output]:
                        outputs[output][elements] = 0.0
                    outputs[output][elements] += update.coeff
        return outputs


class LoopNest:
    '''a nested loop structure'''
    # self.name       = string
    # self.equalities = dictionary of past indexes which have been eliminated/lowered
    # self.conditions = list of Condition objects
    #                   represents the subset of the enumeration space that this loop nest should cover
    # self.symmetries = list of Symmetry objects
    #                   represents the set of symmetries that have yet to be decomposed
    # self.updates    = list of Update objects
    #                   represents the set of work that should be done in the inner-most loop
    def __init__(self, name, conditions=None, symmetries=None, updates=None, equalities=None):
        if not isinstance(name, str):
            die("loop name should be a string")
        if len(name) == 0:
            die("loop name should be non-empty")
        if conditions is None:
            conditions = []
        if equalities is None:
            equalities = {}
        if not isinstance(conditions, list):
            die("loop conditions should be a list of conditions")
        for condition in conditions:
            if not isinstance(condition, Condition):
                die("loop conditions should be a list of Condition")
        if symmetries is None:
            symmetries = []
        if not isinstance(symmetries, list):
            die("loop symmetries should be a list of symmetries")
        for i in range(len(symmetries)):
            symmetry = symmetries[i]
            if isinstance(symmetry, tuple):
                func, lhs, rhs = symmetry
                string_mapping = {}
                for j in range(len(func.indexes)):
                    string_mapping[func.indexes[j]] = j
                if isinstance(lhs, str) or isinstance(lhs, int):
                    lhs = [lhs]
                if isinstance(rhs, str) or isinstance(rhs, int):
                    rhs = [rhs]
                if len(lhs) != len(rhs):
                    die("Symmetry lhs and rhs must be equal length")
                for j in range(len(lhs)):
                    if lhs[j] in string_mapping:
                        lhs[j] = string_mapping[lhs[j]]
                    if rhs[j] in string_mapping:
                        rhs[j] = string_mapping[rhs[j]]
                symmetry = Symmetry(func, lhs, rhs)
                symmetries[i] = symmetry
            elif not isinstance(symmetry, Symmetry):
                die("loop symmetries should be a list of Symmetry")
        if not isinstance(updates, list):
            die("loop updates should be a list of updates")
        for update in updates:
            if not isinstance(update, Update):
                die("loop updates should be a list of Update")
        self.name = name
        self.conditions = conditions
        self.symmetries = symmetries
        self.updates = updates
        self.equalities = equalities

    def indexes(self):
        '''determine the set of indexes referenced by this update'''
        found = set()
        for update in self.updates:
            for index in update.indexes():
                found.add(index)
        return sorted(list(found))

    def split(self, orig_by):
        '''split a nested loop up across a symmetric axis'''
        # apply symmetry transformation
        # this optimization benefits applications where elements can be reused across the updates.
        # given a loop nest, and an assertion that two (sets of) indexes are equivalent/swappable, take advantage of the symmetry to do less work.
        # specifically, break the loop nest apart into 3 pieces:
        #   upper: l < r
        #   equal: l = r
        #   lower: l > r
        # then use index-swapping to rewrite "lower" to look like "upper", and merge them together.
        # the result is "upper" and "equal".
        # "upper" result is a loop-nest with half of the iteration space, but twice the updates.
        # "equal" result is a loop-nest of complexity N-1, covering the leftovers (on the diagonal).

        by = orig_by
        if isinstance(by, int):
            by = self.symmetries[by]
        func = by.func
        indexes = func.indexes.copy()

        lhs_positions = by.lhs
        rhs_positions = by.rhs
        lhs_indexes = [indexes[x] for x in lhs_positions]
        rhs_indexes = [indexes[x] for x in rhs_positions]
        lhs_name = "".join(lhs_indexes)
        rhs_name = "".join(rhs_indexes)
        transpose_indexes = {}
        for l, r in zip(lhs_indexes, rhs_indexes):
            transpose_indexes[l] = r
            transpose_indexes[r] = l
        upper_condition = Condition(lhs_indexes, rhs_indexes, "<")
        equal_conditions = [ Condition(l, r, "==") for l, r in zip(lhs_indexes, rhs_indexes) ]
        lower_condition = Condition(lhs_indexes, rhs_indexes, ">")

        logging.debug("loop.split: lhs is %s", lhs_name)
        logging.debug("loop.split: rhs is %s", rhs_name)
        logging.debug("transpose_indexes is %s", transpose_indexes)

        upper = self.copy()
        equal = self.copy()
        lower = self.copy()
        upper.conditions.append(upper_condition)
        equal.conditions += equal_conditions
        lower.conditions.append(lower_condition)
        upper.name += "_" + lhs_name + "_lt_" + rhs_name
        equal.name += "_" + lhs_name + "_eq_" + rhs_name
        lower.name += "_" + lhs_name + "_gt_" + rhs_name
        lower.update_indexes(transpose_indexes)
        upper.merge_symmetric(lower, by)
        upper.symmetries = [s.copy() for s in self.symmetries if s != by]
        equal.symmetries = [s.copy() for s in self.symmetries if s != by]
        for l, r in zip(lhs_indexes, rhs_indexes):
            equal.equalities[r] = l
        equal.simplify()
        upper.simplify()

        # test whether the results are correct
        # (poor man's test for logical contradictions and stupid bugs)
        testbed_orig  = Testbed([self], self.symmetries)
        testbed_split = Testbed([equal, upper], self.symmetries)
        updates_orig  = testbed_orig.run()
        updates_split = testbed_split.run()
        if updates_orig != updates_split:
            # output LoopNests are not equivalent to input LoopNest
            # report the differences to debug log
            import difflib
            import json
            data_orig  = json.dumps(updates_orig , indent=2, sort_keys=True).split("\n")
            data_split = json.dumps(updates_split, indent=2, sort_keys=True).split("\n")
            logging.debug("difference in computation coverage:")
            diff = difflib.unified_diff(data_orig, data_split, fromfile=self.name, tofile="split zones")
            for line in diff:
                logging.debug(line)

            logging.info("splitting %s by %s failed; leaving it alone", self.name, by)
            self.symmetries = [x for x in self.symmetries if x != by]
            return [self]

        return [upper, equal]

    def update_indexes(self, updates):
        '''apply an index-permutation from a symmetry decomposition'''
        for i in range(len(self.conditions)):
            self.conditions[i].update_indexes(updates)
        for i in range(len(self.updates)):
            self.updates[i].update_indexes(updates)

    def merge_symmetric(self, other, along):
        '''concatenate updates, but preserve the index-order of the func that was split'''
        if len(self.updates) != len(other.updates):
            die("merge_symmetric can only be called on loop nests that are the mirrror image of each other.")
        # find the common element corresponding to the symmetry
        along_element = along.func
        symmetric_element  = None
        transposed_element = None
        for self_update, other_update in zip(self.updates, other.updates):
            if symmetric_element is not None:
                break
            for self_element, other_element in zip(self_update.elements, other_update.elements):
                logging.debug("comparing update element %s to symmetry element %s", self_element, along_element)
                if str(self_element.name) == str(along_element.name):
                    symmetric_element  = self_element
                    transposed_element = other_element
                    break
        else:
            die("symmetry func not found in update elements")

        for self_update, other_update in zip(self.updates, other.updates):
            other_elements = []
            for element in other_update.elements:
                if str(element) == str(transposed_element):
                    element = symmetric_element.copy()
                other_elements.append(element)
            new_update = other_update.copy()
            new_update.elements = other_elements
            self.updates.append(new_update)

    def simplify(self):
        '''consolidatate and simplify everything we can'''
        before = ""
        while str(self) != before:
            self.simplify_conditions()
            self.simplify_equalities()
            self.simplify_updates()
            self.simplify_symmetries()
            before = str(self)

    def simplify_equalities(self):
        '''if i == j, then go and replace j with i in all the symmetries and updates'''
        simplifications = {}
        for condition in self.conditions:
            condition.simplify()
            if condition.op == "==":
                for l, r in zip(condition.lhs, condition.rhs):
                    simplifications[r] = l

        # reduce update indexes
        for update in self.updates:
            update.out.update_indexes(simplifications)
            for element in update.elements:
                element.update_indexes(simplifications)

    def simplify_symmetries(self):
        '''eliminate redundant symmetries, and symmetries that don't apply to any updates'''

        # remove redundant symmetries (symmetries which stringify to the same string)
        new_symmetries = []
        symmetry_strings = {}
        for symmetry in self.symmetries:
            if str(symmetry) not in symmetry_strings:
                symmetry_strings[str(symmetry)] = 1
                new_symmetries.append(symmetry)
        self.symmetries = new_symmetries

        # figure out which symmetries apply to updates
        symmetry_func_names = {}
        update_funcs_by_name = {}
        for symmetry in self.symmetries:
            symmetry_func_names[symmetry.func.name] = symmetry.func
        for update in self.updates:
            for func in update.elements:
                if func.name not in symmetry_func_names:
                    continue
                if func.name not in update_funcs_by_name:
                    update_funcs_by_name[func.name] = [func]
                else:
                    if str(func) != str(update_funcs_by_name[func.name][0]):
                        logging.error(update_funcs_by_name[func.name][0])
                        logging.error(func)
                        die("inconsistent calls to " + func.name + " between updates")
                update_funcs_by_name[func.name].append(func)

        # remove the ones that don't
        new_symmetries = []
        for symmetry in self.symmetries:
            logging.debug("considering symmetry for elimination: %s", symmetry)
            logging.debug("against update funcs: %s", update_funcs_by_name[symmetry.func.name])
            update_indexes = {}
            for func in update_funcs_by_name[symmetry.func.name]:
                for index in func.indexes:
                    update_indexes[index] = index
            func = symmetry.func
            symmetry_affects_updates = False
            for l, r in zip(symmetry.lhs, symmetry.rhs):
                l, r = func.indexes[l], func.indexes[r]
                if l in update_indexes or r in update_indexes:
                    symmetry_affects_updates = True
                    break
            if symmetry_affects_updates:
                new_symmetries.append(symmetry)
        self.symmetries = new_symmetries

    def simplify_conditions(self):
        '''canonicalize/simplify conditions, remove redundant ones'''

        # canonicalize all conditions, remove redundant ones
        new_conditions = []
        for condition in self.conditions:
            condition.canonicalize()
            for new_condition in condition.simplify():
                new_condition.canonicalize()
                new_conditions.append(new_condition)
        self.conditions = new_conditions

        # apply equalities
        equalities = {}
        for condition in self.conditions:
            if condition.op == '==':
                for l, r in zip(condition.lhs, condition.rhs):
                    equalities[r] = l

        # turn tuple conditions into simple conditions
        for condition in self.conditions:
            if len(condition.rhs) == 1:
                continue
            redundant = set()
            lhs_seen = set()
            rhs_seen = set()
            for i in range(len(condition.lhs)):
                l = condition.lhs[i]
                r = condition.rhs[i]
                if l in equalities:
                    l = equalities[l]
                if r in equalities:
                    r = equalities[r]
                if l in lhs_seen and r in rhs_seen:
                    redundant.add(i)
                lhs_seen.add(l)
                rhs_seen.add(r)
            new_lhs = []
            new_rhs = []
            for i in range(len(condition.lhs)):
                if i not in redundant:
                    new_lhs.append(condition.lhs[i])
                    new_rhs.append(condition.rhs[i])
            condition.lhs = new_lhs
            condition.rhs = new_rhs

    def simplify_updates(self):
        '''merge updates which share the same pattern by adding coefficients together'''
        coefficients = {}
        tuples = {}
        for update in self.updates:
            coeff = update.coeff
            key  = [ str(update.out) ]
            key += [ str(e) for e in update.elements]
            key = "/".join(key)
            if key not in coefficients:
                coefficients[key] = 0.0
            coefficients[key] += coeff
            tuples[key] = (update.out, update.elements)
        new_updates = []
        for key in sorted(coefficients.keys()):
            coeff = coefficients[key]
            out, elements = tuples[key]
            update = Update(out=out, elements=elements, coeff=coeff)
            new_updates.append(update)
        self.updates = new_updates

    def split_recursive(self):
        '''split all symmetries recursively'''
        rvs = [self]
        making_progress = True
        while making_progress:
            making_progress = False
            new_rvs = []
            for thing in rvs:
                if len(thing.symmetries):
                    logging.debug("splitting thing %s", thing)
                    new_things = thing.split(0)
                    new_rvs += new_things
                    making_progress = True
                else:
                    new_rvs.append(thing)
            rvs = new_rvs
        return DecomposedLoops(self, rvs)

    def generate_halide(self, app, sizes):
        '''produce a Halide func implementing this loopnest'''
        if len(sizes) < len(self.indexes()):
            raise Exception("called without enough sizes")
        name = self.name
        g = app.funcs["g"]
        g_dens = app.clamps["g_dens"]
        self.simplify()
        logging.info("generating zone %s", name)
        # each symmetry zone has its own iteration space, implemented as an RDom with a where() clause.
        distinct_iters = self.indexes()
        logging.debug("distinct iters: %s", distinct_iters)

        piece_count = len(self.updates)

        iter_name_mapping = {k:k for k in app.vars}
        doing_something_useful = True
        while doing_something_useful:
            doing_something_useful = False
            for condition in self.conditions:
                if condition.op == "==":
                    for lhs, rhs in zip(condition.lhs, condition.rhs):
                        if iter_name_mapping[rhs] != iter_name_mapping[lhs]:
                            doing_something_useful = True
                            iter_name_mapping[rhs] = iter_name_mapping[lhs]

        logging.debug("iter_name_mapping: %s", iter_name_mapping)
        logging.debug("piece_count: %s", piece_count)

        rdom_iters = [(0, piece_count)]
        for size, index in zip(sizes, self.indexes()):
            rdom_iters.append((0, size))
        logging.debug("rdom iters: %s", rdom_iters)
        r = hl.RDom(rdom_iters, name+"_dom")
        # set local variables for RVars
        expanded_iters = {}
        distinct_iters = [r[i] for i in range(len(r))]
        assigned_already = {}
        ru = distinct_iters.pop(0)
        for a, b in iter_name_mapping.items():
            if b in assigned_already:
                expanded_iters[a] = assigned_already[b]
            else:
                iterator = distinct_iters.pop(0)
                expanded_iters[a] = iterator
                assigned_already[b] = iterator
        logging.debug("expanded_iters: %s", expanded_iters)

        for condition in self.conditions:
            if condition.op == "==":
                continue
            logging.debug("generating where condition %s", condition)
            expression = condition.generate(expanded_iters, N=sizes[0])
            if expression is not None:
                r.where(expression)
                logging.debug("resulting where clause: %s", r)

        for update in self.updates:
            logging.debug(update)

        # generate this nested loop

        def maybe_mux(s):
            '''wrap multiple Exprs in mux()'''
            if len(set(s)) == 1:
                return s[0]
            else:
                return hl.mux(hl.Expr(ru), s)

        zone_func = hl.Func(name)
        zone_func_initial_params = list(app.vars.values())[0:len(self.updates[0].out.indexes)]
        zone_func.__setitem__(zone_func_initial_params, hl.f64(0.0))
        left_hand_sides = [[] for x in self.updates[0].out.indexes]
        right_hand_sides = []
        for update in self.updates:
            # LHS indexes
            for i in range(len(update.out.indexes)):
                index = update.out.indexes[i]
                left_hand_sides[i].append(expanded_iters[index])
            # RHS
            rhs = None
            for element in update.elements:
                if element.name in app.clamps:
                    func = app.clamps[element.name]
                elif element.name in app.funcs:
                    func = app.funcs[element.name]
                else:
                    logging.critical("func %s not found"%element.name)
                func_args = [expanded_iters[x] for x in element.indexes]
                value = func.__getitem__(func_args)
                if rhs is None:
                    rhs = value
                else:
                    rhs *= value
            if update.coeff != 1.0:
                rhs *= update.coeff
            right_hand_sides.append(rhs)

        left_hand_sides = [ maybe_mux(x) for x in left_hand_sides ]
        right_hand_sides = maybe_mux(right_hand_sides)
        lhs = zone_func.__getitem__(left_hand_sides)
        zone_func.__setitem__(left_hand_sides, lhs + right_hand_sides)
        logging.debug("%s[%s, %s] += %s", name, left_hand_sides, right_hand_sides)

        app.funcs[name] = zone_func
        app.loopnest_funcs[name] = { "func": zone_func, "loopnest": self, "iters": expanded_iters, "rdom": r, "unroll": ru }
        return zone_func


    def copy(self):
        return LoopNest(
            name=self.name,
            conditions=[c.copy() for c in self.conditions],
            symmetries=[s.copy() for s in self.symmetries],
            updates=[u.copy() for u in self.updates],
            equalities={k: v for k,v in self.equalities.items()},
        )

    def __str__(self):
        lines = ["# name: " + self.name]
        for equality_lhs, equality_rhs in self.equalities.items():
            lines.append("# equality: %s == %s"%(equality_lhs, equality_rhs))
        for condition in self.conditions:
            lines.append("# condition: " + str(condition))
        for symmetry in self.symmetries:
            lines.append("# symmetry: " + str(symmetry))
        for update in self.updates:
            lines.append(str(update))
        return "\n".join(lines)

    def __repr__(self):
        things = [self.name, self.conditions, self.symmetries, self.updates]
        return "LoopNest(" + ", ".join([repr(x) for x in things]) + ")\n"


class DecomposedLoops():
    '''a set of nested loop structures which, together, implement the original algorithm'''
    def __init__(self, orig, loops):
        self.orig = orig
        self.loops = loops

    def generate_halide(self, app, sizes, precursors=None):
        params = list(app.vars.values())[0:len(self.orig.updates[0].out.indexes)]
        expr = None
        if precursors is not None:
            for input in precursors:
                if expr is None:
                    expr = input.__getitem__(params)
                else:
                    expr += input.__getitem__(params)
        for loop in self.loops:
            func = loop.generate_halide(app, sizes)
            if expr is None:
                expr = func.__getitem__(params)
            else:
                expr += func.__getitem__(params)
        output_func = hl.Func(self.orig.name)
        output_func.__setitem__(params, expr)

        app.funcs[self.orig.name] = output_func
        return output_func

    def __str__(self):
        lines = ["original:", str(self.orig)]
        for loop in self.loops:
            lines.append(str(loop))
        return "\n".join(lines)

    def __repr__(self):
        repr_loops = ", ".join([repr(x) for x in self.loops])
        repr_loops = "[" + repr_loops + "]"
        return "DecomposedLoops(%s, %s)"%(repr(self.orig), repr_loops)


# test all of the above
import unittest

class test_Symmetry(unittest.TestCase):
    def test_stringify(self):
        indexes = list("ijkl")
        function = Function("g", indexes)
        symmetry = Symmetry(function, [0,1], [2,3])
        self.assertEqual("<g(i,j,k,l) = g(k,l,i,j)>", str(symmetry))

    def test_copy(self):
        indexes = list("ijkl")
        function = Function("g", indexes)
        symmetry1 = Symmetry(function, [0,1], [2,3])
        symmetry2 = symmetry1.copy()
        self.assertNotEqual(id(symmetry1), id(symmetry2))
        self.assertNotEqual(id(symmetry1.func), id(symmetry2.func))
        self.assertNotEqual(id(symmetry1.lhs), id(symmetry2.lhs))
        self.assertNotEqual(id(symmetry1.rhs), id(symmetry2.rhs))


class test_Update(unittest.TestCase):
    def test_stringify(self):
        indexes = list("ijkl")
        out = indexes[0:2]
        in_ = indexes[2:4]
        g   = Function("g", indexes)
        out = Function("f", out)
        in_ = Function("d", in_)
        update = Update(out, [g, in_])
        self.assertEqual("f(i,j) += g(i,j,k,l) * d(k,l);", str(update))
        update = Update(out, [g, in_], op="*", coeff=-0.5)
        self.assertEqual("f(i,j) *= g(i,j,k,l) * d(k,l) * -0.5;", str(update))

class test_Condition(unittest.TestCase):
    def test_stringify(self):
        i, j, k, l = "ijkl"
        ij = Condition(i, j, "<")
        self.assertEqual("(i < j)", str(ij))
        kl = Condition(k, l, "<=")
        self.assertEqual("(k <= l)", str(kl))
        ijkl = Condition([i,j], [k,l], ">")
        self.assertEqual("(ij > kl)", str(ijkl))

class test_LoopNest(unittest.TestCase):
    def define_original_twoel(self):
        i,j,k,l = 'ijkl'
        g = Function("g", [i,j,k,l])
        zone = LoopNest(
            name="twoel",
            conditions=[],
            symmetries=[
                Symmetry(g, 0, 1),
                Symmetry(g, 2, 3),
                Symmetry(g, [0,1], [2,3]),
            ],
            updates=[
                Update(Function("g_fock", [i,j]), [g, Function("g_dens", [k,l])]),
                Update(Function("g_fock", [i,k]), [g, Function("g_dens", [j,l])], coeff=-0.5),
            ]
        )
        return zone

    def test_definition(self):
        zone = self.define_original_twoel()

    def test_simplified_numeric_symmetries(self):
        i,j,k,l = 'ijkl'
        g = Function("g", [i,j,k,l])
        zone = LoopNest(
            name="twoel",
            # numeric indexes
            symmetries=[ (g, 0, 1), (g, 2, 3), (g, [0,1], [2,3]) ],
            updates=[
                Update(Function("f", [i,j]), [g, Function("d", [k,l])]),
                Update(Function("f", [i,k]), [g, Function("d", [j,l])], coeff=-0.5),
            ]
        )
        zone.split_recursive()

    def test_simplified_string_symmetries(self):
        i,j,k,l = 'ijkl'
        g = Function("g", [i,j,k,l])
        zone = LoopNest(
            name="twoel",
            # numeric indexes
            symmetries=[ (g, "i", "j"), (g, "k", "l"), (g, ["i","j"], ["k","l"]) ],
            updates=[
                Update(Function("f", [i,j]), [g, Function("d", [k,l])]),
                Update(Function("f", [i,k]), [g, Function("d", [j,l])], coeff=-0.5),
            ]
        )
        zone.split_recursive()

    def test_stringify(self):
        zone = self.define_original_twoel()
        s = str(zone)
        self.assertEqual("""
# name: twoel
# symmetry: <g(i,j,k,l) = g(j,i,k,l)>
# symmetry: <g(i,j,k,l) = g(i,j,l,k)>
# symmetry: <g(i,j,k,l) = g(k,l,i,j)>
g_fock(i,j) += g(i,j,k,l) * g_dens(k,l);
g_fock(i,k) += g(i,j,k,l) * g_dens(j,l) * -0.5;
""", "\n"+s+"\n")

    def test_indexes(self):
        zone = self.define_original_twoel()
        self.assertEqual(["i","j","k","l"], list(zone.indexes()))
        zone1, zone2 = zone.split(0)
        self.assertEqual(["i","j","k","l"], list(zone1.indexes()))
        self.assertEqual(["i","k","l"], list(zone2.indexes()))

    def test_split_simple(self):
        zone = self.define_original_twoel()
        zone1, zone2 = zone.split(0)

        i, j, k, l = list("ijkl")
        expected_zone1 = LoopNest(
          name='twoel_i_lt_j', conditions=[Condition(i, j, '<')],
          symmetries=[
            Symmetry(Function('g', [i, j, k, l]), 2, 3),
            Symmetry(Function('g', [i, j, k, l]), [0,1], [2,3]),
          ],
          updates=[
            Update(Function('g_fock', [i, j]), [Function('g', [i, j, k, l]), Function('g_dens', [k, l])]),
            Update(Function('g_fock', [i, k]), [Function('g', [i, j, k, l]), Function('g_dens', [j, l])], coeff=-0.5),
            Update(Function('g_fock', [j, i]), [Function('g', [i, j, k, l]), Function('g_dens', [k, l])]),
            Update(Function('g_fock', [j, k]), [Function('g', [i, j, k, l]), Function('g_dens', [i, l])], coeff=-0.5),
          ]
        )
        expected_zone2 = LoopNest(
          name='twoel_i_eq_j', conditions=[Condition(i, j, '==')],
          symmetries=[
            Symmetry(Function('g', [i, j, k, l]), 2, 3),
            Symmetry(Function('g', [i, j, k, l]), [0,1], [2,3]),
          ],
          equalities={
            j: i
          },
          updates=[
            Update(Function('g_fock', [i, i]), [Function('g', [i, i, k, l]), Function('g_dens', [k, l])]),
            Update(Function('g_fock', [i, k]), [Function('g', [i, i, k, l]), Function('g_dens', [i, l])], coeff=-0.5),
          ]
        )

        self.assertEqual(str(expected_zone1), str(zone1))
        self.assertEqual(str(expected_zone2), str(zone2))

        testbed_orig  = Testbed([zone], zone.symmetries)
        testbed_split = Testbed([zone1, zone2], zone.symmetries)
        updates_orig  = testbed_orig.run()
        updates_split = testbed_split.run()
        self.assertEqual(updates_orig, updates_split)

    def test_split_tuple(self):
        zone = self.define_original_twoel()
        zone1, zone2 = zone.split(2)

        i, j, k, l = list("ijkl")
        expected_zone1 = LoopNest(
          name='twoel_ij_lt_kl', conditions=[Condition([i,j], [k,l], '<')],
          symmetries=[
            Symmetry(Function('g', [i, j, k, l]), 0, 1),
            Symmetry(Function('g', [i, j, k, l]), 2, 3),
          ],
          updates=[
            Update(Function('g_fock', [i, j]), [Function('g', [i, j, k, l]), Function('g_dens', [k, l])]),
            Update(Function('g_fock', [i, k]), [Function('g', [i, j, k, l]), Function('g_dens', [j, l])], coeff=-0.5),
            Update(Function('g_fock', [k, i]), [Function('g', [i, j, k, l]), Function('g_dens', [l, j])], coeff=-0.5),
            Update(Function('g_fock', [k, l]), [Function('g', [i, j, k, l]), Function('g_dens', [i, j])]),
          ]
        )
        expected_zone2 = LoopNest(
          name='twoel_ij_eq_kl', conditions=[Condition(i, k, '=='), Condition(j, l, '==')],
          symmetries=[
            Symmetry(Function('g', [i, j, k, l]), 0, 1),
          ],
          equalities={
            k: i,
            l: j
          },
          updates=[
            Update(Function('g_fock', [i, i]), [Function('g', [i, j, i, j]), Function('g_dens', [j, j])], coeff=-0.5),
            Update(Function('g_fock', [i, j]), [Function('g', [i, j, i, j]), Function('g_dens', [i, j])]),
          ]
        )

        self.assertEqual(str(expected_zone1), str(zone1))
        self.assertEqual(str(expected_zone2), str(zone2))

        testbed_orig  = Testbed([zone], zone.symmetries)
        testbed_split = Testbed([zone1, zone2], zone.symmetries)
        updates_orig  = testbed_orig.run()
        updates_split = testbed_split.run()
        self.assertEqual(updates_orig, updates_split)

    def test_split_recursive(self):
        zone = self.define_original_twoel()
        decomposed = zone.split_recursive()
        zones = decomposed.loops
        self.assertEqual(6, len(zones))
        logging.info("original zone: %s", zone)
        logging.info("split into %d zones.", len(zones))
        ranks = {}
        for newzone in zones:
            rank = len(newzone.indexes())
            if rank not in ranks:
                ranks[rank] = {}
            ranks[rank][newzone.name] = newzone
        for rank in reversed(sorted(ranks.keys())):
            logging.info("# RANK %s", rank)
            for name in sorted(ranks[rank].keys()):
                newzone = ranks[rank][name]
                logging.info("# ZONE")
                logging.info(newzone)

        testbed_orig  = Testbed([zone], zone.symmetries)
        testbed_split = Testbed(zones, zone.symmetries)
        updates_orig  = testbed_orig.run()
        updates_split = testbed_split.run()
        self.maxDiff = 2000
        self.assertEqual(updates_orig, updates_split)

    def test_generate_halide(self):
        zone = self.define_original_twoel()
        decomposed = zone.split_recursive()
        self.vars  = {k: hl.Var(k) for k in "ijkl"}
        i, j, k, l = [self.vars[k] for k in "ijkl"]
        g_dens = hl.Func("g_dens")
        g_dens[i,j] = i * j
        g = hl.Func("g")
        g[i,j,k,l] = hl.cos(i*j) * hl.sin(k*l)
        self.inputs = {"g": g, "g_dens": g_dens}
        self.clamps = {"g": g, "g_dens": g_dens}
        self.funcs = {"g": g, "g_dens": g_dens}
        self.loopnest_funcs = {}
        func = decomposed.generate_halide(self, [8, 8, 8, 8])


class test_Testbed(unittest.TestCase):
    def test_run(self):
        f1 = Function("f1", ["i","j"])
        z = LoopNest(
            name="z",
            symmetries=[ (f1, "i", "j") ],
            updates=[ Update(Function("f2", ["i","j"]), [f1]) ]
        )
        z1 = LoopNest(
            name="z1",
            conditions=[Condition("i","j","<")],
            updates=[
                Update(Function("f2", ["i","j"]), [f1]),
                Update(Function("f2", ["j","i"]), [f1])
            ]
        )
        z2 = LoopNest(
            name="z2",
            conditions=[Condition("i","j","==")],
            updates=[ Update(Function("f2", ["i","i"]), [f1]) ]
        )
        t1 = Testbed([z], z.symmetries)
        t2 = Testbed([z1, z2])
        r1 = t1.run()
        r2 = t2.run()
        import json
        s1 = json.dumps(r1, indent=2, sort_keys=True).split("\n")
        s2 = json.dumps(r2, indent=2, sort_keys=True).split("\n")
        if s1 != s2:
            import difflib
            diff = difflib.unified_diff(s1, s2, fromfile=z.name, tofile="split zones")
            for line in diff:
                logging.error(line)
        self.assertEqual(r1, r2)


if __name__ == '__main__':
    # run the test suite
    logging.basicConfig(level=logging.WARNING)
    unittest.main()
