"""K-plan: a rigorous, convex upper bound on the neuron-level paid constant K of a single-character test.

Input: the paid leak positions of the test, each with family f = (l, a, b) (output class, input classes) and
coefficient 2^{z/2} (z = number of DC inputs).  The nonnegative neuron tensor has entry
   Tn[f] = sum over leak positions in family f of 2^{z/2} |w_m|.
Lemma (block bound): K <= sqrt(I_A^2 + I_B^2 + I_C^2), I_X = injective norm of block X
   (A: output non-L; B: output in L, input-1 non-L; C: output, input 1 in L, input 2 non-L).
Upper bounds for I_X used here (all valid):
   - split the block's (Y,Z) support graph (union over output slices) into connected components: I_X = max over comps;
   - a component with one output slice: matrix M[Y,Z]; if M is a partial permutation -> max entry;
     if M has one row or one column -> l2 norm; else Frobenius;
   - a component with two output slices M1, M2: if their row supports or column supports are disjoint ->
     max(||M1||, ||M2||); else sqrt(||M1||^2 + ||M2||^2); slice norms as above.
The plan is static (depends only on which families are present), and can be evaluated with cvxpy expressions
(for optimisation) or with floats rounded upward (for certificates).
"""
import itertools
import numpy as np
from chanlib import block_assign


def _components(cells):
    """cells: list of (Y, Z); return connected components of the bipartite graph rows Y / cols Z"""
    cells = list(cells)
    parent = {}
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]; u = parent[u]
        return u
    def union(u, v):
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
    for (y, z) in cells:
        parent.setdefault(('Y', y), ('Y', y)); parent.setdefault(('Z', z), ('Z', z))
        union(('Y', y), ('Z', z))
    comps = {}
    for (y, z) in cells:
        comps.setdefault(find(('Y', y)), []).append((y, z))
    return list(comps.values())


def _matrix_rule(cells):
    rows = set(y for y, z in cells); cols = set(z for y, z in cells)
    if len(rows) == len(cells) and len(cols) == len(cells):
        return 'max'
    if len(rows) == 1 or len(cols) == 1:
        return 'l2'
    return 'fro'


def make_plan(fams, L=(1, 2)):
    """fams: iterable of families (l, a, b) present. Returns a nested plan:
       ('l2', [block plans]) with block plan ('max', [component plans]); component plan either
       ('slice', rule, [fams]) or ('two', joiner, slice1, slice2) with joiner in {'max','cs'}."""
    blocks = {}
    for f in fams:
        blocks.setdefault(block_assign(f, L), []).append(f)
    bplans = []
    for bi in sorted(blocks):
        fl = blocks[bi]
        # orient: the 'slice' axis is the output axis for blocks B, C, A alike (output index l)
        cells = [(f[1], f[2]) for f in fl]
        comps = _components(cells)
        cplans = []
        for comp in comps:
            cset = set(comp)
            cf = [f for f in fl if (f[1], f[2]) in cset]
            slices = {}
            for f in cf:
                slices.setdefault(f[0], []).append(f)
            sl = list(slices.values())
            if len(sl) == 1:
                cplans.append(('slice', _matrix_rule([(f[1], f[2]) for f in sl[0]]), sl[0]))
            elif len(sl) == 2:
                s1, s2 = sl
                r1 = set(f[1] for f in s1); r2 = set(f[1] for f in s2)
                c1 = set(f[2] for f in s1); c2 = set(f[2] for f in s2)
                joiner = 'max' if (r1.isdisjoint(r2) or c1.isdisjoint(c2)) else 'cs'
                cplans.append(('two', joiner,
                               ('slice', _matrix_rule([(f[1], f[2]) for f in s1]), s1),
                               ('slice', _matrix_rule([(f[1], f[2]) for f in s2]), s2)))
            else:
                # three or more output slices in one component: fall back to Frobenius of the component
                cplans.append(('slice', 'fro', cf))
        bplans.append(('max', cplans))
    return ('l2', bplans)


# ---------------------------------------------------------------- evaluation (cvxpy)
def eval_cvx(plan, entry):
    import cvxpy as cp
    def ev(pl):
        kind = pl[0]
        if kind == 'l2':
            return cp.norm(cp.hstack([ev(b) for b in pl[1]]), 2)
        if kind == 'max':
            parts = [ev(c) for c in pl[1]]
            return parts[0] if len(parts) == 1 else cp.max(cp.hstack(parts))
        if kind == 'slice':
            rule, fl = pl[1], pl[2]
            vals = [entry[f] for f in fl]
            if len(vals) == 1:
                return vals[0]
            if rule == 'max':
                return cp.max(cp.hstack(vals))
            return cp.norm(cp.hstack(vals), 2)
        if kind == 'two':
            a, b = ev(pl[2]), ev(pl[3])
            return cp.max(cp.hstack([a, b])) if pl[1] == 'max' else cp.norm(cp.hstack([a, b]), 2)
        raise ValueError(kind)
    return ev(plan)


# ---------------------------------------------------------------- evaluation (floats, rounded up)
def _up(x):
    return np.nextafter(x, np.inf)


def _l2_up(vals):
    s = 0.0
    for v in vals:
        s = _up(s + _up(v * v))
    return _up(np.sqrt(s))


def eval_up(plan, entry):
    """entry: dict family -> float upper bound (>= 0).  Returns an upper bound of the plan value."""
    def ev(pl):
        kind = pl[0]
        if kind == 'l2':
            return _l2_up([ev(b) for b in pl[1]])
        if kind == 'max':
            return max(ev(c) for c in pl[1])
        if kind == 'slice':
            rule, fl = pl[1], pl[2]
            vals = [entry[f] for f in fl]
            if len(vals) == 1:
                return vals[0]
            return max(vals) if rule == 'max' else _l2_up(vals)
        if kind == 'two':
            a, b = ev(pl[2]), ev(pl[3])
            return max(a, b) if pl[1] == 'max' else _l2_up([a, b])
        raise ValueError(kind)
    return ev(plan)


if __name__ == '__main__':
    from chanlib import Chan
    CH = Chan(7, (1, 2))
    for case in [('1', '1'), ('1', '2'), ('1', '12'), ('12', '1'), ('12', '12')]:
        fams = set()
        for j, T in ((1, case[0]), (2, case[1])):
            for (m, role, f, d) in CH.component(j, T):
                if role == 'paid':
                    fams.add(tuple(int(t) for t in f))
        print(case, make_plan(fams))
