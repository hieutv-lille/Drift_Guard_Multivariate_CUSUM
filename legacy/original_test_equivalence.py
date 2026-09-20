"""Pathwise regression against the supplied original Python implementation."""
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import numpy as np
from optimize import (ROOT, RESULTS, Design, Model, generate_paths, evaluate,
                      dump_json)

def load_original(path):
    spec = importlib.util.spec_from_file_location("original_dg", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def test(original_path):
    original = load_original(original_path)
    rows = []
    for k1,g,k2,L,h,delta in [(.5,1.5,.25,15,16.4,0.),
                              (.5,1.5,.25,15,16.4,1.5),
                              (.25,.75,.1,30,21.,2.),
                              (1.,2.,.5,5,5.,.75),
                              (.5,1.5,.25,2,2.,0.)]:
        n, H, seed = 80, 1500, 41415
        cfg = original.StudyConfig(k_primary=k1, warning_limit=g,
            k_confirmation=k2, confirmation_horizon=L)
        paths = generate_paths(n, H, seed)
        for method in ["dg", "kf"]:
            for direction in ["dense", "sparse", "mixed"]:
                expected = original.simulate_run_lengths(method,h,n_rep=n,max_steps=H,
                    config=cfg,delta=delta,direction=direction,seed=seed)
                actual,_ = evaluate(paths,Design(method,k1,g,k2,L),h,delta=delta,direction=direction)
                assert np.array_equal(actual,expected), (method,k1,g,k2,L,delta,direction,
                    np.flatnonzero(actual!=expected)[:10])
                rows.append({"method":method,"k1":k1,"g":g,"k2":k2,"L":L,
                    "h":h,"delta":delta,"direction":direction,"n":n,"pass":True})
    # Pathwise threshold monotonicity, including terminal-horizon crossings.
    paths = generate_paths(100,1000,31125)
    previous = np.zeros(100)
    for h in [0.,2.,5.,10.,15.,20.,40.]:
        current,_ = evaluate(paths,Design(),h)
        assert np.all(current>=previous)
        previous=current
    # Zero shift must not depend on the nominal change time; candidate status
    # is observable only on paths surviving until the change.
    a,_ = evaluate(paths,Design(),16.4,tau=1)
    b,active = evaluate(paths,Design(),16.4,tau=201)
    assert np.array_equal(a,b)
    assert np.all(active[b<201]==-1)
    assert np.all(np.isin(active[b>=201],[0,1]))
    # Delayed-shift adapter changes ONLY the injection time in the original
    # simulator; its protected-state recurrences are left untouched.
    source = inspect.getsource(original.simulate_run_lengths)
    source = source.replace("seed: int = 1,", "seed: int = 1, tau: int = 201,")
    source = source.replace("y = level + delta * shift_direction + rng.normal(size=(n_rep, p))",
                            "y = level + (delta if t >= tau else 0.) * shift_direction + rng.normal(size=(n_rep, p))")
    namespace = dict(original.__dict__)
    exec(source, namespace)
    delayed = namespace["simulate_run_lengths"]
    delayed_rows = []
    for design in [Design(), Design("dg", .25, .75, .1, 30), Design("kf")]:
        n, H, seed, h = 80, 1500, 71125, 16.4
        paths = generate_paths(n, H, seed)
        cfg = original.StudyConfig(k_primary=design.k1, warning_limit=design.g,
            k_confirmation=design.k2, confirmation_horizon=design.L)
        for delta in [1., 2.]:
            expected = delayed(design.method,h,n_rep=n,max_steps=H,config=cfg,
                               delta=delta,seed=seed,tau=201)
            actual,_ = evaluate(paths,design,h,delta=delta,tau=201)
            assert np.array_equal(expected,actual), ("delayed",design,delta)
            delayed_rows.append({"design":design.key,"delta":delta,"tau":201,"n":n,"pass":True})
    dump_json(RESULTS/"equivalence_tests.json",{
        "original_source_sha256":hashlib.sha256(Path(original_path).read_bytes()).hexdigest(),
        "pathwise_cases":rows,"pathwise_comparisons":sum(r["n"] for r in rows),
        "delayed_injection_cases":delayed_rows,
        "monotonicity_pass":True,"late_change_bookkeeping_pass":True})
    print(f"PASS: {len(rows)} original-code cases / {sum(r['n'] for r in rows)} exact run-length matches; 480 delayed-injection matches; monotonicity and late-change bookkeeping.",flush=True)

if __name__=="__main__":
    test(Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/"legacy"/"dg_mcusum_study.py")
