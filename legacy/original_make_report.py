"""Generate red LaTeX additions and compact tables from completed run outputs."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "optimization_results"
LABELS = {"DG_original":"DG (reference)", "DG_selected":"DG (selected)",
          "KF_original":"KF (reference)", "KF_selected":"KF (selected)"}

def make_report():
    meta = json.loads((OUT/"run_metadata.json").read_text())
    if meta["status"] != "completed":
        raise RuntimeError("Refusing to report an incomplete simulation")
    df = pd.read_csv(OUT/"independent_tests.csv")
    pairs = pd.read_csv(OUT/"paired_comparisons.csv")
    fine = pd.read_csv(OUT/"refined_search.csv")
    p = meta["protocol"]
    H = p["horizon"]
    def row(case,label,delta=0.):
        return df[(df.case==case)&(df.label==label)&(df.delta==delta)].iloc[0]
    def num(x): return f"{x:.1f}"
    def interval(r): return f"{r['mean']:.1f} [{r['ci_low']:.1f}, {r['ci_high']:.1f}]"
    dg = meta["selected_designs"]["DG_selected"]
    kf = meta["selected_designs"]["KF_selected"]
    h = meta["limits"]
    orig = row("in_control","DG_original")
    sel = row("in_control","DG_selected")
    reductions = [100*(1-row("zero_state","DG_selected",d)["mean"]/
                       row("zero_state","DG_original",d)["mean"]) for d in [1.,1.5,2.]]
    top_scores = fine[fine.method=="dg"].head(2).objective.to_numpy()
    main = []
    main.append(r"\begin{revision}")
    main.append(
        f"The joint search selects $(k_1,g,k_2,L)=({dg['k1']:g},{dg['g']:g},{dg['k2']:g},{dg['L']})$, "
        f"with final limit $h={h['DG_selected']:.4f}$. The selected KF reference uses "
        f"$k={kf['k1']:g}$ and $h={h['KF_selected']:.4f}$. These are empirical selections from the specified "
        "two-stage search, not proven global optima. All four configurations, including the fixed-reference "
        "DG and KF settings, are calibrated on common in-control paths; "
        "their limits are then held fixed for every test below.")
    main.append(
        f"The two lowest refined DG scores are {top_scores[0]:.2f} and {top_scores[1]:.2f}, so the search does not identify a "
        "clearly separated unique optimum. In independent in-control validation, selected DG has mean "
        f"{sel['mean']:.1f}, whereas the two KF references have means "
        f"{row('in_control','KF_original')['mean']:.1f} and {row('in_control','KF_selected')['mean']:.1f}. "
        "Both KF Monte Carlo intervals exclude 500 on the conservative side. False-alarm behavior is "
        "therefore approximately, not exactly, matched; the test results are not used to adjust the limits.")
    main.append(r"\end{revision}")
    main.extend([r"\begin{table}[htbp]",r"\centering\color{warmred}\small",
                 r"\caption{\rev{Independent in-control validation of fixed-reference and selected configurations.}}",
                 r"\label{tab:optimization-calibration}",r"\begin{tabular}{lrrrrrl}",r"\toprule",
                 r"Chart & $k_1$ & $g$ & $k_2$ & $L$ & $h$ & Mean $RL_0$ [95\% MC CI] \\",r"\midrule"])
    for label in LABELS:
        r = row("in_control",label)
        par = meta["selected_designs"][label]
        vals = [f"{par['g']:g}",f"{par['k2']:g}",str(par['L'])] if label.startswith("DG") else ["--"]*3
        main.append(f"{LABELS[label]} & {par['k1']:g} & "+" & ".join(vals)+f" & {h[label]:.4f} & {interval(r)}"+r" \\")
    main.extend([r"\bottomrule",r"\end{tabular}",
        r"\par\smallskip\begin{minipage}{0.98\linewidth}\footnotesize\color{warmred}",
        f"Each limit uses {p['n_final_cal']:,} calibration paths; validation uses {p['n_ic_test']:,} new paths. "
        f"The reported mean is restricted at $H={H:,}$. Intervals quantify test-sample Monte Carlo "
        "uncertainty conditional on the selected parameters and calibrated limit; they do not include "
        "search or calibration uncertainty.",r"\end{minipage}",r"\end{table}"])
    main.extend([r"\begin{table}[htbp]",r"\centering\color{warmred}\small",
                 r"\caption{\rev{Detection performance at calibrated limits: restricted mean delay.}}",
                 r"\label{tab:optimization-delay}",r"\begin{tabular}{lrrrr}",r"\toprule",
                 r"Change / $\Delta$ & DG (reference) & DG (selected) & KF (reference) & KF (selected) \\",r"\midrule"])
    for case,deltas in [("zero_state",p["test_deltas"]),("late_change",p["selection_deltas"])]:
        if case=="late_change": main.append(r"\midrule")
        for d in deltas:
            head = (r"$\tau=1$" if case=="zero_state" else r"$\tau=201$")+f", {d:g}"
            main.append(head+" & "+" & ".join(num(row(case,label,d)["mean"]) for label in LABELS)+r" \\")
    main.extend([r"\bottomrule",r"\end{tabular}",
        r"\par\smallskip\begin{minipage}{0.98\linewidth}\footnotesize\color{warmred}",
        f"Each test scenario starts from {p['n_ooc_test']:,} independently generated paths. For $\\tau=201$, "
        r"the estimand is $\E[\min(T-\tau+1,H-\tau+1)\mid T\ge\tau]$; "
        "each chart has its own surviving subset. This is a finite delayed-change experiment, not a "
        "steady-state claim. Full intervals, tail probabilities, survivor counts and raw stopping times "
        "are supplied with the notebook.",r"\end{minipage}",r"\end{table}"])
    main.append(r"\begin{revision}")
    main.append(
        "At $\\Delta=1,1.5,2$, the selected DG design changes the zero-state mean delay relative to "
        f"the calibrated fixed-reference DG configuration by reductions of {reductions[0]:.1f}\\%, "
        f"{reductions[1]:.1f}\\%, and {reductions[2]:.1f}\\%, respectively. "
        "Common calibration constrains false alarms but does not make detection delays equivalent. "
        "The selection objective concerns chart parameters with fixed Gaussian dynamics; it does not estimate the model "
        "or select the industrial innovation transformation.")
    cis = [pairs[(pairs.case=="zero_state")&(pairs.delta==d)&
                 (pairs.contrast=="DG_selected - DG_original")].iloc[0] for d in [1.,1.5,2.]]
    main.append("The corresponding paired 95\\% Monte Carlo intervals for the mean difference "
                "(selected minus reference) are "+", ".join(
                    f"[{c['ci_low']:.1f}, {c['ci_high']:.1f}]" for c in cis)
                +"; these are per-comparison intervals, without multiplicity adjustment.")
    main.append("")
    main.append(
        "The gain is specific to the selection scenario. At $\\tau=201$, selected DG has mean delays "
        + ", ".join(f"{row('late_change','DG_selected',d)['mean']:.1f}" for d in [1.,1.5,2.])
        + ", compared with "
        + ", ".join(f"{row('late_change','DG_original',d)['mean']:.1f}" for d in [1.,1.5,2.])
        + " for reference DG; both DG settings are slower than either KF reference in this delayed-change "
        "experiment. Protection is already active immediately before the change on "
        f"{100*row('late_change','DG_selected',1.)['candidate_at_change_given_survival']:.1f}\\% "
        "of surviving selected-DG paths. These findings do not support transferring zero-state tuning "
        "to faults arriving during ongoing monitoring without including change-time scenarios in "
        "the selection objective and validating again on fresh paths.")
    main.append("")
    main.append(
        "Mean-delay gains also trade off against typical-path speed: at $\\Delta=2$, selected DG's "
        f"median changes from {row('zero_state','DG_original',2.)['median']:g} to "
        f"{row('zero_state','DG_selected',2.)['median']:g} observations, while $P(T>50)$ changes from "
        f"{row('zero_state','DG_original',2.)['p_gt_50']:.4f} to "
        f"{row('zero_state','DG_selected',2.)['p_gt_50']:.4f}. "
        "Halving the true drift covariance raises its restricted in-control mean to "
        f"{row('drift_half','DG_selected')['mean']:.1f} "
        f"({100*row('drift_half','DG_selected')['censor_rate']:.2f}\\% censored), "
        "and doubling it lowers the mean to "
        f"{row('drift_double','DG_selected')['mean']:.1f}. "
        "The selected parameters therefore do not remove the need for model checking and recalibration.")
    main.append(r"\end{revision}")
    text = "\n".join(main)+"\n"
    (OUT/"results_addition_red.tex").write_text(text,encoding="utf-8")

    methods = r"""\begin{revision}
Calibrating $h$ controls the frequency of false alarms; it does not determine which chart configuration detects a fault most effectively. The remaining parameters act together. A smaller $k_1$ retains weaker live-innovation increments, while a smaller $g$ makes protection easier to trigger, potentially before the live filter absorbs the shift, but also admits weaker noise-driven candidates. A smaller $k_2$ retains weaker protected increments. Increasing $L$ allows more time to confirm a candidate, but also permits a longer forecast without measurement correction, greater forecast uncertainty and longer protection episodes. Changing any of these choices changes the in-control stopping rule and therefore the limit needed to maintain the false-alarm target. Their values should consequently be selected jointly, with $h$ recalibrated for every candidate.

We search 240 configurations $\theta=(k_1,g,k_2,L)$ over
\[
\begin{aligned}
 k_1&\in\{0.25,0.5,0.75,1\},&
 g&\in\{0.75,1,1.5,2\},\\
 k_2&\in\{0.1,0.25,0.5\},&
 L&\in\{5,10,15,20,30\}.
\end{aligned}
\]
For each $\theta$, bisection selects a limit $h_\theta$ giving a simulated in-control mean of 500, restricted at $H=6000$. Among these calibrated candidates, we minimize
\begin{equation}
 \widehat J(\theta)=\frac13\sum_{\Delta\in\{1,1.5,2\}}
 \left[\widehat{\E}_\Delta\min\{T_\theta(h_\theta),H\}
       +50\widehat{\Pp}_\Delta\{T_\theta(h_\theta)>50\}\right].
 \label{eq:optimization-objective}
\end{equation}
The mean term rewards prompt detection; the exceedance term gives extra weight to paths that remain undetected after 50 observations. Its penalty of 50 observations and the equal shift weights are fixed before simulation as illustrative design choices, not estimated industrial costs. Selection uses dense shifts present from the first monitored observation. Thus, it targets zero-state performance rather than an unspecified distribution of fault arrival times.

A coarse pass uses 640 calibration paths and 1,000 selection paths per configuration. The 12 lowest-scoring DG settings and the fixed-reference setting enter refinement with 4,000 fresh calibration and 4,000 fresh selection paths. KF reference values $k\in\{0.25,0.5,0.75,1\}$ undergo the same two stages. The lowest refined score selects each design. After this choice is fixed, 20,000 independent in-control paths determine its final limit; 20,000 further paths validate false-alarm behavior, and 10,000 paths per scenario assess detection. Common paths support paired comparisons within each stage, but calibration, selection, refinement and final testing use disjoint seed streams. Reported test intervals condition on the selected design and calibrated limit; they do not include search or calibration uncertainty.

To assess the scope of the choice, we also introduce shifts at $\tau=201$ without resetting either chart, and halve or double the true drift covariance without retuning. Delayed-change results condition on no alarm before $\tau$ and use the remaining horizon $H-\tau+1$. These scenarios test transfer beyond the selection objective; they are not used to select parameters. The search holds the Gaussian model fixed and is distinct from estimating state dynamics or choosing the industrial innovation transformation.
\end{revision}
"""
    (OUT/"methods_addition_red.tex").write_text(methods,encoding="utf-8")
    # Manuscript-facing CSVs retain full precision in the primary source file.
    df[df.case=="in_control"].to_csv(OUT/"table_new_arl0.csv",index=False)
    df[df.case.isin(["zero_state","late_change"])].to_csv(OUT/"table_new_detection.csv",index=False)
    print(text)
    return text, methods

if __name__=="__main__":
    make_report()
