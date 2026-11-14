"""Figure: P(F) and P(Delta>0) by decile of the learned failure gate's
score, reranking on NQ and TripClick (score-sufficiency amendment) and on
the pre-registered DBPedia and HotpotQA tests. Reads
docs/score_sufficiency_reference.json and docs/iv_profile.json (aggregates
only); writes paper_wsdm/fig_score_profile.tex (a figure* environment)."""

import json
from pathlib import Path

PANELS = (("rerank_nq", "NQ"), ("rerank_tripclick", "TripClick"),
          ("rerank_dbpedia", "DBPedia (pre-registered)"), ("rerank_hotpotqa", "HotpotQA (pre-registered)"))


def coords(prof, key, lo=None):
    if lo is None:
        return " ".join(f"({j + 1},{b[key]:.3f})" for j, b in enumerate(prof))
    return " ".join(f"({j + 1},{b[key + '_ci'][lo]:.3f})" for j, b in enumerate(prof))


def main() -> None:
    r = json.loads(Path("docs/score_sufficiency_reference.json").read_text())
    r.update(json.loads(Path("docs/iv_profile.json").read_text()))
    out = [r"""\begin{figure*}[t]
\centering
\pgfplotsset{profplot/.style={width=0.285\textwidth, height=3.3cm, xmin=0.5, xmax=10.5, ymin=0, ymax=1,
  xtick={1,5,10}, xlabel={failure-score decile}, xlabel style={font=\footnotesize},
  ticklabel style={font=\scriptsize}, title style={font=\footnotesize}, ylabel style={font=\footnotesize},
  mark size=1.2pt, line width=0.7pt, grid=major, grid style={gray!20}}}"""]
    for i, (key, title) in enumerate(PANELS):
        prof = r[key]["figure_profile"]
        extra = r", ylabel={probability}, legend style={font=\tiny, at={(0.97,0.03)}, anchor=south east, inner sep=1pt, row sep=-3pt}" if i == 0 else ""
        out.append(r"\begin{tikzpicture}" + "\n" + rf"\begin{{axis}}[profplot, title={{{title}}}{extra}]")
        for k, col, mk in (("p_fail", "red!75!black", "*"), ("p_helped", "green!50!black", "diamond*")):
            out.append(rf"\addplot[name path={k}lo{i}, draw=none, forget plot] coordinates {{{coords(prof, k, 0)}}};")
            out.append(rf"\addplot[name path={k}hi{i}, draw=none, forget plot] coordinates {{{coords(prof, k, 1)}}};")
            out.append(rf"\addplot[{col}, fill opacity=0.15, forget plot] fill between[of={k}lo{i} and {k}hi{i}];")
            out.append(rf"\addplot[{col}, mark={mk}] coordinates {{{coords(prof, k)}}};")
        if i == 0:
            out.append(r"\legend{$P(F)$, $P(\Delta>0)$}")
        out.append(r"\end{axis}" + "\n" + r"\end{tikzpicture}" + (r"\hfill" if i < 3 else ""))
    out.append(r"""\caption{One score, two targets (reranking, test, 95\% bootstrap bands). By decile
of the learned failure gate's score, the failure rate rises steeply; the share of
queries reranking helps stays flat or falls on NQ, TripClick, and DBPedia but rises
on HotpotQA. NQ, TripClick: exploratory.}
\label{fig:profile}
\end{figure*}""")
    Path("paper_wsdm/fig_score_profile.tex").write_text("\n".join(out) + "\n")
    print("written")


if __name__ == "__main__":
    main()
