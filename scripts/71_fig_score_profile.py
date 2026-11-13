"""Candidate figure: P(F) and P(Delta>0) by decile of the learned failure
gate's score, for the four settings fixed in the score-sufficiency
amendment. Reads docs/score_sufficiency_reference.json (aggregates only);
writes paper_wsdm/fig_score_profile.tex (a figure* environment)."""

import json
from pathlib import Path

PANELS = (("rerank_nq", "NQ reranking"), ("rerank_tripclick", "TripClick reranking"),
          ("rerank_arguana", "ArguAna reranking"), ("rerank_fiqa", "FiQA reranking (pre-reg.)"))


def coords(prof, key, lo=None):
    if lo is None:
        return " ".join(f"({j + 1},{b[key]:.3f})" for j, b in enumerate(prof))
    return " ".join(f"({j + 1},{b[key + '_ci'][lo]:.3f})" for j, b in enumerate(prof))


def main() -> None:
    r = json.loads(Path("docs/score_sufficiency_reference.json").read_text())
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
    out.append(r"""\caption{Same score, two targets: by decile of the learned failure gate's score
(test; 95\% bootstrap bands), the failure rate rises steeply while the share of
queries reranking helps stays flat or falls (exploratory analysis; FiQA from the pre-registered test run).}
\label{fig:profile}
\end{figure*}""")
    Path("paper_wsdm/fig_score_profile.tex").write_text("\n".join(out) + "\n")
    print("written")


if __name__ == "__main__":
    main()
