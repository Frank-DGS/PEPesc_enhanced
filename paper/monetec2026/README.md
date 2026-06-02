# MoNeTec-2026 Springer Paper Draft

This directory contains the working LaTeX draft for the MoNeTec-2026 **Paper in English** submission:

> GRU-Assisted Performance Enhancing Proxy for Adaptive Transmission over Dynamic High-Latency Networks

## Purpose

The paper is converted from the Chinese BUPT undergraduate thesis into a Springer Computer Science Proceedings style regular/full paper. The target submission is the uConfy `Paper in English` track for Springer proceedings.

## Template requirement

This draft uses the Springer LNCS/CCIS LaTeX entry point:

```tex
\documentclass[runningheads]{llncs}
```

Before compiling, download the official Springer LaTeX2e proceedings template from Springer and place these files in this directory:

- `llncs.cls`
- `splncs04.bst`

Official page: <https://www.springer.com/gp/computer-science/lncs/conference-proceedings-guidelines>

Do not copy the BUPT thesis formatting. The review version may be uploaded as PDF/docx/txt in uConfy, but this project is prepared directly in the final Springer style to reduce later camera-ready work.

## Suggested build

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

or simply:

```bash
make
```

## Current structure

```text
main.tex
references.bib
sections/
  01_introduction.tex
  02_related_work.tex
  03_framework.tex
  04_implementation_setup.tex
  05_evaluation.tex
  06_discussion.tex
  07_conclusion.tex
figures/
  README.md
```

## Writing policy

- Do not translate the thesis sentence by sentence.
- Compress thesis Chapters 1--3 into Introduction and Related Work.
- Use thesis Chapters 4--6 as the core of Method, Setup, and Evaluation.
- Keep all experimental claims consistent with the thesis: static throughput +7.6%, dynamic recovery time 1.089 s to 0.664 s, throughput deficit area -20.0%, p95 queue delay 137.6 ms to 44.1 ms, long-duration effective running time 100.9 s to 188.4 s.
- Avoid over-claiming real deployment; describe the evaluation as Mininet-based emulation with controlled dynamic link scenarios.
