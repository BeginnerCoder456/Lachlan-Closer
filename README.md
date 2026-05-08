# QCAA Mathematical Methods Calculator

A full calculus calculator covering the entire QCAA Mathematical Methods syllabus.

## Features

| Tab | What it does |
|-----|-------------|
| **Differentiation** | Any order, any variable |
| **Integration** | Indefinite & definite with exact + numeric answer |
| **Limits** | One-sided and two-sided limits, infinity |
| **Chain / Product / Quotient** | All three rules with step-by-step breakdown |
| **Optimisation** | Critical points, max/min classification via f″, inflection points |
| **Area Between Curves** | Exact area, auto-intersection detection |
| **Differential Equations** | Separable & first-order with initial conditions |
| **Trig & Exp/Log** | Quick diff/int of all QCAA standard functions + reference sheet |

## Setup

```bash
pip install sympy                          # calculator only
pip install sympy opencv-python face-recognition numpy   # + prank
```

## Run

```bash
python calculator.py
```

## Prank Setup (important!)

The calculator silently watches the webcam. When it sees Lachlan, it shuts down the computer.

**To activate the prank:**
1. Get a clear photo of Lachlan's face (front-on, well-lit)
2. Name it exactly `lachlan.jpg`
3. Put it in the same folder as `calculator.py`
4. That's it — the detector starts automatically when the calculator opens

If `lachlan.jpg` is missing or `face-recognition` is not installed, the calculator works perfectly normally — no errors, no hints.
