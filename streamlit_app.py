"""
Fiber-Optics Concept Lab
========================
Interactive practice environment for
  * Chapter 2 - Optics Review  (ray theory, lenses, numerical aperture, diffraction, Gaussian beams)
  * Chapter 3 - Lightwave Fundamentals, (EM waves, dispersion, information rate,
                polarization, resonant cavities, Fresnel reflection, critical-angle reflection)

Run with:   streamlit run app.py
All equations follow the lecture slides (equation numbers are quoted where the slides give them).
"""
import math

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Arc, Polygon
import streamlit as st

# --------------------------------------------------------------------------------------
# Constants & style
# --------------------------------------------------------------------------------------
C0 = 3.0e8            # m/s  (value used in the slides)
ETA0 = 376.73         # ohm, free-space impedance
LOG10E = math.log10(math.e)

INK = "#1b2a41"
TEAL = "#0f8b8d"
AMBER = "#e08e0b"
RED = "#c8553d"
BLUE = "#2a6fdb"
GREEN = "#3a9d5d"
PURPLE = "#7b5ea7"
GREY = "#8a97a6"
LIGHT = "#eef3f7"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.grid": True, "grid.color": "#d9e0e7", "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.prop_cycle": matplotlib.cycler(color=[BLUE, RED, GREEN, AMBER, PURPLE, TEAL]),
    "legend.frameon": False, "figure.dpi": 110,
})

MATERIALS = {"Gas / CO₂ (n = 1)": 1.0, "Water (n = 1.33)": 1.33, "Glass (n ≈ 1.5)": 1.5,
             "Silicon (n = 3.5)": 3.5, "GaAs (n = 3.35)": 3.35, "Custom": None}


# --------------------------------------------------------------------------------------
# Generic UI helpers
# --------------------------------------------------------------------------------------
def show(fig):
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def header(title, subtitle):
    st.markdown(f"## {title}")
    st.caption(subtitle)


def metrics(items, cols=None):
    cols = st.columns(cols or len(items))
    for c, (label, value, *help_) in zip(cols, items):
        c.metric(label, value, help=help_[0] if help_ else None)


def material_input(label, key, default="Glass (n ≈ 1.5)"):
    names = list(MATERIALS)
    choice = st.selectbox(label, names, index=names.index(default), key=key + "_sel")
    if MATERIALS[choice] is None:
        return st.number_input(f"{label} – custom n", 1.0, 5.0, 1.45, 0.01, key=key + "_cus")
    return MATERIALS[choice]


def fmt_time(seconds):
    a = abs(seconds)
    if a == 0:
        return "0 s"
    for unit, scale in (("s", 1), ("ms", 1e-3), ("µs", 1e-6), ("ns", 1e-9), ("ps", 1e-12), ("fs", 1e-15)):
        if a >= scale:
            return f"{seconds / scale:.4g} {unit}"
    return f"{seconds:.3g} s"


def fmt_rate(hz, unit="Hz"):
    a = abs(hz)
    for prefix, scale in (("T", 1e12), ("G", 1e9), ("M", 1e6), ("k", 1e3), ("", 1)):
        if a >= scale:
            return f"{hz / scale:.4g} {prefix}{unit}"
    return f"{hz:.4g} {unit}"


def block_diagram(blocks, arrow_labels=None, figsize=(10, 2.3), height=0.95, colors=None):
    """Simple left-to-right block diagram. blocks = [(title, subtitle), ...]."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    n = len(blocks)
    w, gap = 1.25, 0.75
    colors = colors or [LIGHT] * n
    for i, (title, sub) in enumerate(blocks):
        x = i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0), w, height, boxstyle="round,pad=0.02,rounding_size=0.09",
                                    fc=colors[i], ec=INK, lw=1.4))
        ax.text(x + w / 2, height * 0.72, title, ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(x + w / 2, height * 0.34, sub, ha="center", va="center", fontsize=8, linespacing=1.3)
        if i < n - 1:
            ax.annotate("", xy=(x + w + gap - 0.02, height / 2), xytext=(x + w + 0.02, height / 2),
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6))
            if arrow_labels and arrow_labels[i]:
                ax.text(x + w + gap / 2, height / 2 + 0.08, arrow_labels[i], ha="center", va="bottom",
                        fontsize=7.5, color=RED)
    ax.set_xlim(-0.1, (n - 1) * (w + gap) + w + 0.1)
    ax.set_ylim(-0.15, height + 0.35)
    ax.grid(False)
    return fig


def arrow(ax, p0, p1, color=INK, lw=2.0, style="-|>", ls="-", alpha=1.0, zorder=3):
    ax.annotate("", xy=p1, xytext=p0, zorder=zorder,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, linestyle=ls, alpha=alpha,
                                shrinkA=0, shrinkB=0))


# --------------------------------------------------------------------------------------
# Physics core (pure functions – unit-tested separately)
# --------------------------------------------------------------------------------------
def snell_theta_t(n1, n2, th_i_deg):
    """Eq. (2.3): sin(theta_t) = (n1/n2) sin(theta_i). Returns None if total internal reflection."""
    s = n1 / n2 * math.sin(math.radians(th_i_deg))
    if s > 1.0:
        return None
    return math.degrees(math.asin(s))


def critical_angle_deg(n1, n2):
    """Eq. (3.34): sin(theta_c) = n2/n1, exists only if n2 < n1."""
    return math.degrees(math.asin(n2 / n1)) if n2 < n1 else None


def fresnel(n1, n2, th_deg):
    """Eqs. (3.29) and (3.30) - complex reflection coefficients (slide sign convention)."""
    th = np.radians(np.atleast_1d(np.asarray(th_deg, dtype=float)))
    c = np.cos(th)
    root = np.sqrt(n2 ** 2 - (n1 * np.sin(th)) ** 2 + 0j)
    rho_p = (-n2 ** 2 * c + n1 * root) / (n2 ** 2 * c + n1 * root)
    rho_s = (n1 * c - root) / (n1 * c + root)
    return rho_p, rho_s


def brewster_deg(n1, n2):
    return math.degrees(math.atan(n2 / n1))


def lens_focal_length(n, r1, r2):
    """Eq. (2.4)."""
    return 1.0 / ((n - 1.0) * (1.0 / r1 + 1.0 / r2))


def _trapz(y, x, axis=-1):
    f = getattr(np, "trapezoid", None) or np.trapz
    return f(y, x, axis=axis)


def bessel_j(order, x):
    """Bessel function J0/J1 via the integral representation (no SciPy dependency)."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    tau = np.linspace(0, np.pi, 801)
    arg = order * tau[None, :] - x[:, None] * np.sin(tau[None, :])
    return _trapz(np.cos(arg), tau, axis=1) / np.pi


def airy_intensity(x):
    """Normalised Airy pattern (2 J1(x)/x)^2 (x = pi D r / (lambda f))."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    out = np.ones_like(x)
    nz = np.abs(x) > 1e-9
    out[nz] = (2 * bessel_j(1, x[nz]) / x[nz]) ** 2
    return out


def airy_encircled(x):
    """Fraction of power inside x for the Airy pattern: 1 - J0^2 - J1^2."""
    return 1.0 - bessel_j(0, x) ** 2 - bessel_j(1, x) ** 2


def spot_diameter(lam, f, D):
    """Eq. (2.14): d = 2.44 lambda f / D."""
    return 2.44 * lam * f / D


def gaussian_intensity(r, w, i0=1.0):
    """Eq. (2.5)."""
    return i0 * np.exp(-2.0 * r ** 2 / w ** 2)


M0_MODEL = -0.095   # ps / (nm^2 km)
LAM0_MODEL = 1300.0  # nm


def material_dispersion_model(lam_nm):
    """Slide 'MATERIAL DISPERSION': M = (M0/4)(lambda - lambda0^4/lambda^3), 1200-1600 nm, ps/(nm km)."""
    lam = np.asarray(lam_nm, dtype=float)
    return M0_MODEL / 4.0 * (lam - LAM0_MODEL ** 4 / lam ** 3)


def pulse_spread(M_ps_nm_km, dlam_nm, L_km):
    """Eqs. (3.14)/(3.12): delta_tau = |M| * delta_lambda * L  (returns seconds)."""
    return abs(M_ps_nm_km) * dlam_nm * L_km * 1e-12


def bandwidth_from_spread(dtau):
    """Returns dict of the slide's bandwidth / data-rate results for a pulse spread dtau (s)."""
    return {
        "f3dB_optical": 1.0 / (2.0 * dtau),          # Eq. (5)
        "f3dB_electrical": 0.35 / dtau,               # Eq. (3.19)
        "R_RZ": 0.35 / dtau,                          # Eq. (3.20)
        "R_NRZ": 0.7 / dtau,                          # Eq. (3.21)
    }


def freq_dependent_loss_db(f_over_f3):
    """Eq. (7): L_f = -10 log10( exp(-ln2 (f/f3dB)^2) )."""
    return -10.0 * np.log10(np.exp(-math.log(2.0) * np.asarray(f_over_f3, dtype=float) ** 2))


def cavity_params(L, n, lam0):
    """Eqs. (3.22), (3.25), (3.26). L, lam0 in metres."""
    df = C0 / (2.0 * L * n)
    dlam = lam0 ** 2 * df / C0
    m = 2.0 * n * L / lam0
    return {"df": df, "dlam": dlam, "m": m}


def evanescent_alpha(n1, n2, th_deg, lam0):
    """alpha = k0 sqrt(n1^2 sin^2 theta - n2^2), lam0 in metres -> 1/m."""
    s = (n1 * math.sin(math.radians(th_deg))) ** 2 - n2 ** 2
    return 2 * math.pi / lam0 * math.sqrt(s) if s > 0 else 0.0


# --------------------------------------------------------------------------------------
# HOME
# --------------------------------------------------------------------------------------
def page_home():
    st.markdown("# Fiber-Optics Concept Lab")
    st.markdown(
        "A hands-on practice environment for **Chapter 2 – Optics Review** and "
        "**Chapter 3 – Lightwave Fundamentals**. Every module follows the lecture slides: "
        "the same symbols, the same equations, the same worked examples – but you can now change every "
        "input and watch the physics respond."
    )
    fig = block_diagram(
        [("Source", "LED / laser diode\nλ, Δλ, resonant cavity"),
         ("Coupling optics", "lenses, GRIN rod\nNA, diffraction limit"),
         ("Optical fiber", "Snell, TIR, evanescent\nattenuation α, dispersion M"),
         ("Detector", "responsivity ρ\nbandwidth, data rate")],
        ["§3.6  §3.2", "§2.1–2.4", "§3.3–3.4"], figsize=(10, 2.2), height=1.0,
        colors=["#dff1f1", "#fdf0d5", "#e6ecfa", "#f6e1dc"])
    show(fig)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Chapter 2 – Optics Review")
        st.markdown(
            "- **2.1 Refraction & Snell's law** – ray diagram, bending, TIR\n"
            "- **2.2 Lenses & GRIN rods** – lensmaker equation, ray-tracing rules, quarter-pitch\n"
            "- **2.3 Numerical aperture** – acceptance angle of a receiver or fiber\n"
            "- **2.4 Diffraction & Gaussian beams** – Airy spot, spot size, coupling")
    with c2:
        st.markdown("#### Chapter 3 – Lightwave Fundamentals")
        st.markdown(
            "- **3.1 EM waves & attenuation** – k, λ, ω, lossy waves\n"
            "- **3.2 Source spectrum** – Δλ ↔ Δf, coherence\n"
            "- **3.3 Material dispersion** – M(λ), pulse spreading\n"
            "- **3.4 Bandwidth & data rate** – RZ / NRZ, eye diagram, rate-length product\n"
            "- **3.5 Polarization** – linear, circular, elliptical, unpolarized\n"
            "- **3.6 Resonant cavities** – longitudinal laser modes\n"
            "- **3.7 Fresnel reflection** – ρ, τ, Brewster angle, Fresnel loss\n"
            "- **3.8 Critical-angle reflection** – TIR, evanescent wave, fiber guidance")
    st.info("**How to use it.** Each module has a *Concepts* tab (equations and notes), a *Simulator* "
            "tab (change inputs, read the results, study the plots), and sometimes a *Scenarios* tab "
            "reproducing the slide examples. When you feel ready, go to **Practice Problems** for "
            "randomised numerical questions with worked solutions.")


# --------------------------------------------------------------------------------------
# CH2 – 2.1  Snell
# --------------------------------------------------------------------------------------
def draw_snell(n1, n2, th_i):
    th_t = snell_theta_t(n1, n2, th_i)
    rp, rs = fresnel(n1, n2, th_i)
    R = float((abs(rp[0]) ** 2 + abs(rs[0]) ** 2) / 2)
    T = 1 - R
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.05, 1.15)
    ax.add_patch(Rectangle((-1.35, -1.05), 1.35, 2.2, fc="#e9f4f5", ec="none", alpha=min(0.9, 0.2 + 0.1 * n1)))
    ax.add_patch(Rectangle((0, -1.05), 1.35, 2.2, fc="#f7e9d6", ec="none", alpha=min(0.9, 0.2 + 0.1 * n2)))
    ax.plot([0, 0], [-1.05, 1.15], color=INK, lw=2.5)
    ax.plot([-1.3, 1.3], [0, 0], color=INK, lw=1, ls="--")
    ax.text(-1.3, 1.05, f"$n_1$ = {n1:g}", fontsize=12, fontweight="bold", va="top")
    ax.text(1.3, 1.05, f"$n_2$ = {n2:g}", fontsize=12, fontweight="bold", va="top", ha="right")
    ax.text(1.3, -0.08, "normal", fontsize=8, ha="right", va="top", color=GREY)
    ti = math.radians(th_i)
    arrow(ax, (-math.cos(ti), -math.sin(ti)), (0, 0), color=BLUE, lw=2.6)
    ax.text(-math.cos(ti) - 0.02, -math.sin(ti) - 0.08, "incident", color=BLUE, ha="center", va="top", fontsize=9)
    lw_r = 1.0 + 4.0 * R
    arrow(ax, (0, 0), (-math.cos(ti), math.sin(ti)), color=RED, lw=lw_r)
    ax.text(-math.cos(ti) - 0.02, math.sin(ti) + 0.06, "reflected", color=RED, ha="center", va="bottom", fontsize=9)
    ax.add_patch(Arc((0, 0), 0.7, 0.7, theta1=180, theta2=180 + th_i, color=BLUE, lw=1.6))
    ax.add_patch(Arc((0, 0), 0.9, 0.9, theta1=180 - th_i, theta2=180, color=RED, lw=1.6))
    ax.text(-0.42 * math.cos(ti / 2), -0.42 * math.sin(ti / 2), r"$\theta_i$", color=BLUE, ha="center", va="center")
    ax.text(-0.55 * math.cos(ti / 2), 0.55 * math.sin(ti / 2), r"$\theta_r$", color=RED, ha="center", va="center")
    if th_t is not None:
        tt = math.radians(th_t)
        arrow(ax, (0, 0), (math.cos(tt), math.sin(tt)), color=GREEN, lw=1.0 + 4.0 * T)
        ax.add_patch(Arc((0, 0), 0.8, 0.8, theta1=0, theta2=th_t, color=GREEN, lw=1.6))
        ax.text(0.5 * math.cos(tt / 2), 0.5 * math.sin(tt / 2), r"$\theta_t$", color=GREEN, ha="center", va="center")
        ax.text(math.cos(tt) - 0.02, math.sin(tt) + 0.06, "transmitted", color=GREEN, ha="right", va="bottom", fontsize=9)
    else:
        ax.text(0.65, 0.4, "TOTAL INTERNAL\nREFLECTION", color=RED, fontsize=12, fontweight="bold", ha="center")
    ax.set_title("Reflection & refraction at a boundary (line width ∝ power)")
    return fig, th_t, R


def page_snell():
    header("2.1  Refraction & Snell's Law", "Chapter 2 · Ray Theory and Applications")
    tab_c, tab_s, tab_m = st.tabs(["Concepts", "Simulator", "Speed of light in materials"])
    with tab_c:
        st.markdown("**Index of refraction** – ratio of the free-space light speed to the speed in the medium (usually n > 1):")
        st.latex(r"n=\frac{c}{v},\qquad c=3\times10^{8}\ \mathrm{m/s}")
        st.markdown("**Law of reflection** and **Snell's law** (eq. 2.3):")
        st.latex(r"\theta_r=\theta_i,\qquad \frac{\sin\theta_t}{\sin\theta_i}=\frac{n_1}{n_2}"
                 r"\;\Rightarrow\; \sin\theta_t=\frac{n_1}{n_2}\sin\theta_i")
        st.markdown(
            "- If **n₁ < n₂**: sin θt < sin θi, so θt < θi – the ray bends **toward** the normal.\n"
            "- If **n₁ > n₂**: θt > θi – the ray bends **away** from the normal.\n"
            "- Only angles between 0° and 90° have physical significance; when (n₁/n₂) sin θi > 1 no real θt exists "
            "and the light is totally reflected (see §3.8).")
    with tab_s:
        cl, cr = st.columns([1, 1.5])
        with cl:
            n1 = material_input("Medium 1 (incident side)", "sn1", "Gas / CO₂ (n = 1)")
            n2 = material_input("Medium 2 (far side)", "sn2", "Glass (n ≈ 1.5)")
            th_i = st.slider("Angle of incidence θᵢ (deg)", 0.0, 89.9, 35.0, 0.1, key="sn_th")
        fig, th_t, R = draw_snell(n1, n2, th_i)
        with cr:
            show(fig)
        thc = critical_angle_deg(n1, n2)
        metrics([("θt", f"{th_t:.2f}°" if th_t is not None else "none (TIR)"),
                 ("Critical angle θc", f"{thc:.2f}°" if thc else "n/a (n₁ ≤ n₂)"),
                 ("Reflected power R", f"{100 * R:.2f} %", "Fresnel, unpolarised light (§3.7)"),
                 ("Transmitted power", f"{100 * (1 - R):.2f} %")])
        if th_t is not None:
            st.success(f"n₁ {'<' if n1 < n2 else '>' if n1 > n2 else '='} n₂ → the transmitted ray is bent "
                       f"{'toward' if n1 < n2 else 'away from' if n1 > n2 else 'not at all relative to'} the normal "
                       f"(θt {'<' if th_t < th_i else '>' if th_t > th_i else '='} θi).")
        else:
            st.warning("(n₁/n₂)·sin θi > 1: Snell's law has no real solution – all power is reflected.")
        # theta_t vs theta_i curve
        ang = np.linspace(0, 89.99, 400)
        s = n1 / n2 * np.sin(np.radians(ang))
        tt = np.where(s <= 1, np.degrees(np.arcsin(np.clip(s, -1, 1))), np.nan)
        fig2, ax = plt.subplots(1, 2, figsize=(10, 3.4))
        ax[0].plot(ang, ang, color=GREY, ls=":", label=r"$\theta_t=\theta_i$")
        ax[0].plot(ang, tt, color=GREEN, lw=2.2, label=r"$\theta_t$ (Snell)")
        ax[0].plot([th_i], [th_t if th_t is not None else np.nan], "o", color=RED, ms=8)
        if thc:
            ax[0].axvline(thc, color=RED, ls="--", lw=1.2)
            ax[0].text(thc + 1, 5, r"$\theta_c$", color=RED)
        ax[0].set_xlabel(r"$\theta_i$ (deg)")
        ax[0].set_ylabel(r"$\theta_t$ (deg)")
        ax[0].set_title("Transmitted vs incident angle")
        ax[0].legend()
        ax[1].plot(ang, np.sin(np.radians(ang)), color=BLUE, lw=2, label=r"$\sin\theta$")
        ax[1].axhline(min(1.0, n1 / n2), color=AMBER, ls="--", label=r"$n_1/n_2$ (slope of sin θt vs sin θi)")
        ax[1].set_xlabel("angle (deg)")
        ax[1].set_ylabel("sin θ")
        ax[1].set_title("The sine function (slide 2.3 plot)")
        ax[1].legend(fontsize=8)
        fig2.tight_layout()
        show(fig2)
    with tab_m:
        st.markdown("Speed of light and wavelength inside the materials listed in the slides "
                    "(v = c/n and λ = λ₀/n).")
        lam0 = st.slider("Free-space wavelength λ₀ (nm)", 400.0, 1700.0, 1550.0, 10.0, key="sn_lam0")
        rows = {"Material": [], "n": [], "v (10⁸ m/s)": [], "λ in medium (nm)": []}
        for name, n in MATERIALS.items():
            if n is None:
                continue
            rows["Material"].append(name.split(" (")[0])
            rows["n"].append(n)
            rows["v (10⁸ m/s)"].append(round(3.0 / n, 3))
            rows["λ in medium (nm)"].append(round(lam0 / n, 1))
        st.table(rows)


# --------------------------------------------------------------------------------------
# CH2 – 2.2  Lenses
# --------------------------------------------------------------------------------------
def draw_thin_lens(f, do, h, D):
    """Ray diagram following the slide rules (1: centre ray, 2: parallel ray, 4: through focal point)."""
    di = f * do / (do - f)          # thin-lens equation
    hi = -h * di / do
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    xl = -max(do, f) * 1.18
    xr_real = di if di > 0 else 2.0 * f
    xr = min(max(xr_real, f) * 1.25, 5.0 * f)
    ymax = max(abs(h), abs(hi), D / 2) * 1.35
    ax.set_xlim(xl, xr)
    ax.set_ylim(-ymax, ymax)
    ax.axhline(0, color=INK, lw=1, ls="--")
    y = np.linspace(-D / 2, D / 2, 60)
    t = 0.012 * (xr - xl) * (1 - (y / (D / 2)) ** 2) + 0.0005
    ax.fill(np.r_[t, -t[::-1]], np.r_[y, y[::-1]], color="#b9d8e6", ec=INK, lw=1.2, zorder=2)
    for xf, lab in ((-f, "F"), (f, "F′")):
        ax.plot([xf], [0], "|", color=INK, ms=10)
        ax.text(xf, -ymax * 0.1, lab, ha="center", va="top", fontsize=9)
    arrow(ax, (-do, 0), (-do, h), color=INK, lw=3)
    ax.text(-do, h * 1.05, "object", ha="center", va="bottom", fontsize=9)
    virtual = di < 0
    rays = []
    # ray 2 : parallel to axis -> through F'
    rays.append(((-do, h), (0, h), (di, hi), BLUE, "ray 2 (parallel)"))
    # ray 1 : through lens centre
    rays.append(((-do, h), (0, 0), (di, hi), RED, "ray 1 (centre)"))
    # ray 4 : through F -> parallel
    rays.append(((-do, h), (0, hi), None, GREEN, "ray 4 (through F)"))
    for p0, pl, img, col, lab in rays:
        ax.plot([p0[0], pl[0]], [p0[1], pl[1]], color=col, lw=1.6, label=lab)
        if img is None:  # horizontal exit ray
            ax.plot([0, xr], [hi, hi], color=col, lw=1.6)
            if virtual:
                ax.plot([xl, 0], [hi, hi], color=col, lw=1.1, ls=":")
        else:
            slope = (img[1] - pl[1]) / (img[0] - pl[0]) if abs(img[0] - pl[0]) > 1e-12 else 0.0
            ax.plot([0, xr], [pl[1], pl[1] + slope * xr], color=col, lw=1.6)
            if virtual:
                ax.plot([xl, 0], [pl[1] + slope * xl, pl[1]], color=col, lw=1.1, ls=":")
    if xl < di < xr:
        arrow(ax, (di, 0), (di, hi), color=PURPLE, lw=3, ls="--" if virtual else "-")
        ax.text(di, hi * 1.08, "virtual image" if virtual else "real image", ha="center",
                va="bottom" if hi > 0 else "top", fontsize=9, color=PURPLE)
    ax.set_xlabel("distance along axis (mm)")
    ax.set_ylabel("height (mm)")
    ax.legend(loc="upper left", fontsize=8, ncol=3)
    ax.set_title("Thin-lens ray tracing (rules 1, 2 and 4 of the slides)")
    return fig, di, hi


def draw_grin(g, Rrod, Lrod, rays):
    z = np.linspace(0, Lrod, 500)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.add_patch(Rectangle((0, -Rrod), Lrod, 2 * Rrod, fc="#eef6f7", ec=INK, lw=1.4, zorder=0))
    P = 2 * math.pi / g
    zz = 0.0
    while zz <= Lrod * 1.5 + 1e-9:
        ax.axvline(zz, color=GREY, lw=0.6, ls=":")
        zz += P / 4
    z_out = np.linspace(Lrod, Lrod + 0.6 * max(Lrod, P / 4), 50)
    for r0, a in rays:
        r = r0 * np.cos(g * z) + a * np.sin(g * z)
        ax.plot(z, np.clip(r, -Rrod, Rrod), color=BLUE, lw=1.6)
        slope = -r0 * g * math.sin(g * Lrod) + a * g * math.cos(g * Lrod)
        r_end = r0 * math.cos(g * Lrod) + a * math.sin(g * Lrod)
        ax.plot(z_out, r_end + slope * (z_out - Lrod), color=RED, lw=1.3, ls="--")
    ax.axvline(Lrod, color=INK, lw=2)
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("r (mm)")
    ax.set_ylim(-Rrod * 1.5, Rrod * 1.5)
    ax.set_title(f"GRIN-rod rays: pitch P = 2π/g = {P:.2f} mm (dotted lines every P/4, dashed = exit rays)", fontsize=10)
    return fig


def page_lenses():
    header("2.2  Lenses & GRIN-Rod Lenses", "Chapter 2 · Lenses – focusing, collimating and imaging")
    tab_c, tab_t, tab_g = st.tabs(["Concepts", "Thin lens & ray tracing", "GRIN-rod lens"])
    with tab_c:
        st.markdown("**Thin-lens focal length** (eq. 2.4) with radii of curvature R₁, R₂ and index n:")
        st.latex(r"\frac1f=(n-1)\left(\frac1{R_1}+\frac1{R_2}\right),\qquad f\text{-number}=\frac{f}{D}")
        st.markdown(
            "- Large *f* ⇒ large radii ⇒ nearly flat lens.  Small *f* ⇒ small radii, and the lens size is limited "
            "by **D_max = 2R**.\n"
            "- Lenses couple sources to fibers and are used in some connectors.\n\n"
            "**Ray-tracing rules**\n"
            "1. A ray through the *centre* of the lens is not deviated.\n"
            "2. A ray *parallel* to the axis passes through the focal point after the lens.\n"
            "3. A ray parallel to a central ray meets it in the focal plane.\n"
            "4. A ray through the *focal point* leaves parallel to the axis.\n\n"
            "**GRIN-rod lens:** n = n(r); rays follow sinusoidal paths of period P (the *pitch*). "
            "A **quarter-pitch** rod collimates light from a fiber and focuses a collimated beam onto a fiber.")
        st.caption("Imaging relation used in the simulator: 1/f = 1/dₒ + 1/dᵢ, magnification m = −dᵢ/dₒ.")
    with tab_t:
        cl, cr = st.columns([1, 2.3])
        with cl:
            n = st.slider("Lens index n", 1.3, 2.5, 1.5, 0.01, key="ln_n")
            r1 = st.number_input("R₁ (mm)", 5.0, 500.0, 40.0, 5.0, key="ln_r1")
            r2 = st.number_input("R₂ (mm)", 5.0, 500.0, 40.0, 5.0, key="ln_r2")
            D = st.slider("Lens diameter D (mm)", 2.0, 100.0, 20.0, 1.0, key="ln_D")
            k = st.slider("Object distance dₒ / f", 0.3, 6.0, 2.5, 0.05, key="ln_k")
            h = st.slider("Object height (mm)", 1.0, 20.0, 8.0, 0.5, key="ln_h")
        f = lens_focal_length(n, r1, r2)
        do = k * f
        with cr:
            if abs(k - 1.0) < 1e-6:
                st.warning("dₒ = f: the image forms at infinity (emerging rays are collimated). Move the slider slightly.")
            else:
                fig, di, hi = draw_thin_lens(f, do, h, D)
                show(fig)
        if abs(k - 1.0) >= 1e-6:
            metrics([("Focal length f", f"{f:.2f} mm"), ("f-number f/D", f"{f / D:.2f}"),
                     ("Image distance dᵢ", f"{di:.1f} mm"), ("Magnification m", f"{-di / do:.2f}"),
                     ("Image", "virtual, upright" if di < 0 else "real, inverted")])
        if D > 2 * min(r1, r2):
            st.warning(f"D = {D:g} mm exceeds D_max = 2R = {2 * min(r1, r2):g} mm for this lens – not physically realisable.")
        st.caption("Check: for R₁ = R₂ = R, f = R / (2(n−1)); n = 1.5 gives f = R.")
    with tab_g:
        cl, cr = st.columns([1, 2.6])
        with cl:
            g = st.slider("Gradient constant g (1/mm)", 0.2, 1.5, 0.5, 0.05, key="gr_g")
            Rrod = st.slider("Rod radius (mm)", 0.3, 2.0, 1.0, 0.1, key="gr_R")
            P = 2 * math.pi / g
            fr = {"1/4": 0.25, "1/2": 0.5, "3/4": 0.75, "1": 1.0}
            frac = st.selectbox("Rod length (fraction of pitch)", list(fr) + ["Custom"], key="gr_frac")
            if frac == "Custom":
                Lrod = st.slider("Rod length (mm)", 1.0, 25.0, 5.0, 0.1, key="gr_L")
            else:
                Lrod = P * fr[frac]
            mode = st.radio("Input beam", ["Point source on axis (collimation)", "Parallel beam (focusing)"], key="gr_mode")
        if mode.startswith("Point"):
            slopes = np.array([-0.5, -0.25, 0.0, 0.25, 0.5]) * Rrod * g * 0.9
            rays = [(0.0, sl / g) for sl in slopes]      # r0 = 0, r'(0) = sl  ->  r = (sl/g) sin(gz)
        else:
            rays = [(r0, 0.0) for r0 in np.linspace(-0.8, 0.8, 5) * Rrod]
        with cr:
            show(draw_grin(g, Rrod, Lrod, rays))
        r_end = np.array([r0 * math.cos(g * Lrod) + a * math.sin(g * Lrod) for r0, a in rays])
        s_end = np.array([-r0 * g * math.sin(g * Lrod) + a * g * math.cos(g * Lrod) for r0, a in rays])
        metrics([("Pitch P", f"{P:.2f} mm"), ("Rod length / P", f"{Lrod / P:.3f}"),
                 ("max |r| at exit", f"{np.max(np.abs(r_end)):.3f} mm"), ("max |slope| at exit", f"{np.max(np.abs(s_end)):.4f}")])
        if mode.startswith("Point") and np.max(np.abs(s_end)) < 1e-3:
            st.success("All exit rays are parallel to the axis → the quarter-pitch rod **collimates** the fiber output.")
        elif (not mode.startswith("Point")) and np.max(np.abs(r_end)) < 1e-3:
            st.success("All rays converge to a point on the exit face → the quarter-pitch rod **focuses** onto the fiber.")
        else:
            st.info("Try length = 1/4 pitch. In general r(z) = r₀ cos(gz) + (r′₀/g) sin(gz).")


# --------------------------------------------------------------------------------------
# CH2 – 2.3  Numerical aperture
# --------------------------------------------------------------------------------------
def draw_receiver(f, d, D, phi_deg):
    phi = math.radians(phi_deg)
    s = math.tan(phi)
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    ymax = max(D / 2, d / 2, f * s) * 1.35
    ax.set_xlim(-1.1 * f, 1.25 * f)
    ax.set_ylim(-ymax, ymax)
    ax.axhline(0, color=INK, ls="--", lw=1)
    y = np.linspace(-D / 2, D / 2, 60)
    t = 0.01 * f * (1 - (y / (D / 2)) ** 2) + 0.0005
    ax.fill(np.r_[t, -t[::-1]], np.r_[y, y[::-1]], color="#b9d8e6", ec=INK, lw=1.2, zorder=2)
    ax.add_patch(Rectangle((f, -d / 2), 0.03 * f, d, fc=AMBER, ec=INK, zorder=3))
    ax.text(f + 0.05 * f, d / 2, "photodetector\n(diameter d)", fontsize=8, va="bottom")
    spot = f * s
    hit = abs(spot) <= d / 2 + 1e-12
    col = GREEN if hit else RED
    for yl in (D / 2, 0.0, -D / 2):
        x0 = -1.05 * f
        ax.plot([x0, 0], [yl + x0 * s, yl], color=BLUE, lw=1.4)
        ax.plot([0, f], [yl, spot], color=col, lw=1.6)
    ax.plot([f], [spot], "o", color=col, ms=7, zorder=5)
    ax.text(-0.98 * f, ymax * 0.85, f"incoming rays at φ = {phi_deg:.2f}° to the axis", color=BLUE, fontsize=9)
    ax.set_xlabel("distance along axis")
    ax.set_yticks([])
    ax.set_title("Receiver: focused spot lands at y = f·tan φ  →  " + ("inside the detector" if hit else "OUTSIDE the detector"))
    return fig, hit, spot


def page_na():
    header("2.3  Numerical Aperture", "Chapter 2 · Acceptance angle of a receiver and of a fiber")
    tab_c, tab_r, tab_t, tab_f = st.tabs(["Concepts", "Receiver simulator", "NA ↔ angle table", "Fiber NA"])
    with tab_c:
        st.latex(r"\tan\theta=\frac{d}{2f}\quad(2.11),\qquad NA\equiv n_o\sin\theta")
        st.markdown(
            "*d* is the photodetector diameter, *f* the lens focal length, *n₀* the index of the surrounding medium "
            "(usually 1) and θ the **acceptance-cone half-angle**. Light arriving beyond θ is not focused onto the detector.\n\n"
            "For small angles sin θ ≈ tan θ, hence NA ≈ d/(2f) when n₀ = 1 – this is the approximation used in the slide example "
            "(f = 10 cm, d = 1 cm → NA = 0.05, θ ≈ 2.87°). Typical fibers have **NA = 0.1 … 0.5**.")
    with tab_r:
        cl, cr = st.columns([1, 2.4])
        with cl:
            f = st.number_input("Focal length f (cm)", 1.0, 100.0, 10.0, 1.0, key="na_f")
            d = st.number_input("Detector diameter d (cm)", 0.1, 20.0, 1.0, 0.1, key="na_d")
            D = st.slider("Lens diameter D (cm, drawing only)", 0.5, 40.0, 4.0, 0.5, key="na_D")
            n0 = st.number_input("Surrounding index n₀", 1.0, 2.0, 1.0, 0.01, key="na_n0")
            th = math.degrees(math.atan(d / (2 * f)))
            phi = st.slider("Beam angle φ off the axis (deg)", 0.0, max(10.0, 2.5 * th), min(1.0, th * 0.5), 0.05, key="na_phi")
        with cr:
            fig, hit, spot = draw_receiver(f, d, D, phi)
            show(fig)
        na_exact = n0 * math.sin(math.radians(th))
        metrics([("Acceptance half-angle θ", f"{th:.3f}°", "θ = arctan(d / 2f)"),
                 ("NA = n₀ sin θ", f"{na_exact:.4f}"),
                 ("Small-angle NA ≈ d/2f", f"{n0 * d / (2 * f):.4f}"),
                 ("Full acceptance cone 2θ", f"{2 * th:.3f}°")])
        if hit:
            st.success(f"Spot displacement f·tan φ = {spot:.3f} cm ≤ d/2 = {d / 2:.3f} cm → the beam is collected.")
        else:
            st.error(f"Spot displacement f·tan φ = {spot:.3f} cm > d/2 = {d / 2:.3f} cm → the beam misses the detector.")
    with tab_t:
        na_list = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        st.markdown("Reproduces the slide table (n₀ = 1) using the exact relation θ = arcsin(NA/n₀):")
        st.table({"NA": na_list.tolist(), "θ (deg)": [round(math.degrees(math.asin(x)), 1) for x in na_list],
                  "2θ (deg)": [round(2 * math.degrees(math.asin(x)), 1) for x in na_list]})
        cl, cr = st.columns([1, 2])
        with cl:
            na = st.slider("NA", 0.05, 0.9, 0.3, 0.01, key="na_tab")
            n0b = st.number_input("n₀", 1.0, 2.0, 1.0, 0.01, key="na_n0b")
        th_a = math.degrees(math.asin(min(1.0, na / n0b)))
        with cr:
            fig, ax = plt.subplots(figsize=(7.5, 3.0))
            ax.add_patch(Rectangle((0, -0.5), 6, 1.0, fc="#eef6f7", ec=INK, lw=1.3))
            ax.add_patch(Rectangle((-0.35, -0.12), 0.35, 0.24, fc=GREEN, ec=INK))
            L = 6.0
            tn = math.tan(math.radians(th_a))
            ax.plot([0, -1.6], [0, -1.6 * tn], color=GREY, lw=1)
            ax.plot([0, -1.6], [0, 1.6 * tn], color=GREY, lw=1)
            ax.add_patch(Polygon([[0, 0], [-1.6, 1.6 * tn], [-1.6, -1.6 * tn]], fc="#c9ecd6", alpha=0.5, ec="none"))
            ax.text(-1.55, 0, f"acceptance cone\n2θ = {2 * th_a:.1f}°", ha="left", va="center", fontsize=9)
            ax.set_xlim(-2.0, 6.2)
            ax.set_ylim(-1.3, 1.3)
            ax.axis("off")
            ax.set_title(f"NA = {na:.2f} → θ = {th_a:.2f}°")
            show(fig)
        thr = np.linspace(0, 89, 200)
        fig2, ax2 = plt.subplots(figsize=(6.5, 2.8))
        ax2.plot(np.sin(np.radians(thr)), thr, color=BLUE, lw=2, label="θ = arcsin(NA)")
        ax2.axvspan(0.1, 0.5, color=AMBER, alpha=0.2, label="typical fiber NA")
        ax2.set_xlabel("NA (n₀ = 1)")
        ax2.set_ylabel("θ (deg)")
        ax2.legend()
        show(fig2)
    with tab_f:
        cl, cr = st.columns([1, 2])
        with cl:
            n1 = st.number_input("Core index n₁", 1.3, 2.0, 1.48, 0.005, format="%.3f", key="nf_n1")
            n2 = st.number_input("Cladding index n₂", 1.2, 2.0, 1.46, 0.005, format="%.3f", key="nf_n2")
            n0f = st.number_input("Outside index n₀", 1.0, 1.6, 1.0, 0.01, key="nf_n0")
        if n2 >= n1:
            st.error("The cladding index must be lower than the core index for guidance.")
        else:
            na_f = math.sqrt(n1 ** 2 - n2 ** 2)
            with cr:
                metrics([("NA = √(n₁²−n₂²)", f"{na_f:.4f}"),
                         ("Acceptance half-angle", f"{math.degrees(math.asin(min(1, na_f / n0f))):.2f}°"),
                         ("Δ = (n₁−n₂)/n₁", f"{100 * (n1 - n2) / n1:.2f} %")], cols=3)
                st.caption("Derived from total internal reflection at the core/cladding boundary (see 3.8 for the ray demonstration).")
                if not 0.1 <= na_f <= 0.5:
                    st.info("This NA lies outside the 'typical' 0.1–0.5 range quoted in the slides.")


# --------------------------------------------------------------------------------------
# CH2 – 2.4  Diffraction & Gaussian
# --------------------------------------------------------------------------------------
def page_diffraction():
    header("2.4  Diffraction & Gaussian Beams", "Chapter 2 · Limits of ray theory")
    tab_c, tab_a, tab_g, tab_k = st.tabs(["Concepts", "Diffraction-limited spot", "Gaussian beam", "Lens-to-fiber coupling"])
    with tab_c:
        st.markdown("**Diffraction** is the deviation from the prediction of ray theory. A lens focusing a *uniform* beam "
                    "of diameter D does not give a point but a central spot with surrounding rings:")
        st.latex(r"d=\frac{2.44\,\lambda f}{D}\qquad(2.14)")
        st.markdown("**Gaussian intensity distribution** (eq. 2.5) – emitted by most lasers and the field pattern inside a single-mode fiber:")
        st.latex(r"I=I_o\,e^{-2r^2/w^2},\qquad r=w\Rightarrow \frac{I}{I_o}=e^{-2}=0.135")
        st.markdown("*w* is the **spot size**; the spot **diameter is 2w**.")
    with tab_a:
        cl, cr = st.columns([1, 2])
        with cl:
            lam = st.slider("Wavelength λ (µm)", 0.4, 1.7, 1.0, 0.01, key="df_lam")
            ratio = st.slider("f / D  (lens f-number)", 0.5, 10.0, 2.0, 0.1, key="df_fD")
            Dmm = st.slider("Lens diameter D (mm)", 1.0, 50.0, 10.0, 0.5, key="df_D")
        f_mm = ratio * Dmm
        d = spot_diameter(lam, f_mm, Dmm)            # µm (λ in µm, f/D dimensionless)
        r_airy = d / 2
        with cr:
            metrics([("Central-spot diameter d", f"{d:.3f} µm"), ("f", f"{f_mm:.1f} mm"),
                     ("Encircled power in d", f"{100 * float(airy_encircled(3.8317)[0]):.1f} %")])
            r = np.linspace(0, 3.2 * r_airy, 500)
            x = 3.8317 * r / r_airy
            I1 = airy_intensity(x)
            fig, ax = plt.subplots(1, 2, figsize=(10, 3.5), gridspec_kw={"width_ratios": [1.3, 1]})
            ax[0].plot(np.r_[-r[::-1], r], np.r_[I1[::-1], I1], color=BLUE, lw=2)
            ax[0].axvspan(-r_airy, r_airy, color=AMBER, alpha=0.2, label=f"d = {d:.2f} µm")
            ax[0].set_xlabel("radial position in focal plane (µm)")
            ax[0].set_ylabel("I / I₀")
            ax[0].set_title("Airy pattern  (2J₁(x)/x)²")
            ax[0].legend()
            gx = np.linspace(-3.2 * r_airy, 3.2 * r_airy, 241)
            rr = np.hypot(gx[None, :], gx[:, None])
            img = np.interp(rr, r, I1, right=0.0)
            ax[1].imshow(img ** 0.35, extent=[gx[0], gx[-1], gx[0], gx[-1]], cmap="magma", origin="lower")
            ax[1].set_title("focal-plane image (γ-stretched)")
            ax[1].set_xlabel("x (µm)")
            ax[1].grid(False)
            fig.tight_layout()
            show(fig)
        st.caption("Slide example: f = 2D, λ = 1 µm → d = 2.44·1·2 = 4.88 µm (set f/D = 2, λ = 1.0).")
        if st.checkbox("Show d versus f/D for several wavelengths", key="df_plot"):
            fD = np.linspace(0.5, 10, 100)
            fig2, ax2 = plt.subplots(figsize=(7, 3))
            for lm in (0.82, 1.3, 1.55):
                ax2.plot(fD, 2.44 * lm * fD, label=f"λ = {lm} µm")
            ax2.set_xlabel("f / D")
            ax2.set_ylabel("d (µm)")
            ax2.legend()
            show(fig2)
    with tab_g:
        cl, cr = st.columns([1, 2])
        with cl:
            w = st.slider("Spot size w (µm)", 1.0, 20.0, 5.0, 0.1, key="ga_w")
            rq = st.slider("Query radius r / w", 0.0, 3.0, 1.0, 0.05, key="ga_r")
        Iq = math.exp(-2 * rq ** 2)
        Pq = 1 - Iq
        with cr:
            metrics([("Spot diameter 2w", f"{2 * w:.2f} µm"), ("I(r)/I₀", f"{100 * Iq:.2f} %"),
                     ("Power inside radius r", f"{100 * Pq:.2f} %", "P(r) = 1 − exp(−2r²/w²)")])
        rw = np.linspace(-3, 3, 400)
        fig, ax = plt.subplots(1, 3, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1.2, 1, 1]})
        ax[0].plot(rw, np.exp(-2 * rw ** 2), color=BLUE, lw=2)
        ax[0].axhline(math.exp(-2), color=INK, lw=1.3)
        ax[0].plot([rq, -rq], [Iq, Iq], "o", color=RED)
        ax[0].text(1.6, math.exp(-2) + 0.03, "e⁻² = 0.135", fontsize=8)
        ax[0].set_xlabel("r / w")
        ax[0].set_ylabel("I / I₀")
        ax[0].set_title("Gaussian intensity (slide plot)")
        gx = np.linspace(-3, 3, 200)
        ax[1].imshow(np.exp(-2 * (gx[None, :] ** 2 + gx[:, None] ** 2)), extent=[-3, 3, -3, 3], cmap="inferno", origin="lower")
        ax[1].add_patch(plt.Circle((0, 0), 1, fill=False, ec="white", ls="--"))
        ax[1].set_title("beam cross-section (circle: r = w)")
        ax[1].grid(False)
        rr = np.linspace(0, 3, 200)
        ax[2].plot(rr, 1 - np.exp(-2 * rr ** 2), color=GREEN, lw=2)
        ax[2].plot([rq], [Pq], "o", color=RED)
        ax[2].axhline(0.865, color=GREY, ls=":")
        ax[2].set_xlabel("r / w")
        ax[2].set_title("encircled power")
        fig.tight_layout()
        show(fig)
    with tab_k:
        st.markdown("Fraction of light captured by a fiber core of radius *a* placed at the focus, for a focused "
                    "**uniform** beam (Airy) compared with a **Gaussian** beam of spot size *w*.")
        cl, cr = st.columns([1, 2])
        with cl:
            lam = st.slider("λ (µm)", 0.6, 1.7, 1.31, 0.01, key="ck_lam")
            fD = st.slider("f / D", 1.0, 12.0, 4.0, 0.1, key="ck_fD")
            w = st.slider("Gaussian spot size w (µm)", 1.0, 20.0, 4.5, 0.1, key="ck_w")
            a = st.slider("Core radius a (µm)", 1.0, 30.0, 4.5, 0.1, key="ck_a")
        x_a = math.pi * a / (lam * fD)          # x = π D a /(λ f)
        eta_airy = float(airy_encircled(x_a)[0])
        eta_g = 1 - math.exp(-2 * a ** 2 / w ** 2)
        with cr:
            metrics([("Airy (uniform beam) in core", f"{100 * eta_airy:.1f} %"),
                     ("Gaussian in core", f"{100 * eta_g:.1f} %"),
                     ("Airy spot d = 2.44 λ f/D", f"{2.44 * lam * fD:.2f} µm")])
            aa = np.linspace(0.2, 30, 200)
            fig, ax = plt.subplots(figsize=(7.5, 3.2))
            ax.plot(aa, airy_encircled(math.pi * aa / (lam * fD)), label="Airy (uniform beam)", color=BLUE, lw=2)
            ax.plot(aa, 1 - np.exp(-2 * aa ** 2 / w ** 2), label="Gaussian", color=RED, lw=2)
            ax.plot([a], [eta_airy], "o", color=BLUE)
            ax.plot([a], [eta_g], "o", color=RED)
            ax.set_xlabel("core radius a (µm)")
            ax.set_ylabel("power fraction inside a")
            ax.legend()
            show(fig)
        st.caption("Simplified model: perfectly aligned, geometric overlap of the intensity pattern with the core area "
                   "(it ignores mode-field matching, NA limits and Fresnel losses).")


# --------------------------------------------------------------------------------------
# CH3 – 3.1  EM waves
# --------------------------------------------------------------------------------------
def page_em():
    header("3.1  Electromagnetic Waves", "Chapter 3 · Plane waves, propagation factor, power and loss")
    tab_c, tab_s, tab_l = st.tabs(["Concepts", "Wave simulator", "Attenuation & dB"])
    with tab_c:
        st.latex(r"E=E_o\sin(\omega t-kz)\ (3.1),\qquad k=\frac{\omega}{v}=k_o n=\frac{2\pi}{\lambda}\ (3.6)")
        st.latex(r"\lambda=\frac{\lambda_o}{n}\ (3.7),\qquad f=\frac{v}{\lambda},\ \ \omega=2\pi f,\ \ \phi=\omega t-kz")
        st.latex(r"S=\frac{E^2}{\sqrt{\mu/\varepsilon}}\ \text{(irradiance, W/m}^2),\qquad I\equiv E^2\ \text{(intensity)}")
        st.latex(r"E=E_o e^{-\alpha z}\sin(\omega t-kz)\ (3.8)\ \text{– wave in a lossy medium}")
        st.markdown("Frequency and phase are unchanged by loss; only the **amplitude** E₀e^{−αz} decays. "
                    "Power ∝ E², so power decays as e^{−2αz}.")
    with tab_s:
        cl, cr = st.columns([1, 2.4])
        with cl:
            lam0 = st.slider("Free-space wavelength λ₀ (nm)", 400.0, 1700.0, 1550.0, 10.0, key="em_lam")
            n = st.slider("Refractive index n", 1.0, 4.0, 1.45, 0.01, key="em_n")
            E0 = st.number_input("Amplitude E₀ (V/m)", 1.0, 1e6, 1000.0, 100.0, key="em_E0")
            al = st.slider("Attenuation per wavelength α·λ (Np/λ)", 0.0, 0.3, 0.05, 0.005, key="em_al")
            nlam = st.slider("Show z from 0 to … wavelengths", 2, 12, 6, 1, key="em_nl")
            tau = st.slider("Time t₁ (fraction of period)", 0.0, 1.0, 0.0, 0.05, key="em_tau")
            dt = st.slider("Spacing to t₂, t₃ (fraction of period)", 0.05, 0.4, 0.125, 0.025, key="em_dt")
        f = C0 / (lam0 * 1e-9)
        w = 2 * math.pi * f
        v = C0 / n
        k = w / v
        lam = lam0 / n
        eta = ETA0 / n
        with cr:
            zt = np.linspace(0, nlam, 800)
            fig, ax = plt.subplots(1, 2, figsize=(11, 3.7), gridspec_kw={"width_ratios": [2.2, 1]})
            env = np.exp(-al * zt)
            for i, (col, name) in enumerate(zip((BLUE, RED, GREEN), ("t₁", "t₂", "t₃"))):
                tt = tau + i * dt
                ax[0].plot(zt, env * np.sin(2 * math.pi * (tt - zt)), color=col, lw=1.8, label=f"{name} = {tt:.3g} T")
            ax[0].plot(zt, env, "k:", lw=1)
            ax[0].plot(zt, -env, "k:", lw=1)
            ax[0].set_xlabel("z / λ (λ = wavelength in the medium)")
            ax[0].set_ylabel("E / E₀")
            ax[0].set_title("Wave travelling in +z (snapshots at t₁ < t₂ < t₃)")
            ax[0].legend(loc="upper right", fontsize=8, ncol=3)
            tt = np.linspace(0, 3, 500)
            ax[1].plot(tt, np.sin(2 * math.pi * tt), color=BLUE)
            ax[1].set_xlabel("t / T at z = 0")
            ax[1].set_title("time waveform")
            fig.tight_layout()
            show(fig)
        metrics([("Frequency f", fmt_rate(f)), ("ω = 2πf", f"{w:.3e} rad/s"), ("Speed v = c/n", f"{v:.3e} m/s"),
                 ("λ in medium", f"{lam:.1f} nm"), ("k = 2π/λ", f"{k * 1e-6:.3f} rad/µm")], cols=5)
        S = E0 ** 2 / eta
        st.markdown(f"**Power:** intrinsic impedance η = η₀/n = {eta:.1f} Ω, so the peak irradiance S = E₀²/η = "
                    f"**{S:.3g} W/m²** (I = E² = {E0 ** 2:.3g} in the slide's normalised units). "
                    f"After one wavelength the field has dropped to {100 * math.exp(-al):.1f} % and the power to {100 * math.exp(-2 * al):.1f} %.")
    with tab_l:
        st.markdown("Convert between the slide's **field attenuation coefficient α** (Np/m) and the fiber-industry figure in **dB/km**.")
        cl, cr = st.columns(2)
        with cl:
            adb = st.number_input("Fiber loss (dB/km)", 0.01, 100.0, 0.2, 0.05, key="em_adb")
            L = st.number_input("Length L (km)", 0.1, 500.0, 50.0, 1.0, key="em_L")
        alpha = adb / (20 * LOG10E) / 1000.0             # field, Np/m
        with cr:
            metrics([("α (field, Np/m)", f"{alpha:.3e}"), ("Field remaining", f"{100 * math.exp(-alpha * L * 1000):.2f} %"),
                     ("Power remaining", f"{100 * 10 ** (-adb * L / 10):.3f} %")], cols=3)
        zz = np.linspace(0, L, 200)
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(zz, 100 * 10 ** (-adb * zz / 10), label="power (∝ E²)", color=BLUE, lw=2)
        ax.plot(zz, 100 * np.exp(-alpha * zz * 1000), label="field amplitude (E₀e^{−αz})", color=RED, lw=2)
        ax.set_xlabel("distance (km)")
        ax.set_ylabel("% of launched")
        ax.legend()
        show(fig)


# --------------------------------------------------------------------------------------
# CH3 – 3.2  Source spectrum
# --------------------------------------------------------------------------------------
def page_spectrum():
    header("3.2  Source Spectrum & Coherence", "Chapter 3 · Spectral width Δλ and frequency bandwidth Δf")
    tab_c, tab_s = st.tabs(["Concepts", "Simulator"])
    with tab_c:
        st.latex(r"\frac{\Delta f}{f}=\frac{\Delta\lambda}{\lambda}\quad(3.9)\qquad\Longleftrightarrow\qquad \Delta f=\frac{c\,\Delta\lambda}{\lambda^2}")
        st.markdown(
            "Proof idea (slides): Δf = c(λ₁ − λ₂)/(λ₁λ₂) = cΔλ/(λ₁λ₂); with the mean wavelength λ = √(λ₁λ₂) this becomes cΔλ/λ².\n\n"
            "If Δλ = 0 the source is perfectly coherent (monochromatic). Laser diodes are more coherent than LEDs but not perfectly so. "
            "Source bandwidth will later limit the information capacity of the fiber (§3.3, §3.4).")
        st.table({"Source": ["LED", "Laser diode", "Nd:YAG laser", "HeNe laser"],
                  "Spectral width Δλ (nm)": ["21 – 100", "1 – 5", "0.1", "0.002"]})
    with tab_s:
        cl, cr = st.columns([1, 2])
        with cl:
            lam = st.slider("Centre wavelength λ (nm)", 400.0, 1700.0, 820.0, 5.0, key="sp_lam")
            preset = st.selectbox("Source preset", ["Custom", "LED (60 nm)", "Laser diode (3 nm)", "Nd:YAG (0.1 nm)", "HeNe (0.002 nm)"], key="sp_pre")
            pre = {"LED (60 nm)": 60.0, "Laser diode (3 nm)": 3.0, "Nd:YAG (0.1 nm)": 0.1, "HeNe (0.002 nm)": 0.002}
            if preset == "Custom":
                dlam = st.number_input("Spectral width Δλ (nm)", 0.001, 200.0, 30.0, 1.0, format="%.3f", key="sp_dl")
            else:
                dlam = pre[preset]
                st.write(f"Δλ = {dlam} nm")
        l1, l2 = lam - dlam / 2, lam + dlam / 2
        df_exact = C0 * (1 / (l1 * 1e-9) - 1 / (l2 * 1e-9))
        df_approx = C0 * dlam * 1e-9 / (lam * 1e-9) ** 2
        f0 = C0 / (lam * 1e-9)
        with cr:
            metrics([("Δλ/λ (relative bandwidth)", f"{100 * dlam / lam:.3f} %"), ("Centre frequency f", fmt_rate(f0)),
                     ("Δf = cΔλ/λ²", fmt_rate(df_approx)), ("Δf exact", fmt_rate(df_exact))])
            st.caption("Slide example: λ = 0.82 µm, Δλ = 30 nm → 30/820 = 0.037 → 3.7 % bandwidth.")
        x = np.linspace(-2.5, 2.5, 500) * dlam
        sig = dlam / 2.3548
        spec = np.exp(-0.5 * (x / sig) ** 2)
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
        ax[0].plot(lam + x, spec, color=BLUE, lw=2)
        ax[0].axhline(0.5, color=GREY, ls=":")
        ax[0].axvspan(l1, l2, color=AMBER, alpha=0.25, label=f"Δλ = {dlam:g} nm")
        ax[0].set_xlabel("wavelength (nm)")
        ax[0].set_ylabel("normalised power")
        ax[0].set_title("Emission spectrum vs wavelength")
        ax[0].legend()
        lam_axis = lam + x
        ok = lam_axis > 0
        ax[1].plot(C0 / (lam_axis[ok] * 1e-9) * 1e-12, spec[ok], color=RED, lw=2)
        ax[1].axvspan(C0 / (l2 * 1e-9) * 1e-12, C0 / (l1 * 1e-9) * 1e-12, color=AMBER, alpha=0.25, label=f"Δf = {fmt_rate(df_approx)}")
        ax[1].set_xlabel("frequency (THz)")
        ax[1].set_title("Same spectrum vs frequency")
        ax[1].legend()
        fig.tight_layout()
        show(fig)
        names = ["LED", "Laser diode", "Nd:YAG", "HeNe"]
        widths = [60, 3, 0.1, 0.002]
        fig2, ax2 = plt.subplots(figsize=(8, 2.6))
        ax2.barh(names, widths, color=[RED, AMBER, GREEN, BLUE])
        ax2.set_xscale("log")
        ax2.axvline(dlam, color=INK, ls="--")
        ax2.set_xlabel("Δλ (nm, log scale) – dashed line: your source")
        ax2.grid(True, axis="x")
        show(fig2)
        Lc = (lam * 1e-9) ** 2 / (dlam * 1e-9)
        st.caption(f"Extension (not in the slides): the coherence length ≈ λ²/Δλ = {Lc * 1e3:.3g} mm – the smaller Δλ, the longer the coherence.")


# --------------------------------------------------------------------------------------
# CH3 – 3.3  Material dispersion
# --------------------------------------------------------------------------------------
def pulse_output(T_ps, dlam, M, L_km, n_lam=61):
    """Sum of spectral components, each delayed by  -M*(lambda - lambda_c)*L  (ps)."""
    xs = np.linspace(-1.5, 1.5, n_lam) * dlam
    wts = np.exp(-4 * math.log(2) * (xs / dlam) ** 2) if dlam > 0 else np.ones(1)
    delays = -M * xs * L_km
    span = 2.5 * max(T_ps, abs(M) * dlam * L_km) + 3 * T_ps
    t = np.linspace(-span, span, 2000)
    sig = T_ps / 2.3548
    out = np.zeros_like(t)
    for w_, d_ in zip(wts, delays):
        out += w_ * np.exp(-0.5 * ((t - d_) / sig) ** 2)
    out /= wts.sum()
    inp = np.exp(-0.5 * (t / sig) ** 2)
    return t, inp, out, xs, delays


def fwhm(t, y):
    half = y.max() / 2
    idx = np.where(y >= half)[0]
    return t[idx[-1]] - t[idx[0]]


def page_dispersion():
    header("3.3  Material Dispersion & Pulse Spreading", "Chapter 3 · Wavelength-dependent velocity")
    tab_c, tab_m, tab_p = st.tabs(["Concepts", "M(λ) and pulse-spread calculator", "Pulse broadening simulation"])
    with tab_c:
        st.markdown("Because n depends on λ in glass, different wavelengths travel at different speeds (**dispersion**). "
                    "A pulse from a source with linewidth Δλ therefore spreads by Δτ, growing with **path length** and **source spectral width**.")
        st.latex(r"\Delta\!\left(\frac{\tau}{L}\right)=\left[\frac{d(\tau/L)}{d\lambda}\right]\Delta\lambda\ (3.12),\qquad "
                 r"\left(\frac{\tau}{L}\right)'=-\frac{\lambda}{c}\,n''\ (3.13)")
        st.latex(r"M\equiv\frac{\lambda}{c}\,n'' \quad\Rightarrow\quad \Delta\!\left(\frac{\tau}{L}\right)=-M\,\Delta\lambda\ (3.14),\qquad \Delta\tau=L\,\Delta\!\left(\frac{\tau}{L}\right)")
        st.latex(r"M(\lambda)=\frac{M_o}{4}\left(\lambda-\frac{\lambda_o^{4}}{\lambda^{3}}\right),\quad M_o=-0.095\ \mathrm{ps/(nm^{2}\,km)},\ \lambda_o\approx1300\ \mathrm{nm}\quad(1200\text{–}1600\ \mathrm{nm})")
        st.markdown(
            "- **M > 0** (λ < 1.3 µm): longer wavelength arrives **first**.  **M < 0** (λ > 1.3 µm): shorter wavelength arrives first.\n"
            "- At λ ≈ 1.3 µm, M = 0 → no material-dispersion spreading.\n"
            "- A **soliton** exploits the intensity-dependent index of glass to slow the leading edge and speed the trailing edge, so a properly shaped pulse travels without spreading.")
        lam = np.linspace(1200, 1600, 200)
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.plot(lam, material_dispersion_model(lam), color=RED, lw=2, label="model (1200–1600 nm)")
        ax.plot([820, 1300, 1550], [110, 0, -20], "o", color=INK, label="slide plot points")
        ax.axhline(0, color=INK, lw=1)
        ax.set_xlim(780, 1620)
        ax.set_xlabel("wavelength (nm)")
        ax.set_ylabel("M (ps / nm·km)")
        ax.legend()
        show(fig)
    def get_inputs(prefix, lam_default, dl_default, L_default):
        cl, cr = st.columns([1, 2])
        with cl:
            lam = st.number_input("Wavelength λ (nm)", 700.0, 1700.0, lam_default, 10.0, key=prefix + "lam")
            dl = st.number_input("Source Δλ (nm)", 0.01, 200.0, dl_default, 1.0, key=prefix + "dl")
            L = st.number_input("Fiber length L (km)", 0.1, 500.0, L_default, 1.0, key=prefix + "L")
            inside = 1200.0 <= lam <= 1600.0
            mode = st.radio("Dispersion parameter M", ["Model formula (1200–1600 nm)", "Enter M manually"],
                            index=0 if inside else 1, key=prefix + "mode")
            if mode.startswith("Model") and inside:
                M = float(material_dispersion_model(lam))
                st.write(f"M({lam:g} nm) = **{M:.2f} ps/(nm·km)**")
            else:
                if mode.startswith("Model"):
                    st.warning("Formula valid only for 1200–1600 nm – enter M manually.")
                M = st.number_input("M (ps/(nm·km))", -200.0, 200.0, 110.0 if lam < 1200 else -15.0, 1.0, key=prefix + "M")
        return lam, dl, L, M, cr
    with tab_m:
        sc = st.selectbox("Load a slide example", ["— none —", "LED 0.82 µm, Δλ = 20 nm, L = 10 km (M = 110)",
                                                   "1.5 µm, Δλ = 50 nm, L = 10 km (M = −15)",
                                                   "0.82 µm, Δλ = 1 nm, L = 10 km (M = 110)",
                                                   "1.5 µm, Δλ = 1 nm, L = 10 km (M = −15)"], key="ds_sc")
        presets = {"LED 0.82 µm, Δλ = 20 nm, L = 10 km (M = 110)": (820.0, 20.0, 10.0, 110.0),
                   "1.5 µm, Δλ = 50 nm, L = 10 km (M = −15)": (1500.0, 50.0, 10.0, -15.0),
                   "0.82 µm, Δλ = 1 nm, L = 10 km (M = 110)": (820.0, 1.0, 10.0, 110.0),
                   "1.5 µm, Δλ = 1 nm, L = 10 km (M = −15)": (1500.0, 1.0, 10.0, -15.0)}
        if sc in presets:
            lam, dl, L, M = presets[sc]
            cr = st.container()
            st.info("Slide example loaded (values used exactly as in the slides).")
        else:
            lam, dl, L, M, cr = get_inputs("ds_", 1550.0, 1.0, 50.0)
        dt = pulse_spread(M, dl, L)
        with cr:
            metrics([("M", f"{M:.2f} ps/nm·km"), ("Δ(τ/L) = −MΔλ", f"{-M * dl:.4g} ps/km"), ("Pulse spread Δτ = L·|Δ(τ/L)|", fmt_time(dt))], cols=3)
            if abs(M) < 1e-9 or abs(lam - 1300) < 1:
                st.success("M ≈ 0: no material-dispersive pulse spreading at this wavelength!")
            elif M > 0:
                st.write("M > 0 → Δ(τ/L) < 0: the **longer** wavelength (λ₂) arrives before λ₁.")
            else:
                st.write("M < 0 → Δ(τ/L) > 0: the **shorter** wavelength (λ₁) travels faster.")
            bw = bandwidth_from_spread(dt) if dt > 0 else None
            if bw:
                st.write(f"→ Optical 3-dB bandwidth 1/(2Δτ) = **{fmt_rate(bw['f3dB_optical'])}**; "
                         f"NRZ max rate 0.7/Δτ = **{fmt_rate(bw['R_NRZ'], 'b/s')}** (see §3.4).")
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.3))
        Ls = np.linspace(0, max(L * 1.5, 1), 100)
        ax[0].plot(Ls, abs(M) * dl * Ls * 1e-3, color=BLUE, lw=2)
        ax[0].plot([L], [dt * 1e9], "o", color=RED)
        ax[0].set_xlabel("L (km)")
        ax[0].set_ylabel("Δτ (ns)")
        ax[0].set_title("Spread grows linearly with length")
        dls = np.linspace(0, max(dl * 2, 1), 100)
        ax[1].plot(dls, abs(M) * dls * L * 1e-3, color=GREEN, lw=2)
        ax[1].plot([dl], [dt * 1e9], "o", color=RED)
        ax[1].set_xlabel("Δλ (nm)")
        ax[1].set_ylabel("Δτ (ns)")
        ax[1].set_title("… and linearly with source width")
        fig.tight_layout()
        show(fig)
    with tab_p:
        cl, cr = st.columns([1, 2.4])
        with cl:
            T = st.slider("Input pulse width T (FWHM, ns)", 0.1, 50.0, 5.0, 0.1, key="pb_T")
            dl = st.slider("Source Δλ (nm)", 0.5, 100.0, 20.0, 0.5, key="pb_dl")
            M = st.slider("M (ps/nm·km)", -30.0, 120.0, 110.0, 1.0, key="pb_M")
            L = st.slider("Length L (km)", 0.5, 50.0, 10.0, 0.5, key="pb_L")
        t, inp, out, xs, delays = pulse_output(T * 1e3, dl, M, L)
        with cr:
            fig, ax = plt.subplots(2, 1, figsize=(9.5, 5.2), gridspec_kw={"height_ratios": [1, 1.5]}, sharex=True)
            sel = np.linspace(0, len(xs) - 1, 3).astype(int)
            for j, colr in zip(sel, (BLUE, GREEN, RED)):
                g = np.exp(-0.5 * ((t - delays[j]) / (T * 1e3 / 2.3548)) ** 2)
                ax[0].plot(t * 1e-3, g, color=colr, label=f"λ = λ₀ {xs[j]:+.1f} nm")
            ax[0].set_ylabel("component pulses")
            ax[0].legend(fontsize=8, ncol=3)
            ax[0].set_title("Each spectral component travels at its own speed …")
            ax[1].plot(t * 1e-3, inp, color=GREY, ls="--", lw=2, label=f"input (FWHM {T:.2f} ns)")
            ax[1].plot(t * 1e-3, out, color=TEAL, lw=2.4, label=f"output (FWHM {fwhm(t, out) * 1e-3:.2f} ns)")
            ax[1].set_xlabel("time (ns)")
            ax[1].set_ylabel("normalised power")
            ax[1].set_title("… so the pulse arriving at the receiver is broader and lower")
            ax[1].legend()
            fig.tight_layout()
            show(fig)
        dt = pulse_spread(M, dl, L)
        metrics([("Δτ = |M|ΔλL", fmt_time(dt)), ("Output FWHM (numerical)", f"{fwhm(t, out) * 1e-3:.3f} ns"),
                 ("Peak reduction", f"{100 * (1 - out.max()):.1f} %")], cols=3)
        st.caption("Model: Gaussian spectrum (FWHM Δλ) sampled at 61 wavelengths, delay of each component = −M(λ−λc)L. "
                   "The slide's Δτ is the spread between the extreme wavelengths λ₁ and λ₂ of the linewidth.")


# --------------------------------------------------------------------------------------
# CH3 – 3.4  Bandwidth & data rate
# --------------------------------------------------------------------------------------
def make_bits(n=64, seed=11):
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, n)
    bits[8:12] = 1
    bits[20:24] = 0
    bits[30:34] = [1, 0, 1, 0]
    return bits


def digital_link(bits, x, fmt, sps=40):
    """x = Δτ / T (spread in bit periods). Returns t-axis, input, output, eye-opening."""
    w = np.zeros(len(bits) * sps)
    for i, b in enumerate(bits):
        if b:
            w[i * sps: i * sps + (sps if fmt == "NRZ" else sps // 2)] = 1.0
    sigma = x * sps / 2.3548          # Gaussian impulse response with FWHM = Δτ
    if sigma > 0.3:
        kk = np.arange(-int(5 * sigma), int(5 * sigma) + 1)
        ker = np.exp(-0.5 * (kk / sigma) ** 2)
        ker /= ker.sum()
        y = np.convolve(w, ker, mode="same")
    else:
        y = w.copy()
    ts = sps // 2 if fmt == "NRZ" else sps // 4
    idx = np.arange(3, len(bits) - 3)
    samples = y[idx * sps + ts]
    ones = samples[bits[idx] == 1]
    zeros = samples[bits[idx] == 0]
    opening = float(ones.min() - zeros.max()) if len(ones) and len(zeros) else float("nan")
    return w, y, opening


def page_bandwidth():
    header("3.4  Bandwidth & Data Rate", "Chapter 3 · Information rate limited by pulse spreading")
    tab_c, tab_a, tab_l, tab_d, tab_b = st.tabs(
        ["Concepts", "Sinusoidal modulation", "Frequency response & dB", "RZ / NRZ digital link", "Link calculator"])
    with tab_c:
        st.markdown("Allow the pulse spread to be at most half the modulation period, Δτ ≤ T/2 (eq. 2):")
        st.latex(r"f_{max}=\frac{1}{2\Delta\tau},\qquad f_{3\text{-dB (optical)}}=\frac{1}{2\Delta\tau}\ (5),\qquad f_{3\text{-dB}}\cdot L=\frac{1}{2\Delta(\tau/L)}\ (3.16)")
        st.latex(r"L_f=-10\log\,e^{-\ln2\,(f/f_{3\text{-dB}})^2}\ (7),\qquad Loss=L_a+L_f\ (6)")
        st.markdown("**Detector:** Pₑ = R_L (ρP)² – electrical power goes as the *square* of optical power, so **dB(electrical) = 2 × dB(optical)**. "
                    "The electrical 3-dB bandwidth is therefore reached at 1.5 dB optical loss, f₁.₅dB = 0.71 f₃dB:")
        st.latex(r"f_{3\text{-dB(el)}}=\frac{0.71}{2\Delta\tau}=\frac{0.35}{\Delta\tau}\ ,\qquad f_{3\text{-dB(el)}}\,L=\frac{0.35}{\Delta(\tau/L)}\ (3.19)")
        st.latex(r"R_{RZ}\cdot L=\frac{0.35}{\Delta(\tau/L)}\ (3.20),\qquad R_{NRZ}\cdot L=\frac{0.7}{\Delta(\tau/L)}\ (3.21)\ \Rightarrow\ R_{NRZ}L=2R_{RZ}L")
        st.markdown("Required bandwidth: B_RZ ≈ 1/T = R, B_NRZ ≈ 1/(2T) = R/2. An RZ pulse has width t_p = T/2; NRZ pulses fill the whole bit.")
    with tab_a:
        st.markdown("Two wavelengths carry the same modulation but arrive Δτ apart (slide 27–28). "
                    "`P_R = [1+cos 2πft] + [1+cos 2πf(t+Δτ)]`.")
        cl, cr = st.columns([1, 2.4])
        with cl:
            ratio = st.slider("Modulation frequency f / (1/2Δτ)", 0.0, 2.0, 1.0, 0.05, key="bw_ratio")
        fdt = ratio / 2.0                   # f·Δτ
        t = np.linspace(0, 2, 800)          # in units of T = 1/f
        p1 = 1 + np.cos(2 * np.pi * t)
        p2 = 1 + np.cos(2 * np.pi * (t + fdt))
        pr = p1 + p2
        depth = abs(math.cos(math.pi * fdt))
        with cr:
            fig, ax = plt.subplots(figsize=(9, 3.6))
            ax.plot(t, p1, color=BLUE, ls="--", label="λ₁ (fast)")
            ax.plot(t, p2, color=RED, ls="--", label="λ₂ (delayed by Δτ)")
            ax.plot(t, pr, color=INK, lw=2.4, label="received sum P_R")
            ax.set_xlabel("t / T")
            ax.set_ylabel("power")
            ax.legend(ncol=3, fontsize=8, loc="upper right")
            ax.set_ylim(-0.1, 4.6)
            ax.set_title(f"Δτ = {fdt:.3f} T · received modulation depth = {100 * depth:.0f} % of the transmitted")
            show(fig)
        if abs(ratio - 1.0) < 1e-9:
            st.success("Δτ = T/2: the two components are in anti-phase, P_R = 2 = constant – all modulation is lost. "
                       "This defines f_max = 1/(2Δτ).")
    with tab_l:
        cl, cr = st.columns([1, 2])
        with cl:
            x = st.slider("f / f₃dB", 0.0, 2.0, 0.71, 0.01, key="bw_x")
        Lf = float(freq_dependent_loss_db(x))
        with cr:
            metrics([("Optical loss L_f", f"{Lf:.3f} dB"), ("Electrical loss", f"{2 * Lf:.3f} dB", "dB_el = 2·dB_opt"),
                     ("f / f₃dB", f"{x:.2f}")], cols=3)
        xs = np.linspace(0, 2, 300)
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
        ax[0].plot(xs, freq_dependent_loss_db(xs), color=BLUE, lw=2, label="optical loss (eq. 7)")
        ax[0].plot(xs, 2 * freq_dependent_loss_db(xs), color=RED, lw=2, label="electrical loss")
        ax[0].axhline(3, color=GREY, ls=":")
        ax[0].axhline(1.5, color=GREY, ls=":")
        ax[0].plot([1, 0.706], [3, 1.5], "o", color=INK)
        ax[0].text(1.02, 2.6, "f = f₃dB → 3 dB", fontsize=8)
        ax[0].text(0.3, 1.7, "f = 0.71 f₃dB → 1.5 dB (opt) = 3 dB (el)", fontsize=8)
        ax[0].plot([x], [Lf], "s", color=AMBER, ms=8)
        ax[0].set_xlabel("f / f₃dB")
        ax[0].set_ylabel("loss (dB)")
        ax[0].legend()
        ax[0].invert_yaxis()
        ax[0].set_title("Modulation-frequency dependent loss")
        # dB relation with a photodetector
        P_opt = np.linspace(0.05, 1, 200)
        ax[1].plot(10 * np.log10(P_opt), 10 * np.log10(P_opt ** 2), color=TEAL, lw=2)
        ax[1].plot(10 * np.log10(P_opt), 10 * np.log10(P_opt), color=GREY, ls=":", label="1 : 1")
        ax[1].set_xlabel("optical power change (dB)")
        ax[1].set_ylabel("electrical power change (dB)")
        ax[1].set_title("Photodetector: P_e = R_L ρ² P²  →  slope 2")
        ax[1].legend()
        fig.tight_layout()
        show(fig)
        st.markdown("**Detector calculator**")
        c1, c2, c3 = st.columns(3)
        rho = c1.number_input("Responsivity ρ (A/W)", 0.1, 2.0, 0.8, 0.05, key="bw_rho")
        RL = c2.number_input("Load R_L (Ω)", 1.0, 10000.0, 50.0, 10.0, key="bw_RL")
        P1 = c3.number_input("Optical power P (µW)", 0.1, 1000.0, 10.0, 1.0, key="bw_P")
        Pe1 = RL * (rho * P1 * 1e-6) ** 2
        Pe_half = RL * (rho * P1 * 0.5e-6) ** 2
        st.write(f"i = ρP = {rho * P1:.3g} µA, Pₑ = R_L i² = {Pe1:.3e} W. "
                 f"Halving the optical power (−3.01 dB optical) gives Pₑ = {Pe_half:.3e} W, i.e. "
                 f"{10 * math.log10(Pe_half / Pe1):.2f} dB electrical – twice the optical dB.")
    with tab_d:
        st.markdown("Simulated pulse train through a dispersive fiber. The fiber is modelled by a Gaussian impulse response "
                    "of FWHM Δτ (illustrative). Push the bit rate past the slide limits and watch the eye close.")
        cl, cr = st.columns([1, 2.6])
        with cl:
            fmt = st.radio("Line code", ["NRZ", "RZ"], horizontal=True, key="bw_fmt")
            dtau_ns = st.number_input("Pulse spread Δτ (ns)", 0.01, 100.0, 1.0, 0.1, key="bw_dt")
            bw = bandwidth_from_spread(dtau_ns * 1e-9)
            Rmax = bw["R_NRZ"] if fmt == "NRZ" else bw["R_RZ"]
            frac = st.slider("Bit rate as a fraction of the slide limit R_max", 0.1, 2.5, 1.0, 0.05, key="bw_frac")
        R = frac * Rmax
        T = 1.0 / R
        x = dtau_ns * 1e-9 / T
        bits = make_bits()
        w, y, opening = digital_link(bits, x, fmt)
        sps = 40
        with cr:
            metrics([("Slide limit R_max", fmt_rate(Rmax, "b/s"), "RZ: 0.35/Δτ · NRZ: 0.7/Δτ"), ("Chosen bit rate R", fmt_rate(R, "b/s")),
                     ("Δτ / T", f"{x:.2f}"), ("Eye opening", f"{100 * opening:.0f} %")])
            tt = np.arange(len(y)) / sps
            nshow = 20
            fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.6), gridspec_kw={"width_ratios": [1.7, 1]})
            ax[0].plot(tt[: nshow * sps], w[: nshow * sps], color=GREY, ls="--", lw=1.4, label="transmitted")
            ax[0].plot(tt[: nshow * sps], y[: nshow * sps], color=BLUE, lw=2, label="received")
            for i, b in enumerate(bits[:nshow]):
                ax[0].text(i + 0.5, 1.12, str(b), ha="center", fontsize=8, color=INK)
            ax[0].set_ylim(-0.15, 1.3)
            ax[0].set_xlabel("time (bit periods)")
            ax[0].set_ylabel("power")
            ax[0].legend(loc="lower right", ncol=2, fontsize=8)
            ax[0].set_title(f"{fmt} data at R = {fmt_rate(R, 'b/s')}")
            te = np.arange(2 * sps) / sps
            for k in range(3, len(bits) - 3):
                ax[1].plot(te, y[k * sps:(k + 2) * sps], color=BLUE, alpha=0.25, lw=0.9)
            ts = 0.5 if fmt == "NRZ" else 0.25
            ax[1].axvline(ts, color=RED, ls=":")
            ax[1].axvline(ts + 1, color=RED, ls=":")
            ax[1].set_xlabel("time (bit periods)")
            ax[1].set_title("Eye diagram")
            fig.tight_layout()
            show(fig)
        if frac <= 1.0 and opening > 0.6:
            st.success("Within the slide limit: pulses remain recognisable, the eye is open.")
        elif opening > 0.2:
            st.warning("Beyond the slide limit: inter-symbol interference is closing the eye.")
        else:
            st.error("Eye essentially closed – errors are unavoidable at this rate.")
        st.markdown("**Required bandwidth** (power spectral density of the continuous part):")
        f_r = np.linspace(0, 3, 500)
        sinc2 = lambda a: np.sinc(a) ** 2
        fig2, ax2 = plt.subplots(figsize=(8.5, 2.9))
        ax2.plot(f_r, sinc2(f_r / 2.0), color=RED, lw=2, label="RZ  (t_p = T/2, first null at 2/T)")
        ax2.plot(f_r, sinc2(f_r), color=BLUE, lw=2, label="NRZ (first null at 1/T)")
        ax2.axvline(1.0, color=RED, ls=":")
        ax2.axvline(0.5, color=BLUE, ls=":")
        ax2.text(1.02, 0.7, "B_RZ ≈ R", color=RED)
        ax2.text(0.52, 0.55, "B_NRZ ≈ R/2", color=BLUE)
        ax2.set_xlabel("frequency / R  (R = 1/T)")
        ax2.set_ylabel("PSD (norm.)")
        ax2.legend(fontsize=8)
        show(fig2)
    with tab_b:
        st.markdown("Complete link budget: from source and fiber parameters to bandwidth and data rate.")
        cl, cr = st.columns([1, 2])
        with cl:
            lam = st.number_input("Wavelength λ (nm)", 800.0, 1600.0, 1550.0, 10.0, key="lk_lam")
            dl = st.number_input("Source Δλ (nm)", 0.01, 100.0, 2.0, 0.5, key="lk_dl")
            L = st.number_input("Length L (km)", 0.1, 500.0, 20.0, 1.0, key="lk_L")
            inside = 1200 <= lam <= 1600
            if inside:
                M = float(material_dispersion_model(lam))
                st.write(f"M(model) = {M:.2f} ps/(nm·km)")
            else:
                M = st.number_input("M (ps/nm·km)", -200.0, 200.0, 110.0, 1.0, key="lk_M")
            Rreq = st.number_input("Required bit rate (Mb/s)", 1.0, 100000.0, 622.0, 10.0, key="lk_R")
        dt = pulse_spread(M, dl, L)
        with cr:
            if dt <= 0:
                st.info("M = 0 → no material-dispersion limit at this wavelength (other limits – waveguide, modal, loss – still apply).")
            else:
                bw = bandwidth_from_spread(dt)
                fig = block_diagram(
                    [("Source", f"λ = {lam:g} nm\nΔλ = {dl:g} nm"), ("Fiber", f"L = {L:g} km\nM = {M:.1f} ps/nm·km"),
                     ("Photodetector", f"Δτ = {fmt_time(dt)}\nf₃dB(el) = {fmt_rate(bw['f3dB_electrical'])}"),
                     ("Receiver", f"R_NRZ ≤ {fmt_rate(bw['R_NRZ'], 'b/s')}\nR_RZ ≤ {fmt_rate(bw['R_RZ'], 'b/s')}")],
                    ["", "Δτ = |M|ΔλL", ""], figsize=(10, 2.4), height=1.0, colors=["#dff1f1", "#e6ecfa", "#f6e1dc", "#fdf0d5"])
                show(fig)
                metrics([("Δτ", fmt_time(dt)), ("f₃dB optical 1/(2Δτ)", fmt_rate(bw["f3dB_optical"])),
                         ("R_NRZ,max", fmt_rate(bw["R_NRZ"], "b/s")), ("R_RZ,max", fmt_rate(bw["R_RZ"], "b/s"))])
                Rn = Rreq * 1e6
                verdict = ("✅ NRZ supported" if Rn <= bw["R_NRZ"] else "❌ NRZ NOT supported") + " · " + \
                          ("✅ RZ supported" if Rn <= bw["R_RZ"] else "❌ RZ NOT supported")
                st.write(f"**Required {fmt_rate(Rn, 'b/s')}:** {verdict}")
                Ls = np.logspace(-1, 3, 200)
                bl_nrz = 0.7 / (abs(M) * dl * 1e-12)      # (b/s)·km
                fig2, ax2 = plt.subplots(figsize=(8, 3.2))
                ax2.loglog(Ls, bl_nrz / Ls, color=BLUE, lw=2, label="NRZ  R·L = 0.7/Δ(τ/L)")
                ax2.loglog(Ls, bl_nrz / 2 / Ls, color=RED, lw=2, label="RZ  R·L = 0.35/Δ(τ/L)")
                ax2.plot([L], [Rn], "o", color=INK, label="required")
                ax2.set_xlabel("L (km)")
                ax2.set_ylabel("max bit rate (b/s)")
                ax2.legend(fontsize=8)
                show(fig2)
                st.caption(f"Rate–length products: NRZ {bl_nrz / 1e9:.3g} Gb/s·km, RZ {bl_nrz / 2e9:.3g} Gb/s·km "
                           "(material dispersion only; attenuation and other dispersion are not included).")


# --------------------------------------------------------------------------------------
# CH3 – 3.5  Polarization
# --------------------------------------------------------------------------------------
def page_polarization():
    header("3.5  Polarization", "Chapter 3 · Orientation of the electric field vector")
    tab_c, tab_s, tab_m = st.tabs(["Concepts", "Polarization state simulator", "Polarizer (Malus's law)"])
    with tab_c:
        st.markdown(
            "A wave travelling in z and **linearly polarized in x** has its E-field oscillating along x; likewise for y. "
            "The two orthogonal linear polarizations are the plane-wave *modes* of an unbounded medium. "
            "If the direction of E varies randomly the wave is **un-polarized**. Most fibers depolarize the input light; "
            "special *polarization-maintaining* fibers preserve it.")
        st.latex(r"E_x=A_x\cos(\omega t-kz),\qquad E_y=A_y\cos(\omega t-kz+\delta)")
        st.markdown("Extension beyond the slides: the phase difference δ and amplitude ratio A_y/A_x generate the *linear*, "
                    "*circular* and *elliptical* states that combine the two orthogonal modes.")
    with tab_s:
        cl, cr = st.columns([1, 2.5])
        with cl:
            preset = st.selectbox("Preset", ["Custom", "Linear x", "Linear y", "Linear 45°", "Circular", "Elliptical", "Un-polarized"], key="po_pre")
            vals = {"Linear x": (1.0, 0.0, 0), "Linear y": (0.0, 1.0, 0), "Linear 45°": (1.0, 1.0, 0),
                    "Circular": (1.0, 1.0, 90), "Elliptical": (1.0, 0.5, 60), "Un-polarized": (1.0, 1.0, 0)}
            ax_, ay_, dl_ = vals.get(preset, (1.0, 0.7, 45))
            Ax = st.slider("Amplitude Aₓ", 0.0, 1.0, float(ax_), 0.05, key=f"po_Ax_{preset}")
            Ay = st.slider("Amplitude A_y", 0.0, 1.0, float(ay_), 0.05, key=f"po_Ay_{preset}")
            dl = st.slider("Phase difference δ (deg)", -180, 180, int(dl_), 5, key=f"po_dl_{preset}")
            ph = st.slider("Time phase ωt (deg)", 0, 360, 40, 5, key="po_ph")
        d = math.radians(dl)
        if preset == "Un-polarized":
            rng = np.random.default_rng(3)
            with cr:
                fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
                zc = np.linspace(0, 4 * math.pi, 400)
                ax[0].set_aspect("equal")
                for i in range(60):
                    th = rng.uniform(0, math.pi)
                    a = rng.uniform(0.3, 1)
                    ax[0].plot([-a * math.cos(th), a * math.cos(th)], [-a * math.sin(th), a * math.sin(th)], color=BLUE, alpha=0.25)
                ax[0].set_xlim(-1.1, 1.1)
                ax[0].set_ylim(-1.1, 1.1)
                ax[0].set_title("E-vector direction (many instants)")
                ax[0].set_xlabel("Eₓ")
                ax[0].set_ylabel("E_y")
                tt = np.linspace(0, 10, 500)
                seg = np.cumsum(rng.normal(0, 0.15, 500))
                ax[1].plot(np.cos(seg) * np.exp(-0.0 * tt), np.sin(seg), color=RED, lw=1.2)
                ax[1].set_aspect("equal")
                ax[1].set_title("E-tip wanders randomly in the x–y plane")
                fig.tight_layout()
                show(fig)
            st.info("Un-polarized light: E direction (and phase relation of Eₓ, E_y) changes randomly on time scales ≪ detector response.")
        else:
            s = math.radians(ph)
            zz = np.linspace(0, 3 * 2 * math.pi, 400)
            phi = s - zz
            Ex = Ax * np.cos(phi)
            Ey = Ay * np.cos(phi + d)
            tt = np.linspace(0, 2 * math.pi, 300)
            ex_e = Ax * np.cos(tt)
            ey_e = Ay * np.cos(tt + d)
            with cr:
                fig = plt.figure(figsize=(11, 4.2))
                a3 = fig.add_subplot(1, 2, 1, projection="3d")
                a3.plot(zz, Ex, Ey, color=INK, lw=2)
                a3.plot(zz, Ex, np.full_like(zz, -1.05), color=BLUE, lw=1, alpha=0.6)
                a3.plot(zz, np.full_like(zz, -1.05), Ey, color=RED, lw=1, alpha=0.6)
                for i in range(0, len(zz), 24):
                    a3.plot([zz[i], zz[i]], [0, Ex[i]], [0, Ey[i]], color=GREY, lw=0.8)
                a3.plot([0, zz[-1]], [0, 0], [0, 0], color=INK, lw=0.8, ls="--")
                a3.set_xlabel("z")
                a3.set_ylabel("Eₓ")
                a3.set_zlabel("E_y")
                a3.set_ylim(-1.05, 1.05)
                a3.set_zlim(-1.05, 1.05)
                a3.set_title("E-field along z (snapshot)")
                a2 = fig.add_subplot(1, 2, 2)
                a2.plot(ex_e, ey_e, color=BLUE, lw=2)
                a2.plot([0, Ax * math.cos(s)], [0, Ay * math.cos(s + d)], color=RED, lw=2.5)
                a2.plot([Ax * math.cos(s)], [Ay * math.cos(s + d)], "o", color=RED)
                a2.set_aspect("equal")
                a2.set_xlim(-1.1, 1.1)
                a2.set_ylim(-1.1, 1.1)
                a2.set_xlabel("Eₓ")
                a2.set_ylabel("E_y")
                a2.set_title("Polarization ellipse (looking at the oncoming wave)")
                fig.tight_layout()
                show(fig)
            # classify
            if Ax < 1e-9 or Ay < 1e-9 or abs(math.sin(d)) < 1e-9:
                if Ax < 1e-9 and Ay < 1e-9:
                    kind = "no field"
                else:
                    ang = 90.0 if Ax < 1e-9 else (0.0 if Ay < 1e-9 else math.degrees(math.atan2(Ay, Ax)) * (1 if math.cos(d) > 0 else -1))
                    kind = f"linear (E along {ang:.1f}° from x)"
            elif abs(Ax - Ay) < 1e-9 and abs(abs(dl) - 90) < 1e-9:
                kind = "circular, " + ("clockwise (right-handed)" if dl > 0 else "counter-clockwise (left-handed)") + " as seen facing the oncoming wave"
            else:
                kind = "elliptical, " + ("clockwise" if math.sin(d) > 0 else "counter-clockwise") + " rotation"
            st.success(f"State: **{kind}**")
    with tab_m:
        st.markdown("A linear polarizer transmits I = I₀cos²θ (Malus's law). Unpolarized light through one ideal polarizer gives I₀/2.")
        cl, cr = st.columns([1, 2])
        with cl:
            th = st.slider("Angle between polarization and polarizer axis (deg)", 0, 90, 30, 1, key="po_mal")
            I0 = st.number_input("Incident power (mW)", 0.1, 100.0, 1.0, 0.1, key="po_I0")
        with cr:
            aa = np.linspace(0, 180, 300)
            fig, ax = plt.subplots(figsize=(7, 2.8))
            ax.plot(aa, I0 * np.cos(np.radians(aa)) ** 2, color=BLUE, lw=2)
            ax.plot([th], [I0 * math.cos(math.radians(th)) ** 2], "o", color=RED)
            ax.set_xlabel("θ (deg)")
            ax.set_ylabel("transmitted (mW)")
            show(fig)
            st.write(f"Transmitted: **{I0 * math.cos(math.radians(th)) ** 2:.3f} mW**")


# --------------------------------------------------------------------------------------
# CH3 – 3.6  Resonant cavities
# --------------------------------------------------------------------------------------
def page_cavity():
    header("3.6  Resonant Cavities & Longitudinal Laser Modes", "Chapter 3 · Fabry-Perot cavity between two mirrors")
    tab_c, tab_w, tab_l = st.tabs(["Concepts", "Standing waves in the cavity", "Laser-diode mode spectrum"])
    with tab_c:
        st.markdown("Two perfect mirrors force E = 0 at z = 0 and z = L. With E = E₁(e^{−jkz} − e^{jkz}) the condition at z = L "
                    "gives sin(kL) = 0, i.e. kL = mπ (m = 0, 1, 2, …). Equivalently the round-trip phase is an integer multiple of 2π.")
        st.latex(r"L=\frac{m\lambda}{2}\ (3.22),\qquad \lambda=\frac{2L}{m},\qquad f_m=\frac{c\,m}{2nL}")
        st.latex(r"\Delta f_c=\frac{c}{2nL}\ (3.25),\qquad \Delta\lambda_c=\frac{\lambda_o^{2}\,\Delta f_c}{c}=\frac{\lambda_o^{2}}{2nL}\ (3.26)")
        st.markdown("λ is the wavelength **in the medium filling the cavity**; λ₀ in (3.26) is the free-space wavelength. "
                    "A laser emitting only one longitudinal mode is a **single-mode laser**.")
    with tab_w:
        cl, cr = st.columns([1, 2.4])
        with cl:
            m = st.slider("Mode number m", 1, 8, 3, 1, key="cv_m")
            show_t = st.slider("Time phase ωt (deg)", 0, 360, 60, 10, key="cv_t")
        with cr:
            z = np.linspace(0, 1, 400)
            fig, ax = plt.subplots(figsize=(9, 3.3))
            for mm, colr, alpha in ((m, BLUE, 1.0),):
                ax.plot(z, np.sin(mm * np.pi * z), color=colr, lw=2.2, label=f"envelope  m = {mm}")
                ax.plot(z, -np.sin(mm * np.pi * z), color=colr, lw=2.2, ls="--")
                ax.plot(z, np.sin(mm * np.pi * z) * math.cos(math.radians(show_t)), color=RED, lw=1.5, label="E(z,t) snapshot")
            ax.axvline(0, color=INK, lw=5)
            ax.axvline(1, color=INK, lw=5)
            ax.text(0.0, 1.25, "Mirror", ha="center")
            ax.text(1.0, 1.25, "Mirror", ha="center")
            ax.set_ylim(-1.4, 1.4)
            ax.set_xlabel("z / L")
            ax.set_ylabel("E")
            ax.set_title(f"E = 0 at both mirrors; L = {m}·λ/2 (m half-wavelengths fit in the cavity)")
            ax.legend(loc="lower center", ncol=2, fontsize=8)
            show(fig)
        st.caption("For m = 1 the cavity is exactly half a wavelength long; the round-trip phase 2kL equals m·2π.")
    with tab_l:
        cl, cr = st.columns([1, 2.4])
        with cl:
            L_um = st.number_input("Cavity length L (µm)", 20.0, 2000.0, 300.0, 10.0, key="cv_L")
            n = st.number_input("Refractive index n", 1.0, 4.5, 3.6, 0.1, key="cv_n")
            lam0 = st.number_input("Free-space wavelength λ₀ (µm)", 0.6, 1.7, 0.82, 0.01, key="cv_lam")
            dl = st.number_input("Gain (spectral) width Δλ (nm)", 0.1, 50.0, 2.0, 0.1, key="cv_dl")
        p = cavity_params(L_um * 1e-6, n, lam0 * 1e-6)
        dlam_nm = p["dlam"] * 1e9
        Nmodes = dl / dlam_nm
        with cr:
            metrics([("Mode spacing Δf_c", fmt_rate(p["df"])), ("Δλ_c (free space)", f"{dlam_nm:.3f} nm"),
                     ("Nearest mode number m", f"{p['m']:.0f}"), ("Modes within Δλ", f"{Nmodes:.2f} ≈ {round(Nmodes)}")])
        m0 = round(p["m"])
        ms = np.arange(m0 - 60, m0 + 61)
        lam_m = 2 * n * L_um * 1e-6 / ms * 1e9               # nm
        lam_c = lam0 * 1e3
        lam_c_mode = lam_m[np.argmin(np.abs(lam_m - lam_c))]
        x = np.linspace(lam_c - 1.6 * dl, lam_c + 1.6 * dl, 800)
        gain = np.exp(-4 * math.log(2) * ((x - lam_c) / dl) ** 2)
        fig, ax = plt.subplots(3, 1, figsize=(9.5, 6.4), sharex=True)
        ax[0].plot(x, gain, "--", color=AMBER, lw=2)
        ax[0].set_ylabel("gain")
        ax[0].set_title(f"Gain medium (FWHM = {dl:g} nm) · cavity resonances (spacing {dlam_nm:.3f} nm) · laser output")
        sel = (lam_m > x[0]) & (lam_m < x[-1])
        ax[1].vlines(lam_m[sel], 0, 1, color=BLUE, lw=2.5)
        ax[1].set_ylabel("resonances")
        g_at = np.exp(-4 * math.log(2) * ((lam_m - lam_c) / dl) ** 2)
        emit = sel & (g_at >= 0.5)
        ax[2].plot(x, gain, "--", color=AMBER, lw=1.2)
        ax[2].vlines(lam_m[emit], 0, g_at[emit], color=RED, lw=3)
        ax[2].set_ylabel("output")
        ax[2].set_xlabel("wavelength (nm)")
        ax[2].axvspan(lam_c - dl / 2, lam_c + dl / 2, color=AMBER, alpha=0.12)
        fig.tight_layout()
        show(fig)
        n_emit = int(emit.sum())
        st.write(f"Modes above the half-power level inside Δλ: **{n_emit}** (analytic Δλ/Δλ_c = {Nmodes:.2f}).")
        L_single = (lam0 * 1e-6) ** 2 / (2 * n * dl * 1e-9) * 1e6
        st.info(f"**Single-mode design:** Δλ_c ≥ Δλ requires L ≤ λ₀²/(2nΔλ) = **{L_single:.1f} µm** for this gain width. "
                "Slide example: L = 300 µm, n = 3.6, λ₀ = 0.82 µm → Δλ_c = 0.311 nm → 2 nm / 0.311 nm ≈ 6 modes.")


# --------------------------------------------------------------------------------------
# CH3 – 3.7  Fresnel
# --------------------------------------------------------------------------------------
def page_fresnel():
    header("3.7  Reflection at a Plane Boundary", "Chapter 3 · Fresnel reflection, polarization dependence and Brewster's angle")
    tab_c, tab_n, tab_a, tab_g = st.tabs(["Concepts", "Normal incidence", "Arbitrary incidence & Brewster", "Interface losses"])
    with tab_c:
        st.latex(r"\rho=\frac{E_r}{E_i}=\frac{n_1-n_2}{n_1+n_2},\quad \tau=\frac{E_t}{E_i}=\frac{2n_1}{n_1+n_2},\quad R=|\rho|^2")
        st.markdown("Derived from continuity of E and H at z = 0: E₁ᵢ + E₁ᵣ = E₂ₜ and (E₁ᵢ − E₁ᵣ)/η₁ = E₂ₜ/η₂, with η = √(μ/ε) and "
                    "η₂/η₁ = n₁/n₂ for non-magnetic media. The result is valid for **normal incidence** (Fresnel reflection).")
        st.latex(r"\rho_p=\frac{-n_2^2\cos\theta_i+n_1\sqrt{n_2^2-n_1^2\sin^2\theta_i}}{n_2^2\cos\theta_i+n_1\sqrt{n_2^2-n_1^2\sin^2\theta_i}}\ (3.29),\qquad "
                 r"\rho_s=\frac{n_1\cos\theta_i-\sqrt{n_2^2-n_1^2\sin^2\theta_i}}{n_1\cos\theta_i+\sqrt{n_2^2-n_1^2\sin^2\theta_i}}\ (3.30)")
        st.latex(r"\tan\theta_B=\frac{n_2}{n_1}\ \ (\text{parallel polarization only; }\rho_p=0)")
    with tab_n:
        cl, cr = st.columns([1, 2])
        with cl:
            n1 = material_input("Medium 1", "fr1", "Gas / CO₂ (n = 1)")
            n2 = material_input("Medium 2", "fr2", "Glass (n ≈ 1.5)")
        rho = (n1 - n2) / (n1 + n2)
        tau = 2 * n1 / (n1 + n2)
        R = rho ** 2
        T = 1 - R
        with cr:
            metrics([("ρ = (n₁−n₂)/(n₁+n₂)", f"{rho:+.4f}"), ("τ = 2n₁/(n₁+n₂)", f"{tau:.4f}", "check: 1 + ρ = τ"),
                     ("R = |ρ|²", f"{100 * R:.3f} %"), ("Fresnel loss", f"{-10 * math.log10(T):.3f} dB", "−10 log(1−R)")])
        fig, ax = plt.subplots(figsize=(8, 2.8))
        ax.axis("off")
        ax.set_xlim(0, 10)
        ax.set_ylim(-1.3, 1.3)
        ax.add_patch(Rectangle((5, -1.2), 5, 2.4, fc="#f7e9d6", ec="none", alpha=min(0.9, 0.15 + 0.1 * n2)))
        ax.add_patch(Rectangle((0, -1.2), 5, 2.4, fc="#e9f4f5", ec="none", alpha=min(0.9, 0.15 + 0.1 * n1)))
        ax.plot([5, 5], [-1.2, 1.2], color=INK, lw=3)
        arrow(ax, (1.0, 0.5), (4.9, 0.5), color=BLUE, lw=2.5)
        arrow(ax, (4.9, -0.5), (1.0, -0.5), color=RED, lw=1 + 5 * abs(rho))
        arrow(ax, (5.1, 0.5), (9.0, 0.5), color=GREEN, lw=1 + 3 * min(1, abs(tau)))
        ax.text(1.0, 0.68, "E_i = 1", color=BLUE)
        ax.text(1.0, -0.85, f"E_r = ρ = {rho:+.3f}", color=RED)
        ax.text(6.4, 0.68, f"E_t = τ = {tau:.3f}", color=GREEN)
        ax.text(0.2, 1.05, f"n₁ = {n1:g}", fontsize=11, fontweight="bold")
        ax.text(9.8, 1.05, f"n₂ = {n2:g}", fontsize=11, fontweight="bold", ha="right")
        show(fig)
        if n1 == 1.0 and abs(n2 - 1.5) < 1e-9:
            st.caption("Slide example (air → glass, n₂ = 1.5): R = 0.04 → 4 % reflected, 96 % transmitted, 10 log(0.96) = −0.177 dB ≈ 0.2 dB Fresnel loss.")
    with tab_a:
        cl, cr = st.columns([1, 2.6])
        with cl:
            n1 = material_input("Medium 1 ", "fa1", "Gas / CO₂ (n = 1)")
            n2 = material_input("Medium 2 ", "fa2", "Glass (n ≈ 1.5)")
            th = st.slider("Angle of incidence θᵢ (deg)", 0.0, 89.9, 45.0, 0.1, key="fa_th")
        ang = np.linspace(0, 89.99, 500)
        rp, rs = fresnel(n1, n2, ang)
        rp0, rs0 = fresnel(n1, n2, th)
        thB = brewster_deg(n1, n2)
        thc = critical_angle_deg(n1, n2)
        with cr:
            fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
            ax[0].plot(ang, np.abs(rp), color=RED, lw=2, label="|ρ_p| parallel")
            ax[0].plot(ang, np.abs(rs), color=BLUE, lw=2, ls="--", label="|ρ_s| perpendicular")
            ax[0].axvline(th, color=GREY, ls=":")
            ax[0].axvline(thB, color=GREEN, ls="--", lw=1.2)
            ax[0].text(thB + 1, 0.9, f"θ_B = {thB:.1f}°", color=GREEN, fontsize=9)
            if thc:
                ax[0].axvline(thc, color=AMBER, ls="--", lw=1.2)
                ax[0].text(thc - 1, 0.5, f"θ_c = {thc:.1f}°", color=AMBER, ha="right", fontsize=9)
            ax[0].set_ylim(0, 1.05)
            ax[0].set_xlabel("θᵢ (deg)")
            ax[0].set_ylabel("|ρ|")
            ax[0].set_title(f"Amplitude reflection coefficient ({n1:g} → {n2:g})")
            ax[0].legend(loc="upper left", fontsize=8)
            ax[1].plot(ang, np.abs(rp) ** 2, color=RED, lw=2, label="R_p")
            ax[1].plot(ang, np.abs(rs) ** 2, color=BLUE, lw=2, ls="--", label="R_s")
            ax[1].plot(ang, (np.abs(rp) ** 2 + np.abs(rs) ** 2) / 2, color=INK, lw=1.4, label="unpolarised")
            ax[1].axvline(th, color=GREY, ls=":")
            ax[1].set_ylim(0, 1.05)
            ax[1].set_xlabel("θᵢ (deg)")
            ax[1].set_ylabel("R = |ρ|²")
            ax[1].set_title("Power reflectance")
            ax[1].legend(fontsize=8)
            fig.tight_layout()
            show(fig)
        metrics([("|ρ_p|", f"{abs(rp0[0]):.4f}"), ("|ρ_s|", f"{abs(rs0[0]):.4f}"),
                 ("R_p / R_s", f"{100 * abs(rp0[0]) ** 2:.2f} % / {100 * abs(rs0[0]) ** 2:.2f} %"),
                 ("Brewster angle θ_B", f"{thB:.2f}°")])
        st.caption("Slide examples: air→glass (1 → 1.5): θ_B = 56.3°; glass→air (1.5 → 1): θ_B = 33.7°. "
                   "For perpendicular polarization there is no Brewster angle. The slide plot uses n₂ = 1.48 (|ρ| = 0.194 at 0°).")
        if thc and th >= thc:
            st.warning("θᵢ ≥ θ_c: |ρ| = 1 for both polarizations (complete reflection, see §3.8).")
    with tab_g:
        st.markdown("Fresnel loss of a chain of interfaces (incoherent addition, no multiple-reflection interference) – "
                    "e.g. a fiber-to-fiber connector with an air gap has two glass/air surfaces.")
        cl, cr = st.columns([1, 2])
        with cl:
            ng = st.number_input("Glass index", 1.3, 4.0, 1.5, 0.01, key="fg_n")
            nsurf = st.slider("Number of glass/air interfaces", 1, 8, 2, 1, key="fg_ns")
            gel = st.checkbox("Index-matching gel (n = glass)", key="fg_gel")
        R1 = 0.0 if gel else ((1 - ng) / (1 + ng)) ** 2
        Ttot = (1 - R1) ** nsurf
        with cr:
            metrics([("Per-interface R", f"{100 * R1:.3f} %"), ("Total transmission", f"{100 * Ttot:.2f} %"),
                     ("Total Fresnel loss", f"{-10 * math.log10(Ttot):.3f} dB" if Ttot > 0 else "—")], cols=3)
            ns = np.arange(0, 9)
            fig, ax = plt.subplots(figsize=(7, 2.8))
            ax.bar(ns, -10 * np.log10((1 - R1) ** ns) if R1 > 0 else np.zeros_like(ns), color=TEAL)
            ax.set_xlabel("number of interfaces")
            ax.set_ylabel("loss (dB)")
            show(fig)


# --------------------------------------------------------------------------------------
# CH3 – 3.8  Critical angle
# --------------------------------------------------------------------------------------
def draw_fiber_rays(n1, n2, th0_deg, n0=1.0, n_diam=9.0):
    """Zig-zag meridional ray in the core (widths in units of the core diameter)."""
    s1 = n0 * math.sin(math.radians(th0_deg)) / n1
    th1 = math.asin(min(1.0, s1))          # angle to the axis inside the core
    inc = math.pi / 2 - th1                # angle of incidence at the core/cladding boundary
    thc = critical_angle_deg(n1, n2)
    tir = math.degrees(inc) >= thc - 1e-12
    fig, ax = plt.subplots(figsize=(10, 2.9))
    ax.add_patch(Rectangle((0, -0.5), n_diam, 1.0, fc="#e6ecfa", ec=INK, lw=1.4))
    ax.add_patch(Rectangle((0, 0.5), n_diam, 0.45, fc="#f7e9d6", ec=INK, lw=1))
    ax.add_patch(Rectangle((0, -0.95), n_diam, 0.45, fc="#f7e9d6", ec=INK, lw=1))
    ax.text(0.1, 0.72, f"cladding n₂ = {n2:g}", fontsize=8, va="center")
    ax.text(0.1, -0.25, f"core n₁ = {n1:g}", fontsize=8, va="center", alpha=0.7)
    ax.axhline(0, color=GREY, ls="--", lw=0.6)
    tn = math.tan(th1)
    if th1 < 1e-9:
        arrow(ax, (-0.8, 0), (n_diam, 0), color=BLUE, lw=2)
    else:
        xs, ys = [0.0], [0.0]
        x_b = 0.5 / tn                    # first hit on the boundary (ray enters on the axis)
        side = 1.0
        while x_b < n_diam:
            xs.append(x_b)
            ys.append(0.5 * side)
            if not tir:
                break
            side *= -1
            x_b += 1.0 / tn
        if tir:                            # finish the path at the end face
            y_last = ys[-1]
            dx = n_diam - xs[-1]
            xs.append(n_diam)
            ys.append(y_last - math.copysign(1.0, y_last) * dx * tn)
        ax.plot(xs, ys, color=BLUE, lw=2.2)
        arrow(ax, (-0.8, -0.8 * tn), (0, 0), color=BLUE, lw=2.2)
        if not tir and len(xs) > 1:
            th_t = math.asin(min(1.0, n1 / n2 * math.sin(inc)))      # measured from the normal (vertical)
            dxo, dyo = 1.4 * math.sin(th_t), 1.4 * math.cos(th_t)
            ax.plot([xs[1], xs[1] + dxo], [0.5, 0.5 + dyo * 0.55], color=RED, lw=2, ls="--")
            ax.text(xs[1] + dxo, 0.5 + dyo * 0.55 + 0.05, "refracts into cladding", color=RED, fontsize=9)
    rp, rs = fresnel(n1, n2, math.degrees(inc))
    R_bounce = float((abs(rp[0]) ** 2 + abs(rs[0]) ** 2) / 2)
    ax.set_xlim(-1.0, n_diam + 0.3)
    ax.set_ylim(-1.15, 1.6)
    ax.axis("off")
    ax.set_title(f"Launch {th0_deg:.1f}° → {math.degrees(th1):.1f}° inside core; incidence on cladding {math.degrees(inc):.1f}° "
                 f"({'≥' if tir else '<'} θ_c = {thc:.1f}°)")
    return fig, tir, math.degrees(inc), R_bounce


def page_tir():
    header("3.8  Critical-Angle Reflection & the Evanescent Wave", "Chapter 3 · Total internal reflection – the basis of fiber guidance")
    tab_c, tab_e, tab_f = st.tabs(["Concepts", "Evanescent field", "Fiber guidance demo"])
    with tab_c:
        st.latex(r"\sin\theta_c=\frac{n_2}{n_1}\ (3.34)\quad(\text{exists only if } n_2<n_1)")
        st.markdown(
            "For θᵢ ≥ θ_c the square root √(n₂² − n₁² sin²θᵢ) is purely imaginary, so ρₚ = (−A + jB)/(A + jB) and "
            "ρₛ = (C − jD)/(C + jD): |ρ| = 1 and R = 1 – **complete reflection** for both polarizations.")
        st.markdown("In region n₁ the incident and reflected waves interfere, forming a **standing wave**. In region n₂ the field is not zero: "
                    "it is an **evanescent wave** decaying exponentially and carrying no power in z:")
        st.latex(r"E\propto e^{-\alpha z},\qquad \alpha=k_0\sqrt{n_1^2\sin^2\theta-n_2^2},\qquad k_0=\frac{2\pi}{\lambda_0}")
        st.markdown("At θ = θ_c, α = 0 (no decay, deep penetration); as θ grows toward 90°, α increases and the evanescent field penetrates less.")
    with tab_e:
        cl, cr = st.columns([1, 2.5])
        with cl:
            n1 = st.number_input("n₁ (dense medium)", 1.05, 4.0, 1.48, 0.01, key="ev_n1")
            n2 = st.number_input("n₂ (rare medium)", 1.0, 3.9, 1.46, 0.01, key="ev_n2")
            lam0 = st.number_input("Free-space wavelength λ₀ (nm)", 400.0, 1700.0, 1310.0, 10.0, key="ev_lam")
        if n2 >= n1:
            st.error("Total internal reflection requires n₂ < n₁.")
            return
        thc = critical_angle_deg(n1, n2)
        with cl:
            th = st.slider("Angle of incidence θ (deg)", 0.0, 89.9, float(min(89.0, thc + 1.5)), 0.1, key="ev_th")
        rp, rs = fresnel(n1, n2, th)
        Rs = float(abs(rs[0]) ** 2)
        Rp = float(abs(rp[0]) ** 2)
        with cr:
            if th < thc:
                st.warning(f"θ = {th:.1f}° < θ_c = {thc:.2f}°: a propagating wave is transmitted (R_s = {100 * Rs:.2f} %, R_p = {100 * Rp:.2f} %). Increase θ.")
            else:
                alpha = evanescent_alpha(n1, n2, th, lam0 * 1e-9)         # 1/m
                depth = 1 / alpha * 1e9 if alpha > 0 else float("inf")     # nm
                metrics([("θ_c", f"{thc:.2f}°"), ("R_s = R_p", f"{100 * Rs:.2f} %"), ("α", f"{alpha * 1e-6:.3f} 1/µm" if alpha > 0 else "0"),
                         ("Penetration depth 1/α", f"{depth:.0f} nm" if alpha > 0 else "∞", "field falls to 1/e")])
                k1 = 2 * math.pi * n1 / (lam0 * 1e-9)
                kz = k1 * math.cos(math.radians(th))
                phi = float(np.angle(rs[0]))
                z1 = np.linspace(-1.5 * lam0, 0, 400)
                zmax2 = min(5 * depth, 6 * lam0) if alpha > 0 else 3 * lam0
                z2 = np.linspace(0, zmax2, 300)
                amp1 = 2 * np.abs(np.cos(kz * z1 * 1e-9 + phi / 2))
                tau_abs = abs(1 + rs[0])
                amp2 = tau_abs * np.exp(-alpha * z2 * 1e-9)
                inst1 = 2 * np.cos(kz * z1 * 1e-9 + phi / 2) * math.cos(math.radians(60))
                fig, ax = plt.subplots(figsize=(10, 3.7))
                ax.plot(z1, amp1, color=BLUE, lw=2, label="|E| standing wave (region n₁)")
                ax.plot(z1, -amp1, color=BLUE, lw=2)
                ax.plot(z1, inst1, color=BLUE, lw=1, ls="--", alpha=0.7, label="E at one instant")
                ax.plot(z2, amp2, color=RED, lw=2.2, label=r"evanescent envelope $\tau e^{-\alpha z}$")
                ax.plot(z2, -amp2, color=RED, lw=2.2)
                ax.axvline(0, color=INK, lw=3)
                ax.text(-1.45 * lam0, 2.05, f"n₁ = {n1:g}")
                ax.text(0.05 * zmax2, 2.05, f"n₂ = {n2:g}")
                if alpha > 0:
                    ax.axvline(depth, color=GREY, ls=":")
                    ax.text(depth, 2.05, "1/α", ha="center", fontsize=9)
                ax.set_ylim(-2.4, 2.4)
                ax.set_xlabel("z (nm) – boundary at z = 0")
                ax.set_ylabel("field (s-polarisation, arb. units)")
                ax.legend(loc="lower left", fontsize=8, ncol=3)
                ax.set_title("Field near a totally-reflecting boundary (continuity of E at z = 0)")
                show(fig)
        # decay curves for several angles
        if th >= thc:
            fig2, ax2 = plt.subplots(1, 2, figsize=(11, 3.3))
            for a_th, colr in zip((thc + 0.001, thc + (90 - thc) * 0.05, thc + (90 - thc) * 0.2, thc + (90 - thc) * 0.6, 89.9), (GREEN, TEAL, BLUE, PURPLE, RED)):
                al = evanescent_alpha(n1, n2, a_th, lam0 * 1e-9)
                zz = np.linspace(0, 6 * lam0, 300)
                ax2[0].plot(zz, np.exp(-al * zz * 1e-9), color=colr, label=f"θ = {a_th:.1f}°")
            ax2[0].set_xlabel("z (nm)")
            ax2[0].set_ylabel("|E| / |E(0)|")
            ax2[0].set_title("Decay steepens as θ grows (slide 64)")
            ax2[0].legend(fontsize=8)
            tha = np.linspace(thc, 89.9, 200)
            dep = np.array([1 / evanescent_alpha(n1, n2, a, lam0 * 1e-9) * 1e9 if evanescent_alpha(n1, n2, a, lam0 * 1e-9) > 0 else np.nan for a in tha])
            ax2[1].plot(tha, dep, color=RED, lw=2)
            ax2[1].plot([th], [1 / evanescent_alpha(n1, n2, th, lam0 * 1e-9) * 1e9 if evanescent_alpha(n1, n2, th, lam0 * 1e-9) > 0 else np.nan], "o", color=INK)
            ax2[1].set_yscale("log")
            ax2[1].set_xlabel("θ (deg)")
            ax2[1].set_ylabel("penetration depth 1/α (nm)")
            ax2[1].set_title("Penetration depth → ∞ at θ_c")
            fig2.tight_layout()
            show(fig2)
    with tab_f:
        st.markdown("Launch a ray into a fiber. Guidance requires the angle of incidence at the core/cladding boundary to satisfy θᵢ ≥ θ_c, "
                    "which is equivalent to sin θ₀ ≤ NA = √(n₁² − n₂²).")
        cl, cr = st.columns([1, 2.6])
        with cl:
            n1 = st.number_input("Core n₁", 1.3, 2.0, 1.48, 0.005, format="%.3f", key="fg1")
            n2 = st.number_input("Cladding n₂", 1.2, 1.99, 1.46, 0.005, format="%.3f", key="fg2")
            if n2 >= n1:
                st.error("Need n₂ < n₁")
                return
            NA = math.sqrt(n1 ** 2 - n2 ** 2)
            th_acc = math.degrees(math.asin(min(1.0, NA)))
            th0 = st.slider("Launch angle θ₀ in air (deg)", 0.0, 60.0, min(10.0, 0.8 * th_acc), 0.5, key="fg_th")
        with cr:
            fig, tir, inc, Rb = draw_fiber_rays(n1, n2, th0)
            show(fig)
        metrics([("NA = √(n₁²−n₂²)", f"{NA:.4f}"), ("Acceptance half-angle", f"{th_acc:.2f}°"),
                 ("Incidence at boundary", f"{inc:.2f}°"), ("θ_c", f"{critical_angle_deg(n1, n2):.2f}°")])
        if tir:
            st.success("Total internal reflection at every bounce → the ray is guided (R = 1, no loss from the boundary).")
        else:
            rem = Rb ** 10
            st.error(f"Below the critical angle each bounce loses power: reflectance ≈ {100 * Rb:.1f} % per bounce (unpolarised), "
                     f"so after 10 bounces only {100 * rem:.2g} % remains.")


# --------------------------------------------------------------------------------------
# Practice problems
# --------------------------------------------------------------------------------------
def _prob_snell(rng):
    n1 = float(rng.choice([1.0, 1.33, 1.45, 1.5]))
    n2 = float(rng.choice([1.0, 1.33, 1.5, 1.7, 2.0, 3.5]))
    while n2 == n1:
        n2 = float(rng.choice([1.0, 1.33, 1.5, 1.7, 2.0, 3.5]))
    while True:
        thi = float(rng.integers(10, 70))
        tt = snell_theta_t(n1, n2, thi)
        if tt is not None:
            break
    return dict(q=f"Light travels from a medium with n₁ = {n1:g} into n₂ = {n2:g} at an angle of incidence θᵢ = {thi:g}°. Find the transmission angle θₜ.",
                ans=tt, unit="deg", tol=0.01,
                sol=rf"$\sin\theta_t=\frac{{n_1}}{{n_2}}\sin\theta_i=\frac{{{n1:g}}}{{{n2:g}}}\sin{thi:g}^\circ={n1 / n2 * math.sin(math.radians(thi)):.4f}\ \Rightarrow\ \theta_t={tt:.2f}^\circ$")


def _prob_crit(rng):
    n1 = float(rng.choice([1.45, 1.48, 1.5, 1.6, 2.0, 3.5]))
    n2 = float(rng.choice([1.0, 1.33, 1.4, 1.44]))
    if n2 >= n1:
        n2 = 1.0
    tc = critical_angle_deg(n1, n2)
    return dict(q=f"Find the critical angle for light going from n₁ = {n1:g} to n₂ = {n2:g}.", ans=tc, unit="deg", tol=0.01,
                sol=rf"$\sin\theta_c=\frac{{n_2}}{{n_1}}=\frac{{{n2:g}}}{{{n1:g}}}={n2 / n1:.4f}\ \Rightarrow\ \theta_c={tc:.2f}^\circ$")


def _prob_fresnel(rng):
    n2 = float(rng.choice([1.33, 1.45, 1.5, 1.7, 2.0, 3.5]))
    R = ((1 - n2) / (1 + n2)) ** 2
    loss = -10 * math.log10(1 - R)
    return dict(q=f"At normal incidence from air (n = 1) onto a material with n = {n2:g}, compute the Fresnel loss in dB (positive number).",
                ans=loss, unit="dB", tol=0.02,
                sol=rf"$R=\left(\frac{{1-{n2:g}}}{{1+{n2:g}}}\right)^2={R:.4f}$, transmitted $={1 - R:.4f}$, loss $=-10\log({1 - R:.4f})={loss:.3f}$ dB")


def _prob_brew(rng):
    n1, n2 = (1.0, float(rng.choice([1.33, 1.45, 1.5, 1.7]))) if rng.random() < 0.5 else (float(rng.choice([1.33, 1.45, 1.5, 1.7])), 1.0)
    tb = brewster_deg(n1, n2)
    return dict(q=f"Find Brewster's angle for parallel polarization going from n₁ = {n1:g} to n₂ = {n2:g}.", ans=tb, unit="deg", tol=0.01,
                sol=rf"$\tan\theta_B=\frac{{n_2}}{{n_1}}=\frac{{{n2:g}}}{{{n1:g}}}={n2 / n1:.4f}\ \Rightarrow\ \theta_B={tb:.2f}^\circ$")


def _prob_lens(rng):
    n = float(rng.choice([1.45, 1.5, 1.6, 1.7]))
    R1 = float(rng.choice([10, 20, 25, 40, 50]))
    R2 = float(rng.choice([10, 20, 25, 40, 50]))
    f = lens_focal_length(n, R1, R2)
    return dict(q=f"A thin lens has n = {n:g}, R₁ = {R1:g} mm and R₂ = {R2:g} mm. Compute its focal length (mm).", ans=f, unit="mm", tol=0.01,
                sol=rf"$\frac1f=(n-1)\left(\frac1{{R_1}}+\frac1{{R_2}}\right)={n - 1:.2f}\left(\frac1{{{R1:g}}}+\frac1{{{R2:g}}}\right)={1 / f:.5f}\ \mathrm{{mm^{{-1}}}}\Rightarrow f={f:.2f}$ mm")


def _prob_na(rng):
    f = float(rng.choice([5, 8, 10, 15, 20]))
    d = float(rng.choice([0.5, 0.8, 1.0, 1.5]))
    na = d / (2 * f)
    return dict(q=f"A receiver has f = {f:g} cm and detector diameter d = {d:g} cm (n₀ = 1). Compute the NA using sin θ ≈ tan θ.", ans=na, unit="", tol=0.03,
                sol=rf"$NA\approx\tan\theta=\frac{{d}}{{2f}}=\frac{{{d:g}}}{{{2 * f:g}}}={na:.4f}$ (θ = {math.degrees(math.asin(na)):.2f}°)")


def _prob_diff(rng):
    lam = float(rng.choice([0.82, 1.0, 1.3, 1.55]))
    fD = float(rng.choice([1.0, 1.5, 2.0, 3.0, 5.0]))
    d = 2.44 * lam * fD
    return dict(q=f"A lens with f/D = {fD:g} focuses a uniform beam of wavelength λ = {lam:g} µm. Find the diameter of the central diffraction spot (µm).",
                ans=d, unit="µm", tol=0.01,
                sol=rf"$d=\frac{{2.44\lambda f}}{{D}}=2.44\times{lam:g}\times{fD:g}={d:.3f}\ \mu\mathrm{{m}}$")


def _prob_lambda(rng):
    lam0 = float(rng.choice([0.82, 1.3, 1.55]))
    n = float(rng.choice([1.45, 1.5, 3.5, 3.6]))
    lam = lam0 / n * 1000
    return dict(q=f"Light of free-space wavelength {lam0:g} µm travels in a medium of index n = {n:g}. Find the wavelength in the medium (nm).",
                ans=lam, unit="nm", tol=0.01,
                sol=rf"$\lambda=\lambda_o/n={lam0 * 1000:g}/{n:g}={lam:.1f}$ nm")


def _prob_df(rng):
    lam = float(rng.choice([820, 1310, 1550]))
    dl = float(rng.choice([1, 2, 5, 20, 50]))
    df = C0 * dl * 1e-9 / (lam * 1e-9) ** 2 / 1e9
    return dict(q=f"A source at λ = {lam:g} nm has spectral width Δλ = {dl:g} nm. Find the frequency bandwidth Δf (GHz), using c = 3×10⁸ m/s.",
                ans=df, unit="GHz", tol=0.02,
                sol=rf"$\Delta f=\frac{{f\,\Delta\lambda}}{{\lambda}}=\frac{{c\,\Delta\lambda}}{{\lambda^2}}=\frac{{3\times10^8\cdot{dl:g}\times10^{{-9}}}}{{({lam * 1e-9:.3e})^2}}={df:.1f}$ GHz")


def _prob_spread(rng):
    lam = float(rng.choice([820, 1550]))
    M = 110.0 if lam == 820 else -15.0
    dl = float(rng.choice([1, 2, 5, 20, 50]))
    L = float(rng.choice([2, 5, 10, 25, 50]))
    dt = abs(M) * dl * L / 1000
    return dict(q=f"A source at {lam / 1000:g} µm (M = {M:g} ps/(nm·km)) has Δλ = {dl:g} nm and the fiber is L = {L:g} km long. Find the pulse spread Δτ (ns).",
                ans=dt, unit="ns", tol=0.02,
                sol=rf"$\Delta(\tau/L)=-M\Delta\lambda$, so $\Delta\tau=L|M|\Delta\lambda={L:g}\times{abs(M):g}\times{dl:g}={dt * 1000:g}$ ps $={dt:g}$ ns")


def _prob_nrz(rng):
    M = float(rng.choice([110.0, -15.0]))
    dl = float(rng.choice([1, 2, 5, 20]))
    L = float(rng.choice([2, 5, 10, 25]))
    dt = abs(M) * dl * L * 1e-12
    R = 0.7 / dt / 1e6
    return dict(q=f"M = {M:g} ps/(nm·km), Δλ = {dl:g} nm, L = {L:g} km. Find the maximum NRZ data rate (Mb/s).", ans=R, unit="Mb/s", tol=0.02,
                sol=rf"$\Delta\tau={abs(M):g}\cdot{dl:g}\cdot{L:g}={dt * 1e12:g}$ ps; $R_{{NRZ}}=0.7/\Delta\tau=0.7/({dt:.3e}\ \mathrm{{s}})={R:.1f}$ Mb/s")


def _prob_f3db(rng):
    dt = float(rng.choice([0.1, 0.5, 1, 2, 5, 10]))
    f = 0.35 / (dt * 1e-9) / 1e6
    return dict(q=f"A link has pulse spread Δτ = {dt:g} ns. Find the electrical 3-dB bandwidth (MHz).", ans=f, unit="MHz", tol=0.02,
                sol=rf"$f_{{3\text{{-dB(el)}}}}=\frac{{0.35}}{{\Delta\tau}}=\frac{{0.35}}{{{dt:g}\ \mathrm{{ns}}}}={f:.1f}$ MHz")


def _prob_cavity(rng):
    L = float(rng.choice([200, 250, 300, 400, 500]))
    n = float(rng.choice([3.4, 3.5, 3.6]))
    lam = float(rng.choice([0.82, 0.85, 1.3]))
    p = cavity_params(L * 1e-6, n, lam * 1e-6)
    return dict(q=f"A laser cavity has L = {L:g} µm, n = {n:g} and λ₀ = {lam:g} µm. Find the longitudinal mode wavelength spacing Δλ_c (nm).",
                ans=p["dlam"] * 1e9, unit="nm", tol=0.02,
                sol=rf"$\Delta\lambda_c=\frac{{\lambda_o^2}}{{2nL}}=\frac{{({lam:g}\ \mu m)^2}}{{2({n:g})({L:g}\ \mu m)}}={p['dlam'] * 1e9:.3f}$ nm")


def _prob_evan(rng):
    n1 = float(rng.choice([1.48, 1.5, 1.6]))
    n2 = float(rng.choice([1.0, 1.33, 1.44]))
    thc = critical_angle_deg(n1, n2)
    th = float(round(thc + rng.uniform(3, 20), 0))
    lam = float(rng.choice([0.85, 1.31, 1.55]))
    a = evanescent_alpha(n1, n2, th, lam * 1e-6)
    return dict(q=f"Light in n₁ = {n1:g} strikes a boundary with n₂ = {n2:g} at θ = {th:g}° (beyond θ_c), λ₀ = {lam:g} µm. Find the penetration depth 1/α of the evanescent field (µm).",
                ans=1 / a * 1e6, unit="µm", tol=0.02,
                sol=rf"$\theta_c={thc:.2f}^\circ$. $\alpha=k_0\sqrt{{n_1^2\sin^2\theta-n_2^2}}=\frac{{2\pi}}{{{lam:g}\ \mu m}}\sqrt{{{n1 ** 2 * math.sin(math.radians(th)) ** 2:.4f}-{n2 ** 2:.4f}}}={a * 1e-6:.4f}\ \mu m^{{-1}}$ → $1/\alpha={1 / a * 1e6:.3f}$ µm")


def _prob_lf(rng):
    x = float(rng.choice([0.2, 0.3, 0.5, 0.71, 0.8, 1.0, 1.2]))
    L = float(freq_dependent_loss_db(x))
    return dict(q=f"A link's optical 3-dB bandwidth is f₃dB. Find the modulation-frequency-dependent optical loss L_f (dB) at f = {x:g}·f₃dB.", ans=L, unit="dB", tol=0.03,
                sol=rf"$L_f=-10\log e^{{-\ln2\,({x:g})^2}}=10\cdot\ln2\cdot{x ** 2:.4f}\cdot\log e={L:.3f}$ dB")


def _prob_dbel(rng):
    o = float(rng.choice([0.5, 1.0, 1.5, 2.0, 3.0, 4.5]))
    return dict(q=f"An optical power loss of {o:g} dB occurs in a link. What is the corresponding loss of electrical power at the detector (dB)?", ans=2 * o, unit="dB", tol=0.01,
                sol=rf"$P_e=R_L\rho^2P^2\Rightarrow dB_{{el}}=2\,dB_{{opt}}=2\times{o:g}={2 * o:g}$ dB")


def _prob_gauss(rng):
    k = float(rng.choice([0.5, 0.7, 1.0, 1.5, 2.0]))
    v = math.exp(-2 * k ** 2) * 100
    return dict(q=f"For a Gaussian beam, what percentage of the peak intensity remains at r = {k:g}·w?", ans=v, unit="%", tol=0.02,
                sol=rf"$I/I_o=e^{{-2r^2/w^2}}=e^{{-2({k:g})^2}}={v / 100:.4f}={v:.2f}\%$")


PROBLEMS = {
    "Snell's law – transmission angle": _prob_snell,
    "Critical angle": _prob_crit,
    "Fresnel loss (air → material)": _prob_fresnel,
    "Brewster angle": _prob_brew,
    "Thin-lens focal length": _prob_lens,
    "Numerical aperture of a receiver": _prob_na,
    "Diffraction-limited spot": _prob_diff,
    "Wavelength in a medium": _prob_lambda,
    "Δλ → Δf": _prob_df,
    "Material-dispersion pulse spread": _prob_spread,
    "Maximum NRZ data rate": _prob_nrz,
    "Electrical 3-dB bandwidth": _prob_f3db,
    "Cavity mode spacing": _prob_cavity,
    "Evanescent penetration depth": _prob_evan,
    "Modulation-frequency loss L_f": _prob_lf,
    "Optical dB ↔ electrical dB": _prob_dbel,
    "Gaussian-beam intensity": _prob_gauss,
}


def page_practice():
    header("Practice Problems", "Randomised numerical questions with worked solutions – answers within 1–3 % are accepted")
    ss = st.session_state
    ss.setdefault("pp_score", [0, 0])
    ss.setdefault("pp_seed", 1)
    ss.setdefault("pp_checked", None)
    kinds = ["Random mix"] + list(PROBLEMS)
    kind = st.selectbox("Topic", kinds, key="pp_kind")

    def new_problem():
        ss["pp_seed"] = int(np.random.default_rng().integers(1, 10 ** 9))
        ss["pp_checked"] = None
        ss["pp_answer"] = ""

    rng = np.random.default_rng(ss["pp_seed"])
    if kind == "Random mix":
        name = list(PROBLEMS)[int(rng.integers(0, len(PROBLEMS)))]
    else:
        name = kind
    prob = PROBLEMS[name](rng)
    st.markdown(f"**{name}**")
    st.markdown(f"> {prob['q']}")
    unit = prob["unit"]
    ans_txt = st.text_input(f"Your answer {'(' + unit + ')' if unit else ''}", key="pp_answer", placeholder="e.g. 12.5 or 1.2e-3")
    c1, c2, c3 = st.columns([1, 1, 3])
    check = c1.button("Check answer", key="pp_check")
    c2.button("New problem", key="pp_new", on_click=new_problem)
    if check:
        try:
            val = float(ans_txt.replace(",", "."))
            ok = abs(val - prob["ans"]) <= prob["tol"] * abs(prob["ans"]) + 1e-12
            ss["pp_checked"] = ("ok" if ok else "bad", val)
            ss["pp_score"][1] += 1
            ss["pp_score"][0] += int(ok)
        except ValueError:
            ss["pp_checked"] = ("nan", None)
    res = ss.get("pp_checked")
    if res:
        if res[0] == "ok":
            st.success(f"Correct! {prob['ans']:.4g} {unit}")
        elif res[0] == "bad":
            st.error(f"Not quite – you entered {res[1]:.4g}. Expected about {prob['ans']:.4g} {unit}.")
        else:
            st.warning("Please enter a number.")
    with st.expander("Show worked solution"):
        st.markdown(prob["sol"])
        st.caption(f"Answer: {prob['ans']:.5g} {unit}")
    c3.write(f"Score: **{ss['pp_score'][0]} / {ss['pp_score'][1]}**")


# --------------------------------------------------------------------------------------
# Formula sheet
# --------------------------------------------------------------------------------------
def page_formulas():
    header("Formula Sheet", "Every equation of the two chapters (slide numbering)")
    st.markdown("#### Chapter 2 – Optics review")
    for tex in (r"n=\frac{c}{v}", r"\theta_r=\theta_i,\quad \frac{\sin\theta_t}{\sin\theta_i}=\frac{n_1}{n_2}\ (2.3)",
                r"\frac1f=(n-1)\left(\frac1{R_1}+\frac1{R_2}\right)\ (2.4),\quad f\#=\frac fD",
                r"\tan\theta=\frac d{2f}\ (2.11),\quad NA=n_o\sin\theta",
                r"d=\frac{2.44\lambda f}{D}\ (2.14)", r"I=I_oe^{-2r^2/w^2}\ (2.5),\quad \text{spot diameter}=2w"):
        st.latex(tex)
    st.markdown("#### Chapter 3 – Lightwave fundamentals")
    for tex in (r"E=E_o\sin(\omega t-kz)\ (3.1),\ k=\tfrac\omega v=k_on=\tfrac{2\pi}\lambda\ (3.6),\ \lambda=\tfrac{\lambda_o}n\ (3.7)",
                r"E=E_oe^{-\alpha z}\sin(\omega t-kz)\ (3.8),\quad S=\tfrac{E^2}{\sqrt{\mu/\varepsilon}}",
                r"\frac{\Delta f}f=\frac{\Delta\lambda}\lambda\ (3.9)",
                r"\Delta(\tau/L)=-M\Delta\lambda\ (3.14),\quad M=\tfrac\lambda cn'',\quad \Delta\tau=L|M|\Delta\lambda",
                r"M=\tfrac{M_o}4\left(\lambda-\tfrac{\lambda_o^4}{\lambda^3}\right),\ M_o=-0.095\ \mathrm{ps/nm^2km},\ \lambda_o=1300\ \mathrm{nm}",
                r"f_{3dB}^{opt}=\tfrac1{2\Delta\tau},\quad f_{3dB}^{el}=\tfrac{0.35}{\Delta\tau}\ (3.19),\quad f\cdot L=\tfrac{0.5}{\Delta(\tau/L)}\ \text{(opt)}",
                r"L_f=-10\log e^{-\ln2(f/f_{3dB})^2}\ (7),\quad dB_{el}=2\,dB_{opt}",
                r"R_{RZ}L=\tfrac{0.35}{\Delta(\tau/L)}\ (3.20),\quad R_{NRZ}L=\tfrac{0.7}{\Delta(\tau/L)}\ (3.21),\quad B_{RZ}\approx R,\ B_{NRZ}\approx R/2",
                r"L=\tfrac{m\lambda}2\ (3.22),\quad \Delta f_c=\tfrac c{2nL}\ (3.25),\quad \Delta\lambda_c=\tfrac{\lambda_o^2\Delta f_c}c\ (3.26)",
                r"\rho=\tfrac{n_1-n_2}{n_1+n_2},\quad \tau=\tfrac{2n_1}{n_1+n_2},\quad R=|\rho|^2,\quad \tan\theta_B=\tfrac{n_2}{n_1}\ (3.29\text{–}3.30)",
                r"\sin\theta_c=\tfrac{n_2}{n_1}\ (3.34),\quad \alpha=k_0\sqrt{n_1^2\sin^2\theta-n_2^2}"):
        st.latex(tex)


# --------------------------------------------------------------------------------------
# Navigation
# --------------------------------------------------------------------------------------
PAGES = {
    "Home": page_home,
    "Ch 2 · 2.1 Refraction & Snell's law": page_snell,
    "Ch 2 · 2.2 Lenses & GRIN rods": page_lenses,
    "Ch 2 · 2.3 Numerical aperture": page_na,
    "Ch 2 · 2.4 Diffraction & Gaussian beams": page_diffraction,
    "Ch 3 · 3.1 EM waves & attenuation": page_em,
    "Ch 3 · 3.2 Source spectrum": page_spectrum,
    "Ch 3 · 3.3 Material dispersion": page_dispersion,
    "Ch 3 · 3.4 Bandwidth & data rate": page_bandwidth,
    "Ch 3 · 3.5 Polarization": page_polarization,
    "Ch 3 · 3.6 Resonant cavities": page_cavity,
    "Ch 3 · 3.7 Fresnel reflection": page_fresnel,
    "Ch 3 · 3.8 Critical angle & evanescent wave": page_tir,
    "Practice problems": page_practice,
    "Formula sheet": page_formulas,
}


def main():
    st.set_page_config(page_title="Fiber-Optics Concept Lab", page_icon="🔦", layout="wide")
    st.markdown(
        "<style>div[data-testid='stMetric']{background:#f4f7fa;border:1px solid #d9e0e7;border-radius:8px;padding:8px 12px;}"
        "div[data-testid='stMetricValue']{font-size:1.25rem;}</style>", unsafe_allow_html=True)
    st.sidebar.title("🔦 Fiber-Optics Lab")
    st.sidebar.caption("Optics Review (Ch 2) · Lightwave Fundamentals (Ch 3)")
    page = st.sidebar.radio("Module", list(PAGES), key="nav")
    st.sidebar.markdown("---")
    st.sidebar.caption("Equations follow the lecture slides. c = 3×10⁸ m/s.")
    PAGES[page]()


if __name__ == "__main__":
    main()
