"""
MLHP Chapter 1 Animation #7: Gaussian Process Prior Samples
Shows GP samples with varying length-scale and the connection
to preference modeling via Bradley-Terry.

Run with:
    PATH="/lfs/local/0/sttruong/miniconda3/bin:$PATH" \
    manim -qh --disable_caching --media_dir media/ch1 animations/ch1/gp_prior_samples.py GPPriorSamples
"""

from manim import *
import numpy as np

# ── design tokens ───────────────────────────────────────────────────
ACCENT = "#FFD966"
BG = "#0f0f0f"
TEXT2 = "#aaaaaa"
PAL = ["#5B8DEE", "#45BF7C", "#F0A35C", "#E8637A", "#B07CD8"]
AXIS_CLR = "#888888"


def rbf_kernel(x1, x2, sigma_f=1.0, length_scale=1.0):
    """RBF (squared exponential) kernel matrix."""
    sq_dist = (x1[:, None] - x2[None, :]) ** 2
    return sigma_f ** 2 * np.exp(-sq_dist / (2 * length_scale ** 2))


def matern12_kernel(x1, x2, sigma_f=1.0, length_scale=1.0):
    """Matern-1/2 (exponential) kernel matrix."""
    dist = np.abs(x1[:, None] - x2[None, :])
    return sigma_f ** 2 * np.exp(-dist / length_scale)


def periodic_kernel(x1, x2, sigma_f=1.0, length_scale=1.0, period=2.0):
    """Periodic kernel matrix."""
    dist = np.abs(x1[:, None] - x2[None, :])
    return sigma_f ** 2 * np.exp(
        -2 * np.sin(np.pi * dist / period) ** 2 / length_scale ** 2
    )


def sample_gp(x, length_scale=1.0, n_samples=1, seed=None):
    """Sample functions from a GP prior with RBF kernel."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    K = rbf_kernel(x, x, length_scale=length_scale)
    K += 1e-6 * np.eye(len(x))  # numerical stability
    L = np.linalg.cholesky(K)
    z = rng.randn(len(x), n_samples)
    return (L @ z).T  # shape: (n_samples, len(x))


class GPPriorSamples(Scene):
    def construct(self):
        self.camera.background_color = BG
        np.random.seed(42)

        self.play_title()
        self.play_gp_concept()
        self.play_fixed_samples()
        self.play_length_scale_morph()
        self.play_bt_connection()
        self.play_kernel_comparison()
        self.play_gp_posterior()
        self.play_applications()
        self.play_pipeline_summary()

    # ── title ───────────────────────────────────────────────────────
    def play_title(self):
        t = Text("Gaussian Process Priors", font_size=44,
                 color=WHITE, weight=BOLD)
        st = Text("Flexible nonparametric reward models",
                  font_size=24, color=TEXT2)
        st.next_to(t, DOWN, buff=0.35)
        line = Line(LEFT * 2.5, RIGHT * 2.5, color=ACCENT,
                    stroke_width=1.5)
        line.next_to(st, DOWN, buff=0.3)
        g = VGroup(t, st, line)
        self.play(FadeIn(t, shift=UP * 0.2), run_time=0.8)
        self.play(FadeIn(st), Create(line), run_time=0.6)
        self.wait(2.5)
        self.play(FadeOut(g, shift=UP * 0.4), run_time=0.7)

    # ── GP concept and formulas ─────────────────────────────────────
    def play_gp_concept(self):
        self.gp_header = Text("GP Prior with RBF Kernel", font_size=32,
                              color=WHITE, weight=BOLD)
        self.gp_header.to_edge(UP, buff=0.35)
        self.play(FadeIn(self.gp_header, shift=DOWN * 0.1), run_time=0.6)

        gp_formula = MathTex(
            r"r(x) \sim \mathcal{GP}(0, k)",
            font_size=30, color=ACCENT,
        )
        rbf_formula = MathTex(
            r"k(x, x') = \sigma_f^2 \exp\!\left("
            r"-\frac{\|x - x'\|^2}{2\ell^2}\right)",
            font_size=26, color=ACCENT,
        )
        formulas = VGroup(gp_formula, rbf_formula).arrange(DOWN, buff=0.2)
        formulas.next_to(self.gp_header, DOWN, buff=0.2)
        self.play(Write(gp_formula), run_time=0.8)
        self.play(Write(rbf_formula), run_time=1.0)
        self.wait(15.0)

        self.gp_formulas = formulas

        # Build axes
        self.ax = Axes(
            x_range=[-3, 3, 1], y_range=[-3, 3, 1],
            x_length=9, y_length=4,
            axis_config={"color": AXIS_CLR, "include_numbers": True,
                         "font_size": 18},
            tips=False,
        )
        self.ax.move_to(DOWN * 0.9)
        self.x_lab = self.ax.get_x_axis_label(
            MathTex(r"x", font_size=24, color=AXIS_CLR),
            edge=RIGHT, direction=DOWN,
        )
        self.y_lab = self.ax.get_y_axis_label(
            MathTex(r"r(x)", font_size=24, color=AXIS_CLR),
            edge=UP, direction=LEFT,
        )
        self.play(Create(self.ax), FadeIn(self.x_lab), FadeIn(self.y_lab),
                  run_time=0.7)

    # ── fixed length-scale samples ──────────────────────────────────
    def play_fixed_samples(self):
        x_pts = np.linspace(-3, 3, 200)
        samples = sample_gp(x_pts, length_scale=1.0, n_samples=5, seed=42)

        ls_label = MathTex(r"\ell = 1.0", font_size=26, color=ACCENT)
        ls_label.next_to(self.ax, RIGHT, buff=0.3).shift(UP * 1.0)
        self.play(FadeIn(ls_label), run_time=0.4)

        self.sample_curves = VGroup()
        for i in range(5):
            y_vals = samples[i]
            # Clamp to axis range for clean display
            y_vals = np.clip(y_vals, -2.9, 2.9)

            curve = self.ax.plot_line_graph(
                x_values=x_pts,
                y_values=y_vals,
                add_vertex_dots=False,
                line_color=PAL[i % len(PAL)],
                stroke_width=2.5,
            )
            self.sample_curves.add(curve)
            self.play(Create(curve), run_time=0.8)

        self.wait(15.0)

        # Clean up for morph
        self.play(
            FadeOut(self.sample_curves), FadeOut(ls_label),
            run_time=0.5,
        )

    # ── length-scale morphing ───────────────────────────────────────
    def play_length_scale_morph(self):
        x_pts = np.linspace(-3, 3, 200)

        # Pre-compute samples at many length-scales using same random seed
        # We use a fixed set of standard normal draws for consistency
        rng = np.random.RandomState(123)
        z = rng.randn(len(x_pts))  # single draw

        ls_values = np.concatenate([
            np.linspace(3.0, 0.3, 50),
            np.linspace(0.3, 1.0, 25),
        ])

        # Pre-compute all curves
        all_y = []
        for ls in ls_values:
            K = rbf_kernel(x_pts, x_pts, length_scale=ls)
            K += 1e-6 * np.eye(len(x_pts))
            L = np.linalg.cholesky(K)
            y = L @ z
            y = np.clip(y, -2.9, 2.9)
            all_y.append(y)

        # Use ValueTracker to index into precomputed curves
        idx_tracker = ValueTracker(0)

        def get_curve():
            idx = int(np.clip(idx_tracker.get_value(), 0, len(all_y) - 1))
            return self.ax.plot_line_graph(
                x_values=x_pts,
                y_values=all_y[idx],
                add_vertex_dots=False,
                line_color=PAL[0],
                stroke_width=3,
            )

        def get_ls_label():
            idx = int(np.clip(idx_tracker.get_value(), 0, len(ls_values) - 1))
            ls = ls_values[idx]
            return MathTex(
                rf"\ell = {ls:.2f}",
                font_size=28, color=ACCENT,
            ).next_to(self.ax, RIGHT, buff=0.3).shift(UP * 1.0)

        curve = always_redraw(get_curve)
        ls_label = always_redraw(get_ls_label)

        self.add(curve, ls_label)
        self.wait(0.5)

        note = Text(
            "Large \u2113 = smooth    Small \u2113 = wiggly",
            font_size=20, color=TEXT2,
        )
        note.to_edge(DOWN, buff=0.2)
        self.play(FadeIn(note), run_time=0.4)

        # Animate from smooth (l=3.0) to wiggly (l=0.3)
        self.play(
            idx_tracker.animate.set_value(49),
            run_time=4.0, rate_func=smooth,
        )
        self.wait(1.5)

        # Back to l=1.0
        self.play(
            idx_tracker.animate.set_value(74),
            run_time=2.0, rate_func=smooth,
        )
        self.wait(10.0)

        self.play(FadeOut(curve), FadeOut(ls_label), FadeOut(note),
                  run_time=0.5)

    # ── BT connection ───────────────────────────────────────────────
    def play_bt_connection(self):
        bt_formula = MathTex(
            r"P(A \succ B \mid r) = \sigma(r(x_A) - r(x_B))",
            font_size=30, color=ACCENT,
        )
        bt_formula.move_to(DOWN * 0.5)

        note = Text(
            "GP + Bradley-Terry = flexible preference model",
            font_size=22, color=WHITE, weight=BOLD,
        )
        note.next_to(bt_formula, DOWN, buff=0.4)

        self.play(Write(bt_formula), run_time=1.0)
        self.play(FadeIn(note, shift=UP * 0.1), run_time=0.6)
        self.wait(20.0)

        # Transition: fade BT but keep header + axes
        self.play(FadeOut(bt_formula), FadeOut(note), run_time=0.5)

    # ── kernel comparison ─────────────────────────────────────────
    def play_kernel_comparison(self):
        """Show samples from RBF, Matern-1/2, and Periodic kernels."""
        # Update header
        self.play(
            self.gp_header.animate.become(
                Text("Kernel Comparison", font_size=32,
                     color=WHITE, weight=BOLD).to_edge(UP, buff=0.35)
            ),
            FadeOut(self.gp_formulas),
            run_time=0.6,
        )

        x_pts = np.linspace(-3, 3, 200)
        kernels = [
            ("RBF", rbf_kernel, PAL[0], r"k_{\mathrm{RBF}}"),
            ("Mat\u00e9rn-1/2", matern12_kernel, PAL[1], r"k_{\mathrm{Mat}}"),
            ("Periodic", lambda x1, x2: periodic_kernel(x1, x2, period=2.0),
             PAL[3], r"k_{\mathrm{Per}}"),
        ]

        curves = VGroup()
        labels = VGroup()
        rng = np.random.RandomState(77)

        for i, (name, kern_fn, color, tex) in enumerate(kernels):
            K = kern_fn(x_pts, x_pts)
            K += 1e-6 * np.eye(len(x_pts))
            L = np.linalg.cholesky(K)
            y = np.clip(L @ rng.randn(len(x_pts)), -2.9, 2.9)

            curve = self.ax.plot_line_graph(
                x_values=x_pts, y_values=y,
                add_vertex_dots=False,
                line_color=color, stroke_width=2.5,
            )
            lbl = MathTex(tex, font_size=22, color=color)
            lbl.next_to(self.ax, RIGHT, buff=0.3).shift(UP * (1.0 - i * 0.5))

            curves.add(curve)
            labels.add(lbl)
            self.play(Create(curve), FadeIn(lbl), run_time=0.8)
            self.wait(3.0)

        desc = Text(
            "Kernel choice encodes prior beliefs about smoothness",
            font_size=20, color=TEXT2,
        )
        desc.to_edge(DOWN, buff=0.2)
        self.play(FadeIn(desc), run_time=0.4)
        self.wait(15.0)

        self.play(FadeOut(curves), FadeOut(labels), FadeOut(desc), run_time=0.5)

    # ── GP posterior visualization ────────────────────────────────
    def play_gp_posterior(self):
        """Show GP posterior with observed data, mean, and uncertainty."""
        self.play(
            self.gp_header.animate.become(
                Text("GP Posterior", font_size=32,
                     color=WHITE, weight=BOLD).to_edge(UP, buff=0.35)
            ),
            run_time=0.5,
        )

        x_pts = np.linspace(-3, 3, 200)
        # Observed data
        x_obs = np.array([-2.0, -0.5, 0.5, 1.5, 2.5])
        y_obs = np.array([0.8, -0.5, 1.0, 0.2, -0.8])

        # GP posterior computation
        ls = 1.0
        sigma_n = 0.1
        K_ss = rbf_kernel(x_pts, x_pts, length_scale=ls)
        K_xs = rbf_kernel(x_obs, x_pts, length_scale=ls)
        K_xx = rbf_kernel(x_obs, x_obs, length_scale=ls)
        K_xx += sigma_n ** 2 * np.eye(len(x_obs))

        K_xx_inv = np.linalg.solve(K_xx, np.eye(len(x_obs)))
        mu = K_xs.T @ K_xx_inv @ y_obs
        cov = K_ss - K_xs.T @ K_xx_inv @ K_xs
        std = np.sqrt(np.clip(np.diag(cov), 0, None))

        mu = np.clip(mu, -2.9, 2.9)
        upper = np.clip(mu + 2 * std, -2.9, 2.9)
        lower = np.clip(mu - 2 * std, -2.9, 2.9)

        # Draw uncertainty band
        band_pts = []
        for i in range(len(x_pts)):
            band_pts.append(self.ax.c2p(x_pts[i], upper[i]))
        for i in range(len(x_pts) - 1, -1, -1):
            band_pts.append(self.ax.c2p(x_pts[i], lower[i]))

        band = Polygon(
            *band_pts,
            fill_color=PAL[0], fill_opacity=0.2,
            stroke_width=0,
        )
        self.play(FadeIn(band), run_time=0.8)

        # Draw posterior mean
        mean_curve = self.ax.plot_line_graph(
            x_values=x_pts, y_values=mu,
            add_vertex_dots=False,
            line_color=PAL[0], stroke_width=3,
        )
        self.play(Create(mean_curve), run_time=0.8)

        # Draw observed data points
        dots = VGroup()
        for xi, yi in zip(x_obs, y_obs):
            dot = Dot(self.ax.c2p(xi, yi), radius=0.08, color=ACCENT)
            dots.add(dot)
        self.play(FadeIn(dots, scale=0.5), run_time=0.6)

        # Labels
        obs_lbl = Text("Observed data", font_size=18, color=ACCENT)
        obs_lbl.next_to(self.ax, RIGHT, buff=0.3).shift(UP * 1.0)
        band_lbl = MathTex(r"\pm 2\sigma", font_size=22, color=PAL[0])
        band_lbl.next_to(obs_lbl, DOWN, buff=0.25)
        desc = Text(
            "Uncertainty shrinks near observations",
            font_size=20, color=TEXT2,
        )
        desc.to_edge(DOWN, buff=0.2)
        self.play(FadeIn(obs_lbl), FadeIn(band_lbl), FadeIn(desc), run_time=0.5)
        self.wait(20.0)

        self.play(
            FadeOut(band), FadeOut(mean_curve), FadeOut(dots),
            FadeOut(obs_lbl), FadeOut(band_lbl), FadeOut(desc),
            run_time=0.5,
        )

    # ── application domains ───────────────────────────────────────
    def play_applications(self):
        """Progressive reveal of GP application domains."""
        self.play(
            self.gp_header.animate.become(
                Text("Applications", font_size=32,
                     color=WHITE, weight=BOLD).to_edge(UP, buff=0.35)
            ),
            FadeOut(self.ax), FadeOut(self.x_lab), FadeOut(self.y_lab),
            run_time=0.5,
        )

        apps = [
            ("Robotics", "Learning reward from human demonstrations", PAL[0]),
            ("Bayesian Optimization", "Sample-efficient hyperparameter tuning", PAL[1]),
            ("RLHF / DPO", "Preference learning for language models", PAL[2]),
            ("Drug Discovery", "Molecular property prediction from comparisons", PAL[3]),
        ]

        bullets = VGroup()
        for name, desc, color in apps:
            title = Text(name, font_size=26, color=color, weight=BOLD)
            sub = Text(desc, font_size=20, color=TEXT2)
            sub.next_to(title, DOWN, buff=0.1, aligned_edge=LEFT)
            item = VGroup(title, sub)
            bullets.add(item)

        bullets.arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        bullets.move_to(ORIGIN)

        for item in bullets:
            self.play(FadeIn(item, shift=RIGHT * 0.2), run_time=0.5)
            self.wait(3.0)

        self.wait(8.0)

        # Computational note
        comp = MathTex(
            r"\mathcal{O}(N^3) \rightarrow \mathcal{O}(NM^2)"
            r"\;\mathrm{with\;inducing\;points}",
            font_size=24, color=TEXT2,
        )
        comp.to_edge(DOWN, buff=0.3)
        self.play(FadeIn(comp, shift=UP * 0.1), run_time=0.5)
        self.wait(10.0)

        self.play(FadeOut(bullets), FadeOut(comp), run_time=0.5)

    # ── pipeline summary ──────────────────────────────────────────
    def play_pipeline_summary(self):
        """Three-line GP preference model pipeline."""
        self.play(
            self.gp_header.animate.become(
                Text("GP Preference Model", font_size=32,
                     color=WHITE, weight=BOLD).to_edge(UP, buff=0.35)
            ),
            run_time=0.5,
        )

        lines = [
            (r"r(x) \sim \mathcal{GP}(0, k)", "Prior", PAL[0]),
            (r"P(A \succ B) = \sigma\!\big(r(x_A) - r(x_B)\big)",
             "Likelihood", PAL[1]),
            (r"r \mid \mathcal{D} \;\;\text{(posterior)}",
             "Posterior", PAL[2]),
        ]

        pipeline = VGroup()
        arrows = VGroup()
        for i, (tex, label, color) in enumerate(lines):
            eq = MathTex(tex, font_size=28, color=color)
            lbl = Text(label, font_size=18, color=TEXT2)
            lbl.next_to(eq, RIGHT, buff=0.4)
            row = VGroup(eq, lbl)
            pipeline.add(row)

        pipeline.arrange(DOWN, buff=0.6)
        pipeline.move_to(ORIGIN)

        # Add connecting arrows
        for i in range(len(pipeline) - 1):
            arr = Arrow(
                pipeline[i].get_bottom() + DOWN * 0.05,
                pipeline[i + 1].get_top() + UP * 0.05,
                color=AXIS_CLR, stroke_width=2, max_tip_length_to_length_ratio=0.15,
            )
            arrows.add(arr)

        for i, row in enumerate(pipeline):
            self.play(FadeIn(row, shift=DOWN * 0.1), run_time=0.6)
            if i < len(arrows):
                self.play(Create(arrows[i]), run_time=0.3)
            self.wait(2.0)

        self.wait(20.0)
