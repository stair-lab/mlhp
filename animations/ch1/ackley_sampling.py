"""
MLHP Chapter 1 Animation #3: Stochastic Utility & the Ackley Landscape
Shows the Ackley function as a utility landscape with accept-reject
sampling and pairwise comparisons.

Run with:
    PATH="/lfs/local/0/sttruong/miniconda3/bin:$PATH" \
    manim -qh --disable_caching --media_dir media/ch1 animations/ch1/ackley_sampling.py AckleySampling
"""

from manim import *
import numpy as np

# ── design tokens ───────────────────────────────────────────────────
ACCENT = "#FFD966"
BG = "#0f0f0f"
TEXT2 = "#aaaaaa"
PAL = ["#5B8DEE", "#45BF7C", "#F0A35C", "#E8637A", "#B07CD8"]
AXIS_CLR = "#888888"

ACCEPT_CLR = PAL[0]  # blue
REJECT_CLR = PAL[3]  # red
WINNER_CLR = PAL[1]  # green
LOSER_CLR = PAL[4]   # purple


def ackley(x1, x2):
    """Ackley function (2D)."""
    a, b, c = 20, 0.2, 2 * np.pi
    term1 = -a * np.exp(-b * np.sqrt(0.5 * (x1**2 + x2**2)))
    term2 = -np.exp(0.5 * (np.cos(c * x1) + np.cos(c * x2)))
    return term1 + term2 + a + np.e


class AckleySampling(Scene):
    def construct(self):
        self.camera.background_color = BG
        np.random.seed(42)

        self.play_title()
        self.play_landscape()
        self.play_accept_reject()
        self.play_pairwise()
        self.play_closing()
        self.play_noise_distributions()
        self.play_utility_decomposition()
        self.play_teaser()

    # ── title ───────────────────────────────────────────────────────
    def play_title(self):
        t = Text("Stochastic Utility Models", font_size=44,
                 color=WHITE, weight=BOLD)
        st = Text("The Ackley function as a utility landscape",
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

    # ── landscape contour as pixel image ────────────────────────────
    def play_landscape(self):
        self.header = Text("Ackley Utility Landscape", font_size=32,
                           color=WHITE, weight=BOLD)
        self.header.to_edge(UP, buff=0.35)
        self.play(FadeIn(self.header, shift=DOWN * 0.1), run_time=0.6)

        # Build axes
        self.ax = Axes(
            x_range=[-2, 2, 1], y_range=[-2, 2, 1],
            x_length=6, y_length=6,
            axis_config={"color": AXIS_CLR, "include_numbers": True,
                         "font_size": 18},
            tips=False,
        )
        self.ax.move_to(DOWN * 0.2)

        x_lab = self.ax.get_x_axis_label(
            MathTex(r"x_1", font_size=24, color=AXIS_CLR),
            edge=RIGHT, direction=DOWN,
        )
        y_lab = self.ax.get_y_axis_label(
            MathTex(r"x_2", font_size=24, color=AXIS_CLR),
            edge=UP, direction=LEFT,
        )

        # Create heatmap as pixel array
        res = 200
        xs = np.linspace(-2, 2, res)
        ys = np.linspace(-2, 2, res)
        X, Y = np.meshgrid(xs, ys)
        Z = ackley(X, Y)

        # Normalize to [0, 1] for colormap
        z_min, z_max = Z.min(), Z.max()
        Z_norm = (Z - z_min) / (z_max - z_min)

        # Create color array (warm colormap: dark -> orange -> cream)
        pixels = np.zeros((res, res, 4), dtype=np.uint8)
        for i in range(res):
            for j in range(res):
                v = Z_norm[res - 1 - i, j]  # flip y
                # Dark (low utility) to warm (high utility)
                r = int(40 + 200 * v)
                g = int(20 + 140 * v)
                b = int(20 + 60 * v)
                pixels[i, j] = [r, g, b, 200]

        heatmap = ImageMobject(pixels)
        # Scale to match axes
        heatmap.set_width(self.ax.x_length)
        heatmap.set_height(self.ax.y_length)
        heatmap.move_to(self.ax.get_center())

        self.play(FadeIn(heatmap), run_time=0.8)
        self.play(Create(self.ax), FadeIn(x_lab), FadeIn(y_lab),
                  run_time=0.7)

        note = Text("Bright = high utility, Dark = low utility",
                     font_size=18, color=TEXT2)
        note.to_edge(DOWN, buff=0.2)
        self.play(FadeIn(note), run_time=0.4)
        self.wait(15.0)
        self.play(FadeOut(note), run_time=0.3)

        self.heatmap = heatmap
        self.x_lab = x_lab
        self.y_lab = y_lab

    # ── accept-reject sampling ──────────────────────────────────────
    def play_accept_reject(self):
        note = Text("Accept-Reject Sampling", font_size=22,
                     color=ACCENT, weight=BOLD)
        note.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(note), run_time=0.4)

        threshold_note = Text(
            "Accept (blue) if utility > threshold, Reject (red) otherwise",
            font_size=18, color=TEXT2,
        )
        threshold_note.to_edge(DOWN, buff=0.2)
        self.play(FadeIn(threshold_note), run_time=0.4)

        threshold = 4.0  # midpoint of Ackley range
        dots = VGroup()

        # Slow phase: first 8 items
        for i in range(8):
            x1 = np.random.uniform(-2, 2)
            x2 = np.random.uniform(-2, 2)
            u = ackley(x1, x2)

            color = ACCEPT_CLR if u > threshold else REJECT_CLR
            dot = Dot(
                self.ax.c2p(x1, x2),
                color=color, radius=0.06,
            )
            dots.add(dot)
            self.play(FadeIn(dot, scale=2.0), run_time=0.4)

        # Fast phase: 25 more items
        fast_dots = VGroup()
        for i in range(25):
            x1 = np.random.uniform(-2, 2)
            x2 = np.random.uniform(-2, 2)
            u = ackley(x1, x2)
            color = ACCEPT_CLR if u > threshold else REJECT_CLR
            dot = Dot(
                self.ax.c2p(x1, x2),
                color=color, radius=0.06,
            )
            fast_dots.add(dot)

        self.play(
            *[FadeIn(d, scale=1.5) for d in fast_dots],
            run_time=1.5,
        )
        dots.add(*fast_dots)
        self.wait(15.0)

        # Clean up dots
        self.play(FadeOut(dots), FadeOut(note), FadeOut(threshold_note),
                  run_time=0.5)
        self.dots_group = dots

    # ── pairwise comparisons ────────────────────────────────────────
    def play_pairwise(self):
        note = Text("Pairwise Comparisons", font_size=22,
                     color=ACCENT, weight=BOLD)
        note.to_edge(DOWN, buff=0.5)

        pw_note = Text(
            "Winner (green) has higher utility than loser (purple)",
            font_size=18, color=TEXT2,
        )
        pw_note.to_edge(DOWN, buff=0.2)
        self.play(FadeIn(note), FadeIn(pw_note), run_time=0.4)

        pairs_shown = VGroup()

        for _ in range(6):
            # Sample two random items
            x1a, x2a = np.random.uniform(-2, 2), np.random.uniform(-2, 2)
            x1b, x2b = np.random.uniform(-2, 2), np.random.uniform(-2, 2)
            ua = ackley(x1a, x2a)
            ub = ackley(x1b, x2b)

            win_color = WINNER_CLR
            lose_color = LOSER_CLR

            if ua >= ub:
                dot_a = Dot(self.ax.c2p(x1a, x2a), color=win_color,
                           radius=0.08)
                dot_b = Dot(self.ax.c2p(x1b, x2b), color=lose_color,
                           radius=0.06)
            else:
                dot_a = Dot(self.ax.c2p(x1a, x2a), color=lose_color,
                           radius=0.06)
                dot_b = Dot(self.ax.c2p(x1b, x2b), color=win_color,
                           radius=0.08)

            line = Line(
                self.ax.c2p(x1a, x2a), self.ax.c2p(x1b, x2b),
                color=TEXT2, stroke_width=1, stroke_opacity=0.5,
            )

            pair = VGroup(line, dot_a, dot_b)
            pairs_shown.add(pair)
            self.play(FadeIn(pair), run_time=0.5)
            self.wait(0.5)

        self.wait(15.0)

        # Clean up
        self.play(
            FadeOut(pairs_shown), FadeOut(note), FadeOut(pw_note),
            run_time=0.5,
        )

    # ── closing formula ─────────────────────────────────────────────
    def play_closing(self):
        formula = MathTex(
            r"\tilde{H}_j = V_j + \varepsilon_j",
            font_size=32, color=ACCENT,
        )
        formula.to_edge(DOWN, buff=0.5)

        note = Text(
            "Same landscape, different data collection methods",
            font_size=20, color=TEXT2,
        )
        note.to_edge(DOWN, buff=0.2)

        self.play(Write(formula), run_time=0.8)
        self.play(FadeIn(note), run_time=0.4)
        self.wait(20.0)

        # Transition out of landscape
        self.play(
            FadeOut(self.heatmap), FadeOut(self.ax),
            FadeOut(self.x_lab), FadeOut(self.y_lab),
            FadeOut(formula), FadeOut(note),
            run_time=0.5,
        )

    # ── noise distribution comparison ─────────────────────────────
    def play_noise_distributions(self):
        """Show Gaussian, Gumbel, and Laplace noise distributions."""
        self.play(
            self.header.animate.become(
                Text("Noise Distributions", font_size=32,
                     color=WHITE, weight=BOLD).to_edge(UP, buff=0.35)
            ),
            run_time=0.5,
        )

        ax = Axes(
            x_range=[-5, 5, 1], y_range=[0, 0.5, 0.1],
            x_length=9, y_length=3.5,
            axis_config={"color": AXIS_CLR, "include_numbers": True,
                         "font_size": 16},
            tips=False,
        )
        ax.move_to(DOWN * 0.5)
        x_lbl = ax.get_x_axis_label(
            MathTex(r"\varepsilon", font_size=24, color=AXIS_CLR),
            edge=RIGHT, direction=DOWN,
        )
        self.play(Create(ax), FadeIn(x_lbl), run_time=0.5)

        # Gaussian PDF
        gauss_curve = ax.plot(
            lambda x: (1 / np.sqrt(2 * np.pi)) * np.exp(-x**2 / 2),
            x_range=[-4.5, 4.5], color=PAL[0], stroke_width=2.5,
        )
        gauss_lbl = MathTex(
            r"\mathrm{Gaussian} \rightarrow \mathrm{Probit}",
            font_size=20, color=PAL[0],
        )
        gauss_lbl.next_to(ax, RIGHT, buff=0.3).shift(UP * 1.0)
        self.play(Create(gauss_curve), FadeIn(gauss_lbl), run_time=0.8)
        self.wait(4.0)

        # Gumbel PDF
        def gumbel_pdf(x):
            z = x  # loc=0, scale=1
            return np.exp(-(z + np.exp(-z)))

        gumbel_curve = ax.plot(
            gumbel_pdf, x_range=[-4.5, 4.5], color=PAL[2], stroke_width=2.5,
        )
        gumbel_lbl = MathTex(
            r"\mathrm{Gumbel} \rightarrow \mathrm{Softmax}",
            font_size=20, color=PAL[2],
        )
        gumbel_lbl.next_to(gauss_lbl, DOWN, buff=0.25)
        self.play(Create(gumbel_curve), FadeIn(gumbel_lbl), run_time=0.8)
        self.wait(4.0)

        # Logistic PDF
        def logistic_pdf(x):
            ex = np.exp(-x)
            return ex / (1 + ex)**2

        logistic_curve = ax.plot(
            logistic_pdf, x_range=[-4.5, 4.5], color=PAL[3], stroke_width=2.5,
        )
        logistic_lbl = MathTex(
            r"\mathrm{Logistic} \rightarrow \mathrm{BT/Rasch}",
            font_size=20, color=PAL[3],
        )
        logistic_lbl.next_to(gumbel_lbl, DOWN, buff=0.25)
        self.play(Create(logistic_curve), FadeIn(logistic_lbl), run_time=0.8)
        self.wait(4.0)

        desc = Text(
            "Different noise assumptions lead to different choice models",
            font_size=20, color=TEXT2,
        )
        desc.to_edge(DOWN, buff=0.2)
        self.play(FadeIn(desc), run_time=0.4)
        self.wait(12.0)

        self.play(
            FadeOut(ax), FadeOut(x_lbl),
            FadeOut(gauss_curve), FadeOut(gumbel_curve), FadeOut(logistic_curve),
            FadeOut(gauss_lbl), FadeOut(gumbel_lbl), FadeOut(logistic_lbl),
            FadeOut(desc),
            run_time=0.5,
        )

    # ── utility decomposition ─────────────────────────────────────
    def play_utility_decomposition(self):
        """Number line showing V_j + epsilon samples."""
        self.play(
            self.header.animate.become(
                Text("Utility Decomposition", font_size=32,
                     color=WHITE, weight=BOLD).to_edge(UP, buff=0.35)
            ),
            run_time=0.5,
        )

        # Number line
        nline = NumberLine(
            x_range=[-3, 6, 1], length=10,
            color=AXIS_CLR, include_numbers=True,
            font_size=18,
        )
        nline.move_to(DOWN * 0.3)
        nline_lbl = MathTex(r"\tilde{H}_j", font_size=24, color=AXIS_CLR)
        nline_lbl.next_to(nline, RIGHT, buff=0.3)
        self.play(Create(nline), FadeIn(nline_lbl), run_time=0.6)

        # Fixed utility V_j
        vj_val = 2.0
        vj_dot = Dot(nline.n2p(vj_val), radius=0.12, color=ACCENT)
        vj_lbl = MathTex(r"V_j = 2.0", font_size=24, color=ACCENT)
        vj_lbl.next_to(vj_dot, UP, buff=0.3)
        self.play(FadeIn(vj_dot, scale=1.5), FadeIn(vj_lbl), run_time=0.6)
        self.wait(3.0)

        # Epsilon samples
        rng = np.random.RandomState(99)
        eps_samples = rng.standard_normal(12)
        eps_dots = VGroup()

        for i, eps in enumerate(eps_samples):
            h_val = vj_val + eps
            if -3 < h_val < 6:
                dot = Dot(
                    nline.n2p(h_val), radius=0.06,
                    color=PAL[0], fill_opacity=0.6,
                )
                eps_dots.add(dot)

        self.play(
            *[FadeIn(d, scale=2.0) for d in eps_dots],
            run_time=1.0,
        )

        eps_note = MathTex(
            r"\tilde{H}_j = V_j + \varepsilon_j"
            r"\quad (\varepsilon_j \sim F)",
            font_size=24, color=TEXT2,
        )
        eps_note.next_to(nline, DOWN, buff=0.5)
        self.play(FadeIn(eps_note), run_time=0.5)
        self.wait(5.0)

        desc = Text(
            "Each choice occasion = one noisy utility draw",
            font_size=20, color=WHITE, weight=BOLD,
        )
        desc.to_edge(DOWN, buff=0.2)
        self.play(FadeIn(desc, shift=UP * 0.1), run_time=0.5)
        self.wait(12.0)

        self.play(
            FadeOut(nline), FadeOut(nline_lbl),
            FadeOut(vj_dot), FadeOut(vj_lbl),
            FadeOut(eps_dots), FadeOut(eps_note), FadeOut(desc),
            run_time=0.5,
        )

    # ── teaser for softmax ────────────────────────────────────────
    def play_teaser(self):
        """Preview: Gumbel noise leads to softmax."""
        self.play(
            self.header.animate.become(
                Text("Key Insight", font_size=32,
                     color=WHITE, weight=BOLD).to_edge(UP, buff=0.35)
            ),
            run_time=0.5,
        )

        line1 = MathTex(
            r"\varepsilon_j \sim \mathrm{Gumbel}(0,1)",
            font_size=30, color=PAL[2],
        )
        line2 = MathTex(
            r"\Downarrow",
            font_size=36, color=TEXT2,
        )
        line3 = MathTex(
            r"P(\mathrm{choose}\;j) = "
            r"\frac{e^{V_j}}{\sum_k e^{V_k}}",
            font_size=30, color=ACCENT,
        )
        line4 = Text("The softmax function!", font_size=24,
                      color=WHITE, weight=BOLD)

        content = VGroup(line1, line2, line3, line4)
        content.arrange(DOWN, buff=0.5)
        content.move_to(ORIGIN)

        for item in content:
            self.play(FadeIn(item, shift=DOWN * 0.1), run_time=0.6)
            self.wait(2.0)

        self.wait(10.0)
