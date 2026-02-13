"""
MLHP Chapter 1 Animation #6: Mixture IIA Violation
Shows how two sub-populations each satisfying IIA individually
produce a mixture that violates IIA.

Run with:
    PATH="/lfs/local/0/sttruong/miniconda3/bin:$PATH" \
    manim -qh --disable_caching --media_dir media/ch1 animations/ch1/mixture_iia.py MixtureIIAViolation
"""

from manim import *
import numpy as np

# ── design tokens ───────────────────────────────────────────────────
ACCENT = "#FFD966"
BG = "#0f0f0f"
TEXT2 = "#aaaaaa"
PAL = ["#5B8DEE", "#45BF7C", "#F0A35C", "#E8637A", "#B07CD8"]
AXIS_CLR = "#888888"


def gauss(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class MixtureIIAViolation(Scene):
    def construct(self):
        self.camera.background_color = BG

        self.play_title()
        self.play_two_groups()
        self.play_mixture()
        self.play_iia_test()
        self.play_takeaway()
        self.play_solution_models()

    # ── title ───────────────────────────────────────────────────────
    def play_title(self):
        t = Text("Population Heterogeneity", font_size=44,
                 color=WHITE, weight=BOLD)
        st = Text("Why mixing IIA groups breaks IIA",
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

    # ── two groups as Gaussians ─────────────────────────────────────
    def play_two_groups(self):
        self.header = Text("Two Sub-Populations", font_size=32,
                           color=WHITE, weight=BOLD)
        self.header.to_edge(UP, buff=0.35)
        self.play(FadeIn(self.header, shift=DOWN * 0.1), run_time=0.6)

        # Formula
        self.formula = MathTex(
            r"P(j \succ j') = \sum_{i=1}^{N} \alpha_i \,"
            r"\sigma(V_j^{(i)} - V_{j'}^{(i)})",
            font_size=26, color=ACCENT,
        )
        self.formula.next_to(self.header, DOWN, buff=0.2)
        self.play(Write(self.formula), run_time=1.0)

        # Axes
        self.ax = Axes(
            x_range=[-3, 7, 1], y_range=[0, 0.5, 0.1],
            x_length=10, y_length=4,
            axis_config={"color": AXIS_CLR, "include_numbers": True,
                         "font_size": 18},
            tips=False,
        )
        self.ax.move_to(DOWN * 0.7)

        x_lab = self.ax.get_x_axis_label(
            MathTex(r"\mathrm{Utility\ noise}", font_size=22, color=AXIS_CLR),
            edge=RIGHT, direction=DOWN,
        )
        y_lab = self.ax.get_y_axis_label(
            MathTex(r"f(\varepsilon)", font_size=22, color=AXIS_CLR),
            edge=UP, direction=LEFT,
        )

        self.play(Create(self.ax), FadeIn(x_lab), FadeIn(y_lab),
                  run_time=0.7)

        # Group 1: centered at 0
        g1_curve = self.ax.plot(
            lambda x: gauss(x, 0, 1),
            x_range=[-3, 7, 0.05],
            color=PAL[0], stroke_width=2.5,
        )
        g1_curve.set_stroke(opacity=0.7)
        g1_lbl = Text("Group 1", font_size=20, color=PAL[0])
        g1_lbl.next_to(self.ax.c2p(0, gauss(0, 0, 1)), UP, buff=0.15)

        # Group 2: centered at 4
        g2_curve = self.ax.plot(
            lambda x: gauss(x, 4, 1),
            x_range=[-3, 7, 0.05],
            color=PAL[1], stroke_width=2.5,
        )
        g2_curve.set_stroke(opacity=0.7)
        g2_lbl = Text("Group 2", font_size=20, color=PAL[1])
        g2_lbl.next_to(self.ax.c2p(4, gauss(0, 0, 1)), UP, buff=0.15)

        self.play(Create(g1_curve), FadeIn(g1_lbl), run_time=0.8)
        self.wait(5.0)
        self.play(Create(g2_curve), FadeIn(g2_lbl), run_time=0.8)
        self.wait(15.0)

        self.g1_curve = g1_curve
        self.g2_curve = g2_curve
        self.g1_lbl = g1_lbl
        self.g2_lbl = g2_lbl
        self.x_lab = x_lab
        self.y_lab = y_lab

    # ── show the mixture ────────────────────────────────────────────
    def play_mixture(self):
        mix_curve = self.ax.plot(
            lambda x: 0.5 * gauss(x, 0, 1) + 0.5 * gauss(x, 4, 1),
            x_range=[-3, 7, 0.05],
            color=ACCENT, stroke_width=3,
        )
        mix_lbl = Text("Mixture", font_size=22, color=ACCENT,
                       weight=BOLD)
        mix_lbl.next_to(self.ax.c2p(2, 0.22), UP, buff=0.15)

        self.play(Create(mix_curve), FadeIn(mix_lbl), run_time=1.0)

        bimodal_note = Text(
            "Bimodal: clearly not Gumbel (or Gaussian)",
            font_size=20, color=TEXT2,
        )
        bimodal_note.to_edge(DOWN, buff=0.2)
        self.play(FadeIn(bimodal_note), run_time=0.4)
        self.wait(15.0)

        self.mix_curve = mix_curve
        self.mix_lbl = mix_lbl
        self.bimodal_note = bimodal_note

    # ── numerical IIA ratio test ────────────────────────────────────
    def play_iia_test(self):
        # Clean up curves
        self.play(
            FadeOut(VGroup(
                self.header, self.formula, self.ax,
                self.g1_curve, self.g2_curve, self.mix_curve,
                self.g1_lbl, self.g2_lbl, self.mix_lbl,
                self.x_lab, self.y_lab, self.bimodal_note,
            )),
            run_time=0.7,
        )

        header = Text("IIA Ratio Test", font_size=32,
                      color=WHITE, weight=BOLD)
        header.to_edge(UP, buff=0.4)
        self.play(FadeIn(header, shift=DOWN * 0.1), run_time=0.6)

        # Utilities for two groups
        # Group 1: V_A=2, V_B=1, V_C=0.5
        # Group 2: V_A=0.5, V_B=1, V_C=2
        desc = Text(
            "Group 1: (A=2, B=1, C=0.5)    Group 2: (A=0.5, B=1, C=2)",
            font_size=20, color=TEXT2,
        )
        desc.next_to(header, DOWN, buff=0.3)
        self.play(FadeIn(desc), run_time=0.5)

        # Compute IIA ratios
        v1 = np.array([2.0, 1.0, 0.5])
        v2 = np.array([0.5, 1.0, 2.0])

        # With all 3 items
        e1 = np.exp(v1)
        e2 = np.exp(v2)
        p1_full = e1 / e1.sum()
        p2_full = e2 / e2.sum()
        p_mix_full = 0.5 * p1_full + 0.5 * p2_full

        # Without C (items A, B only)
        e1_ab = np.exp(v1[:2])
        e2_ab = np.exp(v2[:2])
        p1_ab = e1_ab / e1_ab.sum()
        p2_ab = e2_ab / e2_ab.sum()
        p_mix_ab = 0.5 * p1_ab + 0.5 * p2_ab

        # Ratio P(A)/P(B) with and without C
        ratio_g1_full = p1_full[0] / p1_full[1]
        ratio_g1_ab = p1_ab[0] / p1_ab[1]
        ratio_mix_full = p_mix_full[0] / p_mix_full[1]
        ratio_mix_ab = p_mix_ab[0] / p_mix_ab[1]

        # Display results
        group_result = VGroup(
            Text("Group 1:", font_size=22, color=PAL[0], weight=BOLD),
            Text(f"  P(A)/P(B) with C:    {ratio_g1_full:.4f}",
                 font_size=20, color=TEXT2),
            Text(f"  P(A)/P(B) without C: {ratio_g1_ab:.4f}",
                 font_size=20, color=TEXT2),
            Text(f"  Ratio preserved? Yes", font_size=20,
                 color=PAL[1]),
        ).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        group_result.move_to(LEFT * 2.5 + DOWN * 0.3)

        mix_result = VGroup(
            Text("Mixture:", font_size=22, color=ACCENT, weight=BOLD),
            Text(f"  P(A)/P(B) with C:    {ratio_mix_full:.4f}",
                 font_size=20, color=TEXT2),
            Text(f"  P(A)/P(B) without C: {ratio_mix_ab:.4f}",
                 font_size=20, color=TEXT2),
            Text(f"  Ratio preserved? No!", font_size=20,
                 color=PAL[3]),
        ).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        mix_result.move_to(RIGHT * 2.5 + DOWN * 0.3)

        self.play(FadeIn(group_result, shift=RIGHT * 0.15), run_time=0.7)
        self.wait(12.0)
        self.play(FadeIn(mix_result, shift=RIGHT * 0.15), run_time=0.7)
        self.wait(15.0)

        self.play(FadeOut(VGroup(header, desc, group_result, mix_result)),
                  run_time=0.7)

    # ── takeaway ────────────────────────────────────────────────────
    def play_takeaway(self):
        lines = VGroup(
            Text("Each group satisfies IIA individually,",
                 font_size=28, color=WHITE, weight=BOLD),
            Text("but the mixture does not.",
                 font_size=28, color=WHITE, weight=BOLD),
            Text("", font_size=10),
            Text("A mixture of Gumbel noises is not Gumbel.",
                 font_size=24, color=ACCENT),
            Text("", font_size=10),
            Text("This is why population heterogeneity",
                 font_size=22, color=TEXT2),
            Text("is a fundamental challenge for IIA models.",
                 font_size=22, color=TEXT2),
        )
        lines.arrange(DOWN, buff=0.15)
        lines.move_to(UP * 0.2)

        for l in lines:
            self.play(FadeIn(l, shift=DOWN * 0.08), run_time=0.3)
        self.wait(15.0)

        self.play(FadeOut(lines), run_time=0.5)

    # ── solution models ───────────────────────────────────────────
    def play_solution_models(self):
        """Models that handle heterogeneity."""
        header = Text("Beyond IIA: Solution Models", font_size=32,
                       color=WHITE, weight=BOLD)
        header.to_edge(UP, buff=0.35)
        self.play(FadeIn(header, shift=DOWN * 0.1), run_time=0.5)

        models = [
            ("Latent Class Models", "Finite mixture of IIA groups", PAL[0]),
            ("Mixed Logit", "Continuous distribution over preferences", PAL[1]),
            ("Hierarchical Bayes", "Individual-level parameters", PAL[2]),
            ("GP Preference Models", "Nonparametric, flexible correlations", PAL[4]),
        ]

        items = VGroup()
        for name, desc, color in models:
            t = Text(name, font_size=24, color=color, weight=BOLD)
            d = Text(desc, font_size=18, color=TEXT2)
            d.next_to(t, DOWN, buff=0.08, aligned_edge=LEFT)
            item = VGroup(t, d)
            items.add(item)

        items.arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        items.move_to(DOWN * 0.1)

        for item in items:
            self.play(FadeIn(item, shift=RIGHT * 0.15), run_time=0.4)
            self.wait(4.0)

        # Key equation
        eq = MathTex(
            r"P(j \succ k) = \int \sigma(V_j(\theta) - V_k(\theta))\,"
            r"f(\theta)\,d\theta",
            font_size=24, color=ACCENT,
        )
        eq.to_edge(DOWN, buff=0.3)
        self.play(Write(eq), run_time=0.8)
        self.wait(12.0)
