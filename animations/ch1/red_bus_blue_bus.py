"""
MLHP Chapter 1 Animation #5: The Red Bus / Blue Bus Problem
Demonstrates how IIA fails when a clone alternative is added,
incorrectly stealing probability from unrelated options.

Run with:
    PATH="/lfs/local/0/sttruong/miniconda3/bin:$PATH" \
    manim -qh --disable_caching --media_dir media/ch1 animations/ch1/red_bus_blue_bus.py RedBusBlueBus
"""

from manim import *
import numpy as np

# ── design tokens ───────────────────────────────────────────────────
ACCENT = "#FFD966"
BG = "#0f0f0f"
TEXT2 = "#aaaaaa"
PAL = ["#5B8DEE", "#45BF7C", "#F0A35C", "#E8637A", "#B07CD8"]
AXIS_CLR = "#888888"

TRAIN_CLR = PAL[0]
BUS_CLR = PAL[2]
RBUS_CLR = PAL[3]
BBUS_CLR = PAL[0]


def softmax(v):
    """Compute softmax probabilities."""
    ev = np.exp(v)
    return ev / ev.sum()


class RedBusBlueBus(Scene):
    def construct(self):
        self.camera.background_color = BG

        self.play_title()
        self.play_before()
        self.play_clone()
        self.play_fix()
        self.play_takeaway()
        self.play_real_world_examples()

    # ── title ───────────────────────────────────────────────────────
    def play_title(self):
        t = Text("The Red Bus / Blue Bus Problem", font_size=44,
                 color=WHITE, weight=BOLD)
        st = Text("When IIA fails",
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

    # ── helpers ─────────────────────────────────────────────────────
    def make_bar(self, label, prob, color, y_pos, max_width=5.0):
        """Create a labeled horizontal probability bar."""
        lbl = Text(label, font_size=24, color=color, weight=BOLD)
        lbl.move_to(LEFT * 4.5 + UP * y_pos)

        bar = Rectangle(
            width=max_width * prob,
            height=0.45,
            fill_color=color,
            fill_opacity=0.7,
            stroke_color=color,
            stroke_width=1,
        )
        bar.move_to(LEFT * 1.5 + UP * y_pos, aligned_edge=LEFT)

        p_lbl = Text(f"{prob:.1%}", font_size=22, color=WHITE)
        p_lbl.next_to(bar, RIGHT, buff=0.15)

        return lbl, bar, p_lbl

    # ── before: Train vs Bus ────────────────────────────────────────
    def play_before(self):
        header = Text("Original Choice Set", font_size=32,
                      color=WHITE, weight=BOLD)
        header.to_edge(UP, buff=0.35)

        formula = MathTex(
            r"P(j \mid \mathcal{S}) = \frac{e^{V_j}}{\sum_{k} e^{V_k}}",
            font_size=28, color=ACCENT,
        )
        formula.next_to(header, DOWN, buff=0.2)
        self.play(FadeIn(header, shift=DOWN * 0.1), run_time=0.6)
        self.play(Write(formula), run_time=0.8)

        # V_train = 1, V_bus = 2
        v_train, v_bus = 1.0, 2.0
        probs = softmax([v_train, v_bus])

        # Utility labels
        u_train = MathTex(r"V_{\mathrm{train}} = 1", font_size=22,
                          color=TEXT2)
        u_bus = MathTex(r"V_{\mathrm{bus}} = 2", font_size=22,
                        color=TEXT2)
        u_train.move_to(LEFT * 3.0 + UP * 0.5)
        u_bus.move_to(LEFT * 3.0 + DOWN * 0.3)
        self.play(FadeIn(u_train), FadeIn(u_bus), run_time=0.5)

        # Bars
        t_lbl, t_bar, t_plbl = self.make_bar(
            "Train", probs[0], TRAIN_CLR, 0.5)
        b_lbl, b_bar, b_plbl = self.make_bar(
            "Bus", probs[1], BUS_CLR, -0.3)

        self.play(
            FadeIn(t_lbl), GrowFromEdge(t_bar, LEFT), FadeIn(t_plbl),
            run_time=0.6,
        )
        self.play(
            FadeIn(b_lbl), GrowFromEdge(b_bar, LEFT), FadeIn(b_plbl),
            run_time=0.6,
        )
        self.wait(15.0)

        # Store references
        self.before_objs = VGroup(
            header, formula, u_train, u_bus,
            t_lbl, t_bar, t_plbl, b_lbl, b_bar, b_plbl,
        )
        self.probs_before = probs

    # ── clone: add Blue Bus ─────────────────────────────────────────
    def play_clone(self):
        self.play(FadeOut(self.before_objs), run_time=0.5)

        header = Text("After Cloning the Bus", font_size=32,
                      color=WHITE, weight=BOLD)
        header.to_edge(UP, buff=0.35)
        self.play(FadeIn(header, shift=DOWN * 0.1), run_time=0.6)

        # V_train = 1, V_red = 2, V_blue = 2
        v_train, v_red, v_blue = 1.0, 2.0, 2.0
        probs3 = softmax([v_train, v_red, v_blue])

        # Bars
        t_lbl, t_bar, t_plbl = self.make_bar(
            "Train", probs3[0], TRAIN_CLR, 1.0)
        r_lbl, r_bar, r_plbl = self.make_bar(
            "Red Bus", probs3[1], RBUS_CLR, 0.2)
        bl_lbl, bl_bar, bl_plbl = self.make_bar(
            "Blue Bus", probs3[2], BBUS_CLR, -0.6)

        self.play(
            FadeIn(t_lbl), GrowFromEdge(t_bar, LEFT), FadeIn(t_plbl),
            run_time=0.5,
        )
        self.play(
            FadeIn(r_lbl), GrowFromEdge(r_bar, LEFT), FadeIn(r_plbl),
            run_time=0.5,
        )
        self.play(
            FadeIn(bl_lbl), GrowFromEdge(bl_bar, LEFT), FadeIn(bl_plbl),
            run_time=0.5,
        )
        self.wait(10.0)

        # Highlight the problem
        # Train dropped from 26.9% to 15.5%
        problem = Text(
            f"Train dropped from {self.probs_before[0]:.1%} to {probs3[0]:.1%}!",
            font_size=22, color=RBUS_CLR, weight=BOLD,
        )
        problem.to_edge(DOWN, buff=1.2)

        total_bus = probs3[1] + probs3[2]
        bus_note = Text(
            f"Total bus share: {self.probs_before[1]:.1%} \u2192 {total_bus:.1%}",
            font_size=20, color=TEXT2,
        )
        bus_note.next_to(problem, DOWN, buff=0.2)

        self.play(FadeIn(problem), run_time=0.5)
        self.play(FadeIn(bus_note), run_time=0.5)
        self.wait(10.0)

        explain = Text(
            "Adding an identical option shouldn't steal\n"
            "probability from unrelated alternatives!",
            font_size=20, color=ACCENT, line_spacing=1.3,
        )
        explain.next_to(bus_note, DOWN, buff=0.3)
        self.play(FadeIn(explain, shift=UP * 0.1), run_time=0.6)
        self.wait(15.0)

        self.clone_objs = VGroup(
            header, t_lbl, t_bar, t_plbl,
            r_lbl, r_bar, r_plbl,
            bl_lbl, bl_bar, bl_plbl,
            problem, bus_note, explain,
        )

    # ── fix: correlated noise preserves ratios ──────────────────────
    def play_fix(self):
        self.play(FadeOut(self.clone_objs), run_time=0.5)

        header = Text("The Fix: Correlated Noise", font_size=32,
                      color=WHITE, weight=BOLD)
        header.to_edge(UP, buff=0.35)
        self.play(FadeIn(header, shift=DOWN * 0.1), run_time=0.6)

        note = Text(
            "With correlated noise between similar alternatives,\n"
            "the train probability is preserved:",
            font_size=22, color=TEXT2, line_spacing=1.3,
        )
        note.next_to(header, DOWN, buff=0.4)
        self.play(FadeIn(note), run_time=0.5)

        # Corrected probabilities (nested logit style)
        # Train: 26.9%, Red Bus: 36.6%, Blue Bus: 36.6%
        corrected = [0.269, 0.366, 0.366]

        t_lbl, t_bar, t_plbl = self.make_bar(
            "Train", corrected[0], TRAIN_CLR, 0.0)
        r_lbl, r_bar, r_plbl = self.make_bar(
            "Red Bus", corrected[1], RBUS_CLR, -0.8)
        bl_lbl, bl_bar, bl_plbl = self.make_bar(
            "Blue Bus", corrected[2], BBUS_CLR, -1.6)

        self.play(
            FadeIn(t_lbl), GrowFromEdge(t_bar, LEFT), FadeIn(t_plbl),
            run_time=0.5,
        )
        self.play(
            FadeIn(r_lbl), GrowFromEdge(r_bar, LEFT), FadeIn(r_plbl),
            run_time=0.5,
        )
        self.play(
            FadeIn(bl_lbl), GrowFromEdge(bl_bar, LEFT), FadeIn(bl_plbl),
            run_time=0.5,
        )
        self.wait(10.0)

        preserved = Text(
            "Train stays at 26.9% \u2014 cloning doesn't affect it!",
            font_size=22, color=PAL[1], weight=BOLD,
        )
        preserved.to_edge(DOWN, buff=0.3)
        self.play(FadeIn(preserved, shift=UP * 0.1), run_time=0.5)
        self.wait(15.0)

        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.7)

    # ── takeaway ────────────────────────────────────────────────────
    def play_takeaway(self):
        lines = VGroup(
            Text("IIA assumes independent noise", font_size=30,
                 color=WHITE, weight=BOLD),
            Text("between all alternatives.", font_size=30,
                 color=WHITE, weight=BOLD),
            Text("", font_size=10),
            Text("When alternatives are similar,",
                 font_size=24, color=TEXT2),
            Text("this assumption fails.",
                 font_size=24, color=TEXT2),
            Text("", font_size=10),
            Text("Models like nested logit and probit",
                 font_size=22, color=ACCENT),
            Text("allow correlated noise to fix this.",
                 font_size=22, color=ACCENT),
        )
        lines.arrange(DOWN, buff=0.15)
        lines.move_to(UP * 0.2)

        for l in lines:
            self.play(FadeIn(l, shift=DOWN * 0.08), run_time=0.3)
        self.wait(15.0)

        self.play(FadeOut(lines), run_time=0.5)

    # ── real-world examples ───────────────────────────────────────
    def play_real_world_examples(self):
        """IIA violations in practice."""
        header = Text("IIA Violations in Practice", font_size=32,
                       color=WHITE, weight=BOLD)
        header.to_edge(UP, buff=0.35)
        self.play(FadeIn(header, shift=DOWN * 0.1), run_time=0.5)

        examples = [
            ("iPhone 16 vs 16 Pro vs Galaxy",
             "Near-identical options split share", PAL[0]),
            ("LLM paraphrases compete",
             "Similar responses are not independent", PAL[1]),
            ("Highway vs side road vs new highway",
             "Correlated travel routes", PAL[2]),
            ("Netflix: similar movies compete",
             "Genre clusters violate IIA", PAL[3]),
        ]

        items = VGroup()
        for title, desc, color in examples:
            t = Text(title, font_size=22, color=color, weight=BOLD)
            d = Text(desc, font_size=18, color=TEXT2)
            d.next_to(t, DOWN, buff=0.08, aligned_edge=LEFT)
            item = VGroup(t, d)
            items.add(item)

        items.arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        items.move_to(DOWN * 0.3)

        for item in items:
            self.play(FadeIn(item, shift=RIGHT * 0.15), run_time=0.4)
            self.wait(4.0)

        solution = Text(
            "Solution: nested logit, mixed logit, or probit models",
            font_size=20, color=ACCENT,
        )
        solution.to_edge(DOWN, buff=0.3)
        self.play(FadeIn(solution, shift=UP * 0.1), run_time=0.5)
        self.wait(10.0)
