"""
MLHP Chapter 1 Animation #2: From Rasch to Bradley-Terry
Shows how the user parameter U_i cancels when deriving pairwise
comparison probabilities from the Rasch model.

Run with:
    PATH="/lfs/local/0/sttruong/miniconda3/bin:$PATH" \
    manim -qh --disable_caching --media_dir media/ch1 animations/ch1/rasch_to_bt.py RaschToBradleyTerry
"""

from manim import *
import numpy as np

# ── design tokens ───────────────────────────────────────────────────
ACCENT = "#FFD966"
BG = "#0f0f0f"
TEXT2 = "#aaaaaa"
PAL = ["#5B8DEE", "#45BF7C", "#F0A35C", "#E8637A", "#B07CD8"]
AXIS_CLR = "#888888"


class RaschToBradleyTerry(Scene):
    def construct(self):
        self.camera.background_color = BG

        self.play_title()
        self.play_setup()
        self.play_cancellation()
        self.play_implication()
        self.play_comparison_table()
        self.play_historical_timeline()

    # ── title ───────────────────────────────────────────────────────
    def play_title(self):
        t = Text("From Rasch to Bradley-Terry", font_size=44,
                 color=WHITE, weight=BOLD)
        st = Text("Why user appetite cancels",
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

    # ── setup: number line + Rasch utilities ────────────────────────
    def play_setup(self):
        header = Text("The Rasch Model", font_size=32,
                      color=WHITE, weight=BOLD)
        header.to_edge(UP, buff=0.35)
        self.play(FadeIn(header, shift=DOWN * 0.1), run_time=0.6)

        # Number line for utility
        nline = NumberLine(
            x_range=[-3, 5, 1], length=10,
            color=AXIS_CLR, include_numbers=True, font_size=20,
        )
        nline.move_to(DOWN * 0.3)
        nline_lbl = MathTex(r"\mathrm{Utility}", font_size=22,
                            color=AXIS_CLR)
        nline_lbl.next_to(nline, DOWN, buff=0.25)
        self.play(Create(nline), FadeIn(nline_lbl), run_time=0.8)

        # User dot
        user_val = 1.0
        user_dot = Dot(nline.n2p(user_val), color=PAL[2], radius=0.1)
        user_lbl = MathTex(r"U_i", font_size=26, color=PAL[2])
        user_lbl.next_to(user_dot, UP, buff=0.2)
        self.play(FadeIn(user_dot), FadeIn(user_lbl), run_time=0.5)

        # Item j dot
        vj_val = 2.5
        dot_j = Dot(nline.n2p(vj_val), color=PAL[0], radius=0.1)
        lbl_j = MathTex(r"V_j", font_size=26, color=PAL[0])
        lbl_j.next_to(dot_j, UP, buff=0.2)

        # Item k dot
        vk_val = -0.5
        dot_k = Dot(nline.n2p(vk_val), color=PAL[3], radius=0.1)
        lbl_k = MathTex(r"V_k", font_size=26, color=PAL[3])
        lbl_k.next_to(dot_k, UP, buff=0.2)

        self.play(
            FadeIn(dot_j), FadeIn(lbl_j),
            FadeIn(dot_k), FadeIn(lbl_k),
            run_time=0.7,
        )
        self.wait(1.0)

        # Rasch utilities
        eq_j = MathTex(
            r"H_{ij}", r"=", r"U_i", r"+", r"V_j",
            font_size=30, color=ACCENT,
        )
        eq_k = MathTex(
            r"H_{ik}", r"=", r"U_i", r"+", r"V_k",
            font_size=30, color=ACCENT,
        )
        eqs = VGroup(eq_j, eq_k).arrange(DOWN, buff=0.3)
        eqs.next_to(nline, UP, buff=0.8).shift(RIGHT * 2.5)
        self.play(Write(eq_j), run_time=0.8)
        self.play(Write(eq_k), run_time=0.8)
        self.wait(15.0)

        # Store for next act
        self.header = header
        self.nline_group = VGroup(nline, nline_lbl, user_dot, user_lbl,
                                  dot_j, lbl_j, dot_k, lbl_k)
        self.eq_j = eq_j
        self.eq_k = eq_k

    # ── cancellation: the key derivation ────────────────────────────
    def play_cancellation(self):
        # Fade number line, keep equations
        self.play(
            FadeOut(self.nline_group),
            self.eq_j.animate.move_to(UP * 1.5 + LEFT * 2),
            self.eq_k.animate.move_to(UP * 0.7 + LEFT * 2),
            run_time=0.8,
        )

        # Pairwise comparison formula
        pw = MathTex(
            r"P(j \succ k \mid i)",
            r"= \sigma(",
            r"H_{ij}", r"-", r"H_{ik}",
            r")",
            font_size=30, color=WHITE,
        )
        pw.move_to(DOWN * 0.3)
        self.play(Write(pw), run_time=1.0)
        self.wait(10.0)

        # Expand
        expanded = MathTex(
            r"= \sigma(",
            r"(", r"U_i", r"+", r"V_j", r")",
            r"-",
            r"(", r"U_i", r"+", r"V_k", r")",
            r")",
            font_size=30, color=WHITE,
        )
        expanded.next_to(pw, DOWN, buff=0.4)
        self.play(Write(expanded), run_time=1.2)
        self.wait(10.0)

        # Highlight U_i terms
        # Indices: expanded[2] = first U_i, expanded[8] = second U_i
        box1 = SurroundingRectangle(expanded[2], color=PAL[2],
                                     buff=0.05, stroke_width=2)
        box2 = SurroundingRectangle(expanded[8], color=PAL[2],
                                     buff=0.05, stroke_width=2)
        self.play(Create(box1), Create(box2), run_time=0.6)
        self.wait(5.0)

        # Flash and cross out
        cross1 = Line(
            box1.get_corner(UL), box1.get_corner(DR),
            color=PAL[3], stroke_width=3,
        )
        cross2 = Line(
            box2.get_corner(UL), box2.get_corner(DR),
            color=PAL[3], stroke_width=3,
        )
        self.play(Create(cross1), Create(cross2), run_time=0.5)
        self.wait(3.0)

        # Show simplified result
        result = MathTex(
            r"= \sigma(", r"V_j", r"-", r"V_k", r")",
            font_size=34, color=ACCENT,
        )
        result.next_to(expanded, DOWN, buff=0.4)
        self.play(
            FadeOut(box1), FadeOut(box2),
            FadeOut(cross1), FadeOut(cross2),
            Write(result),
            run_time=1.0,
        )
        self.wait(8.0)

        # Box the final formula
        bt_label = Text("This is Bradley-Terry!", font_size=24,
                        color=PAL[1], weight=BOLD)
        bt_label.next_to(result, DOWN, buff=0.3)
        bt_box = SurroundingRectangle(
            VGroup(result, bt_label), color=ACCENT,
            buff=0.2, stroke_width=2,
        )
        self.play(Create(bt_box), FadeIn(bt_label), run_time=0.8)
        self.wait(20.0)

        # Clean up for implication slide
        self.play(
            FadeOut(VGroup(
                self.header, self.eq_j, self.eq_k,
                pw, expanded, result, bt_label, bt_box,
            )),
            run_time=0.7,
        )

    # ── implication: what each data type reveals ────────────────────
    def play_implication(self):
        header = Text("What does each data type reveal?", font_size=32,
                      color=WHITE, weight=BOLD)
        header.to_edge(UP, buff=0.5)
        self.play(FadeIn(header, shift=DOWN * 0.1), run_time=0.6)

        # Row 1: Pairwise
        pw_icon = Text("Pairwise data", font_size=24, color=PAL[3],
                       weight=BOLD)
        pw_arrow = MathTex(r"\longrightarrow", font_size=28, color=TEXT2)
        pw_info = MathTex(r"V_j - V_k \mathrm{\ only}",
                          font_size=26, color=TEXT2)
        row1 = VGroup(pw_icon, pw_arrow, pw_info).arrange(RIGHT, buff=0.3)

        # Row 2: Item-wise
        iw_icon = Text("Item-wise data", font_size=24, color=PAL[0],
                       weight=BOLD)
        iw_arrow = MathTex(r"\longrightarrow", font_size=28, color=TEXT2)
        iw_info = MathTex(r"\mathrm{Both}\ U_i\ \mathrm{and}\ V_j",
                          font_size=26, color=TEXT2)
        row2 = VGroup(iw_icon, iw_arrow, iw_info).arrange(RIGHT, buff=0.3)

        rows = VGroup(row1, row2).arrange(DOWN, buff=0.6,
                                           aligned_edge=LEFT)
        rows.move_to(ORIGIN)

        self.play(FadeIn(row1, shift=RIGHT * 0.2), run_time=0.7)
        self.wait(10.0)
        self.play(FadeIn(row2, shift=RIGHT * 0.2), run_time=0.7)
        self.wait(10.0)

        note = Text(
            "User parameters cancel in pairwise comparisons\n"
            "under the Rasch model's additive structure",
            font_size=20, color=TEXT2, line_spacing=1.3,
        )
        note.to_edge(DOWN, buff=0.3)
        self.play(FadeIn(note, shift=UP * 0.1), run_time=0.5)
        self.wait(15.0)

        self.play(
            FadeOut(VGroup(header, row1, row2, note)),
            run_time=0.5,
        )

    # ── comparison table ──────────────────────────────────────────
    def play_comparison_table(self):
        """Side-by-side comparison of pairwise vs item-wise data."""
        header = Text("Data Type Comparison", font_size=32,
                       color=WHITE, weight=BOLD)
        header.to_edge(UP, buff=0.35)
        self.play(FadeIn(header, shift=DOWN * 0.1), run_time=0.5)

        # Left column: Pairwise
        pw_hdr = Text("Pairwise", font_size=24, color=PAL[3], weight=BOLD)
        pw_items = VGroup(
            Text("Elo / Chatbot Arena", font_size=20, color=TEXT2),
            Text("RLHF preference data", font_size=20, color=TEXT2),
            Text("Tournament results", font_size=20, color=TEXT2),
        )
        pw_items.arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        pw_col = VGroup(pw_hdr, pw_items).arrange(DOWN, buff=0.3, aligned_edge=LEFT)

        # Right column: Item-wise
        iw_hdr = Text("Item-wise", font_size=24, color=PAL[0], weight=BOLD)
        iw_items = VGroup(
            Text("Recommender systems", font_size=20, color=TEXT2),
            Text("Benchmark accuracy", font_size=20, color=TEXT2),
            Text("Likert scale ratings", font_size=20, color=TEXT2),
        )
        iw_items.arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        iw_col = VGroup(iw_hdr, iw_items).arrange(DOWN, buff=0.3, aligned_edge=LEFT)

        cols = VGroup(pw_col, iw_col).arrange(RIGHT, buff=2.0)
        cols.move_to(DOWN * 0.2)

        # Dividing line
        div = Line(UP * 1.5, DOWN * 1.5, color=AXIS_CLR, stroke_width=1)
        div.move_to(cols.get_center())

        self.play(Create(div), run_time=0.3)
        self.play(FadeIn(pw_col, shift=RIGHT * 0.1), run_time=0.5)
        self.wait(5.0)
        self.play(FadeIn(iw_col, shift=LEFT * 0.1), run_time=0.5)
        self.wait(8.0)

        # Key insight
        insight = MathTex(
            r"\mathrm{Pairwise:}\;U_i\;\mathrm{cancels}"
            r"\;\;\Rightarrow\;\;\mathrm{user\text{-}free\;comparisons}",
            font_size=24, color=ACCENT,
        )
        insight.to_edge(DOWN, buff=0.3)
        self.play(FadeIn(insight, shift=UP * 0.1), run_time=0.5)
        self.wait(10.0)

        self.play(
            FadeOut(header), FadeOut(pw_col), FadeOut(iw_col),
            FadeOut(div), FadeOut(insight),
            run_time=0.5,
        )

    # ── historical timeline ───────────────────────────────────────
    def play_historical_timeline(self):
        """Key milestones in paired comparison theory."""
        header = Text("Historical Timeline", font_size=32,
                       color=WHITE, weight=BOLD)
        header.to_edge(UP, buff=0.35)
        self.play(FadeIn(header, shift=DOWN * 0.1), run_time=0.5)

        events = [
            ("1927", "Thurstone", "Law of Comparative Judgment", PAL[0]),
            ("1952", "Bradley-Terry", "Pairwise comparison model", PAL[1]),
            ("1960", "Rasch", "Measurement model (IRT)", PAL[2]),
            ("1973", "McFadden", "Conditional logit (Nobel 2000)", PAL[3]),
        ]

        # Timeline line
        tl_line = Line(LEFT * 5, RIGHT * 5, color=AXIS_CLR, stroke_width=2)
        tl_line.move_to(DOWN * 0.2)
        self.play(Create(tl_line), run_time=0.5)

        for i, (year, name, desc, color) in enumerate(events):
            x_pos = -4.0 + i * 2.7
            dot = Dot(
                tl_line.get_start() + RIGHT * (i + 0.5) * 2.5,
                radius=0.08, color=color,
            )
            dot.move_to(np.array([x_pos, tl_line.get_center()[1], 0]))

            yr = Text(year, font_size=20, color=color, weight=BOLD)
            yr.next_to(dot, UP, buff=0.2)
            nm = Text(name, font_size=18, color=WHITE)
            nm.next_to(yr, UP, buff=0.1)
            ds = Text(desc, font_size=14, color=TEXT2)
            ds.next_to(dot, DOWN, buff=0.2)

            self.play(
                FadeIn(dot), FadeIn(yr), FadeIn(nm), FadeIn(ds),
                run_time=0.5,
            )
            self.wait(3.0)

        self.wait(10.0)
