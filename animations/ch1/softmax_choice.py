"""
MLHP Chapter 1 Animation #4: Softmax Choice / Plackett-Luce
Demonstrates how items are chosen sequentially from a shrinking set
with softmax probabilities recomputing at each step.

Run with:
    PATH="/lfs/local/0/sttruong/miniconda3/bin:$PATH" \
    manim -qh --disable_caching --media_dir media/ch1 animations/ch1/softmax_choice.py SoftmaxChoice
"""

from manim import *
import numpy as np

# ── design tokens ───────────────────────────────────────────────────
ACCENT = "#FFD966"
BG = "#0f0f0f"
TEXT2 = "#aaaaaa"
PAL = ["#5B8DEE", "#45BF7C", "#F0A35C", "#E8637A", "#B07CD8"]
AXIS_CLR = "#888888"


class SoftmaxChoice(Scene):
    def construct(self):
        self.camera.background_color = BG

        self.play_title()
        self.play_softmax_bars()
        self.play_plackett_luce()
        self.play_bt_special_case()
        self.play_equivalence_chain()
        self.play_iia_teaser()

    # ── title ───────────────────────────────────────────────────────
    def play_title(self):
        t = Text("Choice Probabilities under IIA", font_size=44,
                 color=WHITE, weight=BOLD)
        st = Text("From softmax to Plackett-Luce",
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

    # ── softmax bar chart ───────────────────────────────────────────
    def play_softmax_bars(self):
        header = Text("Softmax Choice Probabilities", font_size=32,
                      color=WHITE, weight=BOLD)
        header.to_edge(UP, buff=0.35)

        formula = MathTex(
            r"P(j \mid \mathcal{S}) = "
            r"\frac{e^{V_j}}{\sum_{k \in \mathcal{S}} e^{V_k}}",
            font_size=30, color=ACCENT,
        )
        formula.next_to(header, DOWN, buff=0.2)

        self.play(FadeIn(header, shift=DOWN * 0.1), run_time=0.6)
        self.play(Write(formula), run_time=1.0)

        # Items with utilities
        names = ["A", "B", "C", "D"]
        utilities = [2.0, 1.0, 0.5, 0.0]
        colors = PAL[:4]

        # Compute softmax
        exp_v = np.exp(utilities)
        probs = exp_v / exp_v.sum()

        # Create utility labels and probability bars
        bar_max_width = 3.5
        bar_height = 0.5
        bar_gap = 0.9
        start_y = 0.6

        item_groups = []
        bar_rects = []
        prob_labels = []

        for i, (name, v, p, color) in enumerate(
            zip(names, utilities, probs, colors)
        ):
            y_pos = start_y - i * bar_gap

            # Item name + utility
            name_lbl = Text(f"{name}", font_size=26, color=color,
                           weight=BOLD)
            name_lbl.move_to(LEFT * 4.5 + UP * y_pos)

            v_lbl = MathTex(rf"V = {v:.1f}", font_size=22, color=TEXT2)
            v_lbl.next_to(name_lbl, RIGHT, buff=0.3)

            # Probability bar
            bar = Rectangle(
                width=bar_max_width * p,
                height=bar_height,
                fill_color=color,
                fill_opacity=0.7,
                stroke_color=color,
                stroke_width=1,
            )
            bar.move_to(LEFT * 0.5 + UP * y_pos, aligned_edge=LEFT)

            # Probability label
            p_lbl = Text(f"{p:.1%}", font_size=20, color=WHITE)
            p_lbl.next_to(bar, RIGHT, buff=0.15)

            item_groups.append(VGroup(name_lbl, v_lbl))
            bar_rects.append(bar)
            prob_labels.append(p_lbl)

        # Animate in
        for i in range(4):
            self.play(
                FadeIn(item_groups[i]),
                GrowFromEdge(bar_rects[i], LEFT),
                FadeIn(prob_labels[i]),
                run_time=0.5,
            )
        self.wait(15.0)

        # Store for Plackett-Luce
        self.header = header
        self.formula = formula
        self.item_groups = item_groups
        self.bar_rects = bar_rects
        self.prob_labels = prob_labels
        self.names = names
        self.utilities = utilities
        self.colors = colors

    # ── Plackett-Luce sequential picking ────────────────────────────
    def play_plackett_luce(self):
        # Replace formula
        pl_formula = MathTex(
            r"P(\mathrm{ranking}) = \prod_{t=1}^{M}"
            r"\frac{e^{V_{j_t}}}{\sum_{k \in \mathcal{S}_t} e^{V_k}}",
            font_size=28, color=ACCENT,
        )
        pl_formula.next_to(self.header, DOWN, buff=0.2)
        self.play(
            ReplacementTransform(self.formula, pl_formula),
            run_time=0.8,
        )
        self.wait(10.0)

        # Collect the ranking factors
        ranking_parts = []
        remaining = list(range(4))

        for pick in range(3):  # pick 3 (last one is automatic)
            # Find top item
            top_idx = remaining[0]
            for idx in remaining:
                if self.utilities[idx] > self.utilities[top_idx]:
                    top_idx = idx

            # Highlight the winner
            self.play(
                Indicate(self.bar_rects[top_idx], color=ACCENT,
                         scale_factor=1.05),
                run_time=0.5,
            )

            # Build the factor string
            num = rf"e^{{{self.utilities[top_idx]:.1f}}}"
            denom_parts = [rf"e^{{{self.utilities[k]:.1f}}}"
                          for k in remaining]
            denom = "+".join(denom_parts)
            factor_tex = rf"\frac{{{num}}}{{{denom}}}"
            ranking_parts.append(factor_tex)

            # Lift the winner out
            winner_group = VGroup(
                self.item_groups[top_idx],
                self.bar_rects[top_idx],
                self.prob_labels[top_idx],
            )
            self.play(
                winner_group.animate.shift(RIGHT * 6).set_opacity(0.3),
                run_time=0.7,
            )

            remaining.remove(top_idx)

            # Recompute probabilities for remaining
            if len(remaining) > 1:
                rem_utils = [self.utilities[k] for k in remaining]
                exp_v = np.exp(rem_utils)
                new_probs = exp_v / exp_v.sum()

                anims = []
                for ri, idx in enumerate(remaining):
                    new_width = 3.5 * new_probs[ri]
                    new_bar = Rectangle(
                        width=max(new_width, 0.05),
                        height=0.5,
                        fill_color=self.colors[idx],
                        fill_opacity=0.7,
                        stroke_color=self.colors[idx],
                        stroke_width=1,
                    )
                    new_bar.move_to(
                        self.bar_rects[idx].get_left(), aligned_edge=LEFT
                    )
                    new_p_lbl = Text(
                        f"{new_probs[ri]:.1%}", font_size=20, color=WHITE,
                    )
                    new_p_lbl.next_to(new_bar, RIGHT, buff=0.15)

                    anims.append(
                        Transform(self.bar_rects[idx], new_bar)
                    )
                    anims.append(
                        Transform(self.prob_labels[idx], new_p_lbl)
                    )

                self.play(*anims, run_time=0.8)
            self.wait(8.0)

        self.wait(1.0)

        # Show the full ranking probability product
        full_product = MathTex(
            r"P(A \succ B \succ C \succ D) = "
            + r" \cdot ".join(ranking_parts),
            font_size=24, color=ACCENT,
        )
        full_product.to_edge(DOWN, buff=0.25)
        self.play(Write(full_product), run_time=1.5)
        self.wait(15.0)

        # Clean up
        self.play(
            *[FadeOut(m) for m in self.mobjects],
            run_time=0.7,
        )

    # ── Bradley-Terry as special case ───────────────────────────────
    def play_bt_special_case(self):
        header = Text("Binary Case = Bradley-Terry", font_size=32,
                      color=WHITE, weight=BOLD)
        header.to_edge(UP, buff=0.5)
        self.play(FadeIn(header, shift=DOWN * 0.1), run_time=0.6)

        eq1 = MathTex(
            r"P(j \succ k) = \frac{e^{V_j}}{e^{V_j} + e^{V_k}}",
            font_size=32, color=WHITE,
        )
        eq1.move_to(UP * 0.5)
        self.play(Write(eq1), run_time=1.0)
        self.wait(8.0)

        eq2 = MathTex(
            r"= \sigma(V_j - V_k)",
            font_size=32, color=ACCENT,
        )
        eq2.next_to(eq1, DOWN, buff=0.4)
        self.play(Write(eq2), run_time=0.8)
        self.wait(8.0)

        box = SurroundingRectangle(eq2, color=ACCENT, buff=0.15,
                                    stroke_width=2)
        lbl = Text("Bradley-Terry", font_size=22, color=PAL[1],
                   weight=BOLD)
        lbl.next_to(box, DOWN, buff=0.2)
        self.play(Create(box), FadeIn(lbl), run_time=0.6)
        self.wait(15.0)

        # Transition
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.5)

    # ── chain of equivalences ─────────────────────────────────────
    def play_equivalence_chain(self):
        """Vertical flow: Gumbel -> IIA -> Softmax -> PL -> BT."""
        header = Text("Chain of Equivalences", font_size=32,
                       color=WHITE, weight=BOLD)
        header.to_edge(UP, buff=0.35)
        self.play(FadeIn(header, shift=DOWN * 0.1), run_time=0.5)

        nodes = [
            (r"\varepsilon \sim \mathrm{Gumbel}", PAL[2]),
            (r"\mathrm{IIA\;(Independence\;of\;Irrelevant\;Alternatives)}", TEXT2),
            (r"\mathrm{Softmax\;choice\;probabilities}", PAL[0]),
            (r"\mathrm{Plackett\text{-}Luce\;rankings}", PAL[1]),
            (r"\mathrm{Bradley\text{-}Terry\;(pairwise)}", ACCENT),
        ]

        node_mobs = VGroup()
        for tex, color in nodes:
            m = MathTex(tex, font_size=24, color=color)
            node_mobs.add(m)

        node_mobs.arrange(DOWN, buff=0.55)
        node_mobs.move_to(DOWN * 0.2)

        arrows = VGroup()
        for i in range(len(node_mobs) - 1):
            arr = Arrow(
                node_mobs[i].get_bottom() + DOWN * 0.05,
                node_mobs[i + 1].get_top() + UP * 0.05,
                color=AXIS_CLR, stroke_width=2,
                max_tip_length_to_length_ratio=0.15,
            )
            arrows.add(arr)

        for i, node in enumerate(node_mobs):
            self.play(FadeIn(node, shift=DOWN * 0.1), run_time=0.5)
            if i < len(arrows):
                self.play(Create(arrows[i]), run_time=0.3)
            self.wait(3.0)

        desc = Text(
            "All five are logically equivalent characterizations",
            font_size=20, color=TEXT2,
        )
        desc.to_edge(DOWN, buff=0.2)
        self.play(FadeIn(desc), run_time=0.4)
        self.wait(15.0)

        self.play(
            FadeOut(header), FadeOut(node_mobs), FadeOut(arrows), FadeOut(desc),
            run_time=0.5,
        )

    # ── IIA teaser ────────────────────────────────────────────────
    def play_iia_teaser(self):
        """Preview: what if alternatives are NOT irrelevant?"""
        header = Text("But What If...", font_size=32,
                       color=WHITE, weight=BOLD)
        header.to_edge(UP, buff=0.35)
        self.play(FadeIn(header, shift=DOWN * 0.1), run_time=0.5)

        q = Text(
            "What if alternatives are NOT irrelevant?",
            font_size=28, color=PAL[3], weight=BOLD,
        )
        q.move_to(UP * 0.5)
        self.play(FadeIn(q, shift=DOWN * 0.1), run_time=0.6)
        self.wait(5.0)

        # Parameter efficiency note
        dense = MathTex(
            r"10\;\mathrm{items}: 10! = 3{,}628{,}800\;\mathrm{rankings}",
            font_size=24, color=TEXT2,
        )
        sparse = MathTex(
            r"\mathrm{Plackett\text{-}Luce}: \mathrm{only}\;10\;\mathrm{parameters}",
            font_size=24, color=PAL[1],
        )
        eff = VGroup(dense, sparse).arrange(DOWN, buff=0.3)
        eff.move_to(DOWN * 0.5)

        self.play(FadeIn(dense, shift=DOWN * 0.1), run_time=0.5)
        self.wait(3.0)
        self.play(FadeIn(sparse, shift=DOWN * 0.1), run_time=0.5)
        self.wait(3.0)

        hint = Text(
            "Next: The Red Bus / Blue Bus paradox",
            font_size=22, color=ACCENT,
        )
        hint.to_edge(DOWN, buff=0.3)
        self.play(FadeIn(hint, shift=UP * 0.1), run_time=0.5)
        self.wait(10.0)
