"""
MLHP Chapter 1 — Section title cards & chapter bookends.
These short clips are stitched between the content animations.

Run with:
    PATH="/lfs/local/0/sttruong/miniconda3/bin:$PATH" \
    manim -qh --disable_caching --media_dir media/ch1 animations/ch1/section_titles.py <Scene>

Scenes (render in order):
    ChapterOpening
    Part1Title
    Part2Title
    Part3Title
    Part4Title
    Part5Title
    ChapterClosing
"""

from manim import *

# ── shared design tokens ────────────────────────────────────────────
ACCENT = "#FFD966"
BG = "#0f0f0f"
TEXT2 = "#aaaaaa"
PAL = ["#5B8DEE", "#45BF7C", "#F0A35C", "#E8637A", "#B07CD8"]


class _TitleBase(Scene):
    """Base class for consistent styling."""

    def construct(self):
        self.camera.background_color = BG

    def make_part_card(self, part_num, part_title, subtitle, color):
        """Animate a 'Part N' title card with accent line."""
        part_lbl = Text(
            f"Part {part_num}", font_size=22, color=TEXT2,
        )
        part_lbl.shift(UP * 0.8)

        title = Text(
            part_title, font_size=44, color=color, weight=BOLD,
        )
        title.next_to(part_lbl, DOWN, buff=0.3)

        line = Line(LEFT * 2, RIGHT * 2, color=ACCENT, stroke_width=1.5)
        line.next_to(title, DOWN, buff=0.3)

        sub = Text(subtitle, font_size=22, color=TEXT2)
        sub.next_to(line, DOWN, buff=0.3)

        # Animate in
        self.play(FadeIn(part_lbl, shift=DOWN * 0.1), run_time=0.5)
        self.play(FadeIn(title, shift=DOWN * 0.15), run_time=0.7)
        self.play(Create(line), run_time=0.4)
        self.play(FadeIn(sub, shift=UP * 0.1), run_time=0.5)
        self.wait(4.0)

        # Animate out
        group = VGroup(part_lbl, title, line, sub)
        self.play(FadeOut(group), run_time=0.7)
        self.wait(0.3)


# ════════════════════════════════════════════════════════════════════
#  CHAPTER OPENING
# ════════════════════════════════════════════════════════════════════

class ChapterOpening(_TitleBase):
    def construct(self):
        super().construct()

        # Chapter number
        ch = Text("Chapter 1", font_size=24, color=TEXT2)
        ch.shift(UP * 1.2)

        # Title
        title = Text(
            "Foundations",
            font_size=52, color=WHITE, weight=BOLD,
        )
        title.next_to(ch, DOWN, buff=0.35)

        # Accent line
        line = Line(LEFT * 3, RIGHT * 3, color=ACCENT, stroke_width=2)
        line.next_to(title, DOWN, buff=0.35)

        # Subtitle
        sub = Text(
            "Machine Learning from Human Preferences",
            font_size=28, color=ACCENT,
        )
        sub.next_to(line, DOWN, buff=0.35)

        # Animate
        self.play(FadeIn(ch, shift=DOWN * 0.1), run_time=0.6)
        self.play(FadeIn(title, shift=DOWN * 0.15), run_time=1.0)
        self.play(Create(line), run_time=0.5)
        self.play(FadeIn(sub, shift=UP * 0.1), run_time=0.7)
        self.wait(10.0)

        # Shrink title elements upward to make room
        title_group = VGroup(ch, title, line, sub)
        self.play(
            title_group.animate.scale(0.7).to_edge(UP, buff=0.3),
            run_time=0.8,
        )

        # Progressive domain reveal
        domain_hdr = Text("Preference data is everywhere",
                          font_size=26, color=WHITE, weight=BOLD)
        domain_hdr.next_to(title_group, DOWN, buff=0.4)
        self.play(FadeIn(domain_hdr, shift=DOWN * 0.1), run_time=0.5)

        domains = [
            ("Recommender Systems", "Netflix, Spotify, Amazon", PAL[0]),
            ("Search Ranking", "Google, Bing, retrieval", PAL[1]),
            ("Robotics", "Learning from human demos", PAL[2]),
            ("Elo Ratings", "Chess, Chatbot Arena", PAL[3]),
            ("LLM Alignment", "RLHF, DPO, Constitutional AI", PAL[4]),
        ]

        bullets = VGroup()
        for name, desc, color in domains:
            t = Text(name, font_size=22, color=color, weight=BOLD)
            d = Text(desc, font_size=18, color=TEXT2)
            d.next_to(t, RIGHT, buff=0.3)
            row = VGroup(t, d)
            bullets.add(row)

        bullets.arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        bullets.next_to(domain_hdr, DOWN, buff=0.3)

        for item in bullets:
            self.play(FadeIn(item, shift=RIGHT * 0.15), run_time=0.4)
            self.wait(4.0)

        # Unifying equation
        bt_eq = MathTex(
            r"P(j \succ k) = \sigma(V_j - V_k)",
            font_size=28, color=ACCENT,
        )
        bt_box = SurroundingRectangle(bt_eq, color=ACCENT, buff=0.12,
                                       stroke_width=1.5)
        bt_group = VGroup(bt_eq, bt_box)
        bt_group.next_to(bullets, DOWN, buff=0.35)
        self.play(Write(bt_eq), Create(bt_box), run_time=0.8)
        self.wait(8.0)


# ════════════════════════════════════════════════════════════════════
#  PART TITLE CARDS
# ════════════════════════════════════════════════════════════════════

class Part1Title(_TitleBase):
    def construct(self):
        super().construct()
        self.make_part_card(
            1, "The Setup",
            "Preference data and comparison types",
            PAL[0],
        )


class Part2Title(_TitleBase):
    def construct(self):
        super().construct()
        self.make_part_card(
            2, "Deterministic Utility Models",
            "Rasch, factor models, and the BT connection",
            PAL[1],
        )


class Part3Title(_TitleBase):
    def construct(self):
        super().construct()
        self.make_part_card(
            3, "Stochastic Utility",
            "Random utility models and mean utilities",
            PAL[2],
        )


class Part4Title(_TitleBase):
    def construct(self):
        super().construct()
        self.make_part_card(
            4, "Independence of Irrelevant Alternatives",
            "From Gumbel noise to softmax",
            PAL[3],
        )


class Part5Title(_TitleBase):
    def construct(self):
        super().construct()
        self.make_part_card(
            5, "When IIA Fails",
            "Heterogeneity and the red-bus/blue-bus problem",
            PAL[4],
        )


# ════════════════════════════════════════════════════════════════════
#  CHAPTER CLOSING
# ════════════════════════════════════════════════════════════════════

class ChapterClosing(_TitleBase):
    def construct(self):
        super().construct()

        # Key takeaways
        heading = Text("Key Takeaways", font_size=38,
                       color=WHITE, weight=BOLD)
        heading.to_edge(UP, buff=0.8)
        self.play(FadeIn(heading, shift=DOWN * 0.15), run_time=1.0)
        self.wait(10.0)

        takeaways = [
            "Preference data pervades ML: recommenders,\n"
            "    search, robotics, LLM alignment",
            "Bradley-Terry arises from random utility\n"
            "    models with Gumbel noise (IIA)",
            "IIA reduces parameter complexity from\n"
            "    M! to M, but fails under heterogeneity",
            "Richer models (mixtures, GPs, nested logit)\n"
            "    handle what Bradley-Terry cannot",
        ]

        prev = heading
        items = []
        for i, text in enumerate(takeaways):
            bullet = Text(
                f"{i+1}.", font_size=22, color=ACCENT, weight=BOLD,
            )
            body = Text(
                text, font_size=20, color=TEXT2, line_spacing=1.2,
            )
            row = VGroup(bullet, body).arrange(RIGHT, buff=0.2,
                                                aligned_edge=UP)
            row.next_to(prev, DOWN, buff=0.35, aligned_edge=LEFT)
            row.shift(RIGHT * 0.5)
            items.append(row)
            prev = row
            self.play(FadeIn(row, shift=RIGHT * 0.15), run_time=0.5)
            self.wait(15.0)

        self.wait(15.0)

        # Closing line
        line = Line(LEFT * 3, RIGHT * 3, color=ACCENT, stroke_width=1.5)
        line.next_to(items[-1], DOWN, buff=0.5)

        closing = Text(
            "Every preference tells a story.\n"
            "The model determines what we can learn from it.",
            font_size=24, color=WHITE, line_spacing=1.4,
        )
        closing.next_to(line, DOWN, buff=0.4)

        self.play(Create(line), run_time=0.4)
        self.play(FadeIn(closing, shift=UP * 0.1), run_time=0.8)
        self.wait(10.0)

        # Transition to "Coming Next"
        self.play(
            *[FadeOut(m) for m in self.mobjects],
            run_time=0.7,
        )

        # Coming next preview
        next_hdr = Text("Coming Next", font_size=34,
                         color=WHITE, weight=BOLD)
        next_hdr.to_edge(UP, buff=0.8)
        self.play(FadeIn(next_hdr, shift=DOWN * 0.1), run_time=0.6)

        chapters = [
            ("Chapter 2", "Choice Models", PAL[0]),
            ("Chapter 3", "Parameter Estimation", PAL[1]),
            ("Chapter 4", "Model Selection & Diagnostics", PAL[2]),
        ]

        ch_items = VGroup()
        for ch_num, ch_title, color in chapters:
            num = Text(ch_num, font_size=22, color=TEXT2)
            ttl = Text(ch_title, font_size=26, color=color, weight=BOLD)
            ttl.next_to(num, RIGHT, buff=0.3)
            row = VGroup(num, ttl)
            ch_items.add(row)

        ch_items.arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        ch_items.next_to(next_hdr, DOWN, buff=0.5)

        for item in ch_items:
            self.play(FadeIn(item, shift=RIGHT * 0.15), run_time=0.4)
            self.wait(3.0)

        # Final card
        final_line = Line(LEFT * 2.5, RIGHT * 2.5, color=ACCENT,
                          stroke_width=1.5)
        final_line.next_to(ch_items, DOWN, buff=0.5)
        site = Text("mlhp.stanford.edu", font_size=22, color=ACCENT)
        course = Text("CS329H, Stanford", font_size=18, color=TEXT2)
        site.next_to(final_line, DOWN, buff=0.3)
        course.next_to(site, DOWN, buff=0.15)

        self.play(Create(final_line), run_time=0.4)
        self.play(FadeIn(site), FadeIn(course), run_time=0.5)
        self.wait(8.0)
