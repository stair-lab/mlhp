# Foundations — Video Script

**Chapter 1 of Machine Learning from Human Preferences (MLHP)**
Target length: ~30 minutes
Format: Narrated animation (3Blue1Brown style)

---

## Production Notes

- **Animations** are in `animations/ch1/*.py` (Manim, 1080p60). Each scene listed
  below corresponds to a rendered `.mp4` in `media/ch1/videos/`.
- **Narration** is generated via edge-tts and synced with `stitch_narrated.sh`.
  The `[ANIMATION]` cues mark when each Manim scene plays.
- **Pacing markers:** `[pause]` = ~1 s beat. `[beat]` = ~0.5 s.

---

## PART 1 — THE SETUP (~4:30)

### 1.1 Opening

**NARRATOR:**

Welcome to Chapter 1: Foundations. This chapter is about one of the most fundamental questions in machine learning: how do we learn from human preferences?

[pause]

Preference data is everywhere. When you watch a video on YouTube and skip past another, you've expressed a preference. When you click on one search result instead of another, you've made a comparison. When a human annotator says that one language model response is more helpful than another, they've generated a pairwise observation that will be used to align that model with human values.

[pause]

The mathematical framework we develop in this chapter unifies all of these settings. Whether you're building a recommender system at Netflix, training a robot to move smoothly, running an Elo rating system for chess players, or aligning a large language model using RLHF or DPO — the underlying mathematics is the same. It all traces back to a model that was first proposed in 1952 by Ralph Bradley and Milton Terry for ranking chess players. The Bradley-Terry model says that the probability of preferring item j over item k is the sigmoid function of the difference in their utilities.

[pause]

In this chapter, we'll build up to that model step by step. We'll see where it comes from, why it works so well, and — just as importantly — when it fails. Let's begin.

### 1.2 The Response Matrix

**NARRATOR:**

> [ANIMATION: `response_matrix.py` — `ResponseMatrixSort`]
> Cue: Show the random grid appear

Everything starts with the response matrix. This is a binary matrix where each row represents a user, and each column represents an item. Each cell is either zero or one — did user i engage with item j, or not? In a streaming service, this might be whether a user watched a show. In an e-commerce platform, whether they purchased a product. In a language model evaluation, whether an annotator found a response acceptable.

[pause]

At first glance, this matrix looks like random noise. There's no obvious structure — just a scattered pattern of ones and zeros. The naive approach would be to simply average across each row to get a user's engagement rate, or average down each column to get an item's popularity. But that misses the deeper structure hiding in this data.

[pause]

Watch what happens when we sort.

> Cue: Row sort animation

First, we sort the rows by total engagement — users who accept the most items go to the top, and the most selective users go to the bottom. These row totals measure what we'll call user appetite: some users are enthusiastic and like almost everything, while others are highly selective and only engage with a few items.

> Cue: Column sort animation

Next, we sort the columns by total popularity — the most universally appealing items move to the left, and the niche items move to the right.

> Cue: Diagonal reveal

And suddenly, a striking diagonal pattern emerges. Look at this carefully. Users with high appetite — at the top — engage with almost everything, including niche items. But selective users — at the bottom — only engage with the most popular items. This creates a smooth transition from acceptance to rejection along a diagonal.

[pause]

This pattern is not an accident. It's exactly what a simple probabilistic model predicts. That model is called the Rasch model, developed by the Danish mathematician Georg Rasch in the 1960s. The Rasch model says the probability that user i accepts item j is the sigmoid function of U_i plus V_j — where U_i captures the user's appetite and V_j captures the item's appeal. This additive structure is what produces the diagonal: when U_i and V_j are both high, the probability of engagement is very high. When both are low, it's very low. The diagonal marks the transition boundary.

[pause]

The remarkable thing about this simple additive model is how much structure it reveals. From a matrix of seemingly random binary observations, we can recover two hidden quantities: how enthusiastic each user is, and how appealing each item is. This is the starting point for everything that follows.

---

## PART 2 — DETERMINISTIC UTILITY MODELS (~4:00)

### 2.1 From Rasch to Bradley-Terry

**NARRATOR:**

> [ANIMATION: `rasch_to_bt.py` — `RaschToBradleyTerry`]
> Cue: Number line with dots appears

The Rasch model gives us a utility for every user-item pair: H_ij equals U_i plus V_j. The user appetite U_i is shown here in orange, and two item appeals V_j and V_k are shown in blue and red on this number line. The additive structure means that each user simply shifts all item utilities up or down by their appetite parameter.

[pause]

Now here's the key question: what happens when we shift from item-wise data — where users respond to individual items — to pairwise comparison data, where a user says which of two items they prefer?

[pause]

Under the Rasch model, the probability that user i prefers item j over item k is the sigmoid of the difference in utilities: sigma of H_ij minus H_ik.

> Cue: Expansion written

Let's expand this out. H_ij is U_i plus V_j, and H_ik is U_i plus V_k. So the argument of the sigmoid becomes U_i plus V_j minus U_i minus V_k.

[pause]

Now look carefully at this expression.

> Cue: U_i terms highlighted

Do you see the two U_i terms? One appears in the first group, one in the second. They have opposite signs.

> Cue: U_i terms crossed out

They cancel completely. The user appetite U_i vanishes from the equation.

[pause]

> Cue: Bradley-Terry boxed

What remains is sigma of V_j minus V_k — and this is exactly the Bradley-Terry model. The probability of preferring one item over another depends only on the difference in their item appeals, not on anything about the user.

[pause]

Think about what this means. This is a profound result with far-reaching implications. It tells us that pairwise comparison data is fundamentally limited: it can only reveal differences between items. It cannot tell us anything about the individual users making those comparisons. Whether your users are all very enthusiastic — with high U_i — or all very selective — with low U_i — the pairwise comparison probabilities are identical. The user parameters are invisible.

[pause]

> Cue: Implication rows appear

This has practical consequences for system design. If you have pairwise data — like A/B test results, Elo ratings, or RLHF comparisons — you can only recover item quality differences, not user characteristics. If you want to learn about individual users — to build a personalized recommender, for example — you need item-wise data, where you observe whether each user accepts or rejects each item individually. Item-wise data identifies both user appetites and item appeals. Pairwise data collapses the user dimension away.

[pause]

This derivation also has deep historical roots. The Bradley-Terry model dates to 1952, but the underlying idea — that pairwise preferences arise from differences in utilities — goes back to Louis Leon Thurstone in 1927. The connection between the Rasch model and Bradley-Terry was itself a significant insight: it showed that these seemingly different models from psychometrics and statistics are actually two views of the same mathematical structure.

---

## PART 3 — STOCHASTIC UTILITY (~4:00)

### 3.1 The Ackley Landscape

**NARRATOR:**

> [ANIMATION: `ackley_sampling.py` — `AckleySampling`]
> Cue: Contour plot appears

So far, we've been thinking of utilities as deterministic numbers. Item j has a fixed quality V_j, and that's that. But in the real world, preferences are noisy. The same person might prefer coffee over tea in the morning but reverse that preference in the evening. A language model annotator might rate the same response differently depending on their mood or how carefully they read it. This noise isn't a flaw in the data — it's an inherent feature of human decision-making.

[pause]

To formalize this, we introduce the random utility model, first proposed by Thurstone in 1927 — nearly a century ago. The idea is beautifully simple: each item j has a mean utility V_j, but the actual perceived utility at the moment of choice is V_j plus some random noise epsilon_j. Different noise realizations lead to different choices, even for the same underlying preferences.

[pause]

Let's visualize this concretely. Here we're looking at the Ackley function as a utility landscape over a two-dimensional feature space. Bright regions represent high utility — items that are generally desirable. Dark regions represent low utility. The rugged, multi-modal terrain captures the idea that utility can vary in complex ways across the feature space — there may be multiple "peaks" of desirability and subtle valleys between them.

[pause]

Now, imagine observing decisions from this landscape. There are two fundamentally different types of data we can collect.

> Cue: Accept-reject dots appear

The first is accept-reject data. We show a user an item from this landscape and they either accept it — shown in blue — or reject it — shown in red. This is like a thumbs-up / thumbs-down rating. Notice how the accepted items cluster in the bright, high-utility regions, while rejected items appear more often in the dark, low-utility areas. This is item-wise data: each observation tells us about one item in isolation.

[pause]

> Cue: Pairwise comparison pairs appear

The second type is pairwise comparison data. Here we show two items simultaneously and ask which is preferred. The winner — with higher utility — is shown in green, and the loser in purple. The connecting line shows which two items were being compared. Notice that we don't need to know the absolute utility of either item — we only learn which one was better. This is the kind of data generated in RLHF, in Chatbot Arena battles, and in many A/B testing scenarios.

[pause]

The key insight is that these are two windows into the same underlying reality. The utility landscape doesn't change — what changes is how we sample from it. And as we showed in the last section, when the noise follows certain distributions, pairwise data can only reveal the landscape up to a global shift. The absolute heights of the peaks are invisible; only the relative heights can be recovered.

[pause]

The stochastic utility model gives us the formal equation: the perceived utility H-tilde-j equals V_j plus epsilon_j. The choice of distribution for epsilon turns out to have profound consequences. One particular choice — the Gumbel distribution — leads to the softmax function and the Independence of Irrelevant Alternatives. That's where we're headed next.

---

## PART 4 — INDEPENDENCE OF IRRELEVANT ALTERNATIVES (~4:00)

### 4.1 Softmax and Plackett-Luce

**NARRATOR:**

> [ANIMATION: `softmax_choice.py` — `SoftmaxChoice`]
> Cue: Bar chart with 4 items appears

When we assume that the noise terms epsilon follow independent Gumbel distributions — sometimes called Type-I extreme value distributions — something remarkable happens. The choice probabilities take the form of the softmax function: the probability of choosing item j from a set S is e-to-the-V_j divided by the sum of e-to-the-V_k over all items k in S.

[pause]

This formula has a property called the Independence of Irrelevant Alternatives, or IIA. IIA says that the ratio of choice probabilities between any two items does not depend on what other items are in the set. If you prefer A to B two-to-one when C is available, you'll still prefer A to B two-to-one when C is removed. The irrelevant alternative C doesn't affect the relative comparison.

[pause]

Here we see four items with different utilities. Item A has the highest utility and gets the largest probability. Item D has the lowest and gets the smallest. Notice how the bar widths are proportional to e-to-the-V — this is the softmax in action. Now, let's see what happens when we build a full ranking using these probabilities.

[pause]

> Cue: Plackett-Luce formula appears

The Plackett-Luce model extends softmax to full rankings. The idea is elegant: to generate a ranking, first choose the top-ranked item using softmax over the full set. Then remove it and choose the second-ranked item using softmax over the remaining items. Then remove that one, and continue until only one item is left.

[pause]

> Cue: Item A lifts out, bars rescale

Watch the bars carefully as we pick the highest-utility item and remove it from the set. After item A is chosen first, the softmax is recomputed over the remaining items B, C, and D. The probabilities all increase because the denominator has shrunk — but they increase proportionally. The ratio between any two remaining items stays the same. This is IIA at work.

> Cue: Next items picked

We continue the process — picking B next, then C, with D last by default. Each step is a softmax over a shrinking set.

[pause]

> Cue: Full product formula shown

The probability of the complete ranking A, B, C, D is the product of these sequential softmax terms. Each factor captures one step of the elimination process.

[pause]

Here's why this matters so much: a full ranking of M items is one of M-factorial possible orderings. With just 10 items, that's over 3.6 million possible rankings. Specifying a probability for each would require millions of parameters. But under the Plackett-Luce model, we only need M parameters — one utility V_j per item. IIA collapses an exponential space down to a linear one. This extraordinary parameter efficiency is why the softmax model is so widely used in practice.

> Cue: Binary special case

And when the choice set has just two items — a binary comparison — the softmax fraction simplifies. The probability reduces to sigma of V_j minus V_k — which is exactly the Bradley-Terry model once again.

[pause]

So we can now see the full picture: Bradley-Terry is a special case of Plackett-Luce, which is the softmax model for rankings, which arises from random utility theory with Gumbel noise, which satisfies IIA. It's a beautiful chain of equivalences. But IIA, for all its elegance, has a critical weakness. Let's see what goes wrong.

---

## PART 5 — WHEN IIA FAILS (~10:30)

### 5.1 The Red Bus / Blue Bus Problem

**NARRATOR:**

> [ANIMATION: `red_bus_blue_bus.py` — `RedBusBlueBus`]
> Cue: Before bars shown

The most famous failure of IIA is the red bus / blue bus problem, originally described by the economist Daniel McFadden — who would later win the Nobel Prize for his work on discrete choice models.

[pause]

Here's the setup. Imagine a commuter who can choose between taking a train or taking a bus to work. The bus has a higher utility — maybe it's cheaper and has a more convenient route. Under the softmax model, the train gets about 27 percent and the bus gets about 73 percent. So far, so good.

[pause]

Now imagine that the transit authority paints half of its buses blue and labels them as a separate option: "blue bus." This blue bus is identical to the original — now called "red bus" — in every possible way. Same route, same schedule, same seats, same fare. Common sense says: this cosmetic change shouldn't affect anyone's behavior. The train should still get 27 percent, and the 73 percent for buses should just split between red and blue.

[pause]

> Cue: Clone happens, train probability drops

But look at what IIA predicts. Because the blue bus has the same utility as the red bus, the softmax model treats all three options — train, red bus, blue bus — on equal footing. The train's probability drops to about 15.5 percent. That's a massive drop caused by nothing more than a paint job.

[pause]

The total bus share — red plus blue combined — has jumped from 73 to about 85 percent. The new buses stole probability not just from the original bus, but also from the completely unrelated train option. This is the proportional substitution property of IIA: removing or adding any item affects all other items proportionally. It cannot capture the fact that red bus and blue bus are close substitutes while the train is a completely different mode of transport.

[pause]

> Cue: Fix with correlated noise

The fix is to allow correlated noise between similar alternatives. When two options are close substitutes, their noise terms should be correlated — if you like one, you probably like the other. Models like the nested logit, developed by McFadden, group similar alternatives into nests and allow correlation within each nest. The probit model, which uses Gaussian noise instead of Gumbel noise, naturally handles arbitrary correlation structures. With correlated noise between the two bus options, the train's probability is correctly preserved at about 27 percent — exactly as common sense dictates.

[pause]

This isn't just a theoretical curiosity. In practice, whenever your choice set contains groups of similar alternatives — different flavors of the same product, different variants of the same language model response, different routing options that share the same highway — IIA will incorrectly predict that adding more options to one group will steal probability from every other group. Understanding this limitation is essential for any application of the softmax model.

### 5.2 Population Heterogeneity

**NARRATOR:**

> [ANIMATION: `mixture_iia.py` — `MixtureIIAViolation`]
> Cue: Two Gaussian curves shown

There's a second, subtler way that IIA breaks down: population heterogeneity. Even if every individual decision-maker satisfies IIA perfectly, the aggregate population may not.

[pause]

Here's the intuition. Suppose you have two groups of users with different tastes. Group 1 strongly prefers item A, while Group 2 strongly prefers item C. Within each group, decisions follow the softmax model perfectly — IIA holds for every individual. The formula for the aggregate population probability is a weighted mixture: for each item, we sum the group-weighted softmax probabilities.

[pause]

The key insight is that the mixture of two Gumbel distributions is not itself a Gumbel distribution. Look at this mixture density — it's clearly bimodal, with two peaks. A single Gumbel distribution has only one peak and is always right-skewed. The mathematical foundation of IIA — the Gumbel max-stability property — only works when the noise is Gumbel. Once we mix populations, the effective noise distribution changes, and IIA breaks.

[pause]

> Cue: Mixture curve appears

The bimodal shape tells us that the aggregate population can't be described by a single softmax model. No matter how you choose the utility parameters, a single softmax can't capture the behavior of a mixed population with divergent preferences.

[pause]

> Cue: IIA ratio test

Let's verify this with a concrete numerical test. The IIA property says that the ratio P(A) over P(B) should be the same regardless of what other items are in the choice set. On the left, we test this for Group 1 alone: the ratio is identical whether or not item C is present. IIA holds perfectly within each group, as expected.

[pause]

But on the right, we test the mixture. The ratio P(A) over P(B) changes when we add or remove item C. IIA is violated at the population level. This happens because removing C shifts the relative influence of the two groups — Group 2 cared more about C, so its removal changes the effective population weighting.

[pause]

This finding has direct practical implications. In A/B testing, if your user base has distinct segments with different preferences, aggregate softmax models will give misleading predictions. In survey design, pooling heterogeneous respondents can mask important subgroup effects. The solution is to model heterogeneity explicitly — using latent class models, mixed logit models, or other approaches that allow different users to have different utility parameters. We'll cover these in detail in later chapters.

### 5.3 Gaussian Process Priors

**NARRATOR:**

> [ANIMATION: `gp_prior_samples.py` — `GPPriorSamples`]
> Cue: GP formula and RBF kernel shown

When the standard Bradley-Terry model isn't flexible enough — perhaps because the reward function has complex, nonlinear structure — we need a more expressive model class. One powerful approach is to use a Gaussian process as a prior over the reward function.

[pause]

A Gaussian process, or GP, is a distribution over functions. Instead of assuming that the reward is a fixed linear combination of features, we say: the reward function r of x could be any smooth function, and our prior beliefs about which functions are more likely are encoded in a kernel.

[pause]

The most common kernel is the RBF — the radial basis function — also called the squared exponential kernel. It has the form k(x, x-prime) equals sigma-f-squared times the exponential of negative the squared distance between x and x-prime, divided by two ell squared. The key parameter here is the length-scale ell, which controls how quickly the function can vary. A large length-scale means the function must change slowly — nearby inputs will have similar rewards. A small length-scale allows rapid variation — the function can wiggle freely.

[pause]

> Cue: Sample curves drawn

Here are five sample functions drawn from a GP prior with length-scale equal to one. Each colored curve is one possible reward function — the GP says all of these are plausible before we see any data. Notice how they're all reasonably smooth, crossing zero at different points, with peaks and valleys at different locations. This is the essence of a nonparametric model: we don't commit to any particular functional form. The data will eventually tell us which of these functions best explains the observed preferences.

[pause]

> Cue: Length-scale morph animation

Now watch what happens as we vary the length-scale. Starting with a large value — the function is very smooth, almost linear over this range. As we decrease ell, the function becomes increasingly wiggly. Features at finer and finer scales appear. The extreme of a very small length-scale gives something close to white noise — the function value at one point tells you almost nothing about its value nearby.

[pause]

In practice, the length-scale is learned from data, automatically finding the right level of smoothness. Too smooth, and you'll miss real structure in the preferences. Too wiggly, and you'll overfit to noise in the comparison data. The GP framework provides a principled way to navigate this trade-off through Bayesian model selection.

[pause]

> Cue: BT connection formula

We connect the GP back to preferences via Bradley-Terry: the probability that item A is preferred over item B is sigma of r(x_A) minus r(x_B), where r is drawn from the GP. This gives us a flexible, nonparametric preference model. The GP handles the complexity of the reward landscape, while Bradley-Terry provides the probabilistic link from rewards to observed comparisons.

[pause]

This GP-BT combination has been particularly successful in robotics and control, where the reward function over continuous state-action spaces can be highly nonlinear. It has also been applied to RLHF for language models, where the reward function over the space of possible responses may have complex structure that a linear model can't capture. The main challenge is computational: GP inference scales cubically with the number of data points, though inducing point approximations and other techniques have made it practical for moderately large datasets.

---

## PART 6 — CLOSING (~2:30)

### 6.1 Summary

**NARRATOR:**

> [ANIMATION: `section_titles.py` — `ChapterClosing`]
> Cue: Key Takeaways heading

Let's step back and see the full picture. We've covered a lot of ground in this chapter, and it's worth reflecting on how it all fits together.

[pause]

> Cue: Takeaway 1 appears

First: preference data pervades machine learning. From recommender systems that learn what you'll watch next, to search engines that rank documents, to robotics systems that learn from human demonstrations, to language model alignment via RLHF and DPO — the same mathematical framework underlies all of these applications. The Bradley-Terry model is the common thread.

[pause]

> Cue: Takeaway 2 appears

Second: Bradley-Terry is not just an ad hoc model. It arises naturally from random utility theory. When the noise in perceived utilities follows a Gumbel distribution, the choice probabilities take the softmax form, and pairwise comparisons follow Bradley-Terry. This provides a principled justification for a model that might otherwise seem arbitrary. The derivation from Rasch further shows how pairwise and item-wise models are intimately connected — with the user appetite parameter canceling in pairwise comparisons.

[pause]

> Cue: Takeaway 3 appears

Third: the Independence of Irrelevant Alternatives is both a blessing and a curse. It gives us extraordinary parameter efficiency — collapsing M-factorial possible rankings down to just M utility parameters. But it also makes strong assumptions: that all noise is independent, and that all users have the same preference structure. The red bus / blue bus problem shows what goes wrong with correlated alternatives, and population heterogeneity shows what goes wrong with diverse users.

[pause]

> Cue: Takeaway 4 appears

And fourth: when Bradley-Terry isn't enough, richer models are available. Mixture models handle population heterogeneity. Gaussian processes provide flexible nonparametric reward functions. Nested logit and probit models allow correlated noise between similar alternatives. Each of these extensions trades the simplicity of Bradley-Terry for the ability to capture more complex patterns in human preference data.

[pause]

These tools — Bradley-Terry, softmax, random utility models, Gaussian processes — are not just historical curiosities. They are the working machinery behind modern AI alignment, recommendation systems, and evaluation frameworks. The rest of this book builds on these foundations to show how to learn preference models from data, how to actively collect informative comparisons, and how to make decisions based on what we've learned.

[pause]

Every preference tells a story. The model determines what we can learn from it.

---

## Animation-Scene Mapping

| Script section | Animation file | Scene name |
|---------------|----------------|------------|
| 1.1 Opening | `section_titles.py` | `ChapterOpening` |
| 1.2 Response Matrix | `response_matrix.py` | `ResponseMatrixSort` |
| 2.1 Rasch to BT | `rasch_to_bt.py` | `RaschToBradleyTerry` |
| 3.1 Ackley | `ackley_sampling.py` | `AckleySampling` |
| 4.1 Softmax / PL | `softmax_choice.py` | `SoftmaxChoice` |
| 5.1 Red Bus | `red_bus_blue_bus.py` | `RedBusBlueBus` |
| 5.2 Mixture IIA | `mixture_iia.py` | `MixtureIIAViolation` |
| 5.3 GP Priors | `gp_prior_samples.py` | `GPPriorSamples` |
| 6.1 Summary | `section_titles.py` | `ChapterClosing` |

## Rendering All Animations

```bash
PATH="/lfs/local/0/sttruong/miniconda3/bin:$PATH"

# Content animations
for scene in \
  "response_matrix.py ResponseMatrixSort" \
  "rasch_to_bt.py RaschToBradleyTerry" \
  "ackley_sampling.py AckleySampling" \
  "softmax_choice.py SoftmaxChoice" \
  "red_bus_blue_bus.py RedBusBlueBus" \
  "mixture_iia.py MixtureIIAViolation" \
  "gp_prior_samples.py GPPriorSamples"; do
  set -- $scene
  manim -qh --disable_caching --media_dir media/ch1 animations/ch1/$1 $2
done

# Title cards
for scene in ChapterOpening Part1Title Part2Title Part3Title \
             Part4Title Part5Title ChapterClosing; do
  manim -qh --disable_caching --media_dir media/ch1 animations/ch1/section_titles.py $scene
done
```
