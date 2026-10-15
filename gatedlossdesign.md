# Task: Replace the Current PPI-Style Loss with Gold-Anchored Gradient-Gated Synthetic Training

## Goal

Replace the current PPI/PPI-loss training objective with a new optimization strategy.

The main principle is:

> Gold-labeled data is always the optimization anchor. Synthetic data is auxiliary evidence and should only affect training when its gradient is compatible with the gold gradient.

The desired behavior is:

- If synthetic data is useful, use it to improve training.
- If synthetic data is noisy or misleading, automatically reduce its contribution.
- If synthetic data strongly conflicts with gold data, ignore it.
- In the worst case, training should naturally reduce to ordinary gold-only supervised learning.

Do **not** preserve the old PPI loss formulation unless needed for baseline comparison.

---

## 1. Gold Loss

For a gold-labeled minibatch,

\[
\mathcal{B}_G = \{(x_i,y_i)\},
\]

compute the ordinary supervised loss

\[
L_G
=
\frac{1}{|\mathcal{B}_G|}
\sum_i
\ell(f_\theta(x_i),y_i).
\]

Compute its gradient

\[
g_G = \nabla_\theta L_G.
\]

This is the trusted optimization direction.

Gold supervision must always remain active.

---

## 2. Synthetic Loss

For a synthetic minibatch

\[
\mathcal{B}_S
=
\{(\tilde{x}_j,\tilde{y}_j)\},
\]

compute

\[
L_S
=
\frac{1}{|\mathcal{B}_S|}
\sum_j
\ell(f_\theta(\tilde{x}_j),\tilde{y}_j),
\]

with gradient

\[
g_S=\nabla_\theta L_S.
\]

Do not directly optimize

\[
L_G+\lambda L_S
\]

with a fixed synthetic weight.

Instead, determine the synthetic contribution from its compatibility with the gold gradient.

---

## 3. Measure Gold--Synthetic Gradient Compatibility

Compute cosine similarity between the two gradients:

\[
a_t
=
\frac{
g_G^\top g_S
}{
\|g_G\|_2\|g_S\|_2+\epsilon
}.
\]

Use only the non-conflicting part:

\[
r_t=\max(0,a_t).
\]

Therefore,

\[
a_t\leq0
\quad\Rightarrow\quad
r_t=0.
\]

If the synthetic gradient conflicts with the gold gradient, synthetic data should have zero contribution for that update.

---

## 4. Control Synthetic Gradient Magnitude

Even when the gradients are aligned, the synthetic gradient should not dominate the gold gradient.

Define

\[
c_t
=
\min
\left(
1,
\frac{
\kappa\|g_G\|_2
}{
\|g_S\|_2+\epsilon
}
\right),
\]

where `kappa` controls the maximum synthetic-to-gold gradient magnitude.

Use

```text
kappa = 1.0
```

as the default.

Define the effective synthetic weight

\[
\lambda_t=r_t c_t.
\]

The final gradient is

\[
\boxed{
g_{\mathrm{final}}
=
g_G+\lambda_t g_S.
}
\]

Then perform the normal optimizer update using `g_final`.

---

## 5. Required Gold-Only Fallback

This property is critical.

If

\[
g_G^\top g_S\leq0,
\]

then

\[
r_t=0,
\qquad
\lambda_t=0,
\]

and therefore

\[
\boxed{
g_{\mathrm{final}}=g_G.
}
\]

Thus bad synthetic data must not force the model away from the gold-only optimization direction.

The implementation must preserve this behavior.

---

## 6. Preferred Extension: Sample-Level Synthetic Gating

After the batch-level implementation works, add an optional sample-level mode.

For each synthetic observation \(j\), define

\[
L_{S,j}
=
\ell(f_\theta(\tilde{x}_j),\tilde{y}_j)
\]

and

\[
g_j=\nabla_\theta L_{S,j}.
\]

Compare each synthetic gradient against the gold gradient:

\[
a_j
=
\frac{
g_j^\top g_G
}{
\|g_j\|_2\|g_G\|_2+\epsilon
}.
\]

Define

\[
w_j
=
[\max(0,a_j)]^\gamma.
\]

Use

```text
gamma = 1.0
```

as the default.

Construct the accepted synthetic gradient

\[
g_A
=
\frac{
\sum_j w_j g_j
}{
\sum_j w_j+\epsilon
}.
\]

If

\[
\sum_j w_j \approx 0,
\]

set

\[
g_A=0.
\]

Then clip its magnitude relative to the gold gradient:

\[
\bar g_A
=
\min
\left(
1,
\frac{
\kappa\|g_G\|_2
}{
\|g_A\|_2+\epsilon
}
\right)
g_A.
\]

The final update becomes

\[
\boxed{
g_{\mathrm{final}}
=
g_G+\lambda\bar g_A.
}
\]

Use

```text
lambda = 1.0
```

as the initial default.

---

## 7. Implementation Priority

Implement the **batch-level version first** because it is computationally cheap:

```text
gold batch
    |
    v
L_gold -> g_gold
             \
              -> cosine similarity -> reliability gate
             /
L_syn  -> g_syn
    |
    v
magnitude control
    |
    v
g_final = g_gold + lambda_t * g_syn
    |
    v
optimizer.step()
```

Do not implement expensive per-sample gradients until the batch-level version has been tested.

---

## 8. Important Autograd Requirement

Do not implement this as

```python
loss = gold_loss + lambda_t * synthetic_loss
loss.backward()
```

if `lambda_t` itself remains connected to the gradient computation graph.

The gradient gate should be treated as an optimization controller rather than something the model can manipulate through higher-order derivatives.

Compute the required gradients explicitly and calculate alignment statistics using detached gradient vectors.

Conceptually:

```python
g_gold = grad(gold_loss)
g_syn = grad(synthetic_loss)

cosine = cosine_similarity(
    detach(g_gold),
    detach(g_syn),
)

reliability = max(0, cosine)

scale = min(
    1.0,
    kappa * norm(g_gold) / (norm(g_syn) + eps),
)

synthetic_weight = reliability * scale

g_final = g_gold + synthetic_weight * g_syn

assign_parameter_gradients(g_final)

optimizer.step()
```

Implement this correctly for all trainable model parameters.

Avoid unnecessary second-order gradients.

---

## 9. Logging

For every epoch, log at least:

```text
train_gold_loss
train_synthetic_loss
gold_grad_norm
synthetic_grad_norm
gradient_cosine
synthetic_weight
synthetic_active_fraction
validation_accuracy
validation_balanced_accuracy
validation_macro_f1
```

`synthetic_active_fraction` should measure the fraction of optimization steps for which

\[
\lambda_t>0.
\]

Also report mean and standard deviation of the gradient cosine and synthetic weight.

---

## 10. Baselines

The experiment must support at least the following training modes:

```text
gold
gold_plus_synthetic
gradient_gated
```

### `gold`

Optimize only

\[
L_G.
\]

### `gold_plus_synthetic`

Use the naive baseline

\[
L_G+\lambda L_S.
\]

### `gradient_gated`

Use the proposed gold-anchored gradient update

\[
g_{\mathrm{final}}
=
g_G+\lambda_tg_S.
\]

Keep model architecture, data splits, optimizer, learning rate, random seeds, early stopping, and evaluation metrics identical across the three modes.

---

## 11. Stress Test

The main experiment should explicitly vary synthetic-label quality.

For example, progressively corrupt the synthetic labels:

```text
0%
10%
20%
30%
50%
70%
100%
```

or use the equivalent corruption mechanism appropriate for the current task.

The expected qualitative behavior is:

### High-quality synthetic data

The method should frequently obtain

\[
g_G^\top g_S>0
\]

and therefore use synthetic evidence.

### Low-quality synthetic data

Gradient compatibility should decrease, causing the synthetic weight to shrink.

### Strongly misleading synthetic data

The method should frequently obtain

\[
\lambda_t=0
\]

and behave increasingly like gold-only training.

This adaptive transition is the primary behavior we want to test.

---

## 12. Do Not Assume the New Method Must Win

Do not tune the implementation specifically to manufacture improvement over the gold-only baseline.

The experiment should determine whether the mechanism actually provides the desired behavior.

In particular, inspect:

1. whether good synthetic data improves over gold-only;
2. whether synthetic weight decreases as synthetic quality deteriorates;
3. whether poor synthetic data approaches gold-only performance;
4. whether naive `gold_plus_synthetic` deteriorates more severely under synthetic corruption;
5. whether gradient cosine is predictive of whether synthetic data is useful.

Save all diagnostics necessary to analyze these questions.

---

## Summary

The new method should implement

\[
\boxed{
g_{\mathrm{final}}
=
g_G
+
\underbrace{
\max(0,\cos(g_G,g_S))
}_{\text{gold compatibility}}
\underbrace{
\min
\left(
1,
\frac{\kappa\|g_G\|}
{\|g_S\|+\epsilon}
\right)
}_{\text{magnitude control}}
g_S.
}
\]

The conceptual rule is:

> **Gold determines the direction. Synthetic data is allowed to accelerate learning only when its optimization signal is compatible with gold supervision. When synthetic evidence becomes unreliable, its influence should automatically disappear and training should fall back toward gold-only learning.**