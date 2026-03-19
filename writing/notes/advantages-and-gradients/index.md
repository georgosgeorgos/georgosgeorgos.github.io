---
title: "Advantages and Where to Find (Divergences over) Them"
date: "Mar 18, 2026"
category: "note"
---

This note explores the connections between Reverse KL divergence (Mohamed et al., 2020), Self-Distillation (Agarwal et al., 2024), and Policy Gradient algorithms (Williams, 1991). The core idea is to leverage on-policy samples from the model $\mathbf{z} \sim p_{\theta}(\mathbf{z})$ to optimize w.r.t. $\theta$:

$$\mathbb{E}_{p_{\theta}(\mathbf{z})}\left[\log p_{\theta}(\mathbf{z}) - \log q(\mathbf{z})\right]$$

against a teacher $q(\mathbf{z})$.

We show how this formulation naturally corresponds to on-policy reinforcement learning with a specific advantage function, and show how the teacher distribution $q(\mathbf{z})$ can be constructed from the model itself by conditioning on privileged information such as expert demonstrations or feedback signals.

## Preliminaries

- $p_{\theta}$: model with learnable parameters $\theta$
- $q$: target or expert distribution (no gradients)

### Reverse KL (rKL)

We sample the model/policy $p_{\theta}$ we are learning:

$$\mathbb{KL}[p_{\theta},q] = \int p_{\theta}(\mathbf{z}) \log \frac{p_{\theta}(\mathbf{z})}{q(\mathbf{z})} \, d\mathbf{z} = \mathbb{E}_{p_{\theta}(\mathbf{z})} \left[\log \frac{p_{\theta}(\mathbf{z})}{q(\mathbf{z})}\right]$$

### Gradient Reverse KL (grKL)

<span class="sidenote">We use the score function estimator $\nabla_{\theta} \log p_{\theta}(\mathbf{z})$ throughout. See Mohamed et al. (2020) for pathwise and measure-valued alternatives.</span>

$$\begin{aligned}
\nabla_{\theta} \int p_{\theta}(\mathbf{z}) \log \frac{p_{\theta}(\mathbf{z})}{q(\mathbf{z})} \, d\mathbf{z} &= \int p_{\theta}(\mathbf{z}) \nabla_{\theta} \log p_{\theta}(\mathbf{z}) \log \frac{p_{\theta}(\mathbf{z})}{q(\mathbf{z})} \, d\mathbf{z} \\
&= \mathbb{E}_{p_{\theta}(\mathbf{z})}\left[\nabla_{\theta} \log p_{\theta}(\mathbf{z}) \log \frac{p_{\theta}(\mathbf{z})}{q(\mathbf{z})}\right]
\end{aligned}$$

### Forward KL (fKL)

We sample a different model $q$ or data distribution:

$$\mathbb{KL}[q,p_{\theta}] = \int q(\mathbf{z}) \log \frac{q(\mathbf{z})}{p_{\theta}(\mathbf{z})} \, d\mathbf{z} = \mathbb{E}_{q(\mathbf{z})} \left[\log \frac{q(\mathbf{z})}{p_{\theta}(\mathbf{z})}\right]$$

### Gradient Forward KL (gfKL)

$$\begin{aligned}
\nabla_{\theta} \int q(\mathbf{z}) \log \frac{q(\mathbf{z})}{p_{\theta}(\mathbf{z})} \, d\mathbf{z} &= - \int q(\mathbf{z}) \nabla_{\theta} \log p_{\theta}(\mathbf{z}) \, d\mathbf{z} \\
&= - \mathbb{E}_{q(\mathbf{z})} \left[\nabla_{\theta} \log p_{\theta}(\mathbf{z})\right]
\end{aligned}$$

## From Divergences to Advantages

### Maximum Likelihood (MLE)

We minimize the fKL, optimizing the model likelihood w.r.t. samples from $q(\mathbf{z})$: <span class="sidenote">For standard supervised scenarios, $q(\mathbf{z})$ is the empirical data distribution (samples from a dataset).</span>

$$\mathcal{F}(\mathbf{z}) = \int q(\mathbf{z}) \log p_{\theta}(\mathbf{z}) \, d\mathbf{z} = \mathbb{E}_{q(\mathbf{z})} \left[\log p_{\theta}(\mathbf{z})\right]$$

and compute the gradient (gfKL):

$$\begin{aligned}
\nabla_{\theta} \int q(\mathbf{z}) \log p_{\theta}(\mathbf{z}) \, d\mathbf{z} &= \int q(\mathbf{z}) \nabla_{\theta} \log p_{\theta}(\mathbf{z}) \, d\mathbf{z} \\
&= \mathbb{E}_{q(\mathbf{z})} \left[\nabla_{\theta} \log p_{\theta}(\mathbf{z})\right]
\end{aligned}$$

### Policy Gradient (PG)

Policy gradients are related to the rKL but not equivalent in general. We consider on-policy methods only (i.e. methods that rely on samples from the model $p_{\theta}$). Our goal is to maximize the expected advantage $a(\mathbf{z})$:

$$\mathcal{J}(\mathbf{z}) = \int p_{\theta}(\mathbf{z}) \, a(\mathbf{z}) \, d\mathbf{z} = \mathbb{E}_{p_{\theta}(\mathbf{z})} \left[a(\mathbf{z})\right]$$

Based on how we define the advantage $a(\mathbf{z})$, we can derive a suite of methods all based on the same policy gradient idea. Some useful variants:

| Method | $a(\mathbf{z})$ | References |
|---|---|---|
| Degenerate | $\mathbb{1} \cdot c$ | — |
| Rejection Sampling | $\mathbb{1}(\mathbf{z})$ | Casella et al., 2024; Zelikman et al., 2022; Gulcehre et al., 2023; Dong et al., 2023 |
| REINFORCE | $r(\mathbf{z}) - b$ | Williams, 1992; Ahmadian et al., 2024 |
| PPO | $\left[r(\mathbf{z}) - v_{\psi}(\mathbf{z})\right]_{\epsilon}$ | Schulman et al., 2017 |
| GRPO-style | $\sigma(r(\mathbf{z}) - \bar{r})$ | Shao et al., 2024; Ahmadian et al., 2024; Liu et al., 2026; Yu et al., 2025 |
| Entropy | $-\log p_{\theta}(\mathbf{z})$ | Haarnoja et al., 2018 |
| rKL | $-\log \frac{p_{\theta}(\mathbf{z})}{q(\mathbf{z})}$ | Shenfeld et al., 2026; Hubotter et al., 2026; Rafailov et al., 2024 (off-policy) |

All these methods share the same underlying policy gradient, differing only in the choice of advantage: <span class="sidenote">We assume no gradient on $q$ and $\int p_{\theta}(\mathbf{z}) \nabla_{\theta} \log p_{\theta}(\mathbf{z}) = 0$. The gradient of the policy can also be interpreted as a re-weighted regression task, where samples from the model (not from a dataset) are reweighted based on some scoring rule.</span>

$$\begin{aligned}
\nabla_{\theta} \int p_{\theta}(\mathbf{z}) \, a(\mathbf{z}) \, d\mathbf{z} &= \int p_{\theta}(\mathbf{z}) \nabla_{\theta} \log p_{\theta}(\mathbf{z}) \, a(\mathbf{z}) \, d\mathbf{z} \\
&= \mathbb{E}_{p_{\theta}(\mathbf{z})}\left[\nabla_{\theta} \log p_{\theta}(\mathbf{z}) \, a(\mathbf{z})\right]
\end{aligned}$$

### Knowledge Distillation (KD)

Knowledge Distillation (Hinton et al., 2015), which optimizes the forward KL divergence, is the standard approach for distilling large, expert models into smaller, more efficient ones (Chiang et al., 2023; Guo et al., 2025; Sudalairaj et al., 2024) using supervised fine-tuning. However, effective KD often interferes with the student model's underlying distribution, leading to catastrophic forgetting and brittle behavior: <span class="sidenote">Practitioners typically require additional techniques to mitigate these issues, including tuning on complementary data to refresh the training distribution, careful learning rate scheduling, and various stabilization tricks (Pareja et al., 2024).</span>

$$\mathbb{KL}[q_{\text{teacher}}, p_{\theta}] = \int q_{\text{teacher}}(\mathbf{z}) \log \frac{q_{\text{teacher}}(\mathbf{z})}{p_{\theta}(\mathbf{z})} \, d\mathbf{z}$$

$$\begin{aligned}
\nabla_{\theta} \int q_{\text{teacher}}(\mathbf{z}) \log \frac{q_{\text{teacher}}(\mathbf{z})}{p_{\theta}(\mathbf{z})} \, d\mathbf{z} &= - \mathbb{E}_{q_{\text{teacher}}(\mathbf{z})} \left[\nabla_{\theta} \log p_{\theta}(\mathbf{z})\right]
\end{aligned}$$

### Self-Distillation (SD)

Self-Distillation (Agarwal et al., 2024; Shenfeld et al., 2026; Zhao et al., 2026; Hubotter et al., 2026) is closely related to Knowledge Distillation, but employs a fundamentally different sampling strategy. Rather than distilling teacher knowledge into the student via off-policy sampling from expert demonstrations, Self-Distillation matches student logits to teacher logits using on-policy samples. While both methods share the goal of improving the student model, they differ in their optimization strategies:

- **Knowledge Distillation** samples from the teacher distribution and fine-tunes the student on these samples. This off-policy approach tends to interfere with the student's underlying distribution and optimizes a forward KL (fKL) or maximum likelihood estimation (MLE) objective.

- **Self-Distillation** samples from the student distribution and matches these samples to teacher logits obtained by conditioning on demonstrations, rich feedback, or additional context. This approach optimizes a reverse KL (rKL) or reweighted MLE objective, which can be interpreted as policy gradient optimization with the log density ratio serving as the reward signal.

The key advantage of on-policy updates is that sampling directly from the student distribution reduces distribution interference, yielding improved performance with less catastrophic forgetting. Recent work has demonstrated the effectiveness of Self-Distillation for continual learning (Hubotter et al., 2026) and on-policy reinforcement learning (Shenfeld et al., 2026).

Notably, the source of teacher feedback determines the nature of the learning signal. When demonstrations condition the teacher (Shenfeld et al., 2026), the approach remains driven by external feedback. Conversely, when the student model itself provides feedback (Hubotter et al., 2026), Self-Distillation resembles reinforcement learning with intrinsic motivation, driven by exploration, novelty, or autonomous feedback mechanisms.

The Self-Distillation objective optimizes the reverse KL between the student and the teacher:

$$\mathbb{KL}[p_{\theta}, q_{\text{teacher}}] = \int p_{\theta}(\mathbf{z}) \log \frac{p_{\theta}(\mathbf{z})}{q_{\text{teacher}}(\mathbf{z})} \, d\mathbf{z}$$

$$\begin{aligned}
\nabla_{\theta} \int p_{\theta}(\mathbf{z}) \log \frac{p_{\theta}(\mathbf{z})}{q_{\text{teacher}}(\mathbf{z})} \, d\mathbf{z} &= \mathbb{E}_{p_{\theta}(\mathbf{z})}\left[\nabla_{\theta} \log p_{\theta}(\mathbf{z}) \log \frac{p_{\theta}(\mathbf{z})}{q_{\text{teacher}}(\mathbf{z})}\right]
\end{aligned}$$

or, if the teacher is the model itself with privileged information $\mathbf{h} = h(\mathbf{z})$:

$$\begin{aligned}
\nabla_{\theta} \int p_{\theta}(\mathbf{z}) \log \frac{p_{\theta}(\mathbf{z})}{[p_{\theta}(\mathbf{z} | \mathbf{h})]_{\text{sg}}} \, d\mathbf{z} &= \mathbb{E}_{p_{\theta}(\mathbf{z})}\left[\nabla_{\theta} \log p_{\theta}(\mathbf{z}) \log \frac{p_{\theta}(\mathbf{z})}{p_{\theta}(\mathbf{z} | \mathbf{h})}\right]
\end{aligned}$$

This has the same form as the policy gradient, with the log density ratio acting as the advantage:

$$\mathbb{E}_{p_{\theta}(\mathbf{z})}\left[\nabla_{\theta} \log p_{\theta}(\mathbf{z}) \, a(\mathbf{z})\right], \quad a(\mathbf{z}) = -\log \frac{p_{\theta}(\mathbf{z})}{p_{\theta}(\mathbf{z} | \mathbf{h})}$$

### Sparse and Dense Feedback

Reinforcement Learning from Verifiable Reward (RLVR) provides reward or verification on the target. SD provides feedback at the token level, similar to an intrinsic process reward.

To clarify this point, consider autoregressive rollouts $\mathbf{z}_i = \{\mathbf{z}^1_i, \mathbf{z}^2_i, \ldots, \mathbf{z}^T_i\}$ of length $T$. <span class="sidenote">We omit the prompt $\mathbf{c}$ for clarity.</span> In RLVR, given a group of rollouts $\{\mathbf{z}_i\}^G_{i=1}$ the advantage for rollout $\mathbf{z}_i$ at token $t$ is: <span class="sidenote">RLVR can only provide delayed feedback over the output and over a group. Each token $t$ receives the same reward and only a posteriori.</span>

$$\mathbf{z}_{i} \sim p(\mathbf{z}^1_i) \prod^{T}_{t=2} p_{\theta}(\mathbf{z}^{t}_i | \mathbf{z}^{t-1}_i) \quad i \in \{1, \ldots, G\}$$

$$a(\mathbf{z}^t_{i}) = r_i(\mathbf{z}_{i}) - \frac{1}{G}\sum_{j=1}^G r_j(\mathbf{z}_j), \quad \forall t \in \{1, \ldots, T\}$$

The advantage for SD is:

$$\mathbf{z}^t_{i} \sim p(\mathbf{z}^1_i) \prod^{\hat t}_{t=2} p_{\theta}(\mathbf{z}^{t}_i | \mathbf{z}^{t-1}_i)$$

$$a(\mathbf{z}^t_{i}) = -\log \frac{p_{\theta}(\mathbf{z}^t_i | \mathbf{z}^{t-1}_i)}{p_{\theta}(\mathbf{z}^t_i | \mathbf{z}^{t-1}_i, \mathbf{h})}$$

where $\mathbf{h}$ is some form of additional information (hint, demonstration, env-feedback, expert-feedback, self-feedback, reflexion) $\mathbf{h} \leftarrow f(\mathbf{z}^{1:t}_{i})$, $\mathbf{h} \sim D_{\text{expert}}$. Notice how SD naturally provides feedback on partial trajectories without requiring the group computation. This means that we can decouple batch/group sampling and optimization. <span class="sidenote">In Self-Distillation, the teacher is typically the same model conditioned on additional information. Formally, $q_{\text{teacher}}(\mathbf{z}) := p_{\theta}(\mathbf{z} | \mathbf{h})$, where $\mathbf{h}$ can take various forms: expert demonstrations ($\mathbf{h} \sim D_{\text{expert}}$), feedback signals ($\mathbf{z} \sim p_{\theta}(\mathbf{z}), \mathbf{h} \leftarrow h(\mathbf{z})$), hints (Zelikman et al., 2022; Liao et al., 2026), or other privileged information.</span>

**Implementation notes:**

- Gradient should be stopped over the self-teacher.
- When using the same model with different contexts as the teacher, training can become unstable. EMA updates provide an effective stabilization mechanism (Shenfeld et al., 2026).
- We can use the top-K logits to make the advantage computation inexpensive.
- SD can be used in an SFT or RL pipeline with minimal modifications.
- This method *does not require sampling the teacher model*. We can scale this approach with increase in peak memory but limited increase in computational time.
- Rejection sampling can filter out samples that deviate too far (or too little) from the teacher distribution, effectively implementing an automatic curriculum.

## Related Formulations

### Policy Gradient with KL Regularization

Following Korbak et al. (2022) and Rafailov et al. (2024), PPO-like algorithms can be reinterpreted through the lens of Bayesian inference, connecting reinforcement learning objectives to probabilistic modeling frameworks:

$$\mathcal{F}_{\text{PPO}} = \mathbb{E}_{p_{\theta}(\mathbf{z})} \left[a(\mathbf{z}) - \gamma \log \frac{p_{\theta}(\mathbf{z})}{p_{\text{ref}}(\mathbf{z})}\right]$$

Notice how the regularization term that pushes the model update to stay close to the reference is a reverse KL. <span class="sidenote">The reference is typically a previous model version or the SFT checkpoint.</span> This formulation consists of PG + rKL.

### Entropy Regularization

Many modern post-training pipelines incorporate entropy regularization (Haarnoja et al., 2018) to prevent mode collapse and encourage exploration during training:

$$\mathcal{H}(\mathbf{z}) = - \int p_{\theta}(\mathbf{z}) \log p_{\theta}(\mathbf{z}) \, d\mathbf{z}$$

Note that entropy is closely related to the reverse KL divergence. <span class="sidenote">The two differ by an expectation over a term involving $q(\mathbf{z})$ when $q$ is non-uniform.</span>

### On-policy and Off-policy Sampling

Broadly, *on-policy* learning samples from the same model being optimized, $p_{\theta}$, whereas *off-policy* learning relies on samples produced by a different distribution (e.g., a fixed dataset, a teacher model, or an older snapshot of the same model such as a base SFT checkpoint).

In practice, truly on-policy sampling is difficult to achieve at scale. Most systems therefore use large-batch collection together with variants of importance sampling (or related corrections) to make training efficient, so policy-gradient methods are typically implemented as an alternating loop with a sampling phase followed by an optimization phase.

The shorter the sampling-to-training lag, the closer the procedure is to on-policy. <span class="sidenote">In modern pipelines, sampling is often handled by a dedicated inference stack (e.g., vLLM), while optimization runs in a separate training framework (e.g., RTF or verl); small implementation differences across these components can lead to mismatched logits, which should be accounted for when interpreting results as on-policy learning.</span>

### Amortized Variational Inference (AVI)

Amortized Variational Inference (Jordan et al., 1999; Rezende et al., 2014; Kingma et al., 2019) optimizes a specific instance of the reverse KL divergence. Specifically, we fit an approximate posterior $q_{\phi}(\mathbf{z})$ to match the true posterior $p(\mathbf{z} | \mathbf{x})$ by sampling from the approximate distribution:

$$\mathbb{KL}\left[q_{\phi}, p\right] = \mathbb{E}_{q_{\phi}(\mathbf{z})} \left[\log \frac{q_{\phi}(\mathbf{z})}{p(\mathbf{z} | \mathbf{x})}\right]$$

This KL formulation yields the evidence lower bound (ELBO) on the marginal likelihood: <span class="sidenote">When we fix the variational parameters $\phi$ and optimize only the model parameters $\theta$, the objective reduces to $\mathbb{E}_{q_{\phi}(\mathbf{z})}\left[\log p_{\theta}(\mathbf{x}, \mathbf{z})\right]$, corresponding to the negated forward KL or equivalently the MLE objective. Thus, AVI jointly optimizes reverse KL w.r.t. $\phi$ and forward KL w.r.t. $\theta$.</span>

$$\mathcal{F}(\mathbf{z}) = \mathbb{E}_{q_{\phi}(\mathbf{z})}\left[\log\frac{p_{\theta}(\mathbf{x}, \mathbf{z})}{q_{\phi}(\mathbf{z} | \mathbf{x})}\right]$$

---

The advantage perspective provides a unifying lens: MLE, KD, SD, entropy regularization, and RL methods all share the same policy gradient structure and differ only in how the advantage $a(\mathbf{z})$ is defined. The choice of advantage determines what the model learns — whether it imitates a teacher, maximizes a reward, or matches its own conditionals. The choice of teacher $q(\mathbf{z})$ determines the source of the learning signal — external demonstrations, environment feedback, or the model's own privileged knowledge.
