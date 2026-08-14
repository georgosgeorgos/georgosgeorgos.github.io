---
title: "How Off-Policy is Off-Policy Enough?"
date: "Apr 22, 2026"
category: "note"
---

This note develops the idea of using on-policy learning as the primary driver of policy optimization, and introducing off-policy signal only for what the policy cannot learn from its own rollouts — environment constraints, safety boundaries, privileged state information. We derive the gradient structure for several variants and show that the bounded formulation naturally introduces an adaptive gate, $1 - \sigma(B)$, that shuts off the off-policy signal precisely when the policy already satisfies the environment constraints.

## The Problem

On-policy reinforcement learning (<a href="https://mitpress.mit.edu/9780262039246/" target="_blank">Sutton & Barto, 2018</a>) is sample-inefficient, and this is especially problematic in multi-turn environments (<a href="https://arxiv.org/abs/2010.03768" target="_blank">Shridhar et al., 2021</a>; <a href="https://arxiv.org/abs/2207.01206" target="_blank">Yao et al., 2022</a>) where sparse outcome rewards fail to capture fine-grained intermediate mistakes. A successful end state can mask flawed reasoning or constraint violations along the way. <span class="sidenote">Given enough compute, a learner can eventually arrive at a correct outcome. However, outcome-only reward signals can easily assign high scores to successful end states, even if the agent's path relied on subtle errors or flawed reasoning.</span> To ensure robust compliance in domains where strict adherence to environment constraints is paramount, we must look beyond outcome rewards and leverage privileged states — underlying environment rules, ground-truth object configurations, or optimal intermediate plans — that are accessible during training but unavailable at deployment.

Recent approaches, including GRPO, DAPO, and GSPO (<a href="https://arxiv.org/abs/2501.12948" target="_blank">Guo et al., 2025</a>; <a href="https://arxiv.org/abs/2504.09696" target="_blank">Zheng et al., 2025</a>), have explored the trade-offs between on-policy stability and off-policy sample efficiency through various regularization techniques (<a href="https://arxiv.org/abs/2503.01132" target="_blank">Zhang et al., 2025</a>; <a href="https://arxiv.org/abs/2503.07752" target="_blank">Gao et al., 2025</a>). Despite these advancements, existing literature lacks a unified mechanism to seamlessly integrate and enforce complex environment constraints during the learning process.

## Background

### Policy Optimization

GRPO (<a href="https://arxiv.org/abs/2501.12948" target="_blank">Guo et al., 2025</a>) samples a group of $G$ rollouts per task from the policy, $\{\boldsymbol{\tau}_i\}^G_{i=1} \sim p_{\theta}(\boldsymbol{\tau} \mid q)$, and normalizes their rewards into advantages:

$$a_i = \frac{r_i - \bar{r}}{\sigma_r}, \qquad i = 1, \ldots, G.$$

Each token in rollout $i$ receives the same advantage $a_i$, and the policy is updated via a clipped surrogate objective. This works well when at least some rollouts succeed. When all $G$ rollouts receive the same reward, however, every advantage is zero and the gradient vanishes — the policy receives no learning signal.

We optimize the objective for a given problem $q$:

$$\mathcal{F}(s; \theta) = \mathbb{E}_{p_\theta(\boldsymbol{\tau} \mid q)}\!\left[a(\boldsymbol{\tau}, q)\right],$$

where for a group $G$ we optimize the estimator $\hat{\mathcal{F}}(s; \theta) = \frac{1}{G} \sum^{G}_{i=1} a(\boldsymbol{\tau}^{\theta}_i, q).$ This is the standard policy gradient target (<a href="https://link.springer.com/article/10.1007/BF00992696" target="_blank">Williams, 1992</a>). If a reference model is available, we can use the KL-penalized formulation:

$$\mathcal{F}^{\mathrm{ref}}(s; \theta) = \mathbb{E}_{p_\theta(\boldsymbol{\tau} \mid q)}\!\left[a(\boldsymbol{\tau}, q)\right] - \mathrm{KL}\!\left[p_{\theta}(\boldsymbol{\tau} \mid q) \,\|\, p_{\mathrm{base}}(\boldsymbol{\tau} \mid q)\right].$$

### Learning from Demonstrations

Given a dataset of paired trajectories $(\tau_p, \tau_n)$ for a problem $q$, where $\tau_p$ are positive or compliant trajectories (rollouts we want to learn) and $\tau_n$ are violating ones (rollouts we want to push away), we can write the DPO (<a href="https://arxiv.org/abs/2305.18290" target="_blank">Rafailov et al., 2024</a>) loss as:

$$\mathcal{F}^{\mathrm{dpo}}(\theta) = \log \sigma \left(\mathbb{E}_{q(\tau_p \mid q)} \log \frac{p_{\theta}(\tau_p \mid q)}{p_{\mathrm{base}}(\tau_p \mid q)} - \mathbb{E}_{q(\tau_n \mid q)} \log \frac{p_{\theta}(\tau_n \mid q)}{p_{\mathrm{base}}(\tau_{n} \mid q)}\right).$$


## Setup

Consider a policy $p_\theta$ acting in an environment $\mathcal{E}$ where a privileged state $s$ (object locations, optimal plans, or causal properties) is available during training but unavailable at test or deployment time (<a href="https://arxiv.org/abs/2006.09431" target="_blank">Kamienny et al., 2020</a>; <a href="https://arxiv.org/abs/2310.14800" target="_blank">Cai et al., 2024</a>). At each timestep, the agent observes context $q$ and produces a response $\boldsymbol{\tau}$ (multiple interleaved steps of reasoning followed by actions).

### Guide Model

A guide model $q$ observes three inputs that the student alone cannot access together: the context $q$, the student's on-policy response $\boldsymbol{\tau}_i$, and the privileged state $s$. From these, it produces:

- a *compliant* response $\boldsymbol{\tau}_c \sim q(\boldsymbol{\tau} \mid q, \boldsymbol{\tau}_i, s)$: reasoning and action consistent with $s$,
- a *violating* response $\boldsymbol{\tau}_v \sim q(\boldsymbol{\tau} \mid q, \boldsymbol{\tau}_i, s)$: reasoning that sounds plausible from $q$ alone but contradicts $s$.

A single guide model generates both types using different instructions. Conditioning on $\boldsymbol{\tau}_i$ ensures the contrastive samples target the student's current mistakes rather than generic errors.

### Regularizer

We score the compliant and violating responses under the student's own log-probability and define the regularizer as their weighted difference:

$$B(s; \theta) = \alpha \cdot \mathbb{E}_{q(\boldsymbol{\tau}_c)}\!\left[\log p_\theta(\boldsymbol{\tau}_c \mid q)\right] - (1 - \alpha) \cdot \mathbb{E}_{q(\boldsymbol{\tau}_v)}\!\left[\log p_\theta(\boldsymbol{\tau}_v \mid q)\right],$$

where $\alpha \in [0,1]$ balances the two terms ($\alpha = 0.5$ weights them equally) and log-probabilities are normalized by response length. Maximizing $B$ increases the student's probability of compliant behavior and decreases its probability of plausible violations.

### Learning Objective

We add the regularizer to the on-policy objective, scaled by $\lambda$:

$$\mathcal{F}(s; \theta) = \underbrace{\mathbb{E}_{p_\theta(\boldsymbol{\tau} \mid q)}\!\left[a(\boldsymbol{\tau}, q)\right]}_{\text{on-policy (GRPO)}} + \lambda \cdot \underbrace{\mathbb{E}_{p_\theta(\boldsymbol{\tau} \mid q)}\!\left[B(s; \theta)\right]}_{\text{off-policy regularization}}.$$

When all rollouts succeed or all fail, the GRPO term contributes zero gradient. The off-policy term is independent of rollout outcomes: it provides signal whenever the student assigns higher probability to violating responses than compliant ones. No additional parameters or architecture are introduced — the student's own log-probability is the scoring function.

**Bounded Regularization.** $B(s; \theta)$ is bounded for compliant behavior but unbounded for violations, and the training algorithm can exploit this asymmetry. <span class="sidenote">The unbounded direction is toward violations: the optimizer can drive violating log-probabilities arbitrarily negative to improve the loss without genuinely increasing compliance. Length-based reward hacking is also possible — short trajectories trivially satisfy the regularizer.</span> Drawing on recent preference optimization literature (<a href="https://arxiv.org/abs/2405.14734" target="_blank">Meng et al., 2024</a>) and approximating the expectations with a single sample, the bounded objective is:

$$B^n(s; \theta) = \log \sigma\left(\alpha \cdot \frac{\log p_\theta(\boldsymbol{\tau}_c \mid q)}{|\boldsymbol{\tau}_c|} - (1 - \alpha) \cdot \frac{\log p_\theta(\boldsymbol{\tau}_v \mid q)}{|\boldsymbol{\tau}_v|}\right).$$

We denote $\mathcal{F}^n(\theta; s)$ the corresponding objective, where we substitute $B(s; \theta)$ with $B^n(s; \theta)$.

**Compliant and Violating Samples.** Where generative compliant trajectories require some form of external feedback or privileged state information, violations are easier to get. However, using trivial violations will make the regularization ineffective. A powerful way to build violating samples is to use good but not perfect (based on output reward) samples from the policy. This means using a regularization of the form:

$$\mathbb{E}_{p_\theta(\tau \mid q)}\!\left[\mathbf{1}\{a(\tau, q) \neq a_{\mathrm{max}}\}\, B(s; \theta)\right],$$

where we assume that $q(\boldsymbol{\tau}_v \mid q, \boldsymbol{\tau}, s) \approx p_{\theta}(\boldsymbol{\tau} \mid q)$.


## Derivations

### General Gradients

When taking the gradient with respect to $\theta$, we obtain:

$$\nabla_{\theta} \mathcal{F}(\theta; s) = \nabla_{\theta} \mathbb{E}_{p_\theta(\tau \mid q)}\!\left[a(\tau, q)\right] + \nabla_{\theta} \mathbb{E}_{p_\theta(\tau \mid q)}\!\left[B(\tau; s, \theta)\right].$$

The first term of this equation is the standard policy optimization objective (<a href="https://link.springer.com/article/10.1007/BF00992696" target="_blank">Williams, 1992</a>):

$$\begin{aligned}
\nabla_{\theta} \mathbb{E}_{p_\theta(\tau \mid q)}\!\left[a(\tau, q)\right] &= \nabla_{\theta} \int_{\tau} p_{\theta}(\tau \mid q)\,a(\tau, q)\, d\tau \\
&= \int_{\tau} \nabla_{\theta}\, p_{\theta}(\tau \mid q)\,a(\tau, q)\, d\tau \\
&= \int_{\tau} p_{\theta}(\tau \mid q) \nabla_{\theta} \log p_{\theta}(\tau \mid q)\,a(\tau, q)\, d\tau \\
&= \mathbb{E}_{p_\theta(\tau \mid q)}\!\left[\nabla_{\theta} \log p_{\theta}(\tau \mid q)\,a(\tau, q)\right].
\end{aligned}$$

The second term is also the gradient of an expectation with respect to the policy. However, the regularization term is also a function of the parameters $\theta$. <span class="sidenote">The product rule is needed because $B(\theta)$ depends on $\theta$ through the student's log-probabilities evaluated on the off-policy samples. In contrast, the advantage $a(\tau, q)$ is a fixed scalar from the environment and does not depend on $\theta$.</span> We can then break this term as:

$$\begin{aligned}
\nabla_{\theta} \mathbb{E}_{p_\theta(\tau \mid q)}\!\left[B(\theta)\right] &= \nabla_{\theta} \int_{\tau} p_{\theta}(\tau \mid q)\, B(\theta)\, d\tau \\
&= \int_{\tau} \nabla_{\theta}\left[p_{\theta}(\tau \mid q)\, B(\theta)\right] d\tau \\
&= \int_{\tau} \nabla_{\theta} p_{\theta}(\tau \mid q)\, B(\theta)\, d\tau + \int_{\tau} p_{\theta}(\tau \mid q)\, \nabla_{\theta} B(\theta)\, d\tau \\
&= \int_{\tau} p_{\theta}(\tau \mid q) \nabla_{\theta} \log p_{\theta}(\tau \mid q)\, B(\theta)\, d\tau + \int_{\tau} p_{\theta}(\tau \mid q)\, \nabla_{\theta} B(\theta)\, d\tau \\
&= \mathbb{E}_{p_\theta(\tau \mid q)}\!\left[\nabla_{\theta} \log p_{\theta}(\tau \mid q)\, B(\theta)\right] + \mathbb{E}_{p_\theta(\tau \mid q)}\!\left[\nabla_{\theta} B(\theta)\right].
\end{aligned}$$

We can see that the gradients of $\mathcal{F}(\theta; s)$ are equivalent to an augmented advantage for the gradient of the score function $\log p_{\theta}(\tau \mid q)$ and an additional expectation under the model for the gradient of the regularization term. Putting all together, we get:

$$\nabla_{\theta} \mathcal{F}(\theta; s) = \mathbb{E}_{p_{\theta}(\tau \mid q)}\!\left[\left(a(\tau, q) + B(\tau; s, \theta)\right) \nabla_{\theta} \log p_{\theta}(\tau \mid q)\right] + \mathbb{E}_{p_{\theta}(\tau \mid q)}\!\left[\nabla_{\theta} B(\tau; s, \theta)\right].$$

The shape of this gradient is representative for all variants we consider.

**Base Regularization.** If we limit our attention to the regularization as written in $B(\tau; s, \theta)$, the gradient of the regularizer is simply:

$$\nabla_{\theta} B(\tau; s, \theta) = \mathbb{E}_{q(\tau_c)}\!\left[\nabla_{\theta} \log p_{\theta}(\tau_c \mid q)\right] - \mathbb{E}_{q(\tau_v)}\!\left[\nabla_{\theta} \log p_{\theta}(\tau_v \mid q)\right],$$

where the gradients are stopped over the guide $q$.

### Bounded Regularization

If we consider $B^n(\tau; s, \theta) = \log \sigma(B(\tau; s, \theta))$, we can simply substitute $B^n(\tau; s, \theta)$ in the policy gradient term, and the gradient of the regularizer becomes:

$$\nabla_{\theta} B^n(\theta) = \nabla_{\theta} \log \sigma(B(\theta)) = \frac{\sigma'(B(\theta))\, \nabla_{\theta} B(\theta)}{\sigma(B(\theta))}.$$

We already know how to compute $\nabla_{\theta} B(\theta)$. The gradient of the sigmoid function $\sigma(x) = \frac{1}{1 + e^{-x}}$ can be written as $\sigma'(x) = \sigma(x)(1 - \sigma(x))$, and we obtain the elegant simplification:

$$\nabla_{\theta} B^n(\theta) = \left[1 - \sigma(B(\theta))\right] \nabla_{\theta} B(\theta).$$

So the full gradient is:

$$\nabla_{\theta} \mathcal{F}^n(\theta; s) = \mathbb{E}_{p_{\theta}(\tau \mid q)}\!\left[\left(a(\tau, q) + B^n(\theta)\right) \nabla_{\theta} \log p_{\theta}(\tau \mid q)\right] + \mathbb{E}_{p_{\theta}(\tau \mid q)}\!\left[(1 - \sigma(B(\theta)))\, \nabla_{\theta} B(\theta)\right],$$

where we use $B^n(\theta)$ in the policy gradient estimator and a term $(1 - \sigma(B(\theta)))$ modulates the strength of the gradient over the environment-aware regularization in an adaptive way: if $B(\theta)$ is large, $\sigma(B(\theta))$ is close to 1 and the gradient over the regularizer has less weight, because the policy is already assigning the right preference to compliant and violating samples.

**Regularization with Reference.** If a reference or base model is available to stabilize learning, we can write the regularizer as:

$$B_{\mathrm{ref}}(\theta) = \mathbb{E}_{q(\tau_c)}\!\left[\log \frac{p_\theta(\tau_c \mid q)}{p_{\mathrm{base}}(\tau_c \mid q)}\right] - \mathbb{E}_{q(\tau_v)}\!\left[\log \frac{p_\theta(\tau_v \mid q)}{p_{\mathrm{base}}(\tau_v \mid q)}\right].$$

This mirrors the implicit reward in DPO (<a href="https://arxiv.org/abs/2305.18290" target="_blank">Rafailov et al., 2024</a>) based on the KL-constrained formulation in PPO (<a href="https://arxiv.org/abs/1707.06347" target="_blank">Schulman et al., 2017</a>). This derivation follows the optimal policy $p(\boldsymbol{\tau}) \propto p_{\mathrm{base}}(\boldsymbol{\tau})\, e^{r(\boldsymbol{\tau})}$, applied here to the off-policy regularizer. However, here the preference is based on the environment constraints ($s$) and sample-dependent $\boldsymbol{\tau}_i$. We can then bound this formulation using a log-sigmoid and obtain $B^n_{\mathrm{ref}}(s; \theta) = \log \sigma(B_{\mathrm{ref}}(s; \theta))$.

Because there are no gradients over the base model $p_{\mathrm{base}}$, the gradients are exactly the same as presented before, switching $B(\theta)$ for $B_{\mathrm{ref}}(\theta)$ and $B^n(\theta)$ for $B^n_{\mathrm{ref}}(\theta)$.

### Augmented Advantage

An alternative approach is to directly augment the advantage calculation leveraging the off-policy data. In practice, this means sampling the policy $\boldsymbol{\tau} \sim p_{\theta}(\boldsymbol{\tau} \mid q)$ and the guide $q(\boldsymbol{\tau}_{c/v} \mid q, \boldsymbol{\tau}, s)$ to get $\boldsymbol{\tau}_c$ and $\boldsymbol{\tau}_v$ given the rule-set $s$. Then we can compute the raw reward (assuming $r(\boldsymbol{\tau}) \in (0,1)$) as $(\boldsymbol{\tau}, r(\boldsymbol{\tau}))$, $(\boldsymbol{\tau}_c, 1)$, $(\boldsymbol{\tau}_v, 0)$. We now concatenate the samples $\boldsymbol{\tau}_{\mathrm{aug}} = [\boldsymbol{\tau}_c,\, \boldsymbol{\tau},\, \boldsymbol{\tau}_v]$ and rewards in $r_{\mathrm{aug}} = [1,\, r(\boldsymbol{\tau}),\, 0]$ and maximize:

$$\mathcal{F}^{\mathrm{aug}}(s; \theta) = \mathbb{E}_{p_\theta(\boldsymbol{\tau}_{\mathrm{aug}} \mid q)}\!\left[a_{\mathrm{aug}}(\boldsymbol{\tau}_{\mathrm{aug}}, q)\right],$$

where the augmented advantages are computed as $a_{\mathrm{aug}} = r_{\mathrm{aug}} - \bar{r}_{\mathrm{aug}}$. In this formulation we augmented the batch with the off-policy data and corresponding max min rewards and does not require an additional regularization term. <span class="sidenote">This directly tackles advantage collapse: even when all on-policy rollouts receive the same reward, the compliant and violating anchors at rewards 1 and 0 ensure a non-degenerate advantage distribution.</span>

The gradients are:

$$\nabla_{\theta} \mathcal{F}^{\mathrm{aug}}(s; \theta) = \mathbb{E}_{p_\theta(\boldsymbol{\tau}_{\mathrm{aug}} \mid q)}\!\left[\nabla_{\theta}\log p_\theta(\boldsymbol{\tau}_{\mathrm{aug}} \mid q)\,a_{\mathrm{aug}}(\boldsymbol{\tau}_{\mathrm{aug}}, q)\right],$$

that can be factorized as:

$$\begin{aligned}
\nabla_{\theta} \mathcal{F}^{\mathrm{aug}}(s; \theta) = &\;\mathbb{E}_{p_\theta(\boldsymbol{\tau} \mid q)}\!\left[\nabla_{\theta}\log p_\theta(\boldsymbol{\tau} \mid q)\,a_{\mathrm{aug}}(\boldsymbol{\tau}, q)\right] \\
+\; &\mathbb{E}_{p_\theta(\boldsymbol{\tau}_{c} \mid q)}\!\left[\nabla_{\theta}\log p_\theta(\boldsymbol{\tau}_c \mid q)\,a_{\mathrm{aug}}(\boldsymbol{\tau}_c, q)\right] \\
+\; &\mathbb{E}_{p_\theta(\boldsymbol{\tau}_{v} \mid q)}\!\left[\nabla_{\theta}\log p_\theta(\boldsymbol{\tau}_{v} \mid q)\,a_{\mathrm{aug}}(\boldsymbol{\tau}_{v}, q)\right].
\end{aligned}$$

We can see how training with policy gradient and off-policy environment-aware regularization is equivalent to augmenting the advantage function with the environment-aware regularization summed with the gradient of the regularizer itself. When using a reference model, the structure of the gradient is the same.


## How Off-Policy is Off-Policy Enough?

Traditional off-policy RL leverages experience replay buffers containing old trajectories. This offers no control over how far historical samples drift from the current policy. In contrast, environment-aware regularization formulates the off-policy component as the *minimal policy deviation* required to align with the privileged state.

By conditioning the guide model on $s$, we generate corrections specifically tailored to the current on-policy mistakes. Crucially, this mechanism is adaptive: if the privileged state $s$ does not indicate a constraint violation, the contrastive gap approaches zero ($B \approx 0$). The regularizer pushes trajectories off-policy only by the exact degree necessary to resolve issues identified by the privileged information, leaving core on-policy learning intact. <span class="sidenote">The off-policy samples $q(\tau_{c/v} \mid \tau_i, q, s)$ are a direct function of the on-policy generation $\tau_i$, the task context $q$, and the environment rule-set $s$. This is the key difference from standard replay-based off-policy methods.</span>

Notice how $1 - \sigma(B(\theta))$ adaptively modulates the off-policy intervention in the learning algorithm. When the model can fulfill the environment constraints or rule-set, and there is a large gap between compliant and violating, $\sigma(B(s;\theta))$ tends to 1, and $\nabla_{\theta}B^n(\tau, s; \theta)$ tends to zero. Similarly, $a(\tau, q) + B^n(\tau, s; \theta)$ tends to $a(\tau, q)$ when the gap between compliant and violating is large — the on-policy objective takes over entirely, and the off-policy regularization vanishes.
