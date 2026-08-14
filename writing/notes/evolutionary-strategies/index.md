---
title: "Smoothing Your Way to Gradient Ascent with Evolutionary Strategies"
date: "Aug 14, 2026"
category: "note"
---

This note develops the connection between policy gradients (PG) (<a href="https://link.springer.com/article/10.1007/BF00992696" target="_blank">Williams, 1992</a>) and evolutionary strategies (ES) (<a href="https://arxiv.org/abs/1703.03864" target="_blank">Salimans et al., 2017</a>; <a href="https://arxiv.org/abs/1106.4487" target="_blank">Wierstra et al., 2014</a>). We derive the core ES gradient estimator, interpret ES as gradient ascent on a Gaussian-smoothed objective, and explain why this formulation does not require differentiating the original objective. In this sense, ES is a zeroth-order method (<a href="https://doi.org/10.1007/s10208-015-9296-2" target="_blank">Nesterov et al., 2017</a>) for the original objective: it estimates the gradient of a Gaussian-smoothed objective using only function evaluations of the original objective.

## Notation

Let $\mathbf{z} \in \mathcal{Z}$ denote a sample from a policy or generative model. Depending on the application, $\mathbf{z}$ may represent an action sequence, a trajectory, a generated output, or a latent variable.

Let $p(\mathbf{z};\theta)$ denote a distribution over $\mathbf{z}$, parameterized by $\theta \in \mathbb{R}^d$. Let $f(\mathbf{z}) \in \mathbb{R}$ be a scalar scoring function, such as an environment return, a reward-model score, or a verification signal. Our goal is to maximize the expected score under the model.

We use $q(\theta;\mu)$ for a Gaussian search distribution over parameters $\theta$, with mean $\mu$ and fixed standard deviation $\sigma > 0$. In particular,

$$q(\theta;\mu) = \mathcal{N}(\theta;\mu,\sigma^2 I_d).$$

We use $q(\epsilon)$ to denote standard Gaussian noise over $\epsilon \in \mathbb{R}^d$:

$$\epsilon \sim q(\epsilon), \qquad q(\epsilon) = \mathcal{N}(0, I_d).$$

## Policy Optimization in Sample Space

We define the objective or fitness $F(\theta)$ as the expected score under the model:

$$F(\theta) = \mathbb{E}_{p(\mathbf{z}; \theta)}\left[f(\mathbf{z})\right] = \int p(\mathbf{z}; \theta) f(\mathbf{z})\, d\mathbf{z}.$$

Using the log-derivative trick, its gradient can be written as

$$\nabla_\theta F(\theta) = \mathbb{E}_{p(\mathbf{z}; \theta)} \left[ f(\mathbf{z})\nabla_\theta \log p(\mathbf{z}; \theta) \right].$$

This follows from

$$\begin{aligned}
\nabla_{\theta}\mathbb{E}_{p(\mathbf{z}; \theta)}\left[f(\mathbf{z})\right] &= \nabla_{\theta} \int p(\mathbf{z}; \theta) f(\mathbf{z})\, d\mathbf{z} \\
&= \int \nabla_{\theta}\, p(\mathbf{z}; \theta) f(\mathbf{z})\, d\mathbf{z} \\
&= \int p(\mathbf{z}; \theta) f(\mathbf{z}) \nabla_{\theta} \log p(\mathbf{z}; \theta)\, d\mathbf{z} \\
&= \mathbb{E}_{p(\mathbf{z}; \theta)}\left[f(\mathbf{z}) \nabla_{\theta} \log p(\mathbf{z}; \theta)\right].
\end{aligned}$$

This is the score-function gradient estimator (<a href="https://arxiv.org/abs/1906.10652" target="_blank">Mohamed et al., 2020</a>) and the basis for the REINFORCE algorithm (<a href="https://link.springer.com/article/10.1007/BF00992696" target="_blank">Williams, 1992</a>), a simple form of Monte Carlo Policy Gradient. Intuitively, it increases the probability of sampled trajectories or outputs $\mathbf{z}$ that receive high scores under $f$ in expectation. The function $f$ may represent an environment return, a reward model, or a verification signal.

## Evolutionary Strategies in Parameter Space

Computing or estimating gradients of $F(\theta)$ may be difficult or unfeasible when the objective is non-differentiable, expensive (requiring simulation or solvers), or available only through black-box evaluations (<a href="https://doi.org/10.1137/1.9780898718768" target="_blank">Conn et al., 2009</a>).

Evolutionary strategies (<a href="https://doi.org/10.1023/A:1015059928466" target="_blank">Beyer et al., 2002</a>; <a href="https://arxiv.org/abs/1703.03864" target="_blank">Salimans et al., 2017</a>) address this issue by optimizing a smoothed objective over a distribution of parameters.

Let $q(\theta;\mu)$ be a search distribution over model parameters (for example a Gaussian), with mean $\mu$ (we will assume $\sigma$ fixed).

$$\mathcal{J}(\mu) = \mathbb{E}_{q(\theta; \mu)}\left[F(\theta)\right] = \int q(\theta; \mu) F(\theta)\, d\theta.$$

A sample $\theta \sim q(\theta;\mu)$ corresponds to a perturbed model parameterization. We evaluate its fitness $F(\theta)$ and optimize the expected fitness under the search distribution. We can then compute the gradients w.r.t. the parameters $\mu$:

$$\nabla_{\mu} \mathcal{J}(\mu) = \mathbb{E}_{q(\theta; \mu)}\left[F(\theta) \nabla_{\mu} \log q(\theta; \mu) \right].$$

Notice that this expression does not require differentiating $F(\theta)$. We only need to evaluate it. The remaining gradient, $\nabla_\mu \log q(\theta;\mu)$, still requires gradients. However, the problem is tractable because we choose the search distribution $q$ ourselves.

### Gaussian Search

For this derivation, consider the one-dimensional case, $d=1$. The search distribution is

$$q(\theta;\mu) = \mathcal{N}(\mu,\sigma^2).$$

We can sample from this distribution by reparameterization

$$\theta = \mu + \sigma \epsilon, \qquad \epsilon \sim q(\epsilon).$$

The smoothed objective becomes

$$\mathcal{J}^{\epsilon}(\mu) = \mathbb{E}_{q(\epsilon)} \left[ F(\mu+\sigma\epsilon) \right].$$

Notice that $\mathcal{J}(\mu)=\mathcal{J}^{\epsilon}(\mu)$. In particular, for a one-dimensional Gaussian density, the change of variables $\theta=\mu+\sigma\epsilon$ gives

$$\begin{aligned}
\int q(\theta; \mu) F(\theta)\, d\theta &= \int \dfrac{1}{\sqrt{2 \pi} \sigma} \exp\left(- \dfrac{(\theta - \mu )^2}{2 \sigma^2}\right) F(\theta)\, d\theta \\
&= \int \dfrac{1}{\sqrt{2 \pi} \sigma} \exp\left(- \dfrac{( \sigma \epsilon )^2}{2 \sigma^2}\right) F(\mu + \sigma \epsilon) \sigma\, d\epsilon \\
&= \int \dfrac{1}{\sqrt{2 \pi}} \exp\left(- \dfrac{\epsilon^2}{2}\right) F(\mu + \sigma \epsilon)\, d\epsilon \\
&= \int q(\epsilon) F(\mu + \sigma \epsilon)\, d\epsilon.
\end{aligned}$$

This is the change-of-variables formula (<a href="https://doi.org/10.1201/9781003456285" target="_blank">Casella et al., 2024</a>) applied to the reparameterization $\theta=\mu+\sigma\epsilon$. <span class="sidenote">The Jacobian determinant $\sigma^d$ from the substitution cancels with the $\sigma^d$ in the Gaussian normalizing constant, leaving the standard normal density $q(\epsilon)$.</span> The same change-of-variables argument applies in $d$ dimensions, where $\left| \det \frac{\partial \theta}{\partial \epsilon}\right| = \sigma^d$.

We can then compute the gradients:

$$\begin{aligned}
\int q(\theta; \mu) F(\theta) \nabla_{\mu} \log q(\theta; \mu)\, d\theta &= \int \dfrac{1}{\sqrt{2 \pi} \sigma} \exp\left(- \dfrac{(\theta - \mu )^2}{2 \sigma^2}\right) F(\theta) \nabla_{\mu} \left[- \dfrac{( \theta - \mu )^2}{2 \sigma^2}\right] d\theta \\
&= \int \dfrac{1}{\sqrt{2 \pi} \sigma} \exp\left(- \dfrac{( \theta - \mu )^2}{2 \sigma^2}\right) F(\theta) \dfrac{\theta - \mu}{\sigma^2}\, d\theta \\
&= \int \dfrac{1}{\sigma} q(\epsilon) F(\mu + \sigma \epsilon) \dfrac{\epsilon}{\sigma} \sigma\, d\epsilon \\
&= \int q(\epsilon) F(\mu + \sigma \epsilon) \dfrac{\epsilon}{\sigma}\, d\epsilon,
\end{aligned}$$

where, for the Gaussian distribution, we use

$$\nabla_\mu \log q(\theta;\mu) = \frac{\theta-\mu}{\sigma^2} = \frac{\epsilon}{\sigma}.$$

### ES Gradient

**Score-function derivative.** Summarizing, the gradient of the Gaussian-smoothed objective can be written using the score-function identity as

$$\nabla_{\mu} \mathbb{E}_{q(\theta; \mu)}\left[F(\theta)\right] = \mathbb{E}_{q(\epsilon)}\left[F(\mu + \sigma \epsilon)\dfrac{\epsilon}{\sigma}\right].$$

This expression follows from the log-derivative trick applied to the search distribution $q(\theta;\mu)$, followed by the reparameterization $\theta=\mu+\sigma\epsilon$. It does not require differentiating $F(\theta)$: only the scalar fitness value $F(\mu+\sigma\epsilon)$ is needed. <span class="sidenote">In an automatic-differentiation implementation, gradients can be stopped through the fitness evaluation $F(\mu+\sigma\epsilon)$.</span>

**Pathwise derivative.** If $F(\theta)$ is differentiable, the pathwise gradient can be derived

$$\nabla_{\mu} \mathbb{E}_{q(\epsilon)}\left[F(\mu + \sigma \epsilon)\right] = \mathbb{E}_{q(\epsilon)}\left[\nabla_{\mu}F(\mu + \sigma \epsilon)\right].$$

Since $\theta=\mu+\sigma\epsilon$ and $\partial\theta/\partial\mu=I$, the chain rule gives

$$\nabla_\mu F(\mu+\sigma\epsilon) = \nabla_\theta F(\theta).$$

The pathwise form requires access to $\nabla_\theta F(\theta)$. Evolutionary strategies are useful precisely when this derivative is unavailable, expensive, or undefined — for example, when $F$ includes non-differentiable operations, simulations, external models, solvers, or other black-box components.

### Monte Carlo Estimators

In practice, the expectation is approximated using Monte Carlo samples. Let

$$\epsilon_1,\ldots,\epsilon_S \sim q(\epsilon).$$

For each perturbation, define the single-sample score-function estimate

$$g_\mu^s = F(\mu+\sigma\epsilon_s) \frac{\epsilon_s}{\sigma}.$$

The ES gradient estimator is the sample average

$$\hat{g}_\mu = \frac{1}{S} \sum_{s=1}^{S} g_\mu^s = \frac{1}{S} \sum_{s=1}^{S} F(\mu+\sigma\epsilon_s) \frac{\epsilon_s}{\sigma}.$$

This estimator requires only evaluations of the fitness $F(\mu+\sigma\epsilon_s)$; it does not require gradients through the fitness function. It is therefore a zeroth-order estimator (<a href="https://doi.org/10.1007/s10208-015-9296-2" target="_blank">Nesterov et al., 2017</a>) with respect to the original objective $F$, while estimating the gradient of the smoothed objective $\mathcal{J}(\mu)$. Notice that this estimator is unbiased for $\nabla_{\mu} \mathcal{J}(\mu)$ but not necessarily for $\nabla_{\mu} F(\mu)$. <span class="sidenote">The bias arises because $\mathcal{J}(\mu)$ is a smoothed version of $F(\mu)$. As $\sigma \to 0$, the smoothed objective recovers the original, and the bias vanishes.</span>

If $F$ is differentiable and its gradient is accessible, one may instead use the pathwise estimator. Let

$$g_\mu^p = \nabla_\mu F(\mu+\sigma\epsilon_p) = \nabla_\theta F(\theta) \bigg|_{\theta=\mu+\sigma\epsilon_p}.$$

Using $P$ iid perturbations, the pathwise Monte Carlo estimator is

$$\hat{g}_\mu = \frac{1}{P} \sum_{p=1}^{P} g_\mu^p.$$

## Connections and Extensions

Policy gradients (<a href="https://link.springer.com/article/10.1007/BF00992696" target="_blank">Williams, 1992</a>) and evolutionary strategies (<a href="https://arxiv.org/abs/1703.03864" target="_blank">Salimans et al., 2017</a>) are both based on the score-function identity. In both cases, the objective gradient is written as an expectation involving a scalar score and the gradient of the log probability of the distribution from which samples are drawn.

For policy gradients, the distribution is over actions, trajectories, or other model outputs:

$$\nabla_\theta F(\theta) = \mathbb{E}_{p(\mathbf{z};\theta)} \left[ f(\mathbf{z})\nabla_\theta \log p(\mathbf{z};\theta) \right].$$

For evolutionary strategies, the distribution is over model parameters:

$$\nabla_\mu \mathcal{J}(\mu) = \mathbb{E}_{q(\theta;\mu)} \left[ F(\theta)\nabla_\mu \log q(\theta;\mu) \right],$$

where

$$F(\theta) = \mathbb{E}_{p(\mathbf{z};\theta)} \left[ f(\mathbf{z}) \right].$$

ES can therefore be interpreted as policy-gradient optimization in parameter space. Policy gradients assign credit to sampled actions or trajectories according to their scores. ES instead assigns credit to random perturbations of the model parameters according to the fitness of the perturbed model. <span class="sidenote">This distinction has practical consequences: ES scales with parameter dimensionality $d$ rather than trajectory length, and can be parallelized trivially since each perturbation is evaluated independently.</span>

Unlike the policy-gradient expression, the ES update does not require access to derivatives of $F(\theta)$. It only requires function evaluations of $F$ at perturbed parameter values. The resulting update therefore performs gradient ascent, in expectation, on a Gaussian-smoothed version of the original objective.

This perspective underlies Natural Evolutionary Strategies (NES) (<a href="https://arxiv.org/abs/1106.4487" target="_blank">Wierstra et al., 2014</a>), parameter-based policy-gradient methods (<a href="https://arxiv.org/abs/0805.0826" target="_blank">Sehnke et al., 2010</a>), and large-scale ES optimization (<a href="https://arxiv.org/abs/2509.24372" target="_blank">Qiu et al., 2025</a>; <a href="https://arxiv.org/abs/2511.16652" target="_blank">Sarkar et al., 2025</a>). It is also closely related to variational optimization (<a href="https://arxiv.org/abs/1212.4507" target="_blank">Staines et al., 2012</a>) and has been applied to LLM-based search (<a href="https://arxiv.org/abs/2206.08896" target="_blank">Lehman et al., 2023</a>).

## Ideas

Combine the strengths of policy gradients and evolutionary strategies. When policy-gradient information is available, use PG to efficiently improve the current policy and move toward high-reward regions of parameter space. Then use ES to explore neighborhoods of those regions through coherent perturbations of the full parameter vector, potentially discovering robust, diverse, or otherwise interesting high-performing solutions.

Both methods optimize expected reward, but they operate at different levels. PG updates the parameters of a particular policy to improve its expected score under the policy's output or trajectory distribution. ES instead optimizes the parameters of a search distribution over policies, maximizing the expected score of perturbed parameter vectors. Thus, PG locally refines a specific policy, whereas ES searches over a distribution of related policies and can explore alternatives that may not be reached by local gradient updates alone.

### Constrained Exploration for Deceptive Problems

We propose a hybrid framework for LLM-based scientific discovery that combines policy gradients (PG) and evolutionary strategies (ES). The central idea is to use PG for efficient local improvement when policy-gradient information is available and reliable, while using ES to explore coherent finite changes in the parameters of the scientific agent. The agent may generate hypotheses, retrieve literature, select tools, plan experiments, and interpret evidence. Its objective is not only to maximize immediate task score, but to discover diverse, testable, and high-value scientific solutions.

Let $\theta$ denote the trainable parameters of the agent, such as adapter parameters, retrieval-policy parameters, planner parameters, or continuous prompts. Let $\mathbf{z}$ denote a scientific-discovery trajectory, which may include a hypothesis, retrieved evidence, tool calls, experimental plans, and an interpretation of the resulting evidence. The expected fitness is

$$F(\theta) = \mathbb{E}_{p(\mathbf{z};\theta)} \left[ f(\mathbf{z}) \right].$$

When trajectory log-probabilities are available, policy gradients provide a local update. Given samples $\mathbf{z}_1,\ldots,\mathbf{z}_N \sim p(\mathbf{z};\theta)$ and corresponding advantage estimates $a_1,\ldots,a_N$, a Monte Carlo estimator is

$$\widehat{g}_{\theta,\mathrm{PG}} = \frac{1}{N} \sum_{i=1}^{N} a_i \nabla_\theta \log p(\mathbf{z}_i;\theta).$$

This update exploits regularities among previously validated hypotheses, evidence-gathering strategies, and experiment plans.

**Multiscale ES.** To explore beyond the local policy-gradient direction, ES evaluates perturbed agent parameterizations drawn from Gaussian search distributions centered at $\mu$:

$$\theta = \mu+\sigma_k\epsilon, \qquad \epsilon\sim q(\epsilon), \qquad k=1,\ldots,K,$$

where $\sigma_1 < \cdots < \sigma_K$. <span class="sidenote">The choice of noise scales can itself be adapted during optimization. Natural Evolutionary Strategies (<a href="https://arxiv.org/abs/1106.4487" target="_blank">Wierstra et al., 2014</a>) learn both the mean and covariance of the search distribution.</span> Each noise scale defines a Gaussian-smoothed objective

$$\mathcal{J}_{\sigma_k}(\mu) = \mathbb{E}_{q(\epsilon)} \left[ F(\mu+\sigma_k\epsilon) \right].$$

Small perturbations search for robust local variations of a high-performing scientific workflow. Larger perturbations propose alternative hypothesis-generation, retrieval, planning, or tool-use strategies that may reach distinct high-fitness regions not visible to a local gradient.

For each scale $k$, the score-function ES estimator is

$$\widehat{g}_{\mu,\mathrm{ES},k} = \frac{1}{S_k} \sum_{s=1}^{S_k} F(\mu+\sigma_k\epsilon_{s}) \frac{\epsilon_{s}}{\sigma_k}, \qquad \{\epsilon_{s}\}^{S_k}_{s=1}\sim q(\epsilon).$$
