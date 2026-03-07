---
title: "A Sample Essay: Demonstrating All Features"
date: "Mar 6, 2026"
category: "essay"
---

This is a sample essay to demonstrate the essays section. It exercises every feature supported by the pandoc template, including the three-column layout with table of contents and sidenotes.

## Mathematics

Inline math works naturally: the loss function $\mathcal{L}(\theta)$ is minimized via gradient descent with learning rate $\alpha$. <span class="sidenote">Gradient descent is the workhorse of modern optimization. See <a href="https://arxiv.org/abs/1609.04747">Ruder (2016)</a> for a comprehensive overview of variants.</span>

Display math renders centered:

$$\nabla_\theta \mathcal{L} = \mathbb{E}_{x \sim p(x)} \left[ \nabla_\theta \log p_\theta(x) \right]$$

The evidence lower bound (ELBO) is: <span class="sidenote">The ELBO was introduced by <a href="https://doi.org/10.1023/A:1007665907178">Jordan et al. (1999)</a> and forms the basis of variational inference.</span>

$$\text{ELBO} = \mathbb{E}_{q(z|x)} \left[ \log p(x|z) \right] - D_{\text{KL}}\left( q(z|x) \| p(z) \right)$$

## Code

Here is a Python function with syntax highlighting:

```python
def kl_divergence(mu, log_var):
    """Compute KL divergence for a Gaussian posterior."""
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
```

And inline code like `torch.optim.Adam(lr=1e-3)` renders with a subtle background. <span class="sidenote">Adam combines momentum and adaptive learning rates. See <a href="https://arxiv.org/abs/1412.6980">Kingma & Ba (2015)</a> for the original paper.</span>

## Blockquote

> The purpose of computing is insight, not numbers.
>
> — Richard Hamming

## Table

| Method     | FID ↓  | IS ↑  | Params |
|------------|--------|-------|--------|
| VAE        | 65.2   | 7.1   | 12M    |
| GAN        | 12.8   | 9.2   | 28M    |
| Diffusion  | 4.6    | 11.3  | 55M    |

## Image

![A placeholder diagram](https://via.placeholder.com/600x300?text=Diagram+Placeholder)

*Figure 1: Click to zoom (medium-zoom enabled).*

## Lists

Ordered:

1. Define the generative model
2. Derive the variational objective
3. Train with stochastic gradient descent

Unordered:

- Latent variable models
- Energy-based models
- Autoregressive models

## Conclusion

This sample demonstrates math, code, blockquotes, tables, images, and lists — everything needed for a technical essay. The table of contents on the left tracks your position, and sidenotes appear in the right margin.
