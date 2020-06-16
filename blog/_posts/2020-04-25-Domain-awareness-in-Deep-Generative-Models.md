---
layout: post
title: "Domain awareness in Deep Generative Models"
excerpt: "On internal models and weak-supervision."
mathjax: true
---

### Introduction

Learning generative models for data defined over structures is a fundamental challenge in Machine Learning. Many problems of interest in Physics and Engineering have discrete components, relations defined by graphs and complex geometries. However, the dominant learning paradigm tends to be agnostic of the underlying data-domain, frequently assuming that the data lives in some *continuous dense vacuum*.

I propose to learn domain-aware representations that can handle underlying structure with *weak supervision*. In the weak-supervised learning paradigm, we do not make strong assumptions regarding the input, the task or the target. Instead, we condition on information common to a larger class of problems, reducing the hypothesis space and biasing the model toward compatible solutions.



