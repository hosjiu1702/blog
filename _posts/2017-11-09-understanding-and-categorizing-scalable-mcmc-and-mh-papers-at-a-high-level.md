---
layout:    post
title:     "Understanding and Categorizing Scalable MCMC and MH Papers at a High Level"
date:      2017-11-09 10:00:00
permalink: 2017/11/09/understanding-and-categorizing-scalable-mcmc-and-mh-papers-at-a-high-level/
---

When reading academic papers about a certain subfield, I often find it difficult
to clearly understand how they *connect* with each other. For example, what
algorithms are based on other algorithms? Can the contributions of two papers be
combined? Would combining them result in notable improvements or just
on-the-margin, negligible changes? (The answer to that last question is usually
"the latter" but it's something we should at least *consider*.)

This post is an attempt to unify my understanding of papers related to scalable
Markov Chain Monte Carlo and scalable Metropolis-Hastings. By "scalable," I
refer to the usual meaning of using these algorithms in the large data regime.

These are the papers I'm trying to understand:

- *MCMC Using Hamiltonian Dynamics*, Handbook of MCMC 2010
- *Bayesian Learning via Stochastic Gradient Langevin Dynamics*, ICML 2011
- *Stochastic Gradient Hamiltonian Monte Carlo*, ICML 2014
- *Austerity in MCMC Land: Cutting the Metropolis-Hastings Budget*, ICML 2014
- *Towards Scaling up Markov Chain Monte Carlo: An Adaptive Subsampling Approach*, ICML 2014
- *Firefly Monte Carlo: Exact MCMC with Subsets of Data*, UAI 2014
- *On Markov Chain Monte Carlo Methods For Tall Data*, JMLR 2017.

(All of them are freely available online.)

First, I'll briefly discuss why we care about the problem of scalability with
MCMC and MH. Then, I'll group these papers into categories and explain how they
are connected to each other. This will then motivate [our UAI 2017 paper][3],
*An Efficient Minibatch Acceptance Test for Metropolis-Hastings*.


# Why Markov Chain Monte Carlo?

I'm not going to review MCMC here as you can find many other references, both
online and in textbooks. It may help to look at [my blog post from June 2016][2]
where I describe the general problem setting. [My more recent BAIR Blog post][1]
also contains some potentially useful background material.

But why use MCMC at all? Here's one reason: if we use it to sample some model's
parameter $$\theta$$, then the chain of samples $$\{\theta_1, \ldots,
\theta_T\}$$ should let us quantify useful statistics about properties of
interest. Two of these are the expectation and variance of something, which we
might apply on the parameter itself. We can estimate (for example) the
expectation by taking a sequence of the $$K$$ most recent samples (or a
subsampled sequence) from our chain $$\{\theta_{T-K+1}, \ldots, \theta_T\}$$ and
then taking the sample vector-valued expectation. More generally, letting $$f$$
be a function of the parameters, we can compute the expectation
$$\mathbb{E}[f(\theta)]$$ using the expectation of the sampled values
$$\{f(\theta_{T-K+1}),\ldots, f(\theta_T)\}$$.

We can't do this if we take stochastic gradient steps, because the samples from
SGD *are not from the posterior distribution* of the parameter.  SGD is designed
to converge around a *single point* in the space of possible $$\theta \in
\Theta$$ values, unlike MCMC methods which are supposed to approximate a
*distribution*, which can then be used for sample estimates of expectations and
variances.

My perspective is supported in papers such as the SGLD paper from 2011 (one of
the papers I listed above); the authors (Welling & Teh) claim that:

> Bayesian methods are appealing in their ability to capture uncertainty in
> learned parameters and avoid overfitting. Arguably with large datasets there
> will be little overfitting. Alternatively, as we have access to larger
> datasets and more computational resources, we become interested in building
> more complex models, so that there will always be a need to quantify the
> amount of parameter uncertainty.

So ... that's why we like the Bayesian perspective. These authors are
rock-stars, by the way, so I generally trust their conclusions.

I'll be honest, though: I can't think of something nontrivial I've done in which
the Bayesian perspective was *that* useful to me. In Deep Learning, Deep
Imitation Learning, and Deep Reinforcement Learning, I've never used priors and
posteriors; RMSProp or Adam is good enough, and it seems like this goes for the
rest of the community. Maybe it's just not that necessary in these domains?  I
have two papers on my reading list, [Bootstrapped DQNs][5] and [Robust Bayesian
Neural Networks][4], which might clarify some of my questions regarding how much
of a Bayesian perspective is needed in Deep Learning. I should also definitely
check out the [Bayesian Deep Learning NIPS workshop][6].


# Langevin Dynamics and Hamiltonian Dynamics

This section concerns the following three papers:

- MCMC Using Hamiltonian Dynamics, Handbook of MCMC 2010
- Bayesian Learning via Stochastic Gradient Langevin Dynamics, ICML 2011
- Stochastic Gradient Hamiltonian Monte Carlo, ICML 2014

I gave a brief introduction to Langevin Dynamics [in my earlier blog post][2],
so just to summarize for this one, Langevin Dynamics injects an appropriate
amount of noise so that (in our context) a gradient-based algorithm will
converge to a *distribution* over the posterior of $$\theta$$. The *Stochastic
Gradient* Langevin Dynamics algorithm combines the computational efficiencies of
SGD by using a minibatch gradient, but uses the Langevin noise to appropriately
cover the posterior:

> [...] Langevin dynamics which injects noise into the parameter updates in such
> a way that the trajectory of the parameters will converge to the full
> posterior distribution rather than just the maximum a posteriori mode.

As a follow-up, the Stochastic Gradient *Hamiltonian Monte Carlo* (SGHMC)
algorithm is similar to SGLD in that it uses a minibatch gradient along with
"exploration noise." This time, the noise is from Hamiltonian Monte Carlo, which
is more sophisticated than Langevin Dynamics since HMC introduces extra momentum
variables which allows for larger jumps. 

Radford Neal's excellent 2010 book chapter goes over HMC in great detail, so I
won't go through the details here (though I'd like to write a blog post solely
about HMC --- so stay tuned!). Just to give a quick overview, though, our
problem context is similar, where we have a target posterior:

$$
p(\theta \mid x_1, \ldots, x_N) \propto \exp(-U(\theta))
$$

with *potential energy* function

$$
U(\theta) = -\log p(\theta) - \sum_{i=1}^{N} \log p(x_i \mid \theta).
$$

(Don't worry too much about the "potential energy" terminology; HMC was
originally developed from a physics background. We're still in the same problem
setting.)

HMC generates samples from a *joint distribution* that involves extra *momentum*
variables:

$$
\pi (\theta, r) \propto \exp\left(-U(\theta) - \frac{1}{2}r^TMr\right)
$$

where $$r$$ are the momentum variables and $$M$$ is a mass matrix. The update
rules are:

- $$\theta = \theta + \tau \cdot M^{-1}r$$.
- $$r = r - \tau \cdot \nabla U(\theta)$$.

where $$\tau$$ is some step size. If this doesn't make sense, read Neal's 2010
book chapter.

The result from HMC is a set of samples $$\{(\theta_i,r_i)\}_{i=1}^T$$. But
we're only interested in the $$\theta_i$$s, so ... we simply drop the $$r_i$$
terms to get our samples for $$\theta$$. Amazingly, $$\theta$$ is sampled from
the correct target distribution, which one can show via some "reversibility"
analysis.

SGHMC needs a little massaging to actually get it to sample the target
distribution, since simply taking a subset of the data to compute an
approximation to $$\nabla U(\theta)$$ will lose the "Hamiltonian Dynamics"
property; the authors resolve this by using second-order Langevin Dynamics to
counteract the effect of too much gradient noise in estimating $$\nabla
U(\theta)$$, and the result is a similar algorithm to SGLD except with a
different noise term.

Just to be clear, both SGLD and SGHMC are minibatch, gradient-based algorithms
that are also considered "Bayesian methods." Neither are pure random walks,
i.e., neither use Gaussian *proposals* because the proposals are based on the
*stochastic gradient* value, *plus* some additive noise term. For SGLD, that
extra noise is actually a random walk, but not for SGHMC.

For both SGLD and SGHMC, we have to apply the Metropolis-Hastings test for
computer implementations due to discretization error, even though *in theory* we
shouldn't have to since energy is preserved. In both papers, the authors
decrease step sizes to zero so that the MH rejection rate goes to zero.
Intuitively, smaller step sizes mean samples are concentrated into higher
regions of the posterior, and the gradient ensures going in the direction of
greatest increase of the posterior probability. In addition, decreasing step
sizes also means discretization error decreases, which yet again further reduces
the need for MH tests. While this is great, because the MH test requires
full-batch computation, perhaps we are missing out somehow by keeping our step
sizes small.[^notsure]


# Metropolis-Hastings

In this section, I discuss the remaining papers listed at the introduction of
this post. They are related in some form to the Metropolis-Hastings algorithm,
which is commonly used in MCMC techniques to act as a correction to ensure that
samples do not deviate too frequently away from the target posterior
distribution.

As I mentioned in both of my [earlier][1] [blog][2] posts, conventional MH tests
require a full pass over the entire dataset. This makes them extremely costly,
and is one of the reasons why both SGLD and SGHMC emphasized how decreasing step
sizes results in lower discretization error, so that they could omit the MH
tests.

Their computational cost has raised the question over whether using subsamples
of the data for the MH test computation is feasible. It's not as straightforward
as taking a fixed-sized subset (i.e., minibatch) of the dataset because that
results in a non-trivial target distribution which is not the desired
posterior.

The following two papers propose subsampling-based algorithms that attempt to
tackle the high cost of full-batch MH tests:

- Austerity in MCMC Land: Cutting the Metropolis-Hastings Budget, ICML 2014
- Towards Scaling up Markov Chain Monte Carlo: An Adaptive Subsampling Approach,
  ICML 2014

I discussed the first one in [an earlier blog post][2]. The second one follows a
similar procedure as the first one, except that it uses a slightly different way
of interpreting when to stop data collection. The downside, as I painfully
realized when I tried to implement this, is that due to its concentration
bounds, it requires a real-valued parameter which depends on the *entire*
collection of $$\{\log p(x_i\mid \theta), \log p(x_i\mid \theta')\}_{i=1}^N$$
values *each iteration*, which defeats the point of using a subset of the data.
(Here, I use $$\theta'$$ to denote the proposed distribution.)

The authors of the Adaptive Subsampling paper have a follow-up JMLR 2017 paper
(it was under review for a *long* time) which expands upon this discussion. I
found it quite useful, particularly because of their proof (in Section 6.1)
about how naive subsampling for the MH test results in a nontrivial and
hard-to-interpret target distribution. In Section 6.3, they introduce a novel
contribution where they rely on *subsampling noise for exploration*; that is,
use the minibatch-induced noise (which is Gaussian by the Central Limit Theorem)
to explore the posterior. However, they showed that this approach still seems to
require $$O(n)$$ data points each iteration. On the other hand, they didn't
investigate this method in too much detail, so it's hard to comment on its
usefulness.

The last related work was the Firefly paper, which won the Best Paper Award at
UAI 2014. It can perform exact MCMC, but the main drawback is (emphasis mine):

> FlyMC is compatible with a wide variety of modern MCMC algorithms, and **only
> requires a lower bound on the per-datum likelihood factors**.

To be clear on what this means, they require the existence of functions
$$B_i(\theta)$$ satisfying $$0 \le B_i(\theta) \le p(x_i \mid \theta)$$ for
*all* $$i$$. How realistic is that? I have no idea, honestly, but it seems like
something that is difficult to achieve in practice, especially because it's
conditioning on $$\theta$$ and $$\theta'$$, which will vary considerably
throughout sampling. There is some interesting discussion about this [at
Christian Roberts' excellent blog][7], with Ryan Adams (the professor co-author)
commenting.

The prior work then motivated our work, where we avoided needing these
assumptions and showed that we could cut the M-H test cost time down to one
equivalent with SGD, without loss of performance. There's no free lunch,
though; our algorithm has applicability constraints but those are hopefully not
that restrictive. [Check out our BAIR blog post][1] for more information.


# Conclusion

I've discussed these set of papers and tried grouping them together to see a
coherent theme in all of this. Hopefully this makes it clearer what these papers
are trying to do.

***

[^notsure]: I'm actually not sure if we can even use the Metropolis-*Hastings*
    test (and not just the "Metropolis Algorithm") with SGHMC. The
    authors of the SGHMC paper claim that MH tests are impossible for both SGLD
    and SGHMC since the reverse proposal probability $$q(\theta \mid \theta')$$
    cannot be computed. It seems to me, however, that one can compute the SGLD
    reverse probability because that's a Gaussian centered at the gradient term
    with some known variance. What am I missing here? At the very least,
    applying the MH test to regular HMC should be OK, since we can omit the
    proposal probabilities. And that's what both the SGHMC authors (judging
    from [Tianqi Chen's source code][8]) and Radford Neal do in their experiments.

[1]:http://bair.berkeley.edu/blog/2017/08/02/minibatch-metropolis-hastings/
[2]:https://danieltakeshi.github.io/2016-06-19-some-recent-results-on-minibatch-markov-chain-monte-carlo-methods/
[3]:https://arxiv.org/abs/1610.06848
[4]:https://papers.nips.cc/paper/6117-bayesian-optimization-with-robust-bayesian-neural-networks
[5]:https://papers.nips.cc/paper/6501-deep-exploration-via-bootstrapped-dqn
[6]:http://bayesiandeeplearning.org/
[7]:https://xianblog.wordpress.com/2014/04/02/firefly-monte-carlo/
[8]:https://github.com/tqchen/ML-SGHMC
