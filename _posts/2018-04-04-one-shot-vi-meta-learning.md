---
layout:     post
title:      "One-Shot Visual Imitation Learning via Meta-Learning"
date:       2018-04-04 23:00:00
permalink:  2018/04/04/one-shot-vi-meta-learning/
---

A follow-up paper to the one [I discussed in my previous post][1] is [One-Shot
Visual Imitation Learning via Meta-Learning][2]. The idea is, again, to train
neural network parameters $$\theta$$ on a distribution of tasks such that the
parameters are easy to fine-tune to new tasks sampled from the distribution. In
this paper, the focus is on imitation learning from raw pixels and showing the
effectiveness of a one-shot imitator on a physical PR2 robot.

Recall that the original MAML paper showed the algorithm applied to supervised
regression (for sinusoids), supervised classification (for images), and
reinforcement learning (for MuJoCo). This paper shows how to use MAML for
imitation learning, and the extension is straightforward. First, each imitation
task $$\mathcal{T}_i \sim p(\mathcal{T})$$ contains the following information:

- A trajectory $$\tau = \{o_1,a_1,\ldots,o_T,a_T\} \sim \pi_i^*$$ consists of a
  sequence of states and actions from an *expert policy* $$\pi_i^*$$. Remember,
  this is imitation learning, so we can assume an expert. Also, note that the
  expert policy is *task-specific*.

- A loss function $$\mathcal{L}(a_{1:T},\hat{a}_{1:T}) \to \mathbb{R}$$
  providing feedback on how closely our actions match those of the expert's.

Since the focus of the paper is on "one-shot" learning, we assume we only have
one trajectory available for the "inner" gradient update portion of
meta-training for each task $$\mathcal{T}_i$$. However, if you recall from MAML,
we actually need at least one more trajectory for the "outer" gradient portion
of meta-training, as we need to compute a "validation error" for each sampled
task. This is *not* the overall meta-test time evaluation, which relies on an
entirely new *task* sampled from the distribution (and which only needs one
trajectory, not two or more). Yes, the terminology can be confusing. When I
refer to "test time evaluation" I always refer to when we have trained
$$\theta$$ and we are doing few-shot (or one-shot) learning on a new task that
was not seen during training.

All the tasks in this paper use continuous control, so the loss function for
optimizing our neural network policy $$f_\theta$$ can be described as:

$$
\mathcal{L}_{\mathcal{T}_i}(f_\theta) = \sum_{\tau^{(j)} \sim p(\mathcal{T}_i)}
\sum_{t=1}^T \| f_\theta(o_t^{(j)}) - a_t^{(j)} \|_2^2
$$

where the first sum normally has one trajectory only, hence the "one-shot
learning" terminology, but we can easily extend it to several sampled
trajectories if our task distribution is very challenging. The overall objective
is now:

$$
{\rm minimize}_\theta \sum_{\mathcal{T}_i\sim p(\mathcal{T})} \mathcal{L}_{\mathcal{T}_i} 
(f_{\theta_i'}) = \sum_{\mathcal{T}_i\sim p(\mathcal{T})}
\mathcal{L}_{\mathcal{T}_i} \Big(f_{\theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(f_\theta)}\Big)
$$

and one can simply run Adam to update $$\theta$$.

This paper uses two new techniques for better performance: a two-headed
architecture, and a bias transformation.

- **Two-Headed Architecture**. Let $$y_t^{(j)}$$ be the vector of
  post-activation values just before the last fully connected layer which maps
  to motor torques. The last layer has parameters $$W$$ and $$b$$, so the inner
  loss function $$\mathcal{L}_{\mathcal{T}_i}(f_\theta)$$ can be re-written as:

  $$
  \mathcal{L}_{\mathcal{T}_i}(f_\theta) = \sum_{\tau^{(j)} \sim p(\mathcal{T}_i)}
  \sum_{t=1}^T \| Wy_t^{(j)} + b- a_t^{(j)} \|_2^2
  $$

  where, I suppose, we should write $$\phi = (\theta, W, b)$$ and re-define
  $$\theta$$ to be all the parameters used to compute $$y_t^{(j)}$$.

  In this paper, the test-time single demonstration of the new task is normally
  provided as a sequence of observations (images) and actions. However, they
  also experiment with the more challenging case of removing the provided
  actions for that single test-time demonstration. They simply remove the
  action and use this inner loss function:

  $$
  \mathcal{L}_{\mathcal{T}_i}(f_\theta) = \sum_{\tau^{(j)} \sim p(\mathcal{T}_i)}
  \sum_{t=1}^T \| Wy_t^{(j)} + b\|_2^2
  $$

  This is still a bit confusing to me. I'm not sure why this loss function leads
  to the desired outcome. It's also a bit unclear how the two-headed
  architecture training works. After another read, maybe only the $$W$$ and
  $$b$$ are updated in the inner portion?

  The two-headed architecture seems to be beneficial on the simulated pushing
  task, with performance improving by about 5-6 percentage points. That may not
  sound like a lot, but this was in simulation and they were able to test with
  444 total trials.

  The other confusing part is that if we assume we're allowed to have access to
  expert actions, then the real-world experiment actually used the single-headed
  architecture, and not the two-headed one. So there wasn't a benefit to the
  two-headed one *assuming* we have actions. Without actions, of course, the
  two-headed one is our only option.

- **Bias Transformation**. After a certain neural network layer (which in this
  paper is after the 2D spatial softmax applied after the convolutions to
  process the images), they concatenate this vector of parameters. They claim
  that

  > [...] the bias transformation increases the representational power of the
  > gradient, without affecting the representation power of the network itself.
  > In our experiments, we found this simple addition to the network made
  > gradient-based meta-learning significantly more stable and effective.

  However, the paper doesn't seem to show too much benefit to using the bias
  transformation.  A comparison is reported in the simulated reaching task, with
  a dimension of 10, but it could be argued that performance is similar without
  the bias transformation. For the two other experimental domains, I don't think
  they reported with and without the bias transformation.

  Furthermore, neural networks already have biases. So is there some particular
  advantage to having more biases packed in one layer, and furthermore, with
  that layer being the same spot where the robot configuration is concatenated
  with the processed image ([like what people do with self-supervision][3])? I
  wish I understood. The math that they use to justify the gradient
  representation claim makes sense; I'm just missing a tiny step to figure out
  its practical significance.

They ran their setups on three experimental domains: simulated reaching,
simulated pushing, and (drum roll please) real robotic tasks. For these domains,
they seem to have tested up to 5.5K demonstrations for reaching and 8.5K for
pushing. For the real robot, they used 1.3K demonstrations (ouch, I wonder how
long that took!). The results certainly seem impressive, and I agree that this
paper is a step towards generalist robots.

[1]:https://danieltakeshi.github.io/2018/04/01/maml/
[2]:https://sites.google.com/view/one-shot-imitation
[3]:https://danieltakeshi.github.io/2018/03/30/self-supervision-part-2/
