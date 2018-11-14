---
layout:     post
title:      "Born Again Neural Networks"
date:       2018-05-27 23:00:00
permalink:  2018/05/27/bann/
---

I recently read [Born Again Neural Networks][1] (to appear at ICML 2018) and
enjoyed the paper. Why? First, the title is cool. Second, it's related to the
broader topics of knowledge distillation and machine teaching that I have been
gravitating to lately. The purpose of this blog post is to go over some of the
math in Section 3 and discuss its implications, though I'll assume the reader
has a general idea of the BAN algorithm. As a warning, notation is going to be a
bit tricky/cumbersome but I will generally match with what the paper uses and
supplement it with my preferred notation for clarity.

We have $$\mathbf{z}$$ and $$\mathbf{t}$$ representing *vectors*  corresponding
to the *student* and *teacher* logits, respectively. I'll try to stick to the
convention of boldface meaning vectors, even if they have subscripts to them,
which instead of components means that they are part of a *sequence* of such
vectors. Hence, we have:

$$\mathbf{z} = \langle z_1, \ldots, z_n \rangle \in \mathbb{R}^n$$

or we can also write $$\mathbf{z} = \mathbf{z}_k$$ if we're considering a
minibatch $$\{\mathbf{z}_1, \ldots, \mathbf{z}_b\}$$ of these vectors.

Let $$\mathbf{x}$$ denote input samples (also vectors) and let $$Z=\sum_{k=1}^n
e^{z_k}$$ and $$T=\sum_{k=1}^n e^{t_k}$$ to simplify the subsequent notation,
and consider the *cross entropy loss function*

$$
\mathcal{L}(\mathbf{x}_1, \mathbf{t}_1)= 
-\sum_{k=1}^{n} \left(\frac{e^{t_k}}{T} \log \frac{e^{z_k}}{Z} \right)
$$

which here corresponds to a *single-sample* cross entropy between the student
logits and the teacher's logits, assuming we've applied the usual softmax (with
temperature one) to turn these into probability distributions. The teacher's
probability distribution could be a one-hot vector if we consider the "usual"
classification problem, but the argument made in many knowledge distillation
papers is that if we consider targets that are not one-hot, the student obtains
richer information and achieves lower test error.

The derivative of the cross entropy with respect to a single output $$z_i$$ is
often applied as an exercise in neural network courses, and is good practice:

$$
\begin{align*}
\frac{\partial \mathcal{L}(\mathbf{x}_1, \mathbf{t}_1)}{\partial z_i} &= -\sum_{k=1}^{n} \frac{\partial}{\partial z_i} \left(\frac{e^{t_k}}{T} \log \frac{e^{z_k}}{Z} \right) \\
&= -\frac{\partial}{\partial z_i} \left(\frac{e^{t_i}}{T} \log \frac{e^{z_i}}{Z} \right)  -\sum_{k=1, k\ne i}^{n} \frac{\partial}{\partial z_i} \left(\frac{e^{t_k}}{T} \log \frac{e^{z_k}}{Z} \right) \\
&= -\frac{e^t_i}{T}\frac{Z}{e^{z_i}} \left\{ \frac{\partial}{\partial z_i} \frac{e^{z_i}}{T} \right\} -\sum_{k=1, k\ne i}^{n} \frac{e^{t_k}}{T} \frac{Z}{e^{z_k}} \left\{ \frac{\partial}{\partial z_i} \frac{e^{z_k}}{Z} \right\} \\
&= -\frac{e^{t_i}}{T}\left(1 - \frac{e^{z_i}}{Z}\right) + \sum_{k=1, k\ne i}^{n} \frac{e^{t_k}}{T} \frac{e^{z_k}}{Z} \\
&= \frac{e^{z_i}}{Z} \sum_{k=1}^n\frac{e^{t_k}}{T} - \frac{e^{t_i}}{T} \\
&= \frac{e^{z_i}}{Z} - \frac{e^{t_i}}{T}
\end{align*}
$$

or $$q_i - p_i$$ in the paper's notation. (As a side note, I don't understand
why the paper uses $$\mathcal{L}_i$$ with a subscript $$i$$ when the loss is the
same for all components?) We have $$i \in \{1, 2, \ldots, n\}$$, and following
the paper's notation, let $$*$$ represent the true label. Without loss of
generality, though, we assume that $$n$$ is always the appropriate label (just
re-shuffle the labels as necessary) and now consider the more complete case of a
minibatch with $$b$$ elements and considering all the possible logits.  We have:

$$
\mathcal{L}(\mathbf{x}_1, \mathbf{t}_1, \ldots, \mathbf{x}_b, \mathbf{t}_b) = 
\frac{1}{b}\sum_{s=1}^b \mathcal{L}(\mathbf{x}_s, \mathbf{t}_s)
$$ 

and so the derivative we use is:

$$
\frac{1}{b}\sum_{s=1}^b \sum_{i=1}^n \frac{\partial \mathcal{L}(\mathbf{x}_s,\mathbf{t}_s)}{\partial z_{i,s}} = 
\frac{1}{b}\sum_{s=1}^b (q_{*,s} - p_{*,s}) +\frac{1}{b} \sum_{s=1}^b \sum_{i=1}^{n-1} (q_{i,s} - p_{i,s})
$$

Just to be clear, we sum up across the minibatch and scale by $$1/b$$, which is
often done in practice so that gradient updates are independent of minibatch
size. We also sum across the logits, which might seem odd but remember that the
$$z_{i,s}$$ terms are *not* neural network *parameters* (in which case we
wouldn't be summing them up) but are the outputs of the network. In
backpropagation, computing the gradients with respect to weights requires
[computing derivatives with respect to network nodes][3], of which the $$z$$s
(usually) form the final-layer of nodes, and the sum here arises from an
application of the chain rule.

Indeed, as the paper claims, if we have the ground-truth label $$y_{*,s} = 1$$
then the first term is:

$$\frac{1}{b}\sum_{s=1}^b (q_{*,s} - p_{*,s}y_{*,s})$$

and thus the output of the teacher, $$p_{*,s}$$ is a *weighting factor* on the
original ground-truth label. If we were doing the normal one-hot target, then
the above is the gradient assuming $$p_{*,s}=1$$, and it gets closer and closer
to it the more confident the teacher gets. Again, all of this seems reasonable.

The paper also argues that this is related to importance weighting of the
samples:

$$\frac{1}{b}\sum_{s=1}^b \frac{p_{*,s}}{\sum_{u=1}^b p_{*,u}} (q_{*,s} - y_{*,s})$$

So the question is, does knowledge distillation (called "dark knowledge") from
[(Hinton et al., 2014)][2] work because it is performing a version of importance
weighting? And by "a version of" I assume the paper refers to this because it
seems like the $$q_{*,s}$$ is included in importance weighting, but not in their
*interpretation* of the gradient.

Of course, it could also work due to to the information here:

$$
\frac{1}{b} \sum_{s=1}^b \sum_{i=1}^{n-1} (q_{i,s} - p_{i,s})
$$

which is in the "wrong" labels. This is the claim made by (Hinton et al., 2014),
though it was not backed up by much evidence. It would be interesting to see the
relative contribution of these two gradients in these refined, more
sophisticated experiments with ResNets and DenseNets. How do we do that? The
authors apply two evaluation metrics:

- **Confidence Weighted by Teacher Max (CWTM)**: One which "formally" applies
  importance weighting with the argmax of the teacher.
- **Dark Knowledge with Permuted Predictions (DKPP)**: One which permutes the
  non-argmax labels.

These techniques apply the argmax of the teacher, not the ground-truth label as
discussed earlier. Otherwise, we might as well not be doing machine teaching.

It appears that if CWTM performs very well, one can conclude most of the gains
are from the importance weighting scheme. If not, then it is the information in
the non-argmax labels that is critical. A similar thing applies to DKPP, because
if it performs well, then it can't be due to the non-argmax labels. I was hoping
to see a setup which could remove the importance weighting scheme, but I think
that's too embedded into the real/original training objective to disentangle.

The experiments systematically test a variety of setups (identical teacher and
student architectures, ResNet teacher to DenseNet student, applying CWTM and
DKPP, etc.). They claim improvements across different setups, validating their
hypothesis.

Since I don't have experience programming or using ResNets or DenseNets, it's
hard for me to fully internalize these results. Incidentally, all the values
reported in the various tables appear to have been run with *one random seed*
... which is *extremely* disconcerting to me. I think it would be advantageous
to pick fewer of these experiment setups and run 50 seeds to see the level of
significance. It would also make the results seem less like a laundry list.

It's also disappointing to see the vast majority of the work here on CIFAR-100,
which isn't ImageNet-caliber. There's a brief report on language modeling, but
there needs to be far more.

Most of my criticisms are a matter of doing more training runs, which hopefully
should be less problematic given more time and better computing power (the
authors are affiliated with Amazon, after all...), so hopefully we will have
stronger generalization claims in future work.

**Update 05/29/2018**: After reading the [Policy Distillation paper][4], it
looks like that paper *already* showed that matching a tempered softmax (of
Q-values) from the teacher using the *same* architecture resulted in better
performance in a deep reinforcement learning task. Given that reinforcement
learning on Atari is arguably a harder problem than supervised learning of
CIFAR-100 images, I'm honestly surprised that the Born Again Neural Networks
paper got away without mentioning the Policy Distillation comparison in more
detail, even when considering that the Q-values do not form a probability
distribution.

[1]:https://arxiv.org/abs/1805.04770
[2]:https://arxiv.org/abs/1503.02531
[3]:http://www.offconvex.org/2016/12/20/backprop/
[4]:https://arxiv.org/abs/1511.06295?context=cs
