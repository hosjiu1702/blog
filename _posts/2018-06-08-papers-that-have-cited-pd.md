---
layout:     post
title:      "Papers That Have Cited Policy Distillation"
date:       2018-06-08 23:00:00
permalink:  2018/06/08/papers-that-have-cited-pd/
---

About a week and a half ago, I carefully read the [Policy Distillation paper][1]
from DeepMind. The algorithm is easy to understand yet surprisingly effective.
The basic idea is to have *student* and *teacher* agents (typically
parameterized as neural networks) acting on an environment, such as the Atari
2600 games. The teacher is already skilled at the game, but the student isn't,
and need to learn somehow. Rather than run standard deep reinforcement learning,
DeepMind showed that simply running supervised learning where the student trains
its network to *match a (tempered) softmax of the Q-values of the teacher* is
sufficient to learn how to play an Atari 2600 game. It's surprising that this
works; for one, Q-values are not even a probability distribution, so it's not
straightforward to conclude that a student trained to match the softmaxes would
be able to learn a sequential decision-making task.

It was published in ICLR 2016, and one of the papers that cited this was *Born
Again Neural Networks* (to appear in ICML 2018), a paper which [I blogged about
recently][2]. The algorithms in these two papers are similar, and they apply in
the reinforcement learning (PD) and supervised learning (BANN) domains.

After reading both papers, I developed the urge to understand *all* the Policy
Distillation follow-up work. Thus, I turned to Google Scholar, one of the
greatest research conveniences of modern times; as of this writing, the Policy
Distillation paper has 68 citations.  (Google Scholar sometimes has a delay in
registering certain citations, and it also lists PhD theses and textbooks, so
the previous sentence isn't entirely accurate, but it's close enough.)

I resolved to understand the main idea of *every* paper that cited Policy
Distillation, especially with how relevant the paper is to the algorithm. I
wanted to understand if papers directly extended the algorithm, or if they
simply cited it as related work to try and boost up the citation count for
DeepMind. 

I have never done this before to a paper with more than 15 Google Scholar
citations, so this was new to me. After spending a week and a half on this, I
think I managed to get the gist of Policy Distillation's "follow-up space." You
can see my notes in [this shareable PDF which I've hosted on Dropbox][3].  Feel
free [to send me recommendations][4] about other papers I should read!

[1]:https://arxiv.org/abs/1511.06295
[2]:https://danieltakeshi.github.io/2018/05/27/bann/
[3]:https://www.dropbox.com/s/fnn1a9t25py7co1/Policy_Distillation_Follow_Ups%20%281%29.pdf?dl=0
[4]:https://danieltakeshi.github.io/about.html
