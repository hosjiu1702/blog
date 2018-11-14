---
layout:     post
title:      Max Welling's "Intelligence Per Kilowatt-Hour"
date:       2018-07-23 23:00:00
permalink:  2018/07/23/max-welling/
---

I recently took the time to watch Max Welling's excellent and thought-provoking
ICML keynote. You can view [part 1][1] and [part 2][2] on YouTube. The video
quality is low, but at least the text on the slides is readable. I don't think
slides are accessible anywhere; I don't see them on [his Amsterdam website][3].

As you can tell from his biography, Welling comes from a physics background and
spent undergraduate and graduate school in Amsterdam studying under a Nobel
Laureate, and this background is reflected in the talk.

I will get to the main points of the keynote, but the main reason why I for once
managed to watch a keynote talk (rather than partake in the usual *"Oh, I'll
watch it later when I have time ..."* and then forgetting about it[^blog]) is
that I wanted to test out a new pair of hearing aids and the microphone that
came with it. I am testing the [ReSound ENZO 3D hearing aids][6], along with the
accompanying ReSound Multi Mic.

That microphone will use a 3.5mm mini jack cable to connect to an audio source,
such as my laptop. Then, with an app through my iPhone, I can switch my hearing
aid's mode to "Stream," meaning that the sound from my laptop or audio source,
which is connected to the Multi Mic, goes *directly* into my hearing aids. In
other words, it's like a wireless headphone. I have long wanted to test out
something like this, but never had the chance to do so until the appropriate
technology came for the kind of hearing aid power I need.

The one downside, I suppose, of this is that if I were to listen to music while
I work, there wouldn't be any headphones visible (either wired or wireless) as
would be the case with other students.  This means someone looking at me might
try and talk to me, and think I am ignoring him or her if I do not respond due
to hearing only the sound streaming through the microphone. I will need to plan
this out if I end up getting this microphone.

But anyway, back to the keynote. Welling titled the talk as "Intelligence Per
Kilowatt-Hour" and pointed out early that this could also be expressed as the
following equation:

> Free Energy = Energy - Entropy

After some high-level physics comments, such as connecting gravity, entropy, and
the second law of thermodynamics, Welling moved on to discuss more
familiar[^physics] territory to me: Bayes' Rule, which we should all know by
now. In his notation:

$$P(X) = \int d\Theta P(\Theta,X) =  \int d\Theta P(\Theta)P(X|\Theta)$$

$$P(\Theta|X) = \frac{P(X|\Theta)P(\Theta)}{P(X)}$$

Clearly, there's nothing surprising here.

He then brought up Geoffrey Hinton as the last of his heroes in the
introductory parts of the talks, along with the two papers:

- *Keeping the Neural Networks Simple by Minimizing the Description Length of the Weights* (1993)
- *A View of the EM Algorithm that Justifies Incremental, Sparse, and other Variants* (1998)

I am aware of these papers, but I just cannot find time to read them in detail.
Hopefully I will, someday. Hey, if I can watch Welling's keynote and blog about
it, then I can probably find time to read a paper.

Probably an important, relevant bound to know is:

$$
\begin{align}
\log P(X) &= \int d\Theta Q(\Theta) \log P(X|\Theta) - KL[Q(\Theta)\|P(\Theta)] + KL[Q(\Theta)\|P(\Theta|X)] \\
&\ge \int d\Theta Q(\Theta) \log P(X|\Theta) - KL[Q(\Theta)\|P(\Theta)] \\
&= \int d\Theta Q(\Theta) \log P(X|\Theta)P(\Theta) -\int d\Theta Q(\Theta) \log Q(\Theta)
\end{align}
$$

where the equality to lower bound results because we ignore a KL divergence term
which is always non-negative.  The right hand side of the final line can be
re-thought as negative energy plus entropy.

In the context of discussing the above math, Welling talked about *intractable
distributions*, a thorn in the side of many statisticians and machine learning
practitioners. Thus, he discussed two broad classes of techniques to approximate
intractable distributions: MCMC and Variational methods. The good news is that I
understood this because John Canny and I wrote [a blog post about this last year
on the Berkeley AI Research Blog][4][^canny].

Welling began with his seminal work: Stochastic Gradient Langevin Dynamics,
which gives us a way to use minibatches for large-scale MCMC. I won't belabor
the details of this, since [I wrote a blog post (on this blog!) two years
ago][5] about this very concept. Here's the relevant equation and method
reproduced here, for completeness:

$$\theta_{t+1} = \theta_t + \frac{\epsilon_t}{2}\left(\nabla \log p(\theta_t) + \frac{N}{n}
\sum_{i=1}^n \nabla \log p(x_{ti} \mid \theta_t)\right) + \eta_t$$

$$\eta_t \sim \mathcal{N}(0, \epsilon_t)$$

where we need $$\epsilon_t$$ to vary and decrease towards zero, among other
technical requirements. Incidentally, I like how he says: "sample from the true
posterior." This is what I say in my talks.

Afterwards, he discussed some of the great work that he has done in Variational
Bayesian Learning. I'm most aware of him and his student, Durk Kingma,
introducing
Variational Autoencoders for generative modeling. That paper also popularized
what's known as the *reparameterization trick* in statistics. In Welling's
notation,

$$\Theta = f(\Omega, \Phi) \quad {\rm s.t.} \quad Q_{\Phi}(\Theta)d(\Theta) = P_0(\Omega)d\Omega$$

I will discuss this in more detail, I swear. I have a blog post about the math
here but it's languished in my drafts folder for months.

In addition to the general math overview, Welling discussed:

- How the reparameterization trick helps to decrease variance in REINFORCE.  I'm
  not totally sure about the details, but again, I'll have to mention it in my
  perpetually-in-progress draft blog post previously mentioned.
- The *local* reparameterization trick. I see. What's next, the *tiny*
  reparameterization trick?
- That we need to make Deep Learning more efficient. Right now, our path is not
  sustainable. That's a solid argument; Google can't keep putting this much
  energy into AI projects forever. To do this, we can remove parameters or
  quantize them. For the latter, this is like reducing them from float32 to int,
  to cut down on memory usage. At the extreme, we can use *binary* neural
  networks.
- Welling also mentioned that AI will move to the edge. This means moving from
  servers with massive computational power to everyday smart devices with lower
  compute and power. In fact, his example was *smart hearing aids*, which I
  found amusing since, as you know, the main motivation for me watching this
  video was *precisely* to test out a new pair of hearing aids! I don't think
  there is AI in the ReSound ENZO 3D.

The last point above about AI moving to the edge is what motivates the title of
the talk. Since we are compute- and resource-constrained on the edge, it is
necessary to extract the benefits of AI efficiently, hence AI *per kilowatt
hour*.

Towards the end of the talk, Welling brought up more recent work on Bayesian
Deep Learning for model compression, including:

- Probabilistic Binary Networks
- Differentiable Quantization
- Spiking Neural Networks

These look like some impressive bits of research, especially spiking neural
networks because the name sounds cool.  I wish I had time to read these papers
and blog about them, but Welling gave *juuuuuuust* enough information that I
think I can give one or two sentence explanations of the research contribution.

Welling concluded with a few semi-serious comments, such as inquiring about the
question of life (OK, seriously?), and then ... oh yeah, that Qualcomm AI is
hiring (OK, seriously again?).

Well, advertising aside --- which to be fair, lots of professors do in their
talks if they're part of an industrial AI lab --- the talk was thought-provoking
to me because it forced me to consider energy-efficiency if we are going to make
further progress in AI and to also ensure that we can maximally extract AI
utility in compute-limited devices. These are things worth thinking about at a
high level for our current and future AI projects.

***

[^blog]: To be fair, this happens all the time when I try and write long,
    lengthy blog posts, but then realize I will never have the effort to fix
    up the post to make it acceptable for the wider world.

[^canny]: John Canny also comes from a theoretical physics background, so I bet
    he would like Welling's talk.

[^physics]: I am *trying* to self-study physics. Unfortunately, it is proceeding
    at a snail's pace.

[1]:https://www.youtube.com/watch?time_continue=884&v=7QhkvG4MUbk
[2]:https://www.youtube.com/watch?v=vzoTh0ZAUAk
[3]:https://staff.fnwi.uva.nl/m.welling/
[4]:http://bair.berkeley.edu/blog/2017/08/02/minibatch-metropolis-hastings/
[5]:https://danieltakeshi.github.io/2016-06-19-some-recent-results-on-minibatch-markov-chain-monte-carlo-methods/
[6]:https://www.resound.com/en-us/hearing-aids/resound-hearing-aids/enzo-3d
