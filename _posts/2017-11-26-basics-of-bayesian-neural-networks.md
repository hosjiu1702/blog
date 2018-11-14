---
layout:    post
title:     "Basics of Bayesian Neural Networks"
date:      2017-11-26 10:00:00
permalink: /2017/11/26/basics-of-bayesian-neural-networks/
---

In this post, I try and learn as much about Bayesian Neural Networks (BNNs) as I
can. I borrow the perspective of [Radford Neal][1]: BNNs are updated in two
steps.  The first step samples the hyperparameters, which are typically the
regularizer terms set on a per-layer basis. The second step performs Hamiltonian
Monte Carlo over the data, or through a series of minibatches plus sophisticated
"friction" techniques, if using [Stochastic Gradient Hamiltonian Monte
Carlo][2]. These update the actual weights we use for the neural networks; the
hyperparameters are sampled mainly to invoke a "fully Bayesian" hierarchical
model.

The above is different from the paradigm of using Bayesian Neural Networks with
a technique known as *variational inference*. I will not be discussing that.

To make things concrete, in this blog post I will assume we have the following
neural network which:

- is fully connected.
- takes MNIST data as input (784 dimensions for each data point), has a hidden
  layer of 100 units, and outputs a 10-dimensional vector from a softmax.
- uses the sigmoid activation for the hidden layer.
- uses a regularizer hyperparameter for each of the two sets of weight matrices,
  along with two for the biases.

We can write the network's mathematical meaning using $$A \in \mathbb{R}^{10
\times 100}$$ and $$B \in \mathbb{R}^{784 \times 100}$$ as the weight matrices,
along with $$a \in \mathbb{R}^{10}$$ and $$b \in \mathbb{R}^{100}$$ as the bias
vectors.  Using this, the network output can be expressed as:

$$
P(y=i| x) = 
\frac{\exp\Big(A_i^T\sigma(B^Tx+b)+a_i\Big)}{\sum_{j=0}^9 \exp\Big(A_j^T \sigma(B^Tx+b)+a_j\Big)}
$$

where $$y \in \{0,1,\ldots,9\}$$ indicates the class label and $$A_i$$ is the
$$i$$th column of $$A$$. (The entire output would just be the vector of
$$\langle P(y=0| x), \ldots, P(y=9| x)\rangle$$ values.) I write $$(x,y)$$
without subscripts, but in general we should write
$$\mathcal{D} = \{(x^{(k)},y^{(k)})\}_{k=1}^N$$ to represent the entire dataset. 

We need to incorporate Bayesian assumptions somehow, so first we assume that all
these weights have *Gaussian priors* with zero mean and covariance matrix
$$\Sigma = \sigma_A^2 I$$ set to some multiple of the identity. Intuitively, this
seems to be a reasonable prior, as we'd generally like our weights to be small
and roughly symmetrical about zero. For example, with the weights for $$A$$, we 

$$
P(A)  = \frac{\exp\left(-\frac{1}{2}(A-0)^T(\sigma_A^2
I)^{-1}(A-0)\right)}{(2\pi)^{\frac{n}{2}} |\sigma_A^2 I|^{\frac{1}{2}}} \propto \exp\left(-\frac{\lambda_A}{2}\|A\|_2^2\right)
$$

where we set $$\lambda_A$$ to be the inverse variance, also known as the
*precision term*. Similarly, we have $$P(B) \propto
\exp\left(-\frac{\lambda_B}{2}\|B\|_2^2\right)$$, $$P(a) \propto
\exp\left(-\frac{\lambda_a}{2}\|a\|_2^2\right)$$, and $$P(b) \propto
\exp\left(-\frac{\lambda_b}{2}\|b\|_2^2\right)$$. Notice that I am now
flattening the matrices $$A$$ and $$B$$ so that they are *vectors*. This makes
the notation a bit easier, and it means that when I write $$\|A\|_2$$, I mean
the $$L_2$$ norm, *not* the spectral norm on matrices (a.k.a., the largest
singular value). I will use the flattened notation for the remainder of this
blog post.

The precision terms are hyperparameters which we also endow with their own (IID)
priors:[^rate]

$$
\lambda_A, \lambda_B, \lambda_a, \lambda_b \;{\overset{\rm i.i.d.}{\sim}}\;
\Gamma(\alpha,\beta)
$$

Letting $$\Theta$$ denote our eight major parameters (including the
hyperparameters), the posterior distribution *which we want to sample from*
based on dataset $$\mathcal{D}$$ is

$$
P(\Theta | \mathcal{D}) \propto P(\Theta)P(\mathcal{D} | \Theta) = P(\Theta)\prod_{(y,x)\in \mathcal{D}} P(y | x, \Theta)
$$

where we follow the usual assumption of conditional independence among the data;
think of drawing data points from the true "data distribution" to form our
training data.

BNNs use Bayesian methods to figure out a good set of parameters $$\Theta$$ for
some task, which here is based on digit classification accuracy. I will now go
over how the hyperparameter updates work, followed by the parameter updates.



## The Hyperparameter Updates

This step samples the following:

$$
\lambda_A,\lambda_B, \lambda_a, \lambda_b \sim P( \lambda_A,\lambda_B, \lambda_a, \lambda_b | A, B, a, b)
$$

There is no dependence on the dataset $$\mathcal{D}$$ as the hyperparameters are
sampled based on "data" which consists of the parameter values at the lower 
level. Also, since we assumed an IID prior for the precision terms, and because
the values of the parameters are viewed as independent as well (I admit this
probably isn't the best way of describing it but it feels intuitive) we have:

$$
\begin{align}
P( \lambda_A,\lambda_B, \lambda_a, \lambda_b | A, B, a, b) &\propto P(\lambda_A,\lambda_B, \lambda_a, \lambda_b) P(A,B,a,b | \lambda_A,\lambda_B, \lambda_a, \lambda_b) \\
&= P(\lambda_A)P(\lambda_B)P(\lambda_a)P(\lambda_b) P(A| \lambda_A)P(B|\lambda_B)P(a|\lambda_a)P(b|\lambda_b)
\end{align}
$$

We can literally sample the four precision terms sequentially due to their
independence assumptions. For simplicity, let us assume we are sampling only
$$\lambda_A$$ so that the rest of the computation is straightforward.  The math
turns out to be:

$$
\begin{align}
\lambda_A \sim P(\lambda_A) P(A| \lambda_A) &= \frac{\beta^\alpha \lambda_A^{\alpha-1} e^{-\beta \lambda_A}}{\Gamma(\alpha)} \frac{\exp\left(-\frac{\lambda_A}{2}\|A\|_2^2\right)}{((2\pi)^n |(\lambda_A)^{-1} I|)^{1/2}} \\
&\propto \lambda_A^{\left(\alpha + \frac{n}{2}\right)-1} e^{-\lambda_A\left(\beta + \frac{\|A\|_2^2}{2}\right)} \\
&= \Gamma\left(\alpha + \frac{n}{2},  \beta + \frac{\|A\|_2^2}{2} \right)
\end{align}
$$

and indeed, we have conjugacy: sampling the hyperparameters can be done simply
by sampling from a Gamma distribution with these updated parameters based on the
previous values of $$\alpha$$ and $$\beta$$. For intuition on what these
parameters mean, if we have $$X \sim \Gamma(\alpha, \beta)$$, then
$$\mathbb{E}[X] = \alpha / \beta$$.



## The Parameter Updates

This step samples the following:

$$
A, B, a, b \sim P(A,B,a,b | \mathcal{D}, \Lambda)
$$

where $$\Lambda = \{\lambda_A,\lambda_B, \lambda_a, \lambda_b\}$$ to simplify
the notation if we depend on all the hyperparameters. There is dependence on
$$\Lambda$$ here as those values determine the spread of the Gaussian
distributions. Also, notice the dependence on the data here, unlike the previous
case.

Using Bayes' Rule as we did earlier (with that condensed $$\Theta$$ notation) we
get:

$$
\begin{align}
P(A,B,a,b | \mathcal{D}; \Lambda) &\propto P(A,B,a,b ; \Lambda) \prod_{i=1}^nP(y^{(i)} |x^{(i)},A,B,a,b;\Lambda) \\
&= P(A;\lambda_A)P(B;\lambda_B)P(a;\lambda_a)P(b;\lambda_b) \prod_{i=1}^nP(y^{(i)} |x^{(i)},A,B,a,b;\Lambda)
\end{align}
$$

How do we sample from this distribution? We use **Hamiltonian Monte Carlo**
(HMC). 


### The Hamiltonian, Potential Energy, and Kinetic Energy

Briefly: HMC uses what is known as a **Hamiltonian Function**
$$H(\Theta,p)$$ where $$\Theta$$ are the parameters and $$p$$ refers to
*auxiliary momentum variables*.[^notation] In Bayesian statistics, current
practice is to split the Hamiltonian into two functions: $$H(\Theta,p) =
U(\Theta) + K(p)$$ known as the *potential energy* and *kinetic energy*,
respectively. HMC is designed to sample from the distribution defined as
follows:

$$
P(\Theta,p) = \frac{1}{Z}  \exp\left(-\frac{U(\Theta)}{T}\right) \exp\left(-\frac{K(p)}{T}\right)
$$

where $$Z>0$$ is a normalizing constant and $$T>0$$ is some temperature,
typically used to "flatten" or "diffuse" the target distribution (which here is
$$\propto \exp\left(-\frac{H(\Theta,p)}{T}\right)$$) to make optimization easier.
For the rest of this post, I include $$T$$ for clarity but I keep it separate
from $$H, U,$$ and $$K$$.

In Bayesian statistics, the **Potential Energy** is $$U(\Theta) =
-\log(P(\Theta) \cdot P(\mathcal{D} | \Theta))$$ because if we plug that in, we
get

$$
\begin{align}
P(\Theta,p) &= \frac{1}{Z}  \exp\left(-\frac{(-\log(P(\Theta) P(\mathcal{D} | \Theta))}{T}\right) \exp\left(-\frac{K(p)}{T}\right) \\
&\propto \frac{P(\Theta) P(\mathcal{D} | \Theta)}{T}\exp\left(-\frac{K(p)}{T}\right)
\end{align}
$$

which is exactly what we want for the position variables, assuming that the
momentum is independent so that $$P(\Theta,p)=P(\Theta)P(p)$$, which is standard
practice. To be clear: (a) we're only interested in sampling from the
distribution for $$\Theta$$, not the momentum's distribution, so (b) to get our
desired samples of $$\Theta$$ from the posterior, we generate samples
$$(\Theta^{(i)},p^{(i)})$$ that include the momentum variables, and then we drop
the latter after we're done.

Regarding the **Kinetic Energy**, current practice is to set it to be a
quadratic potential with mass matrix $$M = \sigma^2 I$$ a multiple of the
identity:

$$K(p) = \frac{p^TM^{-1}p}{2} = \frac{p^Tp}{2\sigma^2}$$

To be clear, we *need* to sample from the **target distribution** as specified
by $$\exp\left(-\frac{H(\Theta,p)}{T}\right)$$, which means we must technically
sample from the distributions defined as:

$$\exp\left(-\frac{K(p)}{T}\right) \quad \mbox{and} \quad \exp\left(-\frac{U(\Theta)}{T}\right)$$ 

Here, $$U$$ and $$K$$ are **energy** functions, but they are *not* the same as
the actual distribution we are sampling from. For example, with the kinetic
energy, the actual distribution we sample from is proportional to

$$\exp\left(-\frac{K(p)}{T}\right) = \exp \left( -\frac{p^Tp}{2\sigma^2T}\right)$$

i.e., a zero-mean Gaussian with covariance $$T\sigma^2I$$. We sequentially
sample for each because they are independent by assumption.


### Running Hamiltonian Monte Carlo

Running HMC in computer simulation requires a *Metropolis test*[^hastings] each
iteration (i.e., each sample in our MCMC chain) to correct for discretization
error. This requires computing the follow ratio $$r$$:

$$
r = \frac{1}{T} \Big( (U(\Theta)+K(p)) - (U(\Theta')+K(p'))\Big)
$$

where $$\Theta'$$ and $$p'$$ refer to the *proposed* position and momentum
variables, respectively.  Computing the kinetic energy is typically a matter of
adding squared $$L_2$$ norms, so that part is easy. But what about $$U$$? To
compute this difference for our proposed Bayesian Neural Network, we see that

$$
\begin{align}
U(\Theta) &= -\log P(\Theta; \Lambda) - \log P(\mathcal{D} | \Theta; \Lambda) \\
&= -\log \left(\frac{e^{-\frac{\lambda_A}{2}\|A\|_2^2}}{(2\pi)^{\frac{n}{2}}|\frac{1}{\lambda_A}I|^{\frac{1}{2}}}\right) -\log \left(\frac{e^{-\frac{\lambda_B}{2}\|B\|_2^2}}{(2\pi)^{\frac{n}{2}}|\frac{1}{\lambda_B}I|^{\frac{1}{2}}}\right) -\log \left(\frac{e^{-\frac{\lambda_a}{2}\|a\|_2^2}}{(2\pi)^{\frac{n}{2}}|\frac{1}{\lambda_a}I|^{\frac{1}{2}}}\right) -\log \left(\frac{e^{-\frac{\lambda_b}{2}\|b\|_2^2}}{(2\pi)^{\frac{n}{2}}|\frac{1}{\lambda_b}I|^{\frac{1}{2}}}\right) - \sum_{k=1}^N \log P(y^{(k)} | x^{(k)}, \Theta ; \Lambda) \\
&= C + \frac{\lambda_A}{2}\|A\|_2^2 +\frac{\lambda_B}{2}\|B\|_2^2 +\frac{\lambda_a}{2}\|a\|_2^2 +\frac{\lambda_b}{2}\|b\|_2^2 - \sum_{k=1}^N \log P(y^{(k)} | x^{(k)}, \Theta ; \Lambda).
\end{align}
$$

where $$C$$ represents a constant independent of the parameter $$\Theta$$. The
reason why I ignore this is twofold. 

- First, when computing the Metropolis ratio, that $$C$$ will be the same for
  both $$U(\Theta)$$ and $$U(\Theta')$$ so we can ignore it.

- Second, we also need to use $$U$$ when we *sample* with HMC, and this means
  taking the gradient $$\nabla_\Theta U(\Theta)$$ which will kill $$C$$. (The
  Metropolis test is only to determine whether we accept or reject a proposal,
  but we need some way of actually *getting* that proposal). 

To elaborate on the second point, sampling using "Hamiltonian Dynamics" requires
the momentum update: 

$$
p = p - \frac{\epsilon}{2} \nabla_\Theta \left(\frac{U(\Theta)}{T}\right)
$$

where $$\epsilon$$ is a (leapfrog) step size parameter which we divide by two as
required from the leapfrog method. 

You can immediately see from this that $$p$$ must have the same dimensions as
$$\Theta$$. I think of $$p$$ as concatenating flattened $$A,B,a,b$$ weights, so
it's one giant vector. The gradient updates can be specified weight-by-weight,
which will change the corresponding "slices" of the vector $$p$$.  For instance,
with $$A$$ and abusing notation by re-using $$p$$, we have:

$$
\begin{align}
p &= p - \frac{\epsilon}{2} \nabla_A \left(\frac{U(A)}{T}\right) \\
&= p - \frac{\epsilon}{T2} \left(\lambda_A A - \sum_{k=1}^N \nabla_A \cdot \log P(y^{(k)} | x^{(k)}, A ; \lambda_A) \right)
\end{align}
$$

The $$\lambda_A A$$ term serves as a **weight decay regularizer**, and the sum
over the gradient of probabilities can be computed via backpropagation through
the neural network. 

- Remark 1: hopefully my above explanation clarifies why imposing a Gaussian
  prior on the weights is equivalent to $$L_2$$ regularization.

- Remark 2: consider using TensorFlow to get the gradients corresponding to the
  log probabilities. In particular, TensorFlow can return gradients using
  `tf.gradients`.

One thing I should point out: here, we can view $$- \log P(\mathcal{D} | \Theta;
\Lambda)$$ as a "cost function" that we're trying to minimize. This is
equivalent to minimizing the cross entropy loss between what our the neural
network predicts, and the distribution that consists of one-hot vectors of the
training labels.[^wellknown] That's precisely the loss function I'd use if I
were formulating the classification problem and solving it with stochastic
gradient descent instead of HMC. The implication is that it's OK to try and
maximize the $$\log P(y^{(k)} | x^{(k)}, \Theta ; \Lambda)$$ value that we see
above, which is what happens when we perform gradient ascent on it; intuitively,
the resulting weights will assign higher probability to the correct class. 

After the momentum updates, the position variables are updated using a similar
gradient-based step:

$$\Theta = \Theta + \epsilon \cdot \nabla_p \left(\frac{K(p)}{T}\right) = \Theta + \epsilon \cdot \frac{p}{T\sigma^2}$$

so that, intuitively, $$\Theta$$ is also updated in roughly the same direction
of the gradient. That's how HMC works and uses gradient information.



## Practical Considerations

### Averaging Predictions

Using Bayesian Neural Networks in practice often requires sampling a set of
neural network weights many times and then computing the mean and standard
deviation of the predictions.

<p style="text-align:center;"> 
<img src="{{site.url}}/assets/bnns.png" width="500" alt="bnns"><br>
<i>
A figure copied from the VIME paper (NIPS 2016) showing Bayesian Neural Network
predictions and uncertainty levels.
</i>
</p>

For instance, in the figure above (taken from [the VIME paper][5]) the authors
construct a regression task, where the network takes in a
scalar-valued[^technically] input $$x \in \mathbb{R}$$ and outputs a prediction
$$y \in \mathbb{R}$$. The red dots are the targets, while the green dots are the
predictions. It's clear that the red dots are clustered near the center of the
figure, so logically, our Bayesian Neural Networks should be very confident in
their predictions in those areas, and less confident outside the training data's
dominant regime. Indeed, the shaded areas confirm this, as they represent the
output mean plus/minus one and two standard deviations (I think the second
standard deviation is too far to see for the extremes in this figure) based on
different neural network weight parameters.

These types of figures are typically shown when people talk about Bayesian
Neural Networks, such as in [Yarin Gal's excellent tutorial][6].


### Code Implementation

I am working on implementing Bayesian Neural Networks [in my MCMC
repository][4], which is a TensorFlow implementation based on Tianqi Chen's
earlier pure numpy code. The code is a bit disorganized and not quite ready for
consumption by the public, but I think I'm getting something going with this
code.


<hr>

[^rate]: We're using the characterization using [shape and *rate*][3], not shape
    and scale.

[^notation]: I follow Radford Neal's notation in setting $$p$$ as the auxiliary
    momentum variables for HMC. Neal uses $$q$$ as the position variables, but I
    set them here as $$\Theta$$ for obvious reasons.

[^technically]: Well, technically they preprocessed the input to be
    $$[x,x^2,x^3,x^4]$$ but thinking of it as 1-D makes it much easier to plot.

[^wellknown]: This is a reasonably well-known fact in machine learning, but I
    would like to write up some details on this because I sometimes find myself
    looking up the derivations again.

[^hastings]: The proposals have the same density, so it is not necessary to
    perform a Metropolis-*Hastings* test. Why this is true is based on
    *reversibility*, but it is still somewhat unclear to me.

[1]:http://www.cs.toronto.edu/~radford/
[2]:https://arxiv.org/abs/1402.4102
[3]:https://en.wikipedia.org/wiki/Gamma_distribution#Characterization_using_shape_.CE.B1_and_rate_.CE.B2
[4]:https://github.com/DanielTakeshi/MCMC_and_Dynamics/tree/master/bnn
[5]:https://arxiv.org/abs/1605.09674
[6]:http://mlg.eng.cam.ac.uk/yarin/blog_3d801aa532c1ce.html
