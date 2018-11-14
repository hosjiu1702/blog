---
layout:     post
title:      "Self Supervision and Building Visual Predictive Models"
date:       2018-03-23 23:00:00
permalink:  2018/03/23/self-supervision-part-1/
---

I enjoy reading robotics and deep reinforcement learning papers that cleverly
apply self-supervision to learn some task. There's something oddly appealing
about an agent "semi-randomly" acting in a world and learning something useful
out of the data it collects.  Some papers, for instance, build *visual
predictive models*, which are those that enable the agent to anticipate the
future states of the world, which may be raw images (or more commonly, a latent
feature representation of them). Said another way, the agent learns an internal
physics model.  The agent can then use it to plan because it knows the effect of
its actions, so it can run internal simulations and pick the action that results
in the most desirable outcome.

In this blog post, I'll discuss a few papers about self-supervision and visual
predictive models by providing a brief description of their contributions. A
subsequent blog post will discuss the papers' relationships to each other in
further detail.


## Paper 1: Learning Visual Predictive Models of Physics for Playing Billiards (ICLR 2016)

"Billiards" in this paper refers to a generic, 2-D simulated environment of
balls that move and bounce around walls according to the laws of physics. As the
authors correctly point out, this is an environment that easily enables
extensive experiments: altering the number of balls, changing their sizes or
colors, and so forth.

While the agent "sees" a 2-D image of the environment, that is not the direct
input to the neural network nor is it what the neural network predicts.

- The *input* consists of the past four "glimpses" of the object, and the
  applied forces (which we assume known and tracked). The glimpses should be the
  128x128 RGB image of the environment, but perhaps "blacking out" everything
  except the object. (I'm not sure about the technical details, but the idea is
  intuitive.) Thus, the same network is used for *each* of the balls in the
  environment, which the authors call an "object-centric" model.  As one would
  expect, the input image is passed through a series of convolutional layers and
  then the forces are concatenated with that feature representation.

- The *output* is the object's predicted velocity for the current and subsequent
  (up to $$h$$) times. It is *not* the standard latent feature representation
  that other visual predictive models normally apply, because in billiards, they
  assume it is enough to know the displacements of the balls to track them.

The model is trained by minimizing

$$\sum_{k=1}^h w_k\|\tilde{u}_{t+k} - u_{t+k}\|_2^2$$

where $$w_k$$ is a weighing factor that is larger for shorter-term (smaller
$$k$$) time steps.  Good, this makes sense.

The authors show that they are able to predict the trajectories of balls, and
that this can be generalized and also used for planning.


## Paper 2: Learning to Poke by Poking: Experiental Learning of Intuitive Physics (NIPS 2016)

I discussed this paper in a [previous blog post][2]. Heh, you can tell that I'm
interested in this stuff.


## Paper 3: Learning Hand-Eye Coordination for Robotic Grasping with Deep Learning and Large-Scale Data Collection (IJRR 2017)

This is the famous (or infamous?) "arm-farm" paper from Google.  The dataset
here is *MASSIVE* --- I don't know of a self-supervision paper with real robots
that contains this much data. The authors collected 800,000 (semi-)random grasp
attempts collected over two months by running up to 14 robots in parallel.  In
fact, even this somewhat understates the total amount of data: *each grasp*
consists of $$T$$ training data points of the form $$(I_t^i, p_T^i - p_t^i,
\ell_i)$$ which contains the current camera image, the vector from the current
pose to the one that is eventually reached, and the success of the grasp.

The data then enables the robot to effectively learn hand-eye coordination by
continuous visual servoing, without the need for camera calibration.  Given a
camera image of the workspace, and independently of the calibration or robot
pose, the trained CNN predicts the probability that the motion of the gripper
results in successful grasps.

During data collection, the labels (either a successful grasp or not) must be
automatically supplied. The authors do this with (a) checking if the gripper
closed or not, and (b) an image subtraction test, testing the image before and
after the object was grasped. This makes sense to me. The first test is used,
and then the second is a backup to check for small objects.  I can see how it
might fail, though, such as if the robot grasped the wrong object or pushed
the target object to the side rather than picking it up, either of which would
result in a different image than the starting one

The use of robots running in parallel means that each can collect a diverse
dataset on its own, in part due to different actions and in part due to
different material properties of each gripper. This is an application of the A3C
concept from Deep Reinforcement Learning for real, physical robotics.

There are a lot of things that I like from this paper, but one that really seems
intriguing for future AI applications is that the data enabled the robots to
learn different grasping strategies for different types of objects, such as the
soft vs hard difference the authors observed.

## Paper 4: Learning to Act by Predicting the Future (ICLR 2017)

I discussed this paper in a [previous blog post][1].


## Paper 5: Combining Self-Supervised Learning and Imitation for Vision-Based Rope Manipulation (ICRA 2017)

The same architectural idea from the "Learning to Poke" paper is used in this
one to jointly learn forward and inverse dynamics models. Instead of poking, the
robot learns rope manipulation, a complicated task to model with hard-coded
physics.

In my opinion, one of the weaknesses in the "Learning to Poke" paper was the
greedy planner.  The planner saw the current and *goal* images, and had to infer
the intermediate actions. This prevented the robot from learning longer-horizon
tasks, because the goal image could be quite different from the current one. In
this paper, the authors allow for longer-horizon learning by providing one human
demonstration of the task. The demonstration consists of a sequence of images,
each of which are repeatedly fed into the neural network model at each time
step. Thus, the goal image should be the one that correspond to the next time
step, which appears to be more tractable.

They ran their Baxter robot autonomously for 500 hours, collecting 60,000
training data points.


## Paper 6: Curiosity-Driven Exploration by Self-Supervised Prediction (ICML 2017)

They build on top of an existing RL algorithm, A3C, by modifying the reward
function so that at each time step $$t$$, the reward is $$r_t^{i}+r_t^{e}$$
instead of just $$r_t^{e}$$, where $$r_t^{i}$$ is the *curiosity reward* and
$$r_t^{e}$$ is the reward from the environment.

In sparse rewards, such as the Doom environment from OpenAI they use (and, I
might add, the recent robotics environments, also from OpenAI) the environment
reward is zero almost everywhere, except for 1 at the goal. This makes it
effectively an intractable problem for off-the-shelf RL algorithms. Hence, by
building a predictive model, given current and subsequent states $$s_t$$ and
$$s_{t+1}$$ they can assign the curiosity reward to be

$$r_t^i = \frac{\eta}{2}\|\hat{\phi}(s_{t+1}) - \phi(s_{t+1})\|_2^2$$

which measures the difference in the predicted *latent space* of the successor
state, respectively. The inverse dynamics model takes in $$(s_t,s_{t+1})$$
during training and predicts $$a_t$$. The forward dynamics model predicts the
latent successor state $$\hat{\phi}(s_{t+1})$$ shown above.

They argue that their form of curiosity has three benefits: solving tasks with
sparse rewards, exploring the environment, and learning skills that can be
reused and applied in different scenarios. One interesting conjecture from the
third claim is that if the agent simply does the same thing over and over again,
the curiosity reward will go down to zero because the agent is stuck in the same
latent space. Only by "learning" new actions that substantially change the
latent space will the agent then be able to obtain new rewards.

The results on Doom and Mario environments are impressive.



## Paper 7: Zero-Shot Visual Imitation (ICLR 2018)

Wait, zero-shot visual imitation (learning)? How is this possible? 

First, let's be clear on their technical definition: "zero-shot" means that they
are still allowed to observe a demonstration of the task, but it has to be only
the state space (i.e., images), so actions are *not* included.  The second part
of the definition means that expert demonstrations (regardless of states or
actions) are not allowed during training.

OK, that makes sense. So ... the robot just sees the images of the demo at
inference time, and must imitate it. That's a high bar. The key must be to
develop a sufficient prior --- but how? By having the agent move (semi-)randomly
to learn physics, of course! 

In terms of the visual predictive model, the paper does a nice job describing
four different models, starting from the ICRA 2017 rope manipulation paper and
moving towards the one they use for their experiments. Their final model
conditions on the final goal and uses recurrent neural networks, and is
augmented with a separate neural network that predicts whether the goal has been
attained or not.

The paper presents two sets of experiments. One is a navigation task using a
mobile robot, and the other is a rope manipulation task using the Baxter robot.
With zero-shot visual imitation, the Baxter robot *doubles* the performance of
rope manipulation compared to the results from ICRA 2017. Thus, if I'm thinking
about rope manipulation benchmarks, I better check out this paper and not the
ICRA 2017 one.  I also assume that zero-shot visual imitation would result in
better poking performance than "Learning to Poke" if the poking requires
long-term planning.

Results for the navigation agent are also impressive.

This is not a deep reinforcement learning paper, though one could argue for the
use of Deep RL as an alternative to self-supervision. Indeed, that was a point
raised by one of the reviewers.



## Additional References

Here are a few additional papers that are somewhat related to the above, and
which I don't have time to write about in detail ... yet.

- [Unsupervised Learning for Physical Interaction through Video Prediction][6]
  is another interesting paper on imagining the future based on predicting pixel
  motion.

- [One-Shot Visual Imitation Learning via Meta-Learning][8] allows robots to
  learn how to perform tasks with a single demonstration. It's somewhat related
  to the "Zero-Shot Visual Imitation" paper, except those papers use very
  different solutions for different problems. I'd like to compare them in more
  detail later.

- [Reinforcement Learning with Unsupervised Auxiliary Tasks][7] works by having
  a reinforcement learning agent consider a series of "pseudo" loss functions
  that it considers under its objective function.

- [Diversity is All You Need][5], which argues that by using entropy correctly,
  an agent can automatically learn useful skills in an environment. It's related
  to the "Curiosity" paper in discovering new skills.


[1]:https://danieltakeshi.github.io/2017/10/10/learning-to-act-by-predicting-the-future/
[2]:https://danieltakeshi.github.io/2018/03/03/learning-to-poke-by-poking
[3]:https://www.reddit.com/r/MachineLearning/comments/6bc8ul/r_curiositydriven_exploration_by_selfsupervised/
[4]:https://sites.google.com/view/zero-shot-visual-imitation/home
[5]:https://arxiv.org/abs/1802.06070
[6]:https://arxiv.org/abs/1605.07157
[7]:https://arxiv.org/abs/1611.05397
[8]:https://arxiv.org/abs/1709.04905
