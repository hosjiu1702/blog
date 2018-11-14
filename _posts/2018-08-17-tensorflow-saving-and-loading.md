---
layout:     post
title:      "Saving and Loading TensorFlow Models, Without Reconstruction"
date:       2018-08-17 23:00:00
permalink:  2018/08/17/saving-and-loading-tensorflow/
---

Ever since I started using TensorFlow in late 2016, I've been a happy user of
the software. Yes, the word "happy" is deliberate and not a typo. While I'm
aware that it's fashionable in certain social circles to crap on TensorFlow, to
me, it's a great piece of software that tackles an important problem, and is
undoubtedly worth the time to understand in detail. Today, I did just that by
addressing one of my serious knowledge gaps of TensorFlow: *how to save and load
models*. To put this in perspective, here's how I used to do it:

- Count the number of parameters in my Deep Neural Network and create a
  placeholder vector for it.
- Fetch the parameters (e.g., using `tf.trainable_variables()`) in a list.
- Iterate through the parameters, flatten them, and "assign" them into the
  vector placeholder via `tf.assign` by careful indexing.
- Run a session on the vector placeholder, and save the result in a numpy file.
- When loading the weights, re-construct the TensorFlow model, download the
  numpy file, and re-assign weights.

You can see [some sample code in a blog post I wrote last year][5].

Ouch. I'm embarrassed by my code. It was originally based on John Schulman's
TRPO code, but I think he did that to facilitate the Fisher-Vector products as
part of the algorithm, rather than to save and load weights.

Fortunately, I have matured. I now know that it is standard practice to save and
load using `tf.train.Saver()`. By looking at the [TensorFlow documentation][2]
and [various blog posts][1] --- one aspect where TensorFlow absolutely shines
compared to other Deep learning software --- I realized that such savers could
save weights and meta-data into *checkpoint* files. As of TensorFlow 1.8.0, they
are structured like this:

```
name.data-00000-of-00001
name.index
name.meta
```

where `name` is what we choose. We have `data` representing the actual weights,
`index` representing the connection between variable names and values (like a
dictionary), and `meta` representing various properties of the computational
graph. Then, by reconstructing (i.e., re-running) code that builds the same
network, it's easy to get the same network running.

But then my main thought was: *is it possible to just load a network in a new
Python script without having to call any neural network construction code*?
Suppose I trained a *really* Deep Neural Network and saved the model into
checkpoints. (Nowadays, this would be hundreds of layers, so it's impractical
with the tools I have access to, but never mind.)  How would I load it in a new
script and deploy it, without having to painstakingly reconstruct the network?
And by "reconstruction" I specifically mean having to re-define the same
variables (the names must match!!) and building the same neural network in the
same exact layer order, etc.

The solution is to first use `tf.train.import_meta_graph`. Then, to fetch the
desired placeholders and operations, it is necessary to call
`get_tensor_by_name` from a TensorFlow graph.

I have written a proof of concept of the above high-level description in my
aptly-named ["TensorFlow practice" GitHub code repository][6]. The goal is to
train on (you guessed it) MNIST, save the model after each epoch, then load it
in a separate Python script, and check that each model gets exactly the same
test-time performance. (And it *should* be exact, since there's no
stochasticity.) As a bonus, we'll learn how to use `tf.contrib.slim`, one of the
many convenience wrapper libraries around stock TensorFlow to make it easier to
design and build Deep Neural Networks.

In [my training code][3], I use the keras convenience method for loading in
MNIST. As usual, I check the shapes of the training and testing data (and
labels):

```
(60000, 28, 28) float64 # x_train
(60000,) uint8          # y_train
(10000, 28, 28) float64 # x_test
(10000,) uint8          # y_test
```

Whew, the usual sanity check passed.

Next, I use `tf.slim` to build a simple Convolutional Neural Network. Before
training, I always like to print the state of the tensors after each layer, to
ensure that the sizing and dimensions make sense. The resulting printout is
here, where each line indicates the value of a tensor *after* a layer has been
applied:

```
Tensor("images:0", shape=(?, 28, 28, 1), dtype=float32)
Tensor("Conv/Relu:0", shape=(?, 28, 28, 16), dtype=float32)
Tensor("MaxPool2D/MaxPool:0", shape=(?, 14, 14, 16), dtype=float32)
Tensor("Conv_1/Relu:0", shape=(?, 14, 14, 16), dtype=float32)
Tensor("MaxPool2D_1/MaxPool:0", shape=(?, 7, 7, 16), dtype=float32)
Tensor("Flatten/flatten/Reshape:0", shape=(?, 784), dtype=float32)
Tensor("fully_connected/Relu:0", shape=(?, 100), dtype=float32)
Tensor("fully_connected_1/Relu:0", shape=(?, 100), dtype=float32)
Tensor("fully_connected_2/BiasAdd:0", shape=(?, 10), dtype=float32)
```

For example, the inputs are each 28x28 images. Then, by passing them through a
convolutional layer with 16 filters and with padding set to the same, we get an
output that's *also* 28x28 in the first two axis (ignoring the batch size axis)
but which has 16 as the number of channels. Again, this makes sense.

During training, I get the following output, where I evaluate on the full test
set after each epoch:

```
epoch, test_accuracy, test_loss
0, 0.065, 2.30308
1, 0.908, 0.31122
2, 0.936, 0.20877
3, 0.953, 0.15362
4, 0.961, 0.12030
5, 0.967, 0.10056
6, 0.972, 0.08706
7, 0.975, 0.07774
8, 0.977, 0.07102
9, 0.979, 0.06605
```

At the beginning, the test accuracy is just 0.065, which isn't far from random
guessing (0.1) since no training was applied. Then, after just one pass
through the training data, accuracy is already over 90 percent. This is expected
with MNIST; if anything, my learning rate was probably too small.  Eventually, I
get close to 98 percent.

More importantly for the purposes of this blog post, after each epoch `ep`, I
save the model using:

{% highlight python %}
ckpt_name = "checkpoints/epoch"
saver.save(sess, ckpt_name, global_step=ep) # saver is a `tf.train.Saver`
{% endhighlight %}

I now have all these saved models:

```
total 12M
-rw-rw-r-- 1 daniel daniel   71 Aug 17 17:07 checkpoint
-rw-rw-r-- 1 daniel daniel 1.1M Aug 17 17:06 epoch-0.data-00000-of-00001
-rw-rw-r-- 1 daniel daniel 1.2K Aug 17 17:06 epoch-0.index
-rw-rw-r-- 1 daniel daniel  95K Aug 17 17:06 epoch-0.meta
-rw-rw-r-- 1 daniel daniel 1.1M Aug 17 17:06 epoch-1.data-00000-of-00001
-rw-rw-r-- 1 daniel daniel 1.2K Aug 17 17:06 epoch-1.index
-rw-rw-r-- 1 daniel daniel  95K Aug 17 17:06 epoch-1.meta
-rw-rw-r-- 1 daniel daniel 1.1M Aug 17 17:06 epoch-2.data-00000-of-00001
-rw-rw-r-- 1 daniel daniel 1.2K Aug 17 17:06 epoch-2.index
-rw-rw-r-- 1 daniel daniel  95K Aug 17 17:06 epoch-2.meta
-rw-rw-r-- 1 daniel daniel 1.1M Aug 17 17:06 epoch-3.data-00000-of-00001
-rw-rw-r-- 1 daniel daniel 1.2K Aug 17 17:06 epoch-3.index
-rw-rw-r-- 1 daniel daniel  95K Aug 17 17:06 epoch-3.meta
-rw-rw-r-- 1 daniel daniel 1.1M Aug 17 17:06 epoch-4.data-00000-of-00001
-rw-rw-r-- 1 daniel daniel 1.2K Aug 17 17:06 epoch-4.index
-rw-rw-r-- 1 daniel daniel  95K Aug 17 17:06 epoch-4.meta
-rw-rw-r-- 1 daniel daniel 1.1M Aug 17 17:06 epoch-5.data-00000-of-00001
-rw-rw-r-- 1 daniel daniel 1.2K Aug 17 17:06 epoch-5.index
-rw-rw-r-- 1 daniel daniel  95K Aug 17 17:06 epoch-5.meta
-rw-rw-r-- 1 daniel daniel 1.1M Aug 17 17:06 epoch-6.data-00000-of-00001
-rw-rw-r-- 1 daniel daniel 1.2K Aug 17 17:06 epoch-6.index
-rw-rw-r-- 1 daniel daniel  95K Aug 17 17:06 epoch-6.meta
-rw-rw-r-- 1 daniel daniel 1.1M Aug 17 17:06 epoch-7.data-00000-of-00001
-rw-rw-r-- 1 daniel daniel 1.2K Aug 17 17:06 epoch-7.index
-rw-rw-r-- 1 daniel daniel  95K Aug 17 17:06 epoch-7.meta
-rw-rw-r-- 1 daniel daniel 1.1M Aug 17 17:06 epoch-8.data-00000-of-00001
-rw-rw-r-- 1 daniel daniel 1.2K Aug 17 17:06 epoch-8.index
-rw-rw-r-- 1 daniel daniel  95K Aug 17 17:06 epoch-8.meta
-rw-rw-r-- 1 daniel daniel 1.1M Aug 17 17:07 epoch-9.data-00000-of-00001
-rw-rw-r-- 1 daniel daniel 1.2K Aug 17 17:07 epoch-9.index
-rw-rw-r-- 1 daniel daniel  95K Aug 17 17:07 epoch-9.meta
```

In [my loading/deployment code][4], I call this relevant code snippet for each
epoch:

{% highlight python %}
for ep in range(0,9):
    # It's a two-step process. For restoring, don't include the stuff
    # from the `data`, i.e. use `name`, not `name.data-00000-of-00001`.
    sess = tf.Session()
    saver = tf.train.import_meta_graph('checkpoints/epoch-{}.meta'.format(ep))
    saver.restore(sess, 'checkpoints/epoch-{}'.format(ep))
{% endhighlight %}

Next, we need to get references to *placeholders* and *operations*. Fortunately
we can do precisely that using:

{% highlight python %}
graph = tf.get_default_graph()
images_ph = graph.get_tensor_by_name("images:0")
labels_ph = graph.get_tensor_by_name("labels:0")
accuracy_op = graph.get_tensor_by_name("accuracy_op:0")
cross_entropy_op = graph.get_tensor_by_name("cross_entropy_op:0")
{% endhighlight %}

Note that these names match the names I assigned during my training code, except
that I append an extra `:0` at the end of each name. The importance of getting
names right is why I will start carefully naming TensorFlow variables in my
future code.

After using these same placeholders and operations, I get the following
test-time output:

```
1, 0.908, 0.31122
2, 0.936, 0.20877
3, 0.953, 0.15362
4, 0.961, 0.12030
5, 0.967, 0.10056
6, 0.972, 0.08706
7, 0.975, 0.07774
8, 0.977, 0.07102
9, 0.979, 0.06605
```

(I skipped over epoch 0, as I didn't save that model.)

Whew. The above accuracy and loss values exactly match. And thus, we now know
how to load and use stored TensorFlow checkpoints without having to reconstruct
the entire training graph. Achievement unlocked.


[1]:https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
[2]:https://www.tensorflow.org/api_docs/python/tf/train/Saver
[3]:https://github.com/DanielTakeshi/tf_practice/blob/master/saving_and_loading/train.py
[4]:https://github.com/DanielTakeshi/tf_practice/blob/master/saving_and_loading/load.py
[5]:https://danieltakeshi.github.io/2017/07/06/saving-neural-network-model-weights-using-a-hierarchical-organization/
[6]:https://github.com/DanielTakeshi/tf_practice
