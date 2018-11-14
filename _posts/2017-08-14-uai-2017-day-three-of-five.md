---
layout:     post
title:      "Uncertainty in Artificial Intelligence (UAI) 2017, Day 3 of 5"
date:       2017-08-14 10:00:00
permalink:  2017/08/14/uai-2017-day-three-of-five/
---

The third day of UAI 2017 (August 13) started off with Stanford Professor
[Christopher Ré][1] giving the first keynote talk of the day about his group's
project called [Snorkel][2]. Chris is epitome of a "rock-star academic," and he
has a ridiculous amount of publications in the last few years. His lengthy list
of awards includes the well-known MacArthur "Genius" Fellowship.

<p style="text-align:center;"> 
<img src="{{site.url}}/assets/uai_2017/day3_chrisre.JPG" alt="sydney">
<i>
Stanford Professor Christopher Ré gives the first keynote talk of the day.
</i>
</p>

I really enjoyed Professor Ré's talk, both for the content and for the "style"
(i.e. at the right technical level, good visuals, etc.). He inserted some humor
now and then, as you can see in the slide above. Anyone want to win the parasite
award? Heh. I also try to include humor in my talks.

Anyway, as I mentioned earlier, the main technical part of the talk was about
Snorkel. It's an important project, because it helps us deal with "dark" or
unlabeled data, which is what the vast majority of data will look like in real
life. How can we make sense of dark data, and "clean it up" to make it more
useful?  This is critical because, as Professor Ré said in the talk:

> Training data is the new, new oil.

(Yes, he said "new" twice.)

I was amusingly reminded of [Andrew Ng's famous][3] (or infamous, depending on
your opinion) phrase:

> AI is the new electricity.

In case you are curious, I agree with both of the above quotes.

You can find more information about Snorkel on the project website. What's great
is that there are also lots of blog posts. His group really likes to write blog
posts! At least I have something in common with them.

The first oral session was about "Representations", a research sub-field which I
am unfortunately not familiar with, and so I had an extremely hard time
following the material. I tried to gather pieces of what I could and recorded
anything interesting in my ongoing Google Doc containing my notes from UAI 2017.
[As I stated in my last blog post][4], **I do not try to follow talks in their
entirety** --- I couldn't do that even if I *wanted* to --- but I record *bits
and pieces* of intriguing stuff which are candidates for future investigation.

During breaks, I worked on outlining these blog posts; I drafted them in Google
Docs.

The second oral session was about ... reinforcement learning! Awesome. At least
I should have more background information for this material. Of the four papers
presented, the fourth one seemed to have the most interesting material in it.
The UAI organizers must have agreed, because the authors ([Shayan Doroudi][5]
along with Philip Thomas and Emma Brunskill) won the UAI 2017 Best Paper Award
for the paper "[Importance Sampling for Fair Policy Selection][6]." 

*Fairness* is becoming a recurring theme in research and academia nowadays along
with *safety* and (as you'll see later) *health care*. The talk was excellent
since Shayan has good speaking skills. He motivated the problem with a quick
example of choosing between two policies, one of which was obviously better than
the other. Despite the apparent simplicity of choosing the policies, importance
sampling approaches can actually choose the *worse* policy more often than not.

<p style="text-align:center;"> 
<img src="{{site.url}}/assets/uai_2017/day3_best_paper.JPG" alt="sydney">
<i>
Shayan Doroudi (to the left) being presented with the UAI 2017 Best Paper Award
by the conference chairs. Congratulations!
</i>
</p>

We now have the following award-winning reinforcement learning papers:

- Importance Sampling for Fair Policy Selection (UAI 2017 best paper)
- Modular Multitask Reinforcement Learning with Policy Sketches (ICML 2017
  runner-up best paper)
- Value Iteration Networks (NIPS 2016 best paper)
- Dueling Network Architectures for Deep Reinforcement Learning (ICML 2016 best
  paper)

At some point, I'd like to grind through the details in these papers. I know the
high level idea of each of these but aside from perhaps the dueling networks
paper, the details elude me.

After a brief lunch break with another student from India --- whom I found
standing alone and thus it made sense for us to go to lunch together --- we had
our second keynote talk. Duke Professor [Katherine Heller][7] gave a talk about
machine learning in health care. Gee, is anyone seeing a trend with machine
learning applications?

<p style="text-align:center;"> 
<img src="{{site.url}}/assets/uai_2017/day3_heller.JPG" alt="sydney">
<i>
Katherine Heller gives the second keynote talk of the day.
</i>
</p>

You can see the outline of her talk in my picture above. I remember that she
discussed the following topics:

- Health conditions: chronic kidney disease, sepsis, and multiple sclerosis.
- Health issues: delayed diagnosis results in problems, surgery can introduce
  complications; her exact figure was 15% of the time but there are obvious
  simplifications with numbers like that.
- *Modeling* health issues: using graphical models with latent variables.
  Basically, given health conditions (or a sequence of conditions measured at
  times) what can we say about the patient's health? I also saw a few mentions
  of RNNs and LSTMs there (wow, really?) and would be interested in learning
  more.

Given that much of the talk was about, I believe, modeling health care, I
sometimes wonder how accurate our models are. The United States has one of the
most inefficient health care systems in the developed world, and I wish we could
use some machine learning to cut away at the inefficiency. 

After Professor Heller's talk, we had the usual poster session. I managed to
engage in a few interesting one-on-one conversations, which is good enough for
me!

We then had a special event provided by the conference: a dinner cruise along
Darling Harbor. Awesome! The buffet options included a wide range of food
options: prawns, Indian chicken curry, Thai fish curry, pastas, potatoes, and of
course, lots of salad options.

<p style="text-align:center;"> 
<img src="{{site.url}}/assets/uai_2017/day3_cruise1.JPG" alt="sydney">
<i>
UAI 2017 conference attendees lining up to get dinner.
</i>
</p>

I don't know about how others feel, but *every time* there's an event like this
where I have to pick my own seat amidst a large dinner gathering, I worry and
overthink it *way* too much.  Fortunately, there was a student who I met earlier
at the conference who told me to sit near the center (gulp) of a table filled
with other graduate students, thus saving me the stress of coming up with the
decision myself. I was happy with this because it meant I wasn't sitting by
myself, and because it's better for me to know other graduate students (and
potential future collaborators/colleagues) rather than, for instance, industry
sponsors.

Yes, it was *extremely* noisy in the ship, and I couldn't participate in
substantive conversations, but hey, at least I was sitting with other graduate
students. And it seemed like there was some ongoing discussion regarding my
blog, judging by how several of the other students nearby kept looking at my
name tag in order to correctly spell my name in Google.

Throughout the cruise, we would frequently walk to the top of the ship and view
Darling Harbor.

<p style="text-align:center;"> 
<img src="{{site.url}}/assets/uai_2017/day3_cruise2.JPG" alt="sydney">
<i>
A view of Luna Park.
</i>
</p>

<p style="text-align:center;"> 
<img src="{{site.url}}/assets/uai_2017/day3_cruise3.JPG" alt="sydney">
<i>
A view of the Sydney Opera House.
</i>
</p>

It's times like these when I wish I had a girlfriend, so that we could go on a
vacation together and explore Darling Harbor.

[1]:https://cs.stanford.edu/people/chrismre/
[2]:https://hazyresearch.github.io/snorkel/
[3]:https://www.youtube.com/watch?v=21EiKfQYZXc
[4]:https://danieltakeshi.github.io/2017/08/13/uai-2017-day-two-of-five/
[5]:http://www.cs.cmu.edu/~shayand/
[6]:http://www.cs.cmu.edu/~shayand/papers/UAI2017.pdf
[7]:http://www2.stat.duke.edu/~kheller/
