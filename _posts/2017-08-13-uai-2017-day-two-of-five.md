---
layout:     post
title:      "Uncertainty in Artificial Intelligence (UAI) 2017, Day 2 of 5"
date:       2017-08-13 20:00:00
permalink:  2017/08/13/uai-2017-day-two-of-five/
---

For the second day of UAI 2017 (August 12)[^note], I followed the same initial
routine from the previous day. I woke up early, had a 4:30am gym session, ate a
hearty breakfast at the hotel's buffet, and then walked over to the conference
venue. The talks were held in the room shown in the following image:

<p style="text-align:center;"> 
<img src="{{site.url}}/assets/uai_2017/day2_talk_room.JPG" alt="sydney">
<i>
Room C4.8 in the ICC Sydney building.
</i>
</p>

Yeah, it's a fairly small room. Compare that to [the room used for the ICML
keynote talks][1], which is in the same building. Wow!

Fortunately, the second day of UAI started off [better than the first one][6],
since the captioner (a.k.a. "CART provider" or "stenographer") arrived. Whew.

At around 8:30am, the day began with some opening remarks from one of the
chairs. After that, it was time for MIT robotics professor [Leslie
Kaelbling][2]'s one-hour keynote talk on *Intelligent Robots in an Uncertain
World.* It was a nice, relatively high-level talk which centered on Partially
Observable Markov Decision Processes (POMDPs) and belief states, with
applications that focused on robotics.

<p style="text-align:center;"> 
<img src="{{site.url}}/assets/uai_2017/day2_keynote1.JPG" alt="sydney">
<i>
Professor Kaelbling gives the initial keynote talk for UAI 2017.
</i>
</p>

The experiments that she showed in the slides involved the PR2 robot, which I
assume is one of the primary robots in her lab. I wish I could use the PR2 one
of these days, or at least a robot similar to it. 

The final part of her talk contained a request for the UAI community to figure
out how to perform action selection in belief spaces. In other words, if we
don't know everything about the environment (which is always the case in real
applications) we have to pick actions on the basis of what we *believe* about
the world.

Overall, it was an excellent talk. There were a few sections that were
relatively difficult for me to follow, but I'm not sure if it was because there
was too much information to process in the slides (some of them had a lot!) or
if it was because I had a hard time getting used to the captioning.

After the keynote talk, we had oral sessions. In these, authors of papers
accepted to the conference give 20 minute talks. Not all the papers have oral
talks, though; they're reserved for those with the highest reviews. Also,
typically the *first author* is the one who gives presentations.

Today, there were four oral sessions, each of which consisted of one broad
research topic and three research papers in each (so each session was an
hour long). The first oral session was about deep models. Yay! Ming Jin
started off the oral sessions with his excellent talk on inverse
reinforcement learning.


<p style="text-align:center;"> 
<img src="{{site.url}}/assets/uai_2017/day2_oral1.JPG" alt="sydney">
<i>
UC Berkeley electrical engineering PhD student 
<a href="http://www.jinming.tech/">Ming Jin</a> gives a talk about his paper
<a href="https://arxiv.org/abs/1512.08065">Inverse Reinforcement Learning via Deep Gaussian Process</a>.
</i>
</p>

The two other talks for this oral session were also interesting and perhaps more
related to core Deep Learning research.

The second oral session was on the subject of *machine learning*, which is
probably not the best name for it, but whatever. Unfortunately, the papers were
quite mathematical and the speakers were a little difficult to understand (the
captioner had *major* difficulty) so it was hard for me to get much out of the
talks beyond trying to extract every ounce of information from the slides that I
could.

After a break for lunch --- though I was simply too full from eating earlier and
had to pass --- we had our second keynote of the day, *Expectations in Learning
and Inference* by Professor [Amir Globerson][5] of Tel Aviv University.

<p style="text-align:center;"> 
<img src="{{site.url}}/assets/uai_2017/day2_keynote2.JPG" alt="sydney">
<i>
The second keynote talk of the day.
</i>
</p>

This talk was more about math, in particular about expectations and
probabilities. What happens when we aren't given data but are given the expected
value? What can we determine from it? (See the slide above for the related
context.) The technical contribution was probably the development of bounds for
not probabilities, but the *minimum* of probabilities over a certain class (if
that makes sense?). I unfortunately had a harder time understanding this talk
compared to the first keynote. I thought I could follow slide by slide and
sentence by sentence in the captions when I saw the transcripts, but I couldn't
piece together a good story. Maybe this has happened to other people? 

In any case, for me I long ago decided that for research talks, **I don't try to
understand everything** but instead, I find **any interesting points and then
follow-up on these later**, either by emailing the author or (more likely)
simply searching online. Google has been great for people like me.

We had two more oral presentations after this second keynote. In between the
two, I had an entertaining conversation with another student from the University
of Helsinski who told me that hearing aids should have some way of blocking out
background noise. I told him that, sadly, they're *already* supposed to do that,
though he didn't give up and said that we should use *machine learning* to make
them block out noise. Yeah, that would be great.


<p style="text-align:center;"> 
<img src="{{site.url}}/assets/uai_2017/day2_poster.JPG" alt="sydney">
<i>
The poster session for today.
</i>
</p>

We wrapped up the day with a poster session which featured roughly one-third of
the UAI 2017 papers. (There are also poster sessions in day three and day four
of the conference.)

After this, I went to the nearby mall and found a quick, cheap Middle Eastern
restaurant for dinner. I ate by myself as I didn't know anyone else, and I
couldn't find any lonely person who I could pounce on with an invitation, but
that was OK with me.  I just wanted to see what the city had to offer, and I'm
pleased to say that I was not disappointed. Darling Harbor has a *ridiculous
ton* of awesome restaurants. It's food paradise for someone like me. 

<p style="text-align:center;"> 
<img src="{{site.url}}/assets/uai_2017/day2_harbor_night.JPG" alt="sydney">
<i>
The view of the lovely harbor at night. The conference is in the large building
located at the center-left with the lights on. To its right is a giant mall
(which the photo doesn't entirely reveal) with a LOT of stores and restaurants.
Wow.
</i>
</p>

<hr>

[^note]: The days when the posts are published on this blog do not necessarily
    coincide with the day that the conference took place, e.g., this post was
    published the day after.

[1]:https://twitter.com/Genomicsplc/status/894844813761171456
[2]:http://people.csail.mit.edu/lpk/
[3]:http://www.jinming.tech/
[4]:https://arxiv.org/abs/1512.08065
[5]:http://www.cs.tau.ac.il/~gamir/
[6]:https://danieltakeshi.github.io/2017/08/11/uai-2017-day-one-of-five/
