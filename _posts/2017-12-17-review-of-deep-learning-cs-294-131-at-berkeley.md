---
layout:     post
title:      "Review of Deep Learning (CS 294-131) at Berkeley"
date:       2017-12-17 12:00:00
permalink:  2017/12/17/review-of-deep-learning-cs-294-131-at-berkeley/
---

This semester, I took [CS 294-131, a Deep Learning "special topics" course][1]
which has been offered each semester since Fall 2016 for a variable amount of
class units and will be taught *again* next semester ([the course website is
already up][3]). As usual, it was co-taught by the Trevor Darrell and Dawn Song
team. The course is low-commitment for them because it's a seminar and they
don't have to give lectures or prepare assignments and exams. CS 294-131 meets
only once a week; for us, it was Mondays from 1:00PM to 2:30PM. Each meeting
featured a guest speaker from academia or industry who gave a talk on his or her
cutting-edge Deep Learning research results.

Here were some of the highlights for me:

- Vladlen Koltun's talk about his ICLR 2017 paper *Learning to Act by Predicting
  the Future*. I enjoyed his presentation, though admittedly most of it was
  because he was funny and actively engaging with the audience. I [previously
  blogged about the more technical aspects here][2].

- Barret Zoph and Quoc Le's joint talk on neural architecture search, also from
  ICLR 2017 ([here's the OpenReview link][4]) and also (like Koltun's paper) an
  oral presentation at that conference. I've been hoping to find some time to
  read their paper and perhaps the winter break will afford me that opportunity.
  Zoph and Le's presentation featured *a lot* of aggressive questioning from
  students, to the point where Professor Song asked the students to quiet down
  and let the speakers proceed. Fortunately, at least to me, the *technical
  content* of the presentation was interesting enough to keep my attention.

- Ross Girshik's talk on computer vision and object recognition. Actually, we
  had a fire alarm for this one, which delayed the start class for about 15
  minutes ...  so then we had to find a new room. Unfortunately, it took about
  10 *more* minutes to get the *projector* working, and then we were told we had
  to leave the room at around 2:10PM. At least when Girshik was actually able to
  talk about computer vision, I found the historical overview to be educational.

- Percy Liang's presentation on fighting black boxes and adversaries in Deep
  Learning. This was somewhat more theoretical work but he didn't go too much
  into the details. I am less familiar with his work but would like to get
  accustomed with it as *adversarial learning* is a pretty hot topic in robotics
  these days.

In case you're wondering, yes I had the usual sign language interpreters for the
class. Yes they were unhappy, but they tried, and we were able to agree on a few
terminology-related issues beforehand. And yes, I didn't get all the technical
details from the talks. I tried to allocate two hours before class to do the
background reading, but inevitably that turned into 1.5 hours ... and then 1
hour ... and then 30 minutes. How do people manage to do class readings ahead of
time when they're juggling four major research projects? Or do people with
normal hearing just find it easy to absorb almost all the technical stuff in
these talks without prior preparation?[^tellme]

As I mentioned earlier, CS 294-131 can be taken for a different amount of
credits. This year, we had the option of taking it for 1, 2, or 3 credits. We
got one credit for doing "arXiv summaries" and "discussion leads" and two for
doing a class project. I decided that those summaries and discussions would be
too much of a hassle, so I took CS 294-131 for two credits. It helps that I've
long since finished my course requirements.

While I enjoyed some of the Deep Learning talks, I do have some criticisms about
CS 294-131:

- There are too many websites/links related to the class. We have Piazza, the
  course website, the Google group (seriously?) and Slack channels, with one for
  each new week. I think this is too much information and things should be
  centralized in two spots at most --- the course website and Piazza.

- I'm also not a fan of the arXiv leads, which was new this semester. Students
  had to give 1-minute presentations on Deep Learning papers that appeared on
  arXiv the past week. The problem with this is that the majority of students
  tried to cram as may technical details in their talk as possible, rather than
  give the clear key insight from the paper. In addition, students often went
  over their allotted speaking time (gee, who would have guessed??).

- Finally, I have no idea why class participation is worth 20% of the grading
  here. On Piazza, we were *literally* told that we would get class
  participation credit by simply attending the lectures. Not only does it not
  make sense to award students who attend lectures but don't pay attention, it
  also hurts those who watch the video livestreams to reduce pressure on the
  lecture room, since the first lecture was "standing-room only."

To be honest, I didn't quite enjoy the class as much as I should have, and my
project didn't turn out as well as I would have liked.  I worked on a Deep
Reinforcement Learning project with three other students, and hopefully that
will turn into a research paper later, but in retrospect, it's difficult to
coordinate a four-person project when everyone else has other priorities. 

I don't plan to take the Spring 2018 version of the course, but I'll certainly
keep track of the papers in the background reading. I'm excited to see who the
guest speakers will be this time around ...

<hr>

[^tellme]: For readers of this blog who are EECS PhD students (and yes, I know
    you read this blog) that means I'm not-so-subtly asking you to *tell me how
    much you can absorb from technical talks and lectures*. Posting as a comment
    here or emailing me personally works.

[1]:https://berkeley-deep-learning.github.io/cs294-131-f17/
[2]:https://danieltakeshi.github.io/2017/10/10/learning-to-act-by-predicting-the-future/
[3]:https://berkeley-deep-learning.github.io/cs294-131-s18/
[4]:https://openreview.net/forum?id=r1Ue8Hcxg&utm_campaign=Revue%20newsletter&utm_medium=Newsletter&utm_source=The%20Wild%20Week%20in%20AI
