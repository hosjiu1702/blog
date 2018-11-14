---
layout:     post
title:      "Before Robots Can Take Over the World, We Have to Deal With Calibration"
date:       2017-07-29 10:00:00
permalink:  2017/07/29/before-robots-can-take-over-the-world-we-have-to-deal-with-calibration/
---

<p style="text-align:center;"> 
<img src="{{site.url}}/assets/Robots_Side_by_Side.png" alt="robots">
<i>Clockwise, starting from top left: the Da Vinci, the ABB YuMi, the PR2, and
the Toyota HSR.</i>
</p>

I now have several months of research experience in robotics. I am familiar with
the following four robots, roughly in descending order of my knowledge of them:

- **[Da Vinci][1]**. Price: $2,000,000 (!!!). I'm not sure how much of the full
  set I use, though --- I only use two of the arms, and the components might be
  cheaper versions. After all, even for well-funded Berkeley robotics labs,
  paying $2 million for a robot is impractical. [Smaller hospitals][6] also
  cannot afford the full Da Vinci.
- **[Toyota Human Support Robot][2]** (HSR). Price: ???. Oddly, I can't find a
  price! In fact, I'm not even sure *where to buy it*.
- **[ABB YuMi][3]**. Price: $40,000. At least this one is on the "cheap" end ...
  I think?
- **[Personal Robot 2][4]** (PR2), from Willow Garage[^willow].  Price:
  $280,000.  Yikes! And that's the open source version -- the raw sticker cost
  started as $400,000 when it was released in 2010. Given that Willow Garage no
  longer exists, I'm not sure if it's possible to buy a PR2.

I have sadly never touched or worked with the YuMi and the PR2, though I've
manipulated the Da Vinci on a regular basis. The one-sentence summary: it's a
surgical robotics system which is currently the only such system approved by the
U.S. Food and Drug Administration.

This is interesting. Now let's switch to another topic: suppose you talk to a
layperson about working in robotics. One typical, half-serious conversation
involves this question: *when will your robots take over the world?*

I would respond by pointing out the obvious restrictions placed on the Da Vinci.
It's fixed to a base, with arms that perform a strictly limited set of
surgery-related functions. So ... it can't really "join forces" with other Da
Vincis and somehow navigate the real world.

But perhaps, your conversationalist argues, that we can take the *arms* of the
Da Vinci and *integrate* them to a mobile robot (e.g. the Toyota HSR). If the Da
Vinci works in surgical applications, then it must logically be very
nimble-fingered[^nimble]. Think of the things it can do! It can pick locks, use car
keys, redirect electric wires, and so forth.

Alas, from my experience, it's difficult to even get the robot arms to go where
I want to them to go. To make this more concrete, suppose we're looking at an
image of a flat surgical platform through the Da Vinci camera mounted above.
When we look at the image, we can visually identify the area where we want the
arm (or more precisely, the "end effectors") to go to, and we can figure out the
pixel values. Mathematically, given $$(x_p,y_p)$$ in pixel space, with
$$x_p,y_p$$ positive integers typically bounded by 1080 and 1920 (i.e. the
resolution) we want to find the corresponding six-dimensional *robot*
coordinates $$(x_r,y_r,z_r,y_r,p_r,r_r)$$ where I've added *yaw*, *pitch*, and
*roll* along with the $$r$$ subscript representing "robot."

The problem is that we can't directly convert from pixel to robot points.  The
best strategy I've used for dealing with this is to do some supervised learning.
Given known $$x_p,y_p$$ points, I can manually move the robot end effectors to
where they should be. Then my code can record the robot coordinates. I repeat
this process many times to get a dataset, then perform supervised learning (e.g.
with a random forest) to find the mapping. Finally, I use that map in real
experiments.

This is the process of *calibration*. And unfortunately, it doesn't work that
well. I've found that I consistently get errors of at least 4 millimeters, and
for automated robot surgery that's pretty dangerous. To be clear, I'm focused on
*automated* surgery, not teleoperation, which is when a human expert surgeon
controls some switches which then translate to movement of the Da Vinci arms.

Indeed, calibration is a significant enough problem that it can be part of a
research paper on its own. For instance, here's a [2014 paper from the IEEE
International Conference on Automation Science and Engineering][7] (CASE) which
deals with the problem of kinematic control (which involves calibration).

Calibration --- or more broadly, kinematic control --- is one of those necessary
evils for research. I will tolerate it because I enjoy working with robotics and
with enough manual time, usually calibration becomes workable for running
experiments. 

I hope to continue working with robotics to make them be more autonomous.
Sadly, they won't be taking over the world.

<hr>

[^willow]: Willow Garage also developed the ROS system, which is used in many
    robotics systems, including the Da Vinci and Toyota HSR.  While it's no
    longer around, [it has a long history][5] and is considered a iconic
    robotics company. Many companies have spawned from Willow Garage.  I'm
    embarrassed to admit that I didn't know about Willow Garage until a few
    months ago. I really need to read more about the tech industry; it might be
    more informative for me than skimming over the latest political insults
    hurled on *The New York Times* and *The Wall Street Journal*.

[^nimble]: I think a conscious Da Vinci might take some offense at the YuMi
    being advertised as "[The Most Nimble-Fingered Robot Yet][8]."
    

[1]:http://biorobotics.ri.cmu.edu/robots/daVinci.php
[2]:http://www.toyota-global.com/innovation/partner_robot/family_2.html
[3]:http://new.abb.com/products/robotics/industrial-robots/yumi
[4]:http://www.willowgarage.com/pages/pr2/overview
[5]:https://readwrite.com/2016/08/29/the-legacy-of-willow-garage-the-parc-for-robotics-pl1/
[6]:https://www.technologyreview.com/s/603700/for-hospitals-that-cant-afford-a-surgical-robot-this-500-device-could-fit-the-bill/
[7]:http://goldberg.berkeley.edu/pubs/jeff-case2014-v30.pdf
[8]:https://www.technologyreview.com/s/607931/meet-the-most-nimble-fingered-robot-yet/
