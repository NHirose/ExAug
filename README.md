# ExAug: Robot-Conditioned Navigation Policies via Geometric Experience Augmentation
 
**Summary**: Our method, ExAug can realize the navigation with obstacle avoidance by only using an RGB image. Our control policy is trained from the synthetic images from multiple datasets by minimizing our proposed objectives. In this repository, we release our trained control policies for spherical camera image(double fisheye image), fisheye image(170 degree FOV) and narrow FOV camera. 

Please see the [website](https://sites.google.com/view/exaug-nav/) for more technical details.

#### Paper
**["ExAug: Robot-Conditioned Navigation Policies via Geometric Experience Augmentation"](https://arxiv.org/abs/2210.07450)**  
**[Noriaki Hirose](https://sites.google.com/view/noriaki-hirose/), [Dhruv Shah](https://people.eecs.berkeley.edu/~shah/), Ajay Sridhar, and [Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/?_ga=2.182963686.1720382867.1664319155-2139079238.1651157950)**


System Requirement
=================
Ubuntu 18.04

Pytorch 1.8.0

ROS MELODIC(http://wiki.ros.org/melodic)

Nvidia GPU

How to use ExAug
=================

#### Step1: Download
Code: https://github.com/NHirose/Control-Policy-ExAug.git

Trained Model: https://drive.google.com/drive/folders/1M2d7454caRXPzlXsnjZdEZLopdcndYkj?usp=share_link

#### Step2: Camera Setup
We release three control policies for following different three camera types:

spherical camera(double fisheye camera): Ricoh Theta S

fisheye image(170 degree FOV): ELP 170 degree fisheye

narrow FOV camera: Intel Realsense D435i

The control policy with the spherical camera can show the best performance. We recommend to use RICOH THETA S.(https://theta360.com/en/about/theta/s.html)
Please put the camera in front of your device(robot) at the height 0.35 m not to caputure your robot itself and connect with your PC by USB cable.

#### Step3: Publishing current image
Please publish your camera image on ROS. The topic name is defined as "/topic_name_current_image".
Our control policy is defined as the callback function of this topic. Hence, the frame rate of the image is same as the frame rate of the control policy. We recommend to be larger than 3 fps.

If you use Ricoh Theta S, please check [our previous work](https://github.com/NHirose/DVMPC) and follow how to use it.

#### Step4: Publishing subgoal image
ExAug generates the velocity commands to go to the goal position from the current and subgoal image. Our code subscribes the subgoal image as "/topic_name_goal_image".
By updating the subgoal image, you can control the robot toward the far goal position.

#### Step5: Runing DVMPC
We can run our control policy as follows.

python3 ExAug.py --camera "360" --rsize 0.3 --rsize_t 0.3

Here, "--camera" is the camera type, "--rsize" is the robot radius for collision avoidance, and "--rsize_t" is the robot radius for traversability estimation.
To be conservative, you can set larger robot size than the actual size. Our code publishs the velocity commands as "/cmd_vel".

License
=================
The codes provided on this page are published under the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License(https://creativecommons.org/licenses/by-nc-sa/3.0/). This means that you must attribute the work in the manner specified by the authors, you may not use this work for commercial purposes and if you alter, transform, or build upon this work, you may distribute the resulting work only under the same license. If you are interested in commercial usage you can contact us for further options. 

Citation
=================

If you use DVMPC's software or database, please cite:

@article{hirose2022exaug,  
  title={ExAug: Robot-Conditioned Navigation Policies via Geometric Experience Augmentation},  
  author={Hirose, Noriaki and Shah, Dhruv and Sridhar, Ajay and Levine, Sergey},  
  journal={arXiv preprint arXiv:2210.07450},  
  year={2022}  
}

