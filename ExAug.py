#!/usr/bin/env python
# -*- coding: utf-8 -*-
# From Seigo Ito topic no doki

import os

#ROS
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, CompressedImage
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError

#PIL
from PIL import ImageDraw
from PIL import Image as PILImage

#NN model
from polinet import PoliNet_exaug

#torch
import torch
import torch.nn.functional as F

#others
import yaml
import cv2
import sys
import time
import numpy as np
import math
from datetime import date

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--camera', type=str, default="360", help="camera type, 360 or fisheye or rsense")
parser.add_argument("--rsize", type=float, help="robot size", default=0.3)
parser.add_argument("--rsize_t", type=float, help="robot size for traversability estimation", default=0.3)
args = parser.parse_args()

image360 = False
fisheye = False
rsense = False

if args.camera == "360":
    image360 = True
elif args.camera == "fisheye":
    fisheye = True
elif args.camera == "rsense":
    rsense = True

if image360:
    #for 360 image
    print("Input image is 360 image!!")
    model_file_exaug = os.path.join(os.path.dirname(__file__), 'nn_model_exaug/polinet_360.pth')
elif fisheye:
    #for fisheye
    print("Input image is fisheye image!!")
    model_file_exaug = os.path.join(os.path.dirname(__file__), 'nn_model_exaug/polinet_fisheye.pth')
elif rsense:
    #for realsense
    print("Input image is Realsense image!!")
    model_file_exaug = os.path.join(os.path.dirname(__file__), 'nn_model_exaug/polinet_realsense.pth')

# resize parameters
rsizex = 128
rsizey = 128

#mask for 360 degree image
mask_br360 = np.loadtxt(open(os.path.join(os.path.dirname(__file__), "utils/mask_360view.csv"), "rb"), delimiter=",", skiprows=0)
mask_brr = mask_br360.reshape((1,1,128,256)).astype(np.float32)
mask_brr1 = mask_br360.reshape((1,128,256)).astype(np.float32)
mask_brrc = np.concatenate((mask_brr1, mask_brr1, mask_brr1), axis=0)

#mask for fisheye degree image
mask_recon = np.zeros((1, 128, 416), dtype = 'float32')
center_h = 0.5*128 - 0.5
center_w = 0.5*416 - 0.5
for i in range(416):
    for j in range(128):
        if ((i - center_w)**2)/(0.5*416*0.95)**2 + ((j - center_h)**2)/(0.5*128*1.5)**2 <= 1:
            mask_recon[0,j,i] = 1.0                
mask_recon_batch = torch.from_numpy(mask_recon).float().clone().to("cpu").unsqueeze(0).repeat(1, 3, 1, 1)
mask_recon_polinet = F.interpolate(mask_recon_batch, (128, 128), mode='bilinear', align_corners=False)

i = 0
j = 0


def unddp_state_dict(state_dict):
    if not all([s.startswith("module.") for s in state_dict.keys()]):
        return state_dict

    return OrderedDict((k[7:], v) for k, v in state_dict.items())

def preprocess_image_360(msg):
    cv_img = bridge.imgmsg_to_cv2(msg)
    cv_resize_n = cv2.resize(cv_img, (2*rsizex, rsizey), cv2.INTER_AREA) 
    cv_resizex = cv_resize_n.transpose(2, 0, 1)
    in_imgcc1 = np.array([cv_resizex], dtype=np.float32)
    in_img1 = (in_imgcc1 - 127.5)/127.5

    img_nn_cL = mask_brrc * (in_img1 + 1.0) -1.0 #mask
    img = img_nn_cL.astype(np.float32)

    return img, cv_resize_n

def preprocess_image(msg):
    cv_img = bridge.imgmsg_to_cv2(msg)
    cv_resize_n = cv2.resize(cv_img, (rsizex, rsizey), cv2.INTER_AREA)
    cv_resizex = cv_resize_n.transpose(2, 0, 1)
    in_imgcc1 = np.array([cv_resizex], dtype=np.float32)
    in_img1 = (in_imgcc1 - 127.5)/127.5

    img_nn_cL = (in_img1 + 1.0) -1.0 #mask
    img = img_nn_cL.astype(np.float32)

    return img

def callback_rsense(msg_1):
    global j
    global vwkeep

    j = j + 1

    if j == 1:
        #preprocess of current image
        cur_img = preprocess_image(msg_1)
        xcg = torch.clamp(torch.from_numpy(cur_img).to(device), -1.0, 1.0)

        #preprocess of subgoal image
        #standard deviation and mean for current image
        imgbc = (np.reshape(cur_img[0][0],(1,128,128)) + 1.0)*0.5
        imggc = (np.reshape(cur_img[0][1],(1,128,128)) + 1.0)*0.5
        imgrc = (np.reshape(cur_img[0][2],(1,128,128)) + 1.0)*0.5
        mean_cbgr = np.zeros((3,1))
        std_cbgr = np.zeros((3,1))
        mean_ct = np.zeros((3,1))
        std_ct = np.zeros((3,1))
        mean_cbgr[0] = np.sum(imgbc)/countm
        mean_cbgr[1] = np.sum(imggc)/countm
        mean_cbgr[2] = np.sum(imgrc)/countm
        std_cbgr[0] = np.sqrt(np.sum(np.square(imgbc-mean_cbgr[0]))/countm)
        std_cbgr[1] = np.sqrt(np.sum(np.square(imggc-mean_cbgr[1]))/countm)
        std_cbgr[2] = np.sqrt(np.sum(np.square(imgrc-mean_cbgr[2]))/countm)

        #standard deviation and mean for subgoal image
        imgrt = (np.reshape(goal_img[0][0],(1,128,128)) + 1)*0.5
        imggt = (np.reshape(goal_img[0][1],(1,128,128)) + 1)*0.5
        imgbt = (np.reshape(goal_img[0][2],(1,128,128)) + 1)*0.5
        mean_tbgr = np.zeros((3,1))
        std_tbgr = np.zeros((3,1))
        mean_tbgr[0] = np.sum(imgbt)/countm
        mean_tbgr[1] = np.sum(imggt)/countm
        mean_tbgr[2] = np.sum(imgrt)/countm
        std_tbgr[0] = np.sqrt(np.sum(np.square(imgbt-mean_tbgr[0]))/countm)
        std_tbgr[1] = np.sqrt(np.sum(np.square(imggt-mean_tbgr[1]))/countm)
        std_tbgr[2] = np.sqrt(np.sum(np.square(imgrt-mean_tbgr[2]))/countm)

        mean_ct[0] = mean_cbgr[0]
        mean_ct[1] = mean_cbgr[1]
        mean_ct[2] = mean_cbgr[2]
        std_ct[0] = std_cbgr[0]
        std_ct[1] = std_cbgr[1]
        std_ct[2] = std_cbgr[2]

        imgrtt = (imgrt-mean_tbgr[0])/std_tbgr[0]*std_ct[0]+mean_ct[0]
        imggtt = (imggt-mean_tbgr[1])/std_tbgr[1]*std_ct[1]+mean_ct[1]
        imgbtt = (imgbt-mean_tbgr[2])/std_tbgr[2]*std_ct[2]+mean_ct[2]

        goalt_img = np.array((np.reshape(np.concatenate((imgrt, imggt, imgbt), axis = 0), (1,3,128,128)) - 0.5)*2.0, dtype=np.float32)
        timage = torch.clamp(torch.from_numpy(goalt_img).to(device), -1.0, 1.0)

        #current image
        xcgx = xcg

        #subgoal image
        xpgb = timage

        #robot size
        #for control policy
        robot_size = args.rsize*torch.ones(1, 1, 1, 1).to(device)
        #for traversability estimation
        robot_sizef = args.rsize_t*torch.ones(1, 1, 1, 1).to(device)

	#initial values for integration
        px = torch.zeros(1).to(device)
        pz = torch.zeros(1).to(device)
        ry = torch.zeros(1).to(device)

        with torch.no_grad():
            vwres, ptrav, ptravf = polinet(torch.cat((xcgx, xpgb), axis=1), robot_size, robot_sizef, px, pz, ry)


        msg_pub = Twist()
        #msg_raw = Twist()

        vt = vwres.cpu().numpy()[0,0,0,0]
        wt = vwres.cpu().numpy()[0,1,0,0]

        #TODO fixing traversability estimation. Now 1.0: untraversable, 0.0: traversable 
        if ptravf[0,0] > 0.5:
            vt = 0.0 #if untraversable, we can stop to move.
        
        maxv = 0.2
        maxw = 0.4

        #For safety navigation, we limit the velocity under keeping the required turning radius.
        if np.absolute(vt) < maxv:
            if np.absolute(wt) < maxw:
                msg_pub.linear.x = vt
                msg_pub.linear.y = 0.0
                msg_pub.linear.z = 0.0
                msg_pub.angular.x = 0.0
                msg_pub.angular.y = 0.0
                msg_pub.angular.z = wt
            else:
                rd = vt/wt
                msg_pub.linear.x = maxw * np.sign(vt) * np.absolute(rd)
                msg_pub.linear.y = 0.0
                msg_pub.linear.z = 0.0
                msg_pub.angular.x = 0.0
                msg_pub.angular.y = 0.0
                msg_pub.angular.z = maxw * np.sign(wt)
        else:
            if np.absolute(wt) < 0.001:
                msg_pub.linear.x = maxv * np.sign(vt)
                msg_pub.linear.y = 0.0
                msg_pub.linear.z = 0.0
                msg_pub.angular.x = 0.0
                msg_pub.angular.y = 0.0
                msg_pub.angular.z = 0.0
            else:
                rd = vt/wt
                if np.absolute(rd) > maxv / maxw:
                    msg_pub.linear.x = maxv * np.sign(vt)
                    msg_pub.linear.y = 0.0
                    msg_pub.linear.z = 0.0
                    msg_pub.angular.x = 0.0
                    msg_pub.angular.y = 0.0
                    msg_pub.angular.z = maxv * np.sign(wt) / np.absolute(rd)
                else:
                    msg_pub.linear.x = maxw * np.sign(vt) * np.absolute(rd)
                    msg_pub.linear.y = 0.0
                    msg_pub.linear.z = 0.0
                    msg_pub.angular.x = 0.0
                    msg_pub.angular.y = 0.0
                    msg_pub.angular.z = maxw * np.sign(wt)

        msg_out.publish(msg_pub)
        j = 0

def callback_fisheye(msg_1):
    global j

    j = j + 1
    if j == 1:
        #preprocess of current image
        cur_img = preprocess_image(msg_1)
        xcg = torch.clamp(torch.from_numpy(cur_img).to(device), -1.0, 1.0)

        #preprocess of subgoal image
        #standard deviation and mean for current image
        imgbc = (np.reshape(cur_img[0][0],(1,128,128)) + 1.0)*0.5
        imggc = (np.reshape(cur_img[0][1],(1,128,128)) + 1.0)*0.5
        imgrc = (np.reshape(cur_img[0][2],(1,128,128)) + 1.0)*0.5
        mean_cbgr = np.zeros((3,1))
        std_cbgr = np.zeros((3,1))
        mean_ct = np.zeros((3,1))
        std_ct = np.zeros((3,1))
        mean_cbgr[0] = np.sum(imgbc)/countm
        mean_cbgr[1] = np.sum(imggc)/countm
        mean_cbgr[2] = np.sum(imgrc)/countm
        std_cbgr[0] = np.sqrt(np.sum(np.square(imgbc-mean_cbgr[0]))/countm)
        std_cbgr[1] = np.sqrt(np.sum(np.square(imggc-mean_cbgr[1]))/countm)
        std_cbgr[2] = np.sqrt(np.sum(np.square(imgrc-mean_cbgr[2]))/countm)

        #standard deviation and mean for subgoal image
        imgrt = (np.reshape(goal_img[0][0],(1,128,128)) + 1)*0.5
        imggt = (np.reshape(goal_img[0][1],(1,128,128)) + 1)*0.5
        imgbt = (np.reshape(goal_img[0][2],(1,128,128)) + 1)*0.5
        mean_tbgr = np.zeros((3,1))
        std_tbgr = np.zeros((3,1))
        mean_tbgr[0] = np.sum(imgbt)/countm
        mean_tbgr[1] = np.sum(imggt)/countm
        mean_tbgr[2] = np.sum(imgrt)/countm
        std_tbgr[0] = np.sqrt(np.sum(np.square(imgbt-mean_tbgr[0]))/countm)
        std_tbgr[1] = np.sqrt(np.sum(np.square(imggt-mean_tbgr[1]))/countm)
        std_tbgr[2] = np.sqrt(np.sum(np.square(imgrt-mean_tbgr[2]))/countm)

        mean_ct[0] = mean_cbgr[0]
        mean_ct[1] = mean_cbgr[1]
        mean_ct[2] = mean_cbgr[2]
        std_ct[0] = std_cbgr[0]
        std_ct[1] = std_cbgr[1]
        std_ct[2] = std_cbgr[2]

        imgrtt = (imgrt-mean_tbgr[0])/std_tbgr[0]*std_ct[0]+mean_ct[0]
        imggtt = (imggt-mean_tbgr[1])/std_tbgr[1]*std_ct[1]+mean_ct[1]
        imgbtt = (imgbt-mean_tbgr[2])/std_tbgr[2]*std_ct[2]+mean_ct[2]

        goalt_img = np.array((np.reshape(np.concatenate((imgrtt, imggtt, imgbtt), axis = 0), (1,3,128,128)) - 0.5)*2.0, dtype=np.float32)
        timage = torch.clamp(torch.from_numpy(goalt_img).to(device), -1.0, 1.0)

        #current image
        xcgx = 2.0*(mask_recon_polinet*(0.5*xcg+0.5)-0.5)

        #subgoal image
        xpgb = 2.0*(mask_recon_polinet*(0.5*timage+0.5)-0.5)

        #robot size
        #for control policy
        robot_size = args.rsize*torch.ones(1, 1, 1, 1).to(device)
        #for traversability estimation
        robot_sizef = args.rsize_t*torch.ones(1, 1, 1, 1).to(device)

	#initial values for integration
        px = torch.zeros(1).to(device)
        pz = torch.zeros(1).to(device)
        ry = torch.zeros(1).to(device)

        with torch.no_grad():
            vwres, ptrav, ptravf = polinet(torch.cat((xcgx, xpgb), axis=1), robot_size, robot_sizef, px, pz, ry)

        msg_pub = Twist()

        vt = vwres.cpu().numpy()[0,0,0,0]
        wt = vwres.cpu().numpy()[0,1,0,0]

        #TODO fixing traversability estimation. Now 1.0: untraversable, 0.0: traversable 
        if ptravf[0,0] > 0.5:
            vt = 0.0 #if untraversable, we can stop to move.

        maxv = 0.2
        maxw = 0.4

        #For safety navigation, we limit the velocity under keeping the required turning radius.
        if np.absolute(vt) < maxv:
            if np.absolute(wt) < maxw:
                msg_pub.linear.x = vt
                msg_pub.linear.y = 0.0
                msg_pub.linear.z = 0.0
                msg_pub.angular.x = 0.0
                msg_pub.angular.y = 0.0
                msg_pub.angular.z = wt
            else:
                rd = vt/wt
                msg_pub.linear.x = maxw * np.sign(vt) * np.absolute(rd)
                msg_pub.linear.y = 0.0
                msg_pub.linear.z = 0.0
                msg_pub.angular.x = 0.0
                msg_pub.angular.y = 0.0
                msg_pub.angular.z = maxw * np.sign(wt)
        else:
            if np.absolute(wt) < 0.001:
                msg_pub.linear.x = maxv * np.sign(vt)
                msg_pub.linear.y = 0.0
                msg_pub.linear.z = 0.0
                msg_pub.angular.x = 0.0
                msg_pub.angular.y = 0.0
                msg_pub.angular.z = 0.0
            else:
                rd = vt/wt
                if np.absolute(rd) > maxv / maxw:
                    msg_pub.linear.x = maxv * np.sign(vt)
                    msg_pub.linear.y = 0.0
                    msg_pub.linear.z = 0.0
                    msg_pub.angular.x = 0.0
                    msg_pub.angular.y = 0.0
                    msg_pub.angular.z = maxv * np.sign(wt) / np.absolute(rd)
                else:
                    msg_pub.linear.x = maxw * np.sign(vt) * np.absolute(rd)
                    msg_pub.linear.y = 0.0
                    msg_pub.linear.z = 0.0
                    msg_pub.angular.x = 0.0
                    msg_pub.angular.y = 0.0
                    msg_pub.angular.z = maxw * np.sign(wt)

        msg_out.publish(msg_pub)
        j = 0

def callback_360(msg_1):
    global j

    j = j + 1
    if j == 1:
        #preprocess of current image
        cur_img, cur_img_raw = preprocess_image_360(msg_1) #current image

        #preprocess of subgoal image
        imgrc = (np.reshape(cur_img[0][0],(1,128,256)) + 1.0)*0.5
        imggc = (np.reshape(cur_img[0][1],(1,128,256)) + 1.0)*0.5
        imgbc = (np.reshape(cur_img[0][2],(1,128,256)) + 1.0)*0.5
        mean_cbgr = np.zeros((3,1))
        std_cbgr = np.zeros((3,1))
        mean_ct = np.zeros((3,1))
        std_ct = np.zeros((3,1))
        mean_cbgr[0] = np.sum(imgrc)/countm
        mean_cbgr[1] = np.sum(imggc)/countm
        mean_cbgr[2] = np.sum(imgbc)/countm
        std_cbgr[0] = np.sqrt(np.sum(np.square(imgrc-mask_brr1*mean_cbgr[0]))/countm)
        std_cbgr[1] = np.sqrt(np.sum(np.square(imggc-mask_brr1*mean_cbgr[1]))/countm)
        std_cbgr[2] = np.sqrt(np.sum(np.square(imgbc-mask_brr1*mean_cbgr[2]))/countm)

        #standard deviation and mean for subgoal image
        imgrt = (np.reshape(goal_img[0][0],(1,128,256)) + 1)*0.5
        imggt = (np.reshape(goal_img[0][1],(1,128,256)) + 1)*0.5
        imgbt = (np.reshape(goal_img[0][2],(1,128,256)) + 1)*0.5
        mean_tbgr = np.zeros((3,1))
        std_tbgr = np.zeros((3,1))
        mean_tt = np.zeros((3,1))
        std_tt = np.zeros((3,1))
        mean_tbgr[0] = np.sum(imgrt)/countm
        mean_tbgr[1] = np.sum(imggt)/countm
        mean_tbgr[2] = np.sum(imgbt)/countm
        std_tbgr[0] = np.sqrt(np.sum(np.square(imgrt-mask_brr1*mean_tbgr[0]))/countm)
        std_tbgr[1] = np.sqrt(np.sum(np.square(imggt-mask_brr1*mean_tbgr[1]))/countm)
        std_tbgr[2] = np.sqrt(np.sum(np.square(imgbt-mask_brr1*mean_tbgr[2]))/countm)

        mean_ct[0] = mean_cbgr[0]
        mean_ct[1] = mean_cbgr[1]
        mean_ct[2] = mean_cbgr[2]
        std_ct[0] = std_cbgr[0]
        std_ct[1] = std_cbgr[1]
        std_ct[2] = std_cbgr[2]

        curc_img = np.array((np.reshape(np.concatenate((mask_brr*imgrc, mask_brr*imggc, mask_brr*imgbc), axis = 0), (1,3,128,256)) - 0.5)*2.0, dtype=np.float32)
        cimage = torch.clamp(torch.from_numpy(curc_img).to(device), -1.0, 1.0)


        imgrtt = (imgrt-mean_tbgr[0])/std_tbgr[0]*std_ct[0]+mean_ct[0]
        imggtt = (imggt-mean_tbgr[1])/std_tbgr[1]*std_ct[1]+mean_ct[1]
        imgbtt = (imgbt-mean_tbgr[2])/std_tbgr[2]*std_ct[2]+mean_ct[2]
        goalt_img = np.array((np.reshape(np.concatenate((mask_brr*imgrtt, mask_brr*imggtt, mask_brr*imgbtt), axis = 0), (1,3,128,256)) - 0.5)*2.0, dtype=np.float32)
        timage = torch.clamp(torch.from_numpy(goalt_img).to(device), -1.0, 1.0)

        #current image
        xcgf = cimage[:, :, :, 0:rsizex]
        xcgb = cimage[:, :, :, rsizex:2*rsizex]
        xcgx = torch.cat((xcgf.flip(1), xcgb.flip(1)),axis=1)

        #subgoal image
        xpgf = timage[:, :, :, 0:rsizex]
        xpgb = timage[:, :, :, rsizex:2*rsizex]
        xpgx = torch.cat((xpgf.flip(1), xpgb.flip(1)),axis=1)

        #robot size
        #for control policy
        robot_size = args.rsize*torch.ones(1, 1, 1, 1).to(device)
        #for traversability estimation
        robot_sizef = args.rsize_t*torch.ones(1, 1, 1, 1).to(device)

        px = torch.zeros(1).to(device)
        pz = torch.zeros(1).to(device)
        ry = torch.zeros(1).to(device)

        with torch.no_grad():
            vwres, ptrav, ptravf = polinet(torch.cat((xcgx, xpgx), axis=1), robot_size, robot_sizef, px, pz, ry)

        msg_pub = Twist()

        vt = vwres.cpu().numpy()[0,0,0,0]
        wt = vwres.cpu().numpy()[0,1,0,0]

        #TODO fixing traversability estimation. Now 1.0: untraversable, 0.0: traversable 
        if ptravf[0,0] > 0.5:
            vt = 0.0 #if untraversable, we can stop to move.

        maxv = 0.2
        maxw = 0.4

        #For safety navigation, we limit the velocity under keeping the required turning radius.
        if np.absolute(vt) < maxv:
            if np.absolute(wt) < maxw:
                msg_pub.linear.x = vt
                msg_pub.linear.y = 0.0
                msg_pub.linear.z = 0.0
                msg_pub.angular.x = 0.0
                msg_pub.angular.y = 0.0
                msg_pub.angular.z = wt
            else:
                rd = vt/wt
                msg_pub.linear.x = maxw * np.sign(vt) * np.absolute(rd)
                msg_pub.linear.y = 0.0
                msg_pub.linear.z = 0.0
                msg_pub.angular.x = 0.0
                msg_pub.angular.y = 0.0
                msg_pub.angular.z = maxw * np.sign(wt)
        else:
            if np.absolute(wt) < 0.001:
                msg_pub.linear.x = maxv * np.sign(vt)
                msg_pub.linear.y = 0.0
                msg_pub.linear.z = 0.0
                msg_pub.angular.x = 0.0
                msg_pub.angular.y = 0.0
                msg_pub.angular.z = 0.0
            else:
                rd = vt/wt
                if np.absolute(rd) > maxv / maxw:
                    msg_pub.linear.x = maxv * np.sign(vt)
                    msg_pub.linear.y = 0.0
                    msg_pub.linear.z = 0.0
                    msg_pub.angular.x = 0.0
                    msg_pub.angular.y = 0.0
                    msg_pub.angular.z = maxv * np.sign(wt) / np.absolute(rd)
                else:
                    msg_pub.linear.x = maxw * np.sign(vt) * np.absolute(rd)
                    msg_pub.linear.y = 0.0
                    msg_pub.linear.z = 0.0
                    msg_pub.angular.x = 0.0
                    msg_pub.angular.y = 0.0
                    msg_pub.angular.z = maxw * np.sign(wt)

        msg_out.publish(msg_pub)
        j = 0


def callback_ref_360(msg):
    global goal_img
    goal_img, _ = preprocess_image_360(msg) #subgoal image
    #goal_img = imgmsg_to_numpy_360(msg) #subgoal image

def callback_ref_fisheye(msg):
    global goal_img
    goal_img, _ = preprocess_image(msg) #subgoal image
    #goal_img = imgmsg_to_numpy(msg) #subgoal image
    
def callback_ref_realsense(msg):
    global goal_img
    goal_img, _ = preprocess_image(msg) #subgoal image
    #goal_img = imgmsg_to_numpy(msg) #subgoal image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

step_size = 8
lower_bounds = [0.0, -1.0]
upper_bounds = [+0.5, +1.0]

if image360:
    polinet = PoliNet_exaug(12, step_size, lower_bounds, upper_bounds).to(device)
elif fisheye:
    polinet = PoliNet_exaug(6, step_size, lower_bounds, upper_bounds).to(device)
elif rsense:
    polinet = PoliNet_exaug(6, step_size, lower_bounds, upper_bounds).to(device)

polinet.load_state_dict(unddp_state_dict(torch.load(model_file_exaug, map_location=device)))
polinet.eval()

bridge = CvBridge()

if image360:
    goal_img = np.zeros((1,3,128,256), dtype=np.float32)
else:
    goal_img = np.zeros((1,3,128,128), dtype=np.float32)

# counting the number of pixel with color information
countm = 0
for it in range(128):
    for jt in range(256):
        if mask_brr[0][0][it][jt] > 0.5:
            countm += 1

if image360 is False:
    countm = 128*128 

print(countm)
mask_c = np.concatenate((mask_brr, mask_brr, mask_brr), axis=1)


# main function
if __name__ == '__main__':

    #initialize node
    rospy.init_node('ExAug', anonymous=False)

    #subscribe of topics
    if image360:
        msg1_sub = rospy.Subscriber('/topic_name_current_image', Image, callback_360, queue_size=1)
        msg2_sub = rospy.Subscriber('/topic_name_goal_image', Image, callback_ref_360, queue_size=1)
    elif fisheye:
        #print("kiteruyone??")
        msg1_sub = rospy.Subscriber('/topic_name_current_image', Image, callback_fisheye, queue_size=1)
        msg2_sub = rospy.Subscriber('/topic_name_goal_image', Image, callback_ref_fisheye, queue_size=1)
    elif rsense:
        msg1_sub = rospy.Subscriber('/topic_name_current_image', Image, callback_rsense, queue_size=1)
        msg2_sub = rospy.Subscriber('/topic_name_goal_image', Image, callback_ref_realsense, queue_size=1)

    #publisher of topics
    msg_out = rospy.Publisher('/cmd_vel', Twist,queue_size=1) #velocities for the robot control

    print('waiting message .....')
    rospy.spin()
