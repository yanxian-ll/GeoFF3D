import rosbag
import rospy
import numpy as np
import csv
import os
import cv2
import gtsam
import json
from cv_bridge import CvBridge

######## user params ##########
RGB_TOPIC_NAME = "/device_0/sensor_1/Color_0/image/data"


SAVE_LOC = "/home/<user>/cam_ws/imgs"
BAG_LOC = "/home/<user>/Documents/"
bag = "20250417_141216"


downsample = 8


data_bag = rosbag.Bag(BAG_LOC+bag+".bag")

# make folder structure
os.makedirs(SAVE_LOC + "/" + bag)
os.makedirs(SAVE_LOC + "/" + bag + "/images")


count = 0
index = 0

timestamps = []

for topic, msg, t in data_bag.read_messages(topics=[RGB_TOPIC_NAME]):
    if topic == RGB_TOPIC_NAME:
        count += 1
        if count%downsample==0: 
            timestamps.append(msg.header.stamp)
            bridge = CvBridge()
            cv_image = bridge.imgmsg_to_cv2(msg)
            cv_image.astype(np.uint8)
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(SAVE_LOC + "/" + bag + "/images/" + "rgb_" + str(index) + '.jpg', cv_image)
            

            index += 1

cv2.destroyAllWindows()

