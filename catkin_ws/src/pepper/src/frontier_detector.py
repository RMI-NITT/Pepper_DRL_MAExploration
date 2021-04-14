#!/usr/bin/env python


#--------Include modules---------------
import rospy
from nav_msgs.msg import OccupancyGrid

import numpy as np
import cv2

#-----------------------------------------------------

def getfrontier(mapData):
	global pub
	#global map_topic
	data=mapData.data
	w=mapData.info.width
	h=mapData.info.height

    #Copying infos to frontier_map
	frontier_map=OccupancyGrid()
	frontier_map.header = mapData.header
	frontier_map.info = mapData.info
	
	#img = np.zeros((h, w, 1), np.uint8)
	#img = [[255]*w]*h
	img = np.zeros((h, w), np.uint8)
	
	for i in range(0,h):
		for j in range(0,w):
			if data[i*w+j]!=-1:
				img[i,j]=255
	
	#img = np.asarray(img, np.uint8)
	
	im2,contours, hierarchy = cv2.findContours(img,cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
	
	#ic = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
	#cv2.drawContours(ic, contours, -1, (0, 255, 0), 3)
	#cv2.imshow('window_'+map_topic,ic)
	#cv2.waitKey(10)
	
	#all_pts=[]
	size=w*h
	frontier_data=[0]*size

	for i in contours:
		for j in i:
			_index = w*j[0][1] + j[0][0]
			if data[_index] < 70:
				frontier_data[_index]=100

	frontier_map.data=frontier_data
	pub.publish(frontier_map)	
	return ()

if __name__ == '__main__':
    global pub
    global map_topic
    try:
        rospy.init_node('frontier_detector', anonymous=True)

        map_topic = rospy.get_param('~map_topic', "map")
        frontier_topic = rospy.get_param('~frontier_topic', "frontier_map")

        pub=rospy.Publisher(frontier_topic, OccupancyGrid , queue_size=2)
        rospy.Subscriber(map_topic, OccupancyGrid, getfrontier)

        rospy.spin()
    except rospy.ROSInterruptException:
		#cv2.destroyAllWindows()
		pass
