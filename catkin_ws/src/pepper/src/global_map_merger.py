#!/usr/bin/env python

import rospy
import tf
from nav_msgs.msg import OccupancyGrid, Odometry
import math
import numpy as np
import copy
import time
import threading

merged_map=OccupancyGrid()
map_list=[]
map_object=[]
merge_pending=False

def pub_map():
    global map_list
    global pub
    global _bot_count
    global pub_rate
    global map_length
    global merge_pending

    while not rospy.is_shutdown():
        try:
            if merge_pending:
                merge_pending=False
                map_time=rospy.get_rostime()
                merged_map.header.stamp.secs=map_time.secs
                merged_map.header.stamp.nsecs=map_time.nsecs
                
                #Map merging statement
                merged_map.data = map_list.max(axis=0)

                pub.publish(merged_map)
                pub_rate.sleep()

            sleep_count=0
            while (merge_pending==False) and (not rospy.is_shutdown()):
                time.sleep(0.01)
                sleep_count = sleep_count + 1
                if sleep_count == 200:
                    pub.publish(merged_map) #If nothing is published for the past 2 secs, then re-publish last map
                    sleep_count=0
            
            time.sleep(0.02)    #Waiting for more maps to be received
                
            merged_map.header.seq = merged_map.header.seq + 1

        except rospy.ROSInterruptException:
            break
    return ()

class map_Subscriber():
    def __init__(self,_agent_id):
        global _bot_ns
        global id_offset

        self.agent_id=_agent_id
        rospy.Subscriber("/"+_bot_ns+str(self.agent_id+id_offset)+"/local_map", OccupancyGrid, self.callback)

    def callback(self,_map):
        global map_list
        global merge_pending

        map_list[self.agent_id]=np.array( _map.data, np.int8)
        merge_pending=True

        return ()

if __name__ == '__main__':
    global _bot_count
    global _bot_ns
    global pub
    global pub_rate
    global id_offset
    global map_length
    id_offset=0
    
    try:
        rospy.init_node('local_map_transmitter', anonymous=True)

        _bot_count = rospy.get_param('~bot_count', 1)
        _bot_ns = rospy.get_param('~bot_ns', "Env_0/Agent_")
        _merged_map_topic = rospy.get_param('~merged_map_topic', "global_map")
        _global_frame = rospy.get_param('~global_frame', "Env_0/map")
        _resolution = rospy.get_param('~resolution', 0.1)
        _width = rospy.get_param('~width', 256)
        _height = rospy.get_param('~height', 256)
        _x_origin = rospy.get_param('~x_origin', 0.0)
        _y_origin = rospy.get_param('~y_origin', 0.0)

        #Init merged_map topic values
        merged_map.header.seq=0
        merged_map.header.frame_id=_global_frame

        merged_map.info.map_load_time.secs = 0
        merged_map.info.map_load_time.nsecs = 0
        merged_map.info.resolution = _resolution
        merged_map.info.width = _width
        merged_map.info.height = _height

        merged_map.info.origin.position.x = _x_origin
        merged_map.info.origin.position.y = _y_origin
        merged_map.info.origin.position.z = 0.0
        merged_map.info.origin.orientation.x = 0.0
        merged_map.info.origin.orientation.y = 0.0
        merged_map.info.origin.orientation.z = 0.0
        merged_map.info.origin.orientation.w = 1.0

        map_length = _width * _height
        init_data = np.ones(map_length,np.int8)*-1
        merged_map.data = init_data.copy()
        #merged_map topic init over

        time.sleep(5)

        for id in range(_bot_count):
            if id==0:
                map_list = np.array([init_data])
            else:
                map_list = np.vstack( (map_list , init_data) )
            
            map_object.append(map_Subscriber(id))

        pub=rospy.Publisher(_merged_map_topic, OccupancyGrid , queue_size=2)

        pub_rate=rospy.Rate(20)
        pub_thread = threading.Thread( target=pub_map, args=() )
        pub_thread.start()

        rospy.spin()
    except rospy.ROSInterruptException:
        pass    
