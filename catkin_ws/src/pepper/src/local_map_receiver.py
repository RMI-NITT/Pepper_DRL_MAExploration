#!/usr/bin/env python

import rospy
import tf
from nav_msgs.msg import OccupancyGrid, Odometry
import math
import numpy as np
import copy
import time
import threading
from std_srvs.srv import Trigger

merged_map=OccupancyGrid()
map_list=[]
_started=False
map_object=[]
odom_list=[]
odom_object=[]
map_pending=[]
map_merge_pending=[]
merge_pending=False
maps_merged=True

def return_merge_status(req):
    global _bot_count
    global map_pending
    global map_merge_pending
    global merge_pending
    global maps_merged
    global merge_timeout

    _success=False
    _time_goal=rospy.Time.now() + rospy.Duration(merge_timeout) #Setting (0.5*bot_count) seconds as timeout
    while rospy.Time.now() < _time_goal:
        time.sleep(0.01)
        if _started:
            maps_merged=True
            for i in range(_bot_count):
                if map_pending[i] or map_merge_pending[i]:
                    maps_merged=False
                    break
            
            if (not maps_merged):
                continue
            
            if merge_pending or maps_merged==False:
                continue
            
            _success=True
            break
    
    if _success:
        maps_merged=False
        return {'success': True , 'message': "success"}
    else:
        maps_merged=False
        #print("timeout_Agent_"+str(_bot_id)+" : "+str(map_pending) )
        return {'success': False , 'message': "timeout"}

def local_map_sub(_local_map):
    global _bot_id
    global _bot_count
    global map_list
    global map_object
    global _started
    global pub
    global merged_map
    global map_length
    global odom_list
    global odom_object
    global map_pending
    global merge_pending
    global map_merge_pending

    if _started==False:

        merged_map.header.seq=0
        merged_map.header.frame_id=_local_map.header.frame_id

        merged_map.info=copy.deepcopy(_local_map.info)

        map_length = merged_map.info.width*merged_map.info.height
        init_data = np.ones(map_length,np.int8)*-1
        merged_map.data = init_data.copy()

        for i in range(_bot_count):
            odom_list.append(0)
            map_pending.append(False)
            map_merge_pending.append(False)
            odom_object.append(odom_Subscriber(i))

        time.sleep(5)

        for id in range(_bot_count):
            if id==_bot_id:
                if id==0:
                    map_list = np.array([_local_map.data],np.int8)
                else:
                    map_list = np.vstack( (map_list , np.array(_local_map.data,np.int8) ) )
                
                map_pending[_bot_id]=False
                map_merge_pending[_bot_id]=True
                merge_pending=True

                continue
            if id==0:
                map_list = np.array([init_data])
            else:
                map_list = np.vstack( (map_list , init_data) )
            map_object.append(map_Subscriber(id))

        _started=True

    else:
        map_pending[_bot_id]=True
        map_list[_bot_id]=np.array(_local_map.data, np.int8)
        map_merge_pending[_bot_id]=True
        merge_pending=True
        map_pending[_bot_id]=False

    return ()

def pub_map():
    global map_list
    global pub
    global _bot_count
    global _started
    global pub_rate
    global map_length
    #global map_pending
    global maps_merged
    global merge_pending
    global _bot_id
    global map_merge_pending

    while _started==False:
        time.sleep(0.1)

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
                maps_merged=True
                pub_rate.sleep()

            sleep_count=0
            while (merge_pending==False) and (not rospy.is_shutdown()):
                time.sleep(0.01)
                sleep_count = sleep_count + 1
                if sleep_count == 400:
                    pub.publish(merged_map) #If nothing is published for the past 4 secs, then re-publish last map
                    sleep_count=0
            
            maps_merged=False
            time.sleep(0.01)    #Waiting for more maps to be received
            for i in range(_bot_count):
                if map_merge_pending[i]:
                    map_merge_pending[i]=False
            #print("Agent_"+str(_bot_id)+" :"+str(map_pending))
                
            merged_map.header.seq = merged_map.header.seq + 1

        except rospy.ROSInterruptException:
            break
    return ()

class map_Subscriber():
    def __init__(self,_agent_id):
        global _bot_ns
        global id_offset

        self.agent_id=_agent_id
        rospy.Subscriber("/"+_bot_ns+str(self.agent_id+id_offset)+"/map", OccupancyGrid, self.callback)

    def callback(self,_map):
        global map_list
        global _bot_id
        global _bot_ns
        global _max_comm_dist
        global odom_list
        global map_pending
        global map_merge_pending
        global merge_pending

        distance=math.sqrt( ( ( odom_list[self.agent_id][0] - odom_list[_bot_id][0] )**2)+( ( odom_list[self.agent_id][1] - odom_list[_bot_id][1] )**2) )
        #print("Agent_"+str(_bot_id)+"-"+str(self.agent_id)+" :"+str(distance) )

        if distance <= _max_comm_dist:
            map_pending[self.agent_id]=True
            if not np.array_equal( map_list[self.agent_id] , _map.data ):
                map_list[self.agent_id]=np.array( _map.data, np.int8)
                map_merge_pending[self.agent_id]=True
                merge_pending=True
            map_pending[self.agent_id]=False

        return ()

class odom_Subscriber():
    def __init__(self,_agent_id):
        global  _bot_ns
        global id_offset
        self.agent_id=_agent_id
        self.last_pos=(-1,-1)

        rospy.Subscriber("/"+_bot_ns+str(_agent_id+id_offset)+"/odom", Odometry, self.callback)

    def callback(self,_odom):
        global odom_list
        #global map_pending
        curr_pos=(_odom.pose.pose.position.x , _odom.pose.pose.position.y)
        
        if curr_pos!=self.last_pos:
            odom_list[self.agent_id]=curr_pos
            #map_pending[self.agent_id]=True
            self.last_pos=curr_pos
        
        return ()

if __name__ == '__main__':
    global _bot_id
    global _bot_count
    global _max_comm_dist
    global _bot_ns
    global pub
    global pub_rate
    global merge_timeout
    global id_offset
    id_offset=0
    
    try:
        rospy.init_node('local_map_transmitter', anonymous=True)

        _bot_id = rospy.get_param('~bot_id', 0+id_offset)
        _bot_count = rospy.get_param('~bot_count', 1)
        _max_comm_dist = rospy.get_param('~max_comm_dist', 3.0)
        _bot_ns = rospy.get_param('~bot_ns', "Env_0/Agent_")

        merge_timeout = 1.0*_bot_count
        _bot_id = _bot_id - id_offset

        pub=rospy.Publisher('map', OccupancyGrid , queue_size=2)
        
        rospy.Subscriber('local_map', OccupancyGrid, local_map_sub)

        pub_rate=rospy.Rate(20)
        pub_thread = threading.Thread( target=pub_map, args=() )
        pub_thread.start()

        #Init Merge Status Feedback Service for the Environment to call
        service = rospy.Service("/" + _bot_ns + str(_bot_id + id_offset) + "/merge_status", Trigger, return_merge_status)

        rospy.spin()
    except rospy.ROSInterruptException:
        pass    
