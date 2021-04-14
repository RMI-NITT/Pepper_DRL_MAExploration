#!/usr/bin/env python


#--------Include modules---------------
import rospy
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import UInt64MultiArray
#-----------------------------------------------------

def get_occ_count(mapData):
	global pub
	#global map_topic
	topic_data = UInt64MultiArray()
	data = mapData.data
    #solid_occ_count
	topic_data.data.append( data.count(100) )
	#occupied_count
	topic_data.data.append( len(data) - data.count(-1) - topic_data.data[0] )

	pub.publish(topic_data)	
	return ()

if __name__ == '__main__':
    global pub
    global map_topic
    try:
        rospy.init_node('occupancy_counter', anonymous=True)

        map_topic = rospy.get_param('~map_topic', "map")
        occupancy_topic = rospy.get_param('~occupancy_count_topic', "occupancy_count")

        pub=rospy.Publisher(occupancy_topic, UInt64MultiArray , queue_size=2)
        rospy.Subscriber(map_topic, OccupancyGrid, get_occ_count)

        rospy.spin()
    except rospy.ROSInterruptException:
		pass
