#!/usr/bin/env python

import rospy
import tf
from nav_msgs.msg import OccupancyGrid

_bot_count=3
_bot_ns="tb3_"
_base_link_frame="base_footprint"
_map_frame="map"
fsl=0.50 #footprint side length

def mainprocess(_map):
    _width=_map.info.width
    _height=_map.info.height
    _resolution=_map.info.resolution
    _map_data=list(_map.data)
    try:
        for bot_n in range(_bot_count):
            base_frame = "/" + _bot_ns + str(bot_n) + "/" + _base_link_frame
            (trans,rot) = listener.lookupTransform("map", base_frame, rospy.Time(0))
            _X=trans[0]+10.05
            _Y=trans[1]+10.05	#As the map data's origin is the right side bottom, while the map frame's origin is observed to be close to the center of the map
            lx=int( (_X-fsl)/_resolution )
            ux=int( (_X+fsl)/_resolution )
            ly=int( (_Y-fsl)/_resolution )
            uy=int( (_Y+fsl)/_resolution )
            for y in range(ly,uy):
                for x in range(lx,ux):
                    _map_data[(y*_width)+x]=0
        _map.data=tuple(_map_data)
        pub.publish(_map)

    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        pass

if __name__ == '__main__':
    try:
        rospy.init_node('footprint_clear', anonymous=True)
        listener = tf.TransformListener()
        pub = rospy.Publisher('/map', OccupancyGrid , queue_size=10)
        rospy.Subscriber('/merged_map', OccupancyGrid, mainprocess)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass    
