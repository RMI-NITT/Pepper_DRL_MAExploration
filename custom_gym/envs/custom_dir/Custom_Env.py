import gym
import cv2
import random
import numpy as np
import math

import rospy
from sensor_msgs.msg import LaserScan
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Odometry, Path, OccupancyGrid
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped, PoseStamped, PointStamped
from std_msgs.msg import UInt64MultiArray
from std_srvs.srv import Trigger
from nav_msgs.srv import GetPlan
from hector_nav_msgs.srv import GetRobotTrajectory

from move_base_msgs.msg import MoveBaseActionGoal, MoveBaseActionResult, MoveBaseActionFeedback
from actionlib_msgs.msg import GoalID, GoalStatusArray, GoalStatus

import time
import threading
import subprocess
import multiprocessing
import os, signal
from func_timeout import func_timeout

def flood_fill_single(im, seed_point):
    """Perform a single flood fill operation.

    # Arguments
        image: an image. the image should consist of white background, black lines and black fills.
               the white area is unfilled area, and the black area is filled area.
        seed_point: seed point for trapped-ball fill, a tuple (integer, integer).
    # Returns
        an image after filling.
    """
    pass1 = np.full(im.shape, 255, np.uint8)

    im_inv = cv2.bitwise_not(im)

    mask1 = cv2.copyMakeBorder(im_inv, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)
    size, pass1, _, _ = cv2.floodFill(pass1, mask1, seed_point, 0, 0, 0, 4)

    return size , pass1

def max_flood(im):
	
	res=im.shape
	max=0
	img4=im
	for i in range(res[0]):
		for j in range(res[1]):
			if img4[i,j]==255:
				size, pass2 = flood_fill_single(im, (j,i) )
				
				img4=img4&pass2
				
				if(size > max):
					max=size
					img3=pass2
	return max , img3

def laser_scan(scan_ranges,laser_gen_map,agent_pos,env_size,block_side,scan_min,scan_max,angle_steps,Dr):

    def map_check(x,y):
		if x<env_size[0] and x>=0 and y<env_size[1] and y>=0:
			return laser_gen_map[int(x*env_size[1] + y)]
		else:
			return (0)

    bsh = 0.5*block_side
    _index=0
    for i in range(0,360,angle_steps):
        sn=np.sin(np.radians(i))
        cs=np.cos(np.radians(i))
        r=scan_min	#Minimum r value to start increment from in below while loop
        prev_DX=math.floor((bsh-r*sn)/block_side)
        prev_DY=math.floor((r*cs+bsh)/block_side)
        while r<scan_max:
            DX=math.floor((bsh-r*sn)/block_side)
            DY=math.floor((r*cs+bsh)/block_side)

            if map_check(agent_pos[0]+DX,agent_pos[1]+DY)==0:
                r=r-Dr
                for j in range(1,2):
                    SDr=Dr/pow(10,j)
                    for k in range(10):
                        r=r+SDr
                        DX=math.floor((bsh-r*sn)/block_side)
                        DY=math.floor((r*cs+bsh)/block_side)

                        if(map_check(agent_pos[0]+DX,agent_pos[1]+DY)==0):
                            r=r-SDr
                            break
                break

            elif prev_DX!=DX and prev_DY!=DY:
                if( map_check(agent_pos[0]+prev_DX,agent_pos[1]+DY)==0 or map_check(agent_pos[0]+DX,agent_pos[1]+prev_DY)==0 ):
                    r=r-Dr
                    for j in range(1,2):
                        SDr=Dr/pow(10,j)
                        for k in range(10):
                            r=r+SDr
                            DX=math.floor((bsh-r*sn)/block_side)
                            DY=math.floor((r*cs+bsh)/block_side)

                            if prev_DX!=DX and prev_DY!=DY:
                                r=r-SDr
                                break
                    break

            r=r+Dr
            prev_DX=DX
            prev_DY=DY
        
        if r>= scan_max:
            r=float('inf')
				
        scan_ranges[_index]=r
        _index = _index + 1
		
    return ()

class CustomEnv(gym.Env):
	def __init__(self, env_id=0, size=(32,32), obstacle_density=20, n_agents=1, rrt_exp=False, rrt_mode=0, agents_exp=[0], global_map_exp=False, global_planner=False, laser_range=14.0, max_comm_dist=7.5, nav_recov_timeout=2.0, render_output=False, scaling_factor=20):
		print("Env_"+str(env_id)+" Initializing")
		self.env_ns = "Env_"+str(env_id)	#Environment ROS Namespace

		#Start roscore
		#roscore_t = threading.Thread( target=self.roscore_thread, args=() )
		#roscore_t.start()

		#self.viewer = None
		#self.agent_spawn = True
		
		#Minimum Size Recovery - for compatibility with gmapping package
		_size=[]
		if size[0] >= 4:
			_size.append(int(size[0]))
		else:
			print("Warning: Environment Size must have a minimum value of 4. Resetting size[0] to 4.")
			_size.append(4)
		if size[1] >= 4:
			_size.append(int(size[1]))
		else:
			print("Warning: Environment Size must have a minimum value of 4. Resetting size[1] to 4.")
			_size.append(4)

		#Environment map parameters		
		self.size = tuple(_size)
		self.obstacle_density = float(obstacle_density)/100
		self.scaling_factor = scaling_factor
		self.block_side=0.4			#1 Block side Length in meters
		self.map_pixel_size = 0.1	#1 gmapping map's pixel's side Length in meters
		self.agents_count= n_agents
		self.laser_range = float(laser_range*self.block_side)
		self.max_comm_dist = float(max_comm_dist)
		self.id_offset=0
		self.render_output = render_output

		#Exploration & Navigation parameters
		# Dictionary for labeling exploration packages
		self.Exp_id = {'explore_lite':0, 'rrt_exploration':1, 'hector_exploration':2, 'free_navigation':3, 'free_move':4}
		self.rrt_exp = rrt_exp
		self.rrt_mode = rrt_mode
		self.agents_exp = agents_exp
		self.global_map_exp = global_map_exp
		self.hect_comp_thresh = 0.975	#Hector exploration stops if (exp_prog reaches this value or above) and (the hector_exploration package returns a nav path of zero legth)
		self.elite_thresh = 0.975		#explore_lite exploration stops if this threshold is reached
		self.rrt_thresh = 0.975			#rrt exploration stops if this threshold is reached
		self.hector_planner_timeout = 10  # in seconds - Time Limit for hector get_exploration_path service to return path. If timedout, then hector_exploration_node will be restarted
		self.global_planner = global_planner
		self.move_base_goal_tolerance=0.0				#If goal is unreachable max tolerance (in meters) to make it reachable
		self.move_base_retry_timeout=nav_recov_timeout	#in seconds - Retries timeout on path planning failure
		self.move_base_failed_retries=3					#Maximum times to retry for invalid goal
		self.global_planner_timeout=1500				#in ms - higher values are needed for larger environments bigger than 64x64 blocks

		if render_output:
			cv2.namedWindow(self.env_ns)

		#Generating Random Map
		self.map_gen()
		#Find largest contour in map for future agents spawning and to move within
		max_size, self.img2 = max_flood(self.img)

		self.largest_contour=[]
		for i in range(self.size[0]):
			for j in range(self.size[1]):
				if(self.img2[i,j]==0):
					self.img[i,j]=127
					self.largest_contour.append((i,j))
		#Assigining random starting positions for all the agents from the largest contour
		self.agents_init_poses = random.sample(self.largest_contour,self.agents_count)

		#Calculate no. of occupyable pixels for exp_progress function in agents class
		self.occupiable_pixels = len(self.largest_contour)*(self.block_side/self.map_pixel_size)**2

		#Resizing image for diplay (init code for self.render() )
		self.resized_img=np.zeros((self.size[0]*self.scaling_factor,self.size[1]*self.scaling_factor, 3),dtype=np.uint8)
		#res = cv2.resize(self.img, (512, 512))

		for i in range(0,self.size[0]*self.scaling_factor,self.scaling_factor):
			for j in range(0,self.size[1]*self.scaling_factor,self.scaling_factor):
				shade=self.img[int(i/self.scaling_factor),int(j/self.scaling_factor)]
				self.resized_img[i:i+self.scaling_factor,j:j+self.scaling_factor]=(shade,shade,shade)

		#Initializing ROS Node
		#rospy.init_node("PEPPER_Env_"+str(env_id), anonymous=True)

		#Flag variables
		self.env_alive = True	#This variable is used to exit from threads while closing this environment
		self.agents_initialized = False		#This variable prevents certain agent threads from starting before all agents are initialized

		#Time Parameters
		#self.start_time=time.time()
		self._clock = Clock()
		self.pub_clock = rospy.Publisher('/clock', Clock, queue_size=10)
		#self.clock_rate = rospy.Rate(100) # 100hz

		self.clk_pub = threading.Thread( target=self.clock_pub, args=() )
		self.clk_pub.start()

		#Initilizing TF Publishers
		self.pub_tf_static = rospy.Publisher('/tf_static', TFMessage, queue_size=100)
		self.tf_static_rate = rospy.Rate(self.agents_count) # 1hz per agent
		self.pub_tf = rospy.Publisher('/tf', TFMessage, queue_size=1000)
		self.tf_rate = rospy.Rate(2*self.agents_count) # 2 hz per agent

		#Subscribing to global_map to find global exploration progress & frontier map for user to know the possible nav goals for exploration
		self.map_enlarge_factor = float(self.block_side)/self.map_pixel_size
		self.map_width = int(32.0*math.floor(2.0*self.map_enlarge_factor*self.size[0]/32.0))	#Floor with /32, since gmapping takes 32px as its minimum step as map grows
		self.map_height = int(32.0*math.floor(2.0*self.map_enlarge_factor*self.size[1]/32.0))	#Also gmapping will enlarge the map to twice its actual size
		self.map_len = self.map_width * self.map_height
		block_offset = (0.5*self.block_side)/self.map_pixel_size
		self.map_offset = [0.25*self.map_width + block_offset - 1, 0.25*self.map_height + block_offset - 1]
		self.map = [-1]*self.map_len			#Initializing a map_data with unknown pixels
		self.frontier_map = [0]*self.map_len	#Initializing a map with zeros - denotes no frontier present
		self.exp_prog = 0.0
		self.sub_map = rospy.Subscriber("/" + self.env_ns + "/global_map", OccupancyGrid, self.map_sub)
		# self.frontier_sub = rospy.Subscriber("/" + self.env_ns + "/frontier_map", OccupancyGrid, self.frontier_map_sub)
		self.exp_prog_sub = rospy.Subscriber("/" + self.env_ns + "/occupancy_count", UInt64MultiArray, self.exp_progress)

		#Map_Merging process - Global launch
		self.map_merge_init()

		#CREATE AGENT OBJECTS & SPAWN THEM IN MAP
		self.agent = [0]*self.agents_count
		agent_init_threads=[]
		for id in range(self.agents_count):
			agent_init_threads.append(threading.Thread( target=self.init_agent, args=[id] ))
			agent_init_threads[id].start()
		for id in range(self.agents_count):
			agent_init_threads[id].join()
		self.agents_initialized = True

		#Boundary points for exploration (for self-starting)
		if self.rrt_exp and (rrt_mode!=1):
			self.publish_points()

		print("Env_"+str(env_id)+" Initialized")

	def publish_points(self):
		pub = rospy.Publisher("/" + self.env_ns + "/clicked_point", PointStamped, queue_size=10)
		pub_rate = rospy.Rate(5)
		point = PointStamped()
		point.header.frame_id = self.env_ns+"/map"
		point.point.z = 0

		rospy.sleep(2)

		point.header.seq = 0
		point.header.stamp.secs = self._clock.clock.secs
		point.header.stamp.nsecs = self._clock.clock.nsecs
		point.point.x = -0.4*self.block_side*self.size[0]
		point.point.y = -0.4*self.block_side*self.size[1]
		pub_rate.sleep()
		pub.publish(point)

		point.header.seq = 1
		point.header.stamp.secs = self._clock.clock.secs
		point.header.stamp.nsecs = self._clock.clock.nsecs
		point.point.x = 1.4*self.block_side*self.size[0]
		pub_rate.sleep()
		pub.publish(point)

		point.header.seq = 2
		point.header.stamp.secs = self._clock.clock.secs
		point.header.stamp.nsecs = self._clock.clock.nsecs
		point.point.y = 1.4*self.block_side*self.size[1]
		pub_rate.sleep()
		pub.publish(point)

		point.header.seq = 3
		point.header.stamp.secs = self._clock.clock.secs
		point.header.stamp.nsecs = self._clock.clock.nsecs
		point.point.x = -0.4*self.block_side*self.size[0]
		pub_rate.sleep()
		pub.publish(point)

		point.header.seq = 4
		point.header.stamp.secs = self._clock.clock.secs
		point.header.stamp.nsecs = self._clock.clock.nsecs
		mid_lc = self.largest_contour[int(0.5*len(self.largest_contour))]
		point.point.x = mid_lc[0]*self.block_side
		point.point.y = mid_lc[1]*self.block_side
		pub_rate.sleep()
		pub.publish(point)
		
		return ()

	def map_merge_init(self):
		m_ln0="roslaunch"
		m_ln1="pepper"
		m_ln2="env_map_merge_global.launch"
		env_ns="env_ns:="+self.env_ns
		n_arg="n_agents:="+str(self.agents_count)
		width_arg="width:="+str(self.map_width)
		height_arg="height:="+str(self.map_height)
		delta_arg = "mapping_delta:="+str(0.10)
		x_origin="x_origin:="+str(-0.5*self.block_side*self.size[0])
		y_origin="y_origin:="+str(-0.5*self.block_side*self.size[1])
		if self.rrt_exp:
				start_rrt="start_rrt:=true"
				if self.rrt_mode == 0:
					global_rrt="global_rrt:=true"
					opencv_rrt="opencv_rrt:=false"
				elif (self.rrt_mode == 1) or (self.rrt_mode == 2):
					global_rrt="global_rrt:=false"
					opencv_rrt="opencv_rrt:=true"
				else:
					global_rrt="global_rrt:=true"
					opencv_rrt="opencv_rrt:=true"
		else:
			start_rrt="start_rrt:=false"
			global_rrt="global_rrt:=true"
			opencv_rrt="opencv_rrt:=false"

		map_merge_args = [m_ln0,m_ln1,m_ln2, env_ns, delta_arg, width_arg, height_arg, x_origin, y_origin, n_arg, start_rrt, global_rrt, opencv_rrt]
		self.map_merge_sp=subprocess.Popen(map_merge_args, stdout=subprocess.PIPE, close_fds=True)
		rospy.sleep(10)
		
		return ()

	def clock_pub(self):
		last_time=rospy.get_time() #Gets rostime in seconds as float
		while not rospy.is_shutdown() and self.env_alive:
			#clk=time.time()-self.start_time
			#self._clock.clock.secs=int(clk)
			#self._clock.clock.nsecs=int((clk-float(self._clock.clock.secs))*1000000000.0)
			#self.clock_rate.sleep()

			while rospy.get_time() - last_time < 0.01:
				time.sleep(0.001)

			clk=rospy.get_rostime()
			self._clock.clock.secs=clk.secs
			self._clock.clock.nsecs=clk.nsecs
			
			self.pub_clock.publish(self._clock)
			
			last_time=last_time+0.01

	def init_agent(self,id):
		self.agent[id] = agent( self,id,self.agents_init_poses[id] )
		return ()

	def map_gen(self):
		self.img=np.ones(self.size,dtype=np.uint8)
		self.img=self.img*255

		x=random.sample(range(self.size[0]*self.size[1]),int(self.size[0]*self.size[1]*self.obstacle_density))

		for i in x:
			self.img[int(i/self.size[1]),i%self.size[1]]=0
		return ()

	def map_sub(self,_map):
		self.map = _map.data
		return ()

	"""
	def frontier_map_sub(self,_map):
		self.frontier_map = _map.data
		#print("Global Frontier_map obtained")
		return ()
	"""
	
	def exp_progress(self,occ_topic):
		solid_occ_count = occ_topic.data[0]
		occupied_count = occ_topic.data[1]
		self.exp_prog = (occupied_count+ (0.5*solid_occ_count) ) / self.occupiable_pixels
		#print( "Global_Progress: "+str(self.exp_prog) )
		return ()

	def step(self):
		step_t=[]
		for id in range(self.agents_count):
			step_t.append( threading.Thread( target = self.agent[id].step, args=() ) )
			step_t[id].start()
		for id in range(self.agents_count):
			step_t[id].join()
			#print(str(id)+") Pos:"+str(self.agent[id].goal_pos)+" Pixel:"+str(self.agent[id].goal_pixel))
		
		return ()

	def render(self):
		if self.render_output:
			output_img=self.resized_img.copy()
			#Show all agents
			for id in range(self.agents_count):
				agent_center = ( int( (self.agent[id].agent_pos[1]+0.5)*self.scaling_factor), int( (self.agent[id].agent_pos[0]+0.5)*self.scaling_factor) )
				output_img = cv2.circle(output_img, agent_center , int(self.scaling_factor*0.5) , self.agent[id].color , -1 ) 

			cv2.imshow(self.env_ns,output_img)
		return ()

	def reset(self, n_agents=1, rrt_exp=True, rrt_mode=0, agents_exp=[], global_map_exp=False, global_planner=False, laser_range=14.0, max_comm_dist=7.5, nav_recov_timeout=2.0):
		#Remove alive agents and their child nodes
		self.close_agents()
		self.agents_count= n_agents
		self.rrt_exp = rrt_exp
		self.rrt_mode = rrt_mode
		self.agents_exp = agents_exp
		self.global_map_exp = global_map_exp
		self.laser_range = float(laser_range*self.block_side)
		self.max_comm_dist = float(max_comm_dist)
		self.global_planner = global_planner
		self.move_base_retry_timeout=nav_recov_timeout

		#Re-Assigining random starting positions for all the agents from the largest contour
		self.agents_init_poses = random.sample(self.largest_contour,self.agents_count)

		#Re-Subscribing to global_map & frontier map
		self.map = [-1]*self.map_len			#Initializing a map_data with unknown pixels
		self.frontier_map = [0]*self.map_len	#Initializing a map with zeros - denotes no frontier present
		self.exp_prog = 0.0
		self.exp_prog_sub = rospy.Subscriber("/" + self.env_ns + "/global_map", OccupancyGrid, self.exp_progress)
		# self.frontier_sub = rospy.Subscriber("/" + self.env_ns + "/frontier_map", OccupancyGrid, self.frontier_map_sub)

		#Relaunching Map_Merging process - Global launch
		self.map_merge_init()

		#RE-CREATING AGENT OBJECTS & SPAWN THEM IN MAP
		self.agent = [0]*self.agents_count
		agent_init_threads=[]
		for id in range(self.agents_count):
			agent_init_threads.append(threading.Thread( target=self.init_agent, args=[id] ))
			agent_init_threads[id].start()
		for id in range(self.agents_count):
			agent_init_threads[id].join()

		#Replotting Boundary points for exploration (for self-starting)
		if self.rrt_exp and (self.rrt_mode!=1):
			self.publish_points()

		print(self.env_ns+" Reset Successfully")
		return ()

	def close_agents(self):
		for i in range(self.agents_count):
			self.agent[i].agent_alive=False

			self.agent[i].sub_map.unregister()
			# self.agent[i].frontier_sub.unregister()
			self.agent[i].exp_prog_sub.unregister()

			#print("SIGINT VALUE: "+str(self.agent[i].gmapping_sp.pid))
			os.kill(self.agent[i].gmapping_sp.pid , signal.SIGINT)
			#Delete hect_nav objects
			if self.agent[i].active_exp == self.Exp_id['hector_exploration']:
				del self.agent[i].hect_nav
			#Delete move_base objects
			if self.global_planner:
				if self.agent[i].move_base.action_active:
					self.agent[i].move_base.sub_goal.unregister()
					self.agent[i].move_base.sub_cancel.unregister()
				self.agent[i].move_base.sub_simple_goal.unregister()
				self.agent[i].move_base.sub_global_planner.unregister()
				self.agent[i].move_base.get_plan_srv.shutdown()
				del self.agent[i].move_base

		#print("SIGINT VALUE: "+str(self.map_merge_sp.pid))
		os.kill(self.map_merge_sp.pid , signal.SIGINT)
		self.sub_map.unregister()
		# self.frontier_sub.unregister()
		self.exp_prog_sub.unregister()
		rospy.sleep(2)
		for i in range(self.agents_count):
			del self.agent[0]

		print("Successully removed agents from "+self.env_ns)
		return ()

	def close(self):
		self.close_agents()
		if self.render_output:
			cv2.destroyWindow(self.env_ns)
		self.env_alive=False
		return ()

	def __del__(self):
		print("Env_Deleted")

class agent():
	def __init__(self,_env,_agent_id,_init_pos):
		self.env=_env
		self.agent_id=_agent_id
		self.group_ns=_env.env_ns+"/Agent_"+str(_agent_id + _env.id_offset)
		self.agent_alive=True
		self.exp_completed=False	#If this is set True then step() function won't do anything and just return False if self.active_exp is anything apart from "free_navigation"
		self.step_ret=False	 #This variable is updated during every self.step() execution. [True: (1 step of nav movement is done) or (new path is obtained)], [(False: ( (no nav step) and (no new path) ) or (goal cancelled/completed)]
		self.goal_pos=[-1.0,-1.0]		#Position of goal set by exp_package or user as coordinates of the environment (1 unit = 1 block side) - [-1,-1] denotes unknown
		self.goal_pixel=[-1,-1]			#Position of goal as coordinates on gmapping's map (1 unit = 1 pixel size on map) - [-1,-1] denotes unknown
		self.map = [-1]*_env.map_len			#Initializing a map_data with unknown pixels
		self.frontier_map = [0]*_env.map_len	#Initializing a map with zeros - denotes no frontier present
		if _env.rrt_exp:
			self.active_exp = _env.Exp_id['rrt_exploration']
		elif _env.agents_exp[_agent_id] == 0:
			self.active_exp = _env.Exp_id['explore_lite']
		elif _env.agents_exp[_agent_id] == 1:
			self.active_exp = _env.Exp_id['hector_exploration']
		elif _env.agents_exp[_agent_id] == 2:
			self.active_exp = _env.Exp_id['free_navigation']
		else:
			self.active_exp = _env.Exp_id['free_move']
		print("Initializing Agent_"+str(_agent_id))
		#Spawn agent
		self.agent_pos=[_init_pos[0],_init_pos[1]]	#The variable that always contains the current agents position
		self.agent_pixel=[ int(_init_pos[0]*self.env.map_enlarge_factor + self.env.map_offset[0]), int(_init_pos[1]*self.env.map_enlarge_factor + self.env.map_offset[1]) ]	#This always contains the agents pixels coordinate on gmapping map
		self.color= ( random.randint(0,255),random.randint(0,255),random.randint(0,255) )	#Colour of agent on opencv window
		
		#Init Last known positions of other agents
		self.last_known_pixel=[]	#Contains the pixel coordinate of the all agents within communication range (has [-1,-1] if an agent is not within range)
		for id in range(self.env.agents_count):
			if id==_agent_id:
				self.last_known_pixel.append(self.agent_pixel)
				continue
			self.last_known_pixel.append([-1,-1])	#[-1,-1] refers to unknown position

		#Init set_goal_pixel parameters
		self.simple_goal = PoseStamped()	#For usage in set_goal_pixel() function
		self.simple_goal.header.seq = 0
		self.simple_goal.header.frame_id = self.env.env_ns + "/map"
		self.simple_goal.pose.position.z = 0.0
		self.simple_goal.pose.orientation.x=0.0
		self.simple_goal.pose.orientation.y=0.0
		w_orientation = z_orientation = -1.0/math.sqrt(2)	#We have assumed that the bot is always facing the +ve y axis in this code
		self.simple_goal.pose.orientation.z = z_orientation
		self.simple_goal.pose.orientation.w = w_orientation

		#Start gmapping node
		self.gmapping_sb_init()
		#time.sleep(2.0)
		#Initializing Odom
		self.agent_spawn_pos=self.agent_pos[:]

		self.pub_odom = rospy.Publisher(self.group_ns + "/odom", Odometry, queue_size=1000)
		#self.odom_rate = rospy.Rate(5) # 30hz

		self._odom = Odometry()
		self._odom.header.seq=0
		self._odom.header.frame_id = self.group_ns + "/odom"
		self._odom.child_frame_id = self.group_ns + "/base_link"
		
		self._odom.pose.pose.position.z=0.0		
		self._odom.pose.pose.orientation.x=0.0
		self._odom.pose.pose.orientation.y=0.0
		self._odom.pose.pose.orientation.z=z_orientation
		self._odom.pose.pose.orientation.w=w_orientation
		self._odom.pose.covariance=[]
		for i in range(36):
			self._odom.pose.covariance.append(0.0)

		self._odom.twist.twist.linear.x=0
		self._odom.twist.twist.linear.y=0
		self._odom.twist.twist.linear.z=0
		self._odom.twist.twist.angular.x=0
		self._odom.twist.twist.angular.y=0
		self._odom.twist.twist.angular.z=0
		self._odom.twist.covariance=[]
		for i in range(36):
			self._odom.twist.covariance.append(0.0)

		#Laser Scan Parameters
		self.laser_scan_started=0
		self.bsh=self.env.block_side*0.5
		self.angle_steps=2  #Angle precesion of Laser Scanner in Degree

		self._scan = LaserScan()
		self.pub_scan = rospy.Publisher(self.group_ns + "/scan", LaserScan, queue_size=10000)
		self.laser_rate = rospy.Rate(30) # 30hz

		self.laser_id=0
		self._scan.header.frame_id=self.group_ns + "/base_scan"
		self._scan.angle_min=0.0
		self._scan.angle_max=6.283185307179586
		self._scan.angle_increment=0.017453292519943*self.angle_steps
		self._scan.time_increment=0.0
		self._scan.scan_time=0.0
		self._scan.range_min=self.bsh
		self._scan.range_max=_env.laser_range
		self._scan.intensities=[]
		for i in range(0,360,self.angle_steps):
			self._scan.intensities.append(0.0)
		self.Dr=self.env.block_side/10.0
		self.scan_ranges = multiprocessing.Array('f',int(360/self.angle_steps))
		self.laser_gen_map = multiprocessing.Array('i',self.env.img.flatten())

		self.laser_pub = threading.Thread( target=self.laser_gen, args=() )
		self.laser_pub.start()

		#Intitializing TF Parameters
		self._tf_static=TFMessage()
		self._tf=TFMessage()

		self._tf_bf_cl=TransformStamped()
		self._tf_bf_cl.header.seq=0
		self._tf_bf_cl.header.frame_id=self.group_ns + "/base_link"
		self._tf_bf_cl.child_frame_id=self.group_ns + "/base_scan"
		self._tf_bf_cl.transform.translation.x=0.0
		self._tf_bf_cl.transform.translation.y=0.0
		self._tf_bf_cl.transform.translation.z=0.0
		self._tf_bf_cl.transform.rotation.x=0.0
		self._tf_bf_cl.transform.rotation.y=0.0
		self._tf_bf_cl.transform.rotation.z=0.0
		self._tf_bf_cl.transform.rotation.w=1.0

		self._tf_odom_bf=TransformStamped()
		self._tf_odom_bf.header.seq=0
		self._tf_odom_bf.header.frame_id=self.group_ns + "/odom"
		self._tf_odom_bf.child_frame_id=self.group_ns + "/base_link"
		#self._tf_odom_bf.transform.translation.x=0.0
		#self._tf_odom_bf.transform.translation.y=0.0
		self._tf_odom_bf.transform.translation.z=0.0
		self._tf_odom_bf.transform.rotation.x=0.0
		self._tf_odom_bf.transform.rotation.y=0.0
		self._tf_odom_bf.transform.rotation.z=self._odom.pose.pose.orientation.z
		self._tf_odom_bf.transform.rotation.w=self._odom.pose.pose.orientation.w

		self.tf_static_pub_thread = threading.Thread( target=self.tf_static_pub , args=() )
		self.tf_pub_thread = threading.Thread( target=self.tf_pub , args=() )

		self.tf_static_pub_thread.start()
		
		rospy.sleep(5)
		self.map_merge_service_init()

		#Subscribing to map to find exploration progress
		self.exp_prog = 0.0	#Contains the exploration progress from 0.0 to 1.0 at all times
		self.local_exp_prog = 0.0 #Contains the exploration progress of local map (map built without communication)
		self.sub_map = rospy.Subscriber("/" + self.group_ns + "/map", OccupancyGrid, self.map_sub)
		# self.frontier_sub = rospy.Subscriber("/" + self.group_ns + "/frontier_map", OccupancyGrid, self.frontier_map_sub)
		self.exp_prog_sub = rospy.Subscriber("/" + self.group_ns + "/occupancy_count", UInt64MultiArray, self.exp_progress)
		self.local_exp_prog_sub = rospy.Subscriber("/" + self.group_ns + "/local_occ_count", UInt64MultiArray, self.local_exp_progress)
		self.odom_update(self.agent_pos)	#This function is used to update to position of an agent to any given position
		self.tf_pub_thread.start()
		
		#if (self.active_exp == self.env.Exp_id['explore_lite']) or (self.active_exp == self.env.Exp_id['rrt_exploration']) or (self.active_exp == self.env.Exp_id['free_navigation']):
		if self.env.global_planner:
			self.move_base = Move_base(self)	#Initializing object of Move_base class
			self.move_base.feedback_pub()
		
		if self.active_exp == self.env.Exp_id['hector_exploration']:
			self.hect_nav = hector_navigation(self)
		
		print("Initialized Agent_"+str(_agent_id))

	def gmapping_sb_init(self):
		try:
			gml0="roslaunch"
			gml1="pepper"
			gml2="gmapping_simple_sim.launch"
			id_arg="agent_id:="+str(self.agent_id+self.env.id_offset)
			env_ns="env_ns:="+self.env.env_ns
			n_arg="n_agents:="+str(self.env.agents_count)
			map_size_minx="min_x:="+str(-0.5*self.env.block_side*self.env.size[0])
			map_size_miny="min_y:="+str(-0.5*self.env.block_side*self.env.size[1])
			map_size_maxx="max_x:="+str(1.5*self.env.block_side*self.env.size[0])
			map_size_maxy="max_y:="+str(1.5*self.env.block_side*self.env.size[1])
			delta_arg="mapping_delta:="+str(0.10)
			comm_range_arg="max_comm_dist:="+str(self.env.max_comm_dist*self.env.block_side)
			if self.env.global_planner:
				gp_arg="enable_global_planner:=true"
			else:
				gp_arg="enable_global_planner:=false"
			if (self.active_exp == self.env.Exp_id['rrt_exploration']):
				if self.env.rrt_mode == 1:
					exp_enable="start_rrt:=false"
				else:
					exp_enable="start_rrt:=true"
			elif self.active_exp == self.env.Exp_id['explore_lite']:
				exp_enable="start_e_lite:=true"
			elif self.active_exp == self.env.Exp_id['hector_exploration']:
				exp_enable="start_hector:=true"
			else:
				exp_enable="start_rrt:=false"
			if self.env.global_map_exp:
				map_topic="map_topic:=/"+self.env.env_ns+"/global_map"
			else:
				map_topic="map_topic:=map"

			gmapping_args = [gml0, gml1, gml2, id_arg, env_ns, n_arg, map_size_minx, map_size_miny, map_size_maxx, map_size_maxy, delta_arg, comm_range_arg, gp_arg, exp_enable, map_topic]
			self.gmapping_sp=subprocess.Popen(gmapping_args, stdout=subprocess.PIPE, close_fds=True)
		except:
			pass
		return ()

	def map_check(self,x,y):
		if x<self.env.size[0] and x>=0 and y<self.env.size[1] and y>=0:
			return self.laser_gen_map[int(x*self.env.size[1] + y)]
		else:
			return (0)

	def tf_static_pub(self):
		while not rospy.is_shutdown() and self.agent_alive:
			#self._tf_map_odom.header.stamp.secs=self._clock.clock.secs
			#self._tf_map_odom.header.stamp.nsecs=self._clock.clock.nsecs

			self._tf_bf_cl.header.stamp.secs=self.env._clock.clock.secs
			self._tf_bf_cl.header.stamp.nsecs=self.env._clock.clock.nsecs

			self._tf_static=[]
			#self._tf_static.append(self._tf_map_odom)
			self._tf_static.append(self._tf_bf_cl)

			self.env.pub_tf_static.publish(self._tf_static)
			self.env.tf_static_rate.sleep()
		return ()

	def tf_pub(self):
		while not rospy.is_shutdown() and self.agent_alive:
			if self.odom_updating==False:
				self._odom.header.stamp.secs = self.env._clock.clock.secs
				self._odom.header.stamp.nsecs = self.env._clock.clock.nsecs

				self.pub_odom.publish(self._odom)

				self._odom.header.seq = self._odom.header.seq + 1

				self._tf_odom_bf.header.stamp.secs=self._odom.header.stamp.secs
				self._tf_odom_bf.header.stamp.nsecs=self._odom.header.stamp.nsecs

				self._tf=[]
				self._tf.append(self._tf_odom_bf)

				#self._tf_bf_cl.header.stamp.secs=self._odom.header.stamp.secs
				#self._tf_bf_cl.header.stamp.nsecs=self._odom.header.stamp.nsecs

				#self._tf.append(self._tf_bf_cl)

				self.env.pub_tf.publish(self._tf)
			
			self.env.tf_rate.sleep()
		return ()

	def odom_update(self, pos):
		
		self.odom_updating=True
		self._odom.header.stamp.secs = self.env._clock.clock.secs
		self._odom.header.stamp.nsecs = self.env._clock.clock.nsecs
		self.agent_pos=pos[:]
		self.agent_pixel=[ int(pos[0]*self.env.map_enlarge_factor + self.env.map_offset[0]), int(pos[1]*self.env.map_enlarge_factor + self.env.map_offset[1]) ]

		self._odom.pose.pose.position.x=self.agent_pos[0]*self.env.block_side
		self._odom.pose.pose.position.y=self.agent_pos[1]*self.env.block_side
		#self.odom_rate.sleep()

		self.pub_odom.publish(self._odom)

		self._odom.header.seq = self._odom.header.seq + 1

		self._tf_odom_bf.header.stamp.secs=self._odom.header.stamp.secs
		self._tf_odom_bf.header.stamp.nsecs=self._odom.header.stamp.nsecs
		self._tf_odom_bf.transform.translation.x=self._odom.pose.pose.position.x
		self._tf_odom_bf.transform.translation.y=self._odom.pose.pose.position.y

		self._tf=[]
		self._tf.append(self._tf_odom_bf)

		#self._tf_bf_cl.header.stamp.secs=self._odom.header.stamp.secs
		#self._tf_bf_cl.header.stamp.nsecs=self._odom.header.stamp.nsecs

		#self._tf=[]
		#self._tf.append(self._tf_bf_cl)

		#self.enc.tf_rate.sleep()
		self.env.pub_tf.publish(self._tf)

		self.odom_updating=False
		#Generate and publish laser_scan
		self.laser_gen()
		#self.laser_pub = threading.Thread( target=self.laser_gen, args=() )
		#self.laser_pub.start()
		self.last_known_pixels_update()

		return ()

	def laser_gen(self):
		if not rospy.is_shutdown() and self.agent_alive:
			self._scan.header.stamp.secs= self._odom.header.stamp.secs
			self._scan.header.stamp.nsecs= self._odom.header.stamp.nsecs
					
			self._scan.header.seq=self.laser_id

			scan_process = multiprocessing.Process (target=laser_scan, args=(self.scan_ranges, self.laser_gen_map, self.agent_pos, self.env.size, self.env.block_side, self._scan.range_min, self._scan.range_max, self.angle_steps, self.Dr) )
			scan_process.start()
			scan_process.join()
				
			self._scan.ranges = self.scan_ranges[:]

			if not rospy.is_shutdown() and self.agent_alive:
				self.laser_rate.sleep()
				self.pub_scan.publish(self._scan)

			self.laser_id=self.laser_id+1
		return ()

	def map_merge_service_init(self):
		map_merge_srv_id="/" + self.group_ns + "/merge_status"
		rospy.wait_for_service(map_merge_srv_id)
		self.map_merge_srv = rospy.ServiceProxy(map_merge_srv_id, Trigger)
		return ()

	def map_merge_service(self):
		srv_resp = self.map_merge_srv()
		if srv_resp.success==False:
			print("Agent_" + str(self.agent_id+self.env.id_offset)+": " + srv_resp.message)
		return ()

	def map_with_feedback(self):
		print("Step: "+str(self.agent_id+self.env.id_offset))

		for x in self.env.largest_contour:
			self.odom_update([x[0],x[1]])
			self.map_merge_service()
			self.env.render()
			if cv2.waitKey(1) & 0xFF == ord('x'):
				break

		return()

	def step(self):
		if self.active_exp == self.env.Exp_id['free_navigation'] and self.env.global_planner:
			self.step_ret = self.move_base.step()
			return ()
		elif (self.active_exp == self.env.Exp_id['explore_lite']) or (self.active_exp == self.env.Exp_id['rrt_exploration']):
			if self.active_exp == self.env.Exp_id['explore_lite']:
				if self.env.global_map_exp:
					if self.env.exp_prog > self.env.elite_thresh:
						self.exp_completed = True
				else:
					if self.exp_prog > self.env.elite_thresh:
						self.exp_completed = True
			else:
				if self.env.exp_prog > self.env.rrt_thresh:
					self.exp_completed = True
			if not self.exp_completed and self.env.global_planner:
				self.step_ret = self.move_base.step()	#Returns True only if agent moved a step (Returns True even if a new path was calculated succesfully)
				return ()
			#else:
			#	print(str(self.agent_id)+") Exp_Completed")
		elif self.active_exp == self.env.Exp_id['hector_exploration']:
			if not self.exp_completed:
				self.step_ret = self.hect_nav.step()		#Returns True only if agent moved a step (Returns True even if a new path was calculated succesfully)
				return ()
		elif self.active_exp == self.env.Exp_id['free_move']:
			# ENTER THE CODE FOR RL HERE
			self.step_ret = True
			return ()
		
		self.step_ret = False
		return ()

	def observe(self):
		merged_map = np.reshape(self.map, (self.env.map_height,self.env.map_width)) * 0.005
		# Map -1 to 0.0 & 0 to 100 as 0.5 to 1.0
		neg_i = merged_map < 0.0
		merged_map[~neg_i] += 0.5
		merged_map[neg_i] = 0.0

		# Transforming merged map with agents position as origin
		col = self.agent_pixel[0]
		row = self.agent_pixel[1]
		width = self.env.map_width // 2
		height = self.env.map_height // 2

		transform_col = width - col
		transform_row = height - row
		min_col = max(col - width, 0)
		max_col = min(col + width, self.env.map_width)
		min_row = max(row - height, 0)
		max_row = min(row + height, self.env.map_height)

		map_network = np.full((2, self.env.map_height, self.env.map_width), 0.)

		map_network[0][(min_row + transform_row):(max_row + transform_row),
					(min_col + transform_col):(max_col + transform_col)] = merged_map[min_row:max_row, min_col:max_col]

		# Nearby agents positions
		for i in range(len(self.last_known_pixel)):
			if i != self.agent_id and self.last_known_pixel[i][0] != -1:
				c = self.last_known_pixel[i][0] + transform_col
				r = self.last_known_pixel[i][1] + transform_row
				if r >= 0 and c >= 0:
					map_network[1][r][c] = 1.0

		return map_network

	def set_goal_pixel(self,x,y):
		if not self.env.global_planner:
			print("Exception: Can't set goal pixel. Global Planner not Activated.")
			return ()
		self.simple_goal.pose.position.x = (x-self.env.map_offset[0])*self.env.map_pixel_size
		self.simple_goal.pose.position.y = (y-self.env.map_offset[1])*self.env.map_pixel_size

		self.simple_goal.header.stamp.secs = self.env._clock.clock.secs
		self.simple_goal.header.stamp.nsecs = self.env._clock.clock.nsecs

		self.move_base.simple_goal_sub(self.simple_goal)

		self.simple_goal.header.seq = self.simple_goal.header.seq + 1
		return ()

	def switch_to_free_move(self):
		if self.env.global_planner:
			if self.active_exp != self.env.Exp_id['free_move']:
				if self.move_base.action_active:
					self.move_base.sub_goal.unregister()
					self.move_base.sub_cancel.unregister()
					self.move_base.action_active = False
		self.active_exp = self.env.Exp_id['free_move']

		return ()

	def switch_to_free_nav(self):
		if not self.env.global_planner:
			print("Exception: Can't switch to free navigation mode. Global Planner not Activated.")
			return ()
		if self.active_exp != self.env.Exp_id['free_navigation']:
			if self.move_base.action_active:
				self.move_base.sub_goal.unregister()
				self.move_base.sub_cancel.unregister()
				self.move_base.action_active = False
			self.active_exp = self.env.Exp_id['free_navigation']

		return ()

	def switch_to_exp(self):
		if not self.env.global_planner:
			if self.env.agents_exp[self.agent_id] == 1:
				self.hect_nav.nip = False
				self.active_exp = self.env.Exp_id['hector_exploration']
			elif self.env.agents_exp[self.agent_id] == 3:
				self.active_exp = self.env.Exp_id['free_move']
			else:
				print("Exception: Can't switch to selected mode. Global Planner not Activated.")
			return ()
		if not self.move_base.action_active:
			if not (self.env.agents_exp[self.agent_id] == 1 or self.env.agents_exp[self.agent_id] > 2):
				self.move_base.sub_goal = rospy.Subscriber(self.move_base.ns+"/goal", MoveBaseActionGoal, self.move_base.goal_sub)
				self.move_base.sub_cancel = rospy.Subscriber(self.move_base.ns+"/cancel", GoalID, self.move_base.cancel_sub)
				self.move_base.action_active = True

		if self.env.rrt_exp:
			self.active_exp = self.env.Exp_id['rrt_exploration']
		elif self.env.agents_exp[self.agent_id] == 0:
			self.active_exp = self.env.Exp_id['explore_lite']
		elif self.env.agents_exp[self.agent_id] == 1:
			self.hect_nav.nip = False
			self.active_exp = self.env.Exp_id['hector_exploration']
		elif self.env.agents_exp[self.agent_id] == 2:
			self.active_exp = self.env.Exp_id['free_navigation']
		else:
			self.active_exp = self.env.Exp_id['free_move']
		
		return ()

	def last_known_pixels_update(self):
		if self.env.agents_initialized:
			own_pos=self.agent_pos
			for id in range(self.env.agents_count):
				if id==self.agent_id:
					self.last_known_pixel[id]=self.agent_pixel
					continue
				dist = math.sqrt( ((self.env.agent[id].agent_pos[0]-own_pos[0])**2)+((self.env.agent[id].agent_pos[1]-own_pos[1])**2) )	
				self.last_known_pixel[id] = self.env.agent[id].agent_pixel if (dist<=self.env.max_comm_dist) else [-1,-1]
			#print(str(self.agent_id)+" ) Last_known_pixels: "+str(self.last_known_pixel))
		return()

	def map_sub(self,_map):
		self.map = _map.data
		return ()

	"""
	def frontier_map_sub(self,_map):
		self.frontier_map = _map.data
		#print(str(self.agent_id)+") Frontier_map obtained")
		return ()
	"""

	def exp_progress(self,occ_topic):
		solid_occ_count = occ_topic.data[0]
		occupied_count = occ_topic.data[1]
		self.exp_prog = (occupied_count+ (0.5*solid_occ_count) ) / self.env.occupiable_pixels
		#print( str(self.agent_id)+") Agent_Progress: "+str(self.exp_prog) )
		return ()

	def local_exp_progress(self,occ_topic):
		solid_occ_count = occ_topic.data[0]
		occupied_count = occ_topic.data[1]
		self.local_exp_prog = (occupied_count+ (0.5*solid_occ_count) ) / self.env.occupiable_pixels
		#print( str(self.agent_id)+") Agent_Progress: "+str(self.exp_prog) )
		return ()

class Move_base():
	def __init__(self,_bot):
		self.nip = False	#Navigation in progress = False
		self.goal_tolerance = _bot.env.move_base_goal_tolerance	#If goal is unreachable max tolerance (in meters) to make it reachable
		self.path = []
		self.path_index=0
		self.ns="/"+ _bot.group_ns+"/move_base"
		self.retry_timeout=_bot.env.move_base_retry_timeout	#in seconds
		self.max_fails=_bot.env.move_base_failed_retries	#Maximum number of retries on invalid goal error
		self.goal_processing=False
		self.step_running=False
		self.action_active=True 	#Flag variable to check if move base action topics are active
		self.bot=_bot
		self.largest_contour = _bot.env.largest_contour
		self.block_side = _bot.env.block_side
		self.path_received=False
		self.path_global=[]
		self.prev_goal = []

		self._clock = _bot.env._clock.clock

		self.result=MoveBaseActionResult()
		self.status=GoalStatusArray()
		self.feedback=MoveBaseActionFeedback()

		self.goal_status=GoalStatus()
		self.prev_goal_id=GoalID()
		self.action_goal=MoveBaseActionGoal()	#For use in move_base_simple/goal (simple_goal_sub() )

		#Init topic params
		self.result.header.seq=0

		self.status.header.seq=0

		self.feedback.header.seq=0
		self.feedback.feedback.base_position.header.seq = 0
		self.feedback.status.status=1
		self.feedback.status.text = "This goal has been accepted by the simple action server"

		self.pub_status_rate = rospy.Rate(5)	# 5Hz is the standard rate of publishing status topic
		self.list_shrink_iterations = 0 #The number of iterms in status_list always falls back from 2 to 1 in 5 seconds

		self.pub_result=rospy.Publisher(self.ns+"/result", MoveBaseActionResult , queue_size=10)
		self.pub_status=rospy.Publisher(self.ns+"/status", GoalStatusArray , queue_size=10)
		self.pub_feedback=rospy.Publisher(self.ns+"/feedback", MoveBaseActionFeedback , queue_size=10)
		self.pub_goal=rospy.Publisher("/"+ _bot.group_ns+"/global_planner/goal", PoseStamped , queue_size=10)

		self.pub_status_thread = threading.Thread( target=self.status_pub, args=() )
		self.pub_status_thread.start()	#It needs to be publishing at 5Hz always

		self.sub_goal = rospy.Subscriber(self.ns+"/goal", MoveBaseActionGoal, self.goal_sub)
		self.sub_cancel = rospy.Subscriber(self.ns+"/cancel", GoalID, self.cancel_sub)
		self.sub_simple_goal = rospy.Subscriber(self.ns+"_simple/goal", PoseStamped, self.simple_goal_sub)
		self.sub_global_planner = rospy.Subscriber("/"+ _bot.group_ns+"/global_planner/planner/plan", Path, self.global_path_sub)

		#Remapping global_planner make_plan service to move_base make_plan service
		rospy.wait_for_service("/"+_bot.group_ns+"/global_planner/planner/make_plan")
		self.make_global_plan = rospy.ServiceProxy("/"+_bot.group_ns+"/global_planner/planner/make_plan", GetPlan)
		self.get_plan_srv = rospy.Service("/"+_bot.group_ns+"/move_base_node/NavfnROS/make_plan", GetPlan, self.make_plan_service)

	def goal_sub(self,_goal):
		while self.step_running:
			time.sleep(0.001)
		self.goal_processing=True

		self.path, success = self.global_path(_goal)

		self.goal_status.goal_id.stamp.secs=_goal.goal_id.stamp.secs
		self.goal_status.goal_id.stamp.nsecs=_goal.goal_id.stamp.nsecs
		self.goal_status.goal_id.id=_goal.goal_id.id

		if success:
			last_i = len(self.path) - 1
			pos = [ self.path[last_i][0], self.path[last_i][1] ]
			self.bot.goal_pos = pos[:]
			self.bot.goal_pixel = [ int(pos[0]*self.bot.env.map_enlarge_factor + self.bot.env.map_offset[0]), int(pos[1]*self.bot.env.map_enlarge_factor + self.bot.env.map_offset[1]) ]

			if self.nip:
				#Assigning values for result topic
				self.result.header.seq = self.result.header.seq + 1
				self.result.header.stamp.secs=self._clock.secs
				self.result.header.stamp.nsecs=self._clock.nsecs

				self.result.status.goal_id.stamp.secs=self.prev_goal_id.stamp.secs
				self.result.status.goal_id.stamp.nsecs=self.prev_goal_id.stamp.nsecs
				self.result.status.goal_id.id=self.prev_goal_id.id

				self.result.status.status=2 # The goal received a cancel request after it started executing
				self.result.status.text="This goal was canceled because another goal was recieved by the simple action server"
				#Updating new result in status list too
				self.status.status_list=[]
				self.status.status_list.append(self.result.status)

				#Adding details of new goal in status list
				self.goal_status.status=1
				self.goal_status.text="This goal has been accepted by the simple action server"

				#status_list size is expected to grow from 1 to 2 here
				self.status.status_list.append(self.goal_status)
				self.list_shrink_iterations = 25	#Instructing status_pub() to remove the old status in status_list after 5 secs
				
				self.pub_result.publish(self.result)

			else:
				#Assigning values for status topic
				self.goal_status.status=1
				self.goal_status.text="This goal has been accepted by the simple action server"

				#status_list here is expected to grow from size of 1 to 2
				self.status.status_list.append(self.goal_status)
				self.list_shrink_iterations = 25
				#Start navigation
				self.nip=True

			self.feedback.status.goal_id.stamp.secs = self.prev_goal_id.stamp.secs = _goal.goal_id.stamp.secs
			self.feedback.status.goal_id.stamp.nsecs = self.prev_goal_id.stamp.nsecs = _goal.goal_id.stamp.nsecs
			self.feedback.status.goal_id.id = self.prev_goal_id.id = _goal.goal_id.id

			self.path_index=0

		else:
			self.bot.goal_pos = [-1.0, -1.0]
			self.bot.goal_pixel = [-1, -1]

			self.nip = False

			#Assigning & Publishing values for result topic stating rejection of goal
			self.result.header.seq = self.result.header.seq + 1
			self.result.header.stamp.secs=self._clock.secs
			self.result.header.stamp.nsecs=self._clock.nsecs

			self.result.status.goal_id.stamp.secs=self.prev_goal_id.stamp.secs
			self.result.status.goal_id.stamp.nsecs=self.prev_goal_id.stamp.nsecs
			self.result.status.goal_id.id=self.prev_goal_id.id
			self.result.status.status=2 # The goal received a cancel request after it started executing
			self.result.status.text="This goal was canceled because another goal was recieved by the simple action server"

			self.pub_result.publish(self.result)

			self.result.header.seq = self.result.header.seq + 1

			self.result.status.goal_id.stamp.secs=_goal.goal_id.stamp.secs
			self.result.status.goal_id.stamp.nsecs=_goal.goal_id.stamp.nsecs
			self.result.status.goal_id.id=_goal.goal_id.id
			self.result.status.status=5 # The goal was rejected, because it was unattainable or invalid (Terminal State)
			self.result.status.text="This goal was rejected because it was unattainable or invalid"

			self.pub_result.publish(self.result)

			#status list needs to be cleared and only one rejection status must be added
			self.status.status_list = []
			self.status.status_list.append(self.result.status)

		self.goal_processing = False
		return ()

	def cancel_sub(self,_cancel):
		#Wait if pub_sub is running
		while self.goal_processing:
			time.sleep(0.001)

		self.nip=False
		
		#Assigning & Publishing values for result topic stating cancelation of goal
		self.result.header.seq = self.result.header.seq + 1
		self.result.header.stamp.secs=self._clock.secs
		self.result.header.stamp.nsecs=self._clock.nsecs

		self.result.status.goal_id.stamp.secs=self.prev_goal_id.stamp.secs
		self.result.status.goal_id.stamp.nsecs=self.prev_goal_id.stamp.nsecs
		self.result.status.goal_id.id=self.prev_goal_id.id
		self.result.status.status=2 # The goal received a cancel request after it started executing
		self.result.status.text="The goal received a cancel request after it started executing"

		self.pub_result.publish(self.result)

		#status list needs to be cleared and only one cancelation status must be added
		self.status.status_list = []
		self.status.status_list.append(self.result.status)

		return ()

	def feedback_pub(self):
		#This function is to be called as soon as odom of an agent gets updated everytime

		#feedback publishes only when an ongoing navigation is active (self.nip == True)
		if self.nip:

			self.feedback.header.stamp.secs = self._clock.secs
			self.feedback.header.stamp.nsecs = self._clock.nsecs

			self.feedback.feedback.base_position.header.stamp.secs = self.bot._odom.header.stamp.secs
			self.feedback.feedback.base_position.header.stamp.nsecs = self.bot._odom.header.stamp.nsecs
			self.feedback.feedback.base_position.header.frame_id = self.bot._odom.header.frame_id

			self.feedback.feedback.base_position.pose.position.x = self.bot._odom.pose.pose.position.x
			self.feedback.feedback.base_position.pose.position.y = self.bot._odom.pose.pose.position.y
			self.feedback.feedback.base_position.pose.position.z = self.bot._odom.pose.pose.position.z
			
			self.feedback.feedback.base_position.pose.orientation.x = self.bot._odom.pose.pose.orientation.x
			self.feedback.feedback.base_position.pose.orientation.y = self.bot._odom.pose.pose.orientation.y
			self.feedback.feedback.base_position.pose.orientation.z = self.bot._odom.pose.pose.orientation.z
			self.feedback.feedback.base_position.pose.orientation.w = self.bot._odom.pose.pose.orientation.w

			self.pub_feedback.publish(self.feedback)

			self.feedback.header.seq = self.feedback.header.seq + 1
			self.feedback.feedback.base_position.header.seq = self.feedback.feedback.base_position.header.seq + 1

		return ()

	def status_pub(self):
		while not rospy.is_shutdown() and self.bot.agent_alive:

			self.status.header.stamp.secs = self._clock.secs
			self.status.header.stamp.nsecs = self._clock.nsecs
			# Delete first (old) element from status_list after 5 seconds (as implemented in official move_base package)
			if (len(self.status.status_list) == 2):
				if self.list_shrink_iterations != 0:
					self.list_shrink_iterations = self.list_shrink_iterations - 1
					if self.list_shrink_iterations == 0:
						del self.status.status_list[0]

			self.pub_status.publish(self.status)
			self.pub_status_rate.sleep()

			self.status.header.seq = self.status.header.seq + 1

		return ()

	def simple_goal_sub(self,_sgoal):
		self.action_goal.goal.target_pose = _sgoal

		self.action_goal.goal_id.stamp.secs = self._clock.secs
		self.action_goal.goal_id.stamp.nsecs = self._clock.nsecs

		self.action_goal.goal_id.id = "/move_base-1-" + str(self.action_goal.goal_id.stamp.secs) + "." + str(self.action_goal.goal_id.stamp.nsecs)

		self.goal_sub(self.action_goal)
		return ()

	def global_path_sub(self,_path):
		self.path_global = _path.poses
		self.path_received = True
		return ()

	def global_path(self,_goal):
		self.path_received = False
		self.pub_goal.publish(_goal.goal.target_pose)

		timeout=self.bot.env.global_planner_timeout
		while (self.path_received) == False:
			time.sleep(0.001)
			timeout = timeout - 1

			if timeout < 1:
				print("Timeout while getting path from Global Planner")
				self.prev_goal = _goal
				return [], False	#Denoting failure in getting path

		self.prev_goal = _goal
		len_path = len(self.path_global)

		if len_path == 0:
			if self.goal_tolerance > 0.0:
				#Find if closest reachable spot is present
				goal_pos = _goal.goal.target_pose.pose.position
				goal_xy = (goal_pos.x/self.block_side, goal_pos.y/self.block_side)
				lar_cont = self.largest_contour[:]
				lar_cont.remove( ( int(self.bot.agent_pos[0]), int(self.bot.agent_pos[1]) ) )

				min_distance = math.sqrt( (lar_cont[0][0] - goal_xy[0])**2 + (lar_cont[0][1] - goal_xy[1])**2 )
				closest_goal = lar_cont[0]
				for xy in lar_cont[1:]:
					distance = math.sqrt( (xy[0] - goal_xy[0])**2 + (xy[1] - goal_xy[1])**2 )
					if distance < min_distance:
						closest_goal = xy
						min_distance = distance
				min_distance = min_distance/self.block_side
				#print(str(self.bot.agent_id)+") Curr_pos:"+str(( int(self.bot.agent_pos[0]), int(self.bot.agent_pos[1]) ))+" Goal:"+str(closest_goal)) - for debugging
				if min_distance <= self.goal_tolerance:
					#Recalculate path with closest reachable spot
					_goal.header.seq = _goal.header.seq + 1
					_goal.header.stamp.secs = self._clock.secs
					_goal.header.stamp.nsecs = self._clock.nsecs
					goal_pos.x = self.block_side*closest_goal[0]
					goal_pos.y = self.block_side*closest_goal[1]
					
					self.path_received = False
					self.pub_goal.publish(_goal.goal.target_pose)

					timeout=1500
					while (self.path_received) == False:
						time.sleep(0.001)
						timeout = timeout - 1

						if timeout < 1:
							print("Timeout while getting path from Global Planner")
							self.prev_goal = _goal
							return [], False	#Denoting failure in getting path

					self.prev_goal = _goal
					len_path = len(self.path_global)

					if len_path == 0:
						print("Invalid goal sent to Global Planner")
						return [], False

				else:
					print("Closest reachable goal not within tolerable range")
					return [], False

			else:
				#print("Invalid goal sent to Global Planner")
				return [], False

		path = []
		#Get bot's coordinate as the first coordinate of path
		x_pos = float(self.bot.agent_pos[0])
		y_pos = float(self.bot.agent_pos[1])

		path.append( (x_pos,y_pos) )
		prev_pos = (x_pos,y_pos)
		#Convert odom pos to coordinate on environment
		for i in range(len_path):
			x_pos = math.floor( (self.path_global[i].pose.position.x + 0.5*self.block_side)/self.block_side )
			y_pos = math.floor( (self.path_global[i].pose.position.y + 0.5*self.block_side)/self.block_side )

			if x_pos < 0:
				x_pos = 0
			if y_pos < 0:
				y_pos = 0
			#If pose is same as prev_pose then skip
			if (x_pos,y_pos) == prev_pos:
				continue
			#If pose step is greater than 1 block, then interpolate
			Del_x = int(x_pos-prev_pos[0])
			Del_y = int(y_pos-prev_pos[1])
			if (abs(Del_x) > 1) or (abs(Del_y) > 1):
				if abs(Del_x) >= abs(Del_y):
					x_step = float(Del_x)/abs(Del_x)
					y_step = float(Del_y)/abs(Del_x)
					for j in range(abs(Del_x)):
						x_pos = x_pos + x_step
						y_pos = y_pos + y_step
						path.append( (x_pos,math.floor(y_pos)) )
				else:
					y_step = float(Del_y)/abs(Del_y)
					x_step = float(Del_x)/abs(Del_y)
					for j in range(abs(Del_y)):
						y_pos = y_pos + y_step
						x_pos = x_pos + x_step
						path.append( (math.floor(x_pos),y_pos) )
			else:
				path.append( (x_pos,y_pos) )
			prev_pos=(x_pos,y_pos)
		if len(path) > 1:
			del path[0]	#This deletion is done since the first point in path is just the current position of bot
		#print(str(self.bot.agent_id)+") "+str(self.bot.agent_pos)+" ) "+str(path))
		return path[:], True		#Return path and mark success

	def goal_reached(self):
		self.nip=False
	
		#Assigning & Publishing values for result topic acknowledging reaching of goal
		self.result.header.seq = self.result.header.seq + 1
		self.result.header.stamp.secs = self._clock.secs
		self.result.header.stamp.nsecs = self._clock.nsecs

		self.result.status.goal_id.stamp.secs = self.prev_goal_id.stamp.secs
		self.result.status.goal_id.stamp.nsecs = self.prev_goal_id.stamp.nsecs
		self.result.status.goal_id.id = self.prev_goal_id.id
		self.result.status.status=3 # The goal was achieved successfully
		self.result.status.text="Goal reached."

		self.pub_result.publish(self.result)

		#status list needs to be cleared and only one cancelation status must be added
		self.status.status_list = []
		self.status.status_list.append(self.result.status)

		return ()

	def path_update(self):
		self.path, success = self.global_path(self.prev_goal)	#Recalculating path
		self.path_index=0
		return success

	def make_plan_service(self,req):
		return self.make_global_plan(start = req.start,goal = req.goal,tolerance = req.tolerance)

	def step(self):
		
		while self.goal_processing:
			time.sleep(0.001)
		self.step_running = True

		#Proceed only if navigation in progress is true
		if self.nip:
			path_length=len(self.path)

			if self.path_index > path_length:
				print(str(self.bot.agent_id)+") EXCEPTION: path_index > path_length")
				self.step_running = False
				return (False)
			
			elif self.path_index < path_length:
				#print(str(self.bot.agent_id)+") path_index: "+str(self.path_index) + " path_length: "+str(path_length)+" path: "+str(self.path)) - for debugging
				target = self.path[self.path_index]

				diagonal_obstacle_pass = False
				if (target[0]!=self.bot.agent_pos[0]) and (target[1]!=self.bot.agent_pos[1]):
					if (self.bot.map_check(target[0],self.bot.agent_pos[1]) == 0) and (self.bot.map_check(self.bot.agent_pos[0],target[1])==0):
						diagonal_obstacle_pass = True

				if (self.bot.map_check(target[0],target[1]) == 0) or diagonal_obstacle_pass:
					#if self.path_index == 0:
						#print(str(self.bot.agent_id)+") Goal Cancelled due to potential Deadlock")
						#self.cancel_sub(0)
						#self.step_running = False
						#return (False)
					prev_path = self.path[:]
					failed_count=0
					retries_start=time.time()
					#print(str(self.bot.agent_id)+") Recalculating path")
					#Try to get new path until timeout or max_fails times invalid goal error
					while True:	
						success = self.path_update()	#Recalculating path
						if success == False:
							failed_count = failed_count + 1
							if failed_count <= self.max_fails:
								continue
							#print(str(self.bot.agent_id)+") Recalculating FAIL")
							self.cancel_sub(0)
							return (False)
						if self.path != prev_path:
							#print(str(self.bot.agent_id)+") Recalculating PASS")
							self.step_running = False
							return (True)
						if time.time()-retries_start > self.retry_timeout:
							break
					#print(str(self.bot.agent_id)+") Goal Cancelled due to potential Deadlock")
					self.cancel_sub(0)
					self.step_running = False
					return (False)
				
				#print(str(self.bot.agent_id)+") Inside Step")
				self.bot.odom_update(target)
				#print(str(self.bot.agent_id)+") Updated Odom")
				self.feedback_pub()
				#print(str(self.bot.agent_id)+") Updated Feedback")
				self.bot.map_merge_service()
				#print(str(self.bot.agent_id)+") Map_mergining Done")
				self.path_index = self.path_index + 1
			
			elif path_length != 0:
				#print(str(self.bot.agent_id)+") Goal Reached")
				self.goal_reached()
				self.step_running = False
				return (False)
			
			else:
				#During success of global_path, path_length cannot be 0
				print(str(self.bot.agent_id)+") EXCEPTION: NIP=TRUE & path_length = 0")
				self.step_running = False
				return (False)
		else:
			#	print(str(self.bot.agent_id)+") NIP=FALSE")
			self.step_running = False
			return (False)
		self.step_running = False
		return (True)

class hector_navigation():
	def __init__(self,_bot):
		self.block_side = _bot.env.block_side
		self.get_traj_srv_id = "/"+ _bot.group_ns+"/get_exploration_path"
		self.planner_timeout = _bot.env.hector_planner_timeout
		self.bot = _bot

		rospy.wait_for_service(self.get_traj_srv_id)
		self.get_traj_srv = rospy.ServiceProxy(self.get_traj_srv_id, GetRobotTrajectory)
		#Refer documentation on getting service uri - http://docs.ros.org/kinetic/api/rospy/html/rospy.impl.tcpros_service-pysrc.html#ServiceProxy
		self.old_srv_uri = self.get_traj_srv._get_service_uri( rospy.msg.args_kwds_to_message(self.get_traj_srv.request_class, args=(), kwds=()) )

		self.nip = False
		self.path_index = 0
		self.path_length = 0
		self.path = []

	def get_path(self):
		#print(str(self.bot.agent_id)+") Before calling service")
		#srv_resp = self.get_traj_srv()	#Get path from hector exploration node

		try:
			srv_resp = func_timeout(self.planner_timeout, self.get_traj_srv, args=() )
			hector_path = srv_resp.trajectory.poses
		except:
			print("Agent - "+str(self.bot.agent_id)+" : TIMEOUT in service, KILLING HECTOR NODE")
			#self.get_traj_srv = rospy.ServiceProxy(self.get_traj_srv_id, GetRobotTrajectory) # - this is not needed as the service name won't change due to node restart
			new_srv_uri = self.old_srv_uri
			sb_kill = subprocess.Popen(["rosnode","kill","/"+self.bot.group_ns+"/hector_exploration_node"], stdout=subprocess.PIPE, close_fds=True)
			#Loop to ensure that the rospy.wait_for_service() actually waited for the restarted node and not the same old node
			timeout_start = time.time()
			while (new_srv_uri == self.old_srv_uri) and (time.time() - timeout_start < self.planner_timeout):
				time.sleep(0.1)
				try:
					rospy.wait_for_service(self.get_traj_srv_id)
					new_srv_uri = self.get_traj_srv._get_service_uri(rospy.msg.args_kwds_to_message(self.get_traj_srv.request_class, args=(), kwds=()) )
				except:
					pass
			self.old_srv_uri = new_srv_uri
			os.kill(sb_kill.pid , signal.SIGINT)
			print("Agent - "+str(self.bot.agent_id)+" : RESTARTED HECTOR NODE successfully")
			hector_path = []

		#print(str(self.bot.agent_id)+") After calling service")

		len_path = len(hector_path)

		if len_path == 0:
			self.nip = False
			self.path_index = 0
			self.path_length = 0
			print("Received hector exploration plan with zero length")
			if self.bot.env.global_map_exp:
				if self.bot.env.exp_prog > self.bot.env.hect_comp_thresh:
					self.bot.exp_completed = True
			else:
				if self.bot.exp_prog > self.bot.env.hect_comp_thresh:
					self.bot.exp_completed = True
			return [], False

		path = []
		#Get first coordinate
		x_pos = float(self.bot.agent_pos[0])
		y_pos = float(self.bot.agent_pos[1])

		path.append( (x_pos,y_pos) )
		prev_pos = (x_pos,y_pos)
		#Convert odom pos to coordinate on environment
		for i in range(1,len_path):
			x_pos = math.floor( (hector_path[i].pose.position.x + 0.5*self.block_side)/self.block_side )
			y_pos = math.floor( (hector_path[i].pose.position.y + 0.5*self.block_side)/self.block_side )

			if x_pos < 0:
				x_pos = 0
			if y_pos < 0:
				y_pos = 0
			#If pose is same as prev_pose then skip
			if (x_pos,y_pos) == prev_pos:
				continue
			#If pose step is greater than 1 block, then interpolate
			Del_x = int(x_pos-prev_pos[0])
			Del_y = int(y_pos-prev_pos[1])
			if (abs(Del_x) > 1) or (abs(Del_y) > 1):
				if abs(Del_x) >= abs(Del_y):
					x_step = float(Del_x)/abs(Del_x)
					y_step = float(Del_y)/abs(Del_x)
					for j in range(abs(Del_x)):
						x_pos = x_pos + x_step
						y_pos = y_pos + y_step
						path.append( (x_pos,math.floor(y_pos)) )
				else:
					y_step = float(Del_y)/abs(Del_y)
					x_step = float(Del_x)/abs(Del_y)
					for j in range(abs(Del_y)):
						y_pos = y_pos + y_step
						x_pos = x_pos + x_step
						path.append( (math.floor(x_pos),y_pos) )
			else:
				path.append( (x_pos,y_pos) )
			prev_pos=(x_pos,y_pos)
		
		#The below if statement is to avoid bot to be stuck in the same place if path returns only the current pos of the bot
		#Additionally a case is added for len(path)==0 since sometime hector exp falsely says exp is over when crowded within obstacles
		if len(path) <= 1:
			generate_one_pose=False
			if len(path) == 0:
				x_diff = random.uniform(-1,1)
				y_diff = random.uniform(-1,1)
				generate_one_pose=True
			
			elif (path[0][0] == self.bot.agent_pos[0]) and (path[0][1] == self.bot.agent_pos[1]):
				x_diff = hector_path[len_path-1].pose.position.x - self.bot._odom.pose.pose.position.x
				y_diff = hector_path[len_path-1].pose.position.y - self.bot._odom.pose.pose.position.y
				generate_one_pose=True
			
			if generate_one_pose:
				if abs(x_diff) >= abs(y_diff):
					if x_diff >= 0:
						if self.bot.map_check(path[0][0]+1,path[0][1])!=0:
							path.append( (path[0][0]+1,path[0][1]) )
						else:
							if y_diff >= 0:
								if self.bot.map_check(path[0][0],path[0][1]+1)!=0:
									path.append( (path[0][0],path[0][1]+1) )
								elif self.bot.map_check(path[0][0],path[0][1]-1)!=0:
									path.append( (path[0][0],path[0][1]-1) )
								elif self.bot.map_check(path[0][0]-1,path[0][1])!=0:
									path.append( (path[0][0]-1,path[0][1]) )
							else:
								if self.bot.map_check(path[0][0],path[0][1]-1)!=0:
									path.append( (path[0][0],path[0][1]-1) )
								elif self.bot.map_check(path[0][0],path[0][1]+1)!=0:
									path.append( (path[0][0],path[0][1]+1) )
								elif self.bot.map_check(path[0][0]-1,path[0][1])!=0:
									path.append( (path[0][0]-1,path[0][1]) )
					else:
						if self.bot.map_check(path[0][0]-1,path[0][1])!=0:
							path.append( (path[0][0]-1,path[0][1]) )
						else:
							if y_diff >= 0:
								if self.bot.map_check(path[0][0],path[0][1]+1)!=0:
									path.append( (path[0][0],path[0][1]+1) )
								elif self.bot.map_check(path[0][0],path[0][1]-1)!=0:
									path.append( (path[0][0],path[0][1]-1) )
								elif self.bot.map_check(path[0][0]+1,path[0][1])!=0:
									path.append( (path[0][0]+1,path[0][1]) )
							else:
								if self.bot.map_check(path[0][0],path[0][1]-1)!=0:
									path.append( (path[0][0],path[0][1]-1) )
								elif self.bot.map_check(path[0][0],path[0][1]+1)!=0:
									path.append( (path[0][0],path[0][1]+1) )
								elif self.bot.map_check(path[0][0]+1,path[0][1])!=0:
									path.append( (path[0][0]+1,path[0][1]) )
				else:
					if y_diff >= 0:
						if self.bot.map_check(path[0][0],path[0][1]+1)!=0:
							path.append( (path[0][0],path[0][1]+1) )
						else:
							if x_diff >= 0:
								if self.bot.map_check(path[0][0]+1,path[0][1])!=0:
									path.append( (path[0][0]+1,path[0][1]) )
								elif self.bot.map_check(path[0][0]-1,path[0][1])!=0:
									path.append( (path[0][0]-1,path[0][1]) )
								elif self.bot.map_check(path[0][0],path[0][1]-1)!=0:
									path.append( (path[0][0],path[0][1]-1) )
							else:
								if self.bot.map_check(path[0][0]-1,path[0][1])!=0:
									path.append( (path[0][0]-1,path[0][1]) )
								elif self.bot.map_check(path[0][0]+1,path[0][1])!=0:
									path.append( (path[0][0]+1,path[0][1]) )
								elif self.bot.map_check(path[0][0],path[0][1]-1)!=0:
									path.append( (path[0][0],path[0][1]-1) )
					else:
						if self.bot.map_check(path[0][0],path[0][1]-1)!=0:
							path.append( (path[0][0],path[0][1]-1) )
						else:
							if x_diff >= 0:
								if self.bot.map_check(path[0][0]+1,path[0][1])!=0:
									path.append( (path[0][0]+1,path[0][1]) )
								elif self.bot.map_check(path[0][0]-1,path[0][1])!=0:
									path.append( (path[0][0]-1,path[0][1]) )
								elif self.bot.map_check(path[0][0],path[0][1]+1)!=0:
									path.append( (path[0][0],path[0][1]+1) )
							else:
								if self.bot.map_check(path[0][0]-1,path[0][1])!=0:
									path.append( (path[0][0]-1,path[0][1]) )
								elif self.bot.map_check(path[0][0]+1,path[0][1])!=0:
									path.append( (path[0][0]+1,path[0][1]) )
								elif self.bot.map_check(path[0][0],path[0][1]+1)!=0:
									path.append( (path[0][0],path[0][1]+1) )
		
		if len(path) > 1:
			del path[0]	#This deletion is done since the first point in path is just the current position of bot

		self.path_index = 0
		self.path_length = len(path)
		self.nip = True
		return path[:], True		#Return path and mark success

	def step(self):
		if self.nip:		
			if self.path_index < self.path_length:
				target = self.path[self.path_index]

				diagonal_obstacle_pass = False
				if (target[0]!=self.bot.agent_pos[0]) and (target[1]!=self.bot.agent_pos[1]):
					if (self.bot.map_check(target[0],self.bot.agent_pos[1]) == 0) and (self.bot.map_check(self.bot.agent_pos[0],target[1])==0):
						diagonal_obstacle_pass = True

				if (self.bot.map_check(target[0],target[1]) == 0) or diagonal_obstacle_pass:
					#print(str(self.bot.agent_id)+") Recalculating path")
					self.path, success = self.get_path()	#Recalculating path
					if success:
						last_i = len(self.path) - 1
						pos = [ self.path[last_i][0], self.path[last_i][1] ]
						self.bot.goal_pos = pos[:]
						self.bot.goal_pixel = [ int(pos[0]*self.bot.env.map_enlarge_factor + self.bot.env.map_offset[0]), int(pos[1]*self.bot.env.map_enlarge_factor + self.bot.env.map_offset[1]) ]
						return True
					else:
						self.bot.goal_pos = [-1.0, -1.0]
						self.bot.goal_pixel = [-1, -1]
						return (False)
				#print(str(self.bot.agent_id)+") Inside Step")
				self.bot.odom_update(target)
				#print(str(self.bot.agent_id)+") Updated Odom")
				self.bot.map_merge_service()
				#print(str(self.bot.agent_id)+") Map_mergining Done")
				self.path_index = self.path_index + 1

			elif (self.path_index > self.path_length) or (self.path_length == 0):
				print(str(self.bot.agent_id)+") CRITICAL EXCEPTION: PATH_INDEX > PATH_LENGTH")
				self.path_index = self.path_length
				return (False)	#Invalid cases
			else:
				#print(str(self.bot.agent_id)+") Getting path")		#self.path_index == self.path_length - last goal reached
				self.path, success = self.get_path()	#Recalculating path
				if success:
					last_i = len(self.path) - 1
					pos = [ self.path[last_i][0], self.path[last_i][1] ]
					self.bot.goal_pos = pos[:]
					self.bot.goal_pixel = [ int(pos[0]*self.bot.env.map_enlarge_factor + self.bot.env.map_offset[0]), int(pos[1]*self.bot.env.map_enlarge_factor + self.bot.env.map_offset[1]) ]
					return True
				else:
					self.bot.goal_pos = [-1.0, -1.0]
					self.bot.goal_pixel = [-1, -1]
					return (False)
		else:
			self.path, success = self.get_path()	#Calculating path
			if success:
				last_i = len(self.path) - 1
				pos = [ self.path[last_i][0], self.path[last_i][1] ]
				self.bot.goal_pos = pos[:]
				self.bot.goal_pixel = [ int(pos[0]*self.bot.env.map_enlarge_factor + self.bot.env.map_offset[0]), int(pos[1]*self.bot.env.map_enlarge_factor + self.bot.env.map_offset[1]) ]
				return True
			else:
				self.bot.goal_pos = [-1.0, -1.0]
				self.bot.goal_pixel = [-1, -1]
				return (False)
		
		return (True)
