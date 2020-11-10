import gym
import envs
import time
import cv2
import rospy
import subprocess
import os, signal

#Start roscore
roscore_sp = subprocess.Popen(['roscore'],stdout=subprocess.PIPE)

#Initializing ROS Node
rospy.init_node("PEPPER_Environments", anonymous=True)

kwargs = {  'env_id'            :   0,                  # Must be unique for each parallely running environment
            'size'              :   (32,32),
            'obstacle_density'  :   20,
            'n_agents'          :   5,
            'rrt_exp'           :   False,              # If this is True then all agents will explore with rrt only
            'rrt_mode'          :   0,                  # 0-deafult mode; 1-opencv frontier mode; 2-opencv with local rrt hybrid; 3 or any-opencv, global & local rrt included pro hybrid 
            'agents_exp'        :   [1,1,1,1,1],        # 0-explore_lite ; 1-hector_exploration; 2-free navigation; 3-free move(no planner) [length must be = n_agents]
            'global_map_exp'    :   True,               # If True then explore_lite and hector_exp will use merged global maps instead of local merged maps
            'global_planner'    :   False,              # Must be True for using explore_lite, rrt or free_navigation
            'laser_range'       :   14.0,               # Laser scan range in blocks/boxes (1 unit = 1 block)
            'max_comm_dist'     :   7.5,                # Map & last_known_poses communication range in blocks/boxes (1 unit = 1 block)
            'nav_recov_timeout' :   2.0,                # in seconds - timout for move_base to recover on path planning failure on each step (proportional to agents and map size)
            'render_output'     :   True,
            'scaling_factor'    :   20}                 # Size (in px) of each block in rendered cv2 window

env = gym.make('CustomEnv-v0', **kwargs)
#time.sleep(15)
id_offset=0
n=0
sleep_after_step = 0.0  #Recommended to be proportional to the map size and no. of agents (0.5 for size:(128,128) & n_agents:8); can be 0.0 for low map sizes

while not rospy.is_shutdown():
    env.step()  #Navigation step of all agents
    print("Agent 0 Progresses: Local:", env.agent[0].local_exp_prog,"Merged:", env.agent[0].exp_prog, "Global", env.exp_prog)
    ob_map, ob_poses = env.agent[0].observe()
    print("Agent 0 Observes: Map_Shape:", ob_map.shape, "Locations:", ob_poses)
    env.render()
    if cv2.waitKey(10) & 0xFF == ord('x'):
        break
    time.sleep(sleep_after_step)

while True:
    #env.step()
    env.render()

    k=cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    elif k == ord('w'):
        if env.agent[n].agent_pos[0]>0 and env.agent[n].map_check(env.agent[n].agent_pos[0]-1,env.agent[n].agent_pos[1])!=0 :
            env.agent[n].odom_update( [env.agent[n].agent_pos[0]-1,env.agent[n].agent_pos[1]] )
    elif k == ord('s'):
        if env.agent[n].agent_pos[0]+1<env.size[0] and env.agent[n].map_check(env.agent[n].agent_pos[0]+1,env.agent[n].agent_pos[1])!=0 :
            env.agent[n].odom_update( [env.agent[n].agent_pos[0]+1,env.agent[n].agent_pos[1]] )
    elif k == ord('a'):
        if env.agent[n].agent_pos[1]>0 and env.agent[n].map_check(env.agent[n].agent_pos[0],env.agent[n].agent_pos[1]-1)!=0 :
            env.agent[n].odom_update( [env.agent[n].agent_pos[0],env.agent[n].agent_pos[1]-1] )
    elif k == ord('d'):
        if env.agent[n].agent_pos[1]+1<env.size[1] and env.agent[n].map_check(env.agent[n].agent_pos[0],env.agent[n].agent_pos[1]+1)!=0 :
            env.agent[n].odom_update( [env.agent[n].agent_pos[0],env.agent[n].agent_pos[1]+1] )
    elif k == ord('f'): #f for mapping with map_merging feedback
        env.agent[n].map_with_feedback()
    elif k == ord('n'): #n for navigation
        while not rospy.is_shutdown():
            env.step()  #Navigation step of all agents
            env.render()
            if cv2.waitKey(10) & 0xFF == ord('x'):  #x for exit
                break
            time.sleep(sleep_after_step)
    elif k == ord('m'): #m for mapping
        for x in env.largest_contour:
			env.agent[n].odom_update([x[0],x[1]])
			env.render()
			if cv2.waitKey(1) & 0xFF == ord('x'):   #x for exit
				break
    elif k == ord('u'): #u for un-global planner
        env.agent[n].switch_to_free_move()  #This will switch the agent to free_move mode (for RL with limited actions)
    elif k == ord('g'): #g for goal
        env.agent[n].switch_to_free_nav()  #This is not needed if the active_exp of the agent is already "free_navigation" - it unsubscribes from move_base goal topic
        
        #The below 3 lines finds the approx center point of map which is reachable
        mid_lc = env.largest_contour[int(0.5*len(env.largest_contour))]
        x_pixel = mid_lc[0]*env.map_enlarge_factor + env.map_offset[0]
        y_pixel = mid_lc[1]*env.map_enlarge_factor + env.map_offset[1]
        
        env.agent[n].set_goal_pixel(x_pixel,y_pixel)
        while not rospy.is_shutdown():
            env.step()  #Navigation step of all agents
            running_status=env.agent[n].step_ret    #step_ret of an agent becomes False if it has no goals to navigate to or last goal is reached (updated everytime when step() is run)
            env.render()
            if running_status == False: #running status (step_ret) of an agent becomes false even if a goal was unreachable or failed to reach (more description in __init__() of class agent)
                if env.agent[n].agent_pixel==[x_pixel,y_pixel]:
                    break
                g_key = cv2.waitKey(50)
                if g_key == ord('x'):   #x for exit
                    break
                elif g_key == ord('t'): #t for try again
                    env.agent[n].set_goal_pixel(x_pixel,y_pixel)
            if cv2.waitKey(10) & 0xFF == ord('x'):
                break
        env.agent[n].switch_to_exp() #This subscribes back to move_base goal topic and sets the active_exp flag variable to the previous exploration state (rrt,e_lite or hector)
    elif k == ord('r'): #r for reset
        env.reset(5,False,0,[0,0,0,0,0],True,14.0,7.5,2.0)  #Do not forget to press 'N' key after resetting the environment to start over the exploration (IL)
    elif k >= ord(str(id_offset)) and k <= ord('9'):
        _n=k-ord(str(id_offset))
        if _n< env.agents_count:
            n=_n

#You must execute the close function of an environment followed by "del env" in order to completely close it & maybe start another env with the same id too, without closing the program
env.close()
del env

os.kill(roscore_sp.pid,signal.SIGINT)
print("roscore SIGINT sent")    #SIGINT is the equivalent of pressing CTRL+C

"""
Note-1: Read this documentation carefully.
After reading this, it is highly recommended that you go through the variables and functions in the __init__() function of class CustomEnv() and class agent().
You may get better results by modifying some of the variables, and may find some variables useful to be used in the neural network.

Note-2: It is suggested to use multiprocessing library while running multiple parallel environments instead of threading library for best performance (1 envronment per process).
Also remember that in multiprocessing, the environment variables in a process will remain local to that process, and inter-communication variables must be defined separately.

The following variables can be used while communicating with the neural network:

env.exp_prog - (float) - Present value of exploration progress of global map at any given time (Range: 0.0 to 1.0); 0 - exploration just started; 1 - exploration is over
env.agent[n].exp_prog - (float) - Present value of exploration progress of local within range merged map of an agent at any given time
env.agent[n].local_exp_prog - (float) - Present value of exploration progress of local explored map (without communication data) of an agent at any given time

env.map_width - (int) - has the width of gmapping's map
env.map_height - (int) - has the height og gmapping's map
env.map_length - (int) - length of gmapping's map array (env.map_width*env.map_height)

env.map - (1D int tuple) - Contains the global_map subscribed from the the global_map_merger (same size as gmapping's map)
env.agent[n].map - (1D int tuple) - Contains the local merged map of an agent subscribed from local_map_merger (same size as gmapping's map)

env.frontier_map - (1D int tuple) - Contains the global_map's frontier pixels alone in an empty map (same size as gmapping's map) [Possible values: 0-no frontier; 100-frontier point]
env.agent[n].frontier_map - (1D int tuple) - Contains local_merged_map's frontier pixels (same size as gmapping's map) [Possible values: 0-no frontier; 100-frontier point]
Note: Neural network's output must be one of these frontier points only

env.agent[n].agent_pos - (1D float list) - [x,y] - Agent's position in terms of environment map's coordinate (1 unit = 1 block size) - this array stored int values in float datatype
env.agent[n].agent_pixel - (1D int list) - [x,y] - Agent's position in terms of gmapping map's pixel coordinate (1 unit = 1 pixel)

env.agent[n].last_known_pixel - (2D int list) - [[x,y]*no. of agents] - Description in __init__() function of class agent

env.agent[n].goal_pos - (1D float list) - [x,y] - Stores the agents latest goal point which is in progress or just got completed (updates to [-1,-1] if an invalid goal was received
env.agent[n].goal_pixel - (1D int list) - [x,y] - Stores the pixel coordinate of env.agent[n].goal_pos.

Note-3:
env.agent[n].goal_pos & env.agent[n].goal_pixel are updated whenever a new goal is sent to an agent by an exploration package.
Hence env.agent[n].goal_pixel can by taken as the output of Immitation Learning since the, neural network is expected to operate on (I/O means) gmapping map's pixel coordinates.
Also note that the value of env.agent[n].goal_pos & env.agent[n].goal_pixel will be updated even if a goal is sent by the neural network or user using env.agent[n].set_goal_pixel() function.

Control functions:
Note-4: Most of the control functions are demostrated in the above code itself. Some of the special functions are mentioned below:
env.agent[n].switch_to_free_nav() - If an agent is initialized with an exploration package, but neutal network need to take over this needs to be executed before set_goal_pixel()
env.agent[n].set_goal_pixel(int x,int y) - Set navigation goal by passing the gmapping map pixel's coordinate as the arguments. This goal will be executed in the upcoming step() executions
env.step() - executed exactly 1 step of the environment (execcutes steps function of all agents parallely once, for one step movement or no movement)
env.agent[n].move_base.cancel_sub(0) - Cancel a current navigation goal which is active & halt the agent, whereever it was when this function was called.
Note-5: Whenever a new goal is given to env.agent[n].set_goal_pixel(), while the previous goal is still active, the previous goal will be automatically cancelled and the new goal,
will become the current active goal. Hence, for a neural network, the usage of env.agent[n].move_base.cancel_sub(0) function won't be required.

Side notes related to env.step():
Variable linked with step() function: env.agent[n].step_ret - description in __init__() function of class agent
When neutral network is controlling the navigation of an agent:
if env.agent[n].step_ret == True: Navigation is in progress. The neural network may of may not give the agent a new goal at this state
if env.agent[n].step_ret == False: Navigation is completed/cancelled. The neural network MUST give the agent a new goal at this state, if exploration is not completed.

env.agent[n].switch_to_exp() - If you want to exp_package that was running previously to take over the navigation again then execute this. This basically undoes the function of env.agent[n].switch_to_free_nav()
"""
