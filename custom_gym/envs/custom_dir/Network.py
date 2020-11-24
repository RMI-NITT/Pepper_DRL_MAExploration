import cv2
import random
import numpy as np
import math
import os
import gym
import envs
import time
import tensorflow
import tensorflow.compat.v1 as tf
#import cv2
from tensorflow.compat.v1 import ConfigProto

import rospy
import subprocess
import os, signal
import random
RNN_SIZE = 512
BUFFER_SIZE = 50		#no of episodes before loss update
NUM_BUFFERS = 100
DISCOUNT_RATE = 0.95
GRAD_CLIP = 100
tf.reset_default_graph()
GLOBAL_NET_SCOPE       = 'global'
lr = 1e-5
#groupLocks=[]
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
reset_trainer_op = tf.variables_initializer(trainer.variables())

#trainer=tf.compat.v1.train.AdamOptimizer(learning_rate=lr , use_locking=False,name='Adam')
import tensorflow.contrib.layers as layers
config=ConfigProto()
config.gpu_options.allow_growth=True

class Network():
	def __init__(self,scope, trainer, training, GLOBAL_NET_SCOPE):
		print("**********************************INIT********************************")
		self.graph = tf.Graph()	
		self.scope = scope
		with tf.variable_scope(str(self.scope)+'/qvalues',reuse = tf.AUTO_REUSE):

			self.sess = tf.Session(config=config, graph = self.graph)
			last = []
			with self.graph.as_default():	
				if training == True:
					self.scope = scope
					self.trainer = trainer
					#self.time_step = 0
					self.value,self.policy,self.state_out,self.policy_sig = self.net(2)
					self.gradients,self.loss,self.imitation_loss = self.calculate_loss()

					self.tf_value = tf.summary.scalar('loss', self.loss)
					self.saver = tf.train.Saver(max_to_keep=1)

				#self.policy, self.value, self.state_out, self.state_in, self.state_init, self.valids = self._build_net(self.myinput,self.scalars,RNN_SIZE,TRAINING,a_size)
				
	def normalized_columns_initializer(self,std=1.0):
		def _initializer(shape, dtype=None, partition_info=None):
			out = np.random.randn(*shape).astype(np.float32)
			out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
			return tf.constant(out)
			return _initializer
	def net(self,no_agents):
		print("****************************NET**************************")
		#with self.graph.as_default():
		with tf.variable_scope(str(self.scope)+'/qvalues',reuse = tf.AUTO_REUSE):
			self.inputs  = tf.placeholder(shape = [1,256,256,2] , dtype=tf.int64)
			self.inputs = tf.to_float(self.inputs)
			self.last_known_pixel = tf.placeholder(shape = [1,4],dtype = tf.float32)
			#self.last_known_pixel = np.reshape(self.last_known_pixel,(1,4))
			self.arr = self.last_known_pixel
			#print(last_known_pixel.shape)
			#self.inputs = tf.to_float(inputs)
			self.no_agents = no_agents
			self.no_actions = 9 #change it to 9 if diagonals are included
			#print(self.inputs.shape)
			#self.time_step+=1
			#weigthss shd come here..
			#self.w = layers.variance_scaling_initializer()
			self.w1 = layers.xavier_initializer(seed = 0)
			self.w2 = layers.xavier_initializer(seed = 0)
			self.w3 = layers.xavier_initializer(seed = 0)
			self.w4 = layers.xavier_initializer(seed = 0)
			self.W1 = tf.get_variable("self.W1", [16,16,2,8],initializer = self.w1)
			self.W2 = tf.get_variable("self.W2", [8,8,8,16],initializer = self.w2)
			self.W3 = tf.get_variable("self.W3", [4,4,16,32],initializer = self.w3)
			self.W4 = tf.get_variable("self.W4", [4,4,32,64],initializer = self.w4)
			#print(self.W1)

			self.Z1 = tf.nn.conv2d(self.inputs,self.W1, strides = [1,1,1,1], padding = 'SAME')
			# RELU
			self.A1 = tf.nn.relu(self.Z1)
			# MAXPOOL: window 8x8, stride 8, padding 'SAME'
			self.P1 = tf.nn.max_pool2d(self.A1, ksize = [1,8,8,1], strides = [1,1,1,1], padding = 'SAME')
			# CONV2D: filters W2, stride 1, padding 'SAME'
			self.Z2 = tf.nn.conv2d(self.P1,self.W2, strides = [1,1,1,1], padding = 'SAME')
			# RELU
			self.A2 = tf.nn.relu(self.Z2)
			# MAXPOOL: window 4x4, stride 4, padding 'SAME'
			self.P2 = tf.nn.max_pool2d(self.A2, ksize = [1,4,4,1], strides = [1,1,1,1], padding = 'SAME')
			self.Z3 = tf.nn.conv2d(self.P2,self.W3, strides = [1,1,1,1], padding = 'SAME')
			self.A3 = tf.nn.relu(self.Z3)
			self.P3 = tf.nn.max_pool2d(self.A3, ksize = [1,2,2,1], strides = [1,1,1,1], padding = 'SAME')
			self.Z4 = tf.nn.conv2d(self.P3,self.W4, strides = [1,4,4,1], padding = 'VALID')
			self.A4 = tf.nn.relu(self.Z4)
			self.P4 = tf.nn.max_pool2d(self.A4, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
			#print(self.P4.shape)
			#self.an_arrays = self.P4.eval(session=tf.compat.v1.Session())
			#self.P4=np.mean(self.an_arrays,axis=3)
			#print(self.no_agents)
		
			#self.arr=[[-1]*(self.no_agents*2)]
			#self.arr = self.
			self.F = tensorflow.contrib.layers.flatten(self.P4)
			#self.l1 = layers.fully_connected(inputs=self.F,  num_outputs = self.no_agents*2)

			self.hidden_input = tf.concat([self.F,self.arr],axis=1)

			self.h1 = layers.fully_connected(inputs=self.hidden_input,  num_outputs = RNN_SIZE)
			self.d1 = layers.dropout(self.h1, keep_prob = 0.7, is_training=True)
			self.h2 = layers.fully_connected(inputs = self.d1,  num_outputs = RNN_SIZE, activation_fn=None)
			self.d2 = layers.dropout(self.h2, keep_prob = 0.7, is_training=True)

			self.h3 = tf.nn.relu(self.d2)

			#Recurrent network for temporal dependencies
			self.lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(RNN_SIZE,state_is_tuple=True)
			self.c_init = np.zeros((1, self.lstm_cell.state_size.c), np.float32)
			self.h_init = np.zeros((1, self.lstm_cell.state_size.h), np.float32)
			self.state_init = [self.c_init, self.h_init]
			self.c_in = tf.placeholder(tf.float32, [1, self.lstm_cell.state_size.c])
			self.h_in = tf.placeholder(tf.float32, [1, self.lstm_cell.state_size.h])
			self.state_in = (self.c_in, self.h_in)
			self.rnn_in = tf.expand_dims(self.h3, [0])
			self.step_size = tf.shape(self.inputs)[:1]
			self.state_in = tf.nn.rnn_cell.LSTMStateTuple(self.c_in, self.h_in)
			self.lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(self.lstm_cell, self.rnn_in, 
			initial_state = self.state_in, sequence_length = self.step_size,time_major = False)	
			self.lstm_c, self.lstm_h = self.lstm_state
			state_out = (self.lstm_c[:1, :], self.lstm_h[:1, :])
			self.rnn_out = tf.reshape(self.lstm_outputs, [-1, RNN_SIZE])
			self.policy_layer = layers.fully_connected(inputs=self.rnn_out, num_outputs=self.no_actions,weights_initializer=self.normalized_columns_initializer(1./float(self.no_actions)), biases_initializer=None, activation_fn=None)
			policy       = tf.nn.softmax(self.policy_layer)
			policy_sig   = tf.sigmoid(self.policy_layer)
			value        = layers.fully_connected(inputs=self.rnn_out, num_outputs=1, weights_initializer=self.normalized_columns_initializer(1.0), biases_initializer=None, activation_fn=None)
			#print(self.rnn_out)

			#tf.reset_default_graph()


			#conv1    =  layers.conv2d(inputs=self.inputs,   padding="SAME", num_outputs=RNN_SIZE//4,  kernel_size=[3, 3],   stride=1, data_format="NHWC", weights_initializer=self.W1_init,activation_fn=tf.nn.relu)
			#conv1a   =  layers.conv2d(inputs=conv1,   padding="SAME", num_outputs=RNN_SIZE//4,  kernel_size=[3, 3],   stride=1, data_format="NHWC", weights_initializer=w_init,activation_fn=tf.nn.relu)
			#conv1b   =  layers.conv2d(inputs=conv1a,   padding="SAME", num_outputs=RNN_SIZE//4,  kernel_size=[3, 3],   stride=1, data_format="NHWC", weights_initializer=w_init,activation_fn=tf.nn.relu) 
			#pool1    =  layers.max_pool2d(inputs=conv1b,kernel_size=[2,2])
			#print(pool1)

			return value,policy,state_out,policy_sig
	#@tf.function
	def calculate_loss(self):
		print("******************************LOSS*******************")
		with tf.variable_scope(str(self.scope)+'/qvalues',reuse = tf.AUTO_REUSE) and self.graph.as_default() :#and tf.GradientTape() as t:

			self.a_size = 9
			self.actions               = tf.placeholder(shape=(1,9), dtype=tf.int32)
			self.actions_onehot         = tf.one_hot(self.actions, self.a_size, dtype=tf.float32)
			self.train_valid            = tf.placeholder(dtype=tf.float32)
			self.target_v               = tf.placeholder(tf.float32)
			self.advantages             = tf.placeholder(dtype=tf.float32)
			#self.policy = policy
			print(tf.is_tensor(self.policy))
			#self.value = value
			self.train_value = tf.placeholder(shape=(1,1), dtype=tf.float32)
			self.valids = self.policy_sig
						
			self.actions_onehot    = tf.one_hot(self.actions, self.a_size, dtype=tf.float32)
			self.responsible_outputs    = tf.reduce_sum(tf.reshape(self.policy,shape=[-1]) * self.actions_onehot, [1])

			self.policy_loss = - tf.reduce_sum(tf.log(tf.clip_by_value(self.responsible_outputs,1e-15,1.0)) * self.advantages)
			self.value_loss     = 0.1 * tf.reduce_sum(self.train_value*tf.square(self.target_v - tf.reshape(self.value, shape=[-1])))
			self.entropy        = - 0.01 * tf.reduce_sum(self.policy * tf.log(tf.clip_by_value(self.policy,1e-10,1.0)))
			self.valid_loss     = - 0.1 * tf.reduce_sum(tf.reduce_sum(tf.log(tf.clip_by_value(self.valids,1e-10,1.0)) * self.train_valid + \
										tf.log(tf.clip_by_value(1-self.valids,1e-10,1.0)) * (1-self.train_valid), axis=1))

			self.loss           = self.value_loss + self.policy_loss  - self.entropy  + self.valid_loss
			self.optimal_actions = tf.placeholder(shape=(), dtype=tf.int32)
			self.optimal_actions_onehot = tf.one_hot(self.optimal_actions, self.a_size, dtype=tf.float32)
			self.imitation_loss = tf.reduce_mean(
				tf.keras.backend.categorical_crossentropy(self.optimal_actions_onehot, self.policy))
			local_vars         = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope+'/qvalues')
			#print(self.local_vars)
			#self.gradients     = tf.gradients(self.loss, self.local_vars,unconnected_gradients='zero')
			self.gradients= self.trainer.compute_gradients(self.loss, var_list=local_vars)
			#self.gradients = t.gradient(self.loss, self.local_vars)
			print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
			#print(self.gradients)
			#print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
			#print(self.inputs.shape)
			self.var_norms     = tf.global_norm(local_vars)
			#grads, grad_norms = tf.clip_by_global_norm(self.gradients, GRAD_CLIP)
			#print(self.grads)
			# Apply local gradients to global network
			#global_vars        = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, GLOBAL_NET_SCOPE+'/qvalues')
			#opt = opt.ScipyOptimizerInterface(self.loss, self.var_norms)

			self.apply_grads   = self.trainer.apply_gradients(self.gradients)
			print(len(self.gradients))

			return self.gradients,self.loss,self.imitation_loss
localNet = Network(GLOBAL_NET_SCOPE, trainer, True, GLOBAL_NET_SCOPE)	#local_network


if __name__ == "__main__":
	# localNet = Network(GLOBAL_NET_SCOPE, trainer, True, GLOBAL_NET_SCOPE)	#local_network

	roscore_sp = subprocess.Popen(['roscore'],stdout=subprocess.PIPE)
	
	#self.sess=tf.Session(graph=tf.Graph())
	#with tf.Session(config=config, graph = localNet.graph) as sess:
	#localNet.init = tf.global_variables_initializer()
	with localNet.graph.as_default():		
		localNet.sess.run(tf.global_variables_initializer())
		localNet.sess.run(tf.local_variables_initializer())
		localNet.coord = tf.train.Coordinator()
		#localNet.sess.graph.finalize()
	#Initializing ROS Node
	rospy.init_node("PEPPER_Environments", anonymous=True)
	'''kwargs = {  'env_id'            :   0,                  # Must be unique for each parallely running environment
		    'size'              :   (32,32),
		    'obstacle_density'  :   20,
		    'n_agents'          :   2,
		    'rrt_exp'           :   False,              # If this is True then all agents will explore with rrt only
		    'rrt_mode'          :   0,                  # 0-deafult mode; 1-opencv frontier mode; 2-opencv with local rrt hybrid; 3 or any-opencv, global & local rrt included pro hybrid 
		    'agents_exp'        :   [3,3],        # 0-explore_lite ; 1-hector_exploration; 2 or any - free navigation [length must be = n_agents]
		    'global_map_exp'    :   True,               # If True then explore_lite and hector_exp will use merged global maps instead of local merged maps
		    'laser_range'       :   14.0,               # Laser scan range in blocks/boxes (1 unit = 1 block)
		    'max_comm_dist'     :   7.5,                # Map & last_known_poses communication range in blocks/boxes (1 unit = 1 block)
		    'nav_recov_timeout' :   2.0,                # in seconds - timout for move_base to recover on path planning failure on each step (proportional to agents and map size)
		    'render_output'     :   True,
		    'scaling_factor'    :   20,
			'global_planner'	: False,
            #'session'           :   sess,           
				}                 # Size (in px) of each block in rendered cv2 window
	'''
	#env = gym.make('CustomEnv-v0', **kwargs)
	#time.sleep(15)
	id_offset=0
	n=0
	sleep_after_step = 0.0  #Recommended to be proportional to the map size and no. of agents (0.5 for size:(128,128) & n_agents:8); can be 0.0 for low map sizes
	model_count=2
	while not rospy.is_shutdown():
		gamma = 0.3
		gamma1 = 0.5
		gamma2 = 0.3
		gamma3 = 0.2
		if random.uniform(0, 1)<gamma:
			imitation = True
			r = random.uniform(0,1)
			if r>gamma1:
				print("****************HECTOR************************")
				kwargs = {'env_id': 0,  # Must be unique for each parallely running environment
						  'size': (32, 32),
						  'obstacle_density': 20,
						  'n_agents': 2,
						  'rrt_exp': False,  # If this is True then all agents will explore with rrt only
						  'rrt_mode': 0,
						  # 0-deafult mode; 1-opencv frontier mode; 2-opencv with local rrt hybrid; 3 or any-opencv, global & local rrt included pro hybrid
						  'agents_exp': [2,2],
						  # 0-explore_lite ; 1-hector_exploration; 2 or any - free navigation [length must be = n_agents]
						  'global_map_exp': True,
						  # If True then explore_lite and hector_exp will use merged global maps instead of local merged maps
						  'laser_range': 14.0,  # Laser scan range in blocks/boxes (1 unit = 1 block)
						  'max_comm_dist': 7.5,
						  # Map & last_known_poses communication range in blocks/boxes (1 unit = 1 block)
						  'nav_recov_timeout': 2.0,
						  # in seconds - timout for move_base to recover on path planning failure on each step (proportional to agents and map size)
						  'render_output': True,
						  'scaling_factor': 20,
						  'global_planner': False,
						  #explore lite and rrt True, hector and RL False
						  # 'session'           :   sess,
						  }  # Size (in px) of each block in rendered cv2 window
			elif r<gamma2:
				print('*********************EXPLORELITE******************')
				kwargs = {'env_id': 0,  # Must be unique for each parallely running environment
					  'size': (32, 32),
					  'obstacle_density': 20,
					  'n_agents': 2,
					  'rrt_exp': False,  # If this is True then all agents will explore with rrt only
					  'rrt_mode': 0,
					  # 0-deafult mode; 1-opencv frontier mode; 2-opencv with local rrt hybrid; 3 or any-opencv, global & local rrt included pro hybrid
					  'agents_exp': [0,0],
					  # 0-explore_lite ; 1-hector_exploration; 2 or any - free navigation [length must be = n_agents]
					  'global_map_exp': True,
					  # If True then explore_lite and hector_exp will use merged global maps instead of local merged maps
					  'laser_range': 14.0,  # Laser scan range in blocks/boxes (1 unit = 1 block)
					  'max_comm_dist': 7.5,
					  # Map & last_known_poses communication range in blocks/boxes (1 unit = 1 block)
					  'nav_recov_timeout': 2.0,
					  # in seconds - timout for move_base to recover on path planning failure on each step (proportional to agents and map size)
					  'render_output': True,
					  'scaling_factor': 20,
					  'global_planner': True,
					  # explore lite and rrt True, hector and RL False
					  # 'session'           :   sess,
					  }  # Size (in px) of each block in rendered cv2 window
			else:
				print('*********************RRT*********************')
				kwargs = {'env_id': 0,  # Must be unique for each parallely running environment
						  'size': (32, 32),
						  'obstacle_density': 20,
						  'n_agents': 2,
						  'rrt_exp': True,  # If this is True then all agents will explore with rrt only
						  'rrt_mode': 0,
						  # 0-deafult mode; 1-opencv frontier mode; 2-opencv with local rrt hybrid; 3 or any-opencv, global & local rrt included pro hybrid
						  'agents_exp': [1,1],
						  # 0-explore_lite ; 1-hector_exploration; 2 or any - free navigation [length must be = n_agents]
						  'global_map_exp': True,
						  # If True then explore_lite and hector_exp will use merged global maps instead of local merged maps
						  'laser_range': 14.0,  # Laser scan range in blocks/boxes (1 unit = 1 block)
						  'max_comm_dist': 7.5,
						  # Map & last_known_poses communication range in blocks/boxes (1 unit = 1 block)
						  'nav_recov_timeout': 2.0,
						  # in seconds - timout for move_base to recover on path planning failure on each step (proportional to agents and map size)
						  'render_output': True,
						  'scaling_factor': 20,
						  'global_planner': True,
						  # explore lite and rrt True, hector and RL False
						  # 'session'           :   sess,
						  }  # Size (in px) of each block in rendered cv2 window

		else:
			imitation = False
			kwargs = {'env_id': 0,  # Must be unique for each parallely running environment
					  'size': (32, 32),
					  'obstacle_density': 20,
					  'n_agents': 2,
					  'rrt_exp': False,  # If this is True then all agents will explore with rrt only
					  'rrt_mode': 0,
					  # 0-deafult mode; 1-opencv frontier mode; 2-opencv with local rrt hybrid; 3 or any-opencv, global & local rrt included pro hybrid
					  'agents_exp': [3,3],
					  # 0-explore_lite ; 1-hector_exploration; 2 or any - free navigation [length must be = n_agents]
					  'global_map_exp': True,
					  # If True then explore_lite and hector_exp will use merged global maps instead of local merged maps
					  'laser_range': 14.0,  # Laser scan range in blocks/boxes (1 unit = 1 block)
					  'max_comm_dist': 7.5,
					  # Map & last_known_poses communication range in blocks/boxes (1 unit = 1 block)
					  'nav_recov_timeout': 2.0,
					  # in seconds - timout for move_base to recover on path planning failure on each step (proportional to agents and map size)
					  'render_output': True,
					  'scaling_factor': 20,
					  'global_planner' : False,
					  # 'session'           :   sess,
					  }  # Size (in px) of each block in rendered cv2 window
		env = gym.make('CustomEnv-v0', **kwargs)
		model_count = model_count+1


		env.step(localNet,trainer,model_count,imitation) #Navigation step of all agents
		env.render()
		env.close()
		del env
		if cv2.waitKey(10) & 0xFF == ord('x'):
			break
		time.sleep(sleep_after_step)
	sess.close()

