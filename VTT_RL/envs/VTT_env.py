from ntpath import join
from re import I
import gym
import numpy as np
import math
import pybullet as p
from VTT_RL.resources.VTT import VTT
from VTT_RL.resources.plane import Plane
from VTT_RL.resources.goal import Goal
import matplotlib.pyplot as plt

limit = math.pi*24/4

ang_vel_limit = VTT.ang_vel_limit
pri_vel_limit = VTT.pri_vel_limit
input_size = VTT.input_size

class VTTEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self):
        # action space for [0,1,2,3,4] =: [-2*go, -go, stop, go, 2*go]
        self.action_space = gym.spaces.Discrete(input_size)   
        self.observation_space = gym.spaces.box.Box(
             low=np.array([-10, -10, -1, -1, -1, -1], dtype=np.float32),
            high=np.array([ 10,  10,  1,  1,  1,  1], dtype=np.float32)) # pos x,y , rot matrix x,y , vel x,y , goal x,y
        self.np_random, _ = gym.utils.seeding.np_random()

        # connect method ->p.DIRECT ( without render )
        #                ->p.GUI    ( with render )
        self.client = p.connect(p.DIRECT)
        # Reduce length of episodes for RL algorithms
        p.setTimeStep(1/50, self.client)

        self.vtt = None
        self.goal = None
        self.done = False
        self.prev_pos = None
        self.rendered_img = None
        self.render_rot_matrix = None
        self.reset()
    
    def clientID(self):
        return self.client

    def step(self, action):
        # Feed action to the vtt and get observation of vtt's state
        self.vtt.apply_action(action)
        p.stepSimulation()
        vtt_ob = self.vtt.get_observation()
        
        # dist_to_goal = -(vtt_ob[0] - self.goal[0]) ** 2 - (vtt_ob[1] - self.goal[1]) ** 2
        vtt_z_penalty = vtt_ob[2]-0.1
        
        # discreate reward
        dist_to_goal = 10*(vtt_ob[0] + vtt_ob[1])
        

        joint_penalty = np.abs(vtt_ob[13:25])
        joint_penalty_sum = sum(joint_penalty)
        # reward = dist_to_goal + vtt_z_penalty
        values = 500*(dist_to_goal - self.prev_pos) + 0.1*vtt_z_penalty - 0.01*joint_penalty_sum

        if (vtt_ob[0]<0) or (vtt_ob[1]<0):
            values = -1

        # Done by running off boundaries
        if (vtt_ob[0] >= 10 or vtt_ob[0] <= -10 or vtt_ob[1] >= 10 or vtt_ob[1] <= -10 or vtt_ob[2] <= 0.11):
            self.done = True
            reward = values - 1000
        else:
            reward = values

        print(f"reward : {reward:>10.5f} \t| delta dis ==== {(dist_to_goal - self.prev_pos):>10.5f}")
        
        self.prev_pos = dist_to_goal
        
        ob = np.array(vtt_ob + tuple(action.tolist()), dtype=np.float32) 
        
        return ob, reward, self.done, dict()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]


    def reset(self):
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -10)
        # Reload the plane and vtt
        Plane(self.client)
        self.vtt = VTT(self.client)

        # Set the goal to a random target
        x = (self.np_random.uniform( 5, 9) if self.np_random.randint(2) else
             self.np_random.uniform(-9,-5))
        y = (self.np_random.uniform( 5, 9) if self.np_random.randint(2) else
             self.np_random.uniform(-9,-5))
        self.goal = (x, y)
        self.done = False

        # Visual element of the goal
        Goal(self.client, self.goal)

        # Get observation to return
        vtt_ob = self.vtt.get_observation()

        self.prev_dist_to_goal = math.sqrt(((vtt_ob[0] - self.goal[0]) ** 2 +
                                            (vtt_ob[1] - self.goal[1]) ** 2))
        # return np.array(vtt_ob + self.goal + (0,0,0,0,0,0,0,0,0,0,0,0), dtype=np.float32)
        self.prev_pos = 10*(vtt_ob[0] + vtt_ob[1])
        return np.array(vtt_ob + (0,0,0,0,0,0,0,0,0,0,0,0), dtype=np.float32)
        # return np.array(vtt_ob[7:19], dtype=np.float32)

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect(self.client)