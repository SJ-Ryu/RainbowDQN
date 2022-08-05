import pybullet as p
import os
import math
import torch

class VTT:
    # Joint limit
    ang_mag_limit = math.pi/4*24
    ang_vel_limit = math.pi*0.005*24

    pri_len_upper =  0.
    pri_len_lower = -2.
    pri_vel_limit = 0.08

    # Discrete input size
    input_size = 3
    input_mid = -(-(input_size-1)//2)
    
    def __init__(self, client):
        self.client = client
        f_name = os.path.dirname(__file__) + "/../../_asset/VTT_2dof_v4.1/robot.urdf"
        self.vtt = p.loadURDF(fileName=f_name,
                              basePosition=[0, 0, 0.45],
                              useFixedBase=0,
                              physicsClientId=client)

        # Joint indices as found by p.getJointInfo()
        self.rev_indices = [ 0, 2, 4, 6, 8, 10]
        self.pri_indices = [ 1, 3, 5, 7, 9, 11]

        for i in (self.rev_indices + self.pri_indices):
            p.enableJointForceTorqueSensor(bodyUniqueId = self.vtt,
                                           jointIndex = i,
                                           enableSensor = True,
                                           physicsClientId = self.client)

        self.rev_force = 100
        self.pri_force = 100

        # Joint speed
        self.joint_speed = 0

        # Drag constants
        self.c_rolling = 0.2
        self.c_drag = 0.01

        # linear constant increases "speed" of the vtt
        self.c_linear = 20
        
    def unscale(self, x, lower, upper):
        return (2.0 * x - upper - lower) / (upper - lower)

    def get_ids(self):
        return self.vtt, self.client

    def apply_action(self, action):
        # Expects action to be two dimensional and shift [0,1,2] to [-1,0,1]
        linear_disc, rotation_disc = action[:len(self.pri_indices)], action[len(self.pri_indices):]
        linear = self.pri_vel_limit * (linear_disc - self.input_mid)
        rotation = self.ang_vel_limit * (rotation_disc - self.input_mid)

        # Set the linear velocity of prismatic joints
        p.setJointMotorControlArray(
            bodyUniqueId= self.vtt,
            jointIndices= self.pri_indices,
            controlMode= p.VELOCITY_CONTROL,
            targetVelocities= linear,
            forces= [self.pri_force] * len(self.pri_indices),
            physicsClientId= self.client)
        
        # Set the angular velocity of revolute joints
        p.setJointMotorControlArray(
            bodyUniqueId= self.vtt,
            jointIndices= self.rev_indices,
            controlMode= p.VELOCITY_CONTROL,
            targetVelocities= rotation,
            forces= [self.rev_force] * len(self.rev_indices),
            physicsClientId= self.client)

    def get_observation(self):
        # Get the position and orientation of the vtt in the simulation
        base_pos, base_ang = p.getBasePositionAndOrientation(self.vtt, self.client)
        base_vel, base_angv = p.getBaseVelocity(self.vtt, self.client)

        member_info = ()

        for i in self.pri_indices:
            member_info = member_info + p.getLinkState(self.vtt, i)[0] + p.getLinkState(self.vtt, i)[1]

        rev_react = ()
        pri_react = ()

        # Get the velocity of the vtt
        rev_ang = [self.unscale(p.getJointState(self.vtt, i)[0],-self.ang_mag_limit, self.ang_mag_limit) for i in self.rev_indices]
        rev_vel = [self.unscale(p.getJointState(self.vtt, i)[1],-self.ang_vel_limit, self.ang_vel_limit) for i in self.rev_indices]
        for i in self.rev_indices:
            rev_react = rev_react + p.getJointState(self.vtt, i)[2]
        
        pri_len = [self.unscale(p.getJointState(self.vtt, i)[0], self.pri_len_lower, self.pri_len_upper) for i in self.pri_indices]
        pri_vel = [self.unscale(p.getJointState(self.vtt, i)[1],-self.pri_vel_limit, self.pri_vel_limit) for i in self.pri_indices]
        for i in self.pri_indices:
            pri_react = pri_react + p.getJointState(self.vtt, i)[2]

        joint_info = tuple(rev_ang + pri_len + rev_vel + pri_vel) + rev_react + pri_react

        observation = (base_pos + base_ang + base_vel + base_angv + member_info + joint_info)

        return observation

    def get_nodestate(self):
        nodeposition = [p.getLinkState(self.vtt,i)[0] for i in self.pri_indices]
        return nodeposition




