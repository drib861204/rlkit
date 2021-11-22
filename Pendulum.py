"""
Title: pendulum_pygame
Author: [jadenhensley](https://github.com/jadenhensley)
Last modified: 2021/10/18
Description: Pendulum project, built using pygame and math modules.

Title: wheelPole
Author: [aimetz](https://github.com/aimetz)
Last modified: 2021/04/20

Title: gym/gym/envs/classic_control/pendulum.py
Author: [openai](https://github.com/openai)
Last modified: 2021/10/31
"""
import pygame
from math import pi, sin, cos
import numpy as np


class Pendulum:
    def __init__(self):
        self.theta_rod = 0
        self.theta_wheel = 0
        self.theta_rod_dot = 0
        self.theta_wheel_dot = 0
        self.len_rod = 0.5
        self.len_wheel = 0.9
        self.rad_wheel = 0.1
        self.mass_rod = 0.1
        self.mass_wheel = 0.05
        self.momentum_rod = self.mass_rod*self.len_rod**2/12
        self.momentum_wheel = self.mass_wheel*self.rad_wheel**2/2 #depends on wheel shape
        self.dt = 0.001
        self.gravity = 9.8
        self.max_speed = 100

        width = 800
        height = 600
        self.origin_x = width//2
        self.origin_y = height//2
        self.POS = np.array([self.origin_x, self.origin_y])

        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Pendulum Simulation")


    def reset(self):
        self.theta_rod = np.random.random()*40*pi/180-20*pi/180
        self.theta_wheel = 0
        self.theta_rod_dot = 0
        self.theta_wheel_dot = 0
        state = np.array([self.theta_rod, self.theta_wheel, self.theta_rod_dot, self.theta_wheel_dot], dtype=np.float32)
        return state


    def render(self):
        SCALE = 100
        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)

        tip_x = self.POS[0]+self.len_wheel*sin(self.theta_rod)*SCALE
        tip_y = self.POS[1]-self.len_wheel*cos(self.theta_rod)*SCALE
        POSTIP = np.array([tip_x, tip_y])
        #print(POSTIP)
        self.screen.fill(WHITE)
        pygame.draw.line(self.screen, BLACK, self.POS, POSTIP, 1)
        pygame.display.update()



    def step(self, action):
        thrd = self.theta_rod
        thwl = self.theta_wheel
        trdt = self.theta_rod_dot
        twdt = self.theta_wheel_dot
        lnrd = self.len_rod
        lnwl = self.len_wheel
        msrd = self.mass_rod
        mswl = self.mass_wheel
        mmrd = self.momentum_rod
        mmwl = self.momentum_wheel
        dt = self.dt
        g = self.gravity

        torque = action
        effmm1 = msrd*lnrd**2+mswl*lnwl**2+mmrd+mmwl
        effmm2 = mmwl
        a = (msrd*lnrd+mswl*lnwl)*g*sin(angle_normalize(thrd))

        newtrdt = trdt + ((a-torque)/(effmm1-effmm2))*dt
        #print("rod ang_vel",newtrdt)
        newtrdt = np.clip(newtrdt, -self.max_speed, self.max_speed)
        #print("rod ang_vel",newtrdt)
        newthrd = angle_normalize(angle_normalize(thrd) + newtrdt * dt)
        #print("rod angle",newthrd)

        newtwdt = twdt + ((torque*effmm1-a*effmm2)/effmm2/(effmm1-effmm2))*dt
        newtwdt = np.clip(newtwdt, -self.max_speed, self.max_speed)
        #print("wheel ang_vel",newtwdt)
        newthwl = angle_normalize(angle_normalize(thwl) + newtwdt * dt)
        #print("wheel angle",newthwl)
        #print("torque",torque)
        #print("\n")
        #print([torque, newthrd[0], newthwl[0], newtrdt[0], newtwdt[0]])
        state = np.array([newthrd[0], newthwl[0], newtrdt[0], newtwdt[0]], dtype=np.float32)
        self.theta_rod = newthrd
        self.theta_wheel = newthwl
        self.theta_rod_dot = newtrdt
        self.theta_wheel_dot = newtwdt

        costs = angle_normalize(thrd)**2 + 0.1 * trdt**2 + 0.001 * torque**2

        return state, -costs, False


def angle_normalize(th):
    return ((th+pi)%(2*pi))-pi

