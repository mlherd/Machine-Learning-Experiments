"""
Gym environment for a on-track driving simulator
"""

import sys
import logging
import random
from os import path
from gym import spaces
import numpy as np
import math
import gym

from .flatlands_sim import DrawMap, BicycleModel, WorldMap

LOGGER = logging.getLogger("flatlands_env")


class FlatlandsEnv(gym.Env):
    """
    Gym environment for on-track driving simulator
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        Load the track, draw module, etc.
        """
        self.r = 0
        map_file = path.join(sys.prefix, "map_files\original_circuit_green.csv")
        self.action_space = spaces.Box(low=np.array([-0.1, -1.0]), high=np.array([0.1, 1.0]), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([-10.0, -1.0, -6.29]), high=np.array([1.0, 1.0, 6.29]), dtype=np.float32)
        self.world = WorldMap(map_file)
        self.draw_class = DrawMap(world=self.world)
        self.vehicle_model = BicycleModel(*self.world.path[0], self.world.direction[0], max_velocity=1)

        self.car_info = None

    def calculate_reward(self, r_speed, c_speed, orientation, distance_x, distance_y, head_x, head_y):
        reward = 30 * abs(c_speed) - 10 * abs(r_speed) - 5 * (abs(math.sqrt(distance_x ** 2 + distance_y ** 2))) - 2 * head_x
        return reward

    def step(self, action):
        """
        Accepts an `action` object, consisting of desired accelleration (accel)
        and the steering angle

        Returns on observation object
        """

        accel = action["accel"]
        wheel_angle = action["wheel_angle"]

        self.vehicle_model.move_accel(accel, wheel_angle)
		
        obs = {
            "reward":
            self.r,
            "dist_upcoming_points":
            self.world.get_dist_upcoming_points(self.vehicle_model.position, self.vehicle_model.orientation),
        }
		
        point = obs["dist_upcoming_points"][2]
        head = obs["dist_upcoming_points"][0]
        theta = math.atan(point[0] / point[1])
		
        state = np.array([point[0], point[1], theta])
        r_speed = self.vehicle_model.get_info_object()["r_speed"]
        c_speed = self.vehicle_model.get_info_object()["c_speed"]
        #speed = self.vehicle_model.get_info_object()["car_speed"]
        self.r = self.calculate_reward(r_speed, c_speed, theta, point[0], point[1], head[0], head[1])

        return state, self.r, False, {}

    def reset(self):
        """
        Reset the car to a static place somewhere on the track.
        """

        LOGGER.debug("system resetting")

        idx = random.randint(0, len(self.world.path) - 1)
        LOGGER.debug("Randomly placing the vehicle near map point #{}".format(idx))
        x, y = self.world.path[idx]
        theta = self.world.direction[idx]
        self.vehicle_model.set(x, y, theta)

        self.distance_traveled = 0
		
        obs = {
            "reward":
            0,
            "dist_upcoming_points":
            self.world.get_dist_upcoming_points(self.vehicle_model.position, self.vehicle_model.orientation),
        }

        return obs
		

    def render(self, mode='human', close=False):
        """
        Use pygame to draw the map
        """

        car_info_object = self.vehicle_model.get_info_object()
        self.draw_class.draw_car(car_info_object)
