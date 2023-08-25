import sys
import os
sys.path.append('../')
sys.path.append('.')

import gym
import rtde_receive
import rtde_control
import dashboard_client
import time
import numpy as np
import random
import csv
from src.bendRL_env.targetVisualization import TargetDisplay
import flycapture2 as fc2
import cv2
import tkinter

# install flycap from https://github.com/ethanlarochelle/pyflycapture2
# https://www.flir.ca/support-center/iis/machine-vision/application-note/understanding-buffer-handling/
# sys.path.append("~/git/pyflycapture2")
sys.path.append("~/repositories/pyflycapture2")


#change to 3 for colour image
N_CHANNELS = 3
#observation space type: image, image_joint, image_joint_action

class VisualReacherFiveJoints(gym.Env):
    def __init__(self, random_start=0, log_state_actions=False, save_images=False, goal_threshold=0.25,
                 file_name_prefix="default", env_type="static", obs_space_type='image', target_position=[1800, 100]):
        super(VisualReacherFiveJoints, self).__init__()
        self.env_type = env_type
        self.obs_space_type = obs_space_type
        self.target_position = target_position
        self.count = 0  # steps
        self.episode_count = 0
        self.STEPS_IN_EPISODE = 200
        self.STEP_SIZE = 0.05
        self.log_state_actions = log_state_actions
        self.log_file_path = None
        self.file_name_prefix = file_name_prefix
        self.save_images = save_images
        if self.log_state_actions:
            self.log_file_path = self.file_name_prefix + "_log_state_actions.csv"
            f = open(self.log_file_path, 'w')
            writer = csv.writer(f)
            header = ["Episode", "Step", "Q-Position", "TCP-Coordinates", "Action", "Reward", "Distance to goal",
                      "Protective stop",
                      "Joint causing stop", "Type of stop"]
            writer.writerow(header)

        if not os.path.exists("saved_images/" + self.file_name_prefix + "/"):
            os.makedirs("saved_images/" + self.file_name_prefix + "/")

        # The goal position
        self.GOAL_COORD = [-0.3895592448476226, 0.684725084715733, 0.40715734369523193, 1.6313885170972071,
                           -0.9769981434059597, -1.2117417904968952]
        self.GOAL_THRESHOLD = goal_threshold
        self.dist_to_goal = None
        self.RANDOM_START = random_start
        self.FIXED_START = [4.017988204956055, -1.5178674098900338, 2.1686766783343714, -0.8878425520709534,
                            -0.35228139558901006, 0.16946229338645935]

        # The possible actions (wrist3 should not move)
        self.BASE_CLOCKWISE = 0
        self.BASE_COUNTER_CLOCKWISE = 1
        self.SHOULDER_CLOCKWISE = 2
        self.SHOULDER_COUNTER_CLOCKWISE = 3
        self.ELBOW_CLOCKWISE = 4
        self.ELBOW_COUNTER_CLOCKWISE = 5
        self.WRIST1_CLOCKWISE = 6
        self.WRIST1_COUNTER_CLOCKWISE = 7
        self.WRIST2_CLOCKWISE = 8
        self.WRIST2_COUNTER_CLOCKWISE = 9

        self.LAST_ACTION = None
        self.LAST_STATE = None
        self.LAST_IMAGE_STATE = None

        #UNCOMMENT TO REVERT TO BLACK AND WHITE
        # # CAMERA SETUP
        # self.CAMERA_HOST = "192.168.0.150"
        # self.WIDTH = int(np.round(1600 * 0.25, 0))
        # self.HEIGHT = int(np.round(1200 * 0.25, 0))
        # self.CAMERA = fc2.Context()
        # self.CAMERA.connect(*self.CAMERA.get_camera_from_index(0))
        # # self.CAMERA.set_format7_configuration(8, 0, 0, image_width, image_height, 4194304)
        # self.CAMERA.start_capture()

        self.HOST = "192.168.0.110"  # the IP address 127.0.0.1 is for URSim, 192.168.0.110 for UR10E

        # CAMERA SETUP
        self.CAMERA_HOST = "192.168.0.150"
        self.WIDTH = int(np.round(1600 * 0.25, 0))
        self.HEIGHT = int(np.round(1200 * 0.25, 0))
        self.CAMERA = fc2.Context()
        self.CAMERA.connect(*self.CAMERA.get_camera_from_index(0))
        self.CAMERA.set_format7_configuration(fc2.MODE_8, 0, 0, 1600, 1200, fc2.PIXEL_FORMAT_RGB8)
        # self.CAMERA.set_format7_configuration(8, 0, 0, image_width, image_height, 4194304)
        self.CAMERA.start_capture()

        # Joint limits from our robot
        # The ranges are where the robot should keep the motion
        self.BASE_LOWER_LIMIT_Q = 3.4907
        self.BASE_UPPER_LIMIT_Q = 4.7997
        self.RANGE_BASE = (self.BASE_UPPER_LIMIT_Q - self.BASE_LOWER_LIMIT_Q) * 0.1
        self.SHOULDER_LOWER_LIMIT_Q = -2.3562
        self.SHOULDER_UPPER_LIMIT_Q = -0.6109
        self.RANGE_SHOULDER = (self.SHOULDER_UPPER_LIMIT_Q - self.SHOULDER_LOWER_LIMIT_Q) * 0.25
        self.ELBOW_LOWER_LIMIT_Q = 1.0472
        self.ELBOW_UPPER_LIMIT_Q = 2.7925
        self.RANGE_ELBOW = (self.ELBOW_UPPER_LIMIT_Q - self.ELBOW_LOWER_LIMIT_Q) * 0.25
        self.WRIST1_LOWER_LIMIT_Q = -1.2217
        self.WRIST1_UPPER_LIMIT_Q = 0.0000
        self.RANGE_WRIST1 = (self.WRIST1_UPPER_LIMIT_Q - self.WRIST1_LOWER_LIMIT_Q) * 0.25
        self.WRIST2_LOWER_LIMIT_Q = -0.5236
        self.WRIST2_UPPER_LIMIT_Q = 0.3491
        self.RANGE_WRIST2 = (self.WRIST2_UPPER_LIMIT_Q - self.WRIST2_LOWER_LIMIT_Q) * 0.25
        self.WRIST3_LOWER_LIMIT_Q = -1.5708
        self.WRIST3_UPPER_LIMIT_Q = 3.1416

        self.LOWER_LIMIT_Q = np.array([self.BASE_LOWER_LIMIT_Q, self.SHOULDER_LOWER_LIMIT_Q, self.ELBOW_LOWER_LIMIT_Q, \
                                       self.WRIST1_LOWER_LIMIT_Q, self.WRIST2_LOWER_LIMIT_Q, self.WRIST3_LOWER_LIMIT_Q])

        self.UPPER_LIMIT_Q = np.array([self.BASE_UPPER_LIMIT_Q, self.SHOULDER_UPPER_LIMIT_Q, self.ELBOW_UPPER_LIMIT_Q, \
                                       self.WRIST1_UPPER_LIMIT_Q, self.WRIST2_UPPER_LIMIT_Q, self.WRIST3_UPPER_LIMIT_Q])

        self.control = rtde_control.RTDEControlInterface(self.HOST)
        self.receive = rtde_receive.RTDEReceiveInterface(self.HOST)
        self.dashboard = dashboard_client.DashboardClient(self.HOST)

        self.action_space = gym.spaces.Discrete(10)  # clockwise or counterclockwise, for each of the 5 moving joints
        # self.observation_space = gym.spaces.Box(low=self.LOWER_LIMIT_Q, high=self.UPPER_LIMIT_Q,
        #                                         shape=(6,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(N_CHANNELS, self.HEIGHT, self.WIDTH), dtype=np.uint8)
        # UNCOMMENT TO REVERT TO PRE-OBSERVATION DICTIONARY
        # self.observation_space = gym.spaces.Box(low=self.LOWER_LIMIT_Q, high=self.UPPER_LIMIT_Q,
        #                                         shape=(6,), dtype=np.float32)
        spaces = {
            'image': gym.spaces.Box(low=0, high=255, shape=(N_CHANNELS, self.HEIGHT, self.WIDTH), dtype=np.uint8)
        }
        if 'image_joint' in self.obs_space_type:
            print("to be completed")
            self.LOWER_LIMIT_Q = np.array([self.BASE_LOWER_LIMIT_Q, self.SHOULDER_LOWER_LIMIT_Q, self.ELBOW_LOWER_LIMIT_Q, self.WRIST1_LOWER_LIMIT_Q, self.WRIST2_LOWER_LIMIT_Q, self.WRIST3_LOWER_LIMIT_Q])
            self.UPPER_LIMIT_Q = np.array([self.BASE_UPPER_LIMIT_Q, self.SHOULDER_UPPER_LIMIT_Q, self.ELBOW_UPPER_LIMIT_Q, self.WRIST1_UPPER_LIMIT_Q, self.WRIST2_UPPER_LIMIT_Q, self.WRIST3_UPPER_LIMIT_Q])
            spaces['joint'] = gym.spaces.Box(low=self.LOWER_LIMIT_Q, high=self.UPPER_LIMIT_Q, shape=(6,), dtype=np.float32)
        if self.obs_space_type == 'image_joint_action':
            spaces['action'] = gym.spaces.Discrete(10)
        
        self.observation = gym.spaces.Dict(spaces)

        time.sleep(1)
        self.reconnect()
        self.state = None
        self.image_state = None

        self.root = tkinter.Tk()
        if self.env_type == "reaching":
            self.visualizer = TargetDisplay(self.root, env_type="reaching")
        elif self.env_type == "tracking":
            self.visualizer = TargetDisplay(self.root, env_type="tracking")
        else:
            # the Reacher, but with a static dot position (target never moves)
            self.visualizer = TargetDisplay(self.root, env_type="static", pos=self.target_position)

    def step(self, action):
        if self.state is None:
            raise ValueError("Please call reset first.")
        new_pos = self.state.copy()  # this is a 6D array of the joint positions (from getActualQ())
        new_pos_coord = self.receive.getActualTCPPose()
        last_dist_to_goal = np.sum(np.abs(np.array(new_pos_coord[:3]) - np.array(self.GOAL_COORD[:3])))

        within_limits = True

        if action == self.BASE_CLOCKWISE:
            new_pos[0] += self.STEP_SIZE
            # is this going ABOVE the upper limit - 25% ?
            if new_pos[0] > (self.BASE_UPPER_LIMIT_Q - self.RANGE_BASE):
                within_limits = False
        elif action == self.BASE_COUNTER_CLOCKWISE:
            new_pos[0] -= self.STEP_SIZE
            # is this going BELOW the lower limit + 25% ?
            if new_pos[0] < (self.BASE_LOWER_LIMIT_Q + self.RANGE_BASE):
                within_limits = False

        elif action == self.SHOULDER_CLOCKWISE:
            new_pos[1] += self.STEP_SIZE
            if new_pos[1] > (self.SHOULDER_UPPER_LIMIT_Q - self.RANGE_SHOULDER):
                within_limits = False
        elif action == self.SHOULDER_COUNTER_CLOCKWISE:
            new_pos[1] -= self.STEP_SIZE
            if new_pos[1] < (self.SHOULDER_LOWER_LIMIT_Q + self.RANGE_SHOULDER):
                within_limits = False

        elif action == self.ELBOW_CLOCKWISE:
            new_pos[2] += self.STEP_SIZE
            if new_pos[2] > (self.ELBOW_UPPER_LIMIT_Q - self.RANGE_ELBOW):
                within_limits = False
        elif action == self.ELBOW_COUNTER_CLOCKWISE:
            new_pos[2] -= self.STEP_SIZE
            if new_pos[2] < (self.ELBOW_LOWER_LIMIT_Q + self.RANGE_ELBOW):
                within_limits = False

        elif action == self.WRIST1_CLOCKWISE:
            new_pos[3] += self.STEP_SIZE
            if new_pos[3] > (self.WRIST1_UPPER_LIMIT_Q - self.RANGE_WRIST1):
                within_limits = False
        elif action == self.WRIST1_COUNTER_CLOCKWISE:
            new_pos[3] -= self.STEP_SIZE
            if new_pos[3] < (self.WRIST1_LOWER_LIMIT_Q + self.RANGE_WRIST1):
                within_limits = False

        elif action == self.WRIST2_CLOCKWISE:
            new_pos[4] += self.STEP_SIZE
            if new_pos[4] > (self.WRIST2_UPPER_LIMIT_Q - self.RANGE_WRIST2):
                within_limits = False
        elif action == self.WRIST2_COUNTER_CLOCKWISE:
            new_pos[4] -= self.STEP_SIZE
            if new_pos[4] < (self.WRIST2_LOWER_LIMIT_Q + self.RANGE_WRIST2):
                within_limits = False

        else:
            raise ValueError("Invalid action")

        # Check if the action brings new_pos within the 25% accepted range of the limits
        # if not, revert to previous state
        if not within_limits:
            new_pos = self.state.copy()
        # if new_pos is within the limits, ok
        else:
            new_pos = new_pos

        self.LAST_STATE = self.state.copy()  # save the last state before we take the action
        self.LAST_ACTION = action
        self.LAST_IMAGE_STATE = self.image_state.copy()

        success = self.control.moveJ(new_pos)
        while not success:
            if self.receive.isProtectiveStopped():
                self.reconnect()
                problem, protective_stop, joint, stop_type = self.stop_type()
                time.sleep(6)  # cannot unlock protective stop before it has been stopped for 5 seconds
                self.dashboard.unlockProtectiveStop()
                print("was protective stopped")

                info = {}
                obs = self.image_state.copy()
                self.state = None
                self.image_state = None
                done = True  # should this come before of after?
                reward = -20
                info["ProtectiveStopTerminated"] = True

                pos = self.receive.getActualTCPPose()
                self.dist_to_goal = np.sum(np.abs(np.array(pos[:3]) - np.array(self.GOAL_COORD[:3])))
                if self.log_state_actions:
                    self.log_to_file(action, reward, self.dist_to_goal, protective_stop=protective_stop,
                                     joint_stuck=joint,
                                     type_of_stop=stop_type)
                self.episode_count += 1
                self.count = 0
                # return np.resize(obs.copy(), (HEIGHT, WIDTH, N_CHANNELS)), reward, done, info  # the observation returned is in the joint space
                return obs, reward, done, info,

            else:
                self.reconnect()
                success = self.control.moveJ(new_pos)
        self.state = new_pos

        # if we are in a tracking environment, we move every time step
        if self.env_type == "tracking":
            self.visualizer.move(env_type="tracking")

        catchCount = 0
        while catchCount < 3:
            try:
                #UNCOMMENT TO REVERT TO BLACK AND WHITE
                # im = fc2.Image()
                # self.image_state = cv2.resize(np.array(self.CAMERA.retrieve_buffer(im)), (0, 0), fx=0.25, fy=0.25)
                # self.image_state = np.resize(self.image_state,
                #                              (N_CHANNELS, self.HEIGHT, self.WIDTH))
                im = fc2.Image()
                self.image_state = cv2.resize(np.array(self.CAMERA.retrieve_buffer(im)), (0, 0), fx=0.25, fy=0.25)
                self.image_state = cv2.cvtColor(self.image_state, cv2.COLOR_RGB2BGR)
                self.image_state = np.resize(self.image_state,
                                             (N_CHANNELS, self.HEIGHT, self.WIDTH))
                break
            except:
                print("Oops! Image capture failed (STEP).  Try again...")
        if catchCount == 3:
            self.image_state = self.LAST_IMAGE_STATE
            print("CatchCount == 3 (STEP), setting the last image as current ")
        # TODO: Figure out why the FC2_ERROR_IMAGE_CONSISTENCY_ERROR error is raised
        # TODO: Do we need to empty the buffer?
        # TODO: How can we pull the last n most recent images out of the buffer?
        #  for the moving dot especially
        #  see: install flycap from https://github.com/ethanlarochelle/pyflycapture2
        
        # UNCOMMENT TO REVERT TO PRE-OBSERVATION DICTIONARY
        # obs = self.image_state.copy()
        obs = {
            'image': self.image_state.copy()
        }
        if 'image_joint' in self.obs_space_type:
            obs['joint'] = self.state
        if self.obs_space_type == 'image_joint_action':
            obs['action'] = action
        
        obs_coord = self.receive.getActualTCPPose()
        # make the distance negative to turn it into a reward
        self.dist_to_goal = np.sum(np.abs(np.array(obs_coord[:3]) - np.array(self.GOAL_COORD[:3])))

        # if the tool's x, y, z coordinates are within the threshold region
        if self.dist_to_goal < self.GOAL_THRESHOLD:
            # reward = 100  # we provide a positive reward
            reward = 20  # normalized reward
            done = True
            print("GOAL ATTAINED")
            self.state = None
            info = {}
            if self.log_state_actions:
                self.log_to_file(action, reward, self.dist_to_goal)
            self.count = 0
            self.episode_count += 1
            # return np.resize(obs.copy(), (HEIGHT, WIDTH, N_CHANNELS)), reward, done, info  # the observation returned is in the joint space
            return obs, reward, done, info
        else:
            reward = -1  # INITIAL APPROACH: Penalizing the time step
            # reward = -self.dist_to_goal # OPTION 1: penalizing distance from goal
            # reward = (last_dist_to_goal-self.dist_to_goal) / self.STEPS_IN_EPISODE
            # OPTION 2: reward getting closer and penalize getting farther from goal, with additional timestep penalty
            done = False
        # TODO: Make the reward based on the visual circle

        if self.log_state_actions:
            self.log_to_file(action, reward, self.dist_to_goal)

        info = {}
        self.count += 1

        if self.count == self.STEPS_IN_EPISODE:
            print("episode " + str(self.episode_count) + " terminated")
            self.count = 0
            self.episode_count += 1
            done = True
            info["TimeLimit.truncated"] = True

        if self.save_images:
            cv2.imwrite("saved_images/" + self.file_name_prefix + "/visualReacherFiveJointsGoal_ep" +
                        str(self.episode_count) + "_step" +
                        str(self.count) + ".jpeg", self.image_state)
        # return np.resize(obs.copy(), (HEIGHT, WIDTH, N_CHANNELS)), reward, done, info  # the observation returned is in the joint space
        # self.render(mode="human")
        return obs, reward, done, info

    # Function that writes into file
    def log_to_file(self, action, reward, distance_to_goal, protective_stop=False, joint_stuck=None, type_of_stop=None):
        """For every step, write the action and the Q position, was there a move or not, (POSITION | ACTION | SUCCESS)
        with a new line for each new episode. At the end of the function close the file."""

        f = open(self.log_file_path, 'a')
        writer = csv.writer(f)
        data = [self.episode_count, self.count, self.state, self.receive.getActualTCPPose(), action, reward,
                distance_to_goal, protective_stop, joint_stuck, type_of_stop]
        writer.writerow(data)
        f.close()

    def move_up(self, reset=True):
        """If a protective stop is caused by the table plane, move out of that plane.
        In our case, we can only bump into the z plane, so we always try to move up."""

        print("Robot was stuck getting too close to the table, calling move_up")
        new_pos_coord = self.receive.getActualTCPPose()
        i = 1
        success = False
        while not success and i < 3:
            new_pos_coord[2] += 0.025 * i
            self.control.disconnect()
            self.control.reconnect()
            success = self.control.moveL(new_pos_coord)
            if success:
                print("move_up successful, will now reset")
            else:
                self.dashboard.unlockProtectiveStop()
            i += 1

        if i == 3 and not success:
            print("3 attempts to move up were unsuccessful. "
                  "Please try to move up using manual mode")
        if reset:
            self.go_to_start()

    def move_back(self, reset=True):
        """If possible, this performs the inverse of the last move that was performed. It returns the actual position
        after a move, if it was successful"""

        print("Robot was stuck in illegal joint position, calling move_back")
        new_pos = self.receive.getActualQ()
        i = 1
        success = False
        while not success and i < 5:
            # Check the base
            if new_pos[0] < self.BASE_LOWER_LIMIT_Q + 0.05:  # we are stuck at lower limit of the base
                new_pos[0] += self.STEP_SIZE * i  # perform the opposite and move clockwise a good enough distance.
            elif new_pos[0] > self.BASE_UPPER_LIMIT_Q - 0.05:  # we are stuck at upper limit of the base
                new_pos[0] -= self.STEP_SIZE * i  # perform the opposite and move counter-clockwise
            # Check the shoulder
            if new_pos[1] < self.SHOULDER_LOWER_LIMIT_Q + 0.05:  # we are stuck at lower limit of the shoulder
                new_pos[1] += self.STEP_SIZE * i  # perform the opposite and move clockwise a good enough distance.
            elif new_pos[1] > self.SHOULDER_UPPER_LIMIT_Q - 0.05:  # we are stuck at upper limit of the shoulder
                new_pos[1] -= self.STEP_SIZE * i  # perform the opposite and move counter-clockwise
            # Check the elbow
            if new_pos[2] < self.ELBOW_LOWER_LIMIT_Q + 0.05:  # we are stuck at lower limit of the elbow
                new_pos[2] += self.STEP_SIZE * i  # perform the opposite and move clockwise a good enough distance.
            elif new_pos[2] > self.ELBOW_UPPER_LIMIT_Q - 0.05:  # we are stuck at upper limit of the elbow
                new_pos[2] -= self.STEP_SIZE * i  # perform the opposite and move counter-clockwise
            # Check WRIST1
            if new_pos[3] < self.WRIST1_LOWER_LIMIT_Q + 0.05:  # we are stuck at lower limit of WRIST1
                new_pos[3] += self.STEP_SIZE * i  # perform the opposite and move clockwise a good enough distance.
            elif new_pos[3] > self.WRIST1_UPPER_LIMIT_Q - 0.05:  # we are stuck at upper limit of WRIST1
                new_pos[3] -= self.STEP_SIZE * i  # perform the opposite and move counter-clockwise
            # Check WRIST2
            if new_pos[4] < self.WRIST2_LOWER_LIMIT_Q + 0.05:  # we are stuck at lower limit of WRIST2
                new_pos[4] += self.STEP_SIZE * i  # perform the opposite and move clockwise a good enough distance.
            elif new_pos[4] > self.WRIST2_UPPER_LIMIT_Q - 0.05:  # we are stuck at upper limit of WRIST2
                new_pos[4] -= self.STEP_SIZE * i  # perform the opposite and move counter-clockwise
            # Check WRIST3
            if new_pos[5] < self.WRIST3_LOWER_LIMIT_Q + 0.05:  # we are stuck at lower limit of WRIST3
                new_pos[5] += self.STEP_SIZE * i  # perform the opposite and move clockwise a good enough distance.
            elif new_pos[5] > self.WRIST3_UPPER_LIMIT_Q - 0.05:  # we are stuck at upper limit of WRIST3
                new_pos[5] -= self.STEP_SIZE * i  # perform the opposite and move counter-clockwise

            # these next two lines have proven to be essential by testing
            self.control.disconnect()
            self.control.reconnect()
            success = self.control.moveJ(new_pos)
            if success:
                print("move_back successful, will now reset")
                # return self.receive.getActualQ()
            i += 1

        if i == 5:
            print("5 attempts to move out of the restricted area were unsuccessful. "
                  "Please try to move back using the manual mode")
        if reset:
            self.go_to_start()

    def stop_type(self):
        """ SHOULD ONLY BE CALLED IF IN A PROTECTIVE STOP.
        Assuming the robot is in a protective stop, this returns an integer representing
        the type of stop that is occurring: 1 if this is a joint protective stop and
        2 if there is a plane protective stop"""

        pos = self.receive.getActualQ()
        stop_type_int = 1
        stop_type_str = "Joint protective stop"
        joint_stop = False
        # Check the base
        if pos[0] < self.BASE_LOWER_LIMIT_Q + 0.05:  # we are stuck at lower limit of the base
            joint_stop = True
            joint = 0
        elif pos[0] > self.BASE_UPPER_LIMIT_Q - 0.05:  # we are stuck at upper limit of the base
            joint_stop = True
            joint = 0
        # Check the shoulder
        if pos[1] < self.SHOULDER_LOWER_LIMIT_Q + 0.05:  # we are stuck at lower limit of the shoulder
            joint_stop = True
            joint = 1
        elif pos[1] > self.SHOULDER_UPPER_LIMIT_Q - 0.05:  # we are stuck at upper limit of the shoulder
            joint_stop = True
            joint = 1
        # Check the elbow
        if pos[2] < self.ELBOW_LOWER_LIMIT_Q + 0.05:  # we are stuck at lower limit of the elbow
            joint_stop = True
            joint = 2
        elif pos[2] > self.ELBOW_UPPER_LIMIT_Q - 0.05:  # we are stuck at upper limit of the elbow
            joint_stop = True
            joint = 2
        # Check WRIST1
        if pos[3] < self.WRIST1_LOWER_LIMIT_Q + 0.05:  # we are stuck at lower limit of WRIST1
            joint_stop = True
            joint = 3
        elif pos[3] > self.WRIST1_UPPER_LIMIT_Q - 0.05:  # we are stuck at upper limit of WRIST1
            joint_stop = True
            joint = 3
        # Check WRIST2
        if pos[4] < self.WRIST2_LOWER_LIMIT_Q + 0.05:  # we are stuck at lower limit of WRIST2
            joint_stop = True
            joint = 4
        elif pos[4] > self.WRIST2_UPPER_LIMIT_Q - 0.05:  # we are stuck at upper limit of WRIST2
            joint_stop = True
            joint = 4
        # Check WRIST3
        if pos[5] < self.WRIST3_LOWER_LIMIT_Q + 0.05:  # we are stuck at lower limit of WRIST3
            joint_stop = True
            joint = 5
        elif pos[5] > self.WRIST3_UPPER_LIMIT_Q - 0.05:  # we are stuck at upper limit of WRIST3
            joint_stop = True
            joint = 5

        # if no joints are off limits, then we declare a plane protective stop
        if not joint_stop:
            stop_type_int = 2
            joint = None
            stop_type_str = "Plane protective stop"

        return stop_type_int, True, joint, stop_type_str

    def reconnect(self):

        while not self.control.isConnected():
            print("Control not connected, reconnecting...")
            self.control = rtde_control.RTDEControlInterface(self.HOST)
            time.sleep(5)

        while not self.receive.isConnected():
            print("Receive not connected, reconnecting...")
            self.receive = rtde_receive.RTDEReceiveInterface(self.HOST)
            time.sleep(5)

        while not self.dashboard.isConnected():
            print("Dashboard not connected, reconnecting...")
            self.dashboard = dashboard_client.DashboardClient(self.HOST)
            self.dashboard.connect()
            time.sleep(5)

    def go_to_start(self):
        """Goes back to starting position"""
        if self.RANDOM_START == 0:
            start_q = self.FIXED_START
        else:
            start_q = [
                random.uniform(self.BASE_LOWER_LIMIT_Q + self.RANGE_BASE, self.BASE_UPPER_LIMIT_Q - self.RANGE_BASE),
                random.uniform(self.SHOULDER_LOWER_LIMIT_Q + self.RANGE_SHOULDER,
                               self.SHOULDER_UPPER_LIMIT_Q - self.RANGE_SHOULDER),
                random.uniform(self.ELBOW_LOWER_LIMIT_Q + self.RANGE_ELBOW,
                               self.ELBOW_UPPER_LIMIT_Q - self.RANGE_ELBOW),
                random.uniform(self.WRIST1_LOWER_LIMIT_Q + self.RANGE_WRIST1,
                               self.WRIST1_UPPER_LIMIT_Q - self.RANGE_WRIST1),
                random.uniform(self.WRIST2_LOWER_LIMIT_Q + self.RANGE_WRIST2,
                               self.WRIST2_UPPER_LIMIT_Q - self.RANGE_WRIST2),
                0.16946229338645935]
        self.control.moveJ(start_q)

    def reset(self):
        print("reset has been called")
        self.visualizer.reset(env_type=self.env_type)
        if self.RANDOM_START == 0:
            start_q = self.FIXED_START
        else:
            start_q = [
                random.uniform(self.BASE_LOWER_LIMIT_Q + self.RANGE_BASE, self.BASE_UPPER_LIMIT_Q - self.RANGE_BASE),
                random.uniform(self.SHOULDER_LOWER_LIMIT_Q + self.RANGE_SHOULDER,
                               self.SHOULDER_UPPER_LIMIT_Q - self.RANGE_SHOULDER),
                random.uniform(self.ELBOW_LOWER_LIMIT_Q + self.RANGE_ELBOW,
                               self.ELBOW_UPPER_LIMIT_Q - self.RANGE_ELBOW),
                random.uniform(self.WRIST1_LOWER_LIMIT_Q + self.RANGE_WRIST1,
                               self.WRIST1_UPPER_LIMIT_Q - self.RANGE_WRIST1),
                random.uniform(self.WRIST2_LOWER_LIMIT_Q + self.RANGE_WRIST2,
                               self.WRIST2_UPPER_LIMIT_Q - self.RANGE_WRIST2),
                0.16946229338645935]

        success = self.control.moveJ(start_q)
        count = 0
        problem = 0
        while not success and count < 3:
            self.reconnect()
            if self.receive.isProtectiveStopped():
                problem, _, _, _ = self.stop_type()  # this logs the type of stop in the file as well
                if problem == 1:
                    print("A joint protective stop prevents resetting, will try again")
                else:
                    print("A plane protective stop prevents resetting, will try again")
                time.sleep(6)  # cannot unlock protective stop before it has been stopped for 5 seconds
                self.dashboard.unlockProtectiveStop()
                if self.RANDOM_START == 0:
                    start_q = self.FIXED_START
                # if the robot could not move into the given start coordinate, we try a new one
                else:
                    start_q = [
                        random.uniform(self.BASE_LOWER_LIMIT_Q + self.RANGE_BASE,
                                       self.BASE_UPPER_LIMIT_Q - self.RANGE_BASE),
                        random.uniform(self.SHOULDER_LOWER_LIMIT_Q + self.RANGE_SHOULDER,
                                       self.SHOULDER_UPPER_LIMIT_Q - self.RANGE_SHOULDER),
                        random.uniform(self.ELBOW_LOWER_LIMIT_Q + self.RANGE_ELBOW,
                                       self.ELBOW_UPPER_LIMIT_Q - self.RANGE_ELBOW),
                        random.uniform(self.WRIST1_LOWER_LIMIT_Q + self.RANGE_WRIST1,
                                       self.WRIST1_UPPER_LIMIT_Q - self.RANGE_WRIST1),
                        random.uniform(self.WRIST2_LOWER_LIMIT_Q + self.RANGE_WRIST2,
                                       self.WRIST2_UPPER_LIMIT_Q - self.RANGE_WRIST2),
                        0.16946229338645935]
            self.control.reuploadScript()  # is this necessary ?
            success = self.control.moveJ(start_q)
            count += 1

        if count == 3:
            if problem == 1:
                self.move_back()
            elif problem == 2:
                self.move_up()
            else:
                pass

        self.state = self.receive.getActualQ()  # this is a 6D array, including the position of wrists 2 & 3
        while self.state == []:
            print("State is empty...")
            self.reconnect()
            self.state = self.receive.getActualQ()

        # get frame from camera
        catchCount = 0
        loopFlag = True
        while loopFlag:
            try:
                #UNCOMMENT TO REVERT TO BLACK AND WHITE
                # im = fc2.Image()
                # self.image_state = cv2.resize(np.array(self.CAMERA.retrieve_buffer(im)), (0, 0), fx=0.25, fy=0.25)
                # self.image_state = np.resize(self.image_state,
                #                              (N_CHANNELS, self.HEIGHT, self.WIDTH))
                im = fc2.Image()
                self.image_state = cv2.resize(np.array(self.CAMERA.retrieve_buffer(im)), (0, 0), fx=0.25, fy=0.25)
                self.image_state = cv2.cvtColor(self.image_state, cv2.COLOR_RGB2BGR)
                self.image_state = np.resize(self.image_state,
                                             (N_CHANNELS, self.HEIGHT, self.WIDTH))
                loopFlag = False
                break
            except:
                print("Oops! Image capture failed (RESET).  Try again...")
            if catchCount == 3 and loopFlag:
                catchCount = 0
                print("disconnecting camera")
                self.CAMERA.stop_capture()
                self.CAMERA.disconnect()
                print("connecting camera")
                self.CAMERA.connect(*self.CAMERA.get_camera_from_index(0))
                print("Starting capture")
                self.CAMERA.start_capture()
            catchCount += 1

        # self.image_state = np.array(self.CAMERA.retrieve_buffer(im))
        if self.save_images:
            cv2.imwrite("saved_images/" + self.file_name_prefix + "/visualReacherFiveJointsGoal_ep" +
                        str(self.episode_count) + "_step" + str(self.count) + ".jpeg",
                        self.image_state)
        # return np.resize(self.image_state.copy(), (HEIGHT, WIDTH, N_CHANNELS))
        # self.render(mode="human")
        self.LAST_IMAGE_STATE = self.image_state
        # UNCOMMENT TO REVERT TO PRE-OBSERVATION DICTIONARY
        # obs = self.image_state.copy()
        obs = {
            'image': self.image_state.copy()
        }
        if 'image_joint' in self.obs_space_type:
            obs['joint'] = self.state
        if self.obs_space_type == 'image_joint_action':
            obs['action'] = 0
        return obs
        # return self.image_state.copy()

    def render(self, mode="human"):
        #UNCOMMENT TO REVERT TO BLACK AND WHITE
        # if mode == "human":
        #     cv2.imshow('frame', self.image_state[-1, :, :])
        #     cv2.waitKey(1000)
        if mode == "human":
            cv2.imshow('frame', self.image_state)
            cv2.waitKey(10)
        elif mode == 'rgb_array':
            return self.image_state

    def close(self):
        self.CAMERA.stop_capture()
        self.CAMERA.disconnect()
        self.control.disconnect()
        self.receive.disconnect()
        self.dashboard.disconnect()
        self.visualizer.on_closing()
        self.root = None
        self.visualizer = None