"""
Sim2Real Wrapper helper functions for Jaco Kinova arm.
"""

import time
import numpy as np
from robosuite.wrappers import Wrapper
import socket

from copy import deepcopy
from collections import deque
import numpy as np
import json
from skimage.transform import resize

from robosuite.utils import transform_utils
from PIL import Image


class RobotClient():
    def __init__(self, robot_ip="127.0.0.1", port=9030):
        self.robot_ip = robot_ip
        self.port = port
        self.connected = False
        self.startseq = '<|'
        self.endseq = '|>'
        self.midseq = '**'

    def connect(self):
        while not self.connected:
            print("attempting to connect with robot at {}".format(
                self.robot_ip))
            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR,
                                       1)
            self.tcp_socket.settimeout(100)
            # connect to computer
            self.tcp_socket.connect((self.robot_ip, self.port))
            print('connected')
            self.connected = True
            if not self.connected:
                time.sleep(1)

    def decode_state(self, robot_response, verbose=False):
        # Returned states include the following:
        # Joint Positions  (13)
        # Joint Velocities (13)
        # joint_effort     (13)
        # tool_pose        (7)
        # tool_pose_orig   (7)
        # finger_pose      (1)
        # Total obs dim :  54
        if verbose:
            print('decoding', robot_response)
        ackmsg, resp = robot_response.split('**')
        # successful msg has ACKSTEP
        assert ackmsg[:5] == '<|ACK'
        # make sure we got msg end
        assert resp[-2:] == '|>'
        vals = [x.split(': ')[1] for x in resp[:-2].split('\n')]
        # deal with each data type
        success = bool(vals[0])
        robot_msg = eval(vals[1])
        # not populated
        joint_names = vals[2]
        # num states seen in this step
        self.n_state_updates = int(vals[3])
        timediff = json.loads(vals[4])[-1]
        joint_position = json.loads(vals[5])
        joint_velocity = json.loads(vals[6])
        joint_effort = json.loads(vals[7])
        tool_pose = json.loads(vals[8])

        finger_pose = json.loads(vals[9])
        return timediff, joint_position, joint_velocity, joint_effort, tool_pose, finger_pose

    def send(self, cmd, msg='XX'):
        packet = self.startseq + cmd + self.midseq + msg + self.endseq
        self.tcp_socket.sendall(packet.encode())
        self.tcp_socket.settimeout(100)
        rx = self.tcp_socket.recv(2048).decode()
        return rx

    def render(self):
        packet = self.startseq + "RENDER" + self.midseq + "XX" + self.endseq
        self.tcp_socket.settimeout(100)
        self.tcp_socket.sendall(packet.encode())
        self.tcp_socket.settimeout(100)
        rxl = []
        rxing = True
        cnt = 0
        end = self.endseq.encode()
        while rxing:
            rx = self.tcp_socket.recv(2048)
            rxl.append(rx)
            cnt += 1
            # byte representation of endseq
            if rx[-2:] == end:
                rxing = False
        allrx = b''.join(rxl)[2:-2]
        # height, width
        # The image height and width mysteriously changed
        # Ideally figure this out from the msgs
        #img = np.frombuffer(allrx, dtype=np.uint8).reshape(480, 640, 3)
        img = np.frombuffer(allrx, dtype=np.uint8).reshape(720, 1280, 3)

        return img

    def home(self):
        return self.send('HOME')

    def reset(self):
        print('Robot Client sending reset')
        return self.decode_state(self.send('RESET'))

    def get_state(self):
        return self.decode_state(self.send('GET_STATE'))

    def initialize(self, minx=-10, maxx=10, miny=-10, maxy=10, minz=0.05, maxz=10):
        data = '{},{},{},{},{},{}'.format(minx, maxx, miny, maxy, minz, maxz)
        return self.decode_state(self.send('INIT', data))

    def end(self):
        self.send('END')
        print('disconnected from {}'.format(self.robot_ip))
        self.tcp_socket.close()
        self.connected = False

    def step(self, command_type, relative, unit, data):
        assert (command_type in ['VEL', 'ANGLE', 'TOOL'])
        datastr = ','.join(['%.4f' % x for x in data])
        data = '{},{},{},{}'.format(command_type, int(relative), unit, datastr)
        return self.decode_state(self.send('STEP', data))
 

class JacoSim2RealWrapper(Wrapper):
    def __init__(self, sim_env, robot_server_ip='127.0.0.1', robot_server_port=9030, max_step=np.deg2rad(10), stack_frame=3):
        """
        Initializes the data collection wrapper.

        Args:
            env (MujocoEnv): The environment to monitor.
        """

        super().__init__(sim_env)
        self.sim_env = sim_env
        self.robot_server_ip = robot_server_ip
        self.robot_server_port = robot_server_port
        self.robot_client = RobotClient(robot_ip=self.robot_server_ip,
                                        port=self.robot_server_port)
        self.max_step = max_step
        self.robot_client.connect()
        self.robot_client.initialize()
        self.image_dict = {
            'enc': 'none',
            'width': 0,
            'height': 0,
            'data': 'none'
        }

        self.sim_states = {
            "joint_pose": None,
            "joint_vel": None,
            "effort": None,
            "eef_pos": None,
            "eef_quat": None,
            "finger_pose": None,
            "image_frame": None,
            "stacked_img_frames": None
        }

        self.real_states = {
            "timediff": None,
            "joint_pose": None,
            "joint_vel": None,
            "effort": None,
            "eef_pos": None,
            "eef_quat": None,
            "finger_pose": None,
            "image_frame": None,
            "stacked_img_frames": None
        }

        self._k = stack_frame
        self._real_frames = deque([], maxlen=self._k)
        # whether to use real robot observations or not
        # typicall I use sim_obs for debugging purposes.
        self.use_real_pix_obs = True

        self.frame_height = self.sim_env.camera_heights[0]
        self.frame_width = self.sim_env.camera_widths[0]

        self.keys = [f"{cam_name}_image" for cam_name in self.sim_env.camera_names]

        # control_type options are 'VEL', 'ANGLE', 'TOOL'
        control_type_map = {'JOINT_POSITION':'ANGLE', 
                            'OSC_POSE':'TOOL',
                            'OSC_POSITION':'TOOL'}
 
        self.n_joints = 7
        self.sim_control_type = self.env.robots[0].controller.name
        self.control_type = control_type_map[self.sim_control_type]

        if self.sim_control_type == 'JOINT_POSITION':
            self.control_relative = True
        else:
            self.control_relative = self.sim_env.robots[0].controller.use_delta
        # only relative is tested (much safer on real robot)
        assert self.control_relative == True
        self.control_unit = 'mrad'

    def get_sim_posquat(self):
        sim_eef_pose = deepcopy(self.sim_env.robots[0].pose_in_base_from_name('gripper0_eef'))
        sim_eef_pos = deepcopy(sim_eef_pose)[:3, 3]
        sim_eef_quat = deepcopy(transform_utils.mat2quat(sim_eef_pose))
        return sim_eef_pos, sim_eef_quat
    
    def get_sim2real_posquat(self):
        sim_eef_pose = deepcopy(self.sim_env.robots[0].pose_in_base_from_name('gripper0_eef'))
        angle = np.deg2rad(-90)
        direction_axis = [0, 0, 1]
        rotation_matrix = transform_utils.rotation_matrix(angle, direction_axis)
    
        sim_pose_rotated = np.dot(rotation_matrix, sim_eef_pose)
        sim_eef_pos_rotated = deepcopy(sim_pose_rotated)[:3, 3]
        sim_eef_quat_rotated = deepcopy(transform_utils.mat2quat(sim_pose_rotated))
        return sim_eef_pos_rotated, sim_eef_quat_rotated
    
    # def get_real2sim_posquat(self, pos, quat):
    #     real_eef_pose = transform_utils.pose2mat((pos,quat))
    #     angle = np.deg2rad(90)
    #     direction_axis = [0, 0, 1]
    #     rotation_matrix = transform_utils.rotation_matrix(angle, direction_axis)

    #     real_pose_rotated = np.dot(rotation_matrix, real_eef_pose)
    #     real_eef_pos_rotated = deepcopy(real_pose_rotated)[:3, 3]
    #     real_eef_quat_rotated = deepcopy(transform_utils.mat2quat(real_pose_rotated))
    #     return real_eef_pos_rotated, real_eef_quat_rotated

    def get_real2sim_posquat(self, pos, quat, angle=90):
        rotation_q = transform_utils.quat2mat(quat)
        pose = transform_utils.make_pose(pos, rotation_q)
        angle = np.deg2rad(angle)
        direction_axis = [0, 0, 1]
        rotation_matrix = transform_utils.rotation_matrix(
            angle, direction_axis)

        real_pose_rotated = np.dot(rotation_matrix, pose)
        real_eef_pos_rotated = real_pose_rotated[:3, 3]
        # The rotation doesn't work currently it keeps rotation
        # So don't send it for now
        # real_eef_quat_rotated = transform_utils.mat2quat(real_pose_rotated)

        return real_eef_pos_rotated, quat


    def handle_state(self, state_tuple):
        timediff, joint_position, joint_velocity, joint_effort, tool_pose, finger_pose = state_tuple
        self.timediff = timediff

        self.real_states["joint_pose"] = np.array(joint_position)[:self.n_joints]
        self.real_states["joint_vel"] = np.array(joint_velocity)[:self.n_joints]
        self.real_states["effort"] = np.array(joint_effort)

        converted_pose, converted_quat = self.get_real2sim_posquat(tool_pose[:3], tool_pose[3:], angle=90)

        self.real_states["eef_pos"]= converted_pose
        self.real_states["eef_quat"]= converted_quat
        # Note: The actual finger joint poses are part of join_position
        # finger_pose return here gives finger angles which aren't used in sim.
        self.real_states["finger_pose"] = np.array(joint_position)[7:10]

        return self.real_states

    def reset(self):
        """
        Returns:
            OrderedDict: Environment observation space after reset occurs
        """
        sim_ret = super().reset()
        self.real_states = self.handle_state(self.robot_client.reset())
        self.sim_states["sim_ret"] = sim_ret
        # Extra sim observations
        self.sim_states["joint_pose"] = self.sim_env.sim.data.qpos[self.sim_env.robots[0]._ref_joint_pos_indexes]
        self.sim_states["finger_pose"] = self.sim_env.sim.data.qpos[self.sim_env.robots[0]._ref_joint_gripper_actuator_indexes]

        obs_pix_sim, obs_pix_raw = self._get_sim_image_obs(sim_ret)
        self.sim_states["image_frame"] = obs_pix_raw

        if self.use_real_pix_obs:
            # Real image frames
            obs_pix_real = self.robot_client.render()
            obs_pix_real_train = self.prepare_for_training(obs_pix_real)

            obs_pix_real_render = self.prepare_for_rendering(obs_pix_real)
            self.real_states["image_frame"] = obs_pix_real_render

        for _ in range(self._k):
            if self.use_real_pix_obs:
                self._real_frames.append(obs_pix_real_train)
            else:
                self._real_frames.append(obs_pix_sim)

        self.real_states["stacked_img_frames"] = self._get_stack_obs()

        real_and_sim_states = {"real_robot_state": self.real_states,
                               "sim_robot_state": self.sim_states}

        return real_and_sim_states

    def step(self, input_action):
        """
        Extends vanilla step() function call to accommodate data collection

        Args:
            action (np.array): Action to take in environment

        Returns:
            4-tuple:

                - (OrderedDict) observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """

        # These need to be tuned for the policy to run smoothly on the
        # real arm. The controller on the real is not as smooth as the sim.
        self.input_max = self.sim_env.robots[0].controller.input_max[0]
        self.input_min = self.sim_env.robots[0].controller.input_min[0]
        self.output_max = self.sim_env.robots[0].controller.output_max[0]
        self.output_min = self.sim_env.robots[0].controller.output_min[0]

        # Sim controller is already doing action clipping so send in raw inputs.
        sim_all = super().step(input_action)

        # Apply similar action scaling as done inside scale_action() in base_controller.py
        action_scale = abs(self.output_max - self.output_min) / abs(self.input_max - self.input_min)
        action_output_transform = (self.output_max + self.output_min) / 2.0
        action_input_transform = (self.input_max + self.input_min) / 2.0
        action = np.clip(input_action, self.input_min, self.input_max)
        action = (action - action_input_transform) * action_scale + action_output_transform

        sim_ret = sim_all[0]
        reward = sim_all[1]
        done = sim_all[2]
        misc = sim_all[3]
        print('Policy Action ', input_action)
        print('Clipped Action ', action)

        if self.control_type == "ANGLE":
            converted_action = action
        elif self.control_type == "TOOL":
            if action.shape[0] == 4:
                # OSC Position, no rotations
                for i in range(3):
                    action = np.insert(action, i+3, 0, axis=0)
            converted_pose, converted_quat = self.get_real2sim_posquat(action[:3], action[3:6], angle=-90)
            converted_action = np.concatenate((converted_pose, converted_quat), axis=0)
            # Add finger command
            converted_action = np.append(converted_action, action[-1])
            # print('Sim2Real Convereted action ', converted_action)
        else:
            raise ValueError("Unsupported controller!")

        self.robot_ret = self.handle_state(
                          self.robot_client.step(command_type=self.control_type,
                                   relative=self.control_relative,
                                   unit=self.control_unit,
                                   data=converted_action))
        # self.robot_ret = None

        self.sim_states["sim_ret"] = sim_ret
        # Extra sim observations
        self.sim_states["joint_pose"] = self.sim_env.sim.data.qpos[self.sim_env.robots[0]._ref_joint_pos_indexes]
        self.sim_states["finger_pose"] = self.sim_env.sim.data.qpos[self.sim_env.robots[0]._ref_joint_gripper_actuator_indexes]

        obs_pix_sim, obs_pix_raw = self._get_sim_image_obs(sim_ret)
        self.sim_states["image_frame"] = obs_pix_raw

        if self.use_real_pix_obs:
            # Real image frames
            obs_pix_real = self.robot_client.render()
            obs_pix_real_train = self.prepare_for_training(obs_pix_real)

            obs_pix_real_render = self.prepare_for_rendering(obs_pix_real)
            self.real_states["image_frame"] = obs_pix_real_render

        if self.use_real_pix_obs:
            self._real_frames.append(obs_pix_real_train)
        else:
            self._real_frames.append(obs_pix_sim)

        self.real_states["stacked_img_frames"] = self._get_stack_obs()

        robot_states = {"real_robot_state": self.real_states,
                        "sim_robot_state": self.sim_states}

        ret = (robot_states, reward, done, misc)
        return ret

    def render(self):
        """
         Args:
           height: Viewport height (number of pixels). Optional, defaults to 240.
           width: Viewport width (number of pixels). Optional, defaults to 320.
         Returns:
           The rendered RGB, depth or segmentation image.
        """
        return self.robot_client.render()

    def prepare_for_rendering(self, img, height=640, width=480):
        img = resize(img, (width, height))
        # skcit op changes the type, revert it back
        img = (img * 255).astype(np.uint8)
        return img

    def prepare_for_training(self, img):
        img = resize(img, (self.frame_width, self.frame_height))
        img = (img * 255).astype(np.uint8)
        img = img.reshape(3, img.shape[0],
                          img.shape[1])
        return img

    def _get_stack_obs(self):
        assert len(self._real_frames) == self._k
        concat_frames = np.concatenate(list(self._real_frames), axis=0)
        # For debugging purposes to make sure frames are aligned correctly
        # Grab the first frame
        # obs_pix = concat_frames[0:3, :, :]
        # obs_pix = obs_pix.reshape(obs_pix.shape[1], obs_pix.shape[2], 3)
        # im = Image.fromarray(obs_pix)
        # im.save('Sim2RealWrapper_real_robot_image_frame.png')
        return concat_frames


    def _get_sim_image_obs(self, obs_dict, verbose=False):
        """
        Filters keys of interest out and concatenate the information.
        Args:
            obs_dict (OrderedDict): ordered dictionary of observations
            verbose (bool): Whether to print out to console as observation keys are processed
        Returns:
            np.array: image observations into an array combined across channels
        """
        ob_lst = []
        for key in self.keys:
            if key in obs_dict:
                if verbose:
                    print("adding key: {}".format(key))
                # Sim images are upside-down, flip along z
                obs_pix_raw = obs_dict[key][::-1]
                # Reshaped for training (channel, width, height)
                obs_pix = obs_pix_raw.reshape(3, obs_pix_raw.shape[0],
                                          obs_pix_raw.shape[1])
                ob_lst.append(np.array(obs_pix))
        # concatenate over channels
        # the case for concatenating rgb, depth etc
        return np.concatenate(ob_lst, 2), obs_pix_raw


    def send_robot_to_sim_home(self):
        DOWN_INIT_QPOS = np.array([4.992, 3.680, -0.000, 1.170, 0.050, 3.760, 3.142])

        self.robot_client.step(command_type='ANGLE',
                                relative=False,
                                unit=self.control_unit,
                                data=DOWN_INIT_QPOS)

    def close(self):
        self.robot_client.end()
