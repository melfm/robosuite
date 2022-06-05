"""
This file implements a wrapper for facilitating compatibility with OpenAI gym.
This is useful when using these environments with code that assumes a gym-like
interface.
"""

from typing import Any, NamedTuple
import dm_env
import numpy as np
from gym import spaces
from gym.core import Env
from robosuite.wrappers import Wrapper
from collections import deque
import numpy as np
from dm_env import StepType
from robosuite.wrappers import Wrapper
from robosuite.utils.mjmod import TextureModder, LightingModder, CameraModder, DynamicsModder

DEFAULT_COLOR_ARGS = {
    'geom_names': None,  # all geoms are randomized
    'randomize_local': True,  # sample nearby colors
    'randomize_material':
    True,  # randomize material reflectance / shininess / specular
    'local_rgb_interpolation': 0.3,
    'local_material_interpolation': 0.3,
    'texture_variations': ['rgb', 'checker', 'noise',
                           'gradient'],  # all texture variation types
    'randomize_skybox': True,  # by default, randomize skybox too
}

DEFAULT_CAMERA_ARGS = {
    'camera_names': None,  # all cameras are randomized
    'randomize_position': True,
    'randomize_rotation': True,
    'randomize_fovy': True,
    'position_perturbation_size': 0.01,
    'rotation_perturbation_size': 0.087,
    'fovy_perturbation_size': 5.,
}

DEFAULT_LIGHTING_ARGS = {
    'light_names': None,  # all lights are randomized
    'randomize_position': True,
    'randomize_direction': True,
    'randomize_specular': True,
    'randomize_ambient': True,
    'randomize_diffuse': True,
    'randomize_active': True,
    'position_perturbation_size': 0.1,
    'direction_perturbation_size': 0.35,
    'specular_perturbation_size': 0.5,
    'ambient_perturbation_size': 0.2,
    'diffuse_perturbation_size': 2.0,
}

DEFAULT_DYNAMICS_ARGS = {
    # Opt parameters
    'randomize_density': True,
    'randomize_viscosity': True,
    'density_perturbation_ratio': 0.1,
    'viscosity_perturbation_ratio': 0.1,

    # Body parameters
    'body_names': None,  # all bodies randomized
    'randomize_position': True,
    'randomize_quaternion': True,
    'randomize_inertia': True,
    'randomize_mass': True,
    'position_perturbation_size': 0.0015,
    'quaternion_perturbation_size': 0.003,
    'inertia_perturbation_ratio': 0.02,
    'mass_perturbation_ratio': 0.02,

    # Geom parameters
    'geom_names': None,  # all geoms randomized
    'randomize_friction': True,
    'randomize_solref': True,
    'randomize_solimp': True,
    'friction_perturbation_ratio': 0.1,
    'solref_perturbation_ratio': 0.1,
    'solimp_perturbation_ratio': 0.1,

    # Joint parameters
    'joint_names': None,  # all joints randomized
    'randomize_stiffness': True,
    'randomize_frictionloss': True,
    'randomize_damping': True,
    'randomize_armature': True,
    'stiffness_perturbation_ratio': 0.1,
    'frictionloss_perturbation_size': 0.05,
    'damping_perturbation_size': 0.01,
    'armature_perturbation_size': 0.01,
}

class ExtendedTimeStep(NamedTuple):

    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    state_observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

class GymImageDomainRandomizationWrapper(Wrapper):
    """
    Wrapper that allows for domain randomization mid-simulation.
    Args:
        env (MujocoEnv): The environment to wrap.
        seed (int): Integer used to seed all randomizations from this wrapper. It is
            used to create a np.random.RandomState instance to make sure samples here
            are isolated from sampling occurring elsewhere in the code. If not provided,
            will default to using global random state.
        randomize_color (bool): if True, randomize geom colors and texture colors
        randomize_camera (bool): if True, randomize camera locations and parameters
        randomize_lighting (bool): if True, randomize light locations and properties
        randomize_dyanmics (bool): if True, randomize dynamics parameters
        color_randomization_args (dict): Color-specific randomization arguments
        camera_randomization_args (dict): Camera-specific randomization arguments
        lighting_randomization_args (dict): Lighting-specific randomization arguments
        dynamics_randomization_args (dict): Dyanmics-specific randomization arguments
        randomize_on_reset (bool): if True, randomize on every call to @reset. This, in
            conjunction with setting @randomize_every_n_steps to 0, is useful to
            generate a new domain per episode.
        randomize_every_n_steps (int): determines how often randomization should occur. Set
            to 0 if randomization should happen manually (by calling @randomize_domain)
        randomize_on_init (bool): if True: randomize on initialization and use those initial defaults for remaining randomization
    """
    def __init__(
        self,
        env,
        seed=112,
        randomize_color=False,
        randomize_camera=False,
        randomize_lighting=False,
        randomize_dynamics=False,
        use_proprio_obs=False,
        color_randomization_args=DEFAULT_COLOR_ARGS,
        camera_randomization_args=DEFAULT_CAMERA_ARGS,
        lighting_randomization_args=DEFAULT_LIGHTING_ARGS,
        dynamics_randomization_args=DEFAULT_DYNAMICS_ARGS,
        randomize_on_reset=False,
        randomize_on_init=False,
        randomize_every_n_steps=0,
        frame_stack=3,
        discount=.99,
    ):
        super().__init__(env)

        assert env.use_camera_obs == True
        self._k = frame_stack
        self._frames = deque([], maxlen=self._k)
        self.seed_value = seed
        self.discount = discount
        if seed is not None:
            self.random_state = np.random.RandomState(seed)
        else:
            self.random_state = None

        self.use_proprio_obs = use_proprio_obs

        # Don't change the color of objects of interest or of the robot
        # TODO this works with reach and lift and pickplace, will need to add objects in other envs
        leave_out_color_geoms = [
            'cube', 'sphere', 'gripper', 'robot', 'milk', 'bread', 'cereal',
            'can', 'handle', 'nut'
        ]
        use_color_geoms = []
        for g in env.sim.model.geom_names:
            include = True
            for lo in leave_out_color_geoms:
                if lo.lower() in g.lower():
                    include = False
            if include:
                use_color_geoms.append(g)

        if randomize_color:
            color_randomization_args['geom_names'] = use_color_geoms
            # randomize textures
            # color_randomization_args['texture_variations'] = ("rgb", "checker", "noise", "gradient")
            color_randomization_args['texture_variations'] = ("rgb")
        if randomize_camera:
            camera_randomization_args['camera_names'] = env.camera_names
        self.randomize_color = randomize_color
        self.randomize_camera = randomize_camera
        self.randomize_lighting = randomize_lighting
        self.randomize_dynamics = randomize_dynamics
        self.color_randomization_args = color_randomization_args
        self.camera_randomization_args = camera_randomization_args
        self.lighting_randomization_args = lighting_randomization_args
        self.dynamics_randomization_args = dynamics_randomization_args
        self.randomize_on_reset = randomize_on_reset
        self.randomize_on_init = randomize_on_init
        self.randomize_every_n_steps = randomize_every_n_steps

        self.step_counter = 0

        self.modders = []

        if self.randomize_color:
            self.tex_modder = TextureModder(sim=self.env.sim,
                                            random_state=self.random_state,
                                            **self.color_randomization_args)
            self.modders.append(self.tex_modder)

        if self.randomize_camera:
            self.camera_modder = CameraModder(
                sim=self.env.sim,
                random_state=self.random_state,
                **self.camera_randomization_args,
            )
            self.modders.append(self.camera_modder)

        if self.randomize_lighting:
            self.light_modder = LightingModder(
                sim=self.env.sim,
                random_state=self.random_state,
                **self.lighting_randomization_args,
            )
            self.modders.append(self.light_modder)

        if self.randomize_dynamics:
            self.dynamics_modder = DynamicsModder(
                sim=self.env.sim,
                random_state=self.random_state,
                **self.dynamics_randomization_args,
            )
            self.modders.append(self.dynamics_modder)

        self.save_default_domain()
        self.keys = [f"{cam_name}_image" for cam_name in self.env.camera_names]

        # Gym specific attributes
        self.env.spec = None
        self.metadata = None

        # set up observation and action spaces
        self.obs_dim = self.obs_shape = (3*self._k, self.env.camera_heights[0], self.env.camera_widths[0])
        high = 255 * np.ones(self.obs_dim, dtype=np.uint8)
        low = np.zeros(self.obs_dim, dtype=np.uint8)
        self.observation_space = spaces.Box(low=low, high=high)
        low, high = self.env.action_spec
        self.action_space = spaces.Box(low=low, high=high)
        self.action_shape = low.shape

        if self.use_proprio_obs:
            proprio_dim = self.env.observation_spec()['robot0_proprio-state'].shape[0]
            object_dim = self.env.observation_spec()['object-state'].shape[0]
            self.state_obs_shape = (proprio_dim + object_dim,)

        self._max_episode_steps = self.env.horizon
        if self.randomize_on_init:
            print('setting initial randomization')
            self.randomize_domain()
            # save new default
            self.save_default_domain()

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Dummy function to be compatible with gym interface that simply returns environment reward
        Args:
            achieved_goal: [NOT USED]
            desired_goal: [NOT USED]
            info: [NOT USED]
        Returns:
            float: environment reward
        """
        # Dummy args used to mimic Wrapper interface
        return self.env.reward()

    def seed(self, seed=None):
        """
        Utility function to set numpy seed
        Args:
            seed (None or int): If specified, numpy seed to set
        Raises:
            TypeError: [Seed must be integer]
        """
        # Seed the generator
        if seed is not None:
            try:
                np.random.seed(seed)
            except:
                TypeError("Seed must be an integer type!")

    def _reshape_store_frame(self, pixel_frame, output_file='test_frame'):
        """ For testing the rendered pixels.
            pixel_frame : (3, n, n)
        """
        from PIL import Image

        pixel_frame = pixel_frame.reshape(pixel_frame.shape[1],
                                          pixel_frame.shape[2], 3)
        output_name = output_file + '.png'
        im = Image.fromarray(pixel_frame)
        im.save(output_name)

    def _get_image_obs(self, obs_dict, verbose=False):
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
                obs_pix = obs_dict[key][::-1]
                obs_pix = obs_pix.reshape(3, obs_pix.shape[0],
                                          obs_pix.shape[1])
                # For debugging
                # self._reshape_store_frame(obs_pix,
                #                           'processed_from_get_image_obs.png')
                ob_lst.append(np.array(obs_pix))
        # concatenate over channels
        return np.concatenate(ob_lst, 2)

    def reset(self):
        """
        Extends superclass method to reset the domain randomizer.
        Returns:
            OrderedDict: Environment observation space after reset occurs
        """
        # undo all randomizations
        self.restore_default_domain()

        # normal env reset
        ret = super().reset()

        # save the original env parameters
        self.save_default_domain()

        # reset counter for doing domain randomization at a particular frequency
        self.step_counter = 0

        # update sims
        for modder in self.modders:
            modder.update_sim(self.env.sim)

        if self.randomize_on_reset:
            # domain randomize + regenerate observation
            self.randomize_domain()
            ret = self.env._get_observations()

        img = self._get_image_obs(ret)
        for _ in range(self._k):
            self._frames.append(img)
        obs = self._get_stack_obs()
        action = np.zeros_like(self.action_spec[0]).astype(np.float32)

        # Proprioceptive obs
        state_observation = None
        if self.use_proprio_obs:
            proprio = ret['robot0_proprio-state']
            object = ret['object-state']
            state_observation = np.append(object, proprio)

        return ExtendedTimeStep(step_type=dm_env.StepType.FIRST,
                                discount=self.discount,
                                reward=0.0,
                                observation=obs,
                                state_observation=state_observation,
                                action=action)

    def _get_stack_obs(self):
        assert len(self._frames) == self._k
        concat_frames = np.concatenate(list(self._frames), axis=0)
        # For debugging purposes to make sure frames are aligned correctly
        # Grab the first frame
        # obs_pix = concat_frames[0:3, :, :]
        # self._reshape_store_frame(obs_pix, 'processed_from_get_stack_obs.png')
        return concat_frames

    def step(self, action):
        """
        Extends vanilla step() function call to accommodate domain randomization
        Returns:
            4-tuple:
                - (OrderedDict) observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """

        #assert np.abs(action).max() <= 0.0150
        # functionality for randomizing at a particular frequency
        if self.randomize_every_n_steps > 0:
            if self.step_counter % self.randomize_every_n_steps == 0:
                self.randomize_domain()
        self.step_counter += 1

        ob_dict, reward, done, info = self.env.step(action)
        self._frames.append(self._get_image_obs(ob_dict))
        obs = self._get_stack_obs()

        # Proprioceptive obs
        state_observation = None
        if self.use_proprio_obs:
            proprio = ob_dict['robot0_proprio-state']
            object = ob_dict['object-state']
            state_observation = np.append(object, proprio)

        if not done:
            step_type = dm_env.StepType.MID
        else:
            step_type = dm_env.StepType.LAST
        return ExtendedTimeStep(step_type=step_type,
                                discount=self.discount,
                                reward=reward,
                                observation=obs,
                                state_observation=state_observation,
                                action=action)

    def randomize_domain(self):
        """
        Runs domain randomization over the environment.
        """
        for modder in self.modders:
            modder.randomize()

    def save_default_domain(self):
        """
        Saves the current simulation model parameters so
        that they can be restored later.
        """
        for modder in self.modders:
            modder.save_defaults()

    def restore_default_domain(self):
        """
        Restores the simulation model parameters saved
        in the last call to @save_default_domain.
        """
        for modder in self.modders:
            modder.restore_defaults()

    def render(self,
               width=256,
               height=256,
               depth=False):
        data = self.env.sim.render(camera_name=self.env.camera_names[0],
                                   width=width,
                                   height=height,
                                   depth=depth)
        # original image is upside-down and mirrored, so flip both axis
        if not depth:
            return data[::-1]
        else:
            # Untested
            return data[0][::-1, :], data[1][::-1]
