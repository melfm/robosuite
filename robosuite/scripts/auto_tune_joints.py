from multiprocessing.sharedctypes import Value
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import robosuite
import numpy as np
from imageio import mimwrite
from copy import deepcopy
import os

from matplotlib import cm

c = plt.cm.jet(np.linspace(0, 1, 8))
"""
Max linear velocity of the real jaco is 20cm/s
=======
# JRH summary of Ziegler-Nichols method
1) set damping_ratio and kp to zero
2) increase kp slowly until you get "overshoot", (positive ERR on first half of "diffs", negative ERR on second half of "diffs"
3) increase damping ratio to squash the overshoot
4) watch and make sure you aren't "railing" the torque values on the big diffs (30.5 for the first 4 joints on Jaco). 
If this is happening, you may need to decrease the step size (min_max)
I want it to slightly undershoot the biggest joint_diff in 1 step in a tuned controller
All others should have ~.000x error
Note: Make sure input_min/max and output_min/max match for tuning the configs since no rescaling
is happening here.
"""


def run_test():
    horizon = 1000

    if args.controller_name == 'OSC_POSITION':
        controller_config = robosuite.load_controller_config(
            custom_fpath=os.path.join(
                os.path.dirname(__file__), '..',
                'controllers/config/jaco_osc_position_%shz.json' % args.control_freq))
    elif args.controller_name == 'OSC_POSE':
        controller_config = robosuite.load_controller_config(
            custom_fpath=os.path.join(
                os.path.dirname(__file__), '..',
                'controllers/config/jaco_osc_pose_%shz.json' % args.control_freq))
    else:
        raise ValueError('Controller not supported.')


    print('Tuning :', controller_config)
    result_dir = 'controller_tuning_res'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    robot_name = args.robot_name
    env = robosuite.make("Lift",
                         robots=robot_name,
                         has_renderer=False,
                         has_offscreen_renderer=True,
                         ignore_done=True,
                         use_camera_obs=True,
                         use_object_obs=False,
                         camera_names='frontview',
                         controller_configs=controller_config,
                         control_freq=args.control_freq,
                         horizon=horizon)
    active_robot = env.robots[0]
    init_qpos = deepcopy(active_robot.init_qpos)
    print("before, initial", active_robot.controller.initial_joint)
    env.robots[0].controller.update_initial_joints(init_qpos)
    print("after, initial", active_robot.controller.initial_joint)
    o = env.reset()
    print("after, initial", active_robot.controller.initial_joint)
    positions = []
    orientations = []
    target_positions = []
    target_orientations = []
    joint_torques = []
    frames = []
    action_size = active_robot.controller.control_dim + 1
    action_array = np.zeros((horizon, action_size))
    null_action = np.zeros(action_size)
    step_distance = .2
    max_step_size = np.abs(env.action_spec[0][0])

    eef_pos = o['robot0_eef_pos']
    eef_quat = o['robot0_eef_quat']
    target_position = deepcopy(eef_pos)
    targets = []
    new_target = deepcopy(target_position)
    step_cnt = 0
    for x in np.arange(0, step_distance, max_step_size):
        action_array[step_cnt, 1] = max_step_size
        new_target += action_array[step_cnt, :3]
        targets.append(deepcopy(new_target))
        step_cnt += 1
    for x in range(10):
        targets.append(deepcopy(new_target))
        step_cnt += 1
    for x in np.arange(0, step_distance, max_step_size):
        action_array[step_cnt, 1] = -max_step_size
        new_target += action_array[step_cnt, :3]
        targets.append(deepcopy(new_target))
        step_cnt += 1
    for x in range(10):
        targets.append(deepcopy(new_target))
        step_cnt += 1

    new_target = deepcopy(target_position)
    for x in np.arange(0, step_distance, max_step_size):
        action_array[step_cnt, 0] = max_step_size
        new_target += action_array[step_cnt, :3]
        targets.append(deepcopy(new_target))
        step_cnt += 1
    for x in range(10):
        step_cnt += 1
        targets.append(deepcopy(new_target))
    for x in np.arange(0, step_distance, max_step_size):
        action_array[step_cnt, 0] = -max_step_size
        new_target += action_array[step_cnt, :3]
        targets.append(deepcopy(new_target))
        step_cnt += 1
    for x in range(10):
        targets.append(deepcopy(new_target))
        step_cnt += 1

    new_target = deepcopy(target_position)
    for x in np.arange(0, step_distance, max_step_size):
        action_array[step_cnt, 2] = max_step_size
        new_target += action_array[step_cnt, :3]
        targets.append(deepcopy(new_target))
        step_cnt += 1
    for x in range(10):
        targets.append(deepcopy(new_target))
        step_cnt += 1
    for x in np.arange(0, step_distance, max_step_size):
        action_array[step_cnt, 2] = -max_step_size
        new_target += action_array[step_cnt, :3]
        targets.append(deepcopy(new_target))
        step_cnt += 1
    for x in range(10):
        targets.append(deepcopy(new_target))
        step_cnt += 1

    action_array = action_array[:step_cnt]

    prev_eef = deepcopy(eef_pos)
    plt.figure()
    plt.plot(action_array)
    plt.savefig(os.path.join(result_dir, 'action.png'))
    plt.close()
    print('init pos', init_qpos)
    active_robot = env.robots[0]
    for i in range(action_array.shape[0]):
        if active_robot.controller_config['type'] == 'OSC_POSE':
            action = list(targets[i] - prev_eef) + [0, 0, 0, 0]
        else:
            # OSC_POSITION - add extra dim for finger
            action = list(targets[i] - prev_eef) + [0]

        target_position = prev_eef + action[:3]
        target_positions.append(target_position)
        o, _, _, _ = env.step(action)
        joint_torques.append(active_robot.torques)
        eef_pos = o['robot0_eef_pos']
        positions.append(eef_pos)
        eef_quat = o['robot0_eef_quat']
        orientations.append(eef_quat)
        frames.append(o['frontview_image'][::-1])
        prev_eef = deepcopy(eef_pos)
    video_dir = os.path.join(result_dir, args.movie_file)
    mimwrite(video_dir, frames)

    plt.figure()
    ax = plt.subplot(111)
    target_positions = np.array(target_positions)
    for d in range(target_positions.shape[1]):
        ax.plot(target_positions[:, d],
                 linestyle='--',
                 c=c[d],
                 label=str(d) + ' step target')
        ax.plot(np.array(positions)[:, d], c=c[d],
                label=str(d) + ' actual pose')
        # ax.plot(np.array(targets)[:, d],
        #          linestyle=':',
        #          c=c[d],
        #          label=str(d) + ' ideal target')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(os.path.join(result_dir, 'pos.png'))
    plt.close()

    plt.figure()
    for d in range(target_positions.shape[1]):
        err = target_positions[:, d] - np.array(positions)[:, d]
        plt.plot(err, c=c[d], label=d)
    plt.legend()
    plt.savefig(os.path.join(result_dir, 'error.png'))
    plt.close()

    plt.figure()
    joint_torques = np.array(joint_torques)
    for d in range(joint_torques.shape[1]):
        plt.plot(joint_torques[:, d], label=d, c=c[d])
    plt.legend()
    plt.savefig(os.path.join(result_dir, 'torques.png'))
    plt.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot_name', default='Jaco')
    parser.add_argument('--control_freq', default=5, type=int)
    parser.add_argument('--controller_name',
                        default='OSC_POSITION',
                        type=str,
                        help='controller name')
    parser.add_argument('--movie_file', default='tune.mp4')
    args = parser.parse_args()
    run_test()