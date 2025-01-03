"""
This script demonstrates how to load and rollout a finetuned Octo model.
We use the Octo model finetuned on ALOHA sim data from the examples/02_finetune_new_observation_action.py script.

For installing the ALOHA sim environment, clone: https://github.com/tonyzhaozh/act
Then run:
pip3 install opencv-python modern_robotics pyrealsense2 h5py_cache pyquaternion pyyaml rospkg pexpect mujoco==2.3.3 dm_control==1.0.9 einops packaging h5py

Finally, modify the `sys.path.append` statement below to add the ACT repo to your path.
If you are running this on a head-less server, start a virtual display:
    Xvfb :1 -screen 0 1024x768x16 &
    export DISPLAY=:1

To run this script, run:
    cd examples
    python3 03_eval_finetuned.py --finetuned_path=<path_to_finetuned_aloha_checkpoint>
"""
from functools import partial
import sys

from absl import app, flags, logging
import gym
import jax
import flax
import numpy as np
import wandb

# sys.path.append("/path/to/act")

# keep this to register ALOHA sim env
from envs.metaworld_env import MetaworldEnv  # noqa
from envs.mw_tools import POLICIES

from octo.model.octo_model import OctoModel
from octo.utils.gym_wrappers import HistoryWrapper, NormalizeProprio, RHCWrapper
from octo.utils.train_callbacks import supply_rng

import metaworld.envs.mujoco.env_dict as _env_dict

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "finetuned_path", None, "Path to finetuned Octo checkpoint directory."
)


def main(_):    
    # setup wandb for logging [train | test]
    wandb.init(name="eval_metaworld_ml10_20e_train_5000", project="octo")

    # load finetuned model
    logging.info("Loading finetuned model...")
    model = OctoModel.load_pretrained(FLAGS.finetuned_path, step=4999)

    # make gym environment
    ##################################################################################################################
    # environment needs to implement standard gym interface + return observations of the following form:
    #   obs = {
    #     "image_primary": ...
    #   }
    # it should also implement an env.get_task() function that returns a task dict with goal and/or language instruct.
    #   task = {
    #     "language_instruction": "some string"
    #     "goal": {
    #       "image_primary": ...
    #     }
    #   }
    ##################################################################################################################
    
    benchmark = _env_dict.ML10_V2

    for name in benchmark['train'].keys():
        
        env = MetaworldEnv(name)
        expert = POLICIES[name]()

        # wrap env to normalize proprio
        env = NormalizeProprio(env, model.dataset_statistics)

        # add wrappers for history and "receding horizon control", i.e. action chunking
        env = HistoryWrapper(env, horizon=1)
        env = RHCWrapper(env, exec_horizon=50)

        # the supply_rng wrapper supplies a new random key to sample_actions every time it's called
        policy_fn = supply_rng(
            partial(
                model.sample_actions,
                unnormalization_statistics=model.dataset_statistics["action"],
            ),
        )

        # running rollouts
        total_return = 0
        total_accuracy = 0
        for i in range(20):
            obs, info = env.reset()

            # create task specification --> use model utility to create task dict with correct entries
            language_instruction = env.get_task()["language_instruction"]
            
            task = model.create_tasks(texts=language_instruction)

            # run rollout for 400 steps
            images = [obs["image_primary"][0]]
            episode_return = 0.0
            while len(images) < 400:
                # model returns actions of shape [batch, pred_horizon, action_dim] -- remove batch
                actions = policy_fn(jax.tree_map(lambda x: x[None], obs), task)
                actions = actions[0]

                # step env -- info contains full "chunk" of observations for logging
                # obs only contains observation for final step of chunk
                obs, reward, done, trunc, info = env.step(np.array(actions))
                
                images.append(obs["image_primary"][0])
                episode_return += reward

                # # TODO Simulate expert action
                # info_expert = info['state'][-1]
                # for _ in range(50):
                #     action = np.clip(expert.get_action(info_expert), -1, 1)
                #     bos_expert, reward_expert, done_expert, trunc_expert, info_expert = env.do_one_step(action)
                #     info_expert = info_expert['state']
                    
                #     if done_expert or trunc_expert:
                #         done = done_expert
                #         trunc = trunc_expert
                #         break
                    
                if done or trunc:
                    break
            # print(f"Episode return: {episode_return}, Done: {done}, Trunc: {trunc}")

            total_return += episode_return
            total_accuracy += int(done)
            
            # log rollout video to wandb -- subsample temporally 2x for faster logging
            if i % 5 == 0:
                wandb.log(
                    {"rollout_video": wandb.Video(np.array(images).transpose(0, 3, 1, 2)[::2])}
                )
        
        print(f"Environment: {name}, Average return: {total_return / 20}, Average accuracy: {total_accuracy / 20}")
        wandb.log({name: {"average_return": total_return / 20, "average_accuracy": total_accuracy / 20}})
    
if __name__ == "__main__":
    app.run(main)