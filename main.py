import sys

sys.path.insert(
    0, "projects/mod_GOAT"
)

import json
import os
from pprint import pprint

import numpy as np
from tqdm import tqdm

from config_utils import get_config
from habitat.core.env import Env

from goat_agent import GoatAgent
from objectnav_zoo.core.interfaces import DiscreteNavigationAction
from objectnav_zoo.env.habitat_goat_env.habitat_goat_env import HabitatGoatEnv

def episode_evaluation(env,agent,config, episode_name):
    env.reset()
    agent.reset()

    metrics = {}
    old_distance_to_goal = None
    ctr = 0

    t = 0

    scene_id = env.habitat_env.current_episode.scene_id.split("/")[-1].split(".")[0]
    episode = env.habitat_env.current_episode
    episode_id = episode.episode_id


    agent.planner.set_vis_dir(scene_id, f"{episode_id}_{env.habitat_env.task.current_task_idx}")
    agent.imagenav_visualizer.set_vis_dir(
        f"{scene_id}_{episode_id}_{env.habitat_env.task.current_task_idx}"
    )
    agent.matching.set_vis_dir(f"{scene_id}_{episode_id}_{env.habitat_env.task.current_task_idx}")
    env.visualizer.set_vis_dir(scene_id, f"{episode_id}_{env.habitat_env.task.current_task_idx}")

    all_subtask_metrics = []
    pbar = tqdm(total=config.AGENT.max_steps)
    
    while not env.episode_over:
        current_task_idx = env.habitat_env.task.current_task_idx
        t += 1
        obs = env.get_observation()

        if t == 1:
            obs_tasks = []
            for task in obs.task_observations["tasks"]:
                obs_task = {}
                for key, value in task.items():
                    if key == "image":
                        continue
                    obs_task[key] = value
                obs_tasks.append(obs_task)

            pprint(obs_tasks)

        action, info = agent.act(obs)

        env.apply_action(action, info=info)
        pbar.set_description(
            f"{scene_id}_{episode_id}_{current_task_idx}"
        )
        pbar.update(1)

        if env.get_episode_metrics()["goat_distance_to_sub-goal"] == old_distance_to_goal:
            ctr += 1

            if ctr > 20:
                print("Agent was stuck. Stopping episode.")
                action = DiscreteNavigationAction.STOP
                ctr = 0
        else:
            ctr = 0
        
        old_distance_to_goal = env.get_episode_metrics()["goat_distance_to_sub-goal"]

        if action == DiscreteNavigationAction.STOP:
            ep_metrics = env.get_episode_metrics()
            ep_metrics.pop("goat_top_down_map", None)
            print('-------------------------')
            print(f"{scene_id}_{episode_id}_{current_task_idx}", ep_metrics)
            print('-------------------------')

            all_subtask_metrics.append(ep_metrics)
            if not env.episode_over:
                agent.imagenav_visualizer.set_vis_dir(
                    f"{scene_id}_{episode_id}_{env.habitat_env.task.current_task_idx}"
                )
                agent.matching.set_vis_dir(
                    f"{scene_id}_{episode_id}_{env.habitat_env.task.current_task_idx}"
                )
                agent.planner.set_vis_dir(
                    scene_id, f"{episode_id}_{env.habitat_env.task.current_task_idx}"
                )
                env.visualizer.set_vis_dir(
                    scene_id, f"{episode_id}_{env.habitat_env.task.current_task_idx}"
                )
                pbar.reset()

    pbar.close()

    ep_metrics = env.get_episode_metrics()
    metrics = {"metrics": all_subtask_metrics}
    metrics["total_num_steps"] = t
    metrics["sub_task_timesteps"] = agent.sub_task_timesteps[0]
    metrics["tasks"] = obs_tasks


if __name__ == "__main__":
    config, config_str = get_config(
        "projects/mod_GOAT/configs/habitat_config.yaml",
        "projects/mod_GOAT/configs/modular_goat_hm3d_eval.yaml",
    )
    print("Config:\n", config_str, "\n", "-" * 100)

    all_scenes = os.listdir(os.path.dirname(config.habitat.dataset.data_path.format(split=config.habitat.dataset.split)) + "/content/")
    all_scenes = sorted([x.split('.')[0] for x in all_scenes])
    config.habitat.dataset.content_scenes = all_scenes[0:5]

    config.NUM_ENVIRONMENTS = 1
    config.PRINT_IMAGES = 1

    agent = GoatAgent(config=config)
    habitat_env = Env(config)
    env = HabitatGoatEnv(habitat_env, config=config)

    results_dir = os.path.join(config.DUMP_LOCATION, "results", config.EXP_NAME)
    os.makedirs(results_dir, exist_ok=True)

    for k in range(len(env.habitat_env.episodes)):
        episode_evaluation(env,agent,config, str(k))
        