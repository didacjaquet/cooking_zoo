from __future__ import annotations

import glob
import os
import time

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy, MlpPolicy

from cooking_zoo.environment.cooking_env import env
from pettingzoo.utils.conversions import aec_to_parallel

def train(env_fn, steps: int = 10_000, seed: int | None = 0):
    # Train a single model to play as each agent in an AEC environment
    env = aec_to_parallel(env_fn)

    # Add black death wrapper so the number of agents stays constant
    # MarkovVectorEnv does not support environments with varying numbers of active agents unless black_death is set to True
    env = ss.black_death_v3(env)

    # Pre-process using SuperSuit
    # visual_observation = not env.unwrapped.vector_state
    # if visual_observation:
        # If the observation space is visual, reduce the color channels, resize from 512px to 84px, and apply frame stacking
    #    env = ss.color_reduction_v0(env, mode="B")
    #     env = ss.resize_v1(env, x_size=84, y_size=84)
    #    env = ss.frame_stack_v1(env, 3)

    env.reset()

    print(f"Starting training on {str(env.metadata['name'])}.")

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=1, base_class="stable_baselines3")

    # Use a CNN policy if the observation space is visual
    model = PPO(
        MlpPolicy,
        env,
        verbose=3,
        batch_size=256,
    )

    model.learn(total_timesteps=steps)

    model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()


def eval(env_fn, num_games: int = 100, render_mode: str | None = None):
    # Evaluate a trained agent vs a random agent
    env = env_fn

    # Pre-process using SuperSuit
    # visual_observation = not env.unwrapped.vector_state
    # if visual_observation:
        # If the observation space is visual, reduce the color channels, resize from 512px to 84px, and apply frame stacking
    #   env = ss.color_reduction_v0(env, mode="B")
    #   env = ss.resize_v1(env, x_size=84, y_size=84)
    #   env = ss.frame_stack_v1(env, 3)

    print(
        f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})"
    )

    try:
        latest_policy = max(
            glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime
        )
    except ValueError:
        print("Policy not found.")
        exit(0)

    model = PPO.load(latest_policy)

    rewards = {agent: 0 for agent in env.possible_agents}

    # Note: we evaluate here using an AEC environments, to allow for easy A/B testing against random policies
    # For example, we can see here that using a random agent for archer_0 results in less points than the trained agent
    for i in range(num_games):
        env.reset()
        env.action_space(env.possible_agents[0]).seed(i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            for a in env.agents:
                rewards[a] += env.rewards[a]

            if termination or truncation:
                break
            else:
                if agent == env.possible_agents[0]:
                    act = env.action_space(agent).sample()
                else:
                    act = model.predict(obs, deterministic=True)[0]
            env.step(act)
    env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    avg_reward_per_agent = {
        agent: rewards[agent] / num_games for agent in env.possible_agents
    }
    print(f"Avg reward: {avg_reward}")
    print("Avg reward per agent, per game: ", avg_reward_per_agent)
    print("Full rewards: ", rewards)
    return avg_reward


if __name__ == "__main__":
    num_agents = 2
    max_steps = 400
    render = False
    obs_spaces = ["feature_vector", "feature_vector"]
    action_scheme = "scheme1"
    meta_file = "example"
    level = "coexistence_test"
    recipes = ["TomatoLettuceSalad", "CarrotBanana"]
    end_condition_all_dishes = True
    agent_visualization = ["robot", "human"]
    reward_scheme = {"recipe_reward": 20, "max_time_penalty": -5, "recipe_penalty": -40, "recipe_node_reward": 0}

    env = env(level=level, meta_file=meta_file, num_agents=num_agents,
                                max_steps=max_steps, recipes=recipes, agent_visualization=agent_visualization,
                                obs_spaces=obs_spaces, end_condition_all_dishes=end_condition_all_dishes,
                                action_scheme=action_scheme, render=render, reward_scheme=reward_scheme)

    # Set vector_state to false in order to use visual observations (significantly longer training time)
    #env = aec_to_parallel(env)
    # Train a model (takes ~5 minutes on a laptop CPU)
    train(env, steps=81_920, seed=0)

    # Evaluate 10 games (takes ~10 seconds on a laptop CPU)
    eval(env, num_games=10, render_mode=None)

    # Watch 2 games (takes ~10 seconds on a laptop CPU)
    eval(env, num_games=2, render_mode="human")