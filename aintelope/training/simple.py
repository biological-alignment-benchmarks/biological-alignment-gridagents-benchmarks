import random
from pprint import pprint
import numpy as np
from gym.spaces import Discrete
import gym
from aintelope.agents.q_agent import Agent as Q_agent
from aintelope.agents.memory import ReplayBuffer
from aintelope.aintelope.environments.savanna_zoo import env, move_agent, reward_agent
import os
from gemini import Scene, Entity, sleep




# pygame.display.list_modes()
os.environ["SDL_VIDEODRIVER"] = "dummy" #"x11" # "directfb"

# numerical constants
EPS = 0.0001
INF = 9999999999


class RandomWalkAgent:
    def __call__(self, action_space, observation, reward, info):
        return action_space.sample()


class OneStepPerfectPredictionAgent:
    def __call__(self, action_space, observation, reward, info):
        # FIXME: are you fucking kidding me?!
        agent_pos, grass = observation[:2], observation[2:].reshape(2, -1)
        bestreward = -INF
        ibestaction = 0
        for iaction in range(action_space.n):
            p = move_agent(agent_pos, iaction)
            reward = reward_agent(p, grass)
            if reward > bestreward:
                bestreward = reward
                ibestaction = iaction
        # print(observation)
        # print(reward, iaction)
        return ibestaction


class IterativeWeightOptimizationAgent:
    def __init__(self):
        self.is_initialized = False

    def __call__(self, action_space, observation, reward, info):
        MIN_WEIGHT = 0.05
        learning_rate = 0.01
        learning_randomness = 0.00

        LAST_ACTION_KEY = "last_action"
        LAST_REWARD_KEY = "last_reward"
        ACTIONS_WEIGHTS = "actions_weights"

        if not self.is_initialized:
            info[ACTIONS_WEIGHTS] = np.repeat([1.0], action_space.n)
            self.is_initialized = True
        print('info', info)
        print("step:", reward, observation)
        last_action = info.get(LAST_ACTION_KEY)
        last_reward = info.get(LAST_REWARD_KEY, 0)
        action_weights = info[ACTIONS_WEIGHTS]
        # avoid big weight change on the first valid step
        if last_action is not None and last_reward > EPS:
            last_action_reward_delta = reward - last_reward
            last_action_weight = action_weights[last_action]
            print(
                "dreward",
                last_action_reward_delta,
                last_action,
            )
            last_action_weight += last_action_reward_delta * learning_rate
            last_action_weight = max(MIN_WEIGHT, last_action_weight)
            action_weights[last_action] = last_action_weight
            print("action_weights", action_weights)

            weight_sum = np.sum(action_weights)
            action_weights /= weight_sum

        def cdf(ds):
            res = {}
            x = 0
            for k, v in ds:
                x += v
                res[k] = x
            for k in res:
                res[k] /= x
            return res

        def choose(cdf):
            assert cdf
            x = random.uniform(0, 1 - EPS)
            k = None
            for k, v in cdf.items():
                if x >= v:
                    return k
            return k

        action_weights_cdf = cdf(enumerate(action_weights))
        print(
            "cdf",
            ", ".join(
                [
                    f"{iaction}: {w}"
                    for iaction, w in action_weights_cdf.items()
                ]
            ),
        )

        pprint(action_weights_cdf)
        action = choose(action_weights_cdf)
        if random.uniform(0, 1) < learning_randomness:
            action = action_space.sample()
        info[LAST_ACTION_KEY] = action
        info[LAST_REWARD_KEY] = reward
        print("chose action", action)
        return action


def main(env):
    mode = "ascii"
    # policy = IterativeWeightOptimizationAgent()
    # policy = OneStepPerfectPredictionAgent()
    # policy = RandomWalkAgent()
    
    replay_size = 1000
    env.reset()
    replay_buffer = ReplayBuffer(replay_size)
    policy = Q_agent(gym.make(env), replay_buffer)
    for agent in env.agent_iter():
        observation, reward, done, info = env.last()

        action_space: Discrete = env.action_space(agent)
        if not done:
            action = policy(action_space, observation, reward, info)
            assert action in action_space
        else:
            action = None
        env.step(action)
        env.render(mode)

    # wait = input("Close?")


if __name__ == "__main__":
    env_params = {
        'NUM_ITERS':40,  # duration of the game
        'MAP_MIN':0, 
        'MAP_MAX':20,
        'render_map_max':20,
        'AMOUNT_AGENTS':1,  # for now only one agent
        'AMOUNT_GRASS_PATCHES':2
                }
    e = env(env_params=env_params)
    main(e)

# Local Variables:
# compile-command: "poetry run python simple.py"
# End:
