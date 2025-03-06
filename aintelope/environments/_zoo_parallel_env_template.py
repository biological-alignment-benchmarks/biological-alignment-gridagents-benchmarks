# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository: https://github.com/aintelope/biological-compatibility-benchmarks

from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
from gymnasium.spaces import Box, Discrete
from pettingzoo import AECEnv, ParallelEnv


class ZooParallelEnvTemplate(ParallelEnv):
    def __init__(
        self, env_params: Optional[Dict] = None, ignore_num_iters=False, *args, **kwargs
    ):
        self.max_num_agents = env_params["amount_agents"]
        self.possible_agents = ["agent_" + str(i) for i in range(self.max_num_agents)]
        self.max_iterations = env_params["num_iters"]
        self.render_mode = None  # a required attribute

    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return Box(
            low=0,  # this is a boolean bitmap
            high=1,  # this is a boolean bitmap
            shape=(
                7,  # depth
                5,  # height
                5,  # width
            ),
        )

    def action_space(self, agent):
        return Discrete(5)

    def reset(self, seed=None, options=None, *args, **kwargs):
        self.agents = list(self.possible_agents)  # clone the list
        self.num_moves = 0
        observations = {agent: None for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        self.state = observations

        return observations, infos

    def step(self, actions: Dict[str, int]):
        if not actions:  # empty dict
            return {}, {}, {}, {}, {}

        for agent, action in actions.items():
            if (
                action is None
            ):  # Agent is dead. Dead agents receive one None action call because of Zoo standard.
                pass

            # do your action handling here
        # / for agent, action in actions.items():

        rewards = {}
        for agent in self.agents:
            rewards[agent] = 77

        terminations = {agent: False for agent in self.agents}

        self.num_moves += 1
        env_truncation = self.num_moves >= self.max_iterations
        truncations = {agent: env_truncation for agent in self.agents}

        observations = {agent: np.zeros([7, 5, 5]) for agent in self.agents}
        self.state = observations

        # typically there won't be any information in the infos, but there must
        # still be an entry for each agent
        infos = {agent: {} for agent in self.agents}

        if env_truncation:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(self):
        pass  # not implemented

    def close(self):
        pass
