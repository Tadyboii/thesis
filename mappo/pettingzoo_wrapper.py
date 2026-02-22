from simple_continuous_marl import raw_env

class PettingZooWrapper:
    def __init__(self, env=None, family=None, env_name=None, agent_ids=True, **kwargs):
        if env is not None:
            self.env = env
        else:
            self.env = raw_env(**kwargs)
        self.n_agents = self.env.n_agents

    def reset(self, seed=None):
        obs, infos = self.env.reset(seed=seed)
        state = self.env.get_state()
        return obs, state

    def step(self, actions):
        obs, rewards, dones, truncs, infos = self.env.step(actions)
        state = self.env.get_state()
        # For trainer compatibility
        done = any(dones.values())
        truncated = any(truncs.values())
        return obs, rewards, done, truncated, infos

    def get_obs_size(self):
        return self.env.get_obs_size()

    def get_action_size(self):
        return self.env.get_action_size()

    def get_state_size(self):
        return self.env.get_state_size()

    def sample(self):
        return {agent: self.env.action_space(agent).sample() for agent in self.env.agents}

    def render(self):
        self.env.render()
