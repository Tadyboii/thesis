# import numpy as np
# from particle_pymunk import raw_env

# env = raw_env(render_mode="human")

# while True:  # keep running forever
#     obs, infos = env.reset()
#     done = {agent: False for agent in env.agents}

#     while not all(done.values()):
#         actions = {agent: env.action_space(agent).sample() for agent in env.agents}
#         obs, rewards, terms, truncs, infos = env.step(actions)
#         done = {agent: terms[agent] or truncs[agent] for agent in env.agents}

# # env.close()  # unreachable unless you break manually

from particle_box2d import raw_env

env = raw_env(render_mode="human")

while True:
    obs, infos = env.reset()
    done = {agent: False for agent in env.agents}

    while not all(done.values()):
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        done = {a: terms[a] or truncs[a] for a in env.agents}

