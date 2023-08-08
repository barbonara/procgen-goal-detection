#%%
import gym
from procgen import ProcgenGym3Env
from procgen_tools import visualization, maze
import matplotlib.pyplot as plt

#%%


#venv = ProcgenGym3Env(num=1, env_name='maze')
#env = gym.make('procgen:procgen-maze_yellowstar_redgem-v0')
env = gym.make('procgen:procgen-maze_redgem_yellowstar-v0')


obs = env.reset()
plt.imshow(obs)

# img = venv.env.get_info()[0]["rgb"]
# plt.show(img)

venv = maze.create_venv(1, 0, 0, env_name='maze_redgem_yellowstar')
visualization.visualize_venv(venv, 0)


# %%
