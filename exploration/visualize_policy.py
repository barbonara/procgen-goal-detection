#%%
# Imports
%reload_ext autoreload
%autoreload 2

import os
import pickle

import numpy as np
import pandas as pd
import scipy as sp
from sklearn.feature_selection import f_classif
import torch as t
import torch.nn.functional as f
import xarray as xr
import plotly.express as px
import plotly as py
import plotly.subplots
import plotly.graph_objects as go
from einops import rearrange, repeat
from IPython.display import Video, display
from tqdm.auto import tqdm
import warnings

# NOTE: this is Monte's RL hooking code (and other stuff will be added in the future)
# Install normally with: pip install circrl
import circrl.module_hook as cmh
import circrl.rollouts as cro
import circrl.probing as cpr

import procgen_tools.models as models
import procgen_tools.maze as maze

warnings.filterwarnings("ignore", message=r'.*labels with no predicted samples.*')

# Hack to make sure cwd is the script folder
os.chdir(globals()['_dh'][0])

import procgen_tools
import matplotlib.pyplot as plt
from procgen_tools.imports import *
from procgen_tools import visualization, maze, vfield, patch_utils

#%%
# Load models:
policy1, hook1 = load_model('15', 15)
policy2, hook2 = load_model('1', 15)

#%%
venv = maze.create_venv(1, 0, 0, env_name='maze_yellowstar_redgem')
#venv = maze.create_venv(1, 0, 0, env_name='maze')

#x = visualization.visualize_venv(venv, 0, ax_size=5)

#%%

seed = 0
AX_SIZE = 3

vf = visualization.vector_field(venv, policy1)
visualization.plot_vf(vf)

# x = visualization.plot_vector_field(
#     venv,
#     policy1,
# )



# %%
