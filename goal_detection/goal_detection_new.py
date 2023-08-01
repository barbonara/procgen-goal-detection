#%%
### Imports

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
import circrl.module_hook as cmh
import circrl.rollouts as cro
import circrl.probing as cpr

warnings.filterwarnings("ignore", message=r'.*labels with no predicted samples.*')

import procgen_tools
import matplotlib.pyplot as plt
from procgen_tools.imports import *
from procgen_tools import visualization, maze, models, vfield, patch_utils

import goal_detection_utils as gdu

#%%

### Inputs
# Model(s) to analyze
policy, hook = gdu.load_policy_and_hook('procgen-tools/trained_models/model_rand_region_15.pth')
# Their architecture (layers)
# Procgen environment to run models in 
env_name = 'maze'
num_envs = 200
venv = maze.create_venv(num=num_envs, start_level=0, num_levels=0, env_name=env_name)
# Number of steps to run each model in (observations for probe training)
# Environment objects/abstractions to probe for (their positions)



### Outputs
# prob over objects

#%%