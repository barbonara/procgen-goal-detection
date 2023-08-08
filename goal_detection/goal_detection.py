#%%
# Imports
# %reload_ext autoreload
# %autoreload 2

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

# # Hack to make sure cwd is the script folder
# os.chdir(globals()['_dh'][0])

import procgen_tools
import matplotlib.pyplot as plt
from procgen_tools.imports import *
from procgen_tools import visualization, maze, vfield, patch_utils




# Number of mazes to work with

print('imports done')

num_batch = 200

# Functions

# Takes in a maze object, position in maze, venv (collection of environments), and returns labels regarding whether the position contains object
def is_obj_in_pos(obj, pos, venv):
    square_is_obj = []

    for env_idx in range(venv.num_envs):
        #maze.EnvState(dd[state_bytes_key])

        square_is_obj.append(maze.state_from_venv(venv, env_idx).full_grid()[pos] == obj)

    square_is_obj = np.array(square_is_obj, dtype=bool)
    return square_is_obj

# Train sparse linear probes on observation of model as a baseline

def test_probes_on_obs( obj_is_in_pos_array, hook_to_use, index_nums = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])):
    results, _ = cpr.sparse_linear_probe(hook_to_use, ['embedder.block1.conv_in0'], obj_is_in_pos_array, 
        index_nums=index_nums, random_state=42, class_weight='balanced', max_iter=1000, C=10.)
    #px.line(x=index_nums, y=results.score.isel(value_label=0)).show()
    x = pd.DataFrame({'num_activations': index_nums, 'test_score': results.score.isel(value_label=0).values}).set_index('num_activations')
    print(x)

# Plot the sparse probe scores
def plot_sparse_probe_scores(results, y, index_nums, title, include_limits=True):
    scores_df = results.score.to_dataframe().reset_index()
    scores_df['K'] = index_nums[scores_df['index_num_step']]
    fig = px.line(scores_df, x='value_label', y='score', color='K', title=title)
    if include_limits:
        fig.add_hline(y=1., line_dash="dot", annotation_text="perfect", annotation_position="bottom right")
        baseline_score = abs(y.mean()-0.5) + 0.5
        fig.add_hline(y=baseline_score, line_dash="dot", annotation_text="baseline", 
                annotation_position="bottom right")
    fig.show()

# Apply probes to various layers in the network
def test_probes_layers(obj_is_in_pos_array, hook_to_use, pos, index_nums = np.array([1, 2, 10, 50, 100]),
    value_labels_to_plot = [
        'embedder.block1.conv_in0',
        'embedder.block1.res1.resadd_out',
        'embedder.block1.res2.resadd_out',
        'embedder.block2.res1.resadd_out',
        'embedder.block2.res2.resadd_out',
        'embedder.block3.res1.resadd_out',
        'embedder.block3.res2.resadd_out']
    ):


    results, _ = cpr.sparse_linear_probe(hook_to_use, value_labels_to_plot, obj_is_in_pos_array,
        index_nums = index_nums, random_state=42, class_weight='balanced', max_iter=10000, C=10.)


    plot_sparse_probe_scores(results, obj_is_in_pos_array, index_nums, 
        f'Probe score over layers and K-values for {pos} "is open"')
    

# Get the probe target for object location
def get_obj_loc_targets(venv, obj_value):
    '''Get potential probe targets for y,x (row, col) location of an
    object, where the object is specified by obj_value as the value
    to match on in the maze grid array.'''
    pos_arr = maze.get_object_pos_from_seq_of_states(
        [maze.state_from_venv(venv, env_idx).state_bytes for env_idx in range(num_batch)], obj_value)
    pos = xr.Dataset({
        'y': xr.DataArray(pos_arr[:,0], dims=['batch']),
        'x': xr.DataArray(pos_arr[:,1], dims=['batch'])}).assign_coords(
            {'batch': np.arange(num_batch)})
    return pos

# Get the probe target for top right corner
def get_top_right_loc_targets(venv):
    '''Get potential probe targets for y,x (row, col) location of an
    object, where the object is specified by obj_value as the value
    to match on in the maze grid array.'''
    # pos_arr is array of length len(state_bytes_seq) with (y,x) locations of object
    # pos_arr = maze.get_object_pos_from_seq_of_states(
    #     [maze.state_from_venv(venv, env_idx).state_bytes for env_idx in range(num_batch)], obj_value)
    # Replace with top right corner coords
    pos_arr = np.array([[maze.state_from_venv(venv,env_idx).inner_grid().shape[0]-1, maze.state_from_venv(venv,env_idx).inner_grid().shape[0]-1] for env_idx in range(num_batch)])
    pos = xr.Dataset({
        'y': xr.DataArray(pos_arr[:,0], dims=['batch']),
        'x': xr.DataArray(pos_arr[:,1], dims=['batch'])}).assign_coords(
            {'batch': np.arange(num_batch)})
    return pos


def euclidian_mouse_dist_to_top_right(grid: np.ndarray) -> float:
    """
    Euclidian distance from (x,y) to the cheese. default heuristic for A*
    """
    #print(grid)
    grid_size = grid.shape[0]
    

    # venv = maze.venv_from_grid(grid)
    # visualization.visualize_venv(
    #     venv
    # )
    cx, cy = maze.get_mouse_pos(grid)
    # print(grid_size-1, grid_size-1)
    # print(cx,cy)
    return np.sqrt((grid_size-1 - cx) ** 2 + (grid_size-1 - cy) ** 2)

def get_mouse_dist_to_top_right(venv):
    '''Get potential probe targets for y,x (row, col) location of an
    object, where the object is specified by obj_value as the value
    to match on in the maze grid array.'''
    #print(euclidian_mouse_dist_to_top_right(maze.state_from_venv(venv, 0).inner_grid()))
    dist = [euclidian_mouse_dist_to_top_right(maze.state_from_venv(venv, env_idx).inner_grid()) for env_idx in range(num_batch)]
    dist_arr = np.array(dist)
    return dist_arr

def visualize_maze(state: maze.EnvState) -> None:
    venv = maze.venv_from_grid(state.inner_grid())
    visualization.visualize_venv(venv, render_padding=False)

# Helper functions for conv probes
# 
# Some helper functions
def grid_coord_to_value_ind(full_grid_coord, value_size):
    '''Pick the value index that covers the majority of the grid coord pixel'''
    # TODO: I'm pretty sure this is the best groudned approach, but ch 55 responds better to the approach in probing_main.py
    return np.floor((full_grid_coord+0.5) * value_size/maze.WORLD_DIM).astype(int)

def value_ind_to_grid_coord(value_ind, value_size):
    '''Pick the grid coordinate index whose center is closest to the center of the value pixel'''
    return np.floor((value_ind+0.5) * maze.WORLD_DIM/value_size).astype(int)

def get_obj_pos_data(value_label, object_pos, hook_to_use):
    '''Pick the object location and a random other location without the object so we have a
    balanced dataset of pixels 2x the original size.'''
    rng = np.random.default_rng(15)
    # TODO: vectorize this!    
    value = hook_to_use.get_value_by_label(value_label)
    value_size = value.shape[-1]
    num_pixels = num_batch * 2
    pixels = np.zeros((num_pixels, value.shape[1]))
    is_obj = np.zeros(num_pixels, dtype=bool)
    rows_in_value = np.zeros(num_pixels, dtype=int)
    cols_in_value = np.zeros(num_pixels, dtype=int)
    for bb in tqdm(range(obs_all.shape[0])):
        # Cheese location (transform from full grid row/col to row/col in this value)
        obj_pos_value = (grid_coord_to_value_ind(
                maze.WORLD_DIM-1 - object_pos.y[bb].item(), value_size),
            grid_coord_to_value_ind(object_pos.x[bb].item(), value_size))
        pixels[bb,:] = value[bb,:,obj_pos_value[0],obj_pos_value[1]]
        is_obj[bb] = True
        rows_in_value[bb] = obj_pos_value[0]
        cols_in_value[bb] = obj_pos_value[1]
        # Random pixel that isn't the object location
        bb_rand = bb + num_batch
        random_pos = obj_pos_value
        while random_pos == obj_pos_value:
            random_pos = (rng.integers(value_size), rng.integers(value_size))
        pixels[bb_rand,:] = value[bb,:,random_pos[0],random_pos[1]]
        is_obj[bb_rand] = False
        rows_in_value[bb_rand] = random_pos[0]
        cols_in_value[bb_rand] = random_pos[1]
    return pixels, is_obj, rows_in_value, cols_in_value

# def show_f_test_results(pixels, target, target_name, rows_in_value, cols_in_value):
#     f_test, _ = cpr.f_classif_fixed(pixels, target)
#     f_test_df = pd.Series(f_test).sort_values(ascending=False)

#     fig = px.line(y=f_test_df, title=f'Sorted {target_name} f-test scores for channels of<br>{value_label}',
#         hover_data={'channel': f_test_df.index})
#     fig.update_layout(
#         xaxis_title="channel rank",
#         yaxis_title="f-test score",)
#     fig.show()

#     print(list(f_test_df.index[:20]))

#     for ch_ind in f_test_df.index[:2]:
#         show_pixel_histogram(pixels, target, target_name, ch_ind)

def make_pixel_data(value_labels, obj_pos, hook_to_use):
    f_test_list = []
    pixel_data = {}
    for value_label in value_labels:
        pixels, is_obj, rows_in_value, cols_in_value = get_obj_pos_data(value_label, obj_pos, hook_to_use)
        f_test, _ = cpr.f_classif_fixed(pixels, is_obj)
        sort_inds = np.argsort(f_test)[::-1]
        pixel_data[value_label] = (pixels, is_obj, rows_in_value, cols_in_value, f_test, sort_inds)
        f_test_list.append(pd.DataFrame(
            {'layer': np.full(sort_inds.shape, value_label), 'rank': np.arange(len(sort_inds)),
            'channel': sort_inds, 'f-score': f_test[sort_inds]}))
        #show_f_test_results(pixels, is_obj, 'cheese', rows_in_value, cols_in_value)
    f_test_df = pd.concat(f_test_list, axis='index')
    return pixel_data, f_test_df
    # px.line(f_test_df, x='rank', y='f-score', color='layer', hover_data=['channel'],
    #     title='Ranked f-test scores for "conv pixel contains cheese" over resadd layers').show()

# Generate mazes:
venv = maze.create_venv(num=num_batch, start_level=0, num_levels=0)

# Load models:
policy1, hook1 = load_model('15', 15)
policy2, hook2 = load_model('1', 15)
AX_SIZE = 2

# Collect activations 



values_to_store = [
    'embedder.block1.conv_in0',
    'embedder.block1.res1.relu1_out',
    'embedder.block1.res1.relu2_out',
    'embedder.block1.res1.resadd_out',
    'embedder.block1.res2.relu1_out',
    'embedder.block1.res2.relu2_out',
    'embedder.block1.res2.resadd_out',
    'embedder.block2.res1.relu1_out',
    'embedder.block2.res1.relu2_out',
    'embedder.block2.res1.resadd_out',
    'embedder.block2.res2.relu1_out',
    'embedder.block2.res2.relu2_out',
    'embedder.block2.res2.resadd_out',
    'embedder.block3.res1.relu1_out',
    'embedder.block3.res1.relu2_out',
    'embedder.block3.res1.resadd_out',
    'embedder.block3.res2.relu1_out',
    'embedder.block3.res2.relu2_out',
    'embedder.block3.res2.resadd_out',
    # 'embedder.flatten_out',
    # 'embedder.relufc_out',
    # 'fc_policy_out',
]



# Run obs through model to get all the activations
obs_all = venv.reset().astype('float32')

_ = hook.run_with_input(obs_all, values_to_store=values_to_store)
_ = hook1.run_with_input(obs_all, values_to_store=values_to_store)
_ = hook2.run_with_input(obs_all, values_to_store=values_to_store)

print('activations collected')

# Get the cheese, mouse, and top-right positions for all mazes
cheese_pos = get_obj_loc_targets(venv, maze.CHEESE)
#mouse_pos = get_obj_loc_targets(venv, maze.MOUSE)
#top_right_pos = get_top_right_loc_targets(venv)

print('cheese positions collected')

value_labels_conv = [
    'embedder.block1.res1.resadd_out',
    'embedder.block1.res2.resadd_out',
    'embedder.block2.res1.resadd_out',
    'embedder.block2.res2.resadd_out',
    'embedder.block3.res1.resadd_out',
    'embedder.block3.res2.resadd_out']


value_labels_conv = [
    'embedder.block1.conv_in0',
    'embedder.block1.res1.relu1_out',
    'embedder.block1.res1.relu2_out',
    'embedder.block1.res1.resadd_out',
    'embedder.block1.res2.relu1_out',
    'embedder.block1.res2.relu2_out',
    'embedder.block1.res2.resadd_out',
    'embedder.block2.res1.relu1_out',
    'embedder.block2.res1.relu2_out',
    'embedder.block2.res1.resadd_out',
    'embedder.block2.res2.relu1_out',
    'embedder.block2.res2.relu2_out',
    'embedder.block2.res2.resadd_out',
    'embedder.block3.res1.relu1_out',
    'embedder.block3.res1.relu2_out',
    'embedder.block3.res1.resadd_out',
    'embedder.block3.res2.relu1_out',
    'embedder.block3.res2.relu2_out',
    'embedder.block3.res2.resadd_out',
    # 'embedder.flatten_out',
    # 'embedder.relufc_out',
    # 'fc_policy_out',
]



pixel_data = {}
f_test_dfs = {}

hook_to_use = hook2
pixel_data['cheese2'], f_test_dfs['cheese'] = make_pixel_data(value_labels_conv, cheese_pos, hook2)

pixel_data['cheese1'], f_test_dfs['cheese'] = make_pixel_data(value_labels_conv, cheese_pos, hook1)
# pixel_data['mouse1'], f_test_dfs['mouse'] = make_pixel_data(value_labels_conv, mouse_pos, hook1)
# pixel_data['mouse2'], f_test_dfs['mouse'] = make_pixel_data(value_labels_conv, mouse_pos, hook2)
#pixel_data['dist_to_top_right, hook1'], f_test_dfs['dist_to_top_right'] = make_pixel_data(value_labels_conv, top_right_pos, hook_to_use)
#pixel_data['dist_to_top_right, hook2'], f_test_dfs['dist_to_top_right'] = make_pixel_data(value_labels_conv, top_right_pos, hook2)
# Train probes and show results
# index_nums = np.arange(10)+1
# for obj_name, pixel_data_this in pixel_data.items():
#     scores_list = []
#     for value_label, (pixels, is_obj, rows_in_value, cols_in_value, f_test, sort_inds) in tqdm(pixel_data_this.items()):
#         for K in index_nums:
#             results = cpr.linear_probe(pixels[:,sort_inds[:K]], is_obj, C=10, random_state=42)
#             scores_list.append({'layer': value_label, 'K': K, 'score': results['test_score']})
#     scores_df = pd.DataFrame(scores_list)

#     fig = px.line(scores_df, x='layer', y='score', color='K',
#         title=f'Sparse probe scores for "conv pixel contains {obj_name}" over resadd layers and K values')
#     fig.add_hline(y=1.,  line_dash="dot", annotation_text="perfect",  annotation_position="bottom right")
#     fig.add_hline(y=0.5, line_dash="dot", annotation_text="baseline", annotation_position="bottom right")
#     fig.show()
#%%
print('pixel data collected')
# Train probes and show results
scores_df_list = []
for obj_name, pixel_data_this in pixel_data.items():
    scores_list = []
    for value_label, (pixels, is_obj, rows_in_value, cols_in_value, f_test, sort_inds) in tqdm(pixel_data_this.items()):
        value = hook_to_use.get_value_by_label(value_label)
        
        for ch_ind in tqdm(range(value.shape[1])): 
            results = cpr.linear_probe(pixels[:,ch_ind], is_obj, C=10, random_state=42) 
            scores_list.append({'layer': value_label, 'channel': ch_ind, 'score': results['test_score']}) 
    scores_df = pd.DataFrame(scores_list)
    scores_df_list.append(scores_df)

    fig = px.line(scores_df, x='layer', y='score', color='channel',
        title=f'Sparse probe scores for "conv pixel contains {obj_name}" over resadd layers and K values')
    fig.add_hline(y=1.,  line_dash="dot", annotation_text="perfect",  annotation_position="bottom right")
    fig.add_hline(y=0.5, line_dash="dot", annotation_text="baseline", annotation_position="bottom right")
    fig.show()