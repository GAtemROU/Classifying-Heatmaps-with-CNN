import os
import re
from shutil import copy
from os.path import join


def split_data(project_path, preprocess_heatmaps=True, preprocess_scanpaths=True):

    scanpaths = "data/cropped/scanpaths/"
    heatmaps = "data/cropped/heatmaps/"
    savepath_s = lambda pp, abc, cp: f"data/datasets/scanpaths/{abc}/{cp}/{pp}/"
    savepath_h = lambda pp, cp: f"data/datasets/heatmaps/{cp}/{pp}/"

    # scanpaths
    if preprocess_scanpaths:
        for filename in os.listdir(join(project_path, scanpaths)):
            participant = re.search(r'\d+', filename).group()
            im_type = filename[-5]
            im_class = filename[-7]
            save_to = join(project_path, savepath_s(participant, im_type, im_class))
            if not os.path.exists(save_to):
                os.makedirs(save_to)
            copy(join(project_path, scanpaths)+filename, save_to+filename)

    # heatmaps
    if preprocess_heatmaps:
        for filename in os.listdir(join(project_path, heatmaps)):
            participant = re.search(r'\d+', filename).group()
            im_class = filename[-7]
            save_to = join(project_path, savepath_h(participant, im_class))
            if not os.path.exists(save_to):
                os.makedirs(save_to)
            copy(join(project_path, heatmaps)+filename, save_to+filename)

