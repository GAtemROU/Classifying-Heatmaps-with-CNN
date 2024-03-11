import os
from os.path import join
from PIL import Image


def crop_scanpaths(
        project_path,
        y_bottom=399,
        y_top=599,
        x_left=100,
        x_right=1400):
    orig_scanpaths = "data/rawdata/scanpaths_fixed/"
    save_scanpaths = "data/cropped/scanpaths/"

    save_path = join(project_path, save_scanpaths)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for filename in os.listdir(join(project_path, orig_scanpaths)):
        image = Image.open(join(project_path, orig_scanpaths, filename))
        image = image.crop((x_left, y_bottom, x_right, y_top))
        image.save(save_path + filename)


def crop_heatmaps(
        project_path,
        y_bottom=399,
        y_top=599,
        x_left=0,
        x_right=1500):
    orig_heatmaps = "data/rawdata/heatmap_fixed/"
    save_heatmaps = "data/cropped/heatmaps/"

    save_path = join(project_path, save_heatmaps)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for filename in os.listdir(join(project_path, orig_heatmaps)):
        image = Image.open(join(project_path, orig_heatmaps, filename))
        image = image.crop((x_left, y_bottom, x_right, y_top))
        image.save(save_path + filename)
