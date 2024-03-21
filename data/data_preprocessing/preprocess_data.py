import os
from os.path import join
from shutil import rmtree

from data_splitter import split_data
from data_cropper import crop_scanpaths
from data_cropper import crop_heatmaps


def create_datasets(project_path, create_heatmaps=True, create_scanpaths=True):
    assert (os.path.exists(project_path))
    if create_scanpaths:
        crop_scanpaths(project_path)
    if create_heatmaps:
        crop_heatmaps(project_path)
    split_data(project_path, create_heatmaps, create_scanpaths)
    rmtree(join(project_path, "data/cropped/"))

if __name__ == '__main__':
    create_datasets("/home/gatemrou/uds/Eye_Tracking/")
