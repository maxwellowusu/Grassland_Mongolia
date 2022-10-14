
import sys
import numpy as np
from osgeo import gdal
from osgeo import ogr
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import os
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import rasterio as rio
from rasterio.mask import mask

# function to load raster data
def read_image(dir):
    """This function read raster image, convert to array and get the geoprojection"""
    img = gdal.Open(os.path.join(dir))
    img_arr = img.ReadAsArray()
    img_gt = img.GetGeoTransform()
    img_georef = img.GetProjectionRef()
    return [img_arr, img_gt, img_georef]

# function to get to latitude
def pixel_id_to_lat(GT, Y):
    """This function gets the latitude from raster data"""
    lat = GT[5] * Y + GT[3]
    return lat
    # return round(lat,2)

# function to get the longitude
def pixel_id_to_lon(GT, X):
    """This function gets the longitude from raster data"""
    lon = GT[1] * X + GT[0]
    return lon
    # return round(lon,2)

# function to pixel coordinate
def coord_to_pixel_id(GT, lat, lon):
    """This function gets the pixel coordinate"""
    Y = (lat - GT[3]) // GT[5]
    X = (lon - GT[0]) // GT[1]
    return (X, Y)

# # """ **************************************************************************
# function for plotting image
def hist_stretch_all(arr, bits, clip_extremes):
    """This function perform histogram stretch"""
    n = arr.shape
    per = np.percentile(arr, [2.5, 97.5])
    per_max = per[1]
    per_min = per[0]
    min_arr = np.full(n, per_min)
    max_arr = np.full(n, per_max)
    if (clip_extremes == False):
        new_arr = arr
    else:
        new_arr = np.maximum(min_arr, np.minimum(max_arr, arr))
    if (bits == 0):
        min_ = np.amin(new_arr)
        max_ = np.amax(new_arr)
        new_arr = (new_arr - min_) / (max_ - min_)
    else:
        new_arr = np.floor((2 ** bits - 1) * (new_arr - per_min) / (per_max - per_min))
    return new_arr


def display(arr, img_gt, cmap, x_label, y_label, title, num_ticks=5, colorbar=True):
    """This function display plots, add labels and title"""
    figure(num=None, figsize=(10, 10), dpi=100, facecolor='w', edgecolor='k')
    imgplot = plt.imshow(arr, cmap=cmap)
    # plt.set_yticklabels()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.title('('+mat_ele+') '+title)
    plt.title(title)
    lat_pixels = list(range(0, arr.shape[0], arr.shape[0] // num_ticks))
    lon_pixels = list(range(0, arr.shape[1], arr.shape[1] // num_ticks))

    plt.xticks(lon_pixels, [str(round(pixel_id_to_lon(img_gt, i), 2)) for i in lon_pixels])
    plt.yticks(lat_pixels, [str(round(pixel_id_to_lat(img_gt, i), 2)) for i in lat_pixels])

    if colorbar:
        plt.colorbar()

    # plt.savefig
    plt.show()

def plot_image(img_arr, img_gt, rgb_bands, clip_extremes=False):
    """This function plot image """
    r = hist_stretch_all(img_arr[rgb_bands[0]], 0, clip_extremes)
    b = hist_stretch_all(img_arr[rgb_bands[2]], 0, clip_extremes)
    g = hist_stretch_all(img_arr[rgb_bands[1]], 0, clip_extremes)
    img = np.dstack((r, g, b))
    display(img, img_gt, None, 'Easting (m)', 'Northing (m)', 'Planet Image', num_ticks=5, colorbar=False)


# function to make dataframe
def make_data_frame(img_arr, col_names):
    shp=img_arr.shape
    print(shp)
    #return 0
    if len(shp)>2:
        data=img_arr.flatten().reshape(shp[0],shp[1]*shp[2]).T
    else:
        data = img_arr.flatten()
    df=pd.DataFrame(data, columns=col_names)
    df=df.where(df!=0)
    #print(df)
    return df