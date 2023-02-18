'''The script clip boundary using gdal and rasterio package. 
params = raster, vector  
return = raster. 

Created by Maxwell Owusu, 2023
'''
#%%
# import libraries 

from osgeo import gdal, ogr
from glob import glob
from subprocess import Popen
import numpy as np
import os
import matplotlib.pyplot as plt


#%%
# Clip boundary using shapefiles
polygon = 'D:/Mongolia_Grassland_Mapping/DATA/Naiman_South_Clipping_Geometry/naiman_south.shp'
raster = 'D:/Mongolia_Grassland_Mapping/DATA/spfea_27.vrt'
outfile = 'D:/Mongolia_Grassland_Mapping/DATA/Output'
name = 'SouthClip'

command = f'gdalwarp -dstnodata -9999 -cutline {polygon} -crop_to_cutline -of Gtiff {raster} "{outfile}/{name}.tif"'
Popen(command, shell=True)

#%%
polygon = 'D:/Mongolia_Grassland_Mapping/DATA/Naiman_North_Clipping_Geometry/naiman_north.shp'
raster = 'D:/Mongolia_Grassland_Mapping/DATA/spfea_27.vrt'
outfile = 'D:/Mongolia_Grassland_Mapping/DATA/Output'
name = 'NorthClip'

command = f'gdalwarp -dstnodata -9999 -cutline {polygon} -crop_to_cutline -of Gtiff {raster} "{outfile}/{name}.tif"'
Popen(command, shell=True)

# %%
