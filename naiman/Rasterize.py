'''The script create mask layer using rasterio and geopandas package. 
params = raster, vector  
return = mask layer

Created by Maxwell Owusu, 2023
'''
#%%
# import libraries

import geopandas as gpd
import rasterio
from rasterio import features
import numpy as np
import os
import matplotlib.pyplot as plt

#%%
# Rasterize

def rasterize (In_shapefile, In_raster, FileName, OutfilePath):
    df = gpd.read_file(In_shapefile)
    name = FileName
    with rasterio.open(In_raster, mode='r') as src:
        out_arr = src.read(1)
        out_profile = src.profile.copy()
        out_profile.update(count=1,
                        nodata=0,
                        dtype='float32',
                        width=src.width,
                        height=src.height,
                        crs=src.crs)
        dst_height = src.height
        dst_width = src.width
        shapes = ((geom,value) for geom, value in zip(df.geometry, df.code))
        # print(shapes)
        burned = features.rasterize(shapes=shapes, out_shape=(dst_height, dst_width),fill=0, transform=src.transform)
        plt.imshow(burned) 

    # open in 'write' mode, unpack profile info to dst
    with rasterio.open(f'{OutfilePath}{name}.tif',
                    'w', **out_profile) as dst:
        dst.write_band(1, burned)

#%%
# South Boundary
In_shapefile = 'D:/Mongolia_Grassland_Mapping/DATA/Naiman_South_Clipping_Geometry/TrainData_South.shp'
In_raster = 'D:/Mongolia_Grassland_Mapping/DATA/Output/SouthClip.tif'
FileName = 'South_mask'
OutfilePath = 'D:/Mongolia_Grassland_Mapping/DATA/Output/'

rasterize (In_shapefile, In_raster, FileName, OutfilePath)


# %%
# North Boundary
In_shapefile = 'D:/Mongolia_Grassland_Mapping/DATA/Naiman_North_Clipping_Geometry/TrainData_North.shp'
In_raster = 'D:/Mongolia_Grassland_Mapping/DATA/Output/NorthClip.tif'
FileName = 'North_mask'
OutfilePath = 'D:/Mongolia_Grassland_Mapping/DATA/Output/'

rasterize (In_shapefile, In_raster, FileName, OutfilePath)

# %%
In_shapefile = 'D:/Mongolia_Grassland_Mapping/DATA/Naiman_North_Clipping_Geometry/TrainData_North.shp'

df = gpd.read_file(In_shapefile)
df.head()
# %%
df['class'].unique()
# %%
