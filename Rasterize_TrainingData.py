#%%
import rasterio
from rasterio import features
# import os
import matplotlib.pyplot as plt
from glob import glob
import geopandas as gpd


#%%
def rasterize_me(in_shp, in_raster, outfile):
    # read shapfile 
    # for i in lst:
        df = gpd.read_file(in_shp)
        # add output file name
        # head, tail = os.path.split(i[0])
        name='naiman_train'
    # read raster
        with rasterio.open(in_shp, mode="r") as src:
            out_arr = src.read(1)
            out_profile = src.profile.copy()
            out_profile.update(count=1,
                            nodata=-9999,
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
        with rasterio.open(f'{outfile}{name}.tif',
                        'w', **out_profile) as dst:
            dst.write_band(1, burned)

polygons = sorted(glob('D:/mongolia/mongolia_ml_model/new_train_data/naiman_traindata.geojson'))
RasterTiles = sorted(glob("D:/mongolia/GITHUB/features_full/fourier_sub/fourier_sc10_mean.tif"))
outfile = 'D:/mongolia/mongolia_ml_model/new_train_data/'


rasterize_me(in_shp=polygons, in_raster=RasterTiles, outfile=outfile)
# %%
