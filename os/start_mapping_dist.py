import csv
from re import I
import sys
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os
import matplotlib.pylab as pl
from matplotlib.image import imread
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.img_tiles import Stamen
import cartopy.io.img_tiles as cimgt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.feature import NaturalEarthFeature
import geopandas as gpd
#from matplotlib_scalebar.scalebar import ScaleBar
from geopy.distance import geodesic
from shapely.geometry.point import Point
import cartopy.io.shapereader as shpreader
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from cycler import cycler
from matplotlib.legend import Legend

def setting_up_the_map(ax):
    fname = os.path.join('/Users/susanellenmckinlay/Documents/python/woodcock/input_files/tif_files/', 'HYP_HR_SR_W.tif')
    SOURCE = 'Natural Earth'
    LICENSE = 'public domain'
    source_proj = ccrs.PlateCarree()
    ax.imshow(imread(fname), origin='upper', transform=source_proj, extent=[-180, 180, -90, 90])
    #ax.add_feature(land_50m, edgecolor='gray')
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    #ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, edgecolor='gray')
    grid_lines = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels = True, zorder = 1)
    grid_lines.top_labels = False
    grid_lines.right_labels = False
    grid_lines.xformatter = LONGITUDE_FORMATTER
    grid_lines.yformatter = LATITUDE_FORMATTER
    return ax

def all_birds_map(data, save_fig):
    plt.figure(figsize=(30,10))
    ax = plt.axes(projection= ccrs.PlateCarree())
    ax = setting_up_the_map(ax)
    # Set the background of the map
    ax.set_extent((-20.0, 23.0, 55.0, 0.0), crs=ccrs.PlateCarree()) #(-20.0, 43.0, 55.0, -37.0)
    shpfilename = shpreader.natural_earth(resolution='110m',
                                      category='cultural',
                                      name='admin_0_countries')

    reader = shpreader.Reader(shpfilename)
    # Add names of some countries on map
    countries = reader.records()
    list_countries = ['Ukraine', 'Germany', 'Lithuania', 'Kazakhstan']
    for country in countries:
        for i in list_countries:
            if country.attributes['NAME'] == i:
                x = country.geometry.centroid.x       
                y = country.geometry.centroid.y
                ax.text(x, y, i, color='white', size=11, ha='center', va='center', transform=ccrs.PlateCarree())
    # Drop NaN values in modelat and modelon columns
    #df = df.dropna(subset=["id"])
    data = data.dropna(subset = ['modelat','modelon'])
    data = data.loc[(data['Juv'] == 1) & (data['compl_spr_track'] == 1) & (data['season'] == 'spr')\
    & (data['stationary'] == False) & (data['typeofstopover'] == 'migration')]
    # Set up the tracks per individual that will be represented on the map
    bird_id = data['ID'].unique()
    print('total number of birds is', len(bird_id))
    print('first longitude of 5IK\n',data.loc[(data['ID'] == "5IK"), 'modelon'].iloc[0])
    # I need to choose a proper set of colors
    ax.set_prop_cycle('color',plt.cm.gist_rainbow(np.linspace(0,1,len(bird_id))))
    for bird in bird_id:
        try:
            x = data.loc[(data['ID'] == bird), 'modelon']
            y = data.loc[(data['ID'] == bird), 'modelat']
            x2 = data.loc[(data['ID'] == bird), 'modelon'].iloc[0]
            y2 = data.loc[(data['ID'] == bird), 'modelat'].iloc[0]
            ax.plot(x,y,'-', transform=ccrs.PlateCarree(), linewidth = 1.5, label = bird) # to use dictionary for colors: , color = colors[bird]
            ax.plot(x,y,'.',transform=ccrs.PlateCarree(), label = bird, c = 'black', zorder = 1)
            ax.plot(x2,y2,'.',transform=ccrs.PlateCarree(), label = bird, c = 'red', zorder = 1)
        except ValueError: # raised when there is a NaN value maybe?
            pass
    plt.legend(fontsize='small', loc="upper left")
    plt.savefig('output_files/images/'+save_fig+'.png', dpi=500, format='jpg', bbox_inches="tight")
    plt.show()


def main():
    if len(sys.argv) < 1:
        exit('python3.10 os/start_mapping.py input_files/tracks.csv')
    pd.set_option('display.max_rows', None)
    data = pd.read_csv(sys.argv[1], sep = ';', na_values = ['NA','a'], decimal=',')
    data['date_time'] = pd.to_datetime(data['date_time'])
    # replace NA string with NaN values so that I can convert the whole column to integer values
    # Replace on all selected columns
    # df2 = df.replace(r'^\s*$', np.nan, regex=True)
    # df['Courses'] = df['Courses'].replace('Spark','Apache Spark')
    data['modelat'] = data['modelat'].replace('NA', np.nan, regex=True)
    data['modelon'] = data['modelon'].replace('NA', np.nan, regex=True)
    data['modelat'] = data['modelat'].astype(float)
    data['modelon'] = data['modelon'].astype(float)
    data = data.sort_values(by = ['date_time'])
    # new_df = new_df.sort_values(by = ['date'])
    #print('type column modelon',data['modelon'].dtype, type(data['modelon'].iloc[4]))
    #print('type column modelat',data['modelat'].dtype, type(data['modelat'].iloc[4]))
    #print('first longitude of 5IK\n',data.loc[(data['ID'] == "5IK"), 'modelon'].iloc[0])
    all_birds_map(data, '5IK_spring_juv_non_stationary_migration_map')



if __name__ == "__main__":
    main()