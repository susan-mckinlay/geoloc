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
import geopy.distance
import warnings
from pyproj import Geod
from geographiclib.geodesic import Geodesic
import mplleaflet
import math 
from geopy.distance import Point
from geopy.distance import geodesic

def calc_dist_each_rand_loc(data, season):
    tot_dist_migr = data[(data[f'compl_{season}_track'] == 1) & (data['season'] == season) & (data['Juv'] == 1)].groupby('ID')['distance'].sum().reset_index()
    #& (data[f'adults_{season}_compl_migr'] == 1)
    tot_dist_migr['dist_rand_loc'] = tot_dist_migr['distance'].div(20).round()
    tot_dist_migr['dist_rand_loc'] = tot_dist_migr['dist_rand_loc'].astype(int)
    print(tot_dist_migr)
    return tot_dist_migr

def place_equally_distanced_points(path, n_samples):
    # Distance between each path coord.
    geod = Geodesic.WGS84
    path_distance = [0]
    for (start_lat, start_lon), (end_lat, end_lon) in zip(path[:-1], path[1:]):
        path_distance.append(geod.Inverse(start_lat, start_lon, end_lat, end_lon)['s12'])
    path_distance_cum = np.cumsum(path_distance)
    point_distance = np.linspace(0, path_distance_cum[-1], n_samples)

    points = []
    for pd in point_distance:
    
        # Find segment with.
        i_start = np.argwhere(pd >= path_distance_cum)[:, 0][-1]
    
        # Early exit for ends.
        if np.isclose(pd, path_distance_cum[i_start]):
            points.append(path[i_start])
            continue
        elif np.isclose(pd, path_distance_cum[-1]):
            points.append(path[-1])
            continue
        
        # Distance along segment.
        start_lat, start_lon = path[i_start]
        end_lat, end_lon = path[i_start + 1]
        pd_between = pd - path_distance_cum[i_start]
    
        # Location along segment.
        line = geod.InverseLine(start_lat, start_lon, end_lat, end_lon)
        print(line)
        g_point = line.Position(pd_between, Geodesic.STANDARD | Geodesic.LONG_UNROLL)
        points.append((g_point['lat2'], g_point['lon2']))
    return points


def main():
    if len(sys.argv) < 1:
        exit('python3.9 os/one_way_dist.py output_files/tracks_with_dist.csv output_files/tracks_rand_loc.csv output_files/random_loc_5LK.csv')
    pd.set_option('display.max_rows', None)
    warnings.filterwarnings("ignore")
    data = pd.read_csv(sys.argv[1])
    data['date_time'] = pd.to_datetime(data['date_time'])
    season = 'spr'
    tot_dist_migr = calc_dist_each_rand_loc(data, season)
    data_3cx = data.loc[(data[f'compl_{season}_track'] == 1) & (data['season'] == season)\
         & (data['Juv'] == 1) & (data['ID'] == '5IK')] #& (data['distance'] != 0)]
    # Remove modelat and modelon duplicates
    data_3cx = data_3cx.drop_duplicates(subset=['modelat', 'modelon'], keep='last')
    path = list(zip(data_3cx.modelat, data_3cx.modelon))
    n_samples = 18
    points = place_equally_distanced_points(path, n_samples)
    points_df = pd.DataFrame.from_records(points, columns =['latitude','longitude'])
    print(path, type(path))
    data = pd.DataFrame.from_records(path, columns =['latitude','longitude'])
    plt.plot(data['longitude'], data['latitude'],'b') # Draw blue line
    plt.plot(data['longitude'], data['latitude'],'bs') 
    plt.plot(points_df['longitude'], points_df['latitude'], 'rs') # Draw red squares
    mplleaflet.show()
    #df_loc.to_csv(sys.argv[3], index = False)




if __name__ == "__main__":
    main()