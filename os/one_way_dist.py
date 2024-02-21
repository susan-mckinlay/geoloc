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

def calc_dist_each_rand_loc(data, season):
    tot_dist_migr = data[(data[f'compl_{season}_track'] == 1) & (data['season'] == season) & (data['Juv'] == 1)].groupby('ID')['distance'].sum().reset_index()
    #& (data[f'adults_{season}_compl_migr'] == 1)
    tot_dist_migr['dist_rand_loc'] = tot_dist_migr['distance'].div(20).round()
    tot_dist_migr['dist_rand_loc'] = tot_dist_migr['dist_rand_loc'].astype(int)
    print(tot_dist_migr)
    return tot_dist_migr

def random_loc_on_track(data, tot_dist_migr, season):
    data_3cx = data.loc[(data[f'compl_{season}_track'] == 1) & (data['season'] == season)\
         & (data['Juv'] == 1) & (data['ID'] == '5LK') & (data['distance'] != 0)]
    data_3cx_lat = data_3cx['modelat']
    data_3cx_lon = data_3cx['modelon']
    dist_rand_loc_3cx = tot_dist_migr.loc[tot_dist_migr['ID'] == '5LK', 'dist_rand_loc']
    print('total distance 5II',dist_rand_loc_3cx, len(data_3cx_lat))
    #geoid = Geod(ellps="WGS84")
    #extra_points = geoid.inv_intermediate(data_3cx_lon.iloc[8], data_3cx_lat.iloc[8], \
        #data_3cx_lon.iloc[9], data_3cx_lat.iloc[9], del_s = 350)
    #print(extra_points)
    # I need to fix the values of 5II, res_list has 19 integers instead of 20, I think because of
    # a rounding ERROR
    # For-loop to get column with number of random locations corresponding to each distance row
    dist = 246
    res_list = []
    km_left = 0
    i = 0
    for _, row in data_3cx.iterrows():
        print('km left at the beginning of the for loop', km_left)
        if km_left <= dist:
            if  (i == 0) & (row['distance'] >= dist):
                res = row['distance']//dist
                print('We are in the if of the if-else statement')
                print('the current distance is:', row['distance'])
                print('the current result row[distance]/dist is:', res)
                # append N of random locations that are allowed per segment to list, to add it later as a column
                res_list.append(res)
                # Find how many km remain 
                remainder = row['distance'] % dist
                # Store remaining km
                km_left += remainder
                i += 1
                print('The current remainder is:', remainder, 'The km left are:', km_left)
            else: # instance when row['distance'] < dist
                print("We are in the else of the if-else statement")
                print('The current row[distance] is:', row['distance'], f'it should be less than {dist} on the day', row['date_time'])
                # Append 0 random locations because the segment is too short (row['distance'] < dist)
                #res_list.append(0)
                # Add the segment that is too short to remaining km 
                km_left += row['distance']
                if km_left >= dist:
                    res = km_left//dist
                    res_list.append(res)
                    remainder = km_left % dist
                    # Reset km_left to 0
                    km_left = 0
                    km_left += remainder
                else:
                    # Append 0 random locations because the segment is too short (row['distance'] < dist)
                    res_list.append(0)
                # The probelm is that is km_left >= dist here then I should add res to res_list immediately!
                print('The km left are:', km_left)
        else:
            # append N of random locations that are allowed per segment to list, to add it later as a column
            #res = row['distance']//dist
            print('Here Im in the else of the outer if-else statement' )
            res = km_left//dist
            res_list.append(res)
            remainder = km_left % dist
            # Reset km_left to 0
            km_left = 0
            # Add the remainder of km_left//dist to km_left
            km_left += remainder
            print('the current result km_left/dist is:', res, 'in the else of the outer if-else statement')
            print('the current remainder km_left/dist is:', remainder, 'in the else of the outer if-else statement')
    print(res_list, len(res_list), sum(res_list))

def main():
    if len(sys.argv) < 1:
        exit('python3.9 os/one_way_dist.py output_files/tracks_with_dist.csv')
    pd.set_option('display.max_rows', None)
    warnings.filterwarnings("ignore")
    data = pd.read_csv(sys.argv[1])
    data['date_time'] = pd.to_datetime(data['date_time'])
    tot_dist_migr = calc_dist_each_rand_loc(data, 'spr')
    random_loc_on_track(data, tot_dist_migr, 'spr')



if __name__ == "__main__":
    main()