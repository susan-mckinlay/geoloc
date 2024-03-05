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

def calc_dist_each_rand_loc(data, season):
    tot_dist_migr = data[(data[f'compl_{season}_track'] == 1) & (data['season'] == season) & (data['Juv'] == 1)].groupby('ID')['distance'].sum().reset_index()
    #& (data[f'adults_{season}_compl_migr'] == 1)
    tot_dist_migr['dist_rand_loc'] = tot_dist_migr['distance'].div(20).round()
    tot_dist_migr['dist_rand_loc'] = tot_dist_migr['dist_rand_loc'].astype(int)
    print(tot_dist_migr)
    return tot_dist_migr

def random_loc_on_track(data, tot_dist_migr, season):
    data_3cx = data.loc[(data[f'compl_{season}_track'] == 1) & (data['season'] == season)\
         & (data['Juv'] == 1) & (data['ID'] == '5LK')] #& (data['distance'] != 0)]
    # Remove modelat and modelon duplicates
    data_3cx = data_3cx.drop_duplicates(subset=['modelat', 'modelon'], keep='last')
    data_3cx_lat = data_3cx['modelat']
    data_3cx_lon = data_3cx['modelon']
    dist_rand_loc_3cx = tot_dist_migr.loc[tot_dist_migr['ID'] == '5LK', 'dist_rand_loc']
    print('total distance 5II',dist_rand_loc_3cx, len(data_3cx_lat))
    print('last index of df', data_3cx.index[-1], data_3cx.index[0], len(data_3cx))
    #geoid = Geod(ellps="WGS84")
    #extra_points = geoid.inv_intermediate(data_3cx_lon.iloc[8], data_3cx_lat.iloc[8], \
    #data_3cx_lon.iloc[9], data_3cx_lat.iloc[9], del_s = 350)
    #print(extra_points)
    # I need to fix the values of 5II, res_list has 19 integers instead of 20, I think because of
    # a rounding ERROR
    # For-loop to get column with number of random locations corresponding to each distance row
    dist = 246
    res_list = []
    dist_rand_loc_list = []
    km_left = 0
    i = 0
    for _, row in data_3cx.iterrows():
        print('km left at the beginning of the for loop', km_left)
        if km_left <= dist:
            if  (i == 0) & (row['distance'] >= dist):
                # The distance here will always be more than the dist value, so here I will never append 0
                # to res_list
                res = row['distance']//dist
                print('We are in the if of the if-else statement', i)
                print('the current distance is:', row['distance'])
                print('the current result row[distance]/dist is:', res)
                # append N of random locations that are allowed per segment to list, to add it later as a column
                res_list.append(res)
                # only append N random locations here because it's the first instance of the for loop
                dist_rand_loc_list.append(dist)
                # Find how many km remain 
                remainder = row['distance'] % dist
                # Store remaining km
                km_left += remainder
                print('The current remainder is:', remainder, 'The km left are:', km_left)
            else: # instance when row['distance'] < dist
                print("We are in the else of the if-else statement", i)
                print('The current row[distance] is:', row['distance'], f'it should be less than {dist} on the day', row['date_time'])
                # Add the segment that is too short to remaining km 
                km_left += row['distance']
                if km_left >= dist: # if km_left is divisable by dist then append res to res_list
                    print("Im here because row['distance'] < dist, I added it to km_left and then km_left became >= dist, so I km_left/dist")
                    res = km_left//dist
                    print("and this is the res:", res, "and these are the km_left", km_left)
                    rand_loc_dist = km_left - (dist * res) # so that I get the rand_loc_dist of the first
                    # random location
                    dist_rand_loc_list.append(round(rand_loc_dist, 2))
                    res_list.append(res)
                    remainder = km_left % dist
                    # Reset km_left to 0
                    km_left = 0
                    km_left += remainder
                else:
                    # Append 0 random locations because km_left is still < dist
                    res_list.append(0)
                    dist_rand_loc_list.append(0)
                print('The km left are:', km_left)
        else: # We are here when already at the beginning of the for loop km_left >= dist
            # append N of random locations that are allowed per segment to list, to add it later as a column
            print('Here Im in the else of the outer if-else statement', i )
            res = km_left//dist
            rand_loc_dist = km_left - (dist * res)
            dist_rand_loc_list.append(round(rand_loc_dist), 2)
            res_list.append(res)
            remainder = km_left % dist
            # Reset km_left to 0
            km_left = 0
            # Add the remainder of km_left//dist to km_left
            km_left += remainder
            print('the current result km_left/dist is:', res, 'in the else of the outer if-else statement')
            print('the current remainder km_left/dist is:', remainder, 'in the else of the outer if-else statement')
        i += 1
    print(res_list, len(res_list), sum(res_list))
    print(dist_rand_loc_list, len(dist_rand_loc_list), sum(dist_rand_loc_list))
    data_3cx['number_locations'] = res_list
    data_3cx['distance_locations'] = dist_rand_loc_list
    return data_3cx

def get_coords_from_dist_and_segment(prev_c, current_c, dist):
    #Define the ellipsoid
    geod = Geodesic.WGS84
    # I want to get the same number of coordinates as the number of locations
    # Solve the Inverse problem
    inv = geod.Inverse(prev_c[0], prev_c[1], current_c[0], current_c[1])
    azi1 = inv['azi1']
    print('Initial Azimuth from A to B = ' + str(azi1))
    #Solve the Direct problem
    dir = geod.Direct(prev_c[0],prev_c[1],azi1,dist)
    C = (dir['lat2'],dir['lon2'])
    print('C = ' + str(C))
    return C

def get_coords_multiple_random_loc(prev_c, current_c, dist, numb_loc):
    list_coords = []
    for n in numb_loc:
        C = get_coords_from_dist_and_segment(prev_c, current_c, dist)
        prev_c = C

def get_lat_lon_rand_loc(data_3cx):
    # Strategy:
    # 1. Iterate over dataframe;
    # 2. If the number of locations in the seegment equals 0 then append 0 to res_list
    # 3. If the number of locations in the segment equals 1 then append the coordinates of that random location
    # 4. If the number of locations in the segment > 1 then get the coordinates of each random location
    # I NEED TO FIX THE LAST INSTANCE OF THIS FOR-LOOP
    print('last index of df', data_3cx.index[-1], data_3cx.index[0], len(data_3cx), data_3cx.index, len(data_3cx.index))
    i = 0
    res_list = []
    dist_tot = 246
    # PROBLEM TO FIX: FIRST COORDINATE OF DF SHOULD NOT BE DISCARDED, IT SHOULD BE USEDAS PREV_C IN SECOND INSTANCE
    for index, row in data_3cx.iterrows():
        if (row['number_locations'] == 0) & (i==0): # first instance
            res_list.append(0)
            prev_c = (row['modelon'], row['modelat'])
            #current_c = (data_3cx['modelon'].at[index+1], data_3cx['modelat'].at[index+1])
            print('first instance', i, index)
        elif row['number_locations'] == 0:
            res_list.append(0)
            print('if',i, 'the index is', index)
            print('res_list', res_list)
        elif row['number_locations'] == 1:
            #prev_c = (row['modelon'], row['modelat'])
            #current_c = (data_3cx['modelon'].at[index+1], data_3cx['modelat'].at[index+1])
            current_c = (row['modelon'], row['modelat'])
            print('prev_c', prev_c, 'instance number:', i, 'the index is:', index)
            print('current_c', current_c)
            dist = row['distance_locations']
            C = get_coords_from_dist_and_segment(prev_c, current_c, dist)
            res_list.append(C)
            prev_c = current_c
        else:
            # Make a separate function here
            j = 0
            for n in range(int(row['number_locations'])):
                if j == 0: # first instance
                    # Here the prev_c is the very first row of the df, from the first if statement of the for loop
                    current_c = (row['modelon'], row['modelat'])
                    #current_c = (data_3cx['modelon'].at[index+1], data_3cx['modelat'].at[index+1])
                    print('instance number:', i, 'the index is:', index)
                    print('prev_c in the nested for loop in the if statement', prev_c, row['number_locations'])
                    print('current_c in the nested for loop in the if statement', current_c, row['number_locations'])
                    dist = row['distance_locations']
                    C = get_coords_from_dist_and_segment(prev_c, current_c, dist)
                    res_list.append(C)
                    prev_c = C # the new random point becomes the previous coordinate
                else:
                    # Here the previous coordinate is the random point that was just created previously
                    # Here the current coordinate is the same as in the previous iteration
                    print('prev_c in the nested for loop in the else statement', prev_c, row['number_locations'])
                    print('current_c in the nested for loop in the else statement', current_c, row['number_locations'])
                    # Here I use the standard dist_tot instead of row['distance_locations']
                    sec_coords = get_coords_from_dist_and_segment(prev_c, current_c, dist_tot)
                    res_list.append(sec_coords)
                    # The just produced coordinates of the random location becomes the previous coordinates
                    # for the next calculation
                    prev_c = sec_coords
                j += 1
        i += 1
    print(res_list, len(res_list))


def main():
    if len(sys.argv) < 1:
        exit('python3.9 os/one_way_dist.py output_files/tracks_with_dist.csv')
    pd.set_option('display.max_rows', None)
    warnings.filterwarnings("ignore")
    data = pd.read_csv(sys.argv[1])
    data['date_time'] = pd.to_datetime(data['date_time'])
    tot_dist_migr = calc_dist_each_rand_loc(data, 'spr')
    data_3cx = random_loc_on_track(data, tot_dist_migr, 'spr')
    print('last index of df where I save the df as csv', data_3cx.index[-1], data_3cx.index[0], len(data_3cx))
    data_3cx.to_csv(sys.argv[2], index = False)
    get_lat_lon_rand_loc(data_3cx)



if __name__ == "__main__":
    main()