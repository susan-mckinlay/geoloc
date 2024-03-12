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

def calculate_distance(cA, cB):
    """
    :param cA : pair of point A (long, lat)
    :param cB : pair of point B (long, lat)
    :return distance in Km
    """
    return geopy.distance.geodesic(cA, cB).km


def calc_dist_each_rand_loc(data, season):
    tot_dist_migr = data[(data[f'compl_{season}_track'] == 1) & (data['season'] == season) & (data['Juv'] == 1)].groupby('ID')['distance'].sum().reset_index()
    #& (data[f'adults_{season}_compl_migr'] == 1)
    tot_dist_migr['dist_rand_loc'] = tot_dist_migr['distance'].div(20).round()
    tot_dist_migr['dist_rand_loc'] = tot_dist_migr['dist_rand_loc'].astype(int)
    print(tot_dist_migr)
    return tot_dist_migr

def random_loc_on_track(data, tot_dist_migr, season):
    #geoid = Geod(ellps="WGS84")
    #extra_points = geoid.inv_intermediate(data_3cx_lon.iloc[8], data_3cx_lat.iloc[8], \
    #data_3cx_lon.iloc[9], data_3cx_lat.iloc[9], del_s = 350)
    #print(extra_points)
    # I need to fix the values of 5II, res_list has 19 integers instead of 20, I think because of
    # a rounding ERROR
    # For-loop to get column with number of random locations corresponding to each distance row
    # Strategy:
    # 1. Create a column with number of random locations per segment (at the second row for each pair of coordinates)
    # 2. If the segment is long enough to be divided by the distance (= tot_length_migration/20) divide it and append that number
    # to res_list
    # 3. If the segment is not long enough to be divided by the dist., then append (0) and add the segment to km_left
    # 4. When km_left >= dist then divided it by dist and append the result to res_list
    # 5. Res_list should be the same lengtha as the dataframe, with 0s and integer numbers
    dist = 246
    res_list = []
    dist_rand_loc_list = []
    km_left = 0
    i = 0
    for _, row in data.iterrows():
        print('km left at the beginning of the for loop', km_left)
        if km_left <= dist:
            if  row['distance'] >= dist: #(i == 0) & 
                # The distance here will always be more than the dist value, so here I will never append 0
                # to res_list
                # Add km_left to row['distance'] to take them into account
                res = (row['distance'] + km_left)//dist
                print('We are in the if of the if-else statement', i)
                print('the current distance is:', row['distance'])
                print('the current result row[distance]/dist is:', res)
                # append N of random locations that are allowed per segment to list, to add it later as a column
                res_list.append(res)
                if km_left == 0: # usually at the start of the df
                    dist_rand_loc_list.append(dist)
                else:
                    rand_loc_dist = (km_left + row['distance']) - (dist * res)
                    dist_rand_loc_list.append(round(rand_loc_dist, 2))
                # Find how many km remain 
                remainder = (row['distance'] + km_left) % dist
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
            print('current row[distance]', row['distance'],'km_left:',km_left)
            res = km_left//dist
            rand_loc_dist = km_left - (dist * res)
            print('random location distance:', rand_loc_dist)
            dist_rand_loc_list.append(round(rand_loc_dist, 2))
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
    print(dist_rand_loc_list) # len(dist_rand_loc_list), sum(dist_rand_loc_list))
    data['number_locations'] = res_list
    data['distance_locations'] = dist_rand_loc_list
    return data

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

def get_lat_lon_rand_loc(data):
    # Strategy:
    # 1. Iterate over dataframe;
    # 2. If the number of locations in the seegment equals 0 then append 0 to res_list
    # 3. If the number of locations in the segment equals 1 then append the coordinates of that random location
    # 4. If the number of locations in the segment > 1 then get the coordinates of each random location
    i = 0
    res_list = []
    dist_tot = 246
    # PROBLEM TO FIX: FIRST COORDINATE OF DF SHOULD NOT BE DISCARDED, IT SHOULD BE USEDAS PREV_C IN SECOND INSTANCE
    for index, row in data.iterrows():
        if (row['number_locations'] == 0) & (i==0): # first instance
            res_list.append(0)
            prev_c = (row['modelat'], row['modelon']) # First coordinates of df need to be used
            #current_c = (data_3cx['modelon'].at[index+1], data_3cx['modelat'].at[index+1])
            print('first instance', i, index)
        elif row['number_locations'] == 0:
            res_list.append(0)
            print('if',i, 'the index is', index)
            print('res_list', res_list)
        elif row['number_locations'] == 1:
            #prev_c = (row['modelon'], row['modelat'])
            #current_c = (data_3cx['modelon'].at[index+1], data_3cx['modelat'].at[index+1])
            current_c = (row['modelat'], row['modelon'])
            print('prev_c', prev_c, 'instance number:', i, 'the index is:', index)
            print('current_c', current_c)
            dist = row['distance_locations']
            bearing = calculate_initial_compass_bearing(prev_c,current_c)
            C = Point(geodesic(kilometers=dist).destination(Point(prev_c[0], prev_c[1]), bearing))#.format_decimal()
            list_c = [float(C[0]), float(C[1])]
            C = tuple(list_c)
            res_list.append(C)
            prev_c = current_c
        else:
            # Make a separate function here
            #coords_list = []
            j = 0
            for n in range(int(row['number_locations'])):
                if j == 0: # first instance
                    # Here the prev_c is the very first row of the df, from the first if statement of the for loop
                    current_c = (row['modelat'], row['modelon'])
                    #current_c = (data_3cx['modelon'].at[index+1], data_3cx['modelat'].at[index+1])
                    print('instance number:', i, 'the index is:', index)
                    print('prev_c in the nested for loop in the if statement', prev_c, row['number_locations'])
                    print('current_c in the nested for loop in the if statement', current_c, row['number_locations'])
                    dist = row['distance_locations']
                    print('distance to calculate coordinates:', dist)
                    #C = get_coords_from_dist_and_segment(prev_c, current_c, dist)
                    # I need to make C a tuple
                    bearing = calculate_initial_compass_bearing(prev_c,current_c)
                    C = Point(geodesic(kilometers=dist).destination(Point(prev_c[0], prev_c[1]), bearing))#.format_decimal()
                    list_c = [float(C[0]), float(C[1])]
                    C = tuple(list_c)
                    print('C:', C, type(C))
                    #coords_list.append(C)
                    res_list.append(C)
                    prev_c = C # the new random point becomes the previous coordinate
                else:
                    # Here the previous coordinate is the random point that was just created previously
                    # Here the current coordinate is the same as in the previous iteration
                    print('prev_c in the nested for loop in the else statement', prev_c, row['number_locations'])
                    print('current_c in the nested for loop in the else statement', current_c, row['number_locations'])
                    # Here I use the standard dist_tot instead of row['distance_locations']
                    #sec_coords = get_coords_from_dist_and_segment(prev_c, current_c, dist_tot)
                    #coords_list.append(C)
                    bearing = calculate_initial_compass_bearing(prev_c,current_c)
                    # I need to make sec_coords a tuple
                    sec_coords = geodesic(kilometers=dist_tot).destination(Point(prev_c[0], prev_c[1]), bearing)#.format_decimal()
                    list_sec_coords = [float(sec_coords[0]), float(sec_coords[1])]
                    sec_coords = tuple(list_sec_coords)
                    print('sec coords:', sec_coords)
                    res_list.append(sec_coords)
                    # The just produced coordinates of the random location becomes the previous coordinates
                    # for the next calculation
                    prev_c = sec_coords
                j += 1
        i += 1
    print(res_list, len(res_list))
    return res_list

def calculate_coordinates(prev_c, current_c, dist):
    bearing = calculate_initial_compass_bearing(prev_c,current_c)
    # I need to make new_coords a tuple
    new_coords = geodesic(kilometers=dist).destination(Point(prev_c[1], prev_c[0]), bearing)#.format_decimal()
    # Ned to take it back to coords= lon,lat from lat,lon
    list_new_coords = [float(new_coords[1]), float(new_coords[0])]
    new_coords = tuple(list_new_coords)
    return new_coords

def fix_point_on_segment( prev_c, current_c, dist_points, dist_tot):
    print('Im in fix_point_on_segment function')
    print('prev_c used to calculate coords', prev_c, 'current_c used to calcuate coords', current_c)
    if dist_points == dist_tot:
        dist_calc_coords = dist_points
        print('distance used to calculate new coords', dist_calc_coords)
        rand_point = calculate_coordinates(prev_c, current_c, dist_calc_coords)
        prev_c = rand_point
    else:
        dist_calc_coords = dist_points
        print('distance used to calculate new coords', dist_calc_coords)
        rand_point = calculate_coordinates(prev_c, current_c, dist_calc_coords)
        prev_c = rand_point
    return rand_point, current_c, prev_c

def get_one_dist_points(data):
    i = 0
    dist_tot = 246
    km_left = 0
    list_coords = []
    for index, row in data.iterrows():
        print('The km_left at the beginning of the for loop are', km_left)
        #dist_points = row['distance']
        if i == 0:
            prev_c = (row['modelon'], row['modelat'])
            print('instance number', i ,'prev_c=', prev_c)
            dist_points = data['distance'].at[index+1] # get the first distance, which is at the second row
            print('second instance distance', dist_points)
        elif (dist_points < dist_tot) & (km_left < dist_tot):
            print('instance',i,' when dist_points < dist_tot, dist_points =', dist_points)
            km_left = km_left + dist_points
            dist_points = row['distance']
            #prev_c = (row['modelon'], row['modelat'])
        elif km_left >= dist_tot:
            print('instance number', i, 'where km_left >= dist_tot, km_left =', km_left)
            dist_points = dist_tot #km_left - dist_tot
            current_c = (row['modelon'], row['modelat'])
            print('dist_points  is', dist_points, 'previous_c is', prev_c, 'current_c is', current_c)
            rand_point, current_c, prev_c = fix_point_on_segment(prev_c, current_c, dist_points, dist_tot)
            print('The random point just generated is', rand_point)
            list_coords.append(rand_point)
            dist_points = calculate_distance(rand_point, current_c)
            km_left = 0 # reset km_left
        elif dist_points >= dist_tot:
            j = 0
            for n in range(int(row['number_locations'])): # so that it always keeps the correct current_c for the entire segment instead of going to the next one
                if j == 0:
                    print('instance number', i, 'where dist_points >= dist_tot')
                    dist_points = dist_tot # change dist_points to distance Im going to use in func to get coords of
                    # equally distant point
                    print('this is the dist used to calculate coords',dist_points)
                    current_c = (row['modelon'], row['modelat']) 
                    rand_point, current_c, prev_c = fix_point_on_segment(prev_c, current_c, dist_points, dist_tot)
                    print('The random point just generated is', rand_point)
                    list_coords.append(rand_point)
                    dist_points = calculate_distance(rand_point, current_c)
                    print('rand_point used to calculate distance', rand_point, 'and point B', current_c)
                    print('this is the new distance calculated from random_point to point B', dist_points)
                else:
                    print('instance number', i, 'where dist_points >= dist_tot')
                    dist_points = dist_tot
                    rand_point, current_c, prev_c = fix_point_on_segment(prev_c, current_c, dist_points, dist_tot)
                    print('The random point just generated is', rand_point)
                    list_coords.append(rand_point)
                    dist_points = calculate_distance(rand_point, current_c)
                    print('rand_point used to calculate distance', rand_point, 'and point B', current_c)
                    print('this is the new distance calculated from random_point to point B', dist_points)
                j += 1
        i += 1
    print(list_coords, len(list_coords))
    return list_coords

def calculate_initial_compass_bearing(pointA, pointB):
    """
    Calculates the bearing between two points.
    The formulae used is the following:
    θ = atan2(sin(Δlong).cos(lat2),
              cos(lat1).sin(lat2) − 
    sin(lat1).cos(lat2).cos(Δlong))
    :Parameters:
    - `pointA: The tuple representing the 
    latitude/longitude for the
    first point. Latitude and longitude must be in 
    decimal degrees
    - `pointB: The tuple representing the latitude/longitude for the
    second point. Latitude and longitude must be in decimal degrees
    :Returns:
    The bearing in degrees
    :Returns Type:
    float
    """
    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")

    lat1 = math.radians(pointA[1])
    lat2 = math.radians(pointB[1])

    diffLong = math.radians(pointB[0] - pointA[0])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
        * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing

def create_df_random_points(res_list, data_3cx):
    # Remove zeros from list of coordinates
    #res_list = [i for i in res_list if i != 0]
    #print('this is the final coords list', res_list)
    # Create dataframe with 3 columns: latitude, longitude and bird_id
    df_loc = pd.DataFrame.from_records(res_list, columns =['longitude','latitude'])
    df_loc['ID'] = data_3cx['ID'].iloc[0]
    print(df_loc)
    return df_loc

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
        g_point = line.Position(pd_between, Geodesic.STANDARD | Geodesic.LONG_UNROLL)
        points.append((g_point['lat2'], g_point['lon2']))
    print(points, type(points))
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
         & (data['Juv'] == 1) & (data['ID'] == '5LK')] #& (data['distance'] != 0)]
    # Remove modelat and modelon duplicates
    data_3cx = data_3cx.drop_duplicates(subset=['modelat', 'modelon'], keep='last')
    data_3cx_lat = data_3cx['modelat']
    dist_rand_loc_3cx = tot_dist_migr.loc[tot_dist_migr['ID'] == '5LK', 'dist_rand_loc']
    #print('total distance 5II',dist_rand_loc_3cx, len(data_3cx_lat))
    #data_3cx = random_loc_on_track(data_3cx, tot_dist_migr, 'spr')
    #print('last index of df where I save the df as csv', data_3cx.index[-1], data_3cx.index[0], len(data_3cx))
    #data_3cx.to_csv(sys.argv[2], index = False)
    #res_list = get_one_dist_points(data_3cx)
    #res_list = get_lat_lon_rand_loc(data_3cx)
    #data_rob = data.loc[(data[f'compl_spr_track'] == 1) & (data['season'] == 'spr')\
         #& (data['Juv'] == 1) & ((data['ID'] == '5LK') | (data['ID'] == '5IK'))]
    #data_rob.to_csv(sys.argv[3], index = False)
    #df_loc = create_df_random_points(res_list, data_3cx)
    path = [
    (57.6905, 11.9882),
    (57.6966, 11.9877),
    (57.7006, 12.0164),
    (57.6888, 12.0348),
    ]
    path = list(zip(data_3cx.modelat, data_3cx.modelon))
    n_samples = 18
    points = place_equally_distanced_points(path, n_samples)
    points_df = pd.DataFrame.from_records(points, columns =['latitude','longitude'])
    print(path, type(path))
    data = pd.DataFrame.from_records(path, columns =['latitude','longitude'])
    plt.plot(data['longitude'], data['latitude'],'b') # Draw blue line
    plt.plot(points_df['longitude'], points_df['latitude'], 'rs') # Draw red squares
    mplleaflet.show()
    #df_loc.to_csv(sys.argv[3], index = False)




if __name__ == "__main__":
    main()