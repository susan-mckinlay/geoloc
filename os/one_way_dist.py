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

def add_distance_in_dataframe(group):
    """
    add distance field in the dataframe.
    returns a dataframe with the distance field
    """
    distance = 0
    array_dist = []
    i = 0
    for _, row in group.iterrows():
        if i == 0:
            # first instance
            prev_c = (row['modelat'], row['modelon'])
            current_c = prev_c
        else:
            current_c = (row['modelat'], row['modelon'])
        distance = calculate_distance(prev_c, current_c)
        array_dist.append(distance)
        prev_c = current_c
        i += 1
    group['distance'] = array_dist
    return group

def calc_dist_each_rand_loc(data, season):
    """
    Function to calculate total migration distance for each individual
    """
    tot_dist_migr = data[(data[f'compl_{season}_track'] == 1) & (data['season'] == season) & (data['Juv'] == 1)].groupby('ID')['distance'].sum().reset_index()
    #& (data[f'adults_{season}_compl_migr'] == 1)
    tot_dist_migr['dist_rand_loc'] = tot_dist_migr['distance'].div(20).round()
    tot_dist_migr['dist_rand_loc'] = tot_dist_migr['dist_rand_loc'].astype(int)
    print(tot_dist_migr)
    return tot_dist_migr

def prepare_dataset(data, season, indiv):
    """
    Function to drop duplicates of 'modelat' and 'modelon' column
    """
    data_indiv = data.loc[(data[f'compl_{season}_track'] == 1) & (data['season'] == season)\
         & (data['Juv'] == 1) & (data['ID'] == indiv)]
    # Remove modelat and modelon duplicates
    data_indiv = data_indiv.drop_duplicates(subset=['modelat', 'modelon'], keep='last')
    return data_indiv

def from_df_to_list_of_tuples(data_indiv):
    """
    Function that takes two columns of a dataframe and returns a list of tuples
    """
    # Create a list of tuples of coordinates (lat, long)
    path = list(zip(data_indiv.modelat, data_indiv.modelon))
    return path

def select_shortest_distance(points_indiv_1, points_indiv_2):
    """
    It gets a point 1 in Track A (list of tuples of coords)
    and finds the shortest distance between point 1 and all the other
    random points in Track B (list of tuples of coords) and appends it to a list called min_distance_list 
    and so on until all the points are done. It returns a list with all the minimum distances
    """
    # Reverse the order of the tuples because right now it is (lat, long) but I need it like this
    # (long, lat)
    points_indiv_1 = [tuple(reversed(t)) for t in points_indiv_1]
    points_indiv_2 = [tuple(reversed(t)) for t in points_indiv_2]
    tot_distance_list = []
    min_distance_list = []
    i = 0
    j = 0
    for point_track_a in points_indiv_1:
        for point_track_b in points_indiv_2:
            distance = calculate_distance(point_track_a, point_track_b)
            tot_distance_list.append(distance)
            i += 1
        min_distance_list.append(min(tot_distance_list))
        # reset tot_distance to an empty list so it resets every time it finishes the inner loop, so it only
        # appends the distances of one point with all the other 20 points, the list empties when it gets to point 2
        # of Track A
        tot_distance_list = []
        j += 1
    return min_distance_list

def change_lat_and_long(data_indiv, interpolated_lat_45, interpolated_long_45, interpolated_lat_17, interpolated_long_17):
    """
    It finds the corresponding progressive number (value of the column 'prog_number') of the closest latitude
    to the limits of 45 N and 17 N.
    It then modifies that row with the interpolated latitude and longitude (lat = 45 or = 17).
    It returns the dataframe of the individual with all the latitudes below 45 N and above 17 N.
    """
    # Interpolate location above latitude of 45 N and below latitude of 17 N
    # Modify the coordinates closest to 17 N and 45 N to the interpolation I got from QGIS
    # Get rid of locations above latitude 45 N and below latitude 17 N, so only select
    # locations below 45 N and above 17 N
    first_row_45 = data_indiv.loc[(data_indiv['modelat'] > 45)]
    first_row_17 =  data_indiv.loc[(data_indiv['modelat'] < 17)]
    #print('First coordinates that are above 45 N \n', first_row_45[['date_time','ID','modelat']])
    #print('First coordinates that are below latitude 17 N \n', first_row_17[['date_time','ID','modelat']])
    limit = 45
    data_indiv_fixed = find_closest_values(first_row_45, limit, data_indiv, interpolated_lat_45, interpolated_long_45)
    limit = 17
    data_indiv_fixed = find_closest_values(first_row_17, limit, data_indiv, interpolated_lat_17, interpolated_long_17)
    # Filter out the points above 45 N and below 17 N that we don't need, since we are not interested in the
    # breeding and wintering areas, we only want to consider the migration track
    data_indiv_fixed = data_indiv_fixed.loc[(data_indiv['modelat'] <= 45) & (data_indiv['modelat'] >= 17)]
    print('data indiv fixed\n', data_indiv_fixed[['prog_number','ID','modelon','modelat']])
    return data_indiv_fixed

def find_closest_values(first_row, limit, data_indiv, interpolated_lat, interpolated_long):
    """
    It finds the closest value fo the column 'modelat' to the limits 45 N and 17 N.
    It then gets the index of the closest value to change the value of 'modelat' and 'modelon'
    to the interpolated latitude and longitude.
    """
    index_closest_value = find_index_closest_value(first_row, limit)
    prog_number = first_row['prog_number'].iloc[index_closest_value]
    data_indiv.loc[data_indiv['prog_number'] == prog_number, 'modelat'] = interpolated_lat
    data_indiv.loc[data_indiv['prog_number'] == prog_number, 'modelon'] = interpolated_long
    return data_indiv
        
def place_equally_distanced_points(path, n_samples):
    """
    It takes the list of tuples of coordinates (lat, long) of an individual and creates 20 equally distanced 
    points on the track. It returns the coordinates of these points as a list of tuples (lat, lon).
    """
    # Distance between each path coord.
    geod = Geodesic.WGS84
    path_distance = [0] # the distance in between each coordinate (the original ones)
    for (start_lat, start_lon), (end_lat, end_lon) in zip(path[:-1], path[1:]):
        path_distance.append(geod.Inverse(start_lat, start_lon, end_lat, end_lon)['s12'])
    # Cumulative sum of each distance segment (of the original coordinates)
    path_distance_cum = np.cumsum(path_distance)
    # np.linspace returns evenly spaced numbers over a specified interval, which in this case goes from 0
    # to the last element of the cumulative sum
    # Distance in m between each equally distanced point
    # The distance of each sampled point along the line will be between 0 and the total distance between all path coordinates.
    point_distance = np.linspace(0, path_distance_cum[-1], n_samples)

    points = []
    for pd in point_distance:
        # Find segment with.
        # np.argwhere() returns the indices of elements that are non-zero.
        # In this case it outputs indices of the distance between equally distanced points that is greater than
        # the cumulative sum of the distances of the original coordinates (usually the next following point
        # since we are using an array with cumulative sums)
        # to first find which path segment each point lies on
        i_start = np.argwhere(pd >= path_distance_cum)[:,0][-1]
        # It only returns indices when this condition is met; i_start is the index of when the cum_distance of the equally
        # distanced point is bigger than the array of cumulative distances of the original coordinates
        # Early exit for ends.
        if np.isclose(pd, path_distance_cum[i_start]):
        # When comparing arrays, Numpy’s isclose function returns an array of Boolean values, 
        # with each element indicating whether the corresponding element in pd (cumulative dist. between eq.
        # points) is close to the index cumulative sum of the distances between the original coordinates
            points.append(path[i_start]) # The original coordinates indexed at the segment with the point
            # So if the equally distanced point is very close to an end of a segment just append the coords
            # of that point
            continue
        elif np.isclose(pd, path_distance_cum[-1]):
            # So if the equally distanced point is very close to the last point of the path just append the coords
            # of that point
            points.append(path[-1])
            continue
        # Distance along segment.
        # Get the coordinates of Point A and Point B (original coordinates) of the segment
        start_lat, start_lon = path[i_start]
        end_lat, end_lon = path[i_start + 1]
        # pd_between is the distance along the segment at which I need to place the equally distanced point
        # pd = cumulative sum of equally distanced point, i_start is the index at which 
        pd_between = pd - path_distance_cum[i_start]
        # Location along segment.
        # Construct a geodescic line instead of a straight segement between the original coordinates
        line = geod.InverseLine(start_lat, start_lon, end_lat, end_lon)
        g_point = line.Position(pd_between, Geodesic.STANDARD | Geodesic.LONG_UNROLL)
        points.append((g_point['lat2'], g_point['lon2']))
    return points

def plot_maps_migration(points, path):
    """
    This function converts the list of tuples into a dataframe with 'latitude' and 'longitude' as columns.
    It then maps out the migration track of the individual with the equally distanced points. 
    It returns the dataframe with the coordinates of the equally distanced points.
    """
    points_df = pd.DataFrame.from_records(points, columns =['latitude','longitude'])
    data = pd.DataFrame.from_records(path, columns =['latitude','longitude'])
    plt.plot(data['longitude'], data['latitude'],'b') # Draw blue line
    plt.plot(data['longitude'], data['latitude'],'bs') 
    plt.plot(points_df['longitude'], points_df['latitude'], 'rs') # Draw red squares
    mplleaflet.show()
    return points_df

def find_index_closest_value(data, limit):
    """
    Function that takes the data of the individual and the latitude limit (45 N or 17 N) and returns the
    index of the row with the closest value to the limit
    """
    # Find values that are closest to given values and return the row index
    difference = np.abs(limit - data['modelat'])
    index_closest_value = np.argmin(difference, axis=0)
    return index_closest_value

def get_coordinates_between_interpolation(data_indiv, limit):
    """
    Function that takes the data of the individual and the latitude limit (45 N or 17 N) and returns the 
    coordinates of point A and point B of the last segment of the migration track where I'll need to interpolate
    the longitude of the point that has as latitude the limit (45 N or 17 N)
    """
    # Here I'm selecting all the points of the original migration track of the individual above the limit
    data_bigger = data_indiv.loc[(data_indiv['modelat'] > limit)]
    index_data_bigger = find_index_closest_value(data_bigger, limit)
    # Get the progressive number of the coordinates of the point just above the limit
    prog_number_bigger = data_bigger['prog_number'].iloc[index_data_bigger]
    # With the selected progressive number use .loc to get the exact coordinates
    second_lat = data_bigger.loc[data_bigger['prog_number'] == prog_number_bigger, 'modelat'].values[0]
    second_long = data_bigger.loc[data_bigger['prog_number'] == prog_number_bigger, 'modelon'].values[0]

    # Here I'm selecting all the points of the original migration track of the individual below the limit
    data_smaller = data_indiv.loc[(data_indiv['modelat'] < limit)]
    index_number_smaller = find_index_closest_value(data_smaller, limit)
    # Get the progressive number of the coordinates of the point just below the limit
    prog_number_smaller = data_smaller['prog_number'].iloc[index_number_smaller]
    # With the selected progressive number use .loc to get the exact coordinates
    first_lat = data_smaller.loc[data_smaller['prog_number'] == prog_number_smaller, 'modelat'].values[0]
    first_long = data_smaller.loc[data_smaller['prog_number'] == prog_number_smaller, 'modelon'].values[0]
    
    return first_lat, second_lat, first_long, second_long

def interpolate_coords(data_indiv, limit):
    """
    Function that takes the data of the individual and the latitude limit (45 N or 17 N) where I need to cut
    off the final segment of the migration track of the individual. 
    """
    # Get the coordinates to get the segment where I am going to find points every 100 metres
    first_lat, second_lat, first_long, second_long = get_coordinates_between_interpolation(data_indiv, limit)
    points = []
    geod = Geodesic.WGS84
    # Get the segment where I need to interpolate the longitude at the limit (45 N or 17 N parallel)
    line = geod.InverseLine(first_lat, first_long, second_lat, second_long)
    distance = 50 # I want to interpolate a point every 50 m
    tot_number_points = int(math.ceil(line.s13 / distance)) # math_ceil rounds a number up to its closest integer
    print('This is line.s13', line.s13)
    i = 0
    # s13 – distance of the segment in meters
    # I iterate to get the coordinates of a point every 50 m until I get one that has the
    for point in range(tot_number_points + 1):
        if point == 0:
            print("point 0")
        distance_within_segment = min(distance * point, line.s13)
        interpolated_points = line.Position(distance_within_segment, Geodesic.STANDARD | Geodesic.LONG_UNROLL)
        latitude = interpolated_points['lat2']
        if (np.isclose(latitude, limit)) & (i == 0): # only take the first coordinate with latitude similar
            # to the limit (45N or 17N)
            print('Interpolated coordinates\n',(interpolated_points['lat2'], interpolated_points['lon2']))
            points.append((interpolated_points['lat2'], interpolated_points['lon2']))
            i += 1
    return points[0] # in format list of tuple [(lat, long)], points[0] = (lat, long)

def create_df_with_points(points_df_5IK, points_df_5LK):
    points_df_5IK['ID'] = '5IK'
    points_df_5LK['ID'] = '5LK'
    #points_df_5IK['tot_distance'] = 
    #points_df_5LK['tot_distance'] = 


def main():
    if len(sys.argv) < 1:
        exit('python3.9 os/one_way_dist.py output_files/tracks_with_dist.csv output_files/tracks_rand_loc.csv output_files/random_loc_5LK.csv')
    pd.set_option('display.max_rows', None)
    warnings.filterwarnings("ignore")
    data = pd.read_csv(sys.argv[1])
    data['date_time'] = pd.to_datetime(data['date_time'])

    # Season
    season = 'spr'
    # Only consider juveniles with complete tracks 
    data = data[(data[f'compl_{season}_track'] == 1) & (data['season'] == season) & (data['Juv'] == 1)]
    # Number of equally distanced points
    n_samples = 22

    # For individual 5IK
    indiv = '5IK'
    data_indiv_1 = prepare_dataset(data, season, indiv)
    # Interpolate coordinates at 45 and 17 of latitude (=cut-offs)
    points_45 = interpolate_coords(data_indiv_1, 45) # points_45 = (lat, lon)
    points_17 = interpolate_coords(data_indiv_1, 17) # points_17 = (lat, lon)
    data_indiv_1 = change_lat_and_long(data_indiv_1, points_45[0], points_45[1], points_17[0], points_17[1])
    path_indiv_1 = from_df_to_list_of_tuples(data_indiv_1)
    points_indiv_1 = place_equally_distanced_points(path_indiv_1, n_samples)
    points_df_1 = plot_maps_migration(points_indiv_1, path_indiv_1)
    points_df_1.to_csv(sys.argv[2], index = False)

    # For individual 5LK
    indiv = '5LK'
    data_indiv_2 = prepare_dataset(data, season, indiv)
    # Interpolate coordinates at 45 and 17 of latitude (=cut-offs)
    points_45 = interpolate_coords(data_indiv_2, 45) # points_45 = (lat, lon)
    points_17 = interpolate_coords(data_indiv_2, 17) # points_17 = (lat, lon)
    data_indiv_2 = change_lat_and_long(data_indiv_2, points_45[0], points_45[1], points_17[0], points_17[1])
    path_indiv_2 = from_df_to_list_of_tuples(data_indiv_2)
    points_indiv_2 = place_equally_distanced_points(path_indiv_2, n_samples)
    points_df_2 = plot_maps_migration(points_indiv_2, path_indiv_2)
    points_df_2.to_csv(sys.argv[3], index = False)

    #print('SUM 5IK first and then 5LK\n',data_indiv_5IK['distance'].sum(), data_indiv_5LK['distance'].sum())

    # Calculate one-way distance for individual 5IK
    min_distance_list_1 = select_shortest_distance(points_indiv_1, points_indiv_2)
    print('This is min distance list of 5IK', min_distance_list_1)
    sum_distances_1 = sum(min_distance_list_1)
    print('sum distance list 5IK', sum_distances_1)
    # Remember to use migration track cut off at 45 N and 17 N for tot_distance!
    # Recalculate all the distances of the segments now that the original dataframe is cut off at 17 and 45 lat N
    data_indiv_1 = add_distance_in_dataframe(data_indiv_1)
    tot_sum_indiv_1 = data_indiv_1['distance'].sum()
    print('Total distance sum after recalculation',data_indiv_1['distance'].sum(), tot_sum_indiv_1)
    # Sum distances between track of individual A and track of individual B divided by total distance of track
    # of individual A
    sum_indiv_1 = sum_distances_1 / tot_sum_indiv_1

    # Calculate one-way distance for individual 5LK
    min_distance_list_2 = select_shortest_distance(points_indiv_2, points_indiv_1)
    print('This is min distance list of 5LK', min_distance_list_2)
    sum_distances_2 = sum(min_distance_list_2)
    print('sum distance list 5LK', sum_distances_2)
    # Recalculate all the distances of the segments now that the original dataframe is cut off at 17 and 45 lat N
    data_indiv_2 = add_distance_in_dataframe(data_indiv_2)
    tot_sum_indiv_2 = data_indiv_2['distance'].sum()
    print('Total distance sum after recalculation',data_indiv_2['distance'].sum(), tot_sum_indiv_2)
    # Sum distances between track of individual B and track of individual A divided by total distance of track
    # of individual B
    sum_indiv_2 = sum_distances_2 / tot_sum_indiv_2

    # Final one-way distance
    print('final one-way distance in km', (sum_indiv_1 + sum_indiv_2)/2)



if __name__ == "__main__":
    main()