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
import one_way_dist

# Fixing the progressive number column
def fix_prog_tot_col(data):
    prog_tot = 0
    list_prog = [ ]
    for idx, row in data.iterrows():
        prog_tot += 1
        list_prog.append(prog_tot)
    data['prog_number'] = list_prog
    return data

def calculate_distance(cA, cB):
    """
    :param cA : pair of point A (long, lat)
    :param cB : pair of point B (log, lat)
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

def calculate_avrg_migr_dep(data, season):
    """
    It takes the dataframe with all the locations with their corresponding time, it converts it to julian date and calculates
    the mean of the departure date of migration
    """
    data = data.loc[(data[f'compl_{season}_track'] == 1) & (data['season'] == season)&\
    (data[f'adults_{season}_compl_migr'] == 1) & (data['Juv'] == 0)]
    data['date_time_dt'] = data['date_time'].dt.date
    remove_nat_values = data.loc[~data.date_time_dt.isnull()]
    #print('remove nat values\n',remove_nat_values)
    start_migr = remove_nat_values.groupby('ID')['date_time_dt'].first().reset_index()
    #print('start_migr\n', start_migr)
    start_migr['julian_day_start_migr'] = start_migr['date_time_dt'].apply(lambda x: (x.timetuple().tm_yday) % 365)
    print(f'mean departure {season} migration\n', start_migr['julian_day_start_migr'].mean())

def get_adults_with_compl_migr(data, individuals):                                   
    list_ad_autumn = individuals.loc[(individuals['Juv'] == 0) & (individuals['compl_aut_track'] == 1)].groupby(['Year','Country','Pop'])['ID'].unique().reset_index()['ID']
    print('Number of adults with complete autumn migration\n', list_ad_autumn, 'in total', len(list_ad_autumn))
    list_ad_spring = individuals.loc[(individuals['Juv'] == 0) & (individuals['compl_spr_track'] == 1)].groupby(['Year','Country','Pop'])['ID'].unique().reset_index()['ID']
    data['adults_aut_compl_migr'] = np.where(data['ID'].isin(list_ad_autumn), 1, 0)
    data['adults_spr_compl_migr'] = np.where(data['ID'].isin(list_ad_spring), 1, 0)
    print('Number of adults with complete spring migration\n', list_ad_spring, 'in total', len(list_ad_spring))
    print(np.setdiff1d(list_ad_autumn[6], list_ad_spring[6]), 'how many individuals?', len(np.setdiff1d(list_ad_autumn[6], list_ad_spring[6]))) # 3TR on 5th row in spring but not aut, 5AY on 6th row in spring but not aut
    print('Group by year, pop and country autumn\n', individuals.loc[(individuals['Juv'] == 0) & (individuals['compl_aut_track'] == 1)].groupby(['Year','Country','Pop'])['ID'].unique().reset_index())
    print('Group by year, pop and country autumn\n', individuals.loc[(individuals['Juv'] == 0) & (individuals['compl_aut_track'] == 1)].groupby(['Year','Country','Pop'])['ID'].nunique().reset_index())
    print('Group by year, pop and country spring\n', individuals.loc[(individuals['Juv'] == 0) & (individuals['compl_spr_track'] == 1)].groupby(['Year','Country','Pop'])['ID'].unique().reset_index())
    print('Group by year, pop and country spring\n', individuals.loc[(individuals['Juv'] == 0) & (individuals['compl_spr_track'] == 1)].groupby(['Year','Country','Pop'])['ID'].nunique().reset_index())
    # 'repeated' == 1 column is useless, no idea what it means... some individuals that have both spring and autumn migration do not have the value 1 of the column 'repeated'
    pop_ch_2010 = list(list_ad_spring[0])
    pop_it_1_2010 = list(list_ad_spring[1])
    pop_it_2_2010 = list(list_ad_spring[2])
    pop_ch_2011 = list(list_ad_spring[3])
    pop_it_1_2011 = list(list_ad_autumn[4])
    pop_it_2_2011 = list(list_ad_spring[5])
    pop_it_2_2011 = [i for i in pop_it_2_2011 if i != '5AY'] # remove individual 5AY
    pop_it_1_2012 = list(list_ad_spring[6])
    data['pop_ch_2010_compl_migr'] = np.where(data['ID'].isin(pop_ch_2010), 1, 0)
    data['pop_it_1_2010_compl_migr'] = np.where(data['ID'].isin(pop_it_1_2010), 1, 0)
    data['pop_it_2_2010_compl_migr'] = np.where(data['ID'].isin(pop_it_2_2010), 1, 0)
    data['pop_ch_2011_compl_migr'] = np.where(data['ID'].isin(pop_ch_2011), 1, 0)
    data['pop_it_1_2011_compl_migr'] = np.where(data['ID'].isin(pop_it_1_2011), 1, 0)
    data['pop_it_2_2011_compl_migr'] = np.where(data['ID'].isin(pop_it_2_2011), 1, 0)
    data['pop_it_1_2012_compl_migr'] = np.where(data['ID'].isin(pop_it_1_2012), 1, 0)
    print('Juveniles!!!\n', individuals.loc[(individuals['Juv'] == 1)].groupby(['Year','Country','Pop'])['ID'].unique().reset_index())
    return data

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

def all_birds_map(data, save_fig, season):
    plt.figure(figsize=(30,10))
    ax = plt.axes(projection= ccrs.PlateCarree())
    ax = setting_up_the_map(ax)
    # Set the background of the map
    ax.set_extent((-20.0, 43.0, 55.0, -37.0), crs=ccrs.PlateCarree()) #(-20.0, 43.0, 55.0, -37.0)
    shpfilename = shpreader.natural_earth(resolution='110m',
                                      category='cultural',
                                      name='admin_0_countries')

    reader = shpreader.Reader(shpfilename)
    # Add names of some countries on map
    countries = reader.records()
    list_countries = ['Ukraine', 'Germany', 'Libya']
    for country in countries:
        for i in list_countries:
            if country.attributes['NAME'] == i:
                x = country.geometry.centroid.x       
                y = country.geometry.centroid.y
                ax.text(x, y, i, color='white', size=11, ha='center', va='center', transform=ccrs.PlateCarree())
    #data = data.loc[(data[f'compl_{season}_track'] == 1) & (data['season'] == season) & (data[f'adults_{season}_compl_migr'] == 1)]
    data = data.loc[data['pop_it_1_2012_compl_migr'] == 1]
    # & (data['Juv'] == 1)]
    # Drop NaN values in modelat and modelon columns
    #data = data.drop_duplicates(subset=['modelat', 'modelon'], keep='last')
    #& (data['stationary'] == False) & (data['typeofstopover'] == 'migration')]
    # Set up the tracks per individual that will be represented on the map
    bird_id = data['ID'].unique()
    print('total number of birds is', len(bird_id), bird_id)
    # I need to choose a proper set of colors
    ax.set_prop_cycle('color', plt.cm.gist_rainbow(np.linspace(0,1,len(bird_id))))
    for bird in bird_id:
        try:
            x = data.loc[(data['ID'] == bird), 'modelon']
            y = data.loc[(data['ID'] == bird), 'modelat']
            # ccrs.PlateCarree()
            ax.plot(x,y,'-', transform=ccrs.Geodetic(), linewidth = 1.5, label = bird) # to use dictionary for colors: , color = colors[bird]
            ax.plot(x,y,'.',transform=ccrs.Geodetic(), label = bird, c = 'black', zorder = 1)
        except ValueError: # raised when there is a NaN value maybe?
            pass
    plt.legend(fontsize='small', loc="upper left")
    plt.savefig('output_files/images/'+save_fig+'.png', dpi=500, format='jpg', bbox_inches="tight")
    plt.show()

def main():
    if len(sys.argv) < 1:
        exit('python3.9 os/start_mapping_dist.py input_files/tracks.csv input_files/individuals.xlsx output_files/random_loc_5LK.csv')
    pd.set_option('display.max_rows', None)
    warnings.filterwarnings("ignore")
    data = pd.read_csv(sys.argv[1], sep = ';', na_values = ['NA','a'], decimal=',')
    data['date_time'] = pd.to_datetime(data['date_time'])
    data['modelat'] = data['modelat'].replace('NA', np.nan, regex=True)
    data['modelon'] = data['modelon'].replace('NA', np.nan, regex=True)
    data['modelat'] = data['modelat'].astype(float)
    data['modelon'] = data['modelon'].astype(float)
    data = data.sort_values(by = ['ID','date_time'])
    individuals = pd.read_excel(sys.argv[2])
    data = get_adults_with_compl_migr(data, individuals)
    data = fix_prog_tot_col(data)
    data = data.dropna(subset = ['modelat','modelon'])
    #calculate_avrg_migr_dep(data, 'spr')
    d = {}
    for name, group in data.groupby(['ID','year']):
        d['group_' + str(name)] = group  # group is a dataframe containing information about ONLY ONE BIRD
        group = add_distance_in_dataframe(group)
    new_df = pd.DataFrame([])
    for key in d:
        new_df = new_df.append(d[key])
    season = 'aut'
    #new_df.to_csv(sys.argv[3],index = False)
    data_loc = pd.read_csv(sys.argv[3])
    #all_birds_map(data, f'{season}_test_diff_pop', season)


if __name__ == "__main__":
    main()