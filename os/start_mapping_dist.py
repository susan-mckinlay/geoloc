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
import scipy.stats as stats
import matplotlib.patches as mpatches

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
    data['date_time_dt'] = data['date_time'].dt.date
    remove_nat_values = data.loc[~data.date_time_dt.isnull()]
    #print('remove nat values\n',remove_nat_values)
    start_migr = remove_nat_values.groupby('ID')['date_time_dt'].first().reset_index()
    #print('start_migr\n', start_migr)
    start_migr['julian_day_start_migr'] = start_migr['date_time_dt'].apply(lambda x: (x.timetuple().tm_yday) % 365)
    print(f'mean departure {season} migration\n', start_migr['julian_day_start_migr'].mean(), start_migr['julian_day_start_migr'].std(), len(start_migr['julian_day_start_migr']))
    print(start_migr)
    return start_migr

def t_test_phenology(start_migr_adults, start_migr_juv):
    """
    It performs a two-sample or paired t-test between two variable sets (juveniles vs adults or first track vs second track)
    """
    # variables to test from individuals excel file: migration distance (consec_dist_aut/spr), migration duration (aut/spr_mig_dur),
    # migration straightness (straight_aut/spr), migration speed (speed_aut/spr), departure date
    # from S Sahara (j_win_dep), arrival date to breeding colony (j_BParr), departure date from breeding colony (j_BPdep)
    # arrival date to S Sahara (j_win_arr)
    #  OWD between spring and autumn migration of same individual 
    juv_dep = start_migr_juv['n_sp']     #start_migr_juv['julian_day_start_migr']
    adult_dep = start_migr_adults['n_sp']     #start_migr_adults['julian_day_start_migr']

    print(juv_dep)
    print(adult_dep)

    # Check normal distribution
    print(stats.shapiro(juv_dep))
    print(stats.shapiro(adult_dep))

    # Perform the t-test:
    # When we have measurements from the same people in both data sets (a within-subjects. design), we need to account for this, or the t test will again suggest an inflated (incorrect) value. 
    # We account for this by using a paired t test. 
    t_stat, p_value = stats.ttest_ind(juv_dep, adult_dep, equal_var= False) # stats.ttest_ind(juv_dep, adult_dep, equal_var = False)
    result = stats.ttest_ind(juv_dep, adult_dep, equal_var = False)
    # stats.ttest_ind(equal_var = False) is for non-paired t-tests -> This means that t tests assume there is no relationship between any particular measurement in each of the two data sets being compared. 
    # I use stats.ttest_ind() for t-tests between juveniles and adults
    # paired t-test stats.ttest_rel() -> for t-tests performed between first and second track of the same individual

    # Interpret the results:
    alpha = 0.05
    if p_value < alpha:
        print("Reject the null hypothesis; there is a significant difference between the departure dates of juvenile and adult barn swallows.", 'p-value:', p_value, 't-stat:', t_stat, 'df', result.df)
        print('This is the mean of the first track:', adult_dep.mean(), adult_dep.sem())
        print('This is the mean of the second track:', juv_dep.mean(), juv_dep.sem())        
    else:
        print("Fail to reject the null hypothesis; there is no significant difference between the spring migration distances of juvenile and adult barn swallows.", 'p-value:', p_value, 't-stat:', t_stat, 'df', result.df)
        print('This is the mean of the first track:', adult_dep.mean(), adult_dep.sem())
        print('This is the mean of the second track:', juv_dep.mean(), juv_dep.sem())        
    # Spring departure and arrival dates are signficantly different between juvenile and adult barn swallows

def get_adults_with_compl_migr(data, individuals):
    """
    I am creating columns in the original dataframe to define different adult populations. The year in the name of the
    columns refers to WHEN the individual was tagged. These columns only refer to adults with BOTH spring and autumn
    migration tracks.
    """                                   
    list_ad_autumn = individuals.loc[(individuals['Juv'] == 0) & (individuals['compl_aut_track'] == 1)].groupby(['Year','Country','Pop'])['ID'].unique().reset_index()['ID']
    #print('Number of adults with complete autumn migration\n', list_ad_autumn, 'in total', len(list_ad_autumn))
    list_ad_autumn_column = individuals.loc[(individuals['Juv'] == 0) & (individuals['compl_aut_track'] == 1), 'ID'].unique()
    list_ad_spring = individuals.loc[(individuals['Juv'] == 0) & (individuals['compl_spr_track'] == 1)].groupby(['Year','Country','Pop'])['ID'].unique().reset_index()['ID']
    list_ad_spring_column = individuals.loc[(individuals['Juv'] == 0) & (individuals['compl_spr_track'] == 1), 'ID'].unique()
    data['adults_aut_compl_migr'] = np.where(data['ID'].isin(list_ad_autumn_column), 1, 0)
    data['adults_spr_compl_migr'] = np.where(data['ID'].isin(list_ad_spring_column), 1, 0)
    # Adults with repeated tracks
    repeated_tracks = individuals.loc[individuals['repeated'] == 1, 'RING'].unique()
    #print('unique rings repeated tracks\n', repeated_tracks)
    #print(data['RING'].unique())
    data['repeated_tracks'] = np.where(data['RING'].isin(repeated_tracks), 1, 0)
    #print('Number of adults with complete spring migration\n', list_ad_spring, 'in total', len(list_ad_spring))
    #print(np.setdiff1d(list_ad_autumn[6], list_ad_spring[6]), 'how many individuals?', len(np.setdiff1d(list_ad_autumn[6], list_ad_spring[6]))) # 3TR on 5th row in spring but not aut, 5AY on 6th row in spring but not aut
    #print('Group by year, pop and country autumn\n', individuals.loc[(individuals['Juv'] == 0) & (individuals['compl_aut_track'] == 1)].groupby(['Year','Country','Pop'])['ID'].unique().reset_index())
    #print('Group by year, pop and country autumn\n', individuals.loc[(individuals['Juv'] == 0) & (individuals['compl_aut_track'] == 1)].groupby(['Year','Country','Pop'])['ID'].nunique().reset_index())
    #print('Group by year, pop and country spring\n', individuals.loc[(individuals['Juv'] == 0) & (individuals['compl_spr_track'] == 1)].groupby(['Year','Country','Pop'])['ID'].unique().reset_index())
    #print('Group by year, pop and country spring\n', individuals.loc[(individuals['Juv'] == 0) & (individuals['compl_spr_track'] == 1)].groupby(['Year','Country','Pop'])['ID'].nunique().reset_index())
    # Classify the individuals by population, year and season of migration track
    pop_ch_2010_spr = list(list_ad_spring[0])
    pop_ch_2010_aut = list(list_ad_autumn[0])
    pop_it_1_2010_spr = list(list_ad_spring[1])
    pop_it_1_2010_aut = list(list_ad_autumn[1])
    pop_it_2_2010_spr = list(list_ad_spring[2])
    pop_it_2_2010_aut = list(list_ad_autumn[2])
    pop_ch_2011_spr = list(list_ad_spring[3])
    pop_ch_2011_aut = list(list_ad_autumn[3])
    pop_it_1_2011_spr = list(list_ad_spring[4])
    pop_it_1_2011_aut = list(list_ad_autumn[4])
    pop_it_2_2011_spr = list(list_ad_spring[5])
    pop_it_2_2011_aut = list(list_ad_autumn[5])
    pop_it_1_2012_spr = list(list_ad_spring[6])
    pop_it_1_2012_aut = list(list_ad_autumn[6])
    data['pop_ch_2010_spr'] = np.where(data['ID'].isin(pop_ch_2010_spr), 1, 0)
    data['pop_ch_2010_aut'] = np.where(data['ID'].isin(pop_ch_2010_aut), 1, 0)
    data['pop_it_1_2010_spr'] = np.where(data['ID'].isin(pop_it_1_2010_spr), 1, 0)
    data['pop_it_1_2010_aut'] = np.where(data['ID'].isin(pop_it_1_2010_aut), 1, 0)
    data['pop_it_2_2010_spr'] = np.where(data['ID'].isin(pop_it_2_2010_spr), 1, 0)
    data['pop_it_2_2010_aut'] = np.where(data['ID'].isin(pop_it_2_2010_aut), 1, 0)
    data['pop_ch_2011_spr'] = np.where(data['ID'].isin(pop_ch_2011_spr), 1, 0)
    data['pop_ch_2011_aut'] = np.where(data['ID'].isin(pop_ch_2011_aut), 1, 0)
    data['pop_it_1_2011_spr'] = np.where(data['ID'].isin(pop_it_1_2011_spr), 1, 0)
    data['pop_it_1_2011_aut'] = np.where(data['ID'].isin(pop_it_1_2011_aut), 1, 0)
    data['pop_it_2_2011_spr'] = np.where(data['ID'].isin(pop_it_2_2011_spr), 1, 0)
    data['pop_it_2_2011_aut'] = np.where(data['ID'].isin(pop_it_2_2011_aut), 1, 0)
    data['pop_it_1_2012_spr'] = np.where(data['ID'].isin(pop_it_1_2012_spr), 1, 0)
    data['pop_it_1_2012_aut'] = np.where(data['ID'].isin(pop_it_1_2012_aut), 1, 0)

    #print('Juveniles!!!\n', individuals.loc[(individuals['Juv'] == 1)].groupby(['Year','Country','Pop'])['ID'].unique().reset_index())
    return data


def setting_up_the_map(ax):
    fname = os.path.join('/Users/susanmckinlay/Documents/python/woodcock/input_files/tif_files/', 'HYP_HR_SR_W.tif')
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
    # I am tring to highlight the parallel N 17
    ax.gridlines(draw_labels=False, xlocs=[], ylocs=[17], linestyle='--', color = 'black')
    grid_lines.xformatter = LONGITUDE_FORMATTER
    grid_lines.yformatter = LATITUDE_FORMATTER
    return ax

def all_birds_map(data, save_fig, season):
    plt.figure(figsize=(30,10))
    ax = plt.axes(projection= ccrs.PlateCarree())
    ax = setting_up_the_map(ax)
    # Set the background of the map; extent: 40.0 for spring migration
    ax.set_extent((-20.0, 37.0, 55.0, -33.0), crs=ccrs.PlateCarree()) #(-20.0, 43.0, 55.0, -37.0)
    shpfilename = shpreader.natural_earth(resolution='110m',
                                      category='cultural',
                                      name='admin_0_countries')

    reader = shpreader.Reader(shpfilename)
    # Add names of some countries on map
    countries = reader.records()
    list_countries = ['Ukraine', 'Germany', 'Egypt']
    for country in countries:
        for i in list_countries:
            if country.attributes['NAME'] == i:
                x = country.geometry.centroid.x       
                y = country.geometry.centroid.y
                ax.text(x, y, i, color='white', size=11, ha='center', va='center', transform=ccrs.PlateCarree())
    #data = data.loc[(data[f'compl_{season}_track'] == 1) & (data['season'] == season) & (data[f'adults_{season}_compl_migr'] == 1)]
    #data = data.loc[(data['pop_ch_2011_{season}'] == 1)]
    #data = data.loc[data['season'] == season] # not for wintering
    # For maps of individuals with repeated tracks
    data = data.loc[(data['repeated'] == 1)] # column 'repeated' for wintering map, 'repeated_tracks' for all the others
    # These individuals don't have wintering that is long enough:
    data = data.loc[(data['RING'] != '4A99317')& (data['RING'] != '4A99647')]
    # These individuals only have double autumn migration:
    #data = data.loc[(data['RING'] != '4A99312') & (data['RING'] != '4A99317')& (data['RING'] != '4A99647')]
    #data_first_year = data.loc[data['rep_year_first'] == 1]
    #data_second_year = data.loc[data['rep_year_second'] == 1]
    # Set up the tracks per individual that will be represented on the map
    bird_id = data['RING'].unique() # column 'ID' for all maps, column 'RING' for wintering map
    print('total number of birds is', len(bird_id), bird_id)
    # Add a row with values only for specific columns
    data.loc[len(data)] = {'modelat': -29.420086603577, 'modelon': 30.5620939457841, 'lcllon': 28.621862, 
                           'ucllon': 30.872964, 'lcllat': -29.446047, 'ucllat': -28.525724, 'RING': 'B348329'}
    # Prepare error bars
    xerr_lower = data['lcllat']  # Lower bound errors for longitude (for each bird)
    xerr_upper = data['ucllat']  # Upper bound errors for longitude (for each bird)
    yerr_lower = data['lcllon']  # Lower bound errors for latitude (for each bird)
    yerr_upper = data['ucllon'] # Upper bound errors for longitude (for each bird)
    # Color dictionary for individuals with repated tracks
    color_dict = {'5GN':'aquamarine', '1RH':'aquamarine', '3SP':'yellow','1UP':'yellow', '5GD':'fuchsia','2EU':'fuchsia','3SS':'blue','5SU':'blue',
    '5HC':'red','5PD':'red','1ST':'orange','3ST':'orange','3RD':'lawngreen','1YD':'lawngreen','1RZ':'darkviolet','3RM':'darkviolet'}
    color_dict_winter = {'4A99312':'aquamarine', '4A99312':'aquamarine', '4A99317':'yellow','4A99317':'yellow', '4A99647':'fuchsia','4A99647':'fuchsia','5A27815':'blue','5A27815':'blue',
    '6A49176':'red','6A49176':'red','B348042':'orange','B348042':'orange','B348329':'lawngreen','B348329':'lawngreen','B348767':'darkviolet','B348767':'darkviolet'}
    #print(data_first_year['RING'].unique())
    # I need to choose a proper set of colors
    ax.set_prop_cycle('color', plt.cm.gist_rainbow(np.linspace(0,1,len(bird_id))))
    # Overlay gray diagonal stripes using a patch
    circle = mpatches.Circle((30.5620939457841, -29.420086603577), 1, transform=ccrs.PlateCarree(),
                         edgecolor='dimgray', facecolor='none', hatch='/////', zorder = 5)
    # Loop through each bird ID and its locations
    idx = 0  # This index will be used to loop through the error bounds
    for bird in bird_id:
        try:
            # Get longitude (x) and latitude (y) for the current bird (assuming two locations per bird)
            bird_data = data.loc[data['RING'] == bird]
            x = bird_data['modelon'].values  # Longitude values for the bird
            y = bird_data['modelat'].values  # Latitude values for the bird
            # Error bars for wintering
            # Get lower and upper bounds for longitude and latitude (as coordinates)
            lclon = bird_data['lcllon'].values
            ullon = bird_data['ucllon'].values
            lclat = bird_data['lcllat'].values
            ullat = bird_data['ucllat'].values
            # Convert absolute coordinates to relative error bounds
            xerr_lower = np.abs(x - lclon)
            xerr_upper = np.abs(ullon - x)
            yerr_lower = np.abs(y - lclat)
            yerr_upper = np.abs(ullat - y)
            print(f'Number {idx}: x {x} - lclon {lclon} = {xerr_lower} and ullon {ullon} - x {x} = {xerr_upper}. y {y} - lclat {lclat} = {yerr_lower}. ullat {ullat} - y {y} = {yerr_upper}\n')
            # Combine lower and upper bounds into correct format for errorbar
            xerr = [xerr_lower, xerr_upper]  # Asymmetric x-error
            yerr = [yerr_lower, yerr_upper]  # Asymmetric y-error
            print('len(x) is:', len(x),'number', idx, 'this is the bird:', bird, 'and these are the errors:', xerr, yerr, '\n')
            # Add asymmetric error bars for each point
            # Add asymmetric error bars for each point
            ax.errorbar(
            x, y,
            xerr=xerr, yerr=yerr,
            fmt='o', color = color_dict_winter[bird], ecolor='black', markersize = 12,
            elinewidth=1, capsize=0, transform=ccrs.PlateCarree()
            )
            # Add gray patch for furthest point of individual B348329
            ax.add_patch(circle)
            # Increment idx **after** each location
            idx += len(x)
        except KeyError as e: # raised when there is a NaN value maybe?
            print(f"Data for {bird} not found. KeyError: {e}")
            pass
    #plt.legend(fontsize='small', loc="upper left")
    plt.savefig('output_files/images/'+save_fig+'.png', dpi=500, format='jpg', bbox_inches="tight")
    plt.show()

def calculate_stopover_number(data_adults):
    # Count number of stopovers (actually stationary periods during non-breeding)
    # Replace empty spaces with False
    data_adults['stationary'] = data_adults['stationary'].fillna(False)
    # Group by 'ID' and shift the 'Stationary' column to check previous value
    data_adults['prev_stationary'] = data_adults.groupby('RING')['stationary'].shift(1)
    print(data_adults['stationary'].value_counts())
    # Find where the previous value is False and current value is True
    transition = (data_adults['prev_stationary'] == False) & (data_adults['stationary'] == True)
    print(data_adults[transition]['typeofstopover'].value_counts())
    print(data_adults['typeofstopover'].nunique())
    # Count occurrences of True after False per RING
    ad_n_stopovers = data_adults[transition].groupby('RING').size().reset_index()
    print('sample size ring in df:', data_adults['RING'].nunique())
    print('mean of number stopovers adults\n',ad_n_stopovers)
    return ad_n_stopovers


def main():
    if len(sys.argv) < 1:
        exit('python3.13 os/start_mapping_dist.py output_files/tracks_with_dist.csv input_files/individuals.xlsx output_files/random_loc_5LK.csv')
    pd.set_option('display.max_rows', None)
    warnings.filterwarnings("ignore")
    # This is for original track.csv file
    #data = pd.read_csv(sys.argv[1], sep = ';', na_values = ['NA','a'], decimal=',')
    data = pd.read_csv(sys.argv[1])
    data['date_time'] = pd.to_datetime(data['date_time'])
    data['modelat'] = data['modelat'].replace('NA', np.nan, regex=True)
    data['modelon'] = data['modelon'].replace('NA', np.nan, regex=True)
    data['modelat'] = data['modelat'].astype(float)
    data['modelon'] = data['modelon'].astype(float)
    data = data.sort_values(by = ['ID','date_time'])
    individuals = pd.read_excel(sys.argv[2])
    #print('Average number of stationary periods adults')
    #print(individuals.loc[(individuals['Juv'] == 0) & (individuals['Country'] == 'CH') & (individuals['Year'] == 2011), 'consec_dist_spr'].describe())
    data = get_adults_with_compl_migr(data, individuals)
    data = fix_prog_tot_col(data)
    data = data.dropna(subset = ['modelat','modelon'])
    #calculate_avrg_migr_dep(data, 'spr')
    d = {}
    for name, group in data.groupby(['ID','year']):
        d['group_' + str(name)] = group  # group is a dataframe containing information about ONLY ONE BIRD
        group = add_distance_in_dataframe(group)
    # Use pd.concat to efficiently combine all the groups
    new_df = pd.concat(d.values(), ignore_index=True)
    season = 'spr'
    data_adults = data.loc[(data[f'compl_{season}_track'] == 1) & (data['season'] == season) & (data['pop_ch_2011_'+season] == 1)] #& (data['repeated_tracks'] == 1)
    start_migr_adults = calculate_avrg_migr_dep(data_adults, season)
    start_migr_adults['group'] = 'adults'
    #list_aut_ch = data.loc[data['pop_ch_2011_spr'] == 1, 'RING'].unique()
    #print('Adult of ch_2011 that only has autumn migration and no spring migration\n', set(list(data.loc[data['pop_ch_2011_'+season] == 1, 'RING'].unique())).difference(list_aut_ch))
    # I need to ifnd a way to exclude the individual with repeated tracks pop_it_2011_2012_qut
    data_juv = data.loc[(data[f'compl_{season}_track'] == 1) & (data['season'] == season) & (data['Juv'] == 1)] # &(data['pop_ch_2011'+season]
    start_migr_juv = calculate_avrg_migr_dep(data_juv, season)
    start_migr_juv['group'] = 'juveniles'
    merged_df = start_migr_adults.merge(start_migr_juv, how ='outer')
    #individuals_dep_dates = individuals.loc[(individuals['Year'] == 2011) & (individuals['Country'] == 'CH') & (individuals['ID'] != '3RR')][['ID', 'Juv', 'j_win_dep']]
    merged_df.to_csv(sys.argv[3], index = False)
    # Count number of stopovers for adults and juveniles 
    print('sample size adults:', data_adults['RING'].nunique())
    print('sample size juv:', data_juv['RING'].nunique())
    ad_number_stop = calculate_stopover_number(data_adults)
    juv_number_stop = calculate_stopover_number(data_juv)

    # t-tests between first and second track of individuals with repeated tracks
    individuals_repeated_first_y_aut = individuals.loc[(individuals['repeated'] == 1) & (individuals['rep_year'] == 1)]
    individuals_repeated_second_y_aut = individuals.loc[(individuals['repeated'] == 1) & (individuals['rep_year'] == 2)]
    individuals_repeated_first_y_spr = individuals.loc[(individuals['repeated'] == 1) & (individuals['rep_year'] == 1) & (individuals['RING'] != '4A99312') & 
    (individuals['RING'] != '4A99317') & (individuals['RING'] != '4A99647')]
    individuals_repeated_second_y_spr = individuals.loc[(individuals['repeated'] == 1) & (individuals['rep_year'] == 2) & (individuals['RING'] != '4A99312') &
    (individuals['RING'] != '4A99317') & (individuals['RING'] != '4A99647')]
    #t_test_phenology(individuals_repeated_first_y_aut, individuals_repeated_second_y_aut)

    # t-test between adults and juveniles
    # Individual with just autumn migration and no spring migration is individual with ring B366688
    # for all migration parameters
    individuals_adults_spr = individuals.loc[(individuals['Juv'] == 0) & (individuals['Country'] == 'CH') & (individuals['Year'] == 2011) & (individuals['RING'] != 'B366688')]
    individuals_adults_aut = individuals.loc[(individuals['Juv'] == 0) & (individuals['Country'] == 'CH') & (individuals['Year'] == 2011)]
    individuals_juveniles = individuals.loc[(individuals['Juv'] == 1)]
    #t_test_phenology(individuals_adults_spr, individuals_juveniles)
    # for departure/arrival dates
    adults = start_migr_juv.loc[start_migr_juv['group'] == 'adult']
    juveniles = start_migr_juv.loc[start_migr_juv['group'] == 'juvenile']
    #t_test_phenology(start_migr_adults, start_migr_juv)

    first_year = individuals.loc[individuals['rep_year'] == 1, 'ID'].unique()
    data['rep_year_first'] = np.where(data['ID'].isin(first_year), 1, 0)
    second_year = individuals.loc[individuals['rep_year'] == 2, 'ID'].unique()
    data['rep_year_second'] = np.where(data['ID'].isin(second_year), 1, 0)
   
    all_birds_map(individuals, f'{season}_map_test', season)


if __name__ == "__main__":
    main()