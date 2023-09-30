import networkx as nx
from statistics import median
import pandas as pd
import geopandas as gpd
import shapely as sh
import numpy as np
from tqdm.auto import tqdm, trange
import sys
import random as rd
from scipy import spatial 
import os
import matplotlib.pyplot as plt
import folium

## Functions written for the Jupyter Notebook 

# Aggregation helper functions for simplification of network in corner cases, where lines with different attributes get aggregated.
    
def agg_einbahn(series):
    if '0' in series:
        res = '0'
    elif 'FT' in series and 'TF' in series:
        res = 'Check'
    elif 'TF' in series:
        res = 'TF'
    else:
        res = 'FT'
    return res

def agg_velostreif(series):
    if 'FT' in series and 'TF' in series:
        res = 'Check'
    elif '0' in series:
        res = '0'
    elif 'FT' in series:
        res = 'FT'
    elif 'TF' in series:
        res = 'TF'
    # special case, only once in network and it should be FT
    elif '1' in series:
        res = 'FT'
    else:
        res = '1'
    return res

def agg_objektart(series):
    if '10m Strasse' in series:
        res = '10m Strasse'
    elif '8m Strasse' in series:
        res = '8m Strasse'
    elif '6m Strasse' in series:
        res = '6m Strasse'
    elif '4m Strasse' in series:
        res = '4m Strasse'
    elif '3m Strasse' in series:
        res = '3m Strasse'
    elif '2m Weg' in series:
        res = '2m Weg'
    elif '1m Weg' in series:
        res = '1m Weg'
    else:
        res = 'Verbindung'
    return res

def agg_befahrbark(series):
    if 'Wahr' in series:
        res = 'Wahr'
    elif 'Falsch' in series:
        res = 'Falsch'
    else:
        res = 'k_W'
    return res

def agg_verkehrsbe(series):
    if 'Allgemeine Verkehrsbeschraenkung' in series:
        res = 'Allgemeine Verkehrsbeschraenkung'
    elif 'Zeitlich geregelt' in series:
        res = 'Zeitlich geregelt'
    elif 'Keine' in series:
        res = 'Keine'
    else:
        res = 'k_W'
    return res

def agg_one_way(series):
    if '1 FT' in series and '1 TF' in series:
        res = 'Check'
    elif '1 FT' in series:
        res = '1 FT'
    elif '1 TF' in series:
        res = '1 FT'
    else:
        res = '0'
    return res

def agg_velonetz(series):
    if 'Basisnetz' in series and 'Hauptnetz' in series:
        res = 'Ambiguous'
    elif 'Basisnetz' in series and 'Vorzugsrou' in series:
        res = 'Ambiguous'
    elif 'Hauptnetz' in series and 'Vorzugsrou' in series:
        res = 'Ambiguous'
    elif 'Basisnetz' in series:
        res = 'Basisnetz'
    elif 'Hauptnetz' in series:
        res = 'Hauptnetz'
    elif 'Vorzugsrou' in series:
        res = 'Vorzugsrou'
    else:
        res = '0'
    return res

# This function calculates the median slope of all slopes between the interpolated points per road segment. 
def get_median_slope(interpolated_points, point_spacing):

    # Extracting the z-values per segment
    z_dict = {}
    n_points = len(interpolated_points)
    for i,j in interpolated_points.iterrows():
        if i < n_points -1 and interpolated_points['distance'][i+1] > j[-5]:
            if j[0] in z_dict:
                z_dict[j[0]].append(j[-3])
            else:
                z_dict[j[0]] = [j[-3]]

    # Calculating the slope between the points
    slope_dict = {}
    for key,value in z_dict.items():
        for i,j in enumerate(value):
            if i < len(value)-1:
                # Slope in percent in direction of digitalization
                slope = ((value[i+1]-j)/3)*100 
                if key in slope_dict:
                    slope_dict[key].append(slope)
                else:
                    slope_dict[key] = [slope]

    # Calculating the median slope per segment
    filtered_dict = {}
    for k, v in slope_dict.items():
        filtered_dict[k] = median(v)

    return filtered_dict

# This function matches two road layers based on proximity. The bicycle level of traffic stress network is based on the weakest link principle, therefore this matching is relatively simple and just a little more advanced than centroid matching.
def road_match(l1, l2, buffer):

    # Matching based on only the buffered centroids of one of the layers works well for 90+% of the roads but fails to match perpendicular roads near the centroid. For this this function not only uses centroids but three points along the geometry that can be chosen if wanted.
    inter_list = [0.3, 0.5, 0.7]
    first_point = l1.interpolate(inter_list[0], normalized = True)
    second_point = l1.interpolate(inter_list[1], normalized = True)
    third_point = l1.interpolate(inter_list[2], normalized = True)

    # Combining the three points
    combined_geoms = gpd.GeoDataFrame(geometry = first_point)
    combined_geoms['geom2'] = second_point
    combined_geoms['geom3'] = third_point
    combined_geoms['multi'] = [sh.MultiPoint([x, y, z]) for x, y, z in zip(combined_geoms.geometry, combined_geoms.geom2, combined_geoms.geom3)]
    combined_geoms = combined_geoms.set_geometry('multi').drop(['geometry', 'geom2', 'geom3'], axis=1)
    
    # Buffers of the three points
    buffered = combined_geoms.geometry.buffer(buffer)
    buffered = gpd.GeoDataFrame(geometry = gpd.GeoSeries(buffered), crs = '2056')
    buffered.reset_index(inplace = True)
    buffered.rename(columns = {'index': 'ind'}, inplace = True)
    
    # Calculate the intersections of the three buffered points and the geometries of the second layer, then extracting only the largest overlaps.
    matched_points = gpd.overlay(buffered, l2, how = "intersection", keep_geom_type=False)
    matched_points['length'] = matched_points.geometry.length
    gb = matched_points.groupby('ind')[['length']].max()
    l1 = l1.merge(gb, left_on = 'index', right_on = 'ind', how = 'left')
    matched_points.drop(columns = ['geometry'], inplace = True)

    # Merge them back on to the first road data
    l1 = l1.merge(matched_points, on = 'length', how = 'left')
    l1.drop_duplicates(subset = ['index_x'], inplace = True)
    l1.rename(columns = {'index_x': 'index'}, inplace = True)
    l1.set_index('index', inplace = True)

    # Short output to check if buffer was to small or large
    unmatched_count = 0
    for i in l1['length']:
        if np.isnan(i):
            unmatched_count +=1
    if unmatched_count == 0:
        print("All edges have been matched")
    else:
        print(f'{l1.shape[0]-unmatched_count} out of {l1.shape[0]} edges could be matched with a buffer size of {buffer} meters ({unmatched_count} unmatched edges)')
    
    # Drop unnessecary columns
    l1.drop(columns = ['length', 'index_y', 'ind'], inplace = True)

    return l1

# This functions indicates and attributes the road segments wherever parking spots are placed on the road.
def parking_match(roads, parkings):
    
    # Buffering the roads
    lines_buffered = roads.geometry.buffer(5, cap_style = 2)
    lines_buffered = gpd.GeoDataFrame(geometry = gpd.GeoSeries(lines_buffered), crs = '2056')
    lines_buffered.reset_index(inplace = True)

    # Joining the amount of parkings where they lie in the buffer of the roads
    n_matched_points = lines_buffered.sjoin(parkings).groupby('index').count()
    n_matched_points.drop(columns = ['geometry'], inplace = True)
    n_matched_points.reset_index(inplace = True)
    n_matched_points.rename(columns = {'index_right': 'n_parkings'},inplace = True)
    merged = roads.merge(n_matched_points, left_on = 'index', right_on = 'index', how = 'left')
    
    return(merged)

# This is a function that specifically cleans up the problems of the this dataset. The decisions are all made based on the weakest link principle and are described in the thesis at ???
def clean_variables(matched_dataset):
    
    # Homogenizing Null/NaN values
    matched_dataset = matched_dataset.astype(object).replace(np.nan, None)
    copy_l = matched_dataset.copy()

    # Replacing None values
    copy_l['velostreif'] = copy_l['velostreif'].replace(['BOTH', None], ['1', '0'])
    copy_l['einbahn'] = copy_l['einbahn'].replace([None], ['0'])
    copy_l['vmax'] = copy_l['vmax'].replace([None], ['0'])
    copy_l['OBJEKTART'] = copy_l['OBJEKTART'].replace([None], ['not_matched'])
    copy_l['n_parkings'] = copy_l['n_parkings'].replace([None], ['0'])
    copy_l['DWV_ANZ_Q'] = copy_l['DWV_ANZ_Q'].replace([None], ['0'])
    copy_l['median_slope'] = copy_l['median_slope'].replace([None], ['0'])
    copy_l['velonetz'] = copy_l['velonetz'].replace([None], ['0'])

    # Changing data types
    copy_l['DWV_ANZ_Q'] = pd.to_numeric(copy_l['DWV_ANZ_Q'])
    copy_l['n_parkings'] = pd.to_numeric(copy_l['n_parkings'])
    copy_l['vmax'] = pd.to_numeric(copy_l['vmax'])

    # Losing unnessecary columns
    copy_l = copy_l.drop(columns=['mm_len', 'node_start', 'node_end'])
    
    print('The ambiguous discrete variable values were cleaned up.')
    if copy_l.isnull().values.any() == False:
        print('There are no more Null/None values in the dataframe.')
    else:
        print('There are still Null/None values in the dataframe, Check again')

    return copy_l

# This is a function that transforms the until now onedirectional road network into a bidirectional one. A variable of the input geodataframe defines if the road is a one-way road or can be accessed from both sides. Geometries and other directional variables are reversed when needed. 
def transform_bidirectional(road_network, dir_variable):
    
    # To see how many new edges should be created in the end
    print(road_network.groupby([dir_variable])[dir_variable].count())

    # Some helper variables
    road_network = road_network.copy()
    road_network['rev'] = 0
    cols = [i for i in road_network.columns]
    add_list = []

    # Progress bar
    with tqdm(total=road_network.shape[0], desc = 'Checking/Indexing directions') as pbar: 
        # Loop through all road segments
        for index, row in road_network.iterrows():
            pbar.update(1)

            # For FT, everything stays the same for all other variables
            if row[dir_variable] == 'FT':
                add_list.append(row)
            # Here every variable that has a direction needs to be reversed, apart from the slope they get a new attribute for now
            elif row[dir_variable]== 'TF':
                row['rev'] = 1
                row['median_slope'] = row['median_slope'] * -1
                add_list.append(row)
            # And here the segment needs to be doubled, once in the geometries direction and once reversed
            elif row[dir_variable]== '0':
                row[dir_variable] = 'FT'
                add_list.append(row)
                rowb = row.copy()
                rowb[dir_variable] = 'TF'
                rowb['rev'] = 1
                rowb['median_slope'] = rowb['median_slope'] * -1
                add_list.append(rowb)

    # Some indexing cleaning up
    indexed_lines = gpd.GeoDataFrame(add_list, columns= cols, geometry='geometry', crs = '2056')
    if 'index' not in indexed_lines:
        indexed_lines.reset_index(inplace = True)
    indexed_lines.rename(columns = {'index': 'old_ind'},inplace = True)
    new_lines_1 = indexed_lines[indexed_lines.rev == 0]
    new_lines_2 = indexed_lines[indexed_lines.rev == 1]
    
    # For all other directional variables of the new road segments the directions are reversed
    new_lines_2 = new_lines_2.replace({'FT': 'II'}, regex=True)
    new_lines_2 = new_lines_2.replace({'TF': 'FT'}, regex=True)
    new_lines_2 = new_lines_2.replace({'II': 'TF'}, regex=True)
    
    # Reversing the geometries of those roads
    with tqdm(total=new_lines_2.shape[0], desc = 'Reversing geometries & directional variables') as pbar:
        for i, r in new_lines_2.iterrows():
            pbar.update(1)
            new_lines_2.loc[new_lines_2.index == i, 'geometry'] = r['geometry'].reverse()
    
    new_lines_comb = pd.concat([new_lines_1, new_lines_2]).sort_index()

    # For checking if the right amount of road segments were created
    print(new_lines_comb.groupby([dir_variable])[dir_variable].count())

    return new_lines_comb

# This function is the main classification function and classifies all road segments according to the classification table that is given as an input dictionary. Its output is the road network with all road segments classified.
def lts_classification(road_network, classification_tables, ADT_significance = 'high', slope_inclusion = False, planned_network = False, planned_network_classification = None):

    if slope_inclusion not in [False, True]:
        raise ValueError("Slope inclusion must either be True or False")
    if ADT_significance not in ['high', 'low']:
        raise ValueError("ADT significance must either be high or low")
    if planned_network not in [False, True]:
        raise ValueError("Planned network must either be True or False")
    if 'index' not in road_network:
        road_network.reset_index(inplace = True)

    roads = road_network.copy()
    index_list = [i for i in roads['index']]
    lts_dict = {}
    occurence_dict = dict.fromkeys(classification_tables, 0)
    occurence_dict['bike_lane'] = 0

    with tqdm(total=roads.shape[0], desc = 'Classifying road segments') as pbar: 
        for index, row in roads.iterrows():
            pbar.update(1)
            # Here the wanted result of the planned network can be defined, e.g. all 'Vorzugsrouten' should be LTS 1
            if planned_network is True:
                if row['velonetz'] == 'Vorzugsrou':
                    if planned_network_classification['Vorzugsrouten'] is not None:
                        if planned_network_classification['Vorzugsrouten'] == 1:
                            lts_dict[index] = planned_network_classification['Vorzugsrouten']
                            continue
                elif row['velonetz'] == 'Basisnetz':
                    if planned_network_classification['Basisnetz'] is not None:
                        if planned_network_classification['Basisnetz'] == 1:
                            lts_dict[index] = planned_network_classification['Basisnetz']
                            continue
                elif row['velonetz'] == 'Hauptnetz':
                    if planned_network_classification['Hauptnetz'] is not None:
                        if planned_network_classification['Hauptnetz'] == 1:
                            lts_dict[index] = planned_network_classification['Hauptnetz']
                            continue

            # Firstly all segments that are LTS 1 no matter the other variables
            # Seperated Bikepaths -> LTS 1
            if row['veloweg'] == 1:    
                lts_dict[index] = 1
                occurence_dict['bike_lane'] += 1
            # Some road types -> LTS 1
            elif row['OBJEKTART'] in ['Platz', '1m Weg', '2m Weg', 'not_matched']:
                lts_dict[index] = 1
            # Begegnungszonen -> LTS 1
            elif row['vmax'] == 20.0:
                lts_dict[index] = classification_tables['A 1']
                if row['velostreif'] == '1' or row['velostreif'] == 'FT':
                    occurence_dict['A 1'] += 1
                else:
                    occurence_dict['B 1'] += 1
            # Bike lanes in mixed Traffic
            elif row['velostreif'] == '1' or row['velostreif'] == 'FT':
                if row['vmax'] <= 30.0:
                    if row['OBJEKTART'] == '3m Strasse':
                        if row['one_way'] == '1 FT':
                            lts_dict[index] = classification_tables['A 2.3']
                            occurence_dict['A 2.3'] += 1
                        else:
                            if row['n_parkings'] > 0:
                                lts_dict[index] = classification_tables['A 2.2']
                                occurence_dict['A 2.2'] += 1
                            else:
                                lts_dict[index] = classification_tables['A 2.1'] 
                                occurence_dict['A 2.1'] += 1
                        
                    elif row['OBJEKTART'] == '4m Strasse':
                        if row['one_way'] == '1 FT':
                            lts_dict[index] = classification_tables['A 3.3']
                            occurence_dict['A 3.3'] += 1
                        else:
                            if row['n_parkings'] > 0:
                                lts_dict[index] = classification_tables['A 3.2']
                                occurence_dict['A 3.2'] += 1
                            else:
                                lts_dict[index] = classification_tables['A 3.1'] 
                                occurence_dict['A 3.1'] += 1
                    elif row['OBJEKTART'] in ['6m Strasse', '8m Strasse', '10m Strasse']:
                        if row['one_way'] == '1 FT':
                            lts_dict[index] = classification_tables['A 4.3']
                            occurence_dict['A 4.3'] += 1
                        else:
                            if row['n_parkings'] > 0:
                                lts_dict[index] = classification_tables['A 4.2']
                                occurence_dict['A 4.2'] += 1
                            else:
                                lts_dict[index] = classification_tables['A 4.1']
                                occurence_dict['A 4.1'] += 1
                        lts_dict[index] = 2
                elif row['vmax'] > 30.0:
                    if row['OBJEKTART'] in ['3m Strasse', '4m Strasse']:
                        if row['one_way'] == '1 FT':
                            lts_dict[index] = classification_tables['A 5.3']
                            occurence_dict['A 5.3'] += 1
                        else:
                            if row['n_parkings'] > 0:
                                lts_dict[index] = classification_tables['A 5.2']
                                occurence_dict['A 5.2'] += 1
                            else:
                                lts_dict[index] = classification_tables['A 5.1']
                                occurence_dict['A 5.1'] += 1
                    elif row['OBJEKTART'] in ['6m Strasse']:
                        if row['one_way'] == '1 FT':
                            lts_dict[index] = classification_tables['A 6.3']
                            occurence_dict['A 6.3'] += 1
                        else:
                            if row['n_parkings'] > 0:
                                lts_dict[index] = classification_tables['A 6.2']
                                occurence_dict['A 6.2'] += 1
                            else:
                                lts_dict[index] = classification_tables['A 6.1']
                                occurence_dict['A 6.1'] += 1
                    elif row['OBJEKTART'] in ['8m Strasse']:
                        if row['one_way'] == '1 FT':
                            lts_dict[index] = classification_tables['A 7.3']
                            occurence_dict['A 7.3'] += 1
                        else:
                            if row['n_parkings'] > 0:
                                lts_dict[index] = classification_tables['A 7.2']
                                occurence_dict['A 7.2'] += 1
                            else:
                                lts_dict[index] = classification_tables['A 7.1']
                                occurence_dict['A 7.1'] += 1
                    elif row['OBJEKTART'] in ['10m Strasse']:
                        if row['one_way'] == '1 FT':
                            lts_dict[index] = classification_tables['A 8.3']
                            occurence_dict['A 8.3'] += 1
                        else:
                            if row['n_parkings'] > 0:
                                lts_dict[index] = classification_tables['A 8.2']
                                occurence_dict['A 8.2'] += 1
                            else:
                                lts_dict[index] = classification_tables['A 8.1']
                                occurence_dict['A 8.1'] += 1
            
            elif row['vmax'] <= 30.0:
                if row['OBJEKTART'] in ['3m Strasse']:
                        if row['one_way'] == '1 FT':
                            lts_dict[index] = classification_tables['B 2.7']
                            occurence_dict['B 2.7'] += 1
                        else:
                            if row['n_parkings'] > 0:
                                if row['DWV_ANZ_Q'] >= 3000:
                                    lts_dict[index] = classification_tables['B 2.6']
                                    occurence_dict['B 2.6'] += 1
                                elif row['DWV_ANZ_Q'] > 1000:
                                    lts_dict[index] = classification_tables['B 2.5']
                                    occurence_dict['B 2.5'] += 1
                                elif ADT_significance == 'high' and row['DWV_ANZ_Q'] <= 1000:
                                    lts_dict[index] = classification_tables['B 2.4'][0]
                                    occurence_dict['B 2.4'] += 1
                                elif ADT_significance == 'low' and row['DWV_ANZ_Q'] <= 1000:
                                    lts_dict[index] = classification_tables['B 2.4'][1]
                                    occurence_dict['B 2.4'] += 1
                            else:
                                if row['DWV_ANZ_Q'] >= 3000:
                                    lts_dict[index] = classification_tables['B 2.3']
                                    occurence_dict['B 2.3'] += 1
                                elif row['DWV_ANZ_Q'] > 1000:
                                    lts_dict[index] = classification_tables['B 2.2']
                                    occurence_dict['B 2.2'] += 1
                                elif ADT_significance == 'high' and row['DWV_ANZ_Q'] <= 1000:
                                    lts_dict[index] = classification_tables['B 2.1'][0]
                                    occurence_dict['B 2.1'] += 1
                                elif ADT_significance == 'low' and row['DWV_ANZ_Q'] <= 1000:
                                    lts_dict[index] = classification_tables['B 2.1'][1]
                                    occurence_dict['B 2.1'] += 1
                elif row['OBJEKTART'] in ['4m Strasse']:
                        if row['one_way'] == '1 FT':
                            lts_dict[index] = classification_tables['B 3.7']
                            occurence_dict['B 3.7'] += 1
                        else:
                            if row['n_parkings'] > 0:
                                if row['DWV_ANZ_Q'] >= 3000:
                                    lts_dict[index] = classification_tables['B 3.6']
                                    occurence_dict['B 3.6'] += 1
                                elif row['DWV_ANZ_Q'] > 1000:
                                    lts_dict[index] = classification_tables['B 3.5']
                                    occurence_dict['B 3.5'] += 1
                                elif ADT_significance == 'high' and row['DWV_ANZ_Q'] <= 1000:
                                    lts_dict[index] = classification_tables['B 3.4'][0]
                                    occurence_dict['B 3.4'] += 1
                                elif ADT_significance == 'low' and row['DWV_ANZ_Q'] <= 1000:
                                    lts_dict[index] = classification_tables['B 3.4'][1]
                                    occurence_dict['B 3.4'] += 1
                            else:
                                if row['DWV_ANZ_Q'] >= 3000:
                                    lts_dict[index] = classification_tables['B 3.3']
                                    occurence_dict['B 3.3'] += 1
                                elif row['DWV_ANZ_Q'] > 1000:
                                    lts_dict[index] = classification_tables['B 3.2']
                                    occurence_dict['B 3.2'] += 1
                                elif ADT_significance == 'high' and row['DWV_ANZ_Q'] <= 1000:
                                    lts_dict[index] = classification_tables['B 3.1'][0]
                                    occurence_dict['B 3.1'] += 1
                                elif ADT_significance == 'low' and row['DWV_ANZ_Q'] <= 1000:
                                    lts_dict[index] = classification_tables['B 3.1'][1]
                                    occurence_dict['B 3.1'] += 1
                elif row['OBJEKTART'] in ['6m Strasse', '8m Strasse', '10m Strasse']:
                        if row['one_way'] == '1 FT':
                            lts_dict[index] = classification_tables['B 4.7']
                            occurence_dict['B 4.7'] += 1
                        else:
                            if row['n_parkings'] > 0:
                                if row['DWV_ANZ_Q'] >= 3000:
                                    lts_dict[index] = classification_tables['B 4.6']
                                    occurence_dict['B 4.6'] += 1
                                elif row['DWV_ANZ_Q'] > 1000:
                                    lts_dict[index] = classification_tables['B 4.5']
                                    occurence_dict['B 4.5'] += 1
                                elif ADT_significance == 'high' and row['DWV_ANZ_Q'] <= 1000:
                                    lts_dict[index] = classification_tables['B 4.4'][0]
                                    occurence_dict['B 4.4'] += 1
                                elif ADT_significance == 'low' and row['DWV_ANZ_Q'] <= 1000:
                                    lts_dict[index] = classification_tables['B 4.4'][1]
                                    occurence_dict['B 4.4'] += 1
                            else:
                                if row['DWV_ANZ_Q'] >= 3000:
                                    lts_dict[index] = classification_tables['B 4.3']
                                    occurence_dict['B 4.3'] += 1
                                elif row['DWV_ANZ_Q'] > 1000:
                                    lts_dict[index] = classification_tables['B 4.2']
                                    occurence_dict['B 4.2'] += 1
                                elif ADT_significance == 'high' and row['DWV_ANZ_Q'] <= 1000:
                                    lts_dict[index] = classification_tables['B 4.1'][0]
                                    occurence_dict['B 4.1'] += 1
                                elif ADT_significance == 'low' and row['DWV_ANZ_Q'] <= 1000:
                                    lts_dict[index] = classification_tables['B 4.1'][1]
                                    occurence_dict['B 4.1'] += 1
            elif row['vmax'] > 30.0:
                if row['OBJEKTART'] in ['3m Strasse', '4m Strasse']:
                        if row['one_way'] == '1 FT':
                            lts_dict[index] = classification_tables['B 5.7']
                            occurence_dict['B 5.7'] += 1
                        else:
                            if row['n_parkings'] > 0:
                                if row['DWV_ANZ_Q'] >= 3000:
                                    lts_dict[index] = classification_tables['B 5.6']
                                    occurence_dict['B 5.6'] += 1
                                elif row['DWV_ANZ_Q'] > 1000:
                                    lts_dict[index] = classification_tables['B 5.5']
                                    occurence_dict['B 5.5'] += 1
                                elif  row['DWV_ANZ_Q'] <= 1000:
                                    lts_dict[index] = classification_tables['B 5.4']
                                    occurence_dict['B 5.4'] += 1
                            else:
                                if ADT_significance == 'high' and row['DWV_ANZ_Q'] >= 3000:
                                    lts_dict[index] = classification_tables['B 5.3'][0]
                                    occurence_dict['B 5.3'] += 1
                                elif ADT_significance == 'low' and row['DWV_ANZ_Q'] >= 3000:
                                    lts_dict[index] = classification_tables['B 5.3'][1]
                                    occurence_dict['B 5.3'] += 1
                                elif row['DWV_ANZ_Q'] > 1000:
                                    lts_dict[index] = classification_tables['B 5.2']
                                    occurence_dict['B 5.2'] += 1
                                elif row['DWV_ANZ_Q'] <= 1000:
                                    lts_dict[index] = classification_tables['B 5.1']
                                    occurence_dict['B 5.1'] += 1
                elif row['OBJEKTART'] in ['6m Strasse']:
                        if row['one_way'] == '1 FT':
                            lts_dict[index] = classification_tables['B 6.7']
                            occurence_dict['B 6.7'] += 1
                        else:
                            if row['n_parkings'] > 0:
                                if row['DWV_ANZ_Q'] >= 3000:
                                    lts_dict[index] = classification_tables['B 6.6']
                                    occurence_dict['B 6.6'] += 1
                                elif row['DWV_ANZ_Q'] > 1000:
                                    lts_dict[index] = classification_tables['B 6.5']
                                    occurence_dict['B 6.5'] += 1
                                elif  row['DWV_ANZ_Q'] <= 1000:
                                    lts_dict[index] = classification_tables['B 6.4']
                                    occurence_dict['B 6.4'] += 1
                            else:
                                if ADT_significance == 'high' and row['DWV_ANZ_Q'] >= 3000:
                                    lts_dict[index] = classification_tables['B 6.3'][0]
                                    occurence_dict['B 6.3'] += 1
                                elif ADT_significance == 'low' and row['DWV_ANZ_Q'] >= 3000:
                                    lts_dict[index] = classification_tables['B 6.3'][1]
                                    occurence_dict['B 6.3'] += 1
                                elif row['DWV_ANZ_Q'] > 1000:
                                    lts_dict[index] = classification_tables['B 6.2']
                                    occurence_dict['B 6.2'] += 1
                                elif row['DWV_ANZ_Q'] <= 1000:
                                    lts_dict[index] = classification_tables['B 6.1'] 
                                    occurence_dict['B 6.1'] += 1 
                elif row['OBJEKTART'] in ['8m Strasse']:
                        if row['one_way'] == '1 FT':
                            lts_dict[index] = classification_tables['B 7.5']
                            occurence_dict['B 7.5'] += 1
                        else:
                            if row['n_parkings'] > 0:
                                if row['DWV_ANZ_Q'] >= 5000:
                                    lts_dict[index] = classification_tables['B 7.4']
                                    occurence_dict['B 7.4'] += 1
                                elif  row['DWV_ANZ_Q'] < 5000:
                                    lts_dict[index] = classification_tables['B 7.3']
                                    occurence_dict['B 7.3'] += 1
                            else:
                                if row['DWV_ANZ_Q'] >= 5000:
                                    lts_dict[index] = classification_tables['B 7.2']
                                    occurence_dict['B 7.2'] += 1
                                elif  row['DWV_ANZ_Q'] < 5000:
                                    lts_dict[index] = classification_tables['B 7.1']
                                    occurence_dict['B 7.1'] += 1
                elif row['OBJEKTART'] in ['10m Strasse']:
                        if row['one_way'] == '1 FT':
                            lts_dict[index] = classification_tables['B 8.5']
                            occurence_dict['B 8.5'] += 1
                        else:
                            if row['n_parkings'] > 0:
                                if row['DWV_ANZ_Q'] >= 5000:
                                    lts_dict[index] = classification_tables['B 8.4']
                                    occurence_dict['B 8.4'] += 1
                                elif  row['DWV_ANZ_Q'] < 5000:
                                    lts_dict[index] = classification_tables['B 8.3']
                                    occurence_dict['B 8.3'] += 1
                            else:
                                if row['DWV_ANZ_Q'] >= 5000:
                                    lts_dict[index] = classification_tables['B 8.2']
                                    occurence_dict['B 8.2'] += 1
                                elif  row['DWV_ANZ_Q'] < 5000:
                                    lts_dict[index] = classification_tables['B 8.1']
                                    occurence_dict['B 8.1'] += 1
            
            if row['near_tram'] == 1:
                if lts_dict[index] < 3:
                    lts_dict[index] = 3
                elif lts_dict[index] == 3:
                    lts_dict[index] = 4
            
            if row['ped_island'] == 1:
                lts_dict[index] += 1

            if slope_inclusion is True:
                if row['median_slope'] > 6:
                    if lts_dict[index] < 3:
                        lts_dict[index] += 1

            if planned_network is True:
            # Workaround in case the planned network inputs have lts > 1
                if row['velonetz'] == 'Vorzugsrou':
                        if planned_network_classification['Vorzugsrouten'] is not None:
                            if index in lts_dict:
                                if lts_dict[index] > planned_network_classification['Vorzugsrouten']:
                                    lts_dict[index] = planned_network_classification['Vorzugsrouten']
                                    continue
                elif row['velonetz'] == 'Basisnetz':
                    if planned_network_classification['Basisnetz'] is not None:
                        if index in lts_dict:
                            if lts_dict[index] > planned_network_classification['Basisnetz']:
                                lts_dict[index] = planned_network_classification['Basisnetz']
                                continue
                elif row['velonetz'] == 'Hauptnetz':
                    if planned_network_classification['Hauptnetz'] is not None:
                        if index in lts_dict:
                            if lts_dict[index] > planned_network_classification['Hauptnetz']:
                                lts_dict[index] = planned_network_classification['Hauptnetz']
                                continue

    for k,v in lts_dict.items():
        if v > 4:
            lts_dict[k] = 4
    
    # road connections LTS of higher neighbouring segment
    for index, row in roads.iterrows(): 
        if row['OBJEKTART'] == 'Verbindung':
            if index-1 in lts_dict and index+1 in lts_dict:
                lts_dict[index] = max([lts_dict[index-1], lts_dict[index+1]])
            elif index-1 in lts_dict:
                lts_dict[index] = lts_dict[index-1]
            else:
                lts_dict[index] = lts_dict[index+1]
        
    count_dict = {}
    for i in lts_dict.values():
        y = i
        if y not in count_dict:
            count_dict[y] = 1
        else:
            count_dict[y] += 1
    
    print('This classification results in the following LTS level distribution:')
    print(f'    LTS 1: n = {count_dict[1]}, p = {round((count_dict[1]/len(road_network))*100,2)}%')
    print(f'    LTS 2: n = {count_dict[2]}, p = {round((count_dict[2]/len(road_network))*100,2)}%')
    print(f'    LTS 3: n = {count_dict[3]}, p = {round((count_dict[3]/len(road_network))*100,2)}%')
    print(f'    LTS 4: n = {count_dict[4]}, p = {round((count_dict[4]/len(road_network))*100,2)}%')   
    
    classified = road_network.copy()
    lts_df = pd.DataFrame.from_dict(lts_dict, orient = 'index', columns = ['LTS']) 
    classified['LTS'] = lts_df

    occurence_df = pd.DataFrame.from_dict(occurence_dict, orient = 'index', columns = ['count'])
    
    return classified, occurence_df

# This function splits the road network into 4 networks, for each Level of traffic stress group their own. The roads with higher LTS are not deleted but weighted (LTS + 1 -> weight of 1000, LTS + 2 -> 2000, etc.). The weighting is done like this, so that in the upcoming calculations, shortest path cutoff of 999 can be set for a hard LTS barrier and a cutoff of 1999 or even 2999 can be set for a soft LTS barrier.
def split_weigh_network(network):

    # For each LTS a new copy of the network is created and the road segments that have a LTS that is too high are weighted accordingly.
    Network_LTS_1 = network.copy()
    for S, E, a in Network_LTS_1.edges(data = True):
        if a['LTS'] == 1:
            Network_LTS_1[S][E]['weight'] = 1
        else:
            if a['LTS'] == 2:
                Network_LTS_1[S][E]['weight'] = 1000
            elif a['LTS'] == 3:
                Network_LTS_1[S][E]['weight'] = 2000
            elif a['LTS'] == 4:
                Network_LTS_1[S][E]['weight'] = 3000
    Network_LTS_2 = network.copy()
    for S, E, a in Network_LTS_2.edges(data = True):
            if a['LTS'] <= 2:
                Network_LTS_2[S][E]['weight'] = 1
            else:
                if a['LTS'] == 3:
                    Network_LTS_2[S][E]['weight'] = 1000
                elif a['LTS'] == 4:
                    Network_LTS_2[S][E]['weight'] = 2000
    Network_LTS_3 = network.copy()
    for S, E, a in Network_LTS_3.edges(data = True):
        if a['LTS'] <= 3:
            Network_LTS_3[S][E]['weight'] = 1
        else:
            Network_LTS_3[S][E]['weight'] = 1000
    Network_LTS_4 = network.copy()
    for S, E, a in Network_LTS_4.edges(data = True):
        Network_LTS_4[S][E]['weight'] = 1

    return Network_LTS_1, Network_LTS_2, Network_LTS_3, Network_LTS_4  

# This function choses a node in the given network and randomly choses a second node in the network that is inside of the given radius from the first point.
def random_2_nodes_in_radius(network, radius):
    
    # Choses random node in network and makes a radius around it
    node_1 = rd.choice(list(network.nodes))
    node_2 = None
    poly_radius = sh.Point(node_1).buffer(radius)

    # Choses a second random node that is inside the radius
    node_list = list(network.nodes)
    rd.shuffle(node_list)
    for i in node_list:
        if i != node_1:
            if poly_radius.contains(sh.Point(i)):
                node_2 = i
                break
        else:
            continue
    
    return node_1, node_2

# This function calculates a measure of connectivity of the given network. It does so by generating trips in a certain distance range and checking if they are firstly connected at all with the given LTS network and secondly if the paths dont have an excessive amount of detour compared to the optimal shortest path. 
# Inputs: connectivity_measure(network(nx-network), n_iterations(int), distance(int), lts_barrier = 'soft' ('hard'/'soft'), output = True(bool))
# Output: n trips, n trips connected, n trips unconnected due to LTS, n trips unconnected due to detour, % connected trips
def connectivity_measure(network, n_iterations, distance, lts_barrier = 'soft', output = True):
    
    if lts_barrier not in ['hard', 'soft']:
        raise ValueError("LTS barrier must either be 'hard' or 'soft'")
    if lts_barrier == 'hard':
        cutoff = 999
    else:
        cutoff = 1999
   
    # Some helper variables
    vis_paths_dict = {}
    no_path_counter = 0
    trip_number = 0
    weighted_connected = 0
    weighted_unconnected_lts = 0
    weighted_unconnected_detour = 0
    distances_list = []

    # For further use option that no output is generated to not clog outputs
    if output is False:
        tq = tqdm(range(n_iterations), desc = 'Generating paths and calculating connectivity', file=sys.stdout, leave = False)
    else:
        tq = tqdm(range(n_iterations), desc = 'Generating paths and calculating connectivity', file=sys.stdout)
          
    for i in tq:
        shortest_weighted = None
        # Chose two random nodes in the radius of the distance input with this helper function
        S, E = random_2_nodes_in_radius(network, distance)
        distances_list.append(spatial.distance.euclidean(S, E))
        # Checking for a path at all
        if nx.has_path(network, S, E):
            if E is None:
                no_path_counter += 1
                continue
            else:
                # Calculating the unweighted optimal shortest path
                shortest_unweighted = nx.single_source_dijkstra(network, S, E, weight = 'edge_lengths')
                shortest_unweighted_nodes = shortest_unweighted[1]
                geoms_list = []
                for i,j in enumerate(shortest_unweighted_nodes):
                    if i < len(shortest_unweighted_nodes)-1:
                        geoms_list.append(network[j][shortest_unweighted_nodes[i+1]]['geometry'])
                shortest_unweighted = sh.ops.linemerge(geoms_list)
                len_shortest_unweighted = shortest_unweighted.length
                
                # Calculate the weighted shortest path
                try:
                    shortest_weighted = nx.single_source_dijkstra(network, S, E, cutoff = cutoff, weight = 'weight')
                except:
                    weighted_unconnected_lts += 1
                    pass
                else:
                    geoms_list = []
                    shortest_weighted_nodes = shortest_weighted[1]
                    for i,j in enumerate(shortest_weighted_nodes):
                        if i < len(shortest_weighted_nodes)-1:
                            geoms_list.append(network[j][shortest_weighted_nodes[i+1]]['geometry'])
                    shortest_weighted = sh.ops.linemerge(geoms_list)
                    len_shortest_weighted = shortest_weighted.length
                    
                    # Check if there is an excessive amount of detour between the two
                    if len_shortest_weighted - len_shortest_unweighted <= 500:
                        weighted_connected += 1
                    elif len_shortest_weighted / len_shortest_unweighted <= 1.25:
                        weighted_connected += 1
                    else:
                        weighted_unconnected_detour += 1
        
        else:
            no_path_counter += 1
    
    # Calculate the % of connected trips
    if (n_iterations - no_path_counter) == 0:
        connectivity = 0
    else:
        connectivity = round(weighted_connected / (n_iterations - no_path_counter) *100, 2)

    # Save the outputs in a row of a df for further use
    result_row = pd.DataFrame(columns= ['n_trips', 'n_connected', 'n_unconnected_lts', 'n_unconnected_detour', 'p_connectivity', 'avg_distance']) 
    result_row.loc[len(result_row)] = [n_iterations - no_path_counter, weighted_connected, weighted_unconnected_lts, weighted_unconnected_detour, connectivity, round(np.mean(distances_list),2)]             
    
    if output is True:
        print(f'\nFrom the {n_iterations} iterations, {no_path_counter} times there was no path at all between start and end node.') 
        print(f'Of the remaining {n_iterations - no_path_counter} paths:')
        print(f'    - {weighted_connected} trips were possible with the respective LTS level tolerance.')
        print(f'    - {weighted_unconnected_lts} trips were not connected due to the connecting roads LTS being to high.')
        print(f'    - {weighted_unconnected_detour} trips were deemed "unconnected" due to the involved excessive detour.')
        print(f'Resulting in a connectivity of {connectivity}% of {n_iterations} trips randomly generated in a radius of {distance} meters.')

    return result_row

# This function tests the stability of the connectivity measure by iteratively running it and checking if the different results follow normal distribution. 
def monte_carlo_connectivity(network, loops, distance, n_iterations, lts_barrier = 'hard'):
    
    # Helper variables
    p_connected_list = []
    p_unconnected_lts_list = []
    p_unconnected_detour_list = []
    distances_list = []

    # Running the connectivity measure a certain amount of time
    tq = tqdm(range(loops), desc = 'Monte Carlo Approximation of average values', file=sys.stdout)
    for i in tq:
        res_list = connectivity_measure(network, n_iterations, distance, lts_barrier, output = False)
        p_connected_list.append(res_list.loc[0, 'n_connected']/res_list.loc[0, 'n_trips']*100)
        p_unconnected_lts_list.append(res_list.loc[0, 'n_unconnected_lts']/res_list.loc[0, 'n_trips']*100)
        p_unconnected_detour_list.append(res_list.loc[0, 'n_unconnected_detour']/res_list.loc[0, 'n_trips']*100)
        distances_list.append(res_list.loc[0, 'avg_distance'])

    # Calculating the means
    avg_connected = round(np.mean(p_connected_list), 2)
    avg_unconnected_lts = round(np.mean(p_unconnected_lts_list), 2) 
    avg_unconnected_detour = round(np.mean(p_unconnected_detour_list), 2)
    avg_distances = round(np.mean(distances_list),2)
    
    # Result row
    result_row = pd.DataFrame(columns= ['avg_p_connected', 'avg_p_unconnected_lts', 'avg_p_unconnected_detour', 'avg_distance']) 
    result_row.loc[len(result_row)] = [ str(avg_connected) +' ±' + str(round(2*np.std(p_connected_list),2)), str(avg_unconnected_lts) +' ±'+ str(round(2*np.std(p_unconnected_lts_list),2)), str(avg_unconnected_detour) +' ±'+ str(round(2*np.std(p_unconnected_detour_list),2)), str(avg_distances) +' ±'+ str(round(2*np.std(distances_list),2)) ]             
     
    # Subplots with all the variables
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2, 2, figsize=(10, 10), sharey = True)
    ax1.hist(p_connected_list, bins = 20)
    ax1.set_title('Connected trips (%)')
    ax1.axvline(np.mean(p_connected_list), color='k', linestyle='dashed', linewidth=1)
    ax1.set_xlabel('% trips')
    ax1.text(0.8, 0.9, 'μ = '+str(avg_connected)+' ±' + str(round(2*np.std(p_connected_list),2)), horizontalalignment='center', verticalalignment='center', transform = ax1.transAxes)
    ax2.hist(p_unconnected_lts_list, bins = 20)
    ax2.set_title(f'Unconnected trips due to LTS (%)')
    ax2.set_xlabel('% trips')
    ax2.axvline(np.mean(p_unconnected_lts_list), color='k', linestyle='dashed', linewidth=1)
    ax2.text(0.8, 0.9, 'μ = '+str(avg_unconnected_lts)+' ±'+ str(round(2*np.std(p_unconnected_lts_list),2)), horizontalalignment='center', verticalalignment='center', transform = ax2.transAxes)
    ax3.hist(p_unconnected_detour_list, bins = 20)
    ax3.set_title(f'Unconnected trips due to excessive detour (%)')
    ax3.set_xlabel('% trips')
    ax3.axvline(np.mean(p_unconnected_detour_list), color='k', linestyle='dashed', linewidth=1)
    ax3.text(0.8, 0.9, 'μ = '+str(avg_unconnected_detour)+' ±'+ str(round(2*np.std(p_unconnected_detour_list),2)), horizontalalignment='center', verticalalignment='center', transform = ax3.transAxes)
    ax4.hist(distances_list, bins = 20)
    ax4.set_title(f'Average distances')
    ax4.set_xlabel('Distance between nodes (m)')
    ax4.axvline(np.mean(distances_list), color='k', linestyle='dashed', linewidth=1)
    ax4.text(0.8, 0.9, 'μ = '+str(avg_distances)+' ±'+ str(round(2*np.std(distances_list),2)), horizontalalignment='center', verticalalignment='center', transform = ax4.transAxes)
    fig.suptitle('\n' + str(loops) +' Monte Carlo iterations * '+str(n_iterations)+' node pairs\n', size = 20)
    fig.supylabel('n Monte Carlo iterations')
    plt.tight_layout()

    return result_row
    
# Short helper function to find the closest node from a point
def closest_node(network, point):
    nodes = list(network.nodes())
    return nodes[spatial.distance.cdist([point], nodes).argmin()]

# Short heleper function to find the closest point from a list of points
def closest_point_from_list(point, lst):
            pnt = (point.x, point.y)
            lst_2 = [(i.x, i.y) for i in lst]
            closest = spatial.distance.cdist([pnt], lst_2).argmin()
            return lst[closest]

# This function is similar to the connectivity measure but instead of generating random trips from any nodes, it has specific starting points and specific ending points (poi's). The function can focus on a specific neighbourhood or district and can also focus on a type of poi if the goal is to answer a specific question.
def accessibility_measure(network, start_points, poi, n_iterations, lts_barrier = 'hard', poi_type = None, neighbourhood = None, district = None, nearest_poi = False, output = True):
    
    # Preventing bad inputs
    poi_types = list(poi['type'].unique())
    if poi_type is not None:
        if poi_type not in poi_types:
            raise ValueError(f'POI Type must be one of the following: {poi_types}')
    if neighbourhood is not None and district is not None:
        raise ValueError('Choose either neighbourhood or district, not both.')
    neighbourhoods = list(start_points['qname'].unique())
    if neighbourhood is not None:
        if neighbourhood not in neighbourhoods:
            raise ValueError(f'Neighbourhood must be one of the following: {neighbourhoods}')
    districts = list(start_points['kname'].unique())    
    if district is not None:
        if district not in districts:
            raise ValueError(f'District must be one of the following: {districts}')
    if lts_barrier not in ['hard', 'soft']:
        raise ValueError("Slope inclusion must either be 'hard' or 'soft'")
    if lts_barrier == 'hard':
        cutoff = 999
    else:
        cutoff = 1999
    if nearest_poi not in [True, False]:
        raise ValueError("Nearest Poi must either be True or False")
        
    # Subsetting the input data according to additional filtering input
    if poi_type is not None:
        poi = poi[poi['type'] == poi_type]
    if neighbourhood is not None:
        start_points = start_points[start_points['qname'] == neighbourhood]
    if district is not None:
        start_points = start_points[start_points['kname'] == district]
    
    # Spatial level of focus
    if neighbourhood is not None:
        desc_spatial = neighbourhood
        prnt_desc = 'neighbourhood'
    elif district is not None:
        desc_spatial = district
        prnt_desc = 'district'
    else:
        desc_spatial = None
        prnt_desc = ''
    
    # For further function the option to have no output
    if output is False:
        tq = tqdm(range(n_iterations), desc = f'{prnt_desc} Generating paths and calculating connectivity', file=sys.stdout, leave = False)
    else:
        tq = tqdm(range(n_iterations), desc = f'{prnt_desc} Generating paths and calculating connectivity', file=sys.stdout)

    # Some helper variables
    no_path_counter = 0
    trip_number = 0
    weighted_connected = 0
    weighted_unconnected_lts = 0
    weighted_unconnected_detour = 0

    for i in tq:
        # Chosing a start point
        S_point = start_points.loc[rd.choice(list(start_points.index))]
        S_node = S_point['closest_node']
        E_point = None
        # Generating a buffer around the start point with 3 km, which can be approximated as an average length trip in the city
        poly_radius = sh.Point(S_node).buffer(3000)
        # Finding the pois in the buffer
        target_list = list(poi.index)
        rd.shuffle(target_list)
        pot_targets = []
        for i in target_list:
            if poly_radius.contains(sh.Point(poi.loc[i]['closest_node'])):
                pot_targets.append(i)
                break
            else:
                continue
        # If we want the nearest poi, this does that
        if nearest_poi is True:
            E_point = closest_point_from_list(S_point['geometry'], [poi.loc[i]['geometry'] for i in pot_targets])
            if E_point is not None:
                E_node = poi.loc[poi['geometry'] == E_point]['closest_node'].values[0]
        else:
            E_point = pot_targets[0]
            if E_point is not None:
                E_node = poi.loc[E_point]['closest_node']

        # Now checking for connectivity of the two nodes similar to the connectivity measure
        if nx.has_path(network, S_node, E_node):
            if E_node is None:
                no_path_counter += 1
                continue
            else:
                shortest_unweighted = nx.single_source_dijkstra(network, S_node, E_node, weight = 'edge_lengths')
                shortest_unweighted_nodes = shortest_unweighted[1]
                geoms_list = []
                for i,j in enumerate(shortest_unweighted_nodes):
                    if i < len(shortest_unweighted_nodes)-1:
                        geoms_list.append(network[j][shortest_unweighted_nodes[i+1]]['geometry'])
                shortest_unweighted = sh.ops.linemerge(geoms_list)
                len_shortest_unweighted = shortest_unweighted.length
            
                try:
                    shortest_weighted = nx.single_source_dijkstra(network, S_node, E_node, cutoff = cutoff, weight = 'weight')  
                except:
                    weighted_unconnected_lts += 1
                    pass
                else:
                    geoms_list = []
                    shortest_weighted_nodes = shortest_weighted[1]
                    for i,j in enumerate(shortest_weighted_nodes):
                        if i < len(shortest_weighted_nodes)-1:
                            geoms_list.append(network[j][shortest_weighted_nodes[i+1]]['geometry'])
                    shortest_weighted = sh.ops.linemerge(geoms_list)
                    len_shortest_weighted = shortest_weighted.length
                    
                    if len_shortest_weighted - len_shortest_unweighted <= 500:
                        weighted_connected += 1
                    elif len_shortest_weighted / len_shortest_unweighted <= 1.25:
                        weighted_connected += 1
                    else:
                        weighted_unconnected_detour += 1

        else:
            no_path_counter += 1

    # Result row to use it in a further function
    result_row = pd.DataFrame(columns= ['desc_spatial', 'desc_poi', 'n_trips', 'n_connected', 'n_unconnected_lts', 'n_unconnected_detour', 'p_accessibility']) 
    result_row.loc[len(result_row)] = [desc_spatial, poi_type, n_iterations - no_path_counter, weighted_connected, weighted_unconnected_lts, weighted_unconnected_detour, round(weighted_connected / (n_iterations - no_path_counter) *100, 2) ]             

    # Different outputs depending what input we give
    if output is True:
        if poi_type is not None and neighbourhood is not None:
            print(f'\nFor the neighbourhood {neighbourhood}:')
            print(f'From the {n_iterations} iterations, {no_path_counter} times there was no path at all between start and end point.')
            print(f'Of the remaining {n_iterations - no_path_counter} paths:')
            print(f'    - {weighted_connected} times, a {poi_type} was accessible with the respective LTS level tolerance.')
            print(f'    - {weighted_unconnected_lts} times, a {poi_type} was not accessible due to the connecting roads LTS being to high.')
            print(f'    - {weighted_unconnected_detour} times, a {poi_type} were deemed not accessible due to the involved excessive detour.')
            print(f'Resulting in a accessibility of {weighted_connected / (n_iterations - no_path_counter) *100:.2f}% of trips randomly generated to a {poi_type} in a typical average bicycle trip length of 3 km.')
    
        elif poi_type is not None and district is not None:
            print(f'\nFor the district {district}:')
            print(f'From the {n_iterations} iterations, {no_path_counter} times there was no path at all between start and end point.')
            print(f'Of the remaining {n_iterations - no_path_counter} paths:')
            print(f'    - {weighted_connected} times, a {poi_type} was accessible with the respective LTS level tolerance.')
            print(f'    - {weighted_unconnected_lts} times, a {poi_type} was not accessible due to the connecting roads LTS being to high.')
            print(f'    - {weighted_unconnected_detour} times, a {poi_type} were deemed not accessible due to the involved excessive detour.')
            print(f'Resulting in a accessibility of {weighted_connected / (n_iterations - no_path_counter) *100:.2f}% of trips randomly generated to a {poi_type} in a typical average bicycle trip length of 3 km.')
        
        elif poi_type is not None:
            print(f'\nFrom the {n_iterations} iterations, {no_path_counter} times there was no path at all between start and end point.')
            print(f'Of the remaining {n_iterations - no_path_counter} paths:')
            print(f'    - {weighted_connected} times, a {poi_type} was accessible with the respective LTS level tolerance.')
            print(f'    - {weighted_unconnected_lts} times, a {poi_type} was not accessible due to the connecting roads LTS being to high.')
            print(f'    - {weighted_unconnected_detour} times, a {poi_type} were deemed not accessible due to the involved excessive detour.')
            print(f'Resulting in a accessibility of {weighted_connected / (n_iterations - no_path_counter) *100:.2f}% of trips randomly generated to a {poi_type} in a typical average bicycle trip length of 3 km.')
        
        elif neighbourhood is not None:
            print(f'\nFor the neighbourhood {neighbourhood}:')
            print(f'From the {n_iterations} iterations, {no_path_counter} times there was no path at all between start and end point.')
            print(f'Of the remaining {n_iterations - no_path_counter} paths:')
            print(f'    - {weighted_connected} times, the public service was accessible with the respective LTS level tolerance.')
            print(f'    - {weighted_unconnected_lts} times, the public service was not accessible due to the connecting roads LTS being to high.')
            print(f'    - {weighted_unconnected_detour} times, the public service were deemed not accessible due to the involved excessive detour.')
            print(f'Resulting in a accessibility of {weighted_connected / (n_iterations - no_path_counter) *100:.2f}% of trips randomly generated to public services in a typical average bicycle trip length of 3 km.')
        
        elif district is not None:
            print(f'\nFor the district {district}:')
            print(f'From the {n_iterations} iterations, {no_path_counter} times there was no path at all between start and end point.')
            print(f'Of the remaining {n_iterations - no_path_counter} paths:')
            print(f'    - {weighted_connected} times, the public service was accessible with the respective LTS level tolerance.')
            print(f'    - {weighted_unconnected_lts} times, the public service was not accessible due to the connecting roads LTS being to high.')
            print(f'    - {weighted_unconnected_detour} times, the public service were deemed not accessible due to the involved excessive detour.')
            print(f'Resulting in a accessibility of {weighted_connected / (n_iterations - no_path_counter) *100:.2f}% of trips randomly generated to public services in a typical average bicycle trip length of 3 km.')

        else:
            print(f'\nFrom the {n_iterations} iterations, {no_path_counter} times there was no path at all between start and end point.')
            print(f'Of the remaining {n_iterations - no_path_counter} paths:')
            print(f'    - {weighted_connected} times, the public service was accessible with the respective LTS level tolerance.')
            print(f'    - {weighted_unconnected_lts} times, the public service was not accessible due to the connecting roads LTS being to high.')
            print(f'    - {weighted_unconnected_detour} times, the public service were deemed not accessible due to the involved excessive detour.')
            print(f'Resulting in a accessibility of {weighted_connected / (n_iterations - no_path_counter) *100:.2f}% of trips randomly generated to public services in a typical average bicycle trip length of 3 km.')

    return result_row

# This function uses the accessibility measure with its inuts and calculates them in the same manner for all districts or all neighbourhoods, depending what you choose for the variable spatial level.
def spatial_accessibility_analysis(network, spatial_level, start_points, poi, n_iterations, lts_barrier = 'hard', poi_type = None, nearest_poi = False):
    # Defining the spatial level
    if spatial_level not in ['neighbourhood', 'district']:
        raise ValueError('Spatial level must either be neighbourhood or district')
    res_dict = {}
    
    # Running the accessibility measure with all neighbourhoods
    if spatial_level == 'neighbourhood':
        spatial_list = list(start_points['qname'].unique())
        tq = enumerate(tqdm(spatial_list, desc = f'Calculating accessibility for different {spatial_level}s', file=sys.stdout))
        # Appending them to a gdf 
        for ind, v in tq:
            res_row = accessibility_measure(network, start_points, poi, n_iterations, lts_barrier, poi_type, neighbourhood=v, output = False, nearest_poi = nearest_poi)
            res_dict[ind] = res_row.iloc[0]
    
    # Running the measure with all districts
    else:
        spatial_list = list(start_points['kname'].unique())
        tq = enumerate(tqdm(spatial_list, desc = f'Calculating accessibility for different {spatial_level}s', file=sys.stdout))
        for ind, v in tq:
            res_row = accessibility_measure(network, start_points, poi, n_iterations, lts_barrier, poi_type, district=v, output = False, nearest_poi = nearest_poi)
            res_dict[ind] = res_row.iloc[0]

    res_df = pd.DataFrame.from_dict(res_dict, orient = 'index') 
    res_df.sort_values('p_accessibility', ascending = False, inplace= True)

    return res_df

def vis_spat_analysis(spatial_analysis_df, spatial_division, classified_network):
    # Different spatial level depending on spatial analysis
    if 'Kreis 1' in list(spatial_analysis_df['desc_spatial']):
        spatial = 'kname'
        neighbourhoods_geoms = spatial_division.dissolve(by = 'kname')
        joined = neighbourhoods_geoms.merge(spatial_analysis_df, left_on = spatial, right_on = 'desc_spatial')
        joined.drop(columns = ['qname'], inplace = True)
    else:
        spatial = 'qname'
        neighbourhoods_geoms = spatial_division
        joined = neighbourhoods_geoms.merge(spatial_analysis_df, left_on = spatial, right_on = 'desc_spatial')
        joined.drop(columns = ['kname', 'desc_spatial'], inplace = True)
    
    # LTS of streets to crosscheck
    classified_lts = classified_network[['LTS', 'geometry']]
    
    # Simple folium map
    m = classified_lts.explore(name = "LTS Network", column = 'LTS', categorical = True, cmap = ['#ddea4f', 'orange', '#e62929', '#3A1F04'], show = False, min_zoom = 12, zoom_start = 12)
    m = joined.explore(m = m, legend_kwds = dict(caption = 'Accessibility measure (%)'), name = "Spatial divison", column = 'p_accessibility', cmap = 'Purples', style_kwds = dict(color = 'black', weight = 0.5))
   
    folium.LayerControl().add_to(m)

    return m

def connectivity_overview(Network_1, Network_2, Network_3, Network_4, iterations, distance, lts_barrier):
    
    # Running for all LTS groups
    c1 = connectivity_measure(Network_1, n_iterations = iterations, distance = distance, lts_barrier = lts_barrier, output = False)
    c1['LTS'] = 'LTS 1'
    c2 = connectivity_measure(Network_2, n_iterations = iterations, distance = distance, lts_barrier = lts_barrier, output = False)
    c2['LTS'] = 'LTS 2'
    c3 = connectivity_measure(Network_3, n_iterations = iterations, distance = distance, lts_barrier = lts_barrier, output = False)
    c3['LTS'] = 'LTS 3'
    c4 = connectivity_measure(Network_4, n_iterations = iterations, distance = distance, lts_barrier = lts_barrier, output = False)
    c4['LTS'] = 'LTS 4'

    # Merging them
    con_overview = pd.concat([c1, c2, c3, c4])
    con_overview['p_unconnected_lts'] = round(con_overview['n_unconnected_lts']/con_overview['n_trips']*100,2)
    con_overview['p_unconnected_detour'] = round(con_overview['n_unconnected_detour']/con_overview['n_trips']*100,2)
    con_overview = con_overview[['LTS', 'p_connectivity', 'p_unconnected_lts', 'p_unconnected_detour']].reset_index(drop = True)
    
    return con_overview

def accessibility_overview(Network_1, Network_2, Network_3, Network_4, start_points, poi, iterations, lts_barrier, poi_type):
    
    # Calculating accessibility for all LTS groups
    a1 = accessibility_measure(Network_1, start_points, poi, iterations, lts_barrier = lts_barrier, poi_type = poi_type, nearest_poi = True, output = False)
    a1['LTS'] = 'LTS 1'
    a2 = accessibility_measure(Network_2, start_points, poi, iterations, lts_barrier = lts_barrier, poi_type = poi_type, nearest_poi = True, output = False)
    a2['LTS'] = 'LTS 2'
    a3 = accessibility_measure(Network_3, start_points, poi, iterations, lts_barrier = lts_barrier, poi_type = poi_type, nearest_poi = True, output = False)
    a3['LTS'] = 'LTS 3'
    a4 = accessibility_measure(Network_4, start_points, poi, iterations, lts_barrier = lts_barrier, poi_type = poi_type, nearest_poi = True, output = False)
    a4['LTS'] = 'LTS 4'

    # Merging them
    acc_overview = pd.concat([a1, a2, a3, a4])
    acc_overview['p_unconnected_lts'] = round(acc_overview['n_unconnected_lts']/acc_overview['n_trips']*100,2)
    acc_overview['p_unconnected_detour'] = round(acc_overview['n_unconnected_detour']/acc_overview['n_trips']*100,2)
    acc_overview = acc_overview[['LTS', 'p_accessibility', 'p_unconnected_lts', 'p_unconnected_detour']].reset_index(drop = True)
    
    return acc_overview

## External functions

# Simplify network function from https://stackoverflow.com/questions/70182991/networkx-how-to-combine-edges-together-with-condition

def _is_endpoint(G, node, strict=True):
    """
    Is node a true endpoint of an edge.
    Return True if the node is a "real" endpoint of an edge in the network,
    otherwise False. OSM data includes lots of nodes that exist only as points
    to help streets bend around curves. An end point is a node that either:
    1) is its own neighbor, ie, it self-loops.
    2) or, has no incoming edges or no outgoing edges, ie, all its incident
    edges point inward or all its incident edges point outward.
    3) or, it does not have exactly two neighbors and degree of 2 or 4.
    4) or, if strict mode is false, if its edges have different OSM IDs.
    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    node : int
        the node to examine
    strict : bool
        if False, allow nodes to be end points even if they fail all other rules
        but have edges with different OSM IDs
    Returns
    -------
    bool
    """
    neighbors = set(list(G.predecessors(node)) + list(G.successors(node)))
    n = len(neighbors)
    d = G.degree(node)

    # rule 1
    if node in neighbors:
        # if the node appears in its list of neighbors, it self-loops
        # this is always an endpoint.
        return True

    # rule 2
    elif G.out_degree(node) == 0 or G.in_degree(node) == 0:
        # if node has no incoming edges or no outgoing edges, it is an endpoint
        return True

    # rule 3
    elif not (n == 2 and (d == 2 or d == 4)):
        # else, if it does NOT have 2 neighbors AND either 2 or 4 directed
        # edges, it is an endpoint. either it has 1 or 3+ neighbors, in which
        # case it is a dead-end or an intersection of multiple streets or it has
        # 2 neighbors but 3 degree (indicating a change from oneway to twoway)
        # or more than 4 degree (indicating a parallel edge) and thus is an
        # endpoint
        return True

    # rule 4
    elif not strict:
        # non-strict mode: do its incident edges have different OSM IDs?
        osmids = []

        # add all the edge OSM IDs for incoming edges
        for u in G.predecessors(node):
            for key in G[u][node]:
                osmids.append(G.edges[u, node, key]["osmid"])

        # add all the edge OSM IDs for outgoing edges
        for v in G.successors(node):
            for key in G[node][v]:
                osmids.append(G.edges[node, v, key]["osmid"])

        # if there is more than 1 OSM ID in the list of edge OSM IDs then it is
        # an endpoint, if not, it isn't
        return len(set(osmids)) > 1

    # if none of the preceding rules returned true, then it is not an endpoint
    else:
        return False


def _build_path(G, endpoint, endpoint_successor, endpoints):
    """
    Build a path of nodes from one endpoint node to next endpoint node.
    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    endpoint : int
        the endpoint node from which to start the path
    endpoint_successor : int
        the successor of endpoint through which the path to the next endpoint
        will be built
    endpoints : set
        the set of all nodes in the graph that are endpoints
    Returns
    -------
    path : list
        the first and last items in the resulting path list are endpoint
        nodes, and all other items are interstitial nodes that can be removed
        subsequently
    """
    # start building path from endpoint node through its successor
    path = [endpoint, endpoint_successor]

    # for each successor of the endpoint's successor
    for successor in G.successors(endpoint_successor):
        if successor not in path:
            # if this successor is already in the path, ignore it, otherwise add
            # it to the path
            path.append(successor)
            while successor not in endpoints:
                # find successors (of current successor) not in path
                successors = [n for n in G.successors(successor) if n not in path]

                # 99%+ of the time there will be only 1 successor: add to path
                if len(successors) == 1:
                    successor = successors[0]
                    path.append(successor)

                # handle relatively rare cases or OSM digitization quirks
                elif len(successors) == 0:
                    if endpoint in G.successors(successor):
                        # we have come to the end of a self-looping edge, so
                        # add first node to end of path to close it and return
                        return path + [endpoint]
                    else:  # pragma: no cover
                        # this can happen due to OSM digitization error where
                        # a one-way street turns into a two-way here, but
                        # duplicate incoming one-way edges are present
                        print(
                            f"Unexpected simplify pattern handled near {successor}")
                        return path
                else:  # pragma: no cover
                    # if successor has >1 successors, then successor must have
                    # been an endpoint because you can go in 2 new directions.
                    # this should never occur in practice
                    raise Exception(f"Unexpected simplify pattern failed near {successor}")

            # if this successor is an endpoint, we've completed the path
            return path

    # if endpoint_successor has no successors not already in the path, return
    # the current path: this is usually due to a digitization quirk on OSM
    return path


def _get_paths_to_simplify(G, strict=True):
    """
    Generate all the paths to be simplified between endpoint nodes.
    The path is ordered from the first endpoint, through the interstitial nodes,
    to the second endpoint.
    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    strict : bool
        if False, allow nodes to be end points even if they fail all other rules
        but have edges with different OSM IDs
    Yields
    ------
    path_to_simplify : list
    """
    # first identify all the nodes that are endpoints
    endpoints = set([n for n in G.nodes if _is_endpoint(G, n, strict=strict)])
    print(f"Identified {len(endpoints)} edge endpoints")

    # for each endpoint node, look at each of its successor nodes
    for endpoint in endpoints:
        for successor in G.successors(endpoint):
            if successor not in endpoints:
                # if endpoint node's successor is not an endpoint, build path
                # from the endpoint node, through the successor, and on to the
                # next endpoint node
                yield _build_path(G, endpoint, successor, endpoints)


def simplify_graph(G, strict=True, remove_rings=True, aggregation={}):
    """
    Simplify a graph's topology by removing interstitial nodes.
    Simplifies graph topology by removing all nodes that are not intersections
    or dead-ends. Create an edge directly between the end points that
    encapsulate them, but retain the geometry of the original edges, saved as
    a new `geometry` attribute on the new edge. Note that only simplified
    edges receive a `geometry` attribute. Some of the resulting consolidated
    edges may comprise multiple OSM ways, and if so, their multiple attribute
    values are stored as a list.
    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    strict : bool
        if False, allow nodes to be end points even if they fail all other
        rules but have incident edges with different OSM IDs. Lets you keep
        nodes at elbow two-way intersections, but sometimes individual blocks
        have multiple OSM IDs within them too.
    remove_rings : bool
        if True, remove isolated self-contained rings that have no endpoints
    Returns
    -------
    G : networkx.MultiDiGraph
        topologically simplified graph, with a new `geometry` attribute on
        each simplified edge
    """
    if "simplified" in G.graph and G.graph["simplified"]:  # pragma: no cover
        raise Exception("This graph has already been simplified, cannot simplify it again.")

    print("Begin topologically simplifying the graph...")

    # make a copy to not mutate original graph object caller passed in
    G = G.copy()
    initial_node_count = len(G)
    initial_edge_count = len(G.edges)
    all_nodes_to_remove = []
    all_edges_to_add = []

    # generate each path that needs to be simplified
    for path in _get_paths_to_simplify(G, strict=strict):

        # add the interstitial edges we're removing to a list so we can retain
        # their spatial geometry
        path_attributes = dict()
        for u, v in zip(path[:-1], path[1:]):

            # there should rarely be multiple edges between interstitial nodes
            # usually happens if OSM has duplicate ways digitized for just one
            # street... we will keep only one of the edges (see below)
            edge_count = G.number_of_edges(u, v)
            if edge_count != 1:
                print(f"Found {edge_count} edges between {u} and {v} when simplifying")

            # get edge between these nodes: if multiple edges exist between
            # them (see above), we retain only one in the simplified graph
            edge_data = G.edges[u, v, 0]
            for attr in edge_data:
                if attr in path_attributes:
                    # if this key already exists in the dict, append it to the
                    # value list
                    path_attributes[attr].append(edge_data[attr])
                else:
                    # if this key doesn't already exist, set the value to a list
                    # containing the one value
                    path_attributes[attr] = [edge_data[attr]]

        # consolidate the path's edge segments' attribute values
        for attr in path_attributes:
            if attr in aggregation.keys():
                # if this is an aggregation attribute, aggregate the values
                path_attributes[attr] = aggregation.get(attr)(path_attributes[attr])

        # # construct the new consolidated edge's geometry for this path
        # path_attributes["geometry"] = LineString(
        #     [Point((G.nodes[node]["x"], G.nodes[node]["y"])) for node in path]
        # )

        # add the nodes and edge to their lists for processing at the end
        all_nodes_to_remove.extend(path[1:-1])
        all_edges_to_add.append(
            {"origin": path[0], "destination": path[-1], "attr_dict": path_attributes}
        )

    # for each edge to add in the list we assembled, create a new edge between
    # the origin and destination
    for edge in all_edges_to_add:
        G.add_edge(edge["origin"], edge["destination"], **edge["attr_dict"])

    # finally remove all the interstitial nodes between the new edges
    G.remove_nodes_from(set(all_nodes_to_remove))

    if remove_rings:
        # remove any connected components that form a self-contained ring
        # without any endpoints
        wccs = nx.weakly_connected_components(G)
        nodes_in_rings = set()
        for wcc in wccs:
            if not any(_is_endpoint(G, n) for n in wcc):
                nodes_in_rings.update(wcc)
        G.remove_nodes_from(nodes_in_rings)

    # mark graph as having been simplified
    G.graph["simplified"] = True

    msg = (
        f"Simplified graph: {initial_node_count} to {len(G)} nodes, "
        f"{initial_edge_count} to {len(G.edges)} edges"
    )
    print(msg)
    return G