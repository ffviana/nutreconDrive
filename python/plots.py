import sys
sys.path.append('D:/FV/Projects/NUTRECON/nutreconDrive/python')
from variableCoding import Vars
_v_ = Vars()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def day1_flavorRatings(allRatings_df):

    # all Ratings day 1 plot 

    # remove timestamp cols
    timestamp_cols = [s for s in allRatings_df.columns if 'timestamp' in s]
    day1_ratings_df = allRatings_df[allRatings_df['Day'] == 1].drop(columns=timestamp_cols + ['Day'])

    # Remove Top and right spines, set theme and define font scale 
    custom_params = {"axes.spines.right": False, "axes.spines.top": False};
    sns.set_theme(style="whitegrid", rc=custom_params, palette = 'colorblind');
    sns.set(font_scale=1.3);

    # melt Dataframe for plot with seaborn
    day1_ratings_longdf = day1_ratings_df.melt(id_vars = ['Trial', 'User', _v_.flavorName_colName, _v_.flavorID_colName], var_name='scale', value_name='score')

    # Create Figure Subplots with mapped dataframe
    g = sns.FacetGrid(day1_ratings_longdf, col="scale", col_order = [_v_.novelty_colName, _v_.intensity_colName, _v_.pleasanteness_colName], sharey = False, legend_out = True,  height = 6, aspect = 1);
    # add swarmplot
    g.map_dataframe(sns.swarmplot, x=_v_.flavorName_colName, y="score", hue = 'User', 
                    order = sorted(day1_ratings_longdf[_v_.flavorName_colName].unique()), size=10,
                    hue_order = sorted(list(day1_ratings_longdf['User'].unique())),
                    palette = px.colors.qualitative.Alphabet, linewidth=0 );
    # add boxplot
    g.map_dataframe(sns.boxplot, x=_v_.flavorName_colName, y="score", order = sorted(day1_ratings_longdf[_v_.flavorName_colName].unique()), boxprops=dict(facecolor=(0,0,0,0)));
    # change X tick labels
    g.set_xticklabels(sorted(day1_ratings_longdf[_v_.flavorName_colName].unique()), rotation=45, ha= 'right');
    # change Y-axis limits
    g.axes[0,0].set_ylim([-5,105]);
    g.axes[0,1].set_ylim([-5,105]);
    g.axes[0,2].set_ylim([-78,78]);
    # add horizontal line to pleasantness graph
    g.axes[0,2].axhline(0, ls = '--', c = 'black');

    g.add_legend();

    return g

def fullProtocol_flavorRatings(long_df):

    long_df = long_df.dropna(subset='calorie')
    fig = px.line(long_df.dropna(subset = 'calorie'), 
                facet_col = 'variable', facet_row ="calorie",
                x="Day", y='value', color="User",  markers=True,
                color_discrete_sequence = px.colors.qualitative.Alphabet,
                hover_data = {'User':True, 
                            _v_.group_colName:True,
                            'shape':False,
                            'Flavor':True,
                            'calorie':False,
                            'Day': False,
                            'variable': False,
                            'value': True}, 
                category_orders=
                    {"User": sorted(list(long_df['User'].unique())),
                     "calorie": sorted(list(long_df['calorie'].unique()))}
                )
                
    fig.update_layout(
        title="Flavor Ratings (All Subjects)",
        legend_title="Subject ID",
    )
    fig.layout["yaxis1"].title.text = "CS-"
    fig.layout["yaxis4"].title.text = "CS+"

    fig.update_traces(textposition="bottom right")
    fig.update_xaxes(type='category')

    return fig

def lotteryChoices(df, title):

    hover_cols = ['cohort_id', 'User', 'lottery flavor', 'lottery type']
    hover_data = {key:False for key in df.columns}
    for key in hover_cols:
        hover_data[key] = True

    if len(df['lottery type'].unique()) == 1:
        fig = px.line(df, facet_col = 'lottery p', facet_row ="Day",
                    x="lottery qt", y=_v_.probLotteryChoice_colName, 
                    color="User",  markers=True, line_group = _v_.group_colName,
                    color_discrete_sequence = px.colors.qualitative.Alphabet,
                    hover_data = hover_data, 
                    category_orders= {
                                'Day': sorted(list(df['Day'].unique())),
                                'lottery p': sorted(list(df['lottery p'].unique())),
                                }
                        )
        fig.update_layout(
            title = title,
            legend_title="Subject ID",
        )
    else:
        fig = px.line(df, facet_col = 'lottery p', facet_row ="Day",
                    x="lottery qt", y=_v_.probLotteryChoice_colName, 
                    color="User",  markers=True, line_group = _v_.group_colName,
                    symbol = 'lottery type', line_dash='lottery type',
                    color_discrete_sequence = px.colors.qualitative.Alphabet,
                    hover_data = hover_data, 
                    category_orders= {
                                'Day': sorted(list(df['Day'].unique())),
                                'lottery p': sorted(list(df['lottery p'].unique())),
                                }
                        )
        fig.update_layout(
            title = title,
            legend_title="Subject ID, reward",
        )                                
    
    fig.layout["yaxis1"].title.text = ""
    fig.layout["yaxis11"].title.text = ""

    fig.update_traces(textposition="bottom right")

    return fig

# COLORS

def random_color_palette(n):
    colors = []
    for i in range(n):
        # Generate a random RGB color tuple
        color1 = (random.randint(0, 255)/255, random.randint(0, 255)/255, random.randint(0, 255)/255)
        colors.append(color1)

    hex_palette = [mcolors.to_hex(color) for color in colors]
    return hex_palette


def random_paired_color_palette(n_pairs):
    colors = []
    for i in range(n_pairs):
        # Generate a random RGB color tuple
        color1 = (random.randint(0,255), random.randint(0, 255), random.randint(0, 255))
        
        # Generate a random offset for each color component for the second color
        offset = random.randint(-30, 30)
        r2 = color1[0] + offset
        g2 = color1[1] + offset
        b2 = color1[2] + offset
        # Make sure the second color is still within the valid RGB range (0-255)
        r2 = max(min(r2, 255), 0)/255
        g2 = max(min(g2, 255), 0)/255
        b2 = max(min(b2, 255), 0)/255
        color2 = (r2, g2, b2)
        # Append the color pair to the list of colors
        colors.append((color1[0]/255, color1[1]/255, color1[2]/255))
        colors.append(color2)

    hex_palette = [mcolors.to_hex(color) for color in colors]
    return hex_palette

def cool_warm_palette(n_subs):
    # Generate n pairs of colors
    colors = []
    for i in range(n_subs):

        saturation_cool = random.uniform(0.2, 0.4)
        lightness = random.uniform(0.3, 0.6)
        
        hue_cool = random.uniform(0.5, 0.7) # Cool colors have hue between 0.45 and 0.75
        
        cool_color = mcolors.hsv_to_rgb((hue_cool, lightness, saturation_cool))

        hue_warm = random.choice([random.uniform(0, .2)]) # Warm colors have hue between 0 and 0.2, and between 0.8 and 1
        saturation_warm = random.uniform(0.6, 0.8)

        warm_color = mcolors.hsv_to_rgb((hue_warm, lightness, saturation_warm))

        # Append the color pair to the list of colors
        colors.append(warm_color)
        colors.append(cool_color)
    
    hex_palette = [mcolors.to_hex(color) for color in colors]
    return hex_palette