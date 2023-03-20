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
    g.map_dataframe(sns.swarmplot, x=_v_.flavorName_colName, y="score", hue = 'User', order = sorted(day1_ratings_longdf[_v_.flavorName_colName].unique()), size=6,
                    hue_order = sorted(list(day1_ratings_longdf['User'].unique())));
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