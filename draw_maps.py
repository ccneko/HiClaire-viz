#!/bin/env python3
import os
import sys
from pathlib import Path
import importlib

import re
from datetime import datetime
import logging
import argparse
from collections import defaultdict

import polars as pl 
import pandas as pd
import numpy as np
import math
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import rotate
from scipy.stats import ttest_ind, spearmanr
from intervaltree import IntervalTree

import matplotlib
default_mpl_backend = matplotlib.get_backend()
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib.patches import PathPatch, Rectangle, Ellipse
from matplotlib.ticker import NullLocator, MultipleLocator, FixedLocator, FixedFormatter, FuncFormatter
from matplotlib import colormaps, colors
from matplotlib.colors import LinearSegmentedColormap

timestamp = lambda: datetime.now().strftime('%Y-%m-%d_%H%M%S')
datestamp = lambda: datetime.now().strftime('%Y-%m-%d')

LOG_FORMAT = "%(asctime)s [%(levelname)-5.5s] [%(threadName)-12.12s] [%(module)-10.10s] %(message)s"
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format=LOG_FORMAT)
logger = logging.getLogger(__name__)
logger.info("Start running.")

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams["font.family"] = "Arial"
plt.rcParams['agg.path.chunksize'] = 1000

logger.info(f"Matplotlib backend: {default_mpl_backend}")


def load_config():
    parser = argparse.ArgumentParser(description="Process project configurations and instructions.")

    parser.add_argument(
        "config_path", 
        type=str, 
        help="Path to the figure_config.txt file"
    )

    parser.add_argument(
        "--project_dir", 
        type=str, 
        default=None, 
        help="Path to the project directory"
    )

    parser.add_argument(
        "--sample_sheet", 
        type=str, 
        default="db/sample.xlsx", 
        help="Path to the Excel sample sheet file"
    )

    parser.add_argument(
        "--intrachr", 
        action="store_true", 
        help="Input data is intrachromosome only"
    )

    parser.add_argument(
        "--contact_probability", 
        action='store_true',
        help="Use contact probability instead of Hi-C matrix score for plotting"
    )

    parser.add_argument(
        "--sum_1",
        action='store_true',
        help="Normalize the matrix so that the sum of all entries equals 1 for plotting"
    )

    parser.add_argument(
        "--domain_type", 
        default='condensin', #cohesin
        help="Target domain type. Detect kink within the corresponding size range"
    )

    parser.add_argument(
        "--svg", 
        action='store_true',
        help="Output in SVG instead of PNG"
    )

    parser.add_argument(
        "--target_bed_path", 
        default="db/domain1_all.txt",
        help="Default path to domain location BED file for NSMB 2017 1e style scatter plot"
    )

    args = parser.parse_args()

    logger.info(f"Instructions file: {args.config_path}")
    logger.info(f"Project directory: {args.project_dir}")
    logger.info(f"Sample sheet: {args.sample_sheet}")
    return args


def get_kb_length(length):
    if not isinstance(length, (str, int)):
        kb = False
    elif 'kb' in length:
        kb = int(re.search(r'(\d+)kb', length)[1])
    elif 'bp' in length:
        kb = int(re.search(r'(\d+)bp', length)[1])/1000
    elif 'Mb' in length:
        kb = int(re.search(r'(\d+)Mb', length)[1])*1000
    return kb


def matrix_to_df(matrix_file, sum_1=False):
    df = pd.read_csv(matrix_file, sep='\t', header=0, index_col=0).fillna(0)
    if sum_1:
        df = df / np.nansum(df)
    return df


def get_diff_matrix(df1, df2, log2=False):
    if log2:
        return np.log2(df1/df2)
    else:
        return df1 - df2


def get_matrix_path(project_dir, sample_name, res, norm, extra_dir="", chrom="ALL"):
    if res >= 1:
        if res%1000 == 0:
            res_unit = 'Mb'
            res = int(res/1000)
        else:
            res_unit = 'kb'
    else:
        res = res * 1000
        res_unit = 'bp'
    return Path(f'{project_dir}/data/{extra_dir}/{sample_name}/{res}{res_unit}/{norm}/{chrom}.matrix.gz')


def read_to_draw(fig_instruction_file):
    to_draw = pd.read_csv(fig_instruction_file, sep='\t', header=0, comment="#")
    to_draw['mat1'] = to_draw['mat1'].fillna(-1)
    to_draw['mat2'] = to_draw['mat2'].fillna(-1)
    to_draw['start'] = to_draw['start'].fillna(-1).astype(int)
    to_draw['end'] = to_draw['end'].fillna(-1).astype(int)
    return to_draw


def set_output_path(project_dir, row, extension='png'):
    if row['mat2']!=-1:
        mat2 = row['mat2']
    else:
        mat2 = 'single_matrix'
    if isinstance(row['chr'], (str, int)):
        chrom = row['chr']
    else:
        chrom = 'whole'
    if row['start']!=-1:
        start = row['start']
        end = row['end']
    else:
        start = '_'
        end = '_'
    name_chain = '-'.join(list(map(str, 
                    [   row['fig'], row['mat1'], mat2, row['norm'], row['res'], 
                        row['minmax'], row['cmap'], chrom, start, end]
                )))
    output_path = f'{project_dir}/out/{datestamp()}_hic_figures/{row["subfolder"]}'
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    return f"{output_path}/{name_chain}.{extension}"


new_colormaps_to_use = {
    'hideki_bright': ['#08306B', '#08519C', '#2171B5', '#4292C6', '#6BAED6', '#9ECAE1', '#C6DBEF', '#DEEBF7', 
                      '#FFF5F0', '#FEE0D2', '#FCBBA1', '#FC9272', '#FB6A4A', '#EF3B2C', '#CB181D', '#A50F15', '#67000D'],
    'NSMBish':       [  '#053061',  '#073568',  '#0A3A70',  '#0D4077',  '#10457F',  
                        '#134B86',  '#15508E',  '#185696',  '#1B5B9D',  '#1E61A5',  
                        '#2166AC',  '#246AAE',  '#286FB0',  '#2B74B3',  '#2F78B5',  
                        '#327DB7',  '#3581BA',  '#3986BC',  '#3C8ABE',  '#408FC1',  
                        '#4494C3',  '#4C99C6',  '#549EC9',  '#5CA3CB',  '#64A8CE',  
                        '#6CADD1',  '#74B2D3',  '#7CB7D6',  '#84BCD9',  '#8CC1DC',  
                        '#93C5DE',  '#9AC9E0',  '#A0CCE2',  '#A7CFE4',  '#ADD2E5',  
                        '#B3D6E7',  '#BAD9E9',  '#C0DCEB',  '#C6DFED',  '#CDE3EE',  
                        '#D2E6F0',  '#D7E8F2',  '#DCEBF3',  '#E0EDF5',  '#E5F0F6',  
                        '#EAF3F8',  '#EEF5F9',  '#F3F8FB',  '#F8FBFC',  '#FCFDFE',  
                        '#FEFDFC',  '#FEF9F6',  '#FEF5F0',  '#FEF2EB',  '#FEEEE5',  
                        '#FDEBDF',  '#FDE7DA',  '#FDE3D4',  '#FDE0CE',  '#FDDCC9',  
                        '#FCD7C2',  '#FBD2BB',  '#FACCB4',  '#F9C7AD',  '#F8C1A6',  
                        '#F7BC9F',  '#F7B799',  '#F6B192',  '#F5AC8B',  '#F4A684',  
                        '#F1A07E',  '#EE9978',  '#EB9273',  '#E88B6E',  '#E58468',  
                        '#E27D63',  '#DF765E',  '#DC6F58',  '#D96853',  '#D6614E',  
                        '#D35A4A',  '#CF5246',  '#CB4B43',  '#C8443F',  '#C43D3C',  
                        '#C03539',  '#BD2E35',  '#B92732',  '#B6202E',  '#B2182B',  
                        '#AB1529',  '#A31328',  '#9C1027',  '#940E26',  '#8C0C25',  
                        '#850923',  '#7D0722',  '#760421',  '#6E0220',  '#67001F'  ], # NSMB2017ish_20240130
    'hideki_RdBu_r': ['#053061', '#2166AC', '#4393C3', '#92C5DE', '#D1E5F0', '#FDDBC7', '#F4A582', '#D6604D', '#B2182B', '#67001F']
}

categorical_palettes = {
    'nature_pair': ['#DC0000', '#3C5488'], 
    'nature_cat': ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85']
}

for cm_name in new_colormaps_to_use:
    cm_to_reg = LinearSegmentedColormap.from_list(cm_name, new_colormaps_to_use[cm_name])
    if cm_name not in list(colormaps): 
        colormaps.register(cm_to_reg)
    else:
        colormaps.unregister(cm_name)
        colormaps.register(cm_to_reg)

def plot_colormap(cmap, vmin, vmax, figsize=(2,0.2), label_orient='tangent', output_path='colorscale.png'):
    lw = 1
    majorticklength = 2*lw
    draw_max = 256
    gradient = np.linspace(0, 1, draw_max)
    gradient = np.vstack((gradient, gradient))
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, draw_max)
    ax.imshow(gradient, aspect='auto', cmap=colormaps[cmap])
    
    if -vmin == vmax:
        scale_ticks = [vmin, 0, vmax]
        ax.xaxis.set_major_locator(FixedLocator([0, draw_max/2, draw_max]))
    else:
        scale_ticks = [vmin, vmax]
        ax.xaxis.set_major_locator(FixedLocator([0, draw_max]))
    ax.xaxis.set_major_formatter(FixedFormatter(scale_ticks))
    ax.yaxis.set_major_locator(NullLocator())
    ax.spines[['left', 'right']].set_linewidth(lw)
    ax.spines[['top', 'bottom']].set_visible(False)
    orient_dict = {'tangent':270, 'parallel':0}
    ax.tick_params(
                    axis='both', which='major', width=lw, labelsize=5*majorticklength, length=majorticklength,
                    pad=1.25*majorticklength,  rotation=orient_dict[label_orient], 
                )
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    return ax


# https://stackoverflow.com/questions/19270673/matplotlib-radius-in-polygon-edges-is-it-possible
class RoundedPolygon(PathPatch):
    def _init_(self, xy, pad, **kwargs):
        p = mpath.Path(*self._round(xy=xy, pad=pad))
        super()._init_(path=p, **kwargs)

    def _round(self, xy, pad):
        n = len(xy)

        for i in range(0, n):

            x0, x1, x2 = np.atleast_1d(xy[i - 1], xy[i], xy[(i + 1) % n])

            d01, d12 = x1 - x0, x2 - x1
            d01, d12 = d01 / np.linalg.norm(d01), d12 / np.linalg.norm(d12)

            x00 = x0 + pad * d01
            x01 = x1 - pad * d01
            x10 = x1 + pad * d12
            # x11 = x2 - pad * d12

            if i == 0:
                verts = [x00, x01, x1, x10]
            else:
                verts += [x01, x1, x10]
        codes = [mpath.Path.MOVETO] + n*[mpath.Path.LINETO, mpath.Path.CURVE3, mpath.Path.CURVE3]

        return np.atleast_1d(verts, codes)


## Tanizawa, et al., 2017 Fig 4a-ish
def hic_plot(
        df, centromeres=False, chroms=False, pombe=True, 
        aspect_ratios=False, vmin=False, vmax=False, res=10, # kb
        spinewidth=3, majorticklabelsize=32, majorticklength=24,
        minorticklabelsize=24, minorticklength=8,
        extra_hlines=False, extra_vlines=False,
        gridsep=0.02,
        figsize=(10,10),
        cmap='coolwarm', cbar=False, 
        spine_color='black',
        fontweight = 'regular',
        zoom_range=False, 
        triangle = False,
        output_path='output.png',
        colorscale=True, colorscale_label_orient='parallel'
    ):
    scale=1000
    if not chroms:
        chroms = sorted(list(set([x.split(':')[0]for x in df.columns])))
    if not isinstance(chroms, list):
        chroms = [chroms]
    chrom_lengths = [int(df.filter(regex=f'^{chrom}:').columns[-1].split(':')[-1]) for chrom in chroms]
    if not aspect_ratios:
        logger.debug("aspect_ratios not set")
        logger.debug(f"scale: {scale}")
        aspect_ratios = [int(round(int(df.filter(regex=f'^{chrom}:').columns[-1].split(':')[-1]), -3)/scale) for chrom in chroms]
        if centromeres:
            aspect_ratios = [int(max(aspect_ratios)/10)] + aspect_ratios
    logger.debug(aspect_ratios)
    logger.debug(f'aspect_ratios: {aspect_ratios}')
    fontsize = 20

    grid_num = bool(centromeres) + len(chroms)
    if triangle:
        figsize = (figsize[0], figsize[0]/2)
    fig, ax = plt.subplots(
        grid_num, grid_num, figsize=figsize, gridspec_kw={'width_ratios': aspect_ratios, 'height_ratios': aspect_ratios}
        )

    ## hi-c plot
    for i, chromX in enumerate(chroms):
        for j, chromY in enumerate(chroms):
            working_df = df.filter(regex=f'^{chromX}:', axis=0).filter(regex=f'^{chromY}:', axis=1)
            offset = 0
            if isinstance(ax, np.ndarray):
                working_ax = ax[i+bool(centromeres), j+bool(centromeres)]
            else:
                working_ax = ax
            if zoom_range:
                working_df.index = [int((int(x.split(':')[-1])+1) - res*1000/2) for x in working_df.index]
                working_df.columns = working_df.index
                zoomed = [x for x in working_df.index if x >= zoom_range[0] and x <= zoom_range[1]]
                offset = zoomed[0]
                working_df = working_df.loc[zoomed[0]:zoomed[-1], zoomed[0]:zoomed[-1]]
                working_ax.xaxis.set_major_locator(MultipleLocator(5e5/res/scale))
                working_ax.yaxis.set_major_locator(MultipleLocator(5e5/res/scale))
                working_ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos:f'{round((x*res*scale + offset)/1e6, 1)}'))
                working_ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos:f'{round((x*res*scale + offset)/1e6, 1)}'))
            
            if triangle:
                centromeres = False
                working_df = np.triu(working_df)
                working_df = rotate(working_df, angle=45, reshape=True)
                working_df = working_df[:int(np.ceil(len(working_df)*0.5))]
            if grid_num >1:
                # sns.heatmap(working_df, linewidths=0, vmin=vmin, vmax=vmax, cmap=cmap, ax=working_ax, cbar=cbar)
                # working_ax.set(xticks=[], yticks=[])
                working_ax.imshow(working_df, vmin=vmin, vmax=vmax, cmap=cmap)
                working_ax.set_axis_off()
                
            else:
                # sns.heatmap(working_df, linewidths=0, vmin=vmin, vmax=vmax, cmap=cmap, ax=ax, cbar=cbar)
                working_ax.imshow(working_df, vmin=vmin, vmax=vmax, cmap=cmap)
                working_ax.xaxis.set_ticks_position('top')
                working_ax.tick_params(
                    axis='both', which='minor', width=spinewidth, labelsize=majorticklabelsize, length=majorticklength,
                    pad=1.25*majorticklength,
                )
                working_ax.xaxis.set_major_locator(NullLocator())
                working_ax.yaxis.set_major_locator(NullLocator())
            working_ax.xaxis.set_minor_locator(NullLocator())
            working_ax.yaxis.set_minor_locator(NullLocator())
            if extra_hlines:
                working_ax.hlines(y=[(x - offset)/(working_df.index[-1]-offset)*working_ax.get_ylim()[0] for x in extra_hlines], xmin=0, xmax=working_ax.get_xlim()[1], linewidth=spinewidth, color=spine_color)
            if extra_vlines:
                working_ax.vlines(x=[(x - offset)/(working_df.index[-1]-offset)*working_ax.get_xlim()[1] for x in extra_vlines], ymin=0, ymax=working_ax.get_ylim()[0], linewidth=spinewidth, color=spine_color)

    if centromeres:
        if pombe:
            chrom_lengths = {
                'I': 5579133, 
                'II': 4539804,
                'III': 2452883
            }
            
            centromere_positions = {
                'I': (3753687, 3789421), 
                'II': (1602264, 1644747), 
                'III': (1070904, 1137003)
            }

        for chrom in chroms:
            chrom_lengths[chrom] = chrom_lengths[chrom]/res/scale
            centromere_positions[chrom] = (centromere_positions[chrom][0]/res/scale, centromere_positions[chrom][1]/res/scale)
        
        ax[0, 0].axis('off')
        fig_length = 2.5
        linewidth = round(fig_length/4, 2)
        margin_ratio = 0.01
        magma_edgecolor = '#771515'
        chrom_cmap = colors.LinearSegmentedColormap.from_list('white_to_purple', ['#b3b0d7', '#D1CDE5', '#d3cfe7', '#BFBAD9', '#A09CC6', '#6a68a7'])
        magma_cmap = colors.LinearSegmentedColormap.from_list('magma', ['#ed2926', '#aa1f23'])
        linear_grad = np.atleast_2d(np.linspace(0, 1, 256))
        x = np.linspace(-1, 1, 250)
        y = np.linspace(1, -1, 25)
        xArray, yArray = np.meshgrid(x, y)
        circ_grad = np.sqrt(xArray**2 + yArray**2)

        chrom_round_corner_pad_ratio = 0.4
        extra_element_width_ratio = 0.95
        chrom_start = 1
        
        for i, chrom in enumerate(chroms):
            centromere_start, centromere_end = centromere_positions[chrom]
            chrom_width = 10 ** (int(np.log10(chrom_lengths[chrom])-1)) * 1.2
            chrom_width_start, chrom_width_end = 0, chrom_width
            e_width = chrom_width * extra_element_width_ratio

            chrom_width_margin = (chrom_width_end - chrom_width_start) * margin_ratio
            chrom_length_margin = chrom_lengths[chrom] * margin_ratio * 2
            ax[i + 1, 0].set_xlim(chrom_width_start - chrom_width_margin, chrom_width_end + chrom_width_margin)
            ax[i + 1, 0].set_ylim(chrom_lengths[chrom] + chrom_length_margin, chrom_start - chrom_length_margin) 
            ax[0, i + 1].set_xlim(chrom_start - chrom_length_margin, chrom_lengths[chrom] + chrom_length_margin)
            ax[0, i + 1].set_ylim(chrom_width_start - chrom_width_margin, chrom_width_end + chrom_width_margin)
            
            ax[i + 1, 0].set(xticks=[], yticks=[])
            ax[i + 1, 0].spines[:].set_visible(False)
            ax[0, i + 1].set(xticks=[], yticks=[])
            ax[0, i + 1].spines[:].set_visible(False)

            ax[i + 1, 0].text(-1.2, 0.45, f'Chr {chrom}', transform = ax[i + 1, 0].transAxes, # `aspect_ratios[i + 1]/sum(aspect_ratios[1:])` no need for Arial 12
                                fontsize = fontsize, fontweight=fontweight, rotation = 90)
            ax[0, i + 1].text(0.45, 1.45, f'Chr {chrom}', transform = ax[0, i + 1].transAxes, 
                                fontsize = fontsize, fontweight=fontweight, rotation = 0)

            ### chromosome left arm
            left_arm_start = chrom_start
            left_arm_end = centromere_start
            left_chrom_arm_img_vert = ax[i + 1, 0].imshow(linear_grad, extent=[chrom_width_start, chrom_width_end, left_arm_start, left_arm_end], interpolation='nearest', aspect='auto', cmap=chrom_cmap, origin='lower')
            xy = np.array([(chrom_width_start, left_arm_start), (chrom_width_start, left_arm_end), (chrom_width_end, left_arm_end), (chrom_width_end, left_arm_start)])
            polygon = RoundedPolygon(   xy = xy, pad = chrom_round_corner_pad_ratio * chrom_width, 
                                        facecolor = '#ffffff00', edgecolor = '#68687d', linewidth = linewidth,
                                        zorder = 1) # None facecolor becomes blue
            ax[i + 1, 0].add_patch(polygon)
            left_chrom_arm_img_vert.set_clip_path(polygon)

            left_chrom_arm_img_hori = ax[0, i+1].imshow(linear_grad.T, extent=[left_arm_start, left_arm_end, chrom_width_start, chrom_width_end], interpolation='nearest', aspect='auto', cmap=chrom_cmap, origin='upper')
            polygon = RoundedPolygon(   xy = np.flip(xy), pad = chrom_round_corner_pad_ratio * chrom_width, 
                                        facecolor = '#ffffff00', edgecolor = '#68687d', linewidth = linewidth,
                                        zorder = 1) # None facecolor becomes blue
            ax[0, i + 1].add_patch(polygon)
            left_chrom_arm_img_hori.set_clip_path(polygon)


            ### chromosome right arm
            right_arm_start = centromere_end
            right_arm_end = chrom_lengths[chrom]
            right_chrom_arm_img_vert = ax[i + 1, 0].imshow(linear_grad, extent=[chrom_width_start, chrom_width_end, right_arm_start, right_arm_end], interpolation='nearest', aspect='auto', cmap=chrom_cmap, origin='lower')
            xy = np.array([(chrom_width_start, right_arm_start), (chrom_width_start, right_arm_end), (chrom_width_end, right_arm_end), (chrom_width_end, right_arm_start)])

            polygon = RoundedPolygon(   xy = xy, pad = chrom_round_corner_pad_ratio * chrom_width, 
                                        facecolor = '#ffffff00', edgecolor = '#68687d', linewidth = linewidth,
                                        zorder = 2) # None facecolor becomes blue
            ax[i + 1, 0].add_patch(polygon)
            right_chrom_arm_img_vert.set_clip_path(polygon)

            right_chrom_arm_img_hori = ax[0, i+1].imshow(linear_grad.T, extent=[right_arm_start, right_arm_end, chrom_width_start, chrom_width_end], interpolation='nearest', aspect='auto', cmap=chrom_cmap, origin='upper')
            polygon = RoundedPolygon(   xy = np.flip(xy), pad = chrom_round_corner_pad_ratio * chrom_width, 
                                        facecolor = '#ffffff00', edgecolor = '#68687d', linewidth = linewidth,
                                        zorder = 2) # None facecolor becomes blue
            ax[0, i + 1].add_patch(polygon)
            right_chrom_arm_img_hori.set_clip_path(polygon)
            
            ### centromere polygon
            
            xy = np.array([(chrom_width_start, centromere_start), (chrom_width_start, centromere_end), (chrom_width_end, centromere_end), (chrom_width_end, centromere_start)])
            centromere_img_vert = ax[i + 1, 0].imshow(circ_grad, extent=[chrom_width_start, chrom_width_end, centromere_start, centromere_end], interpolation='bicubic', aspect='auto', cmap=magma_cmap, origin='lower')

            centromere_polygon = RoundedPolygon(    xy = xy, pad = chrom_round_corner_pad_ratio * e_width * (centromere_end - centromere_start) / chrom_lengths[chrom], 
                                                    facecolor = '#aa1f2300', edgecolor = magma_edgecolor, linewidth = linewidth, 
                                                    zorder = 99)
            ax[i + 1, 0].add_patch(centromere_polygon)
            centromere_img_vert.set_clip_path(centromere_polygon)

            centromere_img_hori = ax[0, i + 1].imshow(circ_grad, extent=[centromere_start, centromere_end, chrom_width_start, chrom_width_end], interpolation='bicubic', aspect='auto', cmap=magma_cmap, origin='lower')

            centromere_polygon = RoundedPolygon(    xy = np.flip(xy), pad = chrom_round_corner_pad_ratio * e_width * (centromere_end - centromere_start) / chrom_lengths[chrom], 
                                                    facecolor = '#aa1f2300', edgecolor = magma_edgecolor, linewidth = linewidth, 
                                                    zorder = 99)
            ax[0, i + 1].add_patch(centromere_polygon)

            centromere_img_hori.set_clip_path(centromere_polygon)
        
            ### left telomere
            if chrom != 'III':
                telomere_shift = margin_ratio
            else:
                telomere_shift = -margin_ratio

            left_telomere_polygon = Ellipse(   
                xy = (0.5, 1 - telomere_shift), width = 0.95, height = 0.03, angle = 0, 
                edgecolor = magma_edgecolor, facecolor = '#aa1f23', linewidth = linewidth, transform = ax[i + 1, 0].transAxes,
                zorder = -2
            )
            ax[i + 1, 0].add_patch(left_telomere_polygon)
            left_telomere_polygon.set_clip_on(False)

            left_telomere_polygon = Ellipse(   
                xy = (telomere_shift, 0.5), width = 0.95, height = 0.03, angle = 990, 
                edgecolor = magma_edgecolor, facecolor='#aa1f23', linewidth = linewidth, transform = ax[0, i + 1].transAxes,
                zorder = -2
            )
            ax[0, i + 1].add_patch(left_telomere_polygon)
            left_telomere_polygon.set_clip_on(False)

            ### right telomere
            right_telomere_polygon = Ellipse(   
                xy = (0.5, telomere_shift), width = 0.95, height = 0.03, angle = 0, 
                edgecolor = magma_edgecolor, facecolor='#aa1f23', linewidth = linewidth, transform = ax[i + 1, 0].transAxes,
                zorder = -2
            )
            ax[i + 1, 0].add_patch(right_telomere_polygon)
            right_telomere_polygon.set_clip_on(False)

            right_telomere_polygon = Ellipse(   
                xy = (1 - telomere_shift, 0.5), width = 0.95, height = 0.03, angle = 90, 
                edgecolor = magma_edgecolor, facecolor='#aa1f23', linewidth = linewidth, transform = ax[0, i + 1].transAxes,
                zorder = -2
            )
            ax[0, i + 1].add_patch(right_telomere_polygon)
            right_telomere_polygon.set_clip_on(False)

            if pombe and chrom == 'III':
                ### pombe chrIII rDNA array left
                dark_gray = '#222222'
                rDNA_facecolor = '#e7d9e5'
                left_rDNA_polygon = Ellipse(   
                    xy = (0.5, 1 + telomere_shift), width = 0.95, height = 0.03, angle = 0, 
                    edgecolor = dark_gray, facecolor = rDNA_facecolor, linewidth = linewidth, 
                    transform = ax[i + 1, 0].transAxes,
                    zorder = -1
                )
                ax[i + 1, 0].add_patch(left_rDNA_polygon)
                left_rDNA_polygon.set_clip_on(False)

                left_rDNA_polygon = Ellipse(   
                    xy = (-telomere_shift, 0.5), width = 0.95, height = 0.03, angle = 990, 
                    edgecolor = dark_gray, facecolor = rDNA_facecolor, linewidth = linewidth, 
                    transform = ax[0, i + 1].transAxes,
                    zorder = -2
                )
                ax[0, i + 1].add_patch(left_rDNA_polygon)
                left_rDNA_polygon.set_clip_on(False)

                ### pombe chrIII rDNA array right
                right_rDNA_polygon = Ellipse(   
                    xy = (0.5, -telomere_shift), width = 0.95, height = 0.03, angle = 0, 
                    edgecolor = dark_gray, facecolor = rDNA_facecolor, linewidth = linewidth, 
                    transform = ax[i + 1, 0].transAxes,
                    zorder = -2
                )
                ax[i + 1, 0].add_patch(right_rDNA_polygon)
                right_rDNA_polygon.set_clip_on(False)

                right_rDNA_polygon = Ellipse(   
                    xy = (1 + telomere_shift, 0.5), width = 0.95, height = 0.03, angle = 90, 
                    edgecolor = dark_gray, facecolor = rDNA_facecolor, linewidth = linewidth, 
                    transform = ax[0, i + 1].transAxes,
                    zorder = -2
                )
                ax[0, i + 1].add_patch(right_rDNA_polygon)
                right_rDNA_polygon.set_clip_on(False)

    plt.subplots_adjust(wspace=gridsep, hspace=gridsep)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0)
    if colorscale:
        logger.info('Drawing color scale.')
        plot_colormap(cmap=cmap, vmin=vmin, vmax=vmax, label_orient=colorscale_label_orient, output_path=f"{'-'.join(output_path.rsplit('.', 1))}-colorscale.svg")
    plt.close()
    return ax

def get_dist_df(df, distance_ranges, calc_gradients=False, chroms=False, res=5, verbose=False):
    dist_scores = defaultdict(list)
    if not chroms:
        chroms = sorted(list(set([x.split(':')[0]for x in df.columns])))

    for dist in distance_ranges:
        for chrom in chroms:
            working_df = df.filter(regex=f'^{chrom}:', axis=0).filter(regex=f'^{chrom}:', axis=1)
            working_df.index = [int((int(x.split(':')[-1])+1) - res*1000/2) for x in working_df.index]
            working_df.columns = working_df.index
            dist_scores[dist] += [working_df.loc[x, x + dist*1000] for x in working_df.index if x + dist*1000 <= working_df.index.max()]
            logger.debug(dist, chrom)
            logger.debug(working_df.index[-1])
            logger.debug(dist_scores[dist][:50])

        dist_scores[dist] = np.mean(dist_scores[dist])
        logger.debug(dist, dist_scores[dist])
    scores_by_dist_df = pd.DataFrame.from_dict(dist_scores, orient='index')
    if calc_gradients:
        logger.info(f'gradient: {np.gradient(scores_by_dist_df[0])}')
    scores_by_dist_df = scores_by_dist_df.replace(-np.inf, 0).replace(np.inf, 0)
    return scores_by_dist_df

def log_spaced_ranges(start=10, end=4000, n_points=200, step_multiple=5):
    """
    Generate a monotonically increasing list of distances.
    Intervals grow on a log10 scale, and each step is rounded
    to a multiple of `step_multiple`.
    """
    # log-spaced points
    distances = np.logspace(
        np.log10(start),
        np.log10(end),
        n_points
    )

    # round each value to nearest multiple of step_multiple
    distances = np.round(distances / step_multiple) * step_multiple

    # ensure uniqueness + integer conversion
    distances = np.unique(distances.astype(int))

    return distances.tolist()

## Tanizawa, et al., 2017 Fig 4b-ish
def plot_dist_curves(   dfs, labels=False, distance_ranges=False, chroms=False, res=5,
                        sum_1=True, prob=False, spline_factor=0, #lowess_smooth=False,
                        domain_type = 'condensin',
                        vbounds=[75, 800], rect_colors=['#bce2e8', '#eebbcb', '#c8c2c6'], 
                        vmin=False, vmax=False, colors=['red', 'gray'], 
                        figsize=(4,4),
                        linewidth=False, 
                        spinewidth=False, majorticklabelsize=False, majorticklength=False,
                        minorticklabelsize=False, minorticklength=False,
                        fontweight='bold',
                        legend_fontsize=False, legend_loc=False, legend_bbox_to_anchor=False,
                        output_path='output.png', subfolder='',
                        svg=True, output_array=False
                    ):
    if not linewidth:
        linewidth = 4
    font_size = 12
    if not spinewidth:
        spinewidth = round(linewidth/2)
    if not majorticklabelsize:
        majorticklabelsize = font_size
    if not majorticklength:
        majorticklength = linewidth * 3
    if not minorticklabelsize:
        minorticklabelsize = font_size
    if not minorticklength:
        minorticklength = linewidth
    if not legend_fontsize:
        legend_fontsize = font_size - 2
    if not legend_loc:
        legend_loc = 'best'
    if not legend_bbox_to_anchor:
        legend_bbox_to_anchor = (0,0)

    # score_type = ['Score', 'Score (Sum to 1)', 'Probability']

    if not distance_ranges:
        # distance_ranges = list(range(10, 100, 10)) + list(range(100, 1000, 50)) + list(range(1000, 4000, 250))
        # distance_ranges = list(range(10, 1000, 10)) + list(range(1000, 4000, 100))
        # distance_ranges = list(range(10, 500, 10)) + list(range(500, 2000, 50)) + list(range(2000, 4000, 100))
        # distance_ranges = list(range(10, 100, 10)) + list(range(100, 2000, 50)) + list(range(2000, 4000, 100))
        # distance_ranges = list(range(10, 2000, 10)) + list(range(2000, 4000, 100))
        # distance_ranges = list(range(10, 4000, 10)) # dip at 100 kb
        distance_ranges = log_spaced_ranges(start=10, end=4000, n_points=200, step_multiple=10)

    if not chroms:
        chroms = sorted(list(set([x.split(':')[0]for x in dfs[0].columns])))
        
    score_by_dist_dfs = []
    for i, df in enumerate(dfs):
        if prob:
            pickle_name = f'out/{datestamp()}_hic_figures/{subfolder}/{labels[i]}_{res}kb_dist_df_prob.pickle'
        elif sum_1:
            pickle_name = f'out/{datestamp()}_hic_figures/{subfolder}/{labels[i]}_{res}kb_dist_df_sum1.pickle'
            default_ymax = 1e-4
            default_ymin = 1e-8
        else:
            pickle_name = f'out/{datestamp()}_hic_figures/{subfolder}/{labels[i]}_{res}kb_dist_df.pickle'
            default_ymax = 100
            default_ymin = 0.0001
        if not os.path.isfile(pickle_name):
            dist_df = get_dist_df(df, distance_ranges, chroms=chroms, res=res)
            dist_df.to_pickle(pickle_name)
        else:
            logger.info(f'Reading data from pickle {pickle_name}')
            dist_df = pd.read_pickle(pickle_name)

        score_by_dist_dfs.append(dist_df)
        del dist_df

    fig, ax = plt.subplots(figsize=figsize)
    plt.xscale('log')
    plt.yscale('log')

    if vbounds[0] != 0:
        vbounds = [10] + vbounds
    vbounds.append(score_by_dist_dfs[0].index.max())
    ax.set_xlim(10, vbounds[-1])
    y_label = "Contact Probability" if prob else "Contact Score"
    
    for i in range(1, len(vbounds)):
        score_by_dist_dfs[0][0] = score_by_dist_dfs[0][0].replace(np.inf, np.nan).replace(-np.inf, np.nan)
        # y_min = min([np.nanmin(x[0]) for x in score_by_dist_dfs] + [-1])
        y_min = max(min([np.nanmin(x[0]) for x in score_by_dist_dfs] + [0.02]), default_ymin, vmin)
        logger.debug(f'y_min: {y_min}')
        # comment 2025-11-22 for pmc4 paper exfig6
        y_max = max([np.nanmax(x[0]) for x in score_by_dist_dfs] + [default_ymax, y_min*10])
        ax.add_patch(Rectangle((vbounds[i-1], y_min), vbounds[i] - vbounds[i-1], y_max - y_min, color=rect_colors[i-1]))
        logger.debug(f'y_max: {y_max}')  

    score_by_dist_dfs = [score_by_dist_dfs[-1]] + score_by_dist_dfs[:-1]
    labels = [labels[-1]] + labels[:-1]
    colors = [colors[-1]] + colors[:-1]
    for i, data in enumerate(list(score_by_dist_dfs)):
        if output_array:
            data.to_csv(f'{output_path.rsplit(".", 1)[0]}{labels[i]}.txt', sep='\t', header=None)
        if len(score_by_dist_dfs)==1:
            color = colors[0]
        else:
            color = colors[i%len(colors)]

        line_x_to_plot = np.array(data.index.values)
        line_y_to_plot = np.array(data[0])
        line_y_to_plot[line_y_to_plot == 0] = 1e-10 # to avoid log(0)

        if spline_factor > 0:
            # Drop zero or negative entries (log scale cannot handle them)
            mask = (line_x_to_plot > 0) & (line_y_to_plot > 0)
            line_x_to_plot = line_x_to_plot[mask]
            line_y_to_plot = line_y_to_plot[mask]

            # --- Smoothing spline in log–log space ---
            logx = np.log10(line_x_to_plot)
            logy = np.log10(line_y_to_plot)

            # Smoothing strength (tune this: larger s = smoother)
            s = len(logx) * 0.005
            spl = UnivariateSpline(logx, logy, s=s)

            # High-resolution X axis (still in log space)
            logx_new = np.linspace(logx.min(), logx.max(), 1000)
            logy_smooth = spl(logx_new)

            x_new = 10**logx_new
            y_smooth = 10**logy_smooth

            line_x_to_plot = x_new
            line_y_to_plot = y_smooth

        if labels:
            ax.plot(line_x_to_plot, line_y_to_plot, color=color, linewidth=0.5*linewidth, label=labels[i])
        else:
            ax.plot(line_x_to_plot, line_y_to_plot, color=color, linewidth=0.5*linewidth)
        

        x = np.log10(line_x_to_plot)
        y = np.log10(line_y_to_plot)
        dy = np.gradient(y)
        ddy = np.gradient(dy)
        
        if domain_type == 'condensin':
            kink_x_min = 200
            kink_x_max = 1000
        elif domain_type == 'cohesin':
            kink_x_min = 0
            kink_x_max = 500
            rect_colors=['#eebbcb', '#bce2e8', '#c8c2c6']
        peak_locs, peak_heights = find_peaks(ddy, height=0.005)
        if domain_type:
            for j, peak_loc in enumerate(peak_locs): 

                idx = int(peak_loc)           
                x_bp = line_x_to_plot[idx]

                if x_bp < kink_x_min or x_bp > kink_x_max:
                    continue

                x_to_plot = (x[idx] - np.log10(ax.get_xlim()[0])) / \
                            (np.log10(ax.get_xlim()[1]) - np.log10(ax.get_xlim()[0]))

                y_to_plot = (y[idx] - np.log10(y_min)) / \
                            (np.log10(y_max) - np.log10(y_min))
                
                logger.debug(x_to_plot, y_to_plot)
                circle_r = 0.01
                circle = plt.Circle((x_to_plot, y_to_plot), 
                                    circle_r, 
                                    transform = ax.transAxes,
                                    edgecolor = color, 
                                    facecolor = '#ffffffcc', 
                                    zorder=9999)
                ax.add_artist(circle)
                break


    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['bottom', 'left']].set_linewidth(spinewidth)
    
    ax.xaxis.set_major_locator(FixedLocator([0.001, 10, 100, 1000]))
    ax.xaxis.set_major_formatter(FixedFormatter(['', '10 kb', '100 kb', '1 Mb']))
    ax.xaxis.set_minor_locator(FixedLocator([20, 50, 200, 500, 2000, 5000]))
    ax.xaxis.set_minor_formatter(FixedFormatter([2, 5, 2, 5, 2, 5]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: int(x) if x >= 1 else '{:.1g}'.format(x)))
    logger.debug(f'ax.get_yticks(): {ax.get_yticks()}')
    ax.yaxis.set_minor_locator(NullLocator())
    logger.debug(f'ax.get_ylim(): {ax.get_ylim()}')
    ax.set_ylim(max(ax.get_ylim()[0], y_min), y_max) 
    ax.tick_params(
        axis='both', which='major', width=spinewidth, labelsize=majorticklabelsize, length=majorticklength,
        pad=1.25*majorticklength,
    )
    ax.tick_params(
        axis='x', which='minor', width=spinewidth, labelsize=minorticklabelsize, length=minorticklength,
        pad=1.25*minorticklength,
    )
    ax.set_xlabel('Distance', fontsize=majorticklabelsize, labelpad=majorticklabelsize, fontweight=fontweight)
    
    ax.set_ylabel(y_label, fontsize=majorticklabelsize, labelpad=majorticklabelsize, fontweight=fontweight)

    logger.debug(f'legend_loc: {legend_loc}')
    # if labels:
    #     ax.legend(
    #         ax.get_legend_handles_labels()[1], fontsize=legend_fontsize,
    #         loc=legend_loc, bbox_to_anchor=legend_bbox_to_anchor,
    #     )
    plt.legend()
    logger.info(output_path)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")


## Tanizawa, et al., 2017 Fig 4c-ish
def plot_dist_compare_curve(   df1, df2, distance_ranges, method="log2diff", sum_1=False, chroms=False,
                                        vbounds=[75, 800], rect_colors=['#eebbcb', '#bce2e8', '#c8c2c6'], 
                                        vmin=False, vmax=False, color='red', cbar=False, linewidth=8,
                                        spinewidth=2, majorticklabelsize=32, majorticklength=24,
                                        minorticklabelsize=24, minorticklength=8,
                                        figsize=(8,8),
                                        output_path='output.png', svg=True,
                                        verbose=False, **kwargs):
    if not chroms:
        chroms = sorted(list(set([x.split(':')[0]for x in df1.columns])))
    if method not in ['log2diff', 'diff', 'ratio']:
        logger.error(f"Method {method} not recognized. Please use 'log2diff', 'diff', or 'ratio'.")
        return
    score_type = ['Score', 'Score (Sum to 1)'][sum_1]
    if method == 'log2diff':
        dist_comp_df = (get_dist_df(df2, distance_ranges, chroms=chroms, verbose=verbose) / get_dist_df(df1, distance_ranges, chroms=chroms)).apply(np.log2)
        y_label = f'Contact {score_type} Log2 ratio'
    if method == 'diff':
        dist_comp_df = (get_dist_df(df2, distance_ranges, chroms=chroms, verbose=verbose) - get_dist_df(df1, distance_ranges, chroms=chroms))
        y_label = f'Contact {score_type} Difference'
    if method == 'ratio':
        dist_comp_df = (get_dist_df(df2, distance_ranges, chroms=chroms, verbose=verbose) / get_dist_df(df1, distance_ranges, chroms=chroms))
        y_label = f'Contact {score_type} Ratio'
        
    fig, ax = plt.subplots(figsize=figsize)
    plt.plot(dist_comp_df, color=color, linewidth=linewidth)
    if vbounds[0] != 0:
        vbounds = [10] + vbounds
    vbounds.append(dist_comp_df.index.max())

    y_min = min(vmin, np.floor(np.nanmin(dist_comp_df)))
    y_max = max(vmax, np.ceil(np.nanmax(dist_comp_df)))
    if method == 'diff':
        y_max = max(abs(y_min), y_max)
        y_min = -y_max
        ax.hlines(0, vbounds[0], vbounds[-1])
    elif method == 'ratio':
        if y_min != 0:
            y_max = max(1/y_min, y_max)
            y_min = 0
        ax.hlines(1, vbounds[0], vbounds[-1])
    elif method == 'log2diff':
        ax.hlines(0, vbounds[0], vbounds[-1])
    if math.isinf(y_max):
        y_max = 10
    if math.isinf(y_min):
        y_min = -y_max
    logger.info(f'vmin, vmax: {vmin, vmax}')
    logger.info(f'y_min, y_max: {y_min, y_max}')

    for i in range(1, len(vbounds)):
        dist_comp_df[0].replace(np.inf, np.nan, inplace=True)
        dist_comp_df[0].replace(-np.inf, np.nan, inplace=True)
        ax.add_patch(Rectangle((vbounds[i-1], y_min), vbounds[i] - vbounds[i-1], y_max - y_min, color=rect_colors[i-1]))
    plt.xscale('log')
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['bottom', 'left']].set_linewidth(spinewidth)
    ax.set_xlim(10, vbounds[-1])
    ax.xaxis.set_major_locator(FixedLocator([0.001, 10, 100, 1000]))
    ax.xaxis.set_major_formatter(FixedFormatter(['', '10 kb', '100 kb', '1 Mb']))
    ax.xaxis.set_minor_locator(FixedLocator([20, 50, 200, 500, 2000, 5000]))
    ax.xaxis.set_minor_formatter(FixedFormatter([2, 5, 2, 5, 2, 5]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: int(x) if x >= 1 else '{:.1g}'.format(x)))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylim(y_min, y_max)
    ax.tick_params(
        axis='both', which='major', width=spinewidth, labelsize=majorticklabelsize, length=majorticklength,
        pad=1.25*majorticklength,
    )
    ax.tick_params(
        axis='x', which='minor', width=spinewidth, labelsize=minorticklabelsize, length=minorticklength,
        pad=1.25*minorticklength,
    )
    ax.set_xlabel('Distance', fontsize=majorticklabelsize, labelpad=0.5*majorticklabelsize)
    ax.set_ylabel(y_label, fontsize=majorticklabelsize, labelpad=0.5*majorticklabelsize)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")



## Tanizawa, et al., 2017 Fig 6g
def scatter_diff_scores(df1_path, df2_path, 
                        ctrl1_path=False, ctrl2_path=False,
                        df1_id=False, df2_id=False,
                        ctrl1_id=False, ctrl2_id=False,
                        res=10, min_dist=False, max_dist=False,
                        chroms=False, domain_type='condensin',
                        target_bed_path=False, dotsize=1,
                        figsize=(10,10), vmin=False, vmax=False,
                        basic_color='#66666688', highlight_color='#B82B6488',
                        linewidth=False, spinewidth=False, 
                        majorticklabelsize=False, majorticklength=False,
                        minorticklabelsize=False, minorticklength=False,
                        legend_fontsize=False, legend_loc=False, legend_bbox_to_anchor=False,
                        random_seed = 100, sample_size=100, equal_var=False, output_pval=False,
                        stat_test='ttest_ind', # 'mannwhitneyu',
                        output_path='output.png', svg=False, rasterized=True,
                        ):
    if not linewidth:
        linewidth = figsize[0]
    if not spinewidth:
        spinewidth = int(linewidth/4)
    if not majorticklabelsize:
        majorticklabelsize = linewidth * 4
    if not majorticklength:
        majorticklength = linewidth * 3
    if not minorticklabelsize:
        minorticklabelsize = linewidth * 3
    if not minorticklength:
        minorticklength = linewidth
    if not legend_fontsize:
        legend_fontsize = linewidth * 2
    if not legend_loc:
        legend_loc = 'best'
    if not legend_bbox_to_anchor:
        legend_bbox_to_anchor = (0,0)

    if domain_type == 'condensin':
        if not target_bed_path:
            target_bed_path = importlib.resources.files('hiclaire-viz').joinpath('data/codensin_domain_definition.bed')
        if not min_dist:
            min_dist = 75
        if not max_dist:
            max_dist = 800
    elif domain_type == 'cohesin':
        if not target_bed_path:
            target_bed_path = importlib.resources.files('hiclaire-viz').joinpath('data/cohesin_domain_definition.bed')
        if not min_dist:
            min_dist = 10
        if not max_dist:
            max_dist = 75
    elif not target_bed_path:
        logger.error("Please provide a BED file to specify the domain regions.")
        return
    target_domains = pd.read_csv(target_bed_path, sep='\t')
    domain_by_iv_dict = {}

    if not chroms:
        chroms = target_domains['chrom'].unique()
    ivtrees = {}
    for chrom in target_domains['chrom'].unique():
        ivtrees[chrom] = IntervalTree()
    target_domains.apply(lambda x: ivtrees[x['chrom']].addi(x['start'], x['end'], x['domain_id']), axis=1)

    for chrom in target_domains['chrom'].unique():
        final_start = target_domains[target_domains['chrom']==chrom].iloc[-1]['start']
        for i in range(0, final_start + res*1000, res*1000):
            try:
                domain_by_iv_dict[f'{chrom}:{i}:{i + res*1000 - 1}'] = int(list(ivtrees[chrom][i:i + res*1000 - 1])[0][2])
            except:
                pass

    ivs_by_domain_dict = defaultdict(list)
    for i in range(int(target_domains.iloc[-1]["domain_id"])):
        for k in domain_by_iv_dict:
            if domain_by_iv_dict[k] == i:
                ivs_by_domain_dict[i].append(k)
    logger.debug(f'ivs_by_domain_dict[1]: {ivs_by_domain_dict[1][:10]}')

    def calc_scatter_data(  working_df1_path, working_df2_path, 
                            working_min_dist=False, working_max_dist=False, sum_1=True):
        working_log2_diffs = np.array([])
        working_avg_scores = np.array([])
        working_target_log2_diffs = np.array([])
        working_target_avg_scores = np.array([])
        for chrom in chroms:
            df1 = matrix_to_df(working_df1_path, sum_1=sum_1)
            df1 = df1.filter(regex=f'^{chrom}:', axis=0).filter(regex=f'^{chrom}:', axis=1)
            df2 = matrix_to_df(working_df2_path, sum_1=sum_1)
            df2 = df2.filter(regex=f'^{chrom}:', axis=0).filter(regex=f'^{chrom}:', axis=1)

            logger.debug(f'len(df1): {len(df1)}')
            logger.debug(f'len(df2): {len(df2)}')
            log2_diff_df = np.log2(df1/df2)
            working_log2_diffs_tmp = log2_diff_df.to_numpy().flatten()

            logger.debug(f'len(working_log2_diffs_tmp) before nan filter: {len(working_log2_diffs_tmp)}')
            avg_df = (df1 + df2)/2
            working_avg_scores_tmp = avg_df.to_numpy().flatten()
            working_avg_scores_tmp = working_avg_scores_tmp[np.isfinite(working_log2_diffs_tmp)]
            working_log2_diffs_tmp = working_log2_diffs_tmp[np.isfinite(working_log2_diffs_tmp)]
            logger.debug(f'len(working_log2_diffs_tmp) after nan filter: {len(working_log2_diffs_tmp)}')
            logger.debug(f'len(working_avg_scores_tmp) after nan filter: {len(working_avg_scores_tmp)}')

            working_log2_diffs = np.append(working_log2_diffs, working_log2_diffs_tmp)
            working_avg_scores = np.append(working_avg_scores, working_avg_scores_tmp)
            
            dist_filtered_log2_diff_df = get_dist_filtered_df(log2_diff_df, working_min_dist, working_max_dist)
            dist_filtered_avg_df = get_dist_filtered_df(avg_df, working_min_dist, working_max_dist)
            logger.debug(f'dist_filtered_log2_diff_df.columns: {dist_filtered_log2_diff_df.columns}')
            logger.debug(f'dist_filtered_log2_diff_df.head(): {dist_filtered_log2_diff_df.head()}')

            for i in range(int(target_domains[target_domains['chrom'].isin(chroms)].iloc[-1]["domain_id"])+1):
                working_target_log2_diffs_tmp_df = dist_filtered_log2_diff_df.filter(items=ivs_by_domain_dict[i], axis=0).filter(items=ivs_by_domain_dict[i], axis=1)
                working_target_avg_scores_tmp_df = dist_filtered_avg_df.filter(items=ivs_by_domain_dict[i], axis=0).filter(items=ivs_by_domain_dict[i], axis=1)
                working_target_log2_diffs_tmp = working_target_log2_diffs_tmp_df.to_numpy().flatten()
                working_target_avg_scores_tmp = working_target_avg_scores_tmp_df.to_numpy().flatten()
                working_target_avg_scores_tmp = working_target_avg_scores_tmp[np.isfinite(working_target_log2_diffs_tmp)]
                working_target_log2_diffs_tmp = working_target_log2_diffs_tmp[np.isfinite(working_target_log2_diffs_tmp)]
                working_target_avg_scores = np.append(working_target_avg_scores, working_target_avg_scores_tmp)
                working_target_log2_diffs = np.append(working_target_log2_diffs, working_target_log2_diffs_tmp)
            
            np.random.seed(random_seed)
            row_id_sampled = np.random.randint(0, len(working_target_log2_diffs)-1)
            working_target_df = pd.DataFrame({
                "target_log2_diffs": working_target_log2_diffs, 
                "target_avg_scores": working_target_avg_scores,
                # "target_log2_to_avg": working_target_log2_diffs / working_target_avg_scores,
                "target_log10_avg": [int(round(np.log10(x),0)) for x in working_target_avg_scores]
            })
            working_bg_df = pd.DataFrame({
                "log2_diffs": working_log2_diffs[row_id_sampled],
                "avg_scores": working_avg_scores[row_id_sampled],
                "log10_avg": [int(round(np.log10(x),0)) for x in working_avg_scores]
            })
            logging.debug("working_bg_df calculated")
            working_bg_df = working_bg_df.groupby("log10_avg").agg({"log2_diffs": np.median})
            working_norm_target_df = working_target_df.merge(working_bg_df, left_on="target_log10_avg", right_on="log10_avg", how="inner")
            working_norm_target_df["norm_target_log2_diffs"] = working_norm_target_df["target_log2_diffs"] / working_norm_target_df["log2_diffs"]
            working_target_to_compare = working_norm_target_df["norm_target_log2_diffs"].dropna().tolist()
            # solve random choice and replace
        return working_log2_diffs, working_avg_scores, \
            working_target_log2_diffs, working_target_avg_scores, \
            working_target_to_compare

    log2_diffs, avg_scores, target_log2_diffs, target_avg_scores, target_to_compare = calc_scatter_data(df1_path, df2_path, min_dist, max_dist)
    ctrl_log2_diffs, ctrl_avg_scores, ctrl_target_log2_diffs, ctrl_target_avg_scores, ctrl_target_to_compare = calc_scatter_data(ctrl1_path, ctrl2_path, min_dist, max_dist)

    fig, ax = plt.subplots(figsize=figsize)
    avg_scores_scatter = ax.scatter(avg_scores, log2_diffs, 
                                    color=basic_color, s=dotsize)
    target_avg_scores_scatter = ax.scatter(target_avg_scores, target_log2_diffs, 
                                        color=highlight_color, s=dotsize)
    logger.debug(f"len(target_log2_diffs): {len(target_log2_diffs)}")
    logger.debug(f"len(target_avg_scores): {len(target_avg_scores)}")
    logger.debug(f"vmin: {vmin}")
    logger.debug(f"vmax: {vmax}")
    ax.set_ylim(vmin, vmax)
    if rasterized:
        avg_scores_scatter.set_rasterized(True)
        target_avg_scores_scatter.set_rasterized(True)
    plt.xscale('log')
    ax.tick_params(
        axis = 'both', which = 'major', width = spinewidth, 
        labelsize = majorticklabelsize, length = majorticklength,
        pad = 0.5 * majorticklength,
    )
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.1g}'.format(x)))  
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel('Average score', fontsize = majorticklabelsize, labelpad = 0.5 * majorticklabelsize)
    ax.set_ylabel(f'$log_{2}({df1_id}/{df2_id})$', fontsize = majorticklabelsize, labelpad = 0.5 * majorticklabelsize)

    if output_pval:
        np.random.seed(random_seed)
        pvals = []
        for bootstrap_counter in range(1000):
            logger.debug(f"len(target_log2_diffs): {len(target_log2_diffs)}")
            logger.debug(f"len(ctrl_target_log2_diffs): {len(ctrl_target_log2_diffs)}")

            logger.info(f"stat_test: {stat_test}")
            if stat_test == "ttest_ind":
                pvals.append(ttest_ind(np.random.choice(target_log2_diffs, sample_size, replace=False), 
                                        np.random.choice(ctrl_target_log2_diffs, sample_size, replace=False), 
                                        equal_var=equal_var).pvalue)
            elif stat_test == "spearmanr":
                pvals.append(spearmanr(np.random.choice(target_log2_diffs, sample_size, replace=False),
                                        np.random.choice(ctrl_target_log2_diffs, sample_size, replace=False)).pvalue)
        final_pval = np.median(pvals)
        with open(f'{output_path.rsplit(".", 1)[0]}_pvals.txt', 'w') as f:
            f.write('\n'.join([str(x) for x in pvals]))
        logger.info(f'Pval = {final_pval}')
        with open(f'{output_path.rsplit(".", 1)[0]}_pval.txt', 'w') as f:
            f.write(str(final_pval))
        ax.text(0.6, 0.2, f'P = {format_number(final_pval)}', transform=ax.transAxes, fontsize=int(majorticklabelsize/2))
        ax.text(0.6, 0.1, f'vs $log_{2}({ctrl1_id}/{ctrl2_id})$', transform=ax.transAxes, fontsize=int(majorticklabelsize/2))
    
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['bottom', 'left']].set_linewidth(spinewidth)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if svg:
        fig.savefig(f'{output_path.rsplit(".", 1)[0]}.svg', bbox_inches="tight")
    
    return

# https://stackoverflow.com/questions/42637579/how-to-compute-and-plot-a-lowess-curve-in-python
def lowess(x, y, f=1./3.):
    """
    Basic LOWESS smoother with uncertainty. 
    Note:
        - Not robust (so no iteration) and
            only normally distributed errors. 
        - No higher order polynomials d=1 
            so linear smoother.
    """
    # get some paras
    xwidth = f*(x.max()-x.min()) # effective width after reduction factor
    N = len(x) # number of obs
    # Don't assume the data is sorted
    order = np.argsort(x)
    # storage
    y_sm = np.zeros_like(y)
    y_stderr = np.zeros_like(y)
    # define the weigthing function -- clipping too!
    tricube = lambda d : np.clip((1- np.abs(d)**3)**3, 0, 1)
    # run the regression for each observation i
    for i in range(N):
        dist = np.abs((x[order][i]-x[order]))/xwidth
        w = tricube(dist)
        # form linear system with the weights
        A = np.stack([w, x[order]*w]).T
        b = w * y[order]
        ATA = A.T.dot(A)
        ATb = A.T.dot(b)
        # solve the syste
        sol = np.linalg.solve(ATA, ATb)
        # predict for the observation only
        yest = A[i].dot(sol)# equiv of A.dot(yest) just for k
        place = order[i]
        y_sm[place]=yest 
        sigma2 = (np.sum((A.dot(sol) -y [order])**2)/N )
        # Calculate the standard error
        y_stderr[place] = np.sqrt(sigma2 * 
                                A[i].dot(np.linalg.inv(ATA)
                                                    ).dot(A[i]))
    return y_sm, y_stderr

custom_round = lambda x, b: b * round(x/b)

def natural_sort_matrix_key(item):
    parts = item.split(":")
    first_part = parts[0]
    middle_number = int(parts[1]) if parts[1].isdigit() else float('inf')
    return (first_part, middle_number)

def natural_sort_paths(file_paths, file_name_only=True):
    dirs = {}
    file_paths_to_sort = []
    output_paths = []
    if file_name_only:
        for x in file_paths:
            to_sort = x.rsplit('/', 1)[-1]
            file_paths_to_sort.append(to_sort)
            if '/' in x:
                dirs[to_sort] = x.rsplit('/', 1)[0]
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    file_paths_to_sort = sorted(file_paths_to_sort, key=alphanum_key)
    
    if len(dirs) > 0:
        for file_path in file_paths_to_sort:
            output_paths.append(f"{dirs[file_path]}/{file_path}")
    else:
        for file_path in file_paths_to_sort:
            output_paths.append(file_path)
    return output_paths

def format_number(num):
    if abs(num) < 0.005:
        return f"{num:.2e}"
    else:
        return f"{num:.2f}"


def filter_by_dist(value, row_name, column_name, min_dist=False, max_dist=False):
        dist = abs(int(row_name.split(':')[1]) - int(column_name.split(':')[1]))
        return dist >= min_dist and dist < max_dist


def get_dist_filtered_df(df, min_dist=False, max_dist=False):
    if not min_dist and not max_dist:
        return df
    else:
        return pd.DataFrame({
                        col: [df.at[idx, col] if filter_by_dist(df.at[idx, col], idx, col, min_dist=min_dist, max_dist=max_dist) else np.nan for idx in df.index]
                        for col in df.columns
                    }).set_index(df.index)

def plot_multi_fig( to_draw, project_dir=None, sample_dict=None, centromeres=True, triangle=False,
                    distance_ranges=False, chroms=False, domain_type=False, extra_dir="",
                    sum_1=False, density_rescaling=False, log_score=False, vmin=False, vmax=False, vmax_percentile=80,
                    figsize=(10,10), gridspec=0.05, legend_loc=False, legend_bbox_to_anchor=False, 
                    svg=False, output_array=False, output_project_dir=None, intrachrom=False,
                    output_pval=True, dotsize=1, basic_color='#66666688', highlight_color='#B82B6488',
                    **kwargs):
    logger.info(to_draw)
    inline_chroms = chroms
    for i, row in to_draw.iterrows():
        chroms = inline_chroms
        res = get_kb_length(row['res'])
        sum_1 = True if 'sum_1' in row and row['sum_1'] in [1, 'True'] else False
        svg = True if 'svg' in row and row['svg'] in [1, 'True'] else False
        if not chroms:
            if not isinstance(row['chr'], (str, int)):
                chroms = False
            else:
                chroms = row['chr']

        if not intrachrom:
            chrom_matrix = "ALL"
        else:
            chrom_matrix =  chroms
        if svg:
            extension = 'svg'
        else:
            extension = 'png'
        
        domain_type = row['domain_type'] if 'domain_type' in row else domain_type
        stat_test = row['stat_test'] if 'stat_test' in row else 'ttest_ind'

        if 'hic' in row['type']:
            try:
                if 'triangle' in row['type']:
                    triangle = True
                if row['type'] in ['hic', 'hic_triangle']:
                    logger.info(f'Drawing HiC plot for {row["mat1"]}')
                    logger.info(row['mat1'])
                    logger.info(sample_dict[int(row['mat1'])])
                    mat1_path = get_matrix_path(project_dir, sample_dict[int(row['mat1'])], res, row['norm'], extra_dir, chrom_matrix)
                    df = matrix_to_df(mat1_path, sum_1=sum_1)
                elif 'hic_diff' in row['type']:
                    logger.info(f'Drawing HiC diff plot for {row["mat1"]} and {row["mat2"]}', extra_dir, chrom_matrix)
                    if '-' not in str(row['mat1']):
                        mat1_path = get_matrix_path(project_dir, sample_dict[int(row['mat1'])], res, row['norm'], extra_dir, chrom_matrix)
                        df1 = matrix_to_df(mat1_path, sum_1=sum_1)
                        mat2_path = get_matrix_path(project_dir, sample_dict[int(row['mat2'])], res, row['norm'], extra_dir, chrom_matrix)
                        df2 = matrix_to_df(mat2_path, sum_1=sum_1)
                    else:
                        matA, matB = [int(x) for x  in row['mat1'].split('-')]
                        matA_path = get_matrix_path(project_dir, sample_dict[matA], res, row['norm'], extra_dir, chrom_matrix)
                        matB_path = get_matrix_path(project_dir, sample_dict[matB], res, row['norm'], extra_dir, chrom_matrix)
                        dfA = matrix_to_df(matA_path, sum_1=sum_1)
                        dfB = matrix_to_df(matB_path, sum_1=sum_1)
                        df1 = get_diff_matrix(dfA, dfB)
                        df1 = df1.reindex(sorted(df1.columns, key=natural_sort_matrix_key), axis=0).reindex(sorted(df1.columns, key=natural_sort_matrix_key), axis=1)
                        matC, matD = [int(x) for x  in row['mat2'].split('-')]
                        matC_path = get_matrix_path(project_dir, sample_dict[matC], res, row['norm'], extra_dir, chrom_matrix)
                        matD_path = get_matrix_path(project_dir, sample_dict[matD], res, row['norm'], extra_dir, chrom_matrix)
                        dfC = matrix_to_df(matC_path, sum_1=sum_1)
                        dfD = matrix_to_df(matD_path, sum_1=sum_1)
                        df2 = get_diff_matrix(dfC, dfD)
                        df2 = df2.reindex(sorted(df2.columns, key=natural_sort_matrix_key), axis=0).reindex(sorted(df2.columns, key=natural_sort_matrix_key), axis=1)
                    df = get_diff_matrix(df1, df2)
            except FileNotFoundError as e:
                logger.warning(f"File not found for sample {row['mat1']} {sample_dict[int(row['mat1'])]} with {row['norm']} normalization at res {res}kb. Skipped.")
                logger.warning(f"In {project_dir}")
                logger.warning(e)
                continue
            logger.info(f"Figure name: {row['fig']}")
            logger.info(f"{row['chr']}")
            
            if log_score:
                df = np.log10(df.replace({0, max(df)*0.001}))
                vmax = np.log10(row['minmax'])
            else:
                vmax = row['minmax']
            if np.isnan(vmax):
                logger.info(df.columns)

                tmp_vmax = np.percentile(df, vmax_percentile)
                while tmp_vmax == 0 and vmax_percentile < 100:
                    vmax_percentile = min(100, vmax_percentile + 5)
                    tmp_vmax = np.percentile(df, vmax_percentile)
                    logger.info(f"tmp_vmax: {tmp_vmax}")

                logger.info(10**(np.floor(np.log10(tmp_vmax))))
                vmax = round(custom_round(tmp_vmax, 10**(np.floor(np.log10(tmp_vmax)))), 1)
                row['minmax'] = vmax
                logger.info(f"auto vmax at {vmax}")
            
            if row['start']==-1:
                zoom_range = False
            else:
                zoom_range = (int(row['start']), int(row['end']))
            logger.info(zoom_range)
            if isinstance(row['extra_hlines'], (str, int)):
                extra_hlines = list(map(int, row['extra_hlines'].split(';')))
                logger.info(f'extra_hlines: {extra_hlines}')
            else:
                extra_hlines = False
            if isinstance(row['extra_vlines'], (str, int)):
                extra_vlines = list(map(int, row['extra_vlines'].split(';')))
            else:
                extra_vlines = False
            if row['type'] in ['hic', 'hic_triangle']:
                vmin=0
            else:
                vmin=-row['minmax']
            
            hic_plot(
                        df, 
                        vmin=vmin, vmax=vmax,
                        res=res,
                        extra_hlines=extra_hlines,
                        extra_vlines=extra_vlines,
                        pombe=True,
                        chroms=chroms,
                        centromeres=centromeres,
                        gridsep=gridspec,
                        zoom_range=zoom_range,
                        cmap=row['cmap'],
                        triangle=triangle,
                        output_path=set_output_path(output_project_dir, row, extension=extension)
                    )
        elif row['type'] == 'dist_curve':
            if row['extra_vlines'] is not None:
                vbounds = [75, 800] # cohesin and condensin domain sizes in fission yeast
            else:
                vbounds = [int(x) for x in row['extra_vlines'].split(',')]
            dfs = []
            samples_to_draw = []
            spline_factor = row['spline_factor'] if 'spline_factor' in row and row['spline_factor'] is not None else 0
            if isinstance(row['mat1'], int):
                samples_to_draw.append(row['mat1'])
                dfs.append(matrix_to_df(get_matrix_path(project_dir, sample_dict[int(row['mat1'])], res, row['norm']), sum_1=sum_1))
                if row['mat2'] is not None:
                    samples_to_draw.append(row['mat2'])
                    dfs.append(matrix_to_df(get_matrix_path(project_dir, sample_dict[int(row['mat2'])], res, row['norm']), sum_1=sum_1))
                logger.info(f"Drawing distance curve plot for {samples_to_draw}")
                plot_dist_curves(
                    dfs = dfs,
                    res = res,
                    domain_type = domain_type,
                    subfolder=row['subfolder'] if 'subfolder' in row and row['subfolder'] is not None else "",
                    output_path=set_output_path(output_project_dir, row, extension='svg'),
                    chroms = chroms,
                    labels = samples_to_draw,
                    vbounds = vbounds,
                    output_array = output_array,
                    sum_1 = sum_1,
                    spline_factor = spline_factor,
                    vmin= row['minmax'],
                )
            else:
                logger.info(f"Drawing distance curve plot for {row['mat1']}")
                samples_to_draw = [int(x) for x in row['mat1'].split(';')]
                samples_to_draw = samples_to_draw[1:] + [samples_to_draw[0]]
                dfs = [matrix_to_df(get_matrix_path(project_dir, sample_dict[x], res, row['norm']), sum_1=sum_1) for x in samples_to_draw]
                logger.info(f"len(df.columns) before density rescaling: {[len(df.columns) for df in dfs]}")
                if density_rescaling:
                    dfs = [df * len(df.columns) * len(df.columns) for df in dfs]
                if len(dfs) > 1:
                    if len(dfs) > 2:
                        colors = categorical_palettes['nature_cat'][:len(dfs)-1] + ['gray']
                    else:
                        colors = ['red', 'gray']
                    plot_dist_curves(
                        dfs = dfs, 
                        res = res,
                        domain_type = domain_type,
                        subfolder=row['subfolder'] if 'subfolder' in row and row['subfolder'] is not None else "",
                        output_path=set_output_path(output_project_dir, row, extension='svg'),
                        chroms = chroms,
                        labels = samples_to_draw,
                        colors = colors, legend_loc = legend_loc,
                        legend_bbox_to_anchor = legend_bbox_to_anchor,
                        vbounds = vbounds,
                        sum_1 = sum_1,
                        vmin = row['minmax']
                    )
        elif row['type'] in ['dist_curve_diff', 'dist_curve_ratio', 'dist_curve_log2diff']:
            if row['extra_vlines'] is not None:
                vbounds = [75, 800] # 75kb for cohesin and 800 for condensin domains
            else:
                vbounds = [int(x) for x in row['extra_vlines'].split(',')]
            
            df1 = matrix_to_df(get_matrix_path(project_dir, sample_dict[int(row['mat1'])], res, row['norm']), sum_1=sum_1)
            df2 = matrix_to_df(get_matrix_path(project_dir, sample_dict[int(row['mat2'])], res, row['norm']), sum_1=sum_1)
            if row['type'] == 'dist_curve_diff':
                compare_method = 'diff'
            elif row['type'] == 'dist_curve_ratio':
                compare_method = 'ratio'
            elif row['type'] == 'dist_curve_log2diff':
                compare_method = 'log2diff'
            plot_dist_compare_curve(df1, df2, sum_1=sum_1, method=compare_method, distance_ranges=distance_ranges, chroms=chroms, 
                                        vbounds=vbounds, rect_colors=['#eebbcb', '#bce2e8', '#c8c2c6'], 
                                        output_path=set_output_path(output_project_dir, row), **kwargs)
        elif row['type'] == 'scatter_diff':
            logger.info(f"Drawing scatter HiC score plot for {row['mat1']} and {row['mat2']}")
            df1_id, df2_id = [int(x) for x in row['mat1'].split('-')]
            ctrl1_id, ctrl2_id = [int(x) for x in row['mat2'].split('-')]
            df1_path = get_matrix_path(project_dir, sample_dict[df1_id], res, row['norm'])
            df2_path = get_matrix_path(project_dir, sample_dict[df2_id], res, row['norm'])
            ctrl1_path = get_matrix_path(project_dir, sample_dict[ctrl1_id], res, row['norm'])
            ctrl2_path = get_matrix_path(project_dir, sample_dict[ctrl2_id], res, row['norm'])
            if 'min_dist' in kwargs and 'max_dist' in kwargs:
                min_dist = kwargs['min_dist']
                max_dist = kwargs['max_dist']
            else:
                min_dist = False
                max_dist = False
            vmin = -row['minmax']
            vmax = row['minmax']
            scatter_diff_scores(df1_path, df2_path, ctrl1_path=ctrl1_path, ctrl2_path=ctrl2_path,
                                df1_id=df1_id, df2_id=df2_id, ctrl1_id=ctrl1_id, ctrl2_id=ctrl2_id,
                                min_dist=min_dist, max_dist=max_dist,
                                target_bed_path=target_bed_path,
                                chroms=chroms, domain_type=domain_type,
                                res=res, vmin=vmin, vmax=vmax,
                                output_pval = output_pval, stat_test=stat_test,
                                dotsize=dotsize, basic_color=basic_color, highlight_color=highlight_color, 
                                figsize=figsize, 
                                output_path=set_output_path(output_project_dir, row, extension=extension)
                                )
        plt.close()
    return to_draw

matplotlib.use("Agg")


if __name__ == '__main__':
    args = load_config()
    dfs = {}
    sample_df = pl.read_excel(args.sample_sheet, engine='openpyxl')[['id', 'name']]
    sample_df.columns = ['sample_id', 'sample_name']
    sample_dict = dict(sample_df.iter_rows())
    to_draw = read_to_draw(args.config_path)
    sum_1 = args.sum_1
    domain_type = args.domain_type
    svg = args.svg
    target_bed_path = args.target_bed_path
    plot_multi_fig(
    to_draw, project_dir=args.project_dir, sample_dict=sample_dict, output_project_dir=args.project_dir, 
    centromeres=False, sum_1=sum_1, domain_type=domain_type,
    svg=svg, target_bed_path=target_bed_path)

# cd ~/Project/your_project
# python3 script/2025-01-20_claire_hic_plot_script.py --project_dir `pwd` --sample_sheet db/hic.xlsx out/2025-01-20_hic_figures/sample_instructions.txt
