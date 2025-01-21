# HiClaire-viz
HiC map drawing script for Noma lab (single script version)
- Python script specifically for HiC data processed and folder structure specified by rfy2_hic2 pipeline (https://pubmed.ncbi.nlm.nih.gov/39283450/)
---
## Example usage
`python3 draw_maps.py --project_dir `pwd` --sample_sheet db/hic.xlsx sample_instructions.txt`

---
## Tab-delimited file for multi-plot config
### Columns
1. fig
    - figure name, used as output file prefix
2. type
    - plot type, to specify which type of plot to draw. currently supplots
    - "hic" (normal Hi-C contact map)
    - "hic_diff" (comparison between 2 maps)
    - "hic_triangle" (horizontal triangular plot, previously drawn but not tested this time)
    - "hic_diffdiff" (difference difference map between two pairs of HiC maps)
    - "dist_curve" (distance curve, previously used but it was taking extra time in the Neurospora project data. Test was not complete.)
    - "scatter_diff" (scatter plot to compare average contact per distance between two samples. Current version requires input of BED-format domain region information. Not yet tested on Neurospora data.)
3. res
    - resolution, e.g., 5kb, 20kb, 100kb
4. norm
    - normalization method, to specify which folder to find the normalized data, basically ICE2
5. mat1
    - ID number of sample(s) to draw. Multiple input supported for some plot types
6. mat2
    - ID number of sample(s) to draw. Multiple input supported for some plot types. "N/A" if only one sample is relevant
7. chr
    - chromosome, or sequence name. "N/A" to draw all available.
8. start
    - sequence start position to extract local region. "N/A" to draw the whole chromosome(s).
9. end
    - sequence end position to extract local region. "N/A" to draw the whole chromosome(s).
10. cmap
    - colormap name to select color gradient, acceptable parameters include
    - Matplotlib built-in colormaps: https://matplotlib.org/stable/users/explain/colors/colormaps.html
    - in-script defined colormaps, e.g., NSMBish, hideki_bright
11. minmax
    - Maximum score to show as the most extreme color (vmax). vmin = 0 for Hi-C map and vmin = -vmax in difference maps.
12. extra_hlines
    - advanced option to add horizontal lines.
13. extra_vlines
    - advanced option to add vertical lines or to define regions to highlight in background in distance curve plots.
