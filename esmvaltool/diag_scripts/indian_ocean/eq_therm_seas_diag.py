import logging
from pathlib import Path
from pprint import pformat
import matplotlib.pyplot as plt
import iris # type: ignore
import numpy as np
import numpy.ma as ma

from basic_functions import (get_provenance_record, 
                             load_data,
                             iso_depth_4d,
                             iso_depth)

from esmvaltool.diag_scripts.shared import ( # type: ignore
    group_metadata,
    run_diagnostic,
    save_data,
    save_figure,
    select_metadata,
    sorted_metadata,
)
from esmvaltool.diag_scripts.shared.plot import quickplot # type: ignore

logger = logging.getLogger(Path(__file__).stem)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def extract_eq_IO(cube):
    """
    Extract the IO equatorial region. given an global equatorial band.
    Fold in the NEMO grid at ~73E makes direct extraction not possible.
    """
    
    # Average over the extracted +-2 degrees region
    t20d_mean = np.nanmean(cube.data, axis=1)

    # Extract t20d along equatorial IO (45-100E)
    # Split in grid in ORCA at 73E therefore need to rearrange values
    lon_points = cube[0].coord('longitude').points[0]
    io_inds = np.squeeze(np.where((lon_points >= 45) & (lon_points <= 100)))
    io_lons = lon_points[io_inds]
    io_t20d = t20d_mean[:,io_inds]

    sorted_t20d = []

    for i in range(4):
        sorted_lons, sorted_season = zip(*sorted(zip(io_lons, io_t20d[i])))
        sorted_t20d.append(sorted_season)

    return sorted_t20d    

def get_iso_data(cfg, group_md, variable):
    """
    Compute T20D from 3D temperature fields and return dataset-mean cube dictionary.
    """
    logger.info(f"Processing iso data for variable: {variable}")
    iso_results = {}

# Load temperatures using load_sst_data
    temp_data = load_data(group_md, variable, get_filenames=True)

    if not temp_data or not list(temp_data.keys()):
        logger.warning("No temperature data found. Skipping.")
        return None  

    (dataset, info), = temp_data.items()
    cube = info['cube']
    input_filenames = set()
    file = info['filename']
    input_filenames.update(file if isinstance(file, list) else [file])
    logger.debug(f"Selected dataset: {dataset}")

    t20d_cube = iso_depth(cube, 20, 'season_number')
    t20d_cube.data = np.ma.masked_invalid(t20d_cube.data)
    eq_iso_cube = extract_eq_IO(t20d_cube)
    iso_results[dataset] = {
                'cube': eq_iso_cube,
                'filename': file
            }

    # Save output
    output_basename = f"{dataset}_{variable}_t20d"
    provenance_record = get_provenance_record(output_basename, list(input_filenames))
    save_data(output_basename, provenance_record, cfg, t20d_cube)

    logger.info(f"T20D processed and saved for {dataset}")
    return iso_results

def plot_ts(cfg, plot_dict, title, output_basename):
    """
    Plot all datasets in a single figure.
    """

    seasons = ['DJF','MAM','JJA','SON']

    for n, season in enumerate(seasons):
        logger.info(f"Plotting thermocline for {season}")

        plt.figure(figsize=(10, 5))
        colors = plt.cm.viridis(np.linspace(0, 1, len(plot_dict)))

        input_filenames = set()
    
        for i, (dataset, dict_info) in enumerate(plot_dict.items()):
            cube = dict_info['cube']
            file = dict_info['filename']
            input_filenames.update(file if isinstance(file, list) else [file])
            
            plt.plot(np.linspace(45,100, np.shape(cube[n])[0]), cube[n], label=dataset, color= colors[i])
    
        plt.gca().invert_yaxis()
        plt.xlim(45,100)
        plt.xlabel("Longitude", fontsize=12)
        plt.ylabel("20 $^\\circ$C isotherm depth / m", fontsize=12)
        plt.title(f'{title} - {season}', fontsize=14)
        plt.legend()
        plt.grid(True)
        
        save_title = f'{output_basename}_{season}'
        provenance_record = get_provenance_record(save_title, list(input_filenames))
        save_figure(save_title, provenance_record, cfg)
        logger.info(f"Scatter plot saved: {output_basename}")
        plt.close()


def main(cfg):
    """Compute the 20degree isotherm along the equatorial Indian Ocean and plot results for all datasets."""
    input_data = cfg['input_data'].values()
    grouped_data = group_metadata(input_data, 'dataset')
    iso_results = {}
    for group_name, group_md in grouped_data.items():
        iso_res = get_iso_data(cfg, group_md, 'eq_region')
        iso_results.update(iso_res)
    
    logger.info("Thermocline calculated, now plotting.")
    # Plot results for all datasets
    plot_ts(cfg, iso_results, '20 $^\circ$C isotherm depth along the equatorial Indian Ocean', 'eq_IO_t20d')


if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)

