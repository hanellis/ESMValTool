import logging
from pathlib import Path
from pprint import pformat
import matplotlib.pyplot as plt
import iris # type: ignore
import numpy as np
import numpy.ma as ma
import calendar

from esmvaltool.diag_scripts.shared import ( # type: ignore
    group_metadata,
    run_diagnostic,
    save_data,
    save_figure,
    select_metadata,
    sorted_metadata,
)
from esmvaltool.diag_scripts.shared.plot import quickplot # type: ignore

from basic_functions import (get_provenance_record, 
                             load_data,
                             iso_depth_4d)

logger = logging.getLogger(Path(__file__).stem)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def get_iso_data(cfg, group_md, variable):
    """
    Compute T20D from 3D temperature fields and return dataset-mean cube dictionary.
    """
    logger.info(f"Processing iso data for variable: {variable}")
    iso_results = {}

    # Load temperatures using load_data
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

    t20d_cube = iso_depth_4d(cube, 20, 'month_number')
    t20d_cube.data = np.ma.masked_invalid(t20d_cube.data)
    t20d_mean = t20d_cube.collapsed(['latitude','longitude'], iris.analysis.MEAN)
    iso_results[dataset] = {
                'cube': t20d_mean,
                'filename': file
            }

    # Save output
    output_basename = f"{dataset}_{variable}_t20d"
    provenance_record = get_provenance_record(output_basename, list(input_filenames))
    save_data(output_basename, provenance_record, cfg, t20d_cube)

    logger.info(f"T20D processed and saved for {dataset}")
    return iso_results

def sorted_bycoord(cube, coord):
    '''Sort a cube into order according to one coord, e.g. season number'''

    coord_to_sort = cube.coord(coord)
    assert coord_to_sort.ndim == 1, 'One dim coords only please.'
    dim, = cube.coord_dims(coord_to_sort)
    index = [slice(None)] * cube.ndim
    index[dim] = np.argsort(coord_to_sort.points)
    return cube[tuple(index)]

def plot_ts(cfg, dict, title, output_basename):
    """
    Plot all datasets in a single figure.
    """

    months = list(calendar.month_abbr)[1:]

    plt.figure(figsize=(10, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(dict)))  # Generate distinct colors

    input_filenames = set()

    for i, (dataset, info) in enumerate(dict.items()):
        cube = info['cube']
        file = info['filename']
        input_filenames.update(file if isinstance(file, list) else [file])
        

        color = 'k' if 'HadISST' in dataset else colors[i]
        plt.plot(months, cube.data, label=dataset, color=color)

    plt.gca().invert_yaxis()
    plt.xlabel("Month", fontsize=12)
    plt.ylabel("20 $^\\circ$C isotherm depth / m", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True)

    provenance_record = get_provenance_record(title, list(input_filenames))
    save_figure(output_basename, provenance_record, cfg)
    logger.info(f"SCTR plot saved: {output_basename}")
    plt.close()


def main(cfg):
    """Compute the monthly 20degree isotherm in the SCTR and plot results for all datasets."""
    input_data = cfg['input_data'].values()
    grouped_data = group_metadata(input_data, 'dataset')
    iso_results = {}
    for group_name, group_md in grouped_data.items():
        iso_res = get_iso_data(cfg, group_md, 'sctr_region')
        iso_results.update(iso_res)
           
    # Plot results for all datasets
    plot_ts(cfg, iso_results, '20 $^\circ$C isotherm depth in SCTR', 'sctr_t20d')


if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)

