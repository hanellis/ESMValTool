import logging
from pathlib import Path
from pprint import pformat

import iris
import numpy as np
import matplotlib.pyplot as plt
from basic_functions import get_provenance_record, load_data

from esmvaltool.diag_scripts.shared import (
    group_metadata,
    run_diagnostic,
    save_data,
    save_figure,
    select_metadata,
    sorted_metadata,
    Datasets,
    Variables,
)
import esmvaltool.diag_scripts.shared as e

logger = logging.getLogger(Path(__file__).stem)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)



def scat_plot(cfg, x_dict, y_dict, title, output_basename, provenance_record):
    """
    Create a scatter plot of x vs. y for multiple datasets.

    Args:
        cfg: Configuration dictionary.
        x_dict: Dictionary of x dicts with dataset and correspond.
        y_dict: Dictionary of y cubes.
        title: Title of the plot.
        output_basename: Base name for the saved plot.
        provenance_record: Completed metadata for provenance - ancestor files already added.
    """
    logger.info(f"Generating scatter plot - {output_basename}")

    # Define plot
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(x_dict)))  # Generate distinct colors

    for i, (dataset, x_cube) in enumerate(x_dict.items()):

        if 'HadISST' in dataset:
            y_cube = y_dict['NCEP']
            plt.scatter(x_cube.data, y_cube.data, color='k',
                        label='Obs', alpha=0.6)
        elif dataset in y_dict:
            y_cube = y_dict[dataset]
            plt.scatter(x_cube.data, y_cube.data, color=colors[i],
                        label=dataset, alpha=0.6)
        else:
            print(f"Warning: {dataset} missing in y_dict, skipping...")
            continue


    # Set plot labels and limits
    plt.xlabel("SST / $^\circ$C", fontsize=12)
    plt.ylabel("Zonal wind speed / m $\mathregular{s^{-1}}$", fontsize=12)
    plt.title(title, fontsize=14)
    #plt.xlim(24, 32)
    #plt.ylim(-6, 6)
    plt.legend(title="Datasets", fontsize=10)
    plt.grid(True)

    # Save figure
    save_figure(output_basename, provenance_record, cfg)
    logger.info(f"Figure saved as: {output_basename}")
    plt.show()


def main(cfg):
    """
    Compute and plot scatter plots for all datasets.
    """
    logger.info("Starting diagnostic.")
    input_data = cfg['input_data'].values()
    grouped_data = group_metadata(input_data, 'dataset')
    sst_data, u10_data = {}, {}
    for group_name, group_md in grouped_data.items():
        logger.info(f"Processing group: {group_name}")
        sst_cube = load_data(group_md, 'sst_east')
        if sst_cube:
            sst_data.update(sst_cube)

        u10_cube = load_data(group_md, 'u10_east')
        if u10_cube:
            u10_data.update(u10_cube)

    output_basename = 'sst_vs_u10_in_eeio_scatter'
    provenance_record = get_provenance_record(output_basename, list(cfg['input_data'].keys()))
    scat_plot(cfg, sst_data, u10_data, 'SST vs U10 for eastern IOD pole', output_basename, provenance_record)
    logger.info("Diagnostic completed.")


if __name__ == '__main__':

    with run_diagnostic() as config:
        main(config)

