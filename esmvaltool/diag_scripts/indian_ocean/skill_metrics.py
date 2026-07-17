import logging
from pathlib import Path
from pprint import pformat
import matplotlib.pyplot as plt
import iris # type: ignore
import iris.coord_categorisation 
import numpy as np
import numpy.ma as ma
from datetime import datetime
import calendar
import scipy # type: ignore
from scipy.stats import linregress # type: ignore
from scipy.signal import welch, detrend # type: ignore
import cartopy.crs as ccrs # type: ignore
import cartopy.feature as cfeature # type: ignore


from esmvaltool.diag_scripts.shared import ( # type: ignore
    group_metadata,
    run_diagnostic,
    save_data,
    save_figure,
    select_metadata,
    sorted_metadata,
)
from esmvaltool.diag_scripts.shared.plot import quickplot # type: ignore

from basic_functions import get_provenance_record, load_data, to_set, compute_cube_diff, load_and_update_dict

logger = logging.getLogger(Path(__file__).stem)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)


def _to_file_list(filename_entry):
    if isinstance(filename_entry, list):
        return filename_entry
    return [filename_entry]


def _is_obs_entry(info):
    for file_name in _to_file_list(info['filename']):
        if Path(str(file_name)).name.startswith('OBS'):
            return True
    return False


def _find_obs_dataset_key(dataset_dict):
    obs_keys = [key for key, info in dataset_dict.items() if _is_obs_entry(info)]
    if not obs_keys:
        raise ValueError('Could not find observation entry with filename starting with OBS.')
    return sorted(obs_keys)[0]


def _model_keys(dataset_dict):
    return {key for key, info in dataset_dict.items() if not _is_obs_entry(info)}



def _first_cube(dataset_dict):
    return next(iter(dataset_dict.values()))['cube']


def check_item_load(metadata, variable, dict_name):
    if (item := load_data(metadata, variable, get_filenames=True)):
            dict_name.update(item)


def main(cfg):
    """Compute the Dipole Mode Index and plot results for all datasets."""
    input_data = cfg['input_data'].values()
    grouped_data = group_metadata(input_data, 'dataset')
    dmi_monthly, dmi_seas, nino_anoms_seas, nino_anoms_monthly, west_anoms_seas, west_anoms_monthly, east_anoms_seas, east_anoms_monthly = {}, {}, {}, {}, {}, {}, {}, {}
    for group_name, group_md in grouped_data.items():
        # west_anoms = load_data(group_md, 'sst_anomalies_west', get_filenames=True)
        # east_anoms = load_data(group_md, 'sst_anomalies_east', get_filenames=True)

        load_and_update_dict(group_md, 'sst_anomalies_nino_seasonal', nino_anoms_seas)
        load_and_update_dict(group_md, 'sst_anomalies_nino_monthly', nino_anoms_monthly)
        load_and_update_dict(group_md, 'sst_anomalies_west_seasonal', west_anoms_seas)
        load_and_update_dict(group_md, 'sst_anomalies_west_monthly', west_anoms_monthly)
        load_and_update_dict(group_md, 'sst_anomalies_east_seasonal', east_anoms_seas)
        load_and_update_dict(group_md, 'sst_anomalies_east_monthly', east_anoms_monthly)

        # if (item := load_data(group_md, 'sst_anomalies_nino', get_filenames=True)):
        #     nino_data.update(item)  
        # if (item := load_data(group_md, 'sst_anomalies_global', get_filenames=True)):
        #     sst_anomalies_global.update(item)   


    if west_anoms_seas and east_anoms_seas:
        dmi_res = compute_cube_diff(cfg, west_anoms_seas, east_anoms_seas, 'dmi_seas')
        if dmi_res:
            dmi_seas.update(dmi_res)

    if west_anoms_monthly and east_anoms_monthly:
        dmi_res = compute_cube_diff(cfg, west_anoms_monthly, east_anoms_monthly, 'dmi_monthly')
        if dmi_res:
            dmi_monthly.update(dmi_res)

    

if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)

