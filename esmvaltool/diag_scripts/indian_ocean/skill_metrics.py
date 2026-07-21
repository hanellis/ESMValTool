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

import warnings
from iris.warnings import IrisVagueMetadataWarning

warnings.filterwarnings(
    "ignore",
    category=IrisVagueMetadataWarning
)

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
def find_peak_years(cube, threshold):
    """Find positive/negative event years from a seasonal anomaly time series."""
    if 'year' not in [coord.name() for coord in cube.coords()]:
        iris.coord_categorisation.add_year(cube, 'time', name='year')

    data = ma.filled(cube.data, np.nan)
    years = cube.coord('year').points

    positive_years = years[np.isfinite(data) & (data > threshold)]
    negative_years = years[np.isfinite(data) & (data < -threshold)]

    return sorted(set(positive_years.astype(int))), sorted(set(negative_years.astype(int)))


def add_subplot(ax, cube, cb_label, cb_range, cmap, title):
    # Add map features
    ax.coastlines()
    gl = ax.gridlines(draw_labels=True, linewidth=0.8, color='grey', linestyle='--', alpha=1)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}

    # Contour plot
    contour = ax.contourf(
        cube.coord('longitude').points,
        cube.coord('latitude').points,
        cube.data,
        cmap=cmap,  # Change colormap here
        levels=cb_range, extend='both'
    )

    # Title for each subplot
    ax.set_title(title, fontsize=14)

    # Add individual colorbar
    cb = plt.colorbar(contour, ax=ax, orientation='horizontal', pad=0.07, fraction=0.04)
    cb.set_label(cb_label, fontsize=12)
    cb.ax.tick_params(labelsize=10)

def add_wind_subplot(ax, u_cube, v_cube, title, quiver_step=4):
    """Quiver plot of wind direction with wind speed shading."""
    ax.coastlines()
    gl = ax.gridlines(draw_labels=True, linewidth=0.8, color='grey', linestyle='--', alpha=1)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}

    lons = u_cube.coord('longitude').points
    lats = u_cube.coord('latitude').points
    u = ma.filled(u_cube.data, np.nan)
    v = ma.filled(v_cube.data, np.nan)
    speed = np.sqrt(u ** 2 + v ** 2)

    levels = np.arange(0, 8.5, 0.5)
    contour = ax.contourf(lons, lats, speed, levels=levels, cmap='PuRd', extend='max')

    lons_q = lons[::quiver_step]
    lats_q = lats[::quiver_step]
    u_q = u[::quiver_step, ::quiver_step]
    v_q = v[::quiver_step, ::quiver_step]
    ax.quiver(lons_q, lats_q, u_q, v_q,
              transform=ccrs.PlateCarree(), scale=50, scale_units='width', color='black', width=0.003)

    ax.set_title(title, fontsize=14)
    cb = plt.colorbar(contour, ax=ax, orientation='horizontal', pad=0.07, fraction=0.04)
    cb.set_label('Wind speed / m s$^{-1}$', fontsize=12)
    cb.ax.tick_params(labelsize=10)


def plot_map(cfg, plot_dict, output_basename):

    input_filenames = set()
    (dataset, info), = plot_dict.items() 
    file = info['filename']
    input_filenames.update(file if isinstance(file, list) else [file])
    cube = info['cube']

    fig, axes = plt.subplots(1, figsize=(10,6), subplot_kw={'projection': ccrs.PlateCarree()})
    add_subplot(axes, cube, 'mm $\\mathregular{day^{-1}}$ per std DMI', np.arange(-1.9,2.1,0.2), 'PuOr_r', f'Regression of DMI on precip - {dataset}')
    
    # Adjust layout
    fig.tight_layout()

    save_title = f'{output_basename}_{dataset}'
    provenance_record = get_provenance_record(save_title, list(input_filenames))
    save_figure(save_title, provenance_record, cfg)
    logger.info(f"Map plot saved: {save_title}")
    plt.close()


def plot_double_map(cfg, cube1, pos_count, pos_title, cube2, neg_count, neg_title, dataset, output_basename, prov_files):

    # cube1 is positive events
    # cube2 is negative events

    fig, axes = plt.subplots(2, figsize=(8, 10), subplot_kw={'projection': ccrs.PlateCarree()})

    add_subplot(axes.flat[0], cube1, "SST anomaly / $^\\circ$C", np.arange(-2.5,2.7,0.2), 'RdBu_r', pos_title)
    add_subplot(axes.flat[1], cube2, "SST anomaly / $^\\circ$C", np.arange(-2.5,2.7,0.2), 'RdBu_r', neg_title)

    fig.suptitle(f'Composites for {dataset}', fontsize=16, y=0.98)
    fig.text(0.83, 0.93, f'Event count = {pos_count}')
    fig.text(0.83, 0.46, f'Event count = {neg_count}')
    
    # Adjust layout
    fig.tight_layout()
    provenance_record = get_provenance_record(output_basename, list(prov_files))
    save_figure(output_basename, provenance_record, cfg)
    logger.info(f"Composite plot saved: {output_basename}")
    # Show the figure
    plt.show()
        
def _extract_single_season(cube, season):
    """Extract one season from a seasonal time series (e.g. SON or DJF)."""
    if 'season' not in [coord.name() for coord in cube.coords()]:
        iris.coord_categorisation.add_season(cube, 'time', name='season')

    season_lower = season.lower()
    season_constraint = iris.Constraint(season=lambda value: str(value).lower() == season_lower)
    return cube.extract(season_constraint)


def _spatial_reference(cube):
    """Return a 2D lat-lon reference cube for blank composites."""
    if cube.ndim == 2:
        return cube
    return cube[0]


def extract_composite_cube(years, seasonal_anoms, season):
    """Extract a year composite for a requested season from seasonal anomalies."""
    season_cube = _extract_single_season(seasonal_anoms, season)
    if season_cube is None:
        return create_blank_cube(_spatial_reference(seasonal_anoms))
 
    if len(years) == 0:
        return create_blank_cube(_spatial_reference(season_cube))

    if 'year' not in [coord.name() for coord in season_cube.coords()]:
        iris.coord_categorisation.add_year(season_cube, 'time', name='year')

    year_constraint = iris.Constraint(year=lambda value: value in years)
    composite_cube = season_cube.extract(year_constraint)

    if composite_cube is None:
        return create_blank_cube(_spatial_reference(season_cube))

    # A single matched year may produce a cube without a time dimension.
    if 'time' not in [coord.name() for coord in composite_cube.coords(dim_coords=True)]:
        return composite_cube

    if composite_cube.shape[composite_cube.coord_dims('time')[0]] == 1:
        return composite_cube[0]

    return composite_cube.collapsed('time', iris.analysis.MEAN)

def create_blank_cube(reference_cube):
    """Create a blank cube with the same lat/lon as reference."""
    lat = reference_cube.coord('latitude').points
    lon = reference_cube.coord('longitude').points
    data = ma.masked_all((len(lat), len(lon)))
    
    blank_cube = iris.cube.Cube(
        data,
        long_name='Blank',
        units='',
        dim_coords_and_dims=[
            (reference_cube.coord('latitude'), 0),
            (reference_cube.coord('longitude'), 1)
        ]
    )
    return blank_cube

def get_event_sets(iod_yrs, enso_yrs):
    enso_yrs = [year - 1 for year in enso_yrs] # Shift ENSO years to match SON season for IOD
    both_yrs = list(np.intersect1d(iod_yrs, enso_yrs))
    iod_only_yrs= list(np.setdiff1d(iod_yrs, both_yrs))
    enso_only_yrs = list(np.setdiff1d(enso_yrs, both_yrs))
    return both_yrs, iod_only_yrs, enso_only_yrs

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


def _event_year_groups(dmi_cube, nino_cube, return_seas_year=False):
    dmi_std = dmi_cube.collapsed('time', iris.analysis.STD_DEV).data
    nino_std = nino_cube.collapsed('time', iris.analysis.STD_DEV).data

    pIOD_yrs, nIOD_yrs = find_peak_years(dmi_cube, dmi_std)
    EN_yrs, LN_yrs = find_peak_years(nino_cube, nino_std)

    pIOD_EN_yrs, pIOD_only_yrs, EN_only_yrs = get_event_sets(pIOD_yrs, EN_yrs)
    nIOD_LN_yrs, nIOD_only_yrs, LN_only_yrs = get_event_sets(nIOD_yrs, LN_yrs)
    print('EN_years:', EN_yrs, 'EN_only_years:', EN_only_yrs)
    if return_seas_year:
        return {
            'pIOD_all': pIOD_yrs,
            'nIOD_all': nIOD_yrs,
            'EN_all': EN_yrs,
            'LN_all': LN_yrs,
            'pIOD_only': pIOD_only_yrs,
            'nIOD_only': nIOD_only_yrs,
            'EN_only': [year + 1 for year in EN_only_yrs],
            'LN_only': [year + 1 for year in LN_only_yrs],
            'pIOD_both': pIOD_EN_yrs,
            'EN_both': [year + 1 for year in pIOD_EN_yrs],
            'nIOD_both': nIOD_LN_yrs,
            'LN_both': [year + 1 for year in nIOD_LN_yrs],
        }
    else:
        return {
            'pIOD_all': pIOD_yrs,
            'nIOD_all': nIOD_yrs,
            'EN_all': [year - 1 for year in EN_yrs],
            'LN_all': [year - 1 for year in LN_yrs],
            'pIOD_only': pIOD_only_yrs,
            'nIOD_only': nIOD_only_yrs,
            'EN_only': EN_only_yrs,
            'LN_only': LN_only_yrs,
            'pIOD_both': pIOD_EN_yrs,
            'EN_both': pIOD_EN_yrs,
            'nIOD_both': nIOD_LN_yrs,
            'LN_both': nIOD_LN_yrs,
        }


def _first_cube(dataset_dict):
    return next(iter(dataset_dict.values()))['cube']


def _build_panels_for_event(column_order, years_by_column, season, sst_dict, wind_u_dict, wind_v_dict, precip_dict, obs_keys):
    scalar_row_specs = [
        ('sst', 'SST anomaly / $^\\circ$C', np.arange(-2.5, 2.7, 0.2), 'RdBu_r', sst_dict),
        ('precip', 'Precip anomaly / mm day$^{-1}$', np.arange(-6.0, 6.5, 0.5), 'BrBG', precip_dict),
    ]

    def _extract_col(var_dict, obs_key, column, years):
        dataset_key = obs_key if column == 'OBS' else column
        cube = var_dict[dataset_key]['cube']
        comp = extract_composite_cube(years, cube, season)
        if comp is None:
            comp = create_blank_cube(_spatial_reference(_first_cube(var_dict)))
        return comp

    # --- SST row (index 0) ---
    var_name, cb_label, cb_range, cmap, var_dict = scalar_row_specs[0]
    sst_row = [
        {'type': 'contourf',
         'cube': _extract_col(var_dict, obs_keys[var_name], col, years_by_column[col]),
         'cb_label': cb_label, 'cb_range': cb_range, 'cmap': cmap}
        for col in column_order
    ]

    # --- Wind row (index 1) ---
    wind_row = []
    for col in column_order:
        years = years_by_column[col]
        u_cube = _extract_col(wind_u_dict, obs_keys['wind_u'], col, years)
        v_cube = _extract_col(wind_v_dict, obs_keys['wind_v'], col, years)
        wind_row.append({'type': 'quiver', 'u_cube': u_cube, 'v_cube': v_cube})

    # --- Precip row (index 2) ---
    var_name, cb_label, cb_range, cmap, var_dict = scalar_row_specs[1]
    precip_row = [
        {'type': 'contourf',
         'cube': _extract_col(var_dict, obs_keys[var_name], col, years_by_column[col]),
         'cb_label': cb_label, 'cb_range': cb_range, 'cmap': cmap}
        for col in column_order
    ]

    return [sst_row, wind_row, precip_row]


def plot_3x3_composite_map(cfg, panels, column_labels, row_labels, title, output_basename, prov_files, event_counts):
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), subplot_kw={'projection': ccrs.PlateCarree()})

    for row_idx in range(3):
        for col_idx in range(3):
            panel = panels[row_idx][col_idx]
            panel_title = f"{row_labels[row_idx]} | {column_labels[col_idx]}"
            if panel['type'] == 'quiver':
                add_wind_subplot(axes[row_idx, col_idx], panel['u_cube'], panel['v_cube'], panel_title, quiver_step=10)
            else:
                add_subplot(axes[row_idx, col_idx], panel['cube'], panel['cb_label'], panel['cb_range'], panel['cmap'], panel_title)

    fig.suptitle(title, fontsize=16, y=0.99)
    count_text = (
        f"Event count (Obs / {column_labels[1]} / {column_labels[2]}): "
        f"{event_counts[0]} / {event_counts[1]} / {event_counts[2]}"
    )
    fig.text(0.5, 0.965, count_text, ha='center', va='top', fontsize=10)
    fig.tight_layout()

    provenance_record = get_provenance_record(output_basename, list(prov_files))
    save_figure(output_basename, provenance_record, cfg)
    logger.info(f"Composite plot saved: {output_basename}")
    plt.close()

def monthly_plot_event_years(cfg, dmi_dict_seas, nino_dict_seas, dmi_dict_monthly, east_pole_anoms_monthly, west_pole_anoms_monthly, nino_anoms_monthly):
    '''Plot monthly composites for event years.
    ENSO window: June-May. IOD window: March-February.'''

    def _ensure_time_coords(cube):
        if 'year' not in [coord.name() for coord in cube.coords()]:
            iris.coord_categorisation.add_year(cube, 'time', name='year')
        if 'month_number' not in [coord.name() for coord in cube.coords()]:
            iris.coord_categorisation.add_month_number(cube, 'time', name='month_number')

    def _monthly_composite_series(cube, event_years, start_month):
        work_cube = cube.copy()
        _ensure_time_coords(work_cube)

        data = ma.filled(work_cube.data, np.nan).squeeze()
        years = work_cube.coord('year').points.astype(int)
        months = work_cube.coord('month_number').points.astype(int)
        
        series = []
        for offset in range(12):
            month = ((start_month - 1 + offset) % 12) + 1
            year_offset = 1 if month < start_month else 0

            slot_values = []
            for event_year in event_years:
                target_year = int(event_year) + year_offset
                mask = (years == target_year) & (months == month)
                values = data[mask]
                if np.size(values) > 0:
                    finite_values = values[np.isfinite(values)]
                    if np.size(finite_values) > 0:
                        slot_values.extend(np.atleast_1d(finite_values).tolist())

            if slot_values:
                series.append(float(np.nanmean(slot_values)))
            else:
                series.append(np.nan)

        return np.asarray(series)

    def _month_labels(start_month):
        return [calendar.month_abbr[((start_month - 1 + i) % 12) + 1] for i in range(12)]

    obs_keys = {
        'dmi_seas': _find_obs_dataset_key(dmi_dict_seas),
        'nino_seas': _find_obs_dataset_key(nino_dict_seas),
        'dmi_monthly': _find_obs_dataset_key(dmi_dict_monthly),
        'east_monthly': _find_obs_dataset_key(east_pole_anoms_monthly),
        'west_monthly': _find_obs_dataset_key(west_pole_anoms_monthly),
        'nino_monthly': _find_obs_dataset_key(nino_anoms_monthly),
    }

    common_model_datasets = (
        _model_keys(dmi_dict_seas)
        & _model_keys(nino_dict_seas)
        & _model_keys(dmi_dict_monthly)
        & _model_keys(east_pole_anoms_monthly)
        & _model_keys(west_pole_anoms_monthly)
        & _model_keys(nino_anoms_monthly)
    )
    if len(common_model_datasets) < 2:
        raise ValueError('Need at least two common model datasets for monthly event plots.')

    selected_models = sorted(common_model_datasets)[:2]
    column_order = ['OBS', selected_models[0], selected_models[1]]
    colour_by_column = {
        'OBS': 'k',
        selected_models[0]: 'hotpink',
        selected_models[1]: 'royalblue',
    }
    logger.info(f"Monthly plot column order: {column_order}")

    years_by_column = {
        'OBS': _event_year_groups(
            dmi_dict_seas[obs_keys['dmi_seas']]['cube'],
            nino_dict_seas[obs_keys['nino_seas']]['cube'],
        )
    }
    for dataset in selected_models:
        years_by_column[dataset] = _event_year_groups(dmi_dict_seas[dataset]['cube'], nino_dict_seas[dataset]['cube'])
        
    print(years_by_column)
    def _dataset_key(obs_key, column):
        return obs_key if column == 'OBS' else column

    def _plot_panel(ax, variable_spec, event_key, start_month, panel_event_title):
        var_title, var_dict, obs_key = variable_spec

        for column in column_order:
            data_key = _dataset_key(obs_key, column)
            event_years = years_by_column[column][event_key]
            series = _monthly_composite_series(var_dict[data_key]['cube'], event_years, start_month)
            label = f"{column} (n={len(event_years)})"
            ax.plot(
                np.arange(12),
                series,
                color=colour_by_column[column],
                linestyle='-',
                linewidth=2.0,
                label=label,
            )

        ax.axhline(0.0, color='grey', linewidth=1.0, linestyle='-')
        ax.set_title(f"{panel_event_title} | {var_title}", fontsize=13)
        ax.set_xticks(np.arange(12))
        ax.set_xticklabels(_month_labels(start_month), rotation=45, ha='right')
        ax.set_ylabel('SST anomaly / K')
        ax.grid(True, alpha=0.3)

    figure_specs = [
        {
            'name': 'iod_all',
            'title': 'IOD Monthly SST Anomalies \n'
            '(All Events)',
            'start_month': 3,
            'positive_key': 'pIOD_all',
            'negative_key': 'nIOD_all',
            'positive_title': 'Positive IOD',
            'negative_title': 'Negative IOD',
            'variables': [
                ('West pole', west_pole_anoms_monthly, obs_keys['west_monthly']),
                ('East pole', east_pole_anoms_monthly, obs_keys['east_monthly']),
                ('DMI', dmi_dict_monthly, obs_keys['dmi_monthly']),
            ],
        },
        {
            'name': 'iod_only',
            'title': 'IOD Monthly SST Anomalies \n'
            '(IOD-only Events)',
            'start_month': 3,
            'positive_key': 'pIOD_only',
            'negative_key': 'nIOD_only',
            'positive_title': 'Positive IOD',
            'negative_title': 'Negative IOD',
            'variables': [
                ('West pole', west_pole_anoms_monthly, obs_keys['west_monthly']),
                ('East pole', east_pole_anoms_monthly, obs_keys['east_monthly']),
                ('DMI', dmi_dict_monthly, obs_keys['dmi_monthly']),
            ],
        },
        {
            'name': 'iod_both',
            'title': 'IOD Monthly SST Anomalies \n'
            '(Co-occurring Events)',
            'start_month': 3,
            'positive_key': 'pIOD_both',
            'negative_key': 'nIOD_both',
            'positive_title': 'Positive IOD',
            'negative_title': 'Negative IOD',
            'variables': [
                ('West pole', west_pole_anoms_monthly, obs_keys['west_monthly']),
                ('East pole', east_pole_anoms_monthly, obs_keys['east_monthly']),
                ('DMI', dmi_dict_monthly, obs_keys['dmi_monthly']),
            ],
        },
        {
            'name': 'enso_all',
            'title': 'Niño Monthly SST Anomalies \n'
            '(All Events)',
            'start_month': 6,
            'positive_key': 'EN_all',
            'negative_key': 'LN_all',
            'positive_title': 'El Niño',
            'negative_title': 'La Niña',
            'variables': [
                ('Niño 3.4', nino_anoms_monthly, obs_keys['nino_monthly']),
            ],
        },
        {
            'name': 'enso_only',
            'title': 'Niño Monthly SST Anomalies \n'
            '(ENSO-only Events)',
            'start_month': 6,
            'positive_key': 'EN_only',
            'negative_key': 'LN_only',
            'positive_title': 'El Niño',
            'negative_title': 'La Niña',
            'variables': [
                ('Niño 3.4', nino_anoms_monthly, obs_keys['nino_monthly']),
            ],
        },
        {
            'name': 'enso_both',
            'title': 'Niño Monthly SST Anomalies \n'
            '(Co-occurring Events)',
            'start_month': 6,
            'positive_key': 'EN_both',
            'negative_key': 'LN_both',
            'positive_title': 'El Niño',
            'negative_title': 'La Niña',
            'variables': [
                ('Niño 3.4', nino_anoms_monthly, obs_keys['nino_monthly']),
            ],
        },
    ]

    for spec in figure_specs:
        ncols = len(spec['variables'])
        fig, axes = plt.subplots(2, ncols, figsize=(5.2 * ncols, 8.5), squeeze=False, sharex=True)

        for col_idx, variable_spec in enumerate(spec['variables']):
            _plot_panel(
                axes[0, col_idx],
                variable_spec,
                spec['positive_key'],
                spec['start_month'],
                spec['positive_title'],
            )
            _plot_panel(
                axes[1, col_idx],
                variable_spec,
                spec['negative_key'],
                spec['start_month'],
                spec['negative_title'],
            )

            if col_idx == 0:
                axes[0, col_idx].legend(loc='lower right', fontsize=9)
                axes[1, col_idx].legend(loc='upper right', fontsize=9)

        fig.suptitle(spec['title'], fontsize=16, y=0.99)
        fig.tight_layout()

        prov_files = set()
        for column in column_order:
            dmi_seas_key = _dataset_key(obs_keys['dmi_seas'], column)
            nino_seas_key = _dataset_key(obs_keys['nino_seas'], column)
            prov_files |= to_set(dmi_dict_seas[dmi_seas_key]['filename'])
            prov_files |= to_set(nino_dict_seas[nino_seas_key]['filename'])

        for _, var_dict, var_obs_key in spec['variables']:
            for column in column_order:
                data_key = _dataset_key(var_obs_key, column)
                prov_files |= to_set(var_dict[data_key]['filename'])

        output_basename = f"monthly_event_years_{spec['name']}"
        provenance_record = get_provenance_record(output_basename, list(prov_files))
        save_figure(output_basename, provenance_record, cfg)
        logger.info(f"Monthly event-year plot saved: {output_basename}")
        plt.close()

    


def composite_map(cfg, dmi_dict, nino_dict, sst_anom_dict, wind_u_dict, wind_v_dict, precip_anom_dict):

    obs_keys = {
        'dmi': _find_obs_dataset_key(dmi_dict),
        'nino': _find_obs_dataset_key(nino_dict),
        'sst': _find_obs_dataset_key(sst_anom_dict),
        'wind_u': _find_obs_dataset_key(wind_u_dict),
        'wind_v': _find_obs_dataset_key(wind_v_dict),
        'precip': _find_obs_dataset_key(precip_anom_dict),
    }

    common_model_datasets = (
        _model_keys(dmi_dict)
        & _model_keys(nino_dict)
        & _model_keys(sst_anom_dict)
        & _model_keys(wind_u_dict)
        & _model_keys(wind_v_dict)
        & _model_keys(precip_anom_dict)
    )

    if len(common_model_datasets) < 2:
        raise ValueError(
            'Need at least two common model datasets across all variables for 3x3 plotting.'
        )

    selected_models = sorted(common_model_datasets)[:2]
    column_order = ['OBS', selected_models[0], selected_models[1]]
    logger.info(f"3x3 column order: {column_order}")

    years_by_column = {
        'OBS': _event_year_groups(dmi_dict[obs_keys['dmi']]['cube'], nino_dict[obs_keys['nino']]['cube'], return_seas_year=True)
    }
    for dataset in selected_models:
        years_by_column[dataset] = _event_year_groups(dmi_dict[dataset]['cube'], nino_dict[dataset]['cube'], return_seas_year=True)

    prov_files = set()
    for dataset in selected_models:
        prov_files |= to_set(dmi_dict[dataset]['filename'])
        prov_files |= to_set(nino_dict[dataset]['filename'])
        prov_files |= to_set(sst_anom_dict[dataset]['filename'])
        prov_files |= to_set(wind_u_dict[dataset]['filename'])
        prov_files |= to_set(wind_v_dict[dataset]['filename'])
        prov_files |= to_set(precip_anom_dict[dataset]['filename'])

    prov_files |= to_set(dmi_dict[obs_keys['dmi']]['filename'])
    prov_files |= to_set(nino_dict[obs_keys['nino']]['filename'])
    prov_files |= to_set(sst_anom_dict[obs_keys['sst']]['filename'])
    prov_files |= to_set(wind_u_dict[obs_keys['wind_u']]['filename'])
    prov_files |= to_set(wind_v_dict[obs_keys['wind_v']]['filename'])
    prov_files |= to_set(precip_anom_dict[obs_keys['precip']]['filename'])

    event_specs = [
        ('pIOD_all', 'son', 'Positive IOD - SON'),
        ('nIOD_all', 'son', 'Negative IOD - SON'),
        ('pIOD_only', 'son', 'Only Positive IOD - SON'),
        ('nIOD_only', 'son', 'Only Negative IOD - SON'),
        ('pIOD_both', 'son', 'Positive IOD (Both-Event Years) - SON'),
        ('nIOD_both', 'son', 'Negative IOD (Both-Event Years) - SON'),
        ('EN_all', 'djf', 'El Niño - DJF'),
        ('LN_all', 'djf', 'La Niña - DJF'),
        ('EN_only', 'djf', 'Only El Niño - DJF'),
        ('LN_only', 'djf', 'Only La Niña - DJF'),
        ('EN_both', 'djf', 'El Niño (Both-Event Years) - DJF'),
        ('LN_both', 'djf', 'La Niña (Both-Event Years) - DJF'),
    ]

    row_labels = ['SST', 'Wind', 'Precipitation']
    col_labels = ['Observation', selected_models[0], selected_models[1]]

    for event_key, season, event_title in event_specs:
        event_years = {column: years_by_column[column][event_key] for column in column_order}
        event_counts = [len(event_years[column]) for column in column_order]
        panels = _build_panels_for_event(
            column_order,
            event_years,
            season,
            sst_anom_dict,
            wind_u_dict,
            wind_v_dict,
            precip_anom_dict,
            obs_keys,
        )
        output_basename = f"composite_3x3_{event_key}_{season}"
        plot_3x3_composite_map(
            cfg,
            panels,
            col_labels,
            row_labels,
            event_title,
            output_basename,
            prov_files,
            event_counts,
        )

def check_item_load(metadata, variable, dict_name):
    if (item := load_data(metadata, variable, get_filenames=True)):
            dict_name.update(item)

def plot_dmi_histograms(cfg, dmi_dict, obs_keys, column_order, output_basename, prov_files):
    """Create a 3-row histogram plot of DMI seasonal distribution with skewness."""
    from scipy.stats import skew
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 18))
    
    col_names = ['OBS'] + column_order

    for idx, col in enumerate(col_names):
        ax = axes[idx, 0] if axes.ndim > 1 else axes[idx]

        dataset_key = obs_keys if col == 'OBS' else col
        cube = dmi_dict[dataset_key]['cube']
        
        data = ma.filled(cube.data, np.nan)
        data = data[np.isfinite(data)]
        skewness = skew(data)
        
        ax.hist(data, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(np.nanmean(data), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.nanmean(data):.2f}')
        ax.set_xlabel('DMI / °C', fontsize=16)
        ax.set_ylabel('Frequency', fontsize=16)
        ax.tick_params(labelsize=14)
        
        ax.set_title(f'{col_names[idx]}\nSkewness: {skewness:.3f}', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
    
    fig.suptitle('DMI (SON) Distribution', fontsize=20, fontweight='bold', y=1.00)
    fig.tight_layout()
    
    provenance_record = get_provenance_record(output_basename, list(prov_files))
    save_figure(output_basename, provenance_record, cfg)
    logger.info(f"DMI histogram plot saved: {output_basename}")
    plt.close()



def compute_std_from_cube(cube):
    """Extract data from cube and compute standard deviation."""
    data = ma.filled(cube.data, np.nan)
    data = data[np.isfinite(data)]
    return np.std(data)


def create_skill_scores_table(cfg, dmi_monthly, dmi_seas, nino_monthly, nino_seas, output_basename, prov_files):
    """Create a table of skill scores (standard deviation) for DMI and Nino indices."""
    import pandas as pd
    
    try:
        obs_key = _find_obs_dataset_key(dmi_monthly)
    except ValueError:
        logger.warning("No observation data found for DMI monthly. Skipping skill scores table.")
        return
    
    model_keys = sorted(_model_keys(dmi_monthly))[:2]
    if len(model_keys) < 2:
        model_keys = sorted(_model_keys(dmi_monthly))
    
    dataset_names = ['OBS'] + model_keys
    dataset_keys = [obs_key] + model_keys
    
    # Compute standard deviations for each metric
    metrics = ['DMI Monthly (K)', 'DMI Seasonal (K)', 'Nino Monthly (K)', 'Nino Seasonal (K)']
    data_dicts = [dmi_monthly, dmi_seas, nino_monthly, nino_seas]
    
    scores = {metric: [] for metric in metrics}
    
    for metric, data_dict in zip(metrics, data_dicts):
        for key in dataset_keys:
            if key in data_dict:
                std_value = compute_std_from_cube(data_dict[key]['cube'])
                scores[metric].append(f'{std_value:.3f}')
            else:
                scores[metric].append('N/A')
    
    # Create DataFrame
    df = pd.DataFrame(scores, index=dataset_names).T
    
    # Create figure with table
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = [[str(val) for val in row] for row in df.values]
    row_labels = df.index.tolist()
    col_labels = df.columns.tolist()
    
    table = ax.table(cellText=table_data, rowLabels=row_labels, colLabels=col_labels,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    # Style only existing cells to avoid KeyError across matplotlib versions.
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(weight='bold', color='white')
            continue

        if row % 2 == 0:
            cell.set_facecolor('#f0f0f0')
        else:
            cell.set_facecolor('white')

        # Row labels are typically stored at col == -1 when rowLabels are used.
        if col == -1:
            cell.set_text_props(weight='bold')
    
    plt.title('Skill Scores: Standard Deviation of DMI and Nino Indices', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    provenance_record = get_provenance_record(output_basename, list(prov_files))
    save_figure(output_basename, provenance_record, cfg)
    logger.info(f"Skill scores table saved: {output_basename}")
    
    # Also save as data file
    csv_basename = output_basename.replace('.png', '.csv')
    df.to_csv(csv_basename)
    logger.info(f"Skill scores CSV saved: {csv_basename}")
    
    plt.close()


def main(cfg):
    """Compute the Dipole Mode Index and plot results for all datasets."""
    input_data = cfg['input_data'].values()
    grouped_data = group_metadata(input_data, 'dataset')
    dmi_monthly, dmi_seas, nino_anoms_seas, nino_anoms_monthly, west_anoms_seas, west_anoms_monthly, east_anoms_seas, east_anoms_monthly = {}, {}, {}, {}, {}, {}, {}, {}
    sst_anomalies_global, precip_anoms, zonal_wind_anoms, meridional_wind_anoms = {}, {}, {}, {}
    for group_name, group_md in grouped_data.items():
        west_anoms_monthly_single = load_data(group_md, 'sst_anomalies_west_monthly', get_filenames=True)
        east_anoms_monthly_single = load_data(group_md, 'sst_anomalies_east_monthly', get_filenames=True)
        if west_anoms_monthly_single and east_anoms_monthly_single:
            dmi_result_monthly = compute_cube_diff(cfg, west_anoms_monthly_single, east_anoms_monthly_single, 'dmi_monthly')
            if dmi_result_monthly:
                dmi_monthly.update(dmi_result_monthly)
        west_anoms_seas_single = load_data(group_md, 'sst_anomalies_west_seasonal', get_filenames=True)
        east_anoms_seas_single = load_data(group_md, 'sst_anomalies_east_seasonal', get_filenames=True)
        if west_anoms_seas_single and east_anoms_seas_single:
            dmi_result_seas = compute_cube_diff(cfg, west_anoms_seas_single, east_anoms_seas_single, 'dmi_seas')
            if dmi_result_seas:
                dmi_seas.update(dmi_result_seas)
        
 
        load_and_update_dict(group_md, 'sst_anomalies_global', sst_anomalies_global)
        load_and_update_dict(group_md, 'precip_anomalies_global_seasonal', precip_anoms)
        load_and_update_dict(group_md, 'zonal_wind_anomalies_global', zonal_wind_anoms)
        load_and_update_dict(group_md, 'meridional_wind_anomalies_global', meridional_wind_anoms)
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

    # Collect provenance files
    prov_files = set()
    for data_dict in [dmi_monthly, dmi_seas, nino_anoms_monthly, nino_anoms_seas]:
        for key in data_dict:
            prov_files |= to_set(data_dict[key]['filename'])

    # Generate skill scores table
    create_skill_scores_table(cfg, dmi_monthly, dmi_seas, nino_anoms_monthly, nino_anoms_seas, 
                              'skill_scores_table', prov_files)

    plot_dmi_histograms(cfg, dmi_seas, _find_obs_dataset_key(dmi_seas), sorted(_model_keys(dmi_seas)), 
                        'dmi_histogram_seasonal', [dmi_seas[key]['filename'] for key in dmi_seas])

    monthly_plot_event_years(
        cfg,
        dmi_seas,
        nino_anoms_seas,
        dmi_monthly,
        east_anoms_monthly,
        west_anoms_monthly,
        nino_anoms_monthly,
    )
    
    # composite_map(
    #     cfg,
    #     dmi_seas,
    #     nino_anoms_seas,
    #     sst_anomalies_global,
    #     zonal_wind_anoms,
    #     meridional_wind_anoms,
    #     precip_anoms,
    # )

if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)

