import os

from make_raw_data_heat_map import generateHeatMap
from make_smooth_data_heat_map import generate_smoothed_heatmap_from_file
from make_accuracy_over_time import generate_accuracy_over_time
from make_raw_data_edge_heat_map import generate_raw_data_edge_heat_map
from make_smoothed_data_edge_heat_map import generate_smoothed_data_edge_heat_map

# Use absolute paths so the script works regardless of current working directory.
SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))

MERGED_INPUT_CSV = os.path.join(
    PROJECT_DIR,
    "GatheringPreprocessingTransformation/GeneratedDataFiles/all_games_merged_clean.csv",
)
SMOOTHED_DATA_CSV = os.path.join(
    SCRIPT_DIR,
    "GeneratedDataAndVisualizations/smoothed_heatmap_data.csv",
)

_PREV_CWD = os.getcwd()
try:
    # Some existing generators use relative output directories based on CWD.
    # Chdir into `Visualizations/` so they save into the right folder.
    os.chdir(SCRIPT_DIR)

    #GENERATE RAW DATA HEAT MAP
    generateHeatMap(input_file=MERGED_INPUT_CSV, num_time_bins=40)

    #GENERATE SMOOTHED DATA HEAT MAP
    generate_smoothed_heatmap_from_file(input_file=MERGED_INPUT_CSV)

    #GENERATE TODO [1]–[3] PLOTS
    generate_accuracy_over_time(input_file=MERGED_INPUT_CSV, num_time_bins=40)
    generate_raw_data_edge_heat_map(input_file=MERGED_INPUT_CSV, num_time_bins=40)
    generate_smoothed_data_edge_heat_map(input_file=SMOOTHED_DATA_CSV)
finally:
    os.chdir(_PREV_CWD)
