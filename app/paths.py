from enum import Enum
from pathlib import Path

# Project root, based on this file's location so it works from any directory.
BASE_PATH = Path(__file__).resolve().parent.parent


class ConfigurationPaths(Enum):
    CONFIG_DIR_PATH = BASE_PATH / "config"
    CONFIG_FILE_PATH = CONFIG_DIR_PATH / "config.yaml"


class OutputPaths(Enum):
    OUTPUTS_PATH = BASE_PATH / "outputs"
    DEBUG_DIR_PATH = OUTPUTS_PATH / "debug"

    COLOUR_MAP_100_FILE_PATH = OUTPUTS_PATH / "100.png"
    COLOUR_MAP_1000_FILE_PATH = OUTPUTS_PATH / "1000.png"
    WINE_MAP_FILE_PATH = OUTPUTS_PATH / "wine_dataset.png"

    COLOUR_MAP_100_CLUSTER_MAP_FILE_PATH = OUTPUTS_PATH / "100_clusters.png"
    COLOUR_MAP_1000_CLUSTER_MAP_FILE_PATH = OUTPUTS_PATH / "1000_clusters.png"
    WINE_MAP_CLUSTER_MAP_FILE_PATH = OUTPUTS_PATH / "wine_dataset_clusters.png"

    COLOUR_MAP_100_CLUSTER_SUMMARY_FILE_PATH = OUTPUTS_PATH / "100_cluster_summary.csv"
    COLOUR_MAP_1000_CLUSTER_SUMMARY_FILE_PATH = (
        OUTPUTS_PATH / "1000_cluster_summary.csv"
    )
    WINE_MAP_CLUSTER_SUMMARY_FILE_PATH = (
        OUTPUTS_PATH / "wine_dataset_cluster_summary.csv"
    )

    COLOUR_MAP_100_SNAPSHOTS_FILE_PATH = DEBUG_DIR_PATH / "100_training.png"
    COLOUR_MAP_1000_SNAPSHOTS_FILE_PATH = DEBUG_DIR_PATH / "1000_training.png"
    WINE_MAP_SNAPSHOTS_FILE_PATH = DEBUG_DIR_PATH / "wine_dataset_training.png"

    COLOUR_MAP_100_OPTIMAL_CLUSTERS_FILE_PATH = (
        DEBUG_DIR_PATH / "100_optimal_clusters.csv"
    )
    COLOUR_MAP_1000_OPTIMAL_CLUSTERS_FILE_PATH = (
        DEBUG_DIR_PATH / "1000_optimal_clusters.csv"
    )
    WINE_MAP_OPTIMAL_CLUSTERS_FILE_PATH = (
        DEBUG_DIR_PATH / "wine_dataset_optimal_clusters.csv"
    )


def create_required_dir_paths():
    OutputPaths.OUTPUTS_PATH.value.mkdir(parents=True, exist_ok=True)
    OutputPaths.DEBUG_DIR_PATH.value.mkdir(parents=True, exist_ok=True)
