from app.colour_plots import save_colour_map, save_colour_map_snapshots
from app.config import load_config
from app.data_preprocessing import load_wine_dataset, make_random_colours, standardise
from app.paths import ConfigurationPaths, OutputPaths, create_required_dir_paths
from src.kohonen_som.modelling.self_organising_map import SelfOrganisingMap
from src.kohonen_som.modelling.som_clustering import SOMClustering
from src.kohonen_som.visualisation.som_visualiser import SOMVisualiser


def run_colour_map_100(config):
    """Train a 10x10 map on random colours, then save its clusters and plots."""
    colours = make_random_colours(n_samples=config.colours.n_samples, seed=config.seed)

    som = SelfOrganisingMap(
        width=config.colour_map_100.width,
        height=config.colour_map_100.height,
        n_iterations=config.colour_map_100.n_iterations,
        learning_rate=config.colour_map_100.learning_rate,
        seed=config.seed,
    )
    som.fit(colours, snapshot_every=config.colour_map_100.snapshot_every)

    clustering = SOMClustering(
        som,
        colours,
        feature_names=config.colours.feature_names,
        min_clusters=config.clustering.min_clusters,
        max_clusters=config.clustering.max_clusters,
    )
    silhouette_scores = clustering.silhouette_scores()
    silhouette_scores.to_csv(
        OutputPaths.COLOUR_MAP_100_OPTIMAL_CLUSTERS_FILE_PATH.value, index=False
    )
    # Use the number of clusters set in the config, or the best one if it's null.
    n_clusters = config.colour_map_100.n_clusters or clustering.best_n_clusters()

    cluster_summary = clustering.cluster_summary(n_clusters)
    cluster_summary.to_csv(
        OutputPaths.COLOUR_MAP_100_CLUSTER_SUMMARY_FILE_PATH.value, index=False
    )

    visualiser = SOMVisualiser(som, clustering=clustering)
    save_colour_map(som, OutputPaths.COLOUR_MAP_100_FILE_PATH.value)
    save_colour_map_snapshots(som, OutputPaths.COLOUR_MAP_100_SNAPSHOTS_FILE_PATH.value)
    visualiser.plot_cluster_map(
        n_clusters, OutputPaths.COLOUR_MAP_100_CLUSTER_MAP_FILE_PATH.value
    )


def run_colour_map_1000(config):
    """Train a 100x100 map on random colours, then save its clusters and plots."""
    colours = make_random_colours(n_samples=config.colours.n_samples, seed=config.seed)

    som = SelfOrganisingMap(
        width=config.colour_map_1000.width,
        height=config.colour_map_1000.height,
        n_iterations=config.colour_map_1000.n_iterations,
        learning_rate=config.colour_map_1000.learning_rate,
        seed=config.seed,
    )
    som.fit(colours, snapshot_every=config.colour_map_1000.snapshot_every)

    clustering = SOMClustering(
        som,
        colours,
        feature_names=config.colours.feature_names,
        min_clusters=config.clustering.min_clusters,
        max_clusters=config.clustering.max_clusters,
    )
    silhouette_scores = clustering.silhouette_scores()
    silhouette_scores.to_csv(
        OutputPaths.COLOUR_MAP_1000_OPTIMAL_CLUSTERS_FILE_PATH.value, index=False
    )
    # Use the number of clusters set in the config, or the best one if it's null.
    n_clusters = config.colour_map_1000.n_clusters or clustering.best_n_clusters()

    cluster_summary = clustering.cluster_summary(n_clusters)
    cluster_summary.to_csv(
        OutputPaths.COLOUR_MAP_1000_CLUSTER_SUMMARY_FILE_PATH.value, index=False
    )

    visualiser = SOMVisualiser(som, clustering=clustering)
    save_colour_map(som, OutputPaths.COLOUR_MAP_1000_FILE_PATH.value)
    save_colour_map_snapshots(
        som, OutputPaths.COLOUR_MAP_1000_SNAPSHOTS_FILE_PATH.value
    )
    visualiser.plot_cluster_map(
        n_clusters, OutputPaths.COLOUR_MAP_1000_CLUSTER_MAP_FILE_PATH.value
    )


def run_wine_map(config):
    """Train a map on the wine dataset, then save its clusters and plots.

    The labels are never used for training: fit() only sees the 13 measurements.
    They are only used afterwards, to colour the label map plots.
    """
    original_data, labels, wine_feature_names = load_wine_dataset()
    data = standardise(original_data)

    som = SelfOrganisingMap(
        width=config.wine_map.width,
        height=config.wine_map.height,
        n_iterations=config.wine_map.n_iterations,
        learning_rate=config.wine_map.learning_rate,
        seed=config.seed,
    )
    som.fit(data, snapshot_every=config.wine_map.snapshot_every)

    clustering = SOMClustering(
        som,
        data,
        feature_names=wine_feature_names,
        min_clusters=config.clustering.min_clusters,
        max_clusters=config.clustering.max_clusters,
        original_data=original_data,
    )
    silhouette_scores = clustering.silhouette_scores()
    silhouette_scores.to_csv(
        OutputPaths.WINE_MAP_OPTIMAL_CLUSTERS_FILE_PATH.value, index=False
    )
    # Use the number of clusters set in the config, or the best one if it's null.
    n_clusters = config.wine_map.n_clusters or clustering.best_n_clusters()

    cluster_summary = clustering.cluster_summary(n_clusters)
    cluster_summary.to_csv(
        OutputPaths.WINE_MAP_CLUSTER_SUMMARY_FILE_PATH.value, index=False
    )

    visualiser = SOMVisualiser(som, data, labels, clustering=clustering)
    visualiser.plot_label_map(OutputPaths.WINE_MAP_FILE_PATH.value)
    visualiser.plot_label_map_snapshots(OutputPaths.WINE_MAP_SNAPSHOTS_FILE_PATH.value)
    visualiser.plot_cluster_map(
        n_clusters, OutputPaths.WINE_MAP_CLUSTER_MAP_FILE_PATH.value
    )


def main():
    create_required_dir_paths()
    config = load_config(ConfigurationPaths.CONFIG_FILE_PATH.value)

    run_colour_map_100(config)
    run_colour_map_1000(config)
    run_wine_map(config)


if __name__ == "__main__":
    main()
