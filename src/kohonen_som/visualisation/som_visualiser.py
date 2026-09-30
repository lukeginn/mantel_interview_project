import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


class SOMVisualiser:
    """Plots of a trained SelfOrganisingMap.

    Takes the map and its data once, so each plot only needs a file path.
    `data` and `labels` are only needed for the label plots, and a
    SOMClustering `clustering` only for the cluster map. The snapshot plot
    needs som.fit(..., snapshot_every=N).

    Plots:
        plot_label_map            each node coloured by its true label
        plot_label_map_snapshots  the label map during training, to show it learning
        plot_cluster_map          each node coloured by its k-means cluster

    Helpers, highest level first:
        _plot_group_map           the shared plot behind the label and cluster maps
        _label_nodes              which label each node represents
        _draw_groups              draws a map of group numbers
        _legend_handles           the coloured legend squares
        _save                     tidy, save, close
    """

    def __init__(self, som, data=None, labels=None, clustering=None):
        self.som = som
        self.data = data
        self.labels = labels
        self.clustering = clustering

        # One colour and legend name per true label, e.g. "Label 0", using
        # matplotlib's standard "tab10" colours.
        if labels is not None:
            unique_labels = np.unique(labels)
            self.label_colours = plt.colormaps["tab10"].colors[: len(unique_labels)]
            self.label_names = [f"Label {label}" for label in unique_labels]

    def plot_label_map(self, file_path):
        """Each node coloured by the label of the data row closest to it.

        Weights with many features can't be shown as colours, so this shows
        which known label each part of the map represents instead. If the map
        works, each label should form its own region.
        """
        label_map = self._label_nodes(self.som.weights)
        title = (
            f"{self.som.width}x{self.som.height} map of "
            f"{len(self.data)} rows, {self.data.shape[1]} columns"
        )
        self._plot_group_map(
            label_map, self.label_colours, self.label_names, title, file_path
        )

    def plot_label_map_snapshots(self, file_path):
        """The label map at each saved snapshot, in a grid of 6 per row, to show
        the map learning."""
        n_cols = 6
        n_rows = math.ceil(len(self.som.snapshots) / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2.6 * n_rows))

        # A flat list is easier to loop over than a grid of rows and columns.
        # Hide every box first, so any unused ones on the last row stay empty.
        axes = axes.flatten()
        for ax in axes:
            ax.axis("off")

        for i, snapshot in enumerate(self.som.snapshots):
            ax = axes[i]
            label_map = self._label_nodes(snapshot["weights"])
            self._draw_groups(ax, label_map, self.label_colours)
            ax.set_title(f"Iteration {snapshot['iteration']}")

        # One shared legend, in the bottom-right corner of the figure.
        handles = self._legend_handles(self.label_colours, self.label_names)
        fig.legend(handles=handles, loc="lower right")

        fig.suptitle(f"{self.som.width}x{self.som.height} map during training")
        self._save(fig, file_path, dpi=100)

    def plot_cluster_map(self, n_clusters, file_path):
        """Each node coloured by its k-means cluster, found without any labels.

        Uses matplotlib's softer "Set2" colours, so a cluster never looks like the
        true label that happens to share its number (cluster numbers are arbitrary).
        """
        cluster_map = self.clustering.kmeans_cluster_nodes(n_clusters)
        colours = plt.colormaps["Set2"].colors[:n_clusters]
        names = [f"Cluster {i}" for i in range(n_clusters)]
        title = (
            f"{self.som.width}x{self.som.height} map, "
            f"{n_clusters} clusters found without labels"
        )
        self._plot_group_map(cluster_map, colours, names, title, file_path)

    def _plot_group_map(self, group_map, colours, names, title, file_path):
        """Plot a map of group numbers (labels or clusters) with a legend, and save it."""
        fig, ax = plt.subplots(figsize=(7, 6))
        self._draw_groups(ax, group_map, colours)
        ax.set_title(title)
        ax.set_xlabel("Map column")
        ax.set_ylabel("Map row")

        # Legend to the right of the map.
        handles = self._legend_handles(colours, names)
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1, 1))

        self._save(fig, file_path, dpi=150)

    def _label_nodes(self, weights):
        """For each node, the label of the data row closest to that node's weights.

        Takes the weights rather than using the trained map, so it also works on
        training snapshots. Returns shape (height, width).
        """
        height, width = weights.shape[0], weights.shape[1]
        label_map = np.zeros((height, width))

        for row in range(height):
            for col in range(width):
                # Find the data row closest to this node, and use its label.
                distances = ((self.data - weights[row, col]) ** 2).sum(axis=1)
                closest_row = np.argmin(distances)
                label_map[row, col] = self.labels[closest_row]

        return label_map

    def _draw_groups(self, ax, group_map, colours):
        """Draw a map of group numbers, one colour per group.

        vmin and vmax fix group 0 to the first colour, group 1 to the second, and
        so on, even if some groups are missing from this particular map.
        """
        colour_map = ListedColormap(colours)
        ax.imshow(group_map, cmap=colour_map, vmin=0, vmax=len(colours) - 1)

    def _legend_handles(self, colours, names):
        """One coloured square per group, for the legend."""
        handles = []
        for i in range(len(names)):
            handles.append(Patch(color=colours[i], label=names[i]))
        return handles

    def _save(self, fig, file_path, dpi):
        """Tidy the spacing, save the figure, and close it to free memory."""
        fig.tight_layout()
        fig.savefig(file_path, dpi=dpi)
        plt.close(fig)
