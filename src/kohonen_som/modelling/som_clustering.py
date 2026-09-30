import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class SOMClustering:
    """Groups the nodes of a trained SelfOrganisingMap into clusters, without labels.

    `original_data` is the data before scaling. It is optional, and only used so
    the cluster summary shows real units (e.g. alcohol %) rather than scaled ones.

    Results:
        silhouette_scores           silhouette score for each number of clusters, best marked
        best_n_clusters             the number of clusters with the highest silhouette score
        cluster_summary             compares the clusters side by side on every feature

    Building blocks (kmeans_cluster_nodes is also used by the cluster map plot):
        kmeans_cluster_nodes        which cluster each node is in, found with k-means
        assign_samples_to_clusters  which cluster each data row is in, via its best matching node

    Helpers:
        _node_weights               the trained map as a list of nodes
        _describe_difference        describes a difference from average as high, average or low
    """

    def __init__(
        self,
        som,
        data,
        feature_names,
        min_clusters=2,
        max_clusters=8,
        original_data=None,
    ):
        self.som = som
        self.data = data
        self.feature_names = feature_names
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        if original_data is None:
            original_data = data
        self.original_data = original_data
        # k-means results for each number of clusters (k), so each is only worked
        # out once. The same k is needed several times: for the silhouette
        # scores, the cluster summary and the cluster map plot. Reusing it is
        # faster, and guarantees they all show exactly the same clusters.
        self._saved_cluster_maps = {}

    def silhouette_scores(self):
        """Silhouette score for each number of clusters, with the best one marked.

        The silhouette score measures how well separated the clusters are, from
        -1 (badly) to 1 (very well). For each node it compares:
          a = the average distance to the other nodes in its own cluster
          b = the average distance to the nodes in the nearest other cluster
        and scores it (b - a) / max(a, b); the table shows the average over all
        nodes. It is worked out on the map's nodes rather than the raw data,
        because the map has already summarised the data into smooth prototypes.
        Trust a clear peak; a best k at the edge of the range means "try more".
        """
        nodes = self._node_weights()

        rows = []
        for k in range(self.min_clusters, self.max_clusters + 1):
            clusters = self.kmeans_cluster_nodes(k).reshape(len(nodes))
            score = silhouette_score(nodes, clusters)
            rows.append({"k": k, "silhouette_score": score})

        table = pd.DataFrame(rows)
        table["best"] = table["silhouette_score"] == table["silhouette_score"].max()
        return table

    def best_n_clusters(self):
        """The number of clusters with the highest silhouette score."""
        table = self.silhouette_scores()
        best_row = table["silhouette_score"].idxmax()
        return int(table.loc[best_row, "k"])

    def cluster_summary(self, n_clusters):
        """Compares the clusters side by side on every feature.

        One row per feature, strongest first, and one column per cluster showing
        its average, marked high or low when it is more than half a standard
        deviation from the overall average. The last column, "All", is the overall
        average for comparison.
        """
        sample_clusters = self.assign_samples_to_clusters(n_clusters)
        clusters = np.unique(sample_clusters)
        overall_mean = self.original_data.mean(axis=0)
        overall_std = self.original_data.std(axis=0)

        # --- Stage 1: each cluster's average, and how unusual it is ---
        # The difference is how far the cluster's average is from the overall
        # average, in standard deviations. The ranking puts the cluster's most
        # distinctive features (largest difference, either way) first.
        averages = {}
        differences = {}
        rankings = {}
        for cluster in clusters:
            members = self.original_data[sample_clusters == cluster]
            averages[cluster] = members.mean(axis=0)
            differences[cluster] = (averages[cluster] - overall_mean) / overall_std
            rankings[cluster] = np.argsort(-np.abs(differences[cluster]))

        # --- Stage 2: order the features, strongest first ---
        # Take turns: each cluster's most distinctive feature, then each cluster's
        # second, and so on, skipping features already listed. For wine: cluster
        # 0's top feature (alcohol), cluster 1's (OD280), cluster 2's (proline),
        # then cluster 0's second (colour intensity), ...
        # Simply ranking features by how much they vary across clusters doesn't
        # work as well: it can favour features that separate only some of the
        # clusters, leaving another cluster's defining features near the bottom.
        ordered_features = []
        for rank in range(len(self.feature_names)):
            for cluster in clusters:
                feature = rankings[cluster][rank]
                if feature not in ordered_features:
                    ordered_features.append(feature)

        # --- Stage 3: build the table ---
        # First row: the number of samples in each cluster.
        size_row = {"feature": "n_samples"}
        for cluster in clusters:
            size_row[f"Cluster {cluster}"] = np.sum(sample_clusters == cluster)
        size_row["All"] = len(self.data)
        rows = [size_row]

        # Then one row per feature, e.g. "12.22 (low)" for each cluster.
        for feature in ordered_features:
            row = {"feature": self.feature_names[feature]}
            for cluster in clusters:
                average = averages[cluster][feature]
                level = self._describe_difference(differences[cluster][feature])
                row[f"Cluster {cluster}"] = f"{average:,.2f} ({level})"
            row["All"] = f"{overall_mean[feature]:,.2f}"
            rows.append(row)

        return pd.DataFrame(rows)

    def kmeans_cluster_nodes(self, n_clusters):
        """Returns the cluster number of every node, shape (height, width)."""
        # Already worked out for this number of clusters? Reuse it.
        if n_clusters in self._saved_cluster_maps:
            return self._saved_cluster_maps[n_clusters]

        # Each node's weights are a "prototype" sample, so group them with k-means.
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=self.som.seed)
        node_clusters = kmeans.fit_predict(self._node_weights())

        # Put the cluster numbers back into the shape of the grid.
        cluster_map = node_clusters.reshape(self.som.height, self.som.width)
        self._saved_cluster_maps[n_clusters] = cluster_map
        return cluster_map

    def assign_samples_to_clusters(self, n_clusters):
        """Returns the cluster of each data row: the cluster of its best matching node."""
        cluster_map = self.kmeans_cluster_nodes(n_clusters)

        sample_clusters = []
        for row, col in self.som.predict(self.data):
            sample_clusters.append(cluster_map[row, col])
        return np.array(sample_clusters)

    def _node_weights(self):
        """The trained map as a list of nodes: one row of weights per node."""
        n_nodes = self.som.height * self.som.width
        n_features = self.som.weights.shape[2]
        return self.som.weights.reshape(n_nodes, n_features)

    def _describe_difference(self, difference):
        """Describes a difference from average, in standard deviations, in words."""
        if difference > 0.5:
            return "high"
        if difference < -0.5:
            return "low"
        return "average"
