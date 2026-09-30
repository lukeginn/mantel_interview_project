import numpy as np


class SelfOrganisingMap:
    """Kohonen self-organising map.

    After training, self.weights has shape (height, width, n_features).
    Note: max(width, height) must be at least 3, otherwise the radius won't decay.

    Main methods:
        fit              trains the map on the data
        predict          the (row, col) of each data row's best matching node

    Building block (used by fit and predict):
        find_bmu         the best matching node: the one closest to a sample

    Helpers:
        _gaussian        how strongly each node is pulled towards the sample
        _save_snapshot   stores a copy of the map during training, for plots
    """

    def __init__(
        self,
        width,
        height,
        n_iterations=100,
        learning_rate=0.1,
        seed=42,
    ):
        self.width = width
        self.height = height
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.radius = max(width, height) / 2
        self.seed = seed
        self.weights = None
        self.snapshots = []

    def fit(self, data, snapshot_every=0):
        """Train the map on `data`, shape (n_samples, n_features).

        Repeatedly shows every sample to the map. Each time, the closest node
        (the BMU) and its neighbours are pulled towards the sample. Over time the
        pull gets weaker and the neighbourhood smaller, so the map settles down.

        For debugging, set `snapshot_every` to save a copy of the map every that
        many iterations (0 = off), including the random start and the end. Each
        snapshot in self.snapshots is a dict with the iteration and a copy of the
        weights at that point.
        """
        self.snapshots = []
        # --- Stage 1: start every node with random weights ---
        # Each node gets one random value per feature, drawn between that
        # feature's min and max in the data, so the starting map is in the
        # right range whatever the scale of the data.
        rng = np.random.default_rng(self.seed)
        n_features = data.shape[1]
        weights = rng.uniform(
            data.min(axis=0),
            data.max(axis=0),
            size=(self.height, self.width, n_features),
        )

        # --- Stage 2: things that stay the same for the whole of training ---
        # Row and column numbers of the grid, e.g. [0, 1, ..., 9] for a 10x10
        # map, used to measure how far each node is from the BMU.
        rows = np.arange(self.height)
        cols = np.arange(self.width)
        # Controls how fast the radius and learning rate shrink. Chosen so the
        # radius starts at half the map and ends up at about 1 node.
        time_constant = self.n_iterations / np.log(self.radius)

        # --- Stage 3: loop over the whole dataset n_iterations times ---
        for t in range(self.n_iterations):
            # Shrink the radius and learning rate. Both use the same decay, so
            # it is calculated once. Early on the map makes big, broad changes;
            # later it only makes small, local adjustments.
            decay = np.exp(-t / time_constant)
            radius = self.radius * decay
            learning_rate = self.learning_rate * decay

            # Debug: save the map as it is before this iteration's updates.
            if snapshot_every and t % snapshot_every == 0:
                self._save_snapshot(t, weights)

            # --- Stage 4: show each sample to the map ---
            for sample in data:
                # 4a. Find the BMU: the node that looks most like this sample.
                # `difference` is the gap between the sample and every node, and
                # is reused below to move the nodes.
                difference = sample - weights
                bmu_row, bmu_col = self.find_bmu(difference)

                # 4b. Work out how much each node should move (its influence).
                # Nodes near the BMU on the grid get close to 1, far ones close
                # to 0, following a gaussian (bell curve) of width `radius`.
                #
                # The gaussian can be split into a row part and a column part:
                # distance^2 = row gap^2 + column gap^2 (Pythagoras), and
                # exp(a + b) = exp(a) * exp(b). So instead of one exp per node, we
                # work out one list for the rows and one for the columns, and
                # np.outer multiplies every row value by every column value to
                # fill in the grid. Same result, but on a 100x100 map that's
                # 200 exps instead of 10,000, for every sample.
                row_influence = self._gaussian(rows - bmu_row, radius)
                col_influence = self._gaussian(cols - bmu_col, radius)
                influence = np.outer(row_influence, col_influence)

                # 4c. Move every node part of the way towards the sample.
                # step = learning rate x influence, one number per node, reshaped
                # to (height, width, 1) so it applies to every feature of a node.
                step = learning_rate * influence.reshape(self.height, self.width, 1)
                weights += step * difference

        # --- Stage 5: keep the trained weights for predict() and plotting ---
        self.weights = weights
        if snapshot_every:
            self._save_snapshot(self.n_iterations, weights)

    def predict(self, data):
        """Returns the (row, col) of the best matching node for each sample.

        `data` has shape (n_samples, n_features) and the map must already be
        trained. The result has shape (n_samples, 2): each sample's position
        on the map. The weights are not changed, so it can be used on new data.
        """
        bmu_coordinates = []
        for sample in data:
            # Gap between this sample and every node at once (vectorised over nodes).
            difference = sample - self.weights
            row, col = self.find_bmu(difference)
            bmu_coordinates.append((row, col))

        return np.array(bmu_coordinates)

    def find_bmu(self, difference):
        """Returns the (row, col) of the node that looks most like the sample.

        `difference` is sample - weights: the gap between the sample and every
        node, per feature, with shape (height, width, n_features).
        """
        # The best matching unit (BMU) is the node whose weight vector is closest to the sample.

        # Square each gap so negative and positive gaps both count as distance.
        squared_gaps = difference**2

        # Add up the squared gaps across features, giving one distance per node,
        # shape (height, width). The square root is skipped as it doesn't change
        # which node is closest.
        distances = squared_gaps.sum(axis=2)

        # Position of the smallest distance, counting as if the grid were one
        # long list (e.g. 37 on a 10x10 grid).
        closest_index = np.argmin(distances)

        # Turn that position back into grid coordinates, e.g. 37 -> (3, 7).
        row, col = np.unravel_index(closest_index, distances.shape)
        return row, col

    def _gaussian(self, distance, radius):
        """The standard Kohonen neighbourhood function: a bell curve.

        Returns how strongly a node at `distance` grid squares from the BMU is
        pulled towards the sample: 1 at the BMU, fading smoothly towards 0 the
        further away the node is. `radius` sets how wide the bell is.

        This is the influence from the brief: exp(-distance^2 / (2 * radius^2)).
        """
        # Square the distance: removes minus signs and makes far nodes count more.
        squared_distance = distance**2

        # Scale by the radius: a bigger radius gives a wider bell.
        spread = 2 * radius**2

        # exp of a negative number is between 0 and 1: exp(0) = 1 at the BMU.
        return np.exp(-squared_distance / spread)

    def _save_snapshot(self, iteration, weights):
        """Debug: store a copy of the map at this iteration."""
        self.snapshots.append({"iteration": iteration, "weights": weights.copy()})
