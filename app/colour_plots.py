import math

import matplotlib.pyplot as plt


def save_colour_map(som, file_path):
    """Save a map trained on RGB colours as an image, one pixel per node.

    Only works for colour data: each node's 3 weights are used as its colour.
    """
    plt.imsave(file_path, som.weights)


def save_colour_map_snapshots(som, file_path):
    """The colour map at each saved snapshot, in a grid of 6 per row.

    Needs som.fit(..., snapshot_every=N) to have been run on RGB colours.
    """
    n_cols = 6
    n_rows = math.ceil(len(som.snapshots) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2.6 * n_rows))

    # A flat list is easier to loop over than a grid of rows and columns.
    # Hide every box first, so any unused ones on the last row stay empty.
    axes = axes.flatten()
    for ax in axes:
        ax.axis("off")

    for i, snapshot in enumerate(som.snapshots):
        ax = axes[i]
        ax.imshow(snapshot["weights"])
        ax.set_title(f"Iteration {snapshot['iteration']}")

    fig.suptitle(f"{som.width}x{som.height} map during training")
    fig.tight_layout()
    fig.savefig(file_path, dpi=100)
    plt.close(fig)
