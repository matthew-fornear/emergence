#!/usr/bin/env python3
"""Load world_state.json and visualize entity positions (XYZ) and relations."""
import json
import os
import sys

STATE_FILE = "world_state.json"

def _get_plot_data(state):
    entities = state.get("entities", {})
    if not entities:
        return None, [], [], [], []
    names = list(entities.keys())
    X = [entities[n]["x"] for n in names]
    Y = [entities[n]["y"] for n in names]
    Z = [entities[n]["z"] for n in names]
    return entities, names, X, Y, Z

def render_to_dir(out_dir, turn, state=None):
    """Render 2D and 3D plots to out_dir/turn_<turn>_xy.png and out_dir/turn_<turn>_3d.png."""
    if state is None:
        try:
            with open(STATE_FILE) as f:
                state = json.load(f)
        except Exception:
            return
    entities, names, X, Y, Z = _get_plot_data(state)
    if not entities:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.join(out_dir, f"turn_{turn}")

    # 2D
    fig, ax = plt.subplots(figsize=(12, 10))
    sc = ax.scatter(X, Y, c=Z, s=80, cmap="viridis", alpha=0.8, edgecolors="black", linewidths=0.5)
    for i, n in enumerate(names):
        ax.annotate(n, (X[i], Y[i]), xytext=(4, 4), textcoords="offset points", fontsize=8, alpha=0.9)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Entity positions (color = Z, abstractness)")
    plt.colorbar(sc, label="Z")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(f"{prefix}_xy.png", dpi=150)
    plt.close()

    # 3D
    try:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(X, Y, Z, c=Z, s=60, cmap="viridis", alpha=0.8)
        for i, n in enumerate(names):
            ax.text(X[i], Y[i], Z[i], n, fontsize=7)
        ax.set_xlabel("X (semantic)", fontsize=11, labelpad=8)
        ax.set_ylabel("Y (semantic)", fontsize=11, labelpad=8)
        ax.set_zlabel("Z (abstract)", fontsize=11, labelpad=8)
        ax.set_title("Entity positions in idea space")
        plt.savefig(f"{prefix}_3d.png", dpi=150)
        plt.close()
    except Exception:
        pass

def main():
    try:
        with open(STATE_FILE) as f:
            state = json.load(f)
    except FileNotFoundError:
        print(f"Missing {STATE_FILE}. Run main.py first.")
        sys.exit(1)

    entities = state.get("entities", {})
    relations = state.get("relations", [])

    if not entities:
        print("No entities to plot.")
        sys.exit(0)

    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("Install matplotlib: pip install matplotlib")
        sys.exit(1)

    _, names, X, Y, Z = _get_plot_data(state)

    # 2D plot: X vs Y, color by Z (abstractness)
    fig, ax = plt.subplots(figsize=(12, 10))
    sc = ax.scatter(X, Y, c=Z, s=80, cmap="viridis", alpha=0.8, edgecolors="black", linewidths=0.5)
    for i, n in enumerate(names):
        ax.annotate(n, (X[i], Y[i]), xytext=(4, 4), textcoords="offset points", fontsize=8, alpha=0.9)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Entity positions (color = Z, abstractness)")
    plt.colorbar(sc, label="Z")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    plt.tight_layout()
    out_2d = "world_xy.png"
    plt.savefig(out_2d, dpi=150)
    plt.close()
    print(f"Saved {out_2d} (X,Y with Z as color)")

    # 3D plot
    try:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(X, Y, Z, c=Z, s=60, cmap="viridis", alpha=0.8)
        for i, n in enumerate(names):
            ax.text(X[i], Y[i], Z[i], n, fontsize=7)
        ax.set_xlabel("X (semantic)", fontsize=11, labelpad=8)
        ax.set_ylabel("Y (semantic)", fontsize=11, labelpad=8)
        ax.set_zlabel("Z (abstract)", fontsize=11, labelpad=8)
        ax.set_title("Entity positions in idea space")
        out_3d = "world_3d.png"
        plt.savefig(out_3d, dpi=150)
        plt.close()
        print(f"Saved {out_3d}")
    except Exception as e:
        print(f"3D plot skipped: {e}")

    print(f"\nEntities: {len(entities)}, Relations: {len(relations)}")

if __name__ == "__main__":
    main()
