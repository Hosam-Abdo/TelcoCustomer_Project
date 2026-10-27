import matplotlib.pyplot as plt

def plot_ranked_bars(values, classes, xlabel, title, filename, legend_title,
                     fmt="%.1f", not_significant=None, alpha=None, min_xmax=1):
    values = values.sort_values(ascending=False)

    def bar_color(v):
        for lower, color, _ in classes:
            if v >= lower:
                return color
        return None  

    fig, ax = plt.subplots(figsize=(0.9 * len(values) + 3, 7))

    bars = ax.bar(
        values.index, values.values,
        color=[bar_color(v) for v in values],
        edgecolor="#444444", linewidth=0.5,
        width=0.7,
    )

    if not_significant is not None:
        for bar, feat in zip(bars, values.index):
            if feat in not_significant:
                bar.set_hatch("//")

    ax.bar_label(bars, fmt=fmt, padding=3)

    # Legend
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c,
                             edgecolor="#444444", linewidth=0.5)
               for _, c, _ in classes]
    labels = [label for _, _, label in classes]
    if not_significant is not None:
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="white",
                                     edgecolor="#444444", hatch="//",
                                     linewidth=0.5))
        labels.append(f"not significant (p ≥ {alpha})")
    ax.legend(handles, labels, title=legend_title, loc="upper right")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("")  
    ax.set_title(title)
    ax.set_ylim(0, max(values.max() * 1.15, min_xmax))

    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")

    ax.grid(axis="y", linestyle=":", linewidth=0.8, color="gray", alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)