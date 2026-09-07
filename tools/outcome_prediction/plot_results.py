"""Render a compact scientific figure from frozen prediction artifacts."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def main():
    root = Path("data/analysis/outcome_prediction")
    grouped = json.loads((root / "grouped/metrics.json").read_text())
    nested = json.loads((root / "sensitivity/metrics.json").read_text())["nested"]
    icl = json.loads((root / "icl_combined/metrics.json").read_text())
    rows = [
        json.loads(s)
        for s in (root / "sensitivity/nested_predictions.jsonl").read_text().splitlines()
    ]
    examples = {
        r["example_id"]: r
        for r in [json.loads(s) for s in (root / "examples.jsonl").read_text().splitlines()]
    }
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")
    palette = {"claude-opus-4-8": "#326aab", "claude-opus-5": "#d27832"}
    for model, color in palette.items():
        subset = [r for r in rows if examples[r["example_id"]]["scientist_model"] == model]
        axes[0].scatter(
            [r["y"] for r in subset],
            [r["predictions"]["recipe_text_nested"] for r in subset],
            s=28,
            alpha=0.7,
            color=color,
            label=model.replace("claude-", ""),
        )
    axes[0].plot([0, 0.9], [0, 0.9], color="#555555", linewidth=1, linestyle="--")
    axes[0].set(
        xlim=(0, 0.9),
        ylim=(0, 0.9),
        xlabel="Actual official accuracy",
        ylabel="Predicted accuracy",
        title="Recipe regression: held-out runs\n146 checkpoints, 32 runs",
    )
    axes[0].xaxis.set_major_formatter(PercentFormatter(1))
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(frameon=False, loc="upper left")
    labels = ["Training mean", "Scientist identity", "Current recipe", "Recipe lineage"]
    values = [
        grouped["methods"]["train_mean"]["mae"],
        grouped["methods"]["scientist_only_nuisance"]["mae"],
        nested["methods"]["recipe_text_nested"]["mae"],
        nested["methods"]["lineage_text_nested"]["mae"],
    ]
    for ax, names, numbers, title in [
        (axes[1], labels, values, "Grouped regression comparison\n146 checkpoints, 32 runs"),
        (
            axes[2],
            ["Zero-shot LLM", "Scientist identity", "Nearest recipes", "In-context LLM"],
            [
                icl["methods"][k]["mae"]
                for k in [
                    "zero_shot",
                    "scientist_only_nuisance",
                    "recipe_tfidf_neighbors",
                    "few_shot",
                ]
            ],
            "Blinded in-context comparison\n107 checkpoints, 24 held-out runs",
        ),
    ]:
        ax.barh(
            names[::-1],
            [x * 100 for x in numbers[::-1]],
            color=["#237b65", "#326aab", "#969696", "#c4c4c4"],
        )
        ax.set(xlabel="Mean absolute error (percentage points)", title=title, xlim=(0, 20))
        for i, val in enumerate(numbers[::-1]):
            ax.text(val * 100 + 0.3, i, f"{val * 100:.1f}", va="center")
        ax.grid(axis="x", alpha=0.15)
        ax.set_axisbelow(True)
    fig.suptitle(
        "Experiment recipes predict broad checkpoint quality; scientist effects remain substantial",
        fontsize=13,
    )
    fig.savefig(root / "overview.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
