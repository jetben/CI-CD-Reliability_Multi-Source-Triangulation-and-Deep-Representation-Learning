# visualize_with_labels.py
# Usage:
#   python visualize_with_labels.py --outdir outputs --topk 12 --terms 3
#   python visualize_with_labels.py --outdir outputs --topk 12 --terms 3 --use_name_map 1

import argparse, textwrap
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def short_label(top_terms: str, n=3, wrap=28):
    words = [w.strip() for w in str(top_terms).split(",") if w.strip()]
    label = ", ".join(words[:n]) if words else "—"
    return "\n".join(textwrap.wrap(label, width=wrap))

def build_labels(outdir: Path, topk: int, terms: int, use_name_map: int):
    topics = pd.read_csv(outdir / "topics.csv")               # Topic, Count, ...
    top_terms = pd.read_csv(outdir / "topic_top_terms.csv")   # topic, top_terms

    # Top-k by size (exclude -1 outliers)
    sizes = topics[topics["Topic"] >= 0][["Topic","Count"]].sort_values("Count", ascending=False).head(topk)
    df = sizes.merge(top_terms, left_on="Topic", right_on="topic", how="left")

    # Auto labels from top terms
    df["auto_label"] = df["top_terms"].apply(lambda s: short_label(s, n=terms))

    # Optional: manual overrides for the most important topics (edit here if needed)
    name_map = {
        # 0: "GitHub Actions workflows",
        # 1: "Jenkins pipelines",
        # 2: "Travis CI configuration",
        # 3: "GitLab CI YAML",
        # ...
    }
    if use_name_map:
        df["label"] = df.apply(lambda r: name_map.get(int(r["Topic"]), r["auto_label"]), axis=1)
    else:
        df["label"] = df["auto_label"]

    return df[["Topic","Count","label","top_terms"]]

def plot_barh(df: pd.DataFrame, outdir: Path):
    d = df.sort_values("Count", ascending=True)
    plt.figure(figsize=(10, max(6, 0.5*len(d))))
    plt.barh(d["label"], d["Count"])
    plt.xlabel("Documents")
    plt.title(f"Top {len(d)} Topics by Size (readable labels)")
    plt.tight_layout()
    p = outdir / "fig_topic_sizes_labeled.png"
    plt.savefig(p, dpi=150); plt.close()
    print("✅ saved:", p)

def plot_trends_with_labels(outdir: Path, df_labels: pd.DataFrame, topk_lines: int = 8):
    path = outdir / "topics_over_time.csv"
    if not path.exists():
        print("ℹ️ topics_over_time.csv not found — skipping trends plot.")
        return

    tot = pd.read_csv(path)  # columns: Topic, Timestamp, Frequency
    # Keep only topics we have labels for (topk)
    keep = set(df_labels["Topic"].tolist()[:topk_lines])
    tot = tot[tot["Topic"].isin(keep)]

    # Map topic -> label
    lab_map = dict(zip(df_labels["Topic"], df_labels["label"]))

    plt.figure(figsize=(11,6))
    for t in sorted(keep, key=lambda x: int(x)):
        sub = tot[tot["Topic"] == t].sort_values("Timestamp")
        if sub.empty:
            continue
        plt.plot(pd.to_datetime(sub["Timestamp"]), sub["Frequency"], label=lab_map.get(t, f"T{t}"))
    plt.legend(ncol=2, fontsize=8)
    plt.xlabel("Time"); plt.ylabel("Frequency")
    plt.title("Topic Trends Over Time (labeled)")
    plt.tight_layout()
    p = outdir / "fig_topics_over_time_labeled.png"
    plt.savefig(p, dpi=150); plt.close()
    print("✅ saved:", p)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--terms", type=int, default=3, help="#terms to build short label")
    ap.add_argument("--use_name_map", type=int, default=0, help="1=apply manual overrides in name_map{}")
    ap.add_argument("--trend_lines", type=int, default=8, help="max lines to plot in trends figure")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    df_labels = build_labels(outdir, args.topk, args.terms, args.use_name_map)

    # Save a table for the paper (ID → Label → Docs → Top terms)
    table = df_labels.rename(columns={
        "Topic": "Topic ID", "label": "Short Label", "Count": "Documents", "top_terms": "Top Terms"
    })
    csv_p = outdir / "table_topic_labels.csv"
    table.to_csv(csv_p, index=False)
    print("✅ saved:", csv_p)

    # Plots with labels
    plot_barh(df_labels, outdir)
    plot_trends_with_labels(outdir, df_labels, topk_lines=args.trend_lines)

if __name__ == "__main__":
    main()
