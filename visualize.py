# visualize_ci_cd.py
# Usage:
#   python visualize_ci_cd.py --outputs_dir outputs --topk 12

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_outputs(outdir: str):
    outdir = Path(outdir)
    topics = pd.read_csv(outdir / "topics.csv")
    top_terms = pd.read_csv(outdir / "topic_top_terms.csv")
    docs = pd.read_csv(outdir / "docs_with_topics.csv")
    tovertime_path = outdir / "topics_over_time.csv"
    topics_over_time = pd.read_csv(tovertime_path) if tovertime_path.exists() else None
    return topics, top_terms, docs, topics_over_time

def plot_top_topics(topics_df: pd.DataFrame, outdir: str, topk: int = 12):
    df = topics_df[topics_df["Topic"] >= 0].sort_values("Count", ascending=False).head(topk)
    plt.figure(figsize=(10, 6))
    plt.bar(df["Topic"].astype(str), df["Count"])
    plt.xlabel("Topic ID")
    plt.ylabel("Documents")
    plt.title(f"Top {len(df)} Topics by Size")
    plt.tight_layout()
    out = Path(outdir) / "viz_top_topics.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"saved: {out}")

def prep_monthly_share_from_docs(docs_df: pd.DataFrame) -> pd.DataFrame:
    # expects docs_df to have 'created_at' (YYYY-MM-DD) and 'topic'
    df = docs_df.copy()
    df["month"] = pd.to_datetime(df["created_at"]).dt.to_period("M").astype(str)
    monthly = df.groupby(["month", "topic"]).size().reset_index(name="count")
    totals = monthly.groupby("month")["count"].sum().rename("total").reset_index()
    merged = monthly.merge(totals, on="month")
    merged["share"] = merged["count"] / merged["total"]
    return merged

def plot_topics_over_time(topics_over_time: pd.DataFrame, outdir: str, focus_topics=None, topk: int = 6):
    """
    topics_over_time columns (BERTopic): Topic, Timestamp, Words, Frequency
    We’ll plot Frequency over time for selected topics.
    """
    df = topics_over_time.copy()
    # find top topics by total frequency
    totals = df.groupby("Topic")["Frequency"].sum().sort_values(ascending=False)
    top_topics = list(totals.head(topk).index) if focus_topics is None else focus_topics

    plt.figure(figsize=(11, 6))
    for t in top_topics:
        sub = df[df["Topic"] == t].sort_values("Timestamp")
        plt.plot(pd.to_datetime(sub["Timestamp"]), sub["Frequency"], label=f"T{t}")
    plt.legend(ncol=2, fontsize=9)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.title("Topic Trends Over Time")
    plt.tight_layout()
    out = Path(outdir) / "viz_topics_over_time.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"saved: {out}")

def plot_heatmap_month_share(month_share: pd.DataFrame, outdir: str, topk: int = 12):
    # pivot to month x topic (share)
    # choose topk by overall count to keep it readable
    counts = month_share.groupby("topic")["count"].sum().sort_values(ascending=False)
    keep = set(counts.head(topk).index)
    df = month_share[month_share["topic"].isin(keep)]
    pivot = df.pivot(index="month", columns="topic", values="share").fillna(0.0)
    # simple image heatmap (no seaborn)
    plt.figure(figsize=(12, 7))
    plt.imshow(pivot.values, aspect="auto", interpolation="nearest")
    plt.colorbar(label="Share")
    plt.yticks(ticks=range(len(pivot.index)), labels=pivot.index)
    plt.xticks(ticks=range(len(pivot.columns)), labels=[f"T{t}" for t in pivot.columns], rotation=90)
    plt.title("Topic Share by Month (Top Topics)")
    plt.tight_layout()
    out = Path(outdir) / "viz_heatmap_topics_by_month.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"saved: {out}")

def export_markdown_table(topics_df: pd.DataFrame, top_terms_df: pd.DataFrame, outdir: str, topk: int = 12):
    # Merge sizes + top terms; export a neat table for paper
    sizes = topics_df[topics_df["Topic"] >= 0][["Topic", "Count"]].sort_values("Count", ascending=False).head(topk)
    merged = sizes.merge(top_terms_df, left_on="Topic", right_on="topic", how="left")
    merged = merged.rename(columns={"Count": "Documents", "top_terms": "Top terms"}).drop(columns=["topic"])
    merged.to_csv(Path(outdir) / "table_top_topics.csv", index=False)

    # simple markdown
    md_lines = ["| Topic | Documents | Top terms |", "|---:|---:|---|"]
    for _, row in merged.iterrows():
        md_lines.append(f"| {int(row['Topic'])} | {int(row['Documents'])} | {row['Top terms']} |")
    (Path(outdir) / "table_top_topics.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"saved: {Path(outdir) / 'table_top_topics.csv'}")
    print(f"saved: {Path(outdir) / 'table_top_topics.md'}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_dir", type=str, default="outputs")
    parser.add_argument("--topk", type=int, default=12)
    args = parser.parse_args()

    topics, top_terms, docs, topics_over_time = load_outputs(args.outputs_dir)

    # 1) bar chart of top-N topics
    plot_top_topics(topics, args.outputs_dir, topk=args.topk)

    # 2) trends over time
    if topics_over_time is not None and not topics_over_time.empty:
        plot_topics_over_time(topics_over_time, args.outputs_dir, topk=min(args.topk, 8))
    else:
        # compute monthly shares if topics_over_time.csv wasn’t generated
        print("no topics_over_time.csv found — computing monthly share from docs_with_topics.csv…")
        month_share = prep_monthly_share_from_docs(docs)
        month_share.to_csv(Path(args.outputs_dir) / "topics_monthly_share.csv", index=False)
        plot_heatmap_month_share(month_share, args.outputs_dir, topk=args.topk)

    # 3) export table (CSV + Markdown) of top topics with top terms
    export_markdown_table(topics, top_terms, args.outputs_dir, topk=args.topk)

if __name__ == "__main__":
    main()
