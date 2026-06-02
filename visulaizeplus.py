# visualize_ci_cd_plus.py
# Usage:
#   python visualize_ci_cd_plus.py --outputs_dir outputs --topk 12 --make_wordclouds 1

import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Optional word cloud
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except Exception:
    WORDCLOUD_AVAILABLE = False


def load_outputs(outdir: str):
    outdir = Path(outdir)
    topics = pd.read_csv(outdir / "topics.csv")
    top_terms = pd.read_csv(outdir / "topic_top_terms.csv")
    docs = pd.read_csv(outdir / "docs_with_topics.csv")
    tovertime_path = outdir / "topics_over_time.csv"
    topics_over_time = pd.read_csv(tovertime_path) if tovertime_path.exists() else None
    return topics, top_terms, docs, topics_over_time


def compute_monthly_share_from_docs(docs_df: pd.DataFrame) -> pd.DataFrame:
    df = docs_df.copy()
    df["month"] = pd.to_datetime(df["created_at"]).dt.to_period("M").astype(str)
    monthly = df.groupby(["month", "topic"]).size().reset_index(name="count")
    totals = monthly.groupby("month")["count"].sum().rename("total").reset_index()
    merged = monthly.merge(totals, on="month", how="left")
    merged["share"] = merged["count"] / merged["total"]
    return merged


def fig_top_topics(ax, topics_df: pd.DataFrame, topk: int = 12):
    df = topics_df[topics_df["Topic"] >= 0].sort_values("Count", ascending=False).head(topk)
    ax.bar(df["Topic"].astype(str), df["Count"])
    ax.set_xlabel("Topic ID")
    ax.set_ylabel("Documents")
    ax.set_title(f"Top {len(df)} Topics by Size")
    ax.tick_params(axis="x", rotation=90)


def fig_topics_over_time(ax, topics_over_time: pd.DataFrame, topk: int = 6):
    totals = topics_over_time.groupby("Topic")["Frequency"].sum().sort_values(ascending=False)
    top_topics = list(totals.head(topk).index)
    for t in top_topics:
        sub = topics_over_time[topics_over_time["Topic"] == t].sort_values("Timestamp")
        ax.plot(pd.to_datetime(sub["Timestamp"]), sub["Frequency"], label=f"T{t}")
    ax.legend(ncol=2, fontsize=8)
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency")
    ax.set_title("Topic Trends Over Time")


def fig_heatmap(ax, month_share: pd.DataFrame, topk: int = 12):
    counts = month_share.groupby("topic")["count"].sum().sort_values(ascending=False)
    keep = set(counts.head(topk).index)
    df = month_share[month_share["topic"].isin(keep)]
    pivot = df.pivot(index="month", columns="topic", values="share").fillna(0.0)
    im = ax.imshow(pivot.values, aspect="auto", interpolation="nearest")
    ax.set_title("Topic Share by Month (Top Topics)")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"T{t}" for t in pivot.columns], rotation=90)
    plt.colorbar(im, ax=ax, label="Share")


def export_tables(topics_df: pd.DataFrame, top_terms_df: pd.DataFrame, outdir: Path, topk: int = 12):
    sizes = topics_df[topics_df["Topic"] >= 0][["Topic", "Count"]].sort_values("Count", ascending=False).head(topk)
    merged = sizes.merge(top_terms_df, left_on="Topic", right_on="topic", how="left")
    merged = merged.rename(columns={"Count": "Documents", "top_terms": "Top terms"}).drop(columns=["topic"])
    # CSV & Markdown
    merged.to_csv(outdir / "table_top_topics.csv", index=False)
    md_lines = ["| Topic | Documents | Top terms |", "|---:|---:|---|"]
    for _, row in merged.iterrows():
        md_lines.append(f"| {int(row['Topic'])} | {int(row['Documents'])} | {row['Top terms']} |")
    (outdir / "table_top_topics.md").write_text("\n".join(md_lines), encoding="utf-8")
    # LaTeX tabular
    latex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{r r p{9cm}}",
        r"\hline",
        r"Topic & Documents & Top terms \\ \hline"
    ]
    for _, row in merged.iterrows():
        topic = int(row["Topic"])
        docs = int(row["Documents"])
        terms = str(row["Top terms"]).replace("&", r"\&")
        latex.append(f"{topic} & {docs} & {terms} \\\\")
    latex += [r"\hline", r"\end{tabular}", r"\caption{Top CI/CD topics discovered by BERTopic.}", r"\label{tab:top_topics}", r"\end{table}"]
    (outdir / "table_top_topics.tex").write_text("\n".join(latex), encoding="utf-8")
    print(f"saved: {outdir / 'table_top_topics.csv'}")
    print(f"saved: {outdir / 'table_top_topics.md'}")
    print(f"saved: {outdir / 'table_top_topics.tex'}")


def make_wordclouds(top_terms_df: pd.DataFrame, outdir: Path, topk: int = 8):
    """
    Creates a PNG word cloud per top topic using the comma-separated 'top_terms' list.
    Uses uniform weights (we don't have c-TF-IDF weights in the CSV).
    """
    if not WORDCLOUD_AVAILABLE:
        print("⚠️ wordcloud not installed. Run: pip install wordcloud")
        return
    # pick top-k topics by order in file (assumed sorted by size from previous export)
    df = top_terms_df.head(topk)
    for _, row in df.iterrows():
        t = int(row["topic"])
        terms = [w.strip() for w in str(row["top_terms"]).split(",") if w.strip()]
        # Build a frequency dict with uniform weights
        freqs = {w: 1.0 for w in terms}
        wc = WordCloud(width=1200, height=800, background_color="white").generate_from_frequencies(freqs)
        out_path = outdir / f"wordcloud_T{t}.png"
        wc.to_file(str(out_path))
        print(f"saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_dir", type=str, default="outputs")
    parser.add_argument("--topk", type=int, default=12)
    parser.add_argument("--make_wordclouds", type=int, default=1, help="1 to generate word clouds")
    args = parser.parse_args()

    outdir = Path(args.outputs_dir)
    topics, top_terms, docs, topics_over_time = load_outputs(outdir)

    # ==== Tables (CSV, MD, LaTeX)
    export_tables(topics, top_terms, outdir, topk=args.topk)

    # ==== Multi-panel PDF
    pdf_path = outdir / "paper_figures.pdf"
    with PdfPages(pdf_path) as pdf:
        # Figure 1: Top topics bar
        fig, ax = plt.subplots(figsize=(10, 6))
        fig_top_topics(ax, topics, topk=args.topk)
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # Figure 2: Trends or Heatmap
        if topics_over_time is not None and not topics_over_time.empty:
            fig, ax = plt.subplots(figsize=(11, 6))
            fig_topics_over_time(ax, topics_over_time, topk=min(args.topk, 8))
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)
        else:
            # compute monthly share if needed
            month_share = compute_monthly_share_from_docs(docs)
            month_share.to_csv(outdir / "topics_monthly_share.csv", index=False)
            fig, ax = plt.subplots(figsize=(12, 7))
            fig_heatmap(ax, month_share, topk=args.topk)
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)

    print(f"saved: {pdf_path}")

    # ==== Word clouds
    if args.make_wordclouds:
        make_wordclouds(top_terms, outdir, topk=min(args.topk, 8))


if __name__ == "__main__":
    main()
