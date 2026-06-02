
# so_survival_sentiment.py
import argparse, os, sys
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

def log(*a): print(*a, file=sys.stdout, flush=True)

def load_csv_with_candidates(label, candidates):
  for p in candidates:
      if p and os.path.exists(p):
          log(f"📥 Using {label}: {p}")
          return pd.read_csv(p)
  raise FileNotFoundError(f"Could not find {label}. Tried: {candidates}")

def compute_sentiment(df, text_field="body"):
  try:
      from nltk.sentiment.vader import SentimentIntensityAnalyzer
      from nltk import download as nltk_download
      nltk_download("vader_lexicon", quiet=True)
      sid = SentimentIntensityAnalyzer()
      return df[text_field].astype(str).apply(lambda x: sid.polarity_scores(x)["compound"])
  except Exception as e:
      log(f"⚠️ Sentiment disabled ({e}). Filling zeros.")
      return pd.Series([0.0]*len(df))

def coerce_datetime_cols(df):
  # Prefer StackExchange naming if present, else our CSV fields
  if "creation_date" in df.columns and df["creation_date"].notna().any():
      df["created_at"] = pd.to_datetime(df["creation_date"], unit="s", errors="coerce")
  else:
      df["created_at"] = pd.to_datetime(df.get("created_at"), errors="coerce")

  if "last_activity_date" in df.columns and df["last_activity_date"].notna().any():
      # can be epoch seconds or ISO
      # try epoch first
      lad = pd.to_numeric(df["last_activity_date"], errors="coerce")
      iso = pd.to_datetime(df["last_activity_date"], errors="coerce")
      lad_dt = pd.to_datetime(lad, unit="s", errors="coerce")
      df["last_activity_date"] = lad_dt.fillna(iso)
  else:
      df["last_activity_date"] = df["created_at"]

  return df

def infer_event(df):
  # event = 1 if answered; try columns in this order
  if "is_answered" in df.columns:
      return df["is_answered"].fillna(False).astype(int)
  if "accepted_answer_id" in df.columns:
      return df["accepted_answer_id"].notna().astype(int)
  if "answer_count" in df.columns:
      return (pd.to_numeric(df["answer_count"], errors="coerce").fillna(0) > 0).astype(int)
  return pd.Series([0]*len(df))

def build_survival_data(df):
  df = coerce_datetime_cols(df)
  df["duration"] = (df["last_activity_date"] - df["created_at"]).dt.days
  df["duration"] = df["duration"].fillna(0).clip(lower=0)
  df["event"] = infer_event(df)
  return df

def survival_plot(df, out_path):
  kmf = KaplanMeierFitter()
  plt.figure(figsize=(10,6))
  any_plotted = False
  for topic in sorted([t for t in df["topic"].dropna().unique() if t != -1]):
      sub = df[df["topic"] == topic]
      if len(sub) < 20:
          continue
      kmf.fit(durations=sub["duration"], event_observed=sub["event"], label=f"T{int(topic)}")
      kmf.plot_survival_function(ci_show=False)
      any_plotted = True

  if not any_plotted:
      log("ℹ️ Not enough per-topic data; plotting overall survival.")
      kmf.fit(durations=df["duration"], event_observed=df["event"], label="All topics")
      kmf.plot_survival_function(ci_show=False)

  plt.title("Survival Analysis of Time-to-Resolution")
  plt.xlabel("Days"); plt.ylabel("P(Question still unresolved)")
  plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
  log(f"✅ saved: {out_path}")

def barplot_mean_sentiment(summary, out_path):
  s = summary.dropna(subset=["topic"]).copy()
  s = s[s["topic"] != -1]
  if s.empty: return
  s = s.sort_values("mean_sentiment", ascending=True)
  plt.figure(figsize=(10, max(6, 0.5*len(s))))
  plt.barh([f"T{int(t)}" for t in s["topic"]], s["mean_sentiment"])
  plt.xlabel("Mean sentiment"); plt.title("Mean Sentiment by Topic")
  plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
  log(f"✅ saved: {out_path}")

def barplot_resolution_rate(summary, out_path):
  s = summary.dropna(subset=["topic"]).copy()
  s = s[s["topic"] != -1]
  if s.empty: return
  s = s.sort_values("resolution_rate", ascending=False)
  plt.figure(figsize=(10, max(6, 0.5*len(s))))
  plt.barh([f"T{int(t)}" for t in s["topic"]], s["resolution_rate"])
  plt.xlabel("Resolution rate"); plt.title("Resolution Rate by Topic")
  plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
  log(f"✅ saved: {out_path}")

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--outdir", type=str, default="outputs")
  ap.add_argument("--so_csv", type=str, default=None)
  ap.add_argument("--docs_with_topics_csv", type=str, default=None)
  ap.add_argument("--text_field", type=str, default="body")
  args = ap.parse_args()

  os.makedirs(args.outdir, exist_ok=True)

  so_df = load_csv_with_candidates("so_posts.csv",
           [args.so_csv, os.path.join(args.outdir, "so_posts.csv"), os.path.join("data","so_posts.csv")])

  dwt_df = load_csv_with_candidates("docs_with_topics.csv",
           [args.docs_with_topics_csv, os.path.join(args.outdir, "docs_with_topics.csv")])

  # Ensure 'id' types align for merge
  for df in (so_df, dwt_df):
      if "id" in df.columns:
          df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")

  df = pd.merge(so_df, dwt_df[["id","topic"]], on="id", how="left")

  # Sentiment
  text_field = args.text_field if args.text_field in df.columns else "body"
  df["sentiment"] = compute_sentiment(df, text_field=text_field)
  so_sent_path = os.path.join(args.outdir, "so_with_sentiment.csv")
  df.to_csv(so_sent_path, index=False)

  # Survival data
  df = build_survival_data(df)

  # Save plots and summary
  surv_img = os.path.join(args.outdir, "viz_resolution_survival.png")
  survival_plot(df, surv_img)

  summary = df.groupby("topic", dropna=False).agg(
      mean_sentiment=("sentiment","mean"),
      mean_duration_days=("duration","mean"),
      resolution_rate=("event","mean"),
      n=("id","count")
  ).reset_index()
  summary_path = os.path.join(args.outdir, "resolution_summary.csv")
  summary.to_csv(summary_path, index=False)

  # Extra visualizations
  barplot_mean_sentiment(summary, os.path.join(args.outdir,"viz_mean_sentiment_by_topic.png"))
  barplot_resolution_rate(summary, os.path.join(args.outdir,"viz_resolution_rate_by_topic.png"))

  log("\n📂 Outputs created:")
  for p in [so_sent_path, surv_img, summary_path,
            os.path.join(args.outdir,"viz_mean_sentiment_by_topic.png"),
            os.path.join(args.outdir,"viz_resolution_rate_by_topic.png")]:
      log("  -", p)
  log(f"\n🔎 Check: {os.path.abspath(args.outdir)}")

if __name__ == "__main__":
  main()
