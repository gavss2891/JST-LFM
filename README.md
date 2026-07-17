# Review-Aware Recommender Models

This project studies whether adding **sentiment awareness** to topic models
helps predict how a user will rate a product, using Amazon product reviews.

The core idea: reviews contain more than a star rating — they contain text
that expresses *what* the reviewer talks about and *how they feel* about it.
This project builds a series of recommender models of increasing
sophistication to see how much that extra information actually helps.

## What's in here

**Data pipeline**
- `data_preprocessing.py` — loads the Amazon review data, cleans the review
  text, and splits it into training / validation / test sets.

**Models** (from simplest to most sophisticated)
- `LFM.py` — a plain rating-prediction model, using only past ratings
  (no review text at all). This is the baseline everything else is compared
  against.
- `LDAFirst.py` — first discovers topics in the reviews (what products are
  talked about), then uses those topics as extra information for the rating
  model.
- `LDA_LFM.py` — same idea as above, but the topic discovery and the rating
  prediction are trained together instead of one after the other, so each
  informs the other.
- `JSTFirst.py` — like `LDAFirst.py`, but the topic model also separates
  *positive*, *negative*, and *neutral* sentiment for each topic.
- `JST_LFM.py` — the sentiment-aware version of `LDA_LFM.py`: sentiment
  and topic discovery are trained jointly with the rating model.
- `JST_LFM_asymmetric.py` — a refinement of `JST_LFM.py` that allows a
  different number of topics per sentiment (e.g. more nuance in negative
  reviews than positive ones), instead of a fixed number for all three.

**Running everything**
- `main.py` — runs the full pipeline (load data → clean text → train every
  model → evaluate → save results) for a list of product categories.
- `run_kstar_sweep.py` — a follow-up experiment that tests how much extra
  benefit comes from giving models a few additional "free" rating-only
  dimensions on top of the topic/sentiment ones.

**Summarising results**
- `summarise_results.py` — collects each dataset's results into comparison
  tables (e.g. which model performed best, and by how much).
- `aggregate_ks_results.py` — summarises the topic-count experiments for the
  asymmetric sentiment model.

## How to run it

1. Place the Amazon review files (`.json.gz`) in a `Data/` folder.
2. Run the full pipeline:
   ```
   python main.py
   ```
   This processes each dataset in turn and writes results to
   `results/<dataset name>/`.
3. Once several datasets have been processed, generate the comparison
   tables:
   ```
   python summarise_results.py
   ```
   Output tables land in `results/summary/`.

## What comes out

For each dataset, you get:
- A CSV with each model's final accuracy (error between predicted and
  actual star ratings — lower is better).
- The hyperparameter settings used to get that result.
- Plots showing how each model improved over training.
- The discovered topics/sentiments and their most representative words, for
  inspection.

The summary tables then line datasets and models up side by side, showing
which approach (plain ratings, topics, or topics + sentiment) wins, and by
how much, across product categories.
