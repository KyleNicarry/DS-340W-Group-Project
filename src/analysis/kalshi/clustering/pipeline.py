"""Build economically constrained similarity groups from recent-year market CSVs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from .features import extract_attributes

SEED = 340
COLUMNS = ["ticker", "event_ticker", "market_type", "title", "yes_sub_title", "open_time", "close_time"]


def load_contracts(root: Path) -> tuple[pd.DataFrame, dict]:
    """Read only CSV metadata and preserve supplied event-disjoint splits."""
    frames, counts = [], {}
    for split in ("train", "validation", "test"):
        path = root / "split" / split / "markets.csv"
        frame = pd.read_csv(path, usecols=COLUMNS, dtype=str, keep_default_na=False)
        counts[split] = {"input": len(frame)}
        valid = (
            frame.market_type.eq("binary")
            & ~frame.ticker.str.startswith(("KXMVE", "MVE"))
            & frame.title.str.strip().ne("")
            & frame.event_ticker.str.strip().ne("")
        )
        frame = frame[valid].copy()
        counts[split]["retained"] = len(frame)
        frame["split"] = split
        frames.append(frame)
    contracts = pd.concat(frames, ignore_index=True)
    if contracts.ticker.duplicated().any():
        raise ValueError("Duplicate tickers in supplied splits")
    if contracts.groupby("event_ticker").split.nunique().gt(1).any():
        raise ValueError("Event IDs overlap supplied splits")
    attrs = pd.DataFrame([extract_attributes(row) for row in contracts.to_dict("records")])
    return pd.concat([contracts, attrs], axis=1), counts


def stable_sample(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    keys = frame.ticker.map(lambda value: hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest())
    return frame.loc[keys.sort_values().index[:limit]]


def embed(frame: pd.DataFrame, args) -> np.ndarray:
    """Fit LSA only on training data or use an explicitly supplied local model."""
    if args.backend == "sentence":
        if not args.model or not Path(args.model).is_dir():
            raise ValueError("Sentence backend requires --model pointing to a local sentence-transformers checkpoint")
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(str(args.model), local_files_only=True, device="cpu")
        return model.encode(frame.text.tolist(), normalize_embeddings=True, show_progress_bar=True)
    fit = stable_sample(frame[frame.split.eq("train")], args.fit_limit)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000, sublinear_tf=True)
    matrix = vectorizer.fit_transform(fit.text)
    dimensions = min(64, matrix.shape[1] - 1, matrix.shape[0] - 1)
    if dimensions < 2:
        raise ValueError("Insufficient training vocabulary/contracts for LSA")
    svd = TruncatedSVD(n_components=dimensions, random_state=SEED).fit(matrix)
    # Transform batches to bound peak dense memory.
    return np.vstack(
        [
            normalize(svd.transform(vectorizer.transform(frame.text.iloc[start : start + 10000]))).astype("float32")
            for start in range(0, len(frame), 10000)
        ]
    )


def group_contracts(frame: pd.DataFrame, vectors: np.ndarray, args) -> pd.DataFrame:
    """Fit bounded family hierarchies on train; reject distant or unsupported peers.

    Large families use a deterministic training sample to bound quadratic linkage
    memory. Held-out assignment uses frozen centroids, not reclustering. Minimum
    support counts distinct training events rather than correlated contracts.
    """
    result = frame.copy()
    result["cluster"] = ""
    result["status"] = "unseen_family"
    result["similarity"] = np.nan
    for family, members in frame.groupby("family", sort=True):
        train = members[members.split.eq("train")]
        if train.empty:
            continue
        sample = stable_sample(train, args.family_fit_limit)
        ix = sample.index.to_numpy()
        usable = np.linalg.norm(vectors[ix], axis=1) > 0
        ix = ix[usable]
        result.loc[members.index, "status"] = "insufficient_training_support"
        if len(ix) < 2:
            continue
        if args.method == "hierarchical":
            labels = AgglomerativeClustering(
                n_clusters=None, distance_threshold=args.distance, metric="cosine", linkage="average"
            ).fit_predict(vectors[ix])
        else:
            labels = KMeans(
                n_clusters=min(args.clusters, len(np.unique(vectors[ix], axis=0))), n_init=10, random_state=SEED
            ).fit_predict(vectors[ix])
        centers = normalize(np.vstack([vectors[ix[labels == label]].mean(axis=0) for label in np.unique(labels)]))
        train_ids = train.index.to_numpy()
        # Determine support using all training contracts, never validation/test counts.
        train_scores = vectors[train_ids] @ centers.T
        nearest = train_scores.argmax(axis=1)
        close = train_scores.max(axis=1) >= args.min_similarity
        supported = []
        for label in range(len(centers)):
            peers = train.iloc[np.flatnonzero((nearest == label) & close)]
            if len(peers) >= args.min_contracts and peers.event_ticker.nunique() >= args.min_events:
                supported.append(label)
        ids = members.index.to_numpy()
        scores = vectors[ids] @ centers.T
        nearest = scores.argmax(axis=1)
        similarity = scores.max(axis=1)
        result.loc[ids, "similarity"] = similarity
        result.loc[ids, "status"] = "distant"
        for label in range(len(centers)):
            mask = (nearest == label) & (similarity >= args.min_similarity)
            chosen = ids[mask]
            if label not in supported:
                result.loc[chosen, "status"] = "insufficient_training_support"
                continue
            result.loc[chosen, "cluster"] = f"{family}/{label:03d}"
            result.loc[chosen, "status"] = "assigned"
    return result


def write_reviews(frame: pd.DataFrame, output: Path) -> None:
    rows, reviews = [], []
    assigned = frame[frame.status.eq("assigned")]
    for (split, cluster), group in assigned.groupby(["split", "cluster"]):
        rows.append(
            {
                "split": split,
                "cluster": cluster,
                "contracts": len(group),
                "events": group.event_ticker.nunique(),
                "series": group.series.nunique(),
                "dominant_series_share": group.series.value_counts(normalize=True).iloc[0],
                "mean_centroid_cosine": group.similarity.mean(),
            }
        )
        for kind, examples in [
            ("central", group.nlargest(3, "similarity")),
            ("peripheral", group.nsmallest(3, "similarity")),
            ("random", stable_sample(group, 3)),
        ]:
            for row in examples.to_dict("records"):
                reviews.append(
                    {
                        "review_type": kind,
                        **{
                            k: row[k]
                            for k in [
                                "split",
                                "cluster",
                                "ticker",
                                "event_ticker",
                                "text",
                                "similarity",
                                "attribute_source",
                            ]
                        },
                    }
                )
    pd.DataFrame(rows).to_csv(output / "cluster_profiles.csv", index=False)
    pd.DataFrame(reviews).to_csv(output / "review_examples.csv", index=False)
    frame.groupby(["split", "status"]).size().rename("contracts").to_csv(output / "coverage.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/kalshi_recent_year_csv"))
    parser.add_argument("--output", type=Path, default=Path("output/clustering"))
    parser.add_argument("--backend", choices=["lsa", "sentence"], default="lsa")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--method", choices=["hierarchical", "kmeans"], default="hierarchical")
    parser.add_argument("--fit-limit", type=int, default=20000)
    parser.add_argument("--family-fit-limit", type=int, default=1000)
    parser.add_argument("--clusters", type=int, default=8, help="K-means clusters per family")
    parser.add_argument("--distance", type=float, default=0.35, help="Average-linkage cosine distance cutoff")
    parser.add_argument("--min-similarity", type=float, default=0.75)
    parser.add_argument("--min-contracts", type=int, default=20)
    parser.add_argument("--min-events", type=int, default=10)
    args = parser.parse_args()
    if min(args.fit_limit, args.family_fit_limit, args.clusters, args.min_contracts, args.min_events) < 1:
        parser.error("Counts and limits must be positive")
    if not 0 < args.distance <= 2 or not 0 <= args.min_similarity <= 1:
        parser.error("Require distance in (0, 2] and minimum similarity in [0, 1]")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "completed": False,
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "seed": SEED,
        "input_files": [],
    }
    for split in ["train", "validation", "test"]:
        path = args.data / "split" / split / "markets.csv"
        manifest["input_files"].append(
            {"path": str(path), "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        )
    manifest_path = args.output / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    with threadpool_limits(limits=2):
        frame, manifest["attrition"] = load_contracts(args.data)
        print(f"Representing {len(frame):,} contracts using {args.backend}", flush=True)
        vectors = embed(frame, args)
        result = group_contracts(frame, vectors, args)
        result.to_csv(args.output / "assignments.csv", index=False)
        write_reviews(result, args.output)
    manifest["completed"] = True
    manifest["note"] = (
        "Heuristic families; event-disjoint supplied split, not chronological or unseen-series evaluation."
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(result.groupby(["split", "status"]).size().to_string())
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
