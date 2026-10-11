import os
from collections import defaultdict

import numpy as np
import pandas as pd


class ResultsManager:
    """Persist per-file results immediately and aggregate them across a run."""

    _METRICS = ["AUC-ROC", "AUC-PR", "VUS-PR", "VUS-ROC", "BestF1", "RangeF1"]
    _RUNTIME_METRICS = ["Train_Time_s", "Infer_Time_s"]
    _COMPONENT_TIME_METRICS = ["Graph_Time_s", "Cluster_Time_s"]
    _TIME_METRICS = [*_RUNTIME_METRICS, *_COMPONENT_TIME_METRICS]
    _AVERAGE_TIME_METRICS = {
        "Train_Time_s": "Avg_Train_Time_s",
        "Infer_Time_s": "Avg_Infer_Time_s",
        "Graph_Time_s": "Avg_Graph_Time_s",
        "Cluster_Time_s": "Avg_Cluster_Time_s",
    }
    _SUMMARY_COLUMNS = ["file", "Category", *_METRICS, *_TIME_METRICS]
    _CATEGORICAL_COLUMNS = [
        "Category", "N", *_METRICS, *_AVERAGE_TIME_METRICS.values()
    ]

    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.scores_dir = os.path.join(output_dir, "Filewise_scores")
        os.makedirs(self.scores_dir, exist_ok=True)

        self.summary_path = os.path.join(output_dir, "summary_metrics.csv")
        self.categorical_path = os.path.join(output_dir, "categorical_metrics.csv")

        self._summary_rows = []
        self._cat_rows = defaultdict(list)
        self._file_count = 0
        self._completed_categories = []

    @staticmethod
    def _is_finite_number(value):
        try:
            return bool(np.isfinite(float(value)))
        except (TypeError, ValueError):
            return False

    @classmethod
    def _format_rows(cls, rows):
        """Build a stable CSV frame and round timing values like TSB-AD."""
        frame = pd.DataFrame(rows, columns=cls._SUMMARY_COLUMNS)
        for metric in cls._TIME_METRICS:
            frame[metric] = [
                round(float(value), 6) if cls._is_finite_number(value) else "N/A"
                for value in frame[metric]
            ]
        return frame

    def _rebuild_categories(self):
        self._cat_rows = defaultdict(list)
        for row in self._summary_rows:
            self._cat_rows[row["Category"]].append(row)
        self._file_count = len(self._summary_rows)

    def _write_running_summary(self):
        self._format_rows(self._summary_rows).to_csv(self.summary_path, index=False)

    def restore_from_disk(self):
        """Rebuild in-memory state and migrate legacy summaries for resume."""
        if not os.path.exists(self.summary_path):
            return

        df = pd.read_csv(self.summary_path)
        # finalize() recomputes AVERAGE; keep the last row if a crash duplicated a file.
        df = df[df["file"] != "AVERAGE"].drop_duplicates(subset="file", keep="last").copy()

        for column in self._SUMMARY_COLUMNS:
            if column not in df.columns:
                df[column] = np.nan
        df = df.reindex(columns=self._SUMMARY_COLUMNS)
        for metric in self._TIME_METRICS:
            df[metric] = df[metric].where(df[metric].notna(), "N/A")

        self._summary_rows = df.to_dict("records")
        self._rebuild_categories()
        # Canonicalize the schema before a resumed run appends/replaces any row.
        self._write_running_summary()

    def has_timing(self, file_name):
        """Preserve resume behavior for legacy rows with both total runtimes."""
        for row in reversed(self._summary_rows):
            if row["file"] == file_name:
                return all(self._is_finite_number(row.get(metric))
                           for metric in self._RUNTIME_METRICS)
        return False

    def save_file_result(self, file_name, category, results, full_labels, dist_scores,
                         train_time_s, infer_time_s,
                         graph_time_s=None, cluster_time_s=None):
        """Persist one file's scores, metrics, and phase timings."""
        score_path = os.path.join(self.scores_dir, f"{file_name}_output.csv")
        pd.DataFrame({
            "True Labels": full_labels,
            "Anomaly scores": dist_scores,
        }).to_csv(score_path, index=False)

        row = {
            "file": file_name,
            "Category": category,
            "AUC-ROC": results["AUC-ROC"],
            "AUC-PR": results["AUC-PR"],
            "VUS-PR": results["VUS-PR"],
            "VUS-ROC": results["VUS-ROC"],
            "BestF1": results["Standard-F1"],
            "RangeF1": results["R-based-F1"],
            # Retain full precision in memory; round only at CSV boundaries.
            "Train_Time_s": float(train_time_s),
            "Infer_Time_s": float(infer_time_s),
            "Graph_Time_s": float(graph_time_s) if self._is_finite_number(graph_time_s) else "N/A",
            "Cluster_Time_s": float(cluster_time_s) if self._is_finite_number(cluster_time_s) else "N/A",
        }

        # Upsert by filename so a crash between this write and checkpoint update
        # cannot make a resumed run count the same file twice.
        self._summary_rows = [
            existing for existing in self._summary_rows
            if existing["file"] != file_name
        ]
        self._summary_rows.append(row)
        self._rebuild_categories()
        self._write_running_summary()

    @classmethod
    def _mean_time(cls, rows, metric):
        values = [
            float(row[metric]) for row in rows
            if cls._is_finite_number(row.get(metric))
        ]
        # Do not present partial component coverage as a whole-set average.
        if not values or (metric in cls._COMPONENT_TIME_METRICS and len(values) != len(rows)):
            return "N/A"
        return round(float(np.mean(values)), 6)

    def flush_category_complete(self, category_name):
        """Write a categorical snapshot after one category completes."""
        self._completed_categories.append(category_name)
        self._write_categorical()
        print(f"    [category] '{category_name}' complete — categorical snapshot saved")

    def _write_categorical(self):
        """Write categorical_metrics.csv reflecting current accumulated state."""
        cat_rows = []
        for category in sorted(self._cat_rows):
            rows = self._cat_rows[category]
            cat_row = {"Category": category, "N": len(rows)}
            for metric in self._METRICS:
                cat_row[metric] = float(np.mean([row[metric] for row in rows]))
            for metric, average_metric in self._AVERAGE_TIME_METRICS.items():
                cat_row[average_metric] = self._mean_time(rows, metric)
            cat_rows.append(cat_row)

        if self._summary_rows:
            avg_row = {"Category": "AVERAGE", "N": len(self._summary_rows)}
            for metric in self._METRICS:
                avg_row[metric] = float(np.mean([
                    row[metric] for row in self._summary_rows
                ]))
            for metric, average_metric in self._AVERAGE_TIME_METRICS.items():
                avg_row[average_metric] = self._mean_time(self._summary_rows, metric)
            cat_rows.append(avg_row)

        pd.DataFrame(cat_rows, columns=self._CATEGORICAL_COLUMNS).to_csv(
            self.categorical_path, index=False)

    def finalize(self):
        """Rewrite summaries with final overall and categorical averages."""
        if not self._summary_rows:
            return {}

        avg_row = {"file": "AVERAGE", "Category": ""}
        for metric in self._METRICS:
            avg_row[metric] = float(np.mean([
                row[metric] for row in self._summary_rows
            ]))
        for metric in self._TIME_METRICS:
            avg_row[metric] = self._mean_time(self._summary_rows, metric)

        self._format_rows(self._summary_rows + [avg_row]).to_csv(
            self.summary_path, index=False)
        self._write_categorical()

        averages = {
            metric: float(np.mean([row[metric] for row in self._summary_rows]))
            for metric in self._METRICS
        }
        averages.update({
            metric: self._mean_time(self._summary_rows, metric)
            for metric in self._TIME_METRICS
        })
        return averages

    @property
    def num_files(self):
        return self._file_count
