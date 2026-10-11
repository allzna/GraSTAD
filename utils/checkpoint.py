import csv
import os
import json


class CheckpointManager:
    """Tracks completed files so interrupted runs can resume.

    Priority:
    1. .checkpoint.json  (authoritative, written by mark_done)
    2. summary_metrics.csv (fallback — recovers state when checkpoint
       file doesn't exist yet, e.g. from a run before CheckpointManager
       was added, or when output_dir is pointed at existing results)
    """

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self._ckpt_path = os.path.join(output_dir, ".checkpoint.json")
        self._summary_path = os.path.join(output_dir, "summary_metrics.csv")
        self._completed = self._load()

    # ── public API ────────────────────────────────────────────────────
    def is_done(self, file_name):
        return file_name in self._completed

    def mark_done(self, file_name, prev_category):
        self._completed.add(file_name)
        self._save(meta={"prev_category": prev_category or "",
                         "count": len(self._completed)})

    @property
    def done_count(self):
        return len(self._completed)

    def get_prev_category(self):
        """Return the category of the last completed file, or None."""
        meta = self._load_meta()
        cat = meta.get("prev_category", "")
        return cat if cat else None

    def clear(self):
        """Remove checkpoint (for fresh restart)."""
        self._completed = set()
        if os.path.exists(self._ckpt_path):
            os.remove(self._ckpt_path)

    # ── internal ──────────────────────────────────────────────────────
    def _load(self):
        # 1) checkpoint file is authoritative
        data = self._load_raw()
        if data:
            return set(data.get("completed", []))

        # 2) fallback: rebuild from summary_metrics.csv
        return self._load_from_csv()

    def _load_from_csv(self):
        """Parse summary_metrics.csv to recover completed file names."""
        if not os.path.exists(self._summary_path):
            return set()
        completed = set()
        with open(self._summary_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("file", "").strip()
                if name and name != "AVERAGE":
                    completed.add(name)
        return completed

    def _load_meta(self):
        meta = self._load_raw().get("meta", {})
        if meta:
            return meta
        # fallback: derive prev_category from the last row of summary CSV
        if not os.path.exists(self._summary_path):
            return {}
        last_cat = ""
        with open(self._summary_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cat = row.get("Category", "").strip()
                name = row.get("file", "").strip()
                if name and name != "AVERAGE" and cat:
                    last_cat = cat
        return {"prev_category": last_cat, "count": len(self._completed)}

    def _load_raw(self):
        if not os.path.exists(self._ckpt_path):
            return {}
        with open(self._ckpt_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self, meta=None):
        os.makedirs(self.output_dir, exist_ok=True)
        payload = {
            "completed": sorted(self._completed),
            "meta": meta or {},
        }
        with open(self._ckpt_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
