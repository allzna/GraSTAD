import os

for _thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "LOKY_MAX_CPU_COUNT",
):
    os.environ[_thread_env] = "1"
print("[Threads] CPU thread limits set to 1", flush=True)
import faulthandler

faulthandler.enable()
if os.environ.get("GRASTAD_DEBUG_STACKS") == "1":
    faulthandler.dump_traceback_later(60, repeat=True)
import time
import gc
import random
import warnings
import numpy as np
import torch
from sklearn.exceptions import UndefinedMetricWarning
from model import PatchEncoder
from train import train_model
from utils.data_preprocess import (
    PatchCreator,
    find_length_rank,
    load_and_split_data,
    preprocess_to_patches,
)
from utils.evaluation import (
    create_per_channel_memory_bank,
    calculate_per_channel_anomaly_scores,
    distribute_patch_scores_to_points,
)
from utils.metrics import get_metrics
from utils.results_manager import ResultsManager
from utils.checkpoint import CheckpointManager

warnings.simplefilter("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def _synchronize_cuda(device):
    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


class AnomalyDetection:

    def __init__(
        self,
        data_dir,
        output_dir=None,
        patch_size=96,
        num_iters=100,
        lr=0.0001,
        batch_size=512,
        random_seed=2027,
        use_revin=False,
        file_list=None,
    ):
        self.data_dir = data_dir
        self.output_dir = output_dir or os.path.join(
            os.path.dirname(__file__), "results"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        self.patch_size = patch_size
        self.num_iters = num_iters
        self.lr = lr
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_revin = use_revin
        self.file_list = file_list

    def run(self):
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        tracker = ResultsManager(self.output_dir)
        tracker.restore_from_disk()
        ckpt = CheckpointManager(self.output_dir)
        all_files = sorted([f for f in os.listdir(self.data_dir) if f.endswith(".csv")])
        if self.file_list:
            import pandas as pd

            _df = pd.read_csv(self.file_list)
            _valid = set(_df["file_name"].values)
            all_files = [f for f in all_files if f in _valid]
            print(
                f"    [file_list] {len(all_files)} datasets matched from {self.file_list}"
            )
        total_csv_files = len(all_files)
        prev_category = ckpt.get_prev_category()

        def is_complete(file_name):
            return ckpt.is_done(file_name) and tracker.has_timing(file_name)

        skipped = sum((1 for file_name in all_files if is_complete(file_name)))
        if skipped > 0:
            print(
                f"GraSTAD resuming — {skipped}/{total_csv_files} files already done, {total_csv_files - skipped} remaining"
            )
        else:
            print(f"GraSTAD is running... (found {total_csv_files} files to detect)")
        current_idx = skipped + 1
        for file_name in all_files:
            if is_complete(file_name):
                continue
            file_path = os.path.join(self.data_dir, file_name)
            print(
                f"\x1b[1m══ Running on file ({current_idx}/{total_csv_files})\x1b[0m : {file_name}"
            )
            current_idx += 1
            train_data, train_labels, test_data, test_labels = load_and_split_data(
                file_path
            )
            train_data = np.array(train_data, dtype=np.float32)
            test_data = np.array(test_data, dtype=np.float32)
            test_labels = np.array(test_labels, dtype=np.float32)
            full_data = np.concatenate([train_data, test_data], axis=0)
            full_labels = np.concatenate([train_labels, test_labels], axis=0)
            if full_data.ndim == 1:
                sliding_input = full_data.reshape(-1, 1)
            else:
                sliding_input = full_data[:, 0].reshape(-1, 1)
            slidingWindow = find_length_rank(sliding_input, rank=1)
            patch_creator = PatchCreator(
                L=self.patch_size, s=1, random_seed=self.random_seed
            )
            train_loader, test_loader, true_test_labels = (
                patch_creator.create_dataloaders(
                    train_data, full_data, full_labels, batch_size=self.batch_size
                )
            )
            xb, _ = next(iter(train_loader))
            C = xb.shape[1]
            print(f"[init] inferred in_channels = {C}")
            effective_batch_size = self.batch_size
            if C > 128:
                effective_batch_size = max(32, self.batch_size // (C // 64))
                train_loader, test_loader, true_test_labels = (
                    patch_creator.create_dataloaders(
                        train_data,
                        full_data,
                        full_labels,
                        batch_size=effective_batch_size,
                    )
                )
                print(
                    f"    [OOM guard] C={C}, reduced batch_size to {effective_batch_size}"
                )
            component_times = {"Graph_Time_s": 0.0, "Cluster_Time_s": 0.0}
            _synchronize_cuda(self.device)
            train_started_at = time.time()
            model = PatchEncoder(in_channels=C, use_revin=self.use_revin).to(
                self.device
            )
            train_model(
                model,
                train_loader,
                preprocess_to_patches(train_data, patch_size=self.patch_size, stride=1),
                self.device,
                num_iter=self.num_iters,
                lr=self.lr,
                timings=component_times,
            )
            preparation_started_at = time.time()
            print(
                "    [Prepare] Network training finished; preparing memory and scoring statistics...",
                flush=True,
            )
            per_channel_banks, amp_stats = create_per_channel_memory_bank(
                model, train_loader, self.device, timings=component_times
            )
            print("    [Prepare] Calibrating training scores...", flush=True)
            scoring_stats = calculate_per_channel_anomaly_scores(
                model,
                None,
                per_channel_banks,
                device=self.device,
                train_loader=train_loader,
                amp_stats=amp_stats,
                return_scoring_stats=True,
            )
            _synchronize_cuda(self.device)
            train_time_s = time.time() - train_started_at
            print(
                f"    [Prepare] Completed in {time.time() - preparation_started_at:.2f}s",
                flush=True,
            )
            _synchronize_cuda(self.device)
            infer_started_at = time.time()
            print("    [Inference] Scoring sequence patches...", flush=True)
            all_scores = calculate_per_channel_anomaly_scores(
                model,
                test_loader,
                per_channel_banks,
                device=self.device,
                amp_stats=amp_stats,
                scoring_stats=scoring_stats,
            )
            patch_size_for_dist = self.patch_size
            dist_scores = distribute_patch_scores_to_points(
                all_scores, patch_size=patch_size_for_dist, num_points=len(full_labels)
            )
            _synchronize_cuda(self.device)
            infer_time_s = time.time() - infer_started_at
            print("    >> Anomaly detection completed. Calculating the score...")
            print(
                f"    >> Train time: {train_time_s:.6f}s, Inference time: {infer_time_s:.6f}s"
            )
            print(
                f"    >> Graph construction time: {component_times['Graph_Time_s']:.6f}s, Clustering and selection time: {component_times['Cluster_Time_s']:.6f}s"
            )
            results = get_metrics(
                dist_scores,
                full_labels,
                slidingWindow=slidingWindow,
                pred=None,
                version="opt",
                thre=250,
            )
            category = file_name.split("_")[1]
            if prev_category is not None and category != prev_category:
                tracker.flush_category_complete(prev_category)
            prev_category = category
            print("    [Anomaly Detection Results]")
            print(
                f"    >> AUC-ROC: {results['AUC-ROC']:.4f}, AUC-PR: {results['AUC-PR']:.4f}, VUS-PR: {results['VUS-PR']:.4f}, VUS-ROC: {results['VUS-ROC']:.4f}, BestF1: {results['Standard-F1']:.4f}, RangeF1: {results['R-based-F1']:.4f}"
            )
            tracker.save_file_result(
                file_name,
                category,
                results,
                full_labels,
                dist_scores,
                train_time_s=train_time_s,
                infer_time_s=infer_time_s,
                graph_time_s=component_times["Graph_Time_s"],
                cluster_time_s=component_times["Cluster_Time_s"],
            )
            print(
                f"    >> Saved → {os.path.join(self.output_dir, 'Filewise_scores', file_name + '_output.csv')}"
            )
            ckpt.mark_done(file_name, prev_category)
            del model, all_scores, dist_scores
            gc.collect()
            torch.cuda.empty_cache()
        if prev_category is not None:
            tracker.flush_category_complete(prev_category)
        avg = tracker.finalize()
        print(
            f"GraSTAD's Averaged Final Results: AUROC={avg.get('AUC-ROC', 0):.4f}, AUPRC={avg.get('AUC-PR', 0):.4f}, VUSPR={avg.get('VUS-PR', 0):.4f}, VUSROC={avg.get('VUS-ROC', 0):.4f}, F1-Score={avg.get('BestF1', 0):.4f}, RangeF1={avg.get('RangeF1', 0):.4f}"
        )
        print(
            f">> Average component times (s): graph={avg.get('Graph_Time_s', 'N/A')}, clustering and selection={avg.get('Cluster_Time_s', 'N/A')}"
        )
        print(f">> All results saved to {self.output_dir}")


def build_parser():
    """Expose only the public dataset and training parameters."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run GraSTAD anomaly detection", allow_abbrev=False
    )
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument(
        "--file_list",
        type=str,
        default=None,
        help="CSV file containing a file_name column",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--patch_size", type=int, default=96)
    parser.add_argument("--num_iters", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument(
        "--use_revin",
        action="store_true",
        help="Normalize each patch with RevIN and include amplitude scoring",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    for name in ("patch_size", "num_iters", "batch_size"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be positive")
    if not np.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be finite and positive")
    experiment = AnomalyDetection(
        data_dir=args.data_dir,
        file_list=args.file_list,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        num_iters=args.num_iters,
        batch_size=args.batch_size,
        lr=args.lr,
        random_seed=args.seed,
        use_revin=args.use_revin,
    )
    experiment.run()
