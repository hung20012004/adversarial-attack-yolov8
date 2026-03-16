import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
import torch
from tqdm import tqdm

try:
    from pytorch_msssim import ssim

    SSIM_AVAILABLE = True
except ImportError:
    print("Warning: pytorch-msssim not installed. Install with: pip install pytorch-msssim")
    SSIM_AVAILABLE = False

try:
    import lpips

    LPIPS_AVAILABLE = True
except ImportError:
    print("Warning: lpips not installed. Install with: pip install lpips")
    LPIPS_AVAILABLE = False


class MetricsCalculator:
    """Tính SSIM và LPIPS."""

    def __init__(self):
        self.lpips_model = None
        if LPIPS_AVAILABLE:
            self.lpips_model = lpips.LPIPS(net="alex")
            if torch.cuda.is_available():
                self.lpips_model = self.lpips_model.cuda()
            self.lpips_model.eval()

    def calculate_ssim(self, img1_path, img2_path):
        """Tính SSIM giữa 2 ảnh."""
        if not SSIM_AVAILABLE:
            return None

        try:
            img1 = cv2.imread(str(img1_path))
            img2 = cv2.imread(str(img2_path))

            if img1 is None or img2 is None:
                return None
            if img1.shape != img2.shape:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

            img1_t = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            img2_t = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() / 255.0

            if torch.cuda.is_available():
                img1_t = img1_t.cuda()
                img2_t = img2_t.cuda()

            ssim_value = ssim(img1_t, img2_t, data_range=1.0, size_average=True)
            return float(ssim_value.cpu())
        except Exception as e:
            print(f"  SSIM error: {e}")
            return None

    def calculate_lpips(self, img1_path, img2_path):
        """Tính LPIPS giữa 2 ảnh."""
        if not LPIPS_AVAILABLE or self.lpips_model is None:
            return None

        try:
            img1 = cv2.cvtColor(cv2.imread(str(img1_path)), cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(cv2.imread(str(img2_path)), cv2.COLOR_BGR2RGB)

            if img1 is None or img2 is None:
                return None

            if img1.shape != img2.shape:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

            img1_t = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0 * 2 - 1
            img2_t = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() / 255.0 * 2 - 1

            if torch.cuda.is_available():
                img1_t = img1_t.cuda()
                img2_t = img2_t.cuda()

            with torch.no_grad():
                lpips_value = self.lpips_model(img1_t, img2_t)

            return float(lpips_value.cpu())
        except Exception as e:
            print(f"  LPIPS error: {e}")
            return None


class ExperimentMonitor:
    """Theo dõi tiến trình real-time."""

    def __init__(self, exp_dir, total_images, pbar):
        self.exp_dir = exp_dir
        self.total_images = total_images
        self.pbar = pbar
        self.stop_flag = False
        self.start_time = time.time()

    def monitor(self):
        """Đọc CSV để update progress."""
        summary_file = self.exp_dir / "results" / "attack_summary.csv"
        last_count = 0

        while not self.stop_flag:
            try:
                if summary_file.exists():
                    df = pd.read_csv(summary_file)
                    current_count = len(df)

                    if current_count > last_count:
                        self.pbar.update(current_count - last_count)
                        last_count = current_count

                        if current_count > 0:
                            elapsed = time.time() - self.start_time
                            avg_time = elapsed / current_count
                            remaining = (self.total_images - current_count) * avg_time

                            self.pbar.set_postfix({"avg": f"{avg_time:.1f}s/img", "eta": f"{remaining / 60:.1f}m"})

                    if current_count >= self.total_images:
                        break

            except Exception:
                pass

            time.sleep(0.5)

    def stop(self):
        self.stop_flag = True


class ExperimentRunner:
    """Chạy experiments với SSIM/LPIPS."""

    def __init__(self, base_config, param_pairs, original_images_dir, num_workers=2):
        self.base_config = base_config
        self.param_pairs = param_pairs
        self.original_images_dir = Path(original_images_dir)
        self.num_workers = min(num_workers, len(param_pairs))
        self.results_dir = Path("experiments") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        self.pbar_lock = threading.Lock()
        self.active_pbars = {}
        self.metrics_calc = MetricsCalculator()

    def run_single_experiment(self, epsilon, alpha, exp_index):
        """Chạy một experiment đơn lẻ."""
        exp_name = f"eps_{epsilon}_alpha_{alpha}".replace(".", "_")
        exp_dir = self.results_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        desc = f"ε={epsilon:5.3f} α={alpha:6.4f}"

        with self.pbar_lock:
            pbar = tqdm(
                total=self.base_config["attack_imgs"],
                desc=desc,
                position=exp_index,
                leave=True,
                ncols=100,
                color="cyan",
                dynamic_ncols=False,
            )
            self.active_pbars[exp_index] = pbar

        monitor = ExperimentMonitor(exp_dir, self.base_config["attack_imgs"], pbar)
        monitor_thread = threading.Thread(target=monitor.monitor, daemon=True)
        monitor_thread.start()

        cmd = [
            sys.executable,
            "atk.py",
            "--weights",
            self.base_config["weights"],
            "--data",
            self.base_config["data"],
            "--img-size",
            str(self.base_config["img_size"]),
            "--batch-size",
            "1",
            "--attack_imgs",
            str(self.base_config["attack_imgs"]),
            "--conf-thres",
            str(self.base_config["conf_thres"]),
            "--iou-thres",
            str(self.base_config["iou_thres"]),
            "--iter-atk",
            str(self.base_config["iter_atk"]),
            "--attack_freq",
            self.base_config["attack_freq"],
            "--target_map_reduction",
            str(self.base_config["target_map_reduction"]),
            "--epsilon",
            str(epsilon),
            "--alpha",
            str(alpha),
            "--attack",
            "--project",
            str(exp_dir),
            "--name",
            "results",
            "--exist-ok",
        ]

        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "0", "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"},
            )

            monitor.stop()
            monitor_thread.join(timeout=2)

            time.sleep(2)
            with self.pbar_lock:
                pbar.color = "green"
                pbar.set_description(f"✓ {desc}")
                pbar.update(self.base_config["attack_imgs"] - pbar.n)
                pbar.refresh()

            return exp_dir / "results", epsilon, alpha, True

        except subprocess.CalledProcessError as e:
            monitor.stop()

            with self.pbar_lock:
                pbar.color = "red"
                pbar.set_description(f"✗ {desc}")
                pbar.refresh()

            error_file = exp_dir / "error.log"
            error_file.write_text(f"STDERR:\n{e.stderr}\n\nSTDOUT:\n{e.stdout}")

            return None, epsilon, alpha, False

    def calculate_metrics_for_experiment(self, result_path):
        """Tính SSIM và LPIPS cho tất cả ảnh trong experiment."""
        summary_csv = result_path / "attack_summary.csv"
        if not summary_csv.exists():
            return None

        df = pd.read_csv(summary_csv)

        ssim_scores = []
        lpips_scores = []

        print(f"\n  Computing SSIM/LPIPS for {len(df)} images...")

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="  Metrics", leave=False):
            image_name = row["image_name"]

            original_path = self.original_images_dir / image_name
            stem = Path(image_name).stem
            adv_path = result_path / f"{stem}_adv_FINAL.png"

            if not adv_path.exists():
                adv_path = result_path / f"{stem}_adv_FINAL.jpg"

            if not original_path.exists() or not adv_path.exists():
                print(f"  Missing: {image_name} (original: {original_path.exists()}, adv: {adv_path.exists()})")
                ssim_scores.append(None)
                lpips_scores.append(None)
                continue
            ssim_val = self.metrics_calc.calculate_ssim(original_path, adv_path)
            lpips_val = self.metrics_calc.calculate_lpips(original_path, adv_path)

            ssim_scores.append(ssim_val)
            lpips_scores.append(lpips_val)
        df["ssim"] = ssim_scores
        df["lpips"] = lpips_scores

        enhanced_csv = result_path / "attack_summary_with_metrics.csv"
        df.to_csv(enhanced_csv, index=False)

        print(f"  ✓ Saved: {enhanced_csv.name}")

        return df

    def analyze_results(self, result_path):
        if not result_path or not result_path.exists():
            return None

        df = self.calculate_metrics_for_experiment(result_path)

        if df is None:
            return None

        try:
            summary = {
                "num_images": len(df),
                "success_rate": df["success"].mean() * 100,
                "avg_map_reduction": df["map_reduction_percent"].mean(),
                "avg_attack_time": df["attack_time_s"].mean(),
                "avg_iterations": df["iterations_ran"].mean() if "iterations_ran" in df.columns else 0,
                "avg_original_map": df["original_map"].mean(),
                "avg_adversarial_map": df["adversarial_map"].mean(),
            }

            if SSIM_AVAILABLE:
                ssim_valid = df["ssim"].dropna()
                if len(ssim_valid) > 0:
                    summary["avg_ssim"] = ssim_valid.mean()
                    summary["min_ssim"] = ssim_valid.min()
                    summary["max_ssim"] = ssim_valid.max()

            if LPIPS_AVAILABLE:
                lpips_valid = df["lpips"].dropna()
                if len(lpips_valid) > 0:
                    summary["avg_lpips"] = lpips_valid.mean()
                    summary["min_lpips"] = lpips_valid.min()
                    summary["max_lpips"] = lpips_valid.max()

            return summary
        except Exception as e:
            print(f"  Error analyzing: {e}")
            return None

    def run_parallel(self):
        """Chạy song song."""
        print(f"\n{'=' * 80}")
        print(f"Chạy {self.num_workers} experiments đồng thời")
        print(f"Tổng: {len(self.param_pairs)} experiments")
        print(f"Kết quả: {self.results_dir}")
        print(f"Ảnh gốc: {self.original_images_dir}")
        print(f"{'=' * 80}\n")

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(self.run_single_experiment, eps, alp, i): (eps, alp, i)
                for i, (eps, alp) in enumerate(self.param_pairs)
            }

            for future in as_completed(futures):
                result_path, eps, alp, success = future.result()

                if success:
                    print(f"\n{'=' * 60}")
                    print(f"Analyzing ε={eps} α={alp}...")
                    print(f"{'=' * 60}")

                    analysis = self.analyze_results(result_path)
                    if analysis:
                        self.results.append({"epsilon": eps, "alpha": alp, **analysis})

        time.sleep(1)

        with self.pbar_lock:
            for pbar in self.active_pbars.values():
                pbar.close()

        self.print_summary()

    def print_summary(self):
        """In tổng kết."""
        print(f"\n\n{'=' * 80}")
        print("KẾT QUẢ TỔNG HỢP")
        print(f"{'=' * 80}")

        if not self.results:
            print("Không có experiment nào thành công!")
            return

        df = pd.DataFrame(self.results)
        summary_path = self.results_dir / "overall_summary.csv"
        df.to_csv(summary_path, index=False)

        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        pd.set_option("display.float_format", "{:.4f}".format)

        print(f"\n{df.to_string(index=False)}")
        print(f"\nĐã lưu: {summary_path}")
        print(f"{'=' * 80}\n")


if __name__ == "__main__":
    base_config = {
        "weights": "yolov5_best.pt",
        "data": "data/tt100k.yaml",
        "img_size": 2048,
        "attack_imgs": 50,
        "conf_thres": 0.001,
        "iou_thres": 0.65,
        "iter_atk": 2000,
        "attack_freq": "mid",
        "target_map_reduction": 1.0,
    }

    param_pairs = [
        (0.2, 0.005),
        (0.1, 0.005),
        (0.02, 0.005),
        (0.2, 0.001),
        (0.2, 0.01),
        (0.2, 0.1),
        (2, 0.1),
        (2, 1),
    ]

    original_images_dir = "data/tt100k/images"

    # Số experiments chạy đồng thời
    num_workers = 1

    runner = ExperimentRunner(
        base_config=base_config,
        param_pairs=param_pairs,
        original_images_dir=original_images_dir,
        num_workers=num_workers,
    )

    runner.run_parallel()
