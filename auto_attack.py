import subprocess
import sys
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


class ExperimentRunner:
    def __init__(self, base_config):
        """Base_config: dict chứa các tham số cơ bản."""
        self.base_config = base_config
        self.results_dir = Path("experiments") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        if LPIPS_AVAILABLE:
            self.lpips_model = lpips.LPIPS(net="alex").cuda() if torch.cuda.is_available() else lpips.LPIPS(net="alex")
            self.lpips_model.eval()
        else:
            self.lpips_model = None

    def calculate_ssim(self, img1_path, img2_path):
        """Tính SSIM giữa 2 ảnh."""
        if not SSIM_AVAILABLE:
            return None

        try:
            img1 = cv2.imread(str(img1_path))
            img2 = cv2.imread(str(img2_path))

            if img1 is None or img2 is None:
                return None

            img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() / 255.0

            if torch.cuda.is_available():
                img1_tensor = img1_tensor.cuda()
                img2_tensor = img2_tensor.cuda()

            ssim_value = ssim(img1_tensor, img2_tensor, data_range=1.0, size_average=True)
            return float(ssim_value.cpu())
        except Exception as e:
            print(f"Error calculating SSIM: {e}")
            return None

    def calculate_lpips(self, img1_path, img2_path):
        """Tính LPIPS giữa 2 ảnh."""
        if not LPIPS_AVAILABLE or self.lpips_model is None:
            return None

        try:
            img1 = cv2.imread(str(img1_path))
            img2 = cv2.imread(str(img2_path))

            if img1 is None or img2 is None:
                return None

            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

            img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() / 255.0

            img1_tensor = img1_tensor * 2 - 1
            img2_tensor = img2_tensor * 2 - 1

            if torch.cuda.is_available():
                img1_tensor = img1_tensor.cuda()
                img2_tensor = img2_tensor.cuda()

            with torch.no_grad():
                lpips_value = self.lpips_model(img1_tensor, img2_tensor)

            return float(lpips_value.cpu())
        except Exception as e:
            print(f"Error calculating LPIPS: {e}")
            return None

    def run_experiment(self, epsilon, alpha, exp_name):
        """Chạy một thực nghiệm với epsilon và alpha cụ thể."""
        print(f"\n{'=' * 80}")
        print(f"Running experiment: {exp_name}")
        print(f"Epsilon: {epsilon}, Alpha: {alpha}")
        print(f"{'=' * 80}\n")

        exp_dir = self.results_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "test.py",
            "--weights",
            self.base_config["weights"],
            "--data",
            self.base_config["data"],
            "--img-size",
            str(self.base_config["img_size"]),
            "--batch-size",
            str(self.base_config["batch_size"]),
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
            subprocess.run(cmd, check=True)
            print("\n✓ Experiment completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Error running experiment: {e}")
            return None

        results_path = exp_dir / "results"

        return results_path

    def analyze_results(self, results_path, original_images_dir):
        """Phân tích kết quả và tính SSIM, LPIPS."""
        if not results_path or not results_path.exists():
            return None

        summary_csv = results_path / "attack_summary.csv"
        if not summary_csv.exists():
            print(f"Warning: {summary_csv} not found")
            return None

        df = pd.read_csv(summary_csv)

        ssim_scores = []
        lpips_scores = []

        print(f"\nCalculating SSIM and LPIPS for {len(df)} images...")

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            image_name = row["image_name"]

            original_path = Path(original_images_dir) / image_name

            stem = Path(image_name).stem
            adv_path = results_path / f"{stem}_adv_FINAL.png"

            if not original_path.exists() or not adv_path.exists():
                print(f"Warning: Cannot find images for {image_name}")
                ssim_scores.append(None)
                lpips_scores.append(None)
                continue

            ssim_val = self.calculate_ssim(original_path, adv_path)
            ssim_scores.append(ssim_val)

            lpips_val = self.calculate_lpips(original_path, adv_path)
            lpips_scores.append(lpips_val)

        df["ssim"] = ssim_scores
        df["lpips"] = lpips_scores
        enhanced_csv = results_path / "attack_summary_with_metrics.csv"
        df.to_csv(enhanced_csv, index=False)

        print(f"Saved enhanced results to: {enhanced_csv}")

        return df

    def run_all_experiments(self, param_pairs, original_images_dir):
        """Chạy tất cả các thực nghiệm với các CẶP giá trị epsilon-alpha.

        param_pairs: list of tuples [(epsilon1, alpha1), (epsilon2, alpha2), ...]
        original_images_dir: đường dẫn đến thư mục ảnh gốc để tính SSIM/LPIPS
        """
        all_results = []

        for epsilon, alpha in param_pairs:
            exp_name = f"eps_{epsilon}_alpha_{alpha}".replace(".", "_")

            results_path = self.run_experiment(epsilon, alpha, exp_name)

            if results_path:
                df = self.analyze_results(results_path, original_images_dir)

                if df is not None:
                    summary = {
                        "epsilon": epsilon,
                        "alpha": alpha,
                        "num_images": len(df),
                        "success_rate": df["success"].mean() * 100,
                        "avg_map_reduction": df["map_reduction_percent"].mean(),
                        "avg_attack_time": df["attack_time_s"].mean(),
                        "avg_iterations": df["iterations_ran"].mean(),
                        "avg_original_map": df["original_map"].mean(),
                        "avg_adversarial_map": df["adversarial_map"].mean(),
                    }

                    if SSIM_AVAILABLE:
                        summary["avg_ssim"] = df["ssim"].mean()
                        summary["min_ssim"] = df["ssim"].min()
                        summary["max_ssim"] = df["ssim"].max()

                    if LPIPS_AVAILABLE:
                        summary["avg_lpips"] = df["lpips"].mean()
                        summary["min_lpips"] = df["lpips"].min()
                        summary["max_lpips"] = df["lpips"].max()

                    all_results.append(summary)

                    print(f"\nSummary for {exp_name}:")
                    for key, value in summary.items():
                        print(f"  {key}: {value}")

        if all_results:
            summary_df = pd.DataFrame(all_results)
            summary_path = self.results_dir / "overall_summary.csv"
            summary_df.to_csv(summary_path, index=False)
            print(f"\n{'=' * 80}")
            print("All experiments completed!")
            print(f"Overall summary saved to: {summary_path}")
            print(f"{'=' * 80}")

            print("\nOVERALL RESULTS:")
            print(summary_df.to_string(index=False))

        return all_results


def main():
    base_config = {
        "weights": "yolov8_best.pt",
        "data": "data/tt100k.yaml",
        "img_size": 2048,
        "batch_size": 1,
        "attack_imgs": 250,
        "conf_thres": 0.001,
        "iou_thres": 0.65,
        "iter_atk": 2000,
        "attack_freq": "low",
        "target_map_reduction": 1.0,
    }

    param_pairs = [
        (0.9, 0.2),
        (0.7, 0.15),
        (0.5, 0.15),
        (0.5, 0.1),
        (0.3, 0.1),
        (0.2, 0.05),
        (0.1, 0.05),
        (0.05, 0.01),
    ]
    original_images_dir = "data/tt100k/images"

    runner = ExperimentRunner(base_config)

    runner.run_all_experiments(param_pairs=param_pairs, original_images_dir=original_images_dir)

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED!")
    print(f"Results saved in: {runner.results_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
