from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import zipfile
from multiprocessing import cpu_count
from pathlib import Path


SR_MAP = {"32k": 32000, "40k": 40000, "48k": 48000}
DEFAULT_KAGGLE_PYTHON = Path("/kaggle/working/rvc_venv/bin/python")


def ensure_kaggle_python() -> None:
    if not Path("/kaggle").exists():
        return

    preferred = Path(os.environ.get("RVC_KAGGLE_PYTHON", DEFAULT_KAGGLE_PYTHON))
    current = Path(sys.executable).resolve()
    if preferred.exists() and current != preferred.resolve():
        print(f"Re-executing with Kaggle venv Python: {preferred}", flush=True)
        os.execv(str(preferred), [str(preferred), str(Path(__file__).resolve()), *sys.argv[1:]])

    if not preferred.exists() and current == Path("/usr/bin/python3"):
        raise RuntimeError(
            "Kaggle venv Python 不存在。请先运行 notebook 的“安装 RVC”单元，"
            "创建 /kaggle/working/rvc_venv 并安装 Kaggle 专用依赖。"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RVC training pipeline in Kaggle.")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Generated dataset directory or its audio dir.")
    parser.add_argument("--dataset-zip", type=Path, default=None, help="Dataset zip from /kaggle/input.")
    parser.add_argument("--extract-dir", type=Path, default=Path("/kaggle/working/rvc_datasets"))
    parser.add_argument("--repo-dir", type=Path, default=Path.cwd(), help="RVC repo root.")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--sample-rate", choices=sorted(SR_MAP), default="48k")
    parser.add_argument("--version", choices=["v1", "v2"], default="v2")
    parser.add_argument("--if-f0", type=int, choices=[0, 1], default=1)
    parser.add_argument("--f0-method", choices=["pm", "harvest", "dio", "rmvpe", "rmvpe_gpu"], default="rmvpe")
    parser.add_argument("--gpus", default="0", help="GPU ids joined by '-', for example 0 or 0-1.")
    parser.add_argument("--processes", type=int, default=max(2, min(cpu_count(), 8)))
    parser.add_argument("--speaker-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--total-epoch", type=int, default=200)
    parser.add_argument("--save-every-epoch", type=int, default=20)
    parser.add_argument("--save-latest", type=int, choices=[0, 1], default=1)
    parser.add_argument("--cache-gpu", type=int, choices=[0, 1], default=0)
    parser.add_argument("--save-every-weights", type=int, choices=[0, 1], default=0)
    parser.add_argument("--preprocess-per", type=float, default=3.7)
    parser.add_argument("--fp32", action="store_true", help="Disable half precision.")
    parser.add_argument("--overwrite-dataset", action="store_true")
    parser.add_argument("--skip-train", action="store_true", help="Only preprocess/extract/build filelist/index if possible.")
    parser.add_argument("--export-dir", type=Path, default=None, help="Kaggle output directory for trained artifacts.")
    parser.add_argument("--export-checkpoints", action="store_true", help="Also export logs/<experiment>/G_*.pth and D_*.pth.")
    parser.add_argument("--keep-export-dir", action="store_true", help="Keep unpacked exported files next to the zip package.")
    parser.add_argument("--keep-training-cache", action="store_true", help="Keep copied datasets, feature cache, and checkpoints after export.")
    parser.add_argument("--skip-model-download", action="store_true", help="Do not download minimal Kaggle training models.")
    return parser.parse_args()


def run(cmd: list[str], cwd: Path) -> None:
    printable = " ".join(str(part) for part in cmd)
    print(f"\n$ {printable}", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def download_file(url: str, target: Path) -> None:
    if target.exists() and target.stat().st_size > 0:
        print(f"Model exists: {target}")
        return

    import requests

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    print(f"Downloading {url} -> {target}")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    tmp.replace(target)


def ensure_minimal_training_models(args: argparse.Namespace) -> None:
    if args.skip_model_download:
        return

    base_url = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main"
    download_file(f"{base_url}/hubert_base.pt", args.repo_dir / "assets" / "hubert" / "hubert_base.pt")
    if args.if_f0 and args.f0_method in {"rmvpe", "rmvpe_gpu"}:
        download_file(f"{base_url}/rmvpe.pt", args.repo_dir / "assets" / "rmvpe" / "rmvpe.pt")

    folder = "pretrained" if args.version == "v1" else "pretrained_v2"
    prefix = "f0" if args.if_f0 else ""
    for name in [f"{prefix}G{args.sample_rate}.pth", f"{prefix}D{args.sample_rate}.pth"]:
        download_file(
            f"{base_url}/{folder}/{name}",
            args.repo_dir / "assets" / folder / name,
        )


def load_colab_pipeline(repo_dir: Path):
    script = repo_dir / "rvc_training_data" / "colab" / "train_rvc_colab.py"
    if not script.exists():
        raise FileNotFoundError(f"Missing shared training pipeline: {script}")

    spec = importlib.util.spec_from_file_location("rvc_colab_pipeline", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load shared training pipeline: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dataset_root_from_zip(zip_path: Path) -> str | None:
    roots: set[str] = set()
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            clean = name.strip("/")
            if not clean:
                continue
            roots.add(clean.split("/", 1)[0])
    return next(iter(roots)) if len(roots) == 1 else None


def prepare_kaggle_dataset(args: argparse.Namespace) -> Path:
    if args.dataset_zip is None and args.dataset_dir is None:
        raise ValueError("请设置 --dataset-zip 或 --dataset-dir")

    if args.dataset_zip is not None:
        zip_path = args.dataset_zip.resolve()
        if not zip_path.exists():
            raise FileNotFoundError(f"数据 zip 不存在: {zip_path}")
        root_name = dataset_root_from_zip(zip_path)
        args.extract_dir.mkdir(parents=True, exist_ok=True)
        print(f"Extracting {zip_path} to {args.extract_dir}")
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(args.extract_dir)
        if args.dataset_dir is None:
            args.dataset_dir = args.extract_dir / root_name if root_name else args.extract_dir

    assert args.dataset_dir is not None
    dataset_dir = args.dataset_dir.resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {dataset_dir}")
    return dataset_dir


def create_package(export_dir: Path, package_path: Path) -> Path:
    if package_path.exists():
        package_path.unlink()
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(export_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(export_dir.parent))
    print(f"Created package: {package_path}")
    return package_path


def cleanup_after_export(args: argparse.Namespace, export_dir: Path) -> None:
    if args.keep_training_cache:
        return

    targets = [
        args.repo_dir / "datasets" / args.experiment,
        args.repo_dir / "logs" / args.experiment,
    ]
    if args.dataset_zip is not None and Path("/kaggle/working") in args.extract_dir.resolve().parents:
        targets.append(args.extract_dir)
    for target in targets:
        if target.exists():
            shutil.rmtree(target)
            print(f"Removed training cache: {target}")

    weights_dir = args.repo_dir / "assets" / "weights"
    for pattern in [f"{args.experiment}.pth", f"{args.experiment}_e*_s*.pth"]:
        for path in weights_dir.glob(pattern):
            path.unlink()
            print(f"Removed exported weight copy: {path}")

    if not args.keep_export_dir and export_dir.exists():
        shutil.rmtree(export_dir)
        print(f"Removed unpacked export dir: {export_dir}")


def export_artifacts(args: argparse.Namespace, summary_path: Path) -> Path:
    export_dir = args.export_dir or (Path("/kaggle/working/rvc_models") / args.experiment)
    export_dir.mkdir(parents=True, exist_ok=True)

    patterns = [
        args.repo_dir / "assets" / "weights" / f"{args.experiment}.pth",
        args.repo_dir / "logs" / args.experiment / "*.index",
        args.repo_dir / "logs" / args.experiment / "train.log",
        summary_path,
    ]
    if args.export_checkpoints:
        patterns.extend(
            [
                args.repo_dir / "logs" / args.experiment / "G_*.pth",
                args.repo_dir / "logs" / args.experiment / "D_*.pth",
            ]
        )

    exported: list[Path] = []
    for pattern in patterns:
        for src_name in glob.glob(str(pattern)):
            src = Path(src_name)
            dst = export_dir / src.name
            shutil.copy2(src, dst)
            exported.append(dst)

    print(f"Exported {len(exported)} file(s) to {export_dir}")
    for path in sorted(exported):
        print(path)
    if not exported:
        raise RuntimeError(f"没有导出任何文件，请检查训练是否生成了最终权重和 index: {export_dir}")

    package_path = export_dir.with_suffix(".zip")
    create_package(export_dir, package_path)
    cleanup_after_export(args, export_dir)
    return package_path


def main() -> None:
    ensure_kaggle_python()
    args = parse_args()
    args.repo_dir = args.repo_dir.resolve()
    args.dataset_dir = prepare_kaggle_dataset(args)
    ensure_minimal_training_models(args)

    pipeline = load_colab_pipeline(args.repo_dir)

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    raw_dir = pipeline.prepare_raw_dataset(args)
    exp_dir = args.repo_dir / "logs" / args.experiment
    is_half = not args.fp32

    pipeline.ensure_config(args.repo_dir, args.experiment, args.version, args.sample_rate)
    run(
        [
            sys.executable,
            "infer/modules/train/preprocess.py",
            str(raw_dir),
            str(SR_MAP[args.sample_rate]),
            str(args.processes),
            str(exp_dir),
            "False",
            str(args.preprocess_per),
        ],
        args.repo_dir,
    )
    pipeline.extract_f0(args, exp_dir, is_half)
    pipeline.extract_features(args, exp_dir, is_half)
    pipeline.build_filelist(args.repo_dir, args.experiment, args.version, args.if_f0, args.speaker_id, args.sample_rate)

    if not args.skip_train:
        pretrain_g, pretrain_d = pipeline.pretrained_paths(args.repo_dir, args.version, args.sample_rate, args.if_f0)
        train_cmd = [
            sys.executable,
            "infer/modules/train/train.py",
            "-e",
            args.experiment,
            "-sr",
            args.sample_rate,
            "-f0",
            str(args.if_f0),
            "-bs",
            str(args.batch_size),
            "-g",
            args.gpus,
            "-te",
            str(args.total_epoch),
            "-se",
            str(args.save_every_epoch),
            "-l",
            str(args.save_latest),
            "-c",
            str(args.cache_gpu),
            "-sw",
            str(args.save_every_weights),
            "-v",
            args.version,
        ]
        if pretrain_g:
            train_cmd.extend(["-pg", pretrain_g])
        if pretrain_d:
            train_cmd.extend(["-pd", pretrain_d])
        run(train_cmd, args.repo_dir)

    index_path = pipeline.build_index(args.repo_dir, args.experiment, args.version)
    export_dir = args.export_dir or (Path("/kaggle/working/rvc_models") / args.experiment)
    summary = {
        "environment": "kaggle",
        "experiment": args.experiment,
        "dataset_dir": str(args.dataset_dir),
        "log_dir": str(exp_dir),
        "index": str(index_path),
        "weights_dir": str(args.repo_dir / "assets" / "weights"),
        "export_dir": str(export_dir),
        "package": str(export_dir.with_suffix(".zip")),
    }
    summary_path = exp_dir / "kaggle_train_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    export_artifacts(args, summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
