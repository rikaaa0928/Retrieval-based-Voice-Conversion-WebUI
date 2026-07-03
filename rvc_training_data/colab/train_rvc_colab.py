from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from multiprocessing import cpu_count
from pathlib import Path


SR_MAP = {"32k": 32000, "40k": 40000, "48k": 48000}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RVC training pipeline in Colab.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Generated dataset directory or its audio dir.")
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
    parser.add_argument("--save-every-weights", type=int, choices=[0, 1], default=1)
    parser.add_argument("--preprocess-per", type=float, default=3.7)
    parser.add_argument("--fp32", action="store_true", help="Disable half precision.")
    parser.add_argument("--overwrite-dataset", action="store_true")
    parser.add_argument("--skip-train", action="store_true", help="Only preprocess/extract/build filelist/index if possible.")
    return parser.parse_args()


def run(cmd: list[str], cwd: Path) -> None:
    printable = " ".join(str(part) for part in cmd)
    print(f"\n$ {printable}", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def audio_source_dir(dataset_dir: Path) -> Path:
    return dataset_dir / "audio" if (dataset_dir / "audio").is_dir() else dataset_dir


def prepare_raw_dataset(args: argparse.Namespace) -> Path:
    source = audio_source_dir(args.dataset_dir.resolve())
    if not source.exists():
        raise FileNotFoundError(f"数据目录不存在: {source}")

    target = args.repo_dir / "datasets" / args.experiment / "raw"
    if target.exists() and args.overwrite_dataset:
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    audio_suffixes = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac", ".opus"}
    copied = 0
    for path in sorted(source.iterdir()):
        if path.suffix.lower() not in audio_suffixes:
            continue
        dst = target / path.name
        if not dst.exists() or args.overwrite_dataset:
            shutil.copy2(path, dst)
        copied += 1

    if copied == 0:
        raise RuntimeError(f"没有找到音频文件: {source}")
    print(f"Prepared {copied} audio file(s) in {target}")
    return target


def pretrained_paths(repo_dir: Path, version: str, sample_rate: str, if_f0: int) -> tuple[str, str]:
    folder = "pretrained" if version == "v1" else "pretrained_v2"
    prefix = "f0" if if_f0 else ""
    g_path = repo_dir / "assets" / folder / f"{prefix}G{sample_rate}.pth"
    d_path = repo_dir / "assets" / folder / f"{prefix}D{sample_rate}.pth"
    return (str(g_path) if g_path.exists() else "", str(d_path) if d_path.exists() else "")


def ensure_config(repo_dir: Path, experiment: str, version: str, sample_rate: str) -> None:
    exp_dir = repo_dir / "logs" / experiment
    exp_dir.mkdir(parents=True, exist_ok=True)
    config_name = f"v1/{sample_rate}.json" if version == "v1" or sample_rate == "40k" else f"v2/{sample_rate}.json"
    source = repo_dir / "configs" / config_name
    target = exp_dir / "config.json"
    if not source.exists():
        raise FileNotFoundError(f"缺少配置文件: {source}")
    if not target.exists():
        shutil.copy2(source, target)


def build_filelist(repo_dir: Path, experiment: str, version: str, if_f0: int, speaker_id: int, sample_rate: str) -> Path:
    exp_dir = repo_dir / "logs" / experiment
    gt_dir = exp_dir / "0_gt_wavs"
    feature_dir = exp_dir / ("3_feature256" if version == "v1" else "3_feature768")
    if not gt_dir.exists() or not feature_dir.exists():
        raise FileNotFoundError("请先完成 preprocess 和 feature extraction")

    gt_names = {path.stem for path in gt_dir.glob("*.wav")}
    feature_names = {path.stem for path in feature_dir.glob("*.npy")}
    names = gt_names & feature_names

    if if_f0:
        f0_dir = exp_dir / "2a_f0"
        f0nsf_dir = exp_dir / "2b-f0nsf"
        f0_names = {path.name.split(".")[0] for path in f0_dir.glob("*.npy")}
        f0nsf_names = {path.name.split(".")[0] for path in f0nsf_dir.glob("*.npy")}
        names &= f0_names & f0nsf_names

    if not names:
        raise RuntimeError("没有可训练样本，请检查预处理/F0/特征提取日志")

    lines: list[str] = []
    for name in sorted(names):
        if if_f0:
            lines.append(
                f"{gt_dir / (name + '.wav')}|{feature_dir / (name + '.npy')}|"
                f"{exp_dir / '2a_f0' / (name + '.wav.npy')}|"
                f"{exp_dir / '2b-f0nsf' / (name + '.wav.npy')}|{speaker_id}"
            )
        else:
            lines.append(f"{gt_dir / (name + '.wav')}|{feature_dir / (name + '.npy')}|{speaker_id}")

    feature_dim = 256 if version == "v1" else 768
    mute_root = repo_dir / "logs" / "mute"
    for _ in range(2):
        if if_f0:
            lines.append(
                f"{mute_root / '0_gt_wavs' / ('mute' + sample_rate + '.wav')}|"
                f"{mute_root / ('3_feature' + str(feature_dim)) / 'mute.npy'}|"
                f"{mute_root / '2a_f0' / 'mute.wav.npy'}|"
                f"{mute_root / '2b-f0nsf' / 'mute.wav.npy'}|{speaker_id}"
            )
        else:
            lines.append(
                f"{mute_root / '0_gt_wavs' / ('mute' + sample_rate + '.wav')}|"
                f"{mute_root / ('3_feature' + str(feature_dim)) / 'mute.npy'}|{speaker_id}"
            )

    filelist = exp_dir / "filelist.txt"
    filelist.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} training rows to {filelist}")
    return filelist


def extract_features(args: argparse.Namespace, exp_dir: Path, is_half: bool) -> None:
    import torch

    gpu_ids = [gpu for gpu in args.gpus.split("-") if gpu != ""]
    if torch.cuda.is_available() and gpu_ids:
        for idx, gpu in enumerate(gpu_ids):
            run(
                [
                    sys.executable,
                    "infer/modules/train/extract_feature_print.py",
                    "cuda",
                    str(len(gpu_ids)),
                    str(idx),
                    gpu,
                    str(exp_dir),
                    args.version,
                    str(is_half).lower(),
                ],
                args.repo_dir,
            )
    else:
        run(
            [
                sys.executable,
                "infer/modules/train/extract_feature_print.py",
                "cpu",
                "1",
                "0",
                str(exp_dir),
                args.version,
                "false",
            ],
            args.repo_dir,
        )


def extract_f0(args: argparse.Namespace, exp_dir: Path, is_half: bool) -> None:
    if not args.if_f0:
        return

    import torch

    gpu_ids = [gpu for gpu in args.gpus.split("-") if gpu != ""]
    if args.f0_method == "rmvpe_gpu" and torch.cuda.is_available() and gpu_ids:
        for idx, gpu in enumerate(gpu_ids):
            run(
                [
                    sys.executable,
                    "infer/modules/train/extract/extract_f0_rmvpe.py",
                    str(len(gpu_ids)),
                    str(idx),
                    gpu,
                    str(exp_dir),
                    str(is_half).lower(),
                ],
                args.repo_dir,
            )
    else:
        method = "rmvpe" if args.f0_method == "rmvpe_gpu" else args.f0_method
        run(
            [
                sys.executable,
                "infer/modules/train/extract/extract_f0_print.py",
                str(exp_dir),
                str(args.processes),
                method,
            ],
            args.repo_dir,
        )


def build_index(repo_dir: Path, experiment: str, version: str) -> Path:
    import faiss
    import numpy as np
    from sklearn.cluster import MiniBatchKMeans

    exp_dir = repo_dir / "logs" / experiment
    feature_dir = exp_dir / ("3_feature256" if version == "v1" else "3_feature768")
    npys = [np.load(path) for path in sorted(feature_dir.glob("*.npy"))]
    if not npys:
        raise RuntimeError(f"没有特征文件: {feature_dir}")

    big_npy = np.concatenate(npys, 0)
    order = np.arange(big_npy.shape[0])
    np.random.shuffle(order)
    big_npy = big_npy[order]
    if big_npy.shape[0] > 2e5:
        big_npy = MiniBatchKMeans(
            n_clusters=10000,
            verbose=True,
            batch_size=256 * max(1, cpu_count()),
            compute_labels=False,
            init="random",
        ).fit(big_npy).cluster_centers_

    np.save(exp_dir / "total_fea.npy", big_npy)
    dim = 256 if version == "v1" else 768
    n_ivf = max(1, min(int(16 * np.sqrt(big_npy.shape[0])), max(1, big_npy.shape[0] // 39)))
    index = faiss.index_factory(dim, f"IVF{n_ivf},Flat")
    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 1
    index.train(big_npy)
    trained_path = exp_dir / f"trained_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{experiment}_{version}.index"
    faiss.write_index(index, str(trained_path))
    for start in range(0, big_npy.shape[0], 8192):
        index.add(big_npy[start:start + 8192])
    added_path = exp_dir / f"added_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{experiment}_{version}.index"
    faiss.write_index(index, str(added_path))
    print(f"Built index: {added_path}")
    return added_path


def main() -> None:
    args = parse_args()
    args.repo_dir = args.repo_dir.resolve()
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    raw_dir = prepare_raw_dataset(args)
    exp_dir = args.repo_dir / "logs" / args.experiment
    is_half = not args.fp32

    ensure_config(args.repo_dir, args.experiment, args.version, args.sample_rate)
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
    extract_f0(args, exp_dir, is_half)
    extract_features(args, exp_dir, is_half)
    build_filelist(args.repo_dir, args.experiment, args.version, args.if_f0, args.speaker_id, args.sample_rate)

    if not args.skip_train:
        pretrain_g, pretrain_d = pretrained_paths(args.repo_dir, args.version, args.sample_rate, args.if_f0)
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

    index_path = build_index(args.repo_dir, args.experiment, args.version)
    summary = {
        "experiment": args.experiment,
        "log_dir": str(exp_dir),
        "index": str(index_path),
        "weights_dir": str(args.repo_dir / "assets" / "weights"),
    }
    (exp_dir / "colab_train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
