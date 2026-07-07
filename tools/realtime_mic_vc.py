from __future__ import annotations

import argparse
import os
import sys
import time
import zipfile
from multiprocessing import Process, Queue, cpu_count, freeze_support
from pathlib import Path

now_dir = Path(__file__).resolve().parents[1]
os.chdir(now_dir)
sys.path.insert(0, str(now_dir))

import librosa
import numpy as np
import sounddevice as sd
import torch
import torch.nn.functional as F
import torchaudio.transforms as tat
from dotenv import load_dotenv


ASSET_URL = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main"
ARTIFACT_SUFFIXES = {".pth", ".index"}


class Harvest(Process):
    def __init__(self, inp_q: Queue, opt_q: Queue):
        super().__init__()
        self.inp_q = inp_q
        self.opt_q = opt_q

    def run(self) -> None:
        import numpy as np
        import pyworld

        while True:
            idx, x, res_f0, n_cpu, ts = self.inp_q.get()
            f0, _ = pyworld.harvest(
                x.astype(np.double),
                fs=16000,
                f0_ceil=1100,
                f0_floor=50,
                frame_period=10,
            )
            res_f0[idx] = f0
            if len(res_f0.keys()) >= n_cpu:
                self.opt_q.put(ts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Realtime RVC microphone-to-speaker voice conversion."
    )
    parser.add_argument(
        "--list-devices", action="store_true", help="List audio devices and exit."
    )
    parser.add_argument(
        "--package", type=Path, help="Kaggle exported rvc_models/<experiment>.zip."
    )
    parser.add_argument("--extract-dir", type=Path, help="Directory for extracted package files.")
    parser.add_argument("--model", type=Path, help="Final RVC .pth model.")
    parser.add_argument("--index", type=Path, default=Path(""), help="RVC added_*.index file.")
    parser.add_argument("--input-device", type=int, help="sounddevice input device index.")
    parser.add_argument("--output-device", type=int, help="sounddevice output device index.")
    parser.add_argument("--samplerate", type=int, default=0, help="0 uses model sample rate.")
    parser.add_argument("--input-channels", type=int, default=1)
    parser.add_argument("--output-channels", type=int, default=1)
    parser.add_argument("--test-tone", action="store_true", help="Play a test tone and exit.")
    parser.add_argument("--passthrough", action="store_true", help="Monitor mic input directly and exit on Ctrl+C.")
    parser.add_argument("--duration", type=float, default=3.0, help="Duration for --test-tone.")
    parser.add_argument("--pitch", type=int, default=0, help="Pitch shift in semitones.")
    parser.add_argument("--formant", type=float, default=0.0)
    parser.add_argument("--index-rate", type=float, default=0.3)
    parser.add_argument(
        "--f0method",
        choices=["pm", "harvest", "crepe", "rmvpe", "fcpe"],
        default="rmvpe",
    )
    parser.add_argument("--block-time", type=float, default=0.25)
    parser.add_argument("--crossfade-time", type=float, default=0.05)
    parser.add_argument("--extra-time", type=float, default=2.5)
    parser.add_argument("--threshold", type=int, default=-60, help="Silence gate dB. -60 disables.")
    parser.add_argument("--rms-mix-rate", type=float, default=0.0)
    parser.add_argument("--n-cpu", type=int, default=min(cpu_count(), 4))
    parser.add_argument("--device", default=None, help="cuda:0, cpu, mps, etc. Defaults to RVC config.")
    parser.add_argument("--fp32", action="store_true", help="Disable half precision.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-auto-download-assets", action="store_true")
    return parser.parse_args()


def list_devices() -> None:
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    hostapi_by_idx = {idx: hostapi["name"] for idx, hostapi in enumerate(hostapis)}
    print("Audio devices:")
    for idx, device in enumerate(devices):
        hostapi = hostapi_by_idx.get(device["hostapi"], "")
        in_ch = device["max_input_channels"]
        out_ch = device["max_output_channels"]
        marker = []
        if in_ch:
            marker.append(f"in={in_ch}")
        if out_ch:
            marker.append(f"out={out_ch}")
        if marker:
            print(f"  {idx:>3}: {device['name']} [{hostapi}] ({', '.join(marker)})")
    print(f"Default input/output: {sd.default.device}")


def extract_package(
    package: Path, extract_dir: Path | None, require_index: bool
) -> tuple[Path, Path]:
    package = package.expanduser().resolve()
    if not package.exists():
        raise FileNotFoundError(f"Package not found: {package}")

    target = extract_dir or package.with_name(package.stem + "_extracted")
    target = target.expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(package) as archive:
        archive.extractall(target)

    candidates = [
        path for path in target.rglob("*") if path.suffix.lower() in ARTIFACT_SUFFIXES
    ]
    models = [
        path
        for path in candidates
        if path.suffix.lower() == ".pth" and not path.name.startswith(("G_", "D_"))
    ]
    indexes = [
        path
        for path in candidates
        if path.suffix.lower() == ".index" and "added" in path.name.lower()
    ]
    if not models:
        raise FileNotFoundError(f"No final .pth model found in {target}")
    if require_index and not indexes:
        raise FileNotFoundError(f"No added_*.index found in {target}")
    return sorted(models)[0], sorted(indexes)[0] if indexes else Path("")


def resolve_artifacts(args: argparse.Namespace) -> tuple[Path, Path]:
    model = args.model
    index = args.index
    if args.package:
        model, index = extract_package(
            args.package, args.extract_dir, args.index_rate > 0
        )
    if model is None:
        raise ValueError("Use --package or --model.")
    model = model.expanduser().resolve()
    index = index.expanduser().resolve() if str(index) else Path("")
    if not model.exists():
        raise FileNotFoundError(f"Model not found: {model}")
    if args.index_rate > 0 and (not str(index) or not index.exists()):
        raise FileNotFoundError(
            f"Index not found: {index}. Set --index-rate 0 to disable index."
        )
    return model, index


def download_file(url: str, target: Path) -> None:
    if target.exists() and target.stat().st_size > 0:
        return
    import requests

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    print(f"Downloading {target.name}...")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    tmp.replace(target)


def ensure_assets(f0method: str, auto_download: bool) -> None:
    required = [Path("assets/hubert/hubert_base.pt")]
    if f0method == "rmvpe":
        required.append(Path("assets/rmvpe/rmvpe.pt"))
    missing = [path for path in required if not path.exists()]
    if not missing:
        return
    if not auto_download:
        raise FileNotFoundError(
            "Missing assets: "
            + ", ".join(str(path) for path in missing)
            + ". Run python tools/download_models.py or omit --no-auto-download-assets."
        )
    for path in missing:
        download_file(f"{ASSET_URL}/{path.name}", path)


def default_output_samplerate(output_device: int | None, fallback: int = 48000) -> int:
    if output_device is None:
        return fallback
    device = sd.query_devices(output_device, "output")
    return int(device.get("default_samplerate") or fallback)


def fill_output(outdata: np.ndarray, mono: np.ndarray, channels: int) -> None:
    samples = min(outdata.shape[0], mono.shape[0])
    outdata[:] = 0
    if channels == 1:
        outdata[:samples, 0] = mono[:samples]
    else:
        outdata[:samples, :] = np.repeat(mono[:samples, None], channels, axis=1)


def run_test_tone(args: argparse.Namespace) -> None:
    samplerate = args.samplerate or default_output_samplerate(args.output_device)
    frames = int(args.duration * samplerate)
    t = np.arange(frames, dtype=np.float32) / samplerate
    tone = 0.15 * np.sin(2 * np.pi * 440 * t)
    audio = (
        tone[:, None]
        if args.output_channels == 1
        else np.repeat(tone[:, None], args.output_channels, axis=1)
    )
    print(
        f"Playing {args.duration:.1f}s test tone on output device "
        f"{args.output_device}, samplerate {samplerate}, channels {args.output_channels}"
    )
    sd.play(audio, samplerate=samplerate, device=args.output_device, blocking=True)


def run_passthrough(args: argparse.Namespace) -> None:
    samplerate = args.samplerate or default_output_samplerate(args.output_device)

    def callback(indata: np.ndarray, outdata: np.ndarray, frames, times, status) -> None:
        if status:
            print(status, file=sys.stderr)
        mono = indata.mean(axis=1)
        fill_output(outdata, mono, args.output_channels)

    print("Mic passthrough started. Press Ctrl+C to stop.")
    print(
        f"Input: {args.input_device}, output: {args.output_device}, "
        f"samplerate: {samplerate}, channels: {args.input_channels}->{args.output_channels}"
    )
    with sd.Stream(
        device=(args.input_device, args.output_device),
        channels=(args.input_channels, args.output_channels),
        callback=callback,
        samplerate=samplerate,
        dtype="float32",
    ):
        while True:
            time.sleep(0.5)


class RealtimeMicVC:
    def __init__(self, args: argparse.Namespace, model_path: Path, index_path: Path) -> None:
        from configs.config import Config
        from infer.lib import rtrvc as rvc_for_realtime

        self.args = args
        self.model_path = str(model_path)
        self.index_path = str(index_path) if str(index_path) else ""
        self.inp_q: Queue = Queue()
        self.opt_q: Queue = Queue()
        self.harvest_workers: list[Process] = []

        self.config = Config()
        self.config.use_jit = False
        if args.device:
            self.config.device = args.device
        if args.fp32 or str(self.config.device) == "cpu":
            self.config.is_half = False
        if not args.verbose:
            rvc_for_realtime.printt = lambda *_, **__: None

        if args.f0method == "harvest" and args.n_cpu > 1:
            for _ in range(args.n_cpu):
                worker = Harvest(self.inp_q, self.opt_q)
                worker.daemon = True
                worker.start()
                self.harvest_workers.append(worker)

        self.rvc = rvc_for_realtime.RVC(
            args.pitch,
            args.formant,
            self.model_path,
            self.index_path,
            args.index_rate,
            args.n_cpu,
            self.inp_q,
            self.opt_q,
            self.config,
        )
        if not hasattr(self.rvc, "tgt_sr"):
            raise RuntimeError("RVC model initialization failed. Check the traceback above.")
        if args.f0method == "rmvpe":
            from infer.lib.rmvpe import RMVPE

            print("Preloading rmvpe model...")
            self.rvc.model_rmvpe = RMVPE(
                "assets/rmvpe/rmvpe.pt",
                is_half=self.rvc.is_half,
                device=self.rvc.device,
                use_jit=self.config.use_jit,
            )
        elif args.f0method == "fcpe":
            from torchfcpe import spawn_bundled_infer_model

            print("Preloading fcpe model...")
            self.rvc.device_fcpe = self.rvc.device
            self.rvc.model_fcpe = spawn_bundled_infer_model(self.rvc.device_fcpe)
        self.samplerate = args.samplerate or self.rvc.tgt_sr
        self.zc = self.samplerate // 100
        self.block_frame = int(np.round(args.block_time * self.samplerate / self.zc)) * self.zc
        self.block_frame_16k = 160 * self.block_frame // self.zc
        self.crossfade_frame = int(np.round(args.crossfade_time * self.samplerate / self.zc)) * self.zc
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
        self.sola_search_frame = self.zc
        self.extra_frame = int(np.round(args.extra_time * self.samplerate / self.zc)) * self.zc

        self.input_wav = torch.zeros(
            self.extra_frame + self.crossfade_frame + self.sola_search_frame + self.block_frame,
            device=self.config.device,
            dtype=torch.float32,
        )
        self.input_wav_res = torch.zeros(
            160 * self.input_wav.shape[0] // self.zc,
            device=self.config.device,
            dtype=torch.float32,
        )
        self.rms_buffer = np.zeros(4 * self.zc, dtype="float32")
        self.sola_buffer = torch.zeros(
            self.sola_buffer_frame,
            device=self.config.device,
            dtype=torch.float32,
        )
        self.skip_head = self.extra_frame // self.zc
        self.return_length = (
            self.block_frame + self.sola_buffer_frame + self.sola_search_frame
        ) // self.zc
        self.fade_in_window = (
            torch.sin(
                0.5
                * np.pi
                * torch.linspace(
                    0.0,
                    1.0,
                    steps=self.sola_buffer_frame,
                    device=self.config.device,
                    dtype=torch.float32,
                )
            )
            ** 2
        )
        self.fade_out_window = 1 - self.fade_in_window
        self.resampler = tat.Resample(
            orig_freq=self.samplerate,
            new_freq=16000,
            dtype=torch.float32,
        ).to(self.config.device)
        self.resampler2 = (
            tat.Resample(
                orig_freq=self.rvc.tgt_sr,
                new_freq=self.samplerate,
                dtype=torch.float32,
            ).to(self.config.device)
            if self.rvc.tgt_sr != self.samplerate
            else None
        )

    def callback(self, indata: np.ndarray, outdata: np.ndarray, frames, times, status) -> None:
        if status:
            print(status, file=sys.stderr)
        started = time.perf_counter()
        indata_mono = librosa.to_mono(indata.T)
        if self.args.threshold > -60:
            indata_mono = np.append(self.rms_buffer, indata_mono)
            rms = librosa.feature.rms(
                y=indata_mono, frame_length=4 * self.zc, hop_length=self.zc
            )[:, 2:]
            self.rms_buffer[:] = indata_mono[-4 * self.zc :]
            indata_mono = indata_mono[2 * self.zc - self.zc // 2 :]
            db_threshold = (
                librosa.amplitude_to_db(rms, ref=1.0)[0] < self.args.threshold
            )
            for idx, muted in enumerate(db_threshold):
                if muted:
                    indata_mono[idx * self.zc : (idx + 1) * self.zc] = 0
            indata_mono = indata_mono[self.zc // 2 :]

        self.input_wav[: -self.block_frame] = self.input_wav[self.block_frame :].clone()
        self.input_wav[-indata_mono.shape[0] :] = torch.from_numpy(
            indata_mono
        ).to(self.config.device)
        self.input_wav_res[: -self.block_frame_16k] = self.input_wav_res[
            self.block_frame_16k :
        ].clone()
        self.input_wav_res[-160 * (indata_mono.shape[0] // self.zc + 1) :] = (
            self.resampler(self.input_wav[-indata_mono.shape[0] - 2 * self.zc :])[160:]
        )

        infer_wav = self.rvc.infer(
            self.input_wav_res,
            self.block_frame_16k,
            self.skip_head,
            self.return_length,
            self.args.f0method,
        )
        if self.resampler2 is not None:
            infer_wav = self.resampler2(infer_wav)

        if self.args.rms_mix_rate < 1:
            input_wav = self.input_wav[self.extra_frame :]
            rms1 = librosa.feature.rms(
                y=input_wav[: infer_wav.shape[0]].cpu().numpy(),
                frame_length=4 * self.zc,
                hop_length=self.zc,
            )
            rms1 = torch.from_numpy(rms1).to(self.config.device)
            rms1 = F.interpolate(
                rms1.unsqueeze(0),
                size=infer_wav.shape[0] + 1,
                mode="linear",
                align_corners=True,
            )[0, 0, :-1]
            rms2 = librosa.feature.rms(
                y=infer_wav[:].cpu().numpy(),
                frame_length=4 * self.zc,
                hop_length=self.zc,
            )
            rms2 = torch.from_numpy(rms2).to(self.config.device)
            rms2 = F.interpolate(
                rms2.unsqueeze(0),
                size=infer_wav.shape[0] + 1,
                mode="linear",
                align_corners=True,
            )[0, 0, :-1]
            rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-3)
            infer_wav *= torch.pow(
                rms1 / rms2,
                torch.tensor(
                    1 - self.args.rms_mix_rate,
                    device=self.config.device,
                    dtype=torch.float32,
                ),
            )

        conv_input = infer_wav[
            None, None, : self.sola_buffer_frame + self.sola_search_frame
        ]
        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = torch.sqrt(
            F.conv1d(
                conv_input**2,
                torch.ones(1, 1, self.sola_buffer_frame, device=self.config.device),
            )
            + 1e-8
        )
        sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0]).item()
        infer_wav = infer_wav[sola_offset:]
        infer_wav[: self.sola_buffer_frame] *= self.fade_in_window
        infer_wav[: self.sola_buffer_frame] += self.sola_buffer * self.fade_out_window
        self.sola_buffer[:] = infer_wav[
            self.block_frame : self.block_frame + self.sola_buffer_frame
        ]

        chunk = infer_wav[: self.block_frame].detach().cpu().numpy()
        fill_output(outdata, chunk, self.args.output_channels)
        if self.args.verbose:
            print(f"Infer time: {(time.perf_counter() - started) * 1000:.1f} ms")

    def run(self) -> None:
        print("Realtime RVC started. Press Ctrl+C to stop.")
        print(f"Model: {self.model_path}")
        print(f"Index: {self.index_path or 'disabled'}")
        print(
            f"Device: {self.config.device}, samplerate: {self.samplerate}, "
            f"block: {self.block_frame}"
        )
        with sd.Stream(
            device=(self.args.input_device, self.args.output_device),
            channels=(self.args.input_channels, self.args.output_channels),
            callback=self.callback,
            blocksize=self.block_frame,
            samplerate=self.samplerate,
            dtype="float32",
        ):
            while True:
                time.sleep(0.5)

    def stop(self) -> None:
        for worker in self.harvest_workers:
            worker.terminate()


def main() -> None:
    if sys.platform == "win32":
        freeze_support()
    load_dotenv()
    args = parse_args()
    sys.argv = sys.argv[:1]
    if args.list_devices:
        list_devices()
        return
    if args.test_tone:
        run_test_tone(args)
        return
    if args.passthrough:
        try:
            run_passthrough(args)
        except KeyboardInterrupt:
            print("\nStopped.")
        return
    if sys.platform == "darwin" and str(args.device).startswith("mps"):
        raise RuntimeError(
            "Realtime audio callbacks are unstable with PyTorch MPS on macOS. "
            "Use --device cpu --fp32 instead."
        )

    model_path, index_path = resolve_artifacts(args)
    ensure_assets(args.f0method, not args.no_auto_download_assets)
    app = RealtimeMicVC(args, model_path, index_path)
    try:
        app.run()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        app.stop()


if __name__ == "__main__":
    main()
