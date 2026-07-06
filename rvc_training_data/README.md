# RVC TTS 训练数据与 Colab 训练流程

本目录是独立的 uv 项目，用来做三件事：

1. 从公版文本源下载中文/英文/日文文本。
2. 调用 `src/rvc_data_tools/tts_client.py` 的 TTS 接口生成 RVC 训练音频，每条音频控制在 5-15 秒。
3. 在 Colab 或 Kaggle 上用生成的数据集训练 RVC 模型并构建 index。

请只用你有授权的声音、音色和文本。默认文本源限制在公版/可公开再利用来源；你也可以替换 `sources/catalog.json`。

## 目录结构

```text
rvc_training_data/
  pyproject.toml                 # uv 项目
  sources/catalog.json           # 中/英/日默认文本源
  src/rvc_data_tools/            # 本地数据生成工具
  colab/train_rvc_colab.py       # Colab 训练入口
  kaggle/train_rvc_kaggle.py     # Kaggle 训练入口
  kaggle/train_rvc_kaggle.ipynb  # Kaggle Notebook
  kaggle/requirements-kaggle.txt # Kaggle Python 3.12 依赖
  data/                          # 本地生成结果，默认不入 git
```

## 1. 本地环境

在仓库根目录执行：

```bash
uv sync --project rvc_training_data
cp rvc_training_data/.env.example rvc_training_data/.env
```

编辑 `rvc_training_data/.env`：

```dotenv
TTS_API_KEY=你的_key
TTS_BASE_URL=https://tts.api.c.yiling.top/v1
TTS_MODEL=utf-8-tts
TTS_VOICE=leijun
```

TTS 客户端已经支持 429/超时/临时 5xx 的重试，并会优先读取环境变量或 `rvc_training_data/.env`。根目录 `clone.py` 只是兼容旧用法的薄入口。

## 2. 下载并检查文本源

每次只生成一种语言。支持值：

- `zh`: 简体中文，默认使用本地 `sources/zh_luxun_selected_simplified.txt`
- `en`: 英文，默认 `Alice's Adventures in Wonderland`
- `ja`: 日文，默认 `吾輩は猫である`

```bash
uv run --project rvc_training_data rvc-download-texts \
  --language zh \
  --out rvc_training_data/data/source_cache
```

如果要替换书目，编辑 `rvc_training_data/sources/catalog.json`，增加同语言的 txt 或青空文库 zip 源。

## 3. 生成 30-60 分钟训练数据

示例：生成中文 45 分钟、单条 5-15 秒的音频。

```bash
uv run --project rvc_training_data rvc-generate-dataset \
  --language zh \
  --minutes 45 \
  --voice leijun \
  --out rvc_training_data/data/zh_leijun_45m
```

英文和日文只改 `--language` 与输出目录：

```bash
uv run --project rvc_training_data rvc-generate-dataset \
  --language en \
  --minutes 45 \
  --voice leijun \
  --out rvc_training_data/data/en_leijun_45m
```

```bash
uv run --project rvc_training_data rvc-generate-dataset \
  --language ja \
  --minutes 45 \
  --voice leijun \
  --out rvc_training_data/data/ja_leijun_45m
```

输出结构：

```text
zh_leijun_45m/
  audio/                 # 合格音频，训练用
  rejected/              # 时长不在 5-15 秒的音频
  texts/                 # 每条音频对应文本
  metadata.csv           # 文件、时长、语言、文本清单
  summary.json           # 总时长与来源
  source_cache/          # 下载的原始文本
```

生成完成后校验：

```bash
uv run --project rvc_training_data rvc-validate-dataset \
  rvc_training_data/data/zh_leijun_45m
```

打包上传 Google Drive：

```bash
uv run --project rvc_training_data rvc-zip-dataset \
  rvc_training_data/data/zh_leijun_45m
```

把生成的 `zh_leijun_45m.zip` 上传到 Google Drive，或直接上传整个目录。

## 4. Colab 训练 RVC

推荐直接打开 `rvc_training_data/colab/train_rvc_colab.ipynb`，按单元格执行。Notebook 会 clone 你的 GitHub 仓库，训练脚本会随仓库一起下载，不需要再单独上传 `train_rvc_colab.py`。下面是脚本版等价命令，适合你想复制到自己的 notebook 或 Colab shell。

在 Colab 里建议使用 GPU Runtime。先挂载 Drive：

```python
from google.colab import drive
drive.mount("/content/drive")
```

准备 RVC 仓库和依赖：

```bash
%cd /content
!git clone https://github.com/rikaaa0928/Retrieval-based-Voice-Conversion-WebUI.git RVC
%cd /content/RVC
import sys
requirements_file = "requirements-py311.txt" if sys.version_info >= (3, 11) else "requirements.txt"
print(f"Python: {sys.version.split()[0]}, installing {requirements_file}")
!pip install -r "$requirements_file"
!python tools/download_models.py
```

如果你把数据 zip 上传到 Drive，先解压：

```bash
!mkdir -p /content/drive/MyDrive/rvc_datasets
!unzip -q /content/drive/MyDrive/zh_leijun_45m.zip -d /content/drive/MyDrive/rvc_datasets
```

开始训练：

```bash
!python rvc_training_data/colab/train_rvc_colab.py \
  --repo-dir /content/RVC \
  --dataset-dir /content/drive/MyDrive/rvc_datasets/zh_leijun_45m \
  --experiment leijun_zh_v2_48k \
  --sample-rate 48k \
  --version v2 \
  --if-f0 1 \
  --f0-method rmvpe \
  --gpus 0 \
  --batch-size 8 \
  --total-epoch 200 \
  --save-every-epoch 20 \
  --save-every-weights 1
```

训练产物位置：

- 模型权重：`/content/RVC/assets/weights/leijun_zh_v2_48k.pth` 或 `logs/leijun_zh_v2_48k/G_*.pth`
- 索引：`/content/RVC/logs/leijun_zh_v2_48k/added_IVF*_leijun_zh_v2_48k_v2.index`
- 训练日志：`/content/RVC/logs/leijun_zh_v2_48k/train.log`

训练完成后复制回 Drive：

```bash
!mkdir -p /content/drive/MyDrive/rvc_models/leijun_zh_v2_48k
!cp /content/RVC/assets/weights/leijun_zh_v2_48k.pth /content/drive/MyDrive/rvc_models/leijun_zh_v2_48k/ || true
!cp /content/RVC/logs/leijun_zh_v2_48k/*.index /content/drive/MyDrive/rvc_models/leijun_zh_v2_48k/
!cp /content/RVC/logs/leijun_zh_v2_48k/train.log /content/drive/MyDrive/rvc_models/leijun_zh_v2_48k/
```

## 5. Kaggle 训练 RVC

推荐直接打开 `rvc_training_data/kaggle/train_rvc_kaggle.ipynb`，按单元格执行。Kaggle 右侧设置里需要开启 GPU 和 Internet，然后把本地生成的 `zh_leijun_45m.zip` 添加为 Kaggle Dataset。Kaggle Python 3.12 下不要直接装根目录的 `requirements-py311.txt`，请使用 `rvc_training_data/kaggle/requirements-kaggle.txt`。

Kaggle 的路径和 Colab 不同：

- 输入数据：`/kaggle/input/...`，只读。
- 临时工作区和输出：`/kaggle/working/...`，训练产物会作为 Notebook Output 保存。

Notebook 里的关键参数示例：

```python
REPO_URL = "https://github.com/rikaaa0928/Retrieval-based-Voice-Conversion-WebUI.git"
REPO_DIR = Path("/kaggle/working/RVC")
DATASET_ZIP = Path("/kaggle/input/zh-leijun-45m/zh_leijun_45m.zip")
EXPERIMENT = "leijun_zh_v2_48k"
EXPORT_DIR = Path("/kaggle/working/rvc_models") / EXPERIMENT
```

脚本版等价命令：

```bash
cd /kaggle/working
git clone https://github.com/rikaaa0928/Retrieval-based-Voice-Conversion-WebUI.git RVC
cd /kaggle/working/RVC
python -m pip install --upgrade pip setuptools wheel
pip install -r rvc_training_data/kaggle/requirements-kaggle.txt
python -c "import fairseq; print(fairseq.__file__)"
python tools/download_models.py

python rvc_training_data/kaggle/train_rvc_kaggle.py \
  --repo-dir /kaggle/working/RVC \
  --dataset-zip /kaggle/input/zh-leijun-45m/zh_leijun_45m.zip \
  --experiment leijun_zh_v2_48k \
  --sample-rate 48k \
  --version v2 \
  --if-f0 1 \
  --f0-method rmvpe \
  --gpus 0 \
  --batch-size 8 \
  --total-epoch 200 \
  --save-every-epoch 20 \
  --save-every-weights 1 \
  --export-dir /kaggle/working/rvc_models/leijun_zh_v2_48k
```

训练完成后下载 Kaggle Output 里的：

- `/kaggle/working/rvc_models/leijun_zh_v2_48k/leijun_zh_v2_48k.pth`
- `/kaggle/working/rvc_models/leijun_zh_v2_48k/added_IVF*.index`
- `/kaggle/working/rvc_models/leijun_zh_v2_48k/train.log`

## 6. 本地加载变声器

把 Colab/Kaggle 产物下载回本地仓库：

```text
assets/weights/leijun_zh_v2_48k.pth
logs/leijun_zh_v2_48k/added_IVF...index
```

本地 RVC 环境也可以用 uv 建：

```bash
uv venv --python 3.10
uv pip install -r requirements.txt
python tools/download_models.py
```

启动 WebUI：

```bash
uv run python infer-web.py
```

在 WebUI 里选择刚下载的 `.pth` 和 `.index`，进入实时变声或普通推理流程。

## 常用参数建议

- 数据总长：30-60 分钟；先用 30-45 分钟试训。
- 单条音频：保持默认 `--min-seconds 5 --max-seconds 15`。
- 采样率：优先 `48k` + `v2`。
- F0：说话/唱歌都建议 `--if-f0 1`；Colab 上 `rmvpe` 稳定，显存充足可试 `rmvpe_gpu`。
- batch size：T4 可从 `8` 开始；显存不够降到 `4`。
