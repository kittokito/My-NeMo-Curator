# RAPIDS/NeMo-Curator インストールエラー解決ガイド

## 問題の概要

`cugraph-cu12`などのRAPIDSパッケージをpipでインストールしようとすると、以下のエラーが発生する：

```
urllib.error.HTTPError: HTTP Error 404: Not Found
RuntimeError: Failed to open project URL https://pypi.nvidia.com/cugraph-cu12/
```

## 原因

- `pypi.nvidia.com`が2025年6月27日から配布を停止している（[RSN #49](https://docs.rapids.ai/notices/rsn0049/)）
- RAPIDS関連のパッケージ（cudf、cugraph、cuml等）は、PyPIに実体のwheelを置かず、`wheel_stub`が`pypi.nvidia.com`から本物のwheelを取得するプレースホルダー形式になっている
- そのため、`pypi.nvidia.com`にアクセスできないとインストールが失敗する

## 解決方法

RAPIDS公式の推奨に従い、Conda/Mambaを使用してインストールする。

### 手順

#### 1. Minicondaのインストール

```bash
# Minicondaインストーラーをダウンロード
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# インストール実行（-bオプションでバッチモード、-pでインストール先指定）
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3

# condaを初期化
~/miniconda3/bin/conda init bash

# bashrcを再読み込み
source ~/.bashrc
```

#### 2. Mambaのインストール（高速なパッケージ解決のため）

```bash
conda install -n base -c conda-forge mamba -y
```

#### 3. RAPIDS環境の作成

```bash
# Python 3.10とRAPIDSを含む環境を作成
mamba create -n curator python=3.10 -c rapidsai -c conda-forge rapids=25.04 -y
```

#### 4. 環境のアクティベート

```bash
conda activate curator
```

#### 5. NeMo-Curatorのインストール

```bash
# NeMo-Curatorのディレクトリに移動
cd /home/ubuntu/NeMo-Curator

# 開発モードでインストール
pip install -e .
```

#### 6. インストールの確認

```bash
python -c "import nemo_curator; print('NeMo-Curator imported successfully')"
```

## 注意事項

- pipとcondaの環境が混在するため、依存関係の競合に関する警告が表示される場合がありますが、NeMo-Curatorは正常に動作します
- 今後この環境を使用する際は、必ず`conda activate curator`でconda環境をアクティベートしてください
- RAPIDS 25.04を使用していますが、バージョンは必要に応じて調整してください

## 代替案（参考）

もしCondaを使用できない場合の代替案：

1. **Dockerを使用**: RAPIDS公式のDockerイメージを使用
2. **CPU版のみを使用**: GPU関連の機能を諦めて、CPU版のみで動作させる
3. **別のインデックスURLを使用**: 一時的な回避策として、別のパッケージインデックスを探す（ただし公式サポートなし）

## 参考リンク

- [RAPIDS Release Support Notice #49](https://docs.rapids.ai/notices/rsn0049/)
- [RAPIDS Installation Guide](https://docs.rapids.ai/install)
- [NeMo-Curator GitHub](https://github.com/NVIDIA/NeMo-Curator)
