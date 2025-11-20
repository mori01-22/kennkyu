# 画像分類サンプル（PyTorch）

このリポジトリは、PyTorch を使ったシンプルな画像分類のサンプルです。CIFAR-10 を使った学習・評価の最小構成を提供します。

## 準備

1. 仮想環境を作成して有効化（例: venv または conda）
2. 依存関係をインストール

```bash
pip install -r requirements.txt
```

## 実行方法

以下のコマンドで学習を開始できます（デフォルトは CIFAR-10、GPU があれば自動利用）:

```bash
python src/train.py --epochs 10 --batch-size 128 --lr 0.001
```

主なファイル:

- `src/train.py`: データ読み込み、学習ループ、評価、モデル保存
- `src/model.py`: シンプルなCNN定義
- `src/utils.py`: 学習/評価ヘルパー

## 備考

- 小さな実験や学習方法の教育用途向け。実運用にはデータ拡張や学習率スケジューラ、より大きなモデルを検討してください。

## モデルアーティファクトの取り扱い

- 本リポジトリでは学習済みモデル（例: `tansu_detector.keras`）はリポジトリ本体に含めず、外部ストレージに置く運用を推奨します。モデルはサイズが大きく、Git 履歴を肥大化させるためです。
- ローカルでモデルを使う手順（例）:
	1. 指定のクラウドストレージ（Google Drive / S3 / Dropbox など）から `tansu_detector.keras` をダウンロードします。
	2. ダウンロードしたファイルをリポジトリのルート（`kennkyu/`）に置きます。
	3. ノートブックや推論スクリプトでモデルを参照します（例: `tf.keras.models.load_model('tansu_detector.keras')`）。
- 共有したい場合は、管理者がアップロード先の URL をここに追記してください。例:

```
モデルのダウンロード: https://drive.google.com/drive/folders/DRIVE_LINK_HERE
```

- もしリポジトリでモデルを扱う必要がある場合は Git LFS の導入を検討してください（大きなファイル管理用）。

