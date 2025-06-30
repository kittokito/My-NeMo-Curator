import pandas as pd
import dask
from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator import Sequential, AddId, ExactDuplicates, FuzzyDuplicates, FuzzyDuplicatesConfig, SemDedup, SemDedupConfig
import logging
import os
import shutil

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def pre_imports() -> None:
    """GPU使用時に必要なライブラリを事前インポート"""
    import cudf  # noqa: F401


def show_duplicate_examples(dataset, duplicates, id_field="id", text_field="text", n_examples=5, max_text_length=100):
    """
    重複ドキュメントの例を表示する関数
    
    Parameters:
    -----------
    dataset: DocumentDataset
        元のデータセット
    duplicates: DocumentDataset
        重複ドキュメントのデータセット（IDとハッシュを含む）
    id_field: str
        IDフィールド名
    text_field: str
        テキストフィールド名
    n_examples: int
        表示する例の数
    max_text_length: int
        表示するテキストの最大長
    """
    logger.info("重複ドキュメントの例を表示中...")
    
    # 重複データを計算して取得
    dup_df = duplicates.df.compute()
    
    if len(dup_df) == 0:
        logger.info("重複ドキュメントが見つかりませんでした。")
        return
    
    # cuDFとpandasの互換性のため、pandasに変換
    if hasattr(dup_df, 'to_pandas'):
        dup_df = dup_df.to_pandas()
    
    # 元のデータセットから重複IDに該当するドキュメントを抽出
    dup_ids = set(dup_df[id_field].tolist())
    
    # 元のデータセットをフィルタリング
    original_df = dataset.df.compute()
    if hasattr(original_df, 'to_pandas'):
        original_df = original_df.to_pandas()
    
    dup_docs = original_df[original_df[id_field].isin(dup_ids)]
    
    # ハッシュ情報をマージ
    dup_docs_with_hash = dup_docs.merge(dup_df, on=id_field, how='inner')
    
    # ハッシュでグループ化
    grouped = dup_docs_with_hash.groupby('_hashes')
    
    print("\n" + "="*80)
    print("重複ドキュメントの例")
    print("="*80)
    
    group_count = 0
    total_duplicates = 0
    
    for hash_val, group in grouped:
        if group_count >= n_examples:
            break
            
        group_size = len(group)
        total_duplicates += group_size
        
        print(f"\n--- 重複グループ {group_count + 1} ---")
        print(f"ハッシュ値: {hash_val}")
        print(f"重複数: {group_size}件")
        print("-" * 60)
        
        for idx, (_, row) in enumerate(group.iterrows()):
            if idx >= 3:  # 各グループで最大3件まで表示
                print(f"... 他 {group_size - 3} 件")
                break
                
            print(f"  ID: {row[id_field]}")
            text = str(row[text_field])
            if len(text) > max_text_length:
                text = text[:max_text_length] + "..."
            print(f"  テキスト: {text}")
            print(f"  文字数: {len(str(row[text_field]))}")
            print()
        
        group_count += 1
    
    # 統計情報を表示
    print("\n" + "="*80)
    print("重複統計")
    print("="*80)
    print(f"総ドキュメント数: {len(original_df):,}")
    print(f"重複ドキュメント数: {len(dup_df):,}")
    print(f"重複率: {len(dup_df)/len(original_df)*100:.2f}%")
    print(f"重複グループ数: {len(grouped):,}")
    print(f"表示したグループ数: {min(group_count, len(grouped))}")


def show_duplicate_stats(dataset, duplicates, id_field="id"):
    """
    重複の統計情報を表示する関数
    
    Parameters:
    -----------
    dataset: DocumentDataset
        元のデータセット
    duplicates: DocumentDataset
        重複ドキュメントのデータセット
    id_field: str
        IDフィールド名
    """
    logger.info("重複統計を計算中...")
    
    # データを計算して取得
    original_df = dataset.df.compute()
    dup_df = duplicates.df.compute()
    
    # cuDFとpandasの互換性のため、pandasに変換
    if hasattr(original_df, 'to_pandas'):
        original_df = original_df.to_pandas()
    if hasattr(dup_df, 'to_pandas'):
        dup_df = dup_df.to_pandas()
    
    total_docs = len(original_df)
    dup_docs = len(dup_df)
    
    if dup_docs == 0:
        print("\n重複ドキュメントが見つかりませんでした。")
        return
    
    # ハッシュでグループ化して重複グループの統計を取得
    grouped = dup_df.groupby('_hashes')
    group_sizes = grouped.size()
    
    print("\n" + "="*60)
    print("詳細統計")
    print("="*60)
    print(f"総ドキュメント数: {total_docs:,}")
    print(f"重複ドキュメント数: {dup_docs:,}")
    print(f"重複率: {dup_docs/total_docs*100:.2f}%")
    print(f"ユニークドキュメント数: {total_docs - dup_docs:,}")
    print(f"重複グループ数: {len(group_sizes):,}")
    print(f"平均重複数/グループ: {group_sizes.mean():.2f}")
    print(f"最大重複数: {group_sizes.max()}")
    print(f"最小重複数: {group_sizes.min()}")
    
    # 重複数の分布
    print("\n重複数の分布:")
    distribution = group_sizes.value_counts().sort_index()
    
    # cuDFとpandasの互換性のため、pandasに変換してから反復処理
    if hasattr(distribution, 'to_pandas'):
        distribution = distribution.to_pandas()
    
    for dup_count, group_count in distribution.items():
        print(f"  {dup_count}件重複: {group_count}グループ")


def main():
    # ========================================
    # 設定パラメータ
    # ========================================
    # デバイス設定: 'cpu' または 'gpu' を選択
    DEVICE = 'gpu'  # 'cpu' または 'gpu' を指定
    
    # キャッシュディレクトリのクリア（オプション）
    CLEAR_CACHE = True  # 必要に応じてFalseに変更
    
    # ワーカー数の設定
    N_WORKERS = 1
    
    # 入力ファイルパス
    INPUT_FILE = "/home/ubuntu/NeMo-Curator/data/raw/plc_normal_04-1.jsonl"
    
    # キャッシュディレクトリを絶対パスで設定
    BASE_DIR = "/home/ubuntu/NeMo-Curator"
    EXACT_CACHE_DIR = os.path.join(BASE_DIR, "exact_dedup_cache")
    FUZZY_CACHE_DIR = os.path.join(BASE_DIR, "fuzzy_dedup_cache")
    
    # 表示設定
    N_EXAMPLES = 20  # 表示する重複グループ数
    MAX_TEXT_LENGTH = 200  # 表示するテキストの最大文字数
    
    logger.info(f"実行モード: {DEVICE.upper()}")
    logger.info(f"入力ファイル: {INPUT_FILE}")
    
    if CLEAR_CACHE:
        cache_dirs = [EXACT_CACHE_DIR, FUZZY_CACHE_DIR]
        for cache_dir in cache_dirs:
            if os.path.exists(cache_dir):
                logger.info(f"キャッシュディレクトリをクリア中: {cache_dir}")
                shutil.rmtree(cache_dir)
    
    # バックエンド設定
    backend = "cudf" if DEVICE == "gpu" else "pandas"
    
    # Dask設定を適用
    with dask.config.set({"dataframe.backend": backend}):
        # Dask クライアント起動
        if DEVICE == 'gpu':
            try:
                import cudf
                client = get_client(
                    cluster_type="gpu",
                    n_workers=N_WORKERS,
                    device_memory_limit="0.85",
                    rmm_pool_size="0.75"
                )
                logger.info("GPU (cuDF) バックエンドを使用します")
                client.run(pre_imports)
            except ImportError:
                logger.warning("cuDFが利用できません。CPUモードに切り替えます")
                DEVICE = 'cpu'
                backend = "pandas"
        
        if DEVICE == 'cpu':
            client = get_client(
                cluster_type="cpu",
                n_workers=N_WORKERS
            )
            logger.info("CPU (pandas) バックエンドを使用します")
        
        # データセットの読み込み
        logger.info("データセットを読み込み中...")
        ds = DocumentDataset.read_json(
            INPUT_FILE, 
            backend=backend,
            blocksize="256MiB"
        )
        
        # データの前処理
        logger.info("データの前処理を実行中...")
        ds.df = ds.df.dropna(subset=['text'])
        ds.df = ds.df[ds.df['text'].str.strip() != '']
        
        # ID追加
        ds = AddId(id_field="id")(ds)
        
        logger.info(f"処理対象のドキュメント数: {len(ds.df)}")
        
        # ========================================
        # ExactDuplicates（重複検出のみ）
        # ========================================
        exact = ExactDuplicates(
            id_field="id",
            text_field="text",
            hash_method="md5",
            perform_removal=False,  # 重複を削除せず、検出のみ
            cache_dir=EXACT_CACHE_DIR
        )
        
        try:
            logger.info("完全一致重複を検出中...")
            duplicates = exact(ds)
            
            # 重複統計を表示
            show_duplicate_stats(ds, duplicates)
            
            # 重複例を表示
            show_duplicate_examples(
                ds, 
                duplicates, 
                n_examples=N_EXAMPLES,
                max_text_length=MAX_TEXT_LENGTH
            )
            
        except Exception as e:
            logger.error(f"重複検出中にエラーが発生しました: {e}")
            logger.error(f"エラーの詳細: {type(e).__name__}")
            raise
        
        logger.info("重複分析が完了しました！")
        
        # クライアントのクローズ
        client.close()


if __name__ == '__main__':
    main()
