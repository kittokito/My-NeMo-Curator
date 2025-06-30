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


def main():
    # ========================================
    # 設定パラメータ
    # ========================================
    # デバイス設定: 'cpu' または 'gpu' を選択
    DEVICE = 'gpu'  # 'cpu' または 'gpu' を指定
    
    # キャッシュディレクトリのクリア（オプション）
    # パラメータ変更時や前回の実行が中断された場合は、キャッシュをクリアすることを推奨
    CLEAR_CACHE = True  # 必要に応じてFalseに変更
    
    # ワーカー数の設定
    N_WORKERS = 1
    
    # 入出力ファイルパス
    INPUT_FILE = "/home/ubuntu/NeMo-Curator/data/raw/plc_normal_04-1.jsonl"
    OUTPUT_FILE = "/home/ubuntu/NeMo-Curator/data/deduped/dedup_plc_normal_04-1.jsonl"
    
    # キャッシュディレクトリを絶対パスで設定
    BASE_DIR = "/home/ubuntu/NeMo-Curator"
    EXACT_CACHE_DIR = os.path.join(BASE_DIR, "exact_dedup_cache")
    FUZZY_CACHE_DIR = os.path.join(BASE_DIR, "fuzzy_dedup_cache")
    SEM_CACHE_DIR = os.path.join(BASE_DIR, "sem_cache")
    
    logger.info(f"実行モード: {DEVICE.upper()}")
    logger.info(f"入力ファイル: {INPUT_FILE}")
    logger.info(f"出力ファイル: {OUTPUT_FILE}")
    
    if CLEAR_CACHE:
        cache_dirs = [EXACT_CACHE_DIR, FUZZY_CACHE_DIR, SEM_CACHE_DIR]
        for cache_dir in cache_dirs:
            if os.path.exists(cache_dir):
                logger.info(f"キャッシュディレクトリをクリア中: {cache_dir}")
                shutil.rmtree(cache_dir)
    
    # バックエンド設定
    backend = "cudf" if DEVICE == "gpu" else "pandas"
    
    # Dask設定を適用
    with dask.config.set({"dataframe.backend": backend}):
        # Dask クライアント起動（デバイスに応じて設定を変更）
        if DEVICE == 'gpu':
            # GPU用の設定
            try:
                import cudf
                client = get_client(
                    cluster_type="gpu",
                    n_workers=N_WORKERS,
                    device_memory_limit="0.85",  # GPUメモリ制限（メモリの85%）
                    rmm_pool_size="0.75"  # RMMプールサイズ（メモリの75%）
                )
                logger.info("GPU (cuDF) バックエンドを使用します")
                # GPU使用時は事前インポートを実行
                client.run(pre_imports)
            except ImportError:
                logger.warning("cuDFが利用できません。CPUモードに切り替えます")
                DEVICE = 'cpu'
                backend = "pandas"
        
        if DEVICE == 'cpu':
            # CPU用の設定
            client = get_client(
                cluster_type="cpu",
                n_workers=N_WORKERS
            )
            logger.info("CPU (pandas) バックエンドを使用します")
        
        # データセットの読み込み（バックエンドを指定して読み込み）
        logger.info("データセットを読み込み中...")
        ds = DocumentDataset.read_json(
            INPUT_FILE, 
            backend=backend,
            blocksize="256MiB"  # メモリ効率のため適切なブロックサイズを設定
        )
        
        # データの前処理：null値の確認と除去
        logger.info("データの前処理を実行中...")
        # null値を含む行を除去
        ds.df = ds.df.dropna(subset=['text'])
        
        # 空文字列の行も除去
        ds.df = ds.df[ds.df['text'].str.strip() != '']
        
        # ID追加
        ds = AddId(id_field="id")(ds)
        
        # データ数を確認
        logger.info(f"処理対象のドキュメント数: {len(ds.df)}")
        
        # ========================================
        # 1. ExactDuplicates（完全一致重複除去）
        # ========================================
        exact = ExactDuplicates(
            id_field="id",
            text_field="text",
            hash_method="md5",
            perform_removal=True,
            cache_dir=EXACT_CACHE_DIR
        )
        
        # ========================================
        # 2. FuzzyDuplicates（ファジー重複除去）
        # ========================================
        
        # メモリ問題を回避するため、設定を調整
        fuzzy_cfg = FuzzyDuplicatesConfig(
            cache_dir=FUZZY_CACHE_DIR,
            id_field="id",
            text_field="text",
            char_ngrams=12,
            num_buckets=40,  
            hashes_per_bucket=6,
            jaccard_threshold=0.92,
            perform_removal=True,
            false_positive_check=True,
            num_anchors=3
        )
        fuzzy = FuzzyDuplicates(config=fuzzy_cfg, perform_removal=True)
        
        # # ========================================
        # # 3. SemDedup（セマンティック重複除去）
        # # ========================================
        
        # # メモリ効率を考慮した設定
        # sem_cfg = SemDedupConfig(
        #     cache_dir=SEM_CACHE_DIR,
        #     embedding_model_name_or_path="jinaai/jina-embeddings-v2-base-code",
        #     embedding_batch_size=512,  # バッチサイズを小さく（1024→512）
        #     eps_to_extract=0.08,
        #     n_clusters=1000,  # クラスタ数を減らす（1000→100）
        #     max_iter=50,  # 反復回数を減らす（100→50）
        #     batched_cosine_similarity=512,
        #     which_to_keep="hard"  # バッチサイズを調整
        # )
        # semantic = SemDedup(config=sem_cfg, input_column="text", id_column="id", perform_removal=True)
        
        # パイプラインの構築と実行
        try:
            # 各ステージを個別に実行（デバッグしやすくするため）
            logger.info("パイプラインを実行中...")
            
            # Stage 1: Exact deduplication
            ds_after_exact = exact(ds)
            logger.info(f"完全一致重複除去後のドキュメント数: {len(ds_after_exact.df)}")
            
            # # Stage 2: Fuzzy deduplication
            # ds_after_fuzzy = fuzzy(ds_after_exact)
            # logger.info(f"ファジー重複除去後のドキュメント数: {len(ds_after_fuzzy.df)}")
            
            # # Stage 3: Semantic deduplication
            # clean_ds = semantic(ds_after_fuzzy)
            # logger.info(f"セマンティック重複除去後のドキュメント数: {len(clean_ds.df)}")
            
            # オプション
            clean_ds = ds_after_exact
            # clean_ds = ds_after_fuzzy
            
        except Exception as e:
            logger.error(f"パイプライン実行中にエラーが発生しました: {e}")
            logger.error(f"エラーの詳細: {type(e).__name__}")
            
            # デバッグ情報を出力
            if 'ds_after_exact' in locals():
                logger.info(f"Exact後のデータフレームタイプ: {type(ds_after_exact.df)}")
                logger.info(f"Exact後のカラム: {ds_after_exact.df.columns.tolist() if hasattr(ds_after_exact.df, 'columns') else 'N/A'}")
                
            # エラー時は中間結果を保存
            if 'ds_after_exact' in locals():
                try:
                    intermediate_exact_path = os.path.join(BASE_DIR, "intermediate_exact_dedup")
                    ds_after_exact.to_json(intermediate_exact_path, write_to_filename=True)
                    logger.info("Exact重複除去後の中間結果を保存しました")
                except:
                    logger.error("中間結果の保存に失敗しました")
                    
            if 'ds_after_fuzzy' in locals():
                try:
                    intermediate_fuzzy_path = os.path.join(BASE_DIR, "intermediate_fuzzy_dedup")
                    ds_after_fuzzy.to_json(intermediate_fuzzy_path, write_to_filename=True)
                    logger.info("Fuzzy重複除去後の中間結果を保存しました")
                except:
                    logger.error("中間結果の保存に失敗しました")
            raise
        
        # 結果の保存
        logger.info("結果を保存中...")
        
        # 出力形式に応じて保存
        if OUTPUT_FILE.endswith('.parquet'):
            # Parquet形式で保存（より効率的）
            clean_ds.to_parquet(OUTPUT_FILE)
        else:
            # JSONL形式で保存
            # 複数パーティションを1つに統合してから保存
            if hasattr(clean_ds.df, 'repartition'):
                logger.info("データフレームを1つのパーティションに統合中...")
                clean_ds.df = clean_ds.df.repartition(npartitions=1)
            
            # 単一ファイルとして保存
            clean_ds.to_json(OUTPUT_FILE, write_to_filename=True)
        
        logger.info("処理が完了しました！")
        logger.info(f"使用したデバイス: {DEVICE.upper()}")
        logger.info(f"出力ファイル: {OUTPUT_FILE}")
        
        # クライアントのクローズ
        client.close()


if __name__ == '__main__':
    main()
