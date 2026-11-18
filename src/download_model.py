"""预下载模型脚本 —— 使用国内镜像加速"""
import os
from pathlib import Path

# 国内加速：HF 镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 如果你的梯子开了 TUN，HTTP 请求理论上自动走代理。如果还是很慢，取消下面注释：
# os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"


def download_from_hf_mirror(repo_id: str, save_dir: str):
    """从 HF 镜像下载模型"""
    from huggingface_hub import snapshot_download

    print(f"下载 {repo_id} → {save_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=save_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print("  完成")


def download_from_modelscope(model_id: str, save_dir: str):
    """从 ModelScope 下载（国内最快）"""
    try:
        from modelscope import snapshot_download as ms_snapshot_download
        print(f"从 ModelScope 下载 {model_id} → {save_dir}")
        ms_snapshot_download(
            model_id=model_id,
            local_dir=save_dir,
            revision="master",
        )
        print("  完成")
        return True
    except ImportError:
        print("  modelscope 未安装，回退到 HF 镜像")
        return False
    except Exception as e:
        print(f"  ModelScope 下载失败: {e}，回退到 HF 镜像")
        return False


def main():
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    models = [
        {
            "repo_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "ms_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "save_dir": str(models_dir / "paraphrase-multilingual-MiniLM-L12-v2"),
        },
        {
            "repo_id": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "ms_id": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "save_dir": str(models_dir / "ms-marco-MiniLM-L-6-v2"),
        },
    ]

    for m in models:
        save_dir = m["save_dir"]
        # 跳过已下载的
        if Path(save_dir).exists() and any(Path(save_dir).iterdir()):
            print(f"已存在，跳过: {save_dir}")
            continue

        # 优先 ModelScope（国内最快），不行再 HF 镜像
        downloaded = download_from_modelscope(m["ms_id"], save_dir)
        if not downloaded:
            download_from_hf_mirror(m["repo_id"], save_dir)

    print(f"\n模型已下载到 {models_dir.resolve()}")
    print("使用时设置环境变量或修改 config.yaml 中 embedding.model_name 为本地路径:")
    for m in models:
        print(f"  {m['save_dir']}")


if __name__ == "__main__":
    main()
