"""GraphRAG 统一入口 —— CLI 参数解析，支持 ingest 和 query 模式"""

import argparse
import logging
import os
import sys

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config.settings import Settings
from src.llm.deepseek import DeepSeekAdapter
from src.pipeline.ingestion import IngestionPipeline
from src.pipeline.query import QueryPipeline
from src.pipeline.conversation import ConversationManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("graphrag")


def create_llm(settings: Settings):
    """根据配置创建 LLM 实例"""
    provider = settings.llm.get("provider", "deepseek")
    if provider == "deepseek":
        api_key = settings.llm.get("api_key", os.getenv("LLM_API_KEY", ""))
        if not api_key:
            raise ValueError("LLM API key not set. Set LLM_API_KEY env var or api_key in config.yaml")
        return DeepSeekAdapter(
            api_key=api_key,
            base_url=settings.llm.get("base_url", "https://api.siliconflow.cn/v1"),
            default_model=settings.llm.get("default_model", "Pro/deepseek-ai/DeepSeek-V3"),
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def cmd_ingest(args):
    """入库命令"""
    settings = Settings.load(args.config)
    llm = create_llm(settings)

    pipeline = IngestionPipeline(settings, llm)

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            texts = [f.read()]
        logger.info(f"Loaded file: {args.file} ({len(texts[0])} chars)")
    elif args.text:
        texts = [args.text]
    elif args.text_dir:
        import glob
        patterns = ["*.txt", "*.md"]
        text_files = []
        for pat in patterns:
            text_files.extend(glob.glob(os.path.join(args.text_dir, pat)))
        for tf in sorted(text_files):
            with open(tf, "r", encoding="utf-8") as f:
                text = f.read()
            pipeline.run_from_texts([text], source_doc=args.source or os.path.basename(tf), skip_resolution=args.skip_resolution)
        logger.info(f"Processed {len(text_files)} text files from {args.text_dir}")
        return
    else:
        logger.error("Provide --file, --text, or --text_dir")
        return

    pipeline.run_from_texts(texts, source_doc=args.source or (os.path.basename(args.file) if args.file else ""), skip_resolution=args.skip_resolution)


def cmd_query(args):
    """查询命令"""
    settings = Settings.load(args.config)
    llm = create_llm(settings)

    pipeline = QueryPipeline(settings, llm)
    conv = ConversationManager()

    try:
        if args.query:
            # 单次查询
            resolved = conv.resolve_pronouns(args.query)
            result = pipeline.ask(resolved)
            print(f"\n{'='*60}\n{result['answer']}\n{'='*60}")
            if args.verbose:
                print(f"\n[上下文路径]\n{result.get('context', 'N/A')}")
        else:
            # 交互模式
            print("领域GraphRAG 交互查询模式 (输入 quit 退出, verbose 显示详细上下文)")
            while True:
                try:
                    query = input("\n> ").strip()
                    if not query:
                        continue
                    if query.lower() == "quit":
                        break
                    if query.lower() == "verbose":
                        args.verbose = not args.verbose
                        print(f"verbose = {args.verbose}")
                        continue

                    resolved = conv.resolve_pronouns(query)
                    result = pipeline.ask(resolved)

                    print(f"\n{result['answer']}\n")

                    if args.verbose:
                        print(f"[上下文路径]\n{result.get('context', 'N/A')}\n")

                    # 从答案中提取实体用于下一轮指代消解
                    conv.add_turn(query, result["answer"], [p["entry_entity"] for p in result.get("paths", []) if p.get("entry_entity")])

                except KeyboardInterrupt:
                    print("\n")
                    break
    finally:
        pipeline.close()


def main():
    parser = argparse.ArgumentParser(description="领域GraphRAG - Domain Knowledge Graph RAG System")
    parser.add_argument("--config", "-c", default="config.yaml", help="配置文件路径")

    sub = parser.add_subparsers(dest="command", help="子命令")

    # ingest
    ingest = sub.add_parser("ingest", help="入库：文本 → 知识图谱")
    ingest.add_argument("--file", "-f", help="单个文件路径（.md/.txt）")
    ingest.add_argument("--text", help="入库文本")
    ingest.add_argument("--text_dir", help="文本目录（.txt/.md）")
    ingest.add_argument("--source", default="", help="来源文档名")
    ingest.add_argument("--skip-resolution", action="store_true", help="跳过实体消歧")

    # query
    query = sub.add_parser("query", help="查询：自然语言问答")
    query.add_argument("--query", "-q", help="单次查询（不指定则进入交互模式）")
    query.add_argument("--verbose", "-v", action="store_true", help="显示检索上下文")

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "query":
        cmd_query(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
