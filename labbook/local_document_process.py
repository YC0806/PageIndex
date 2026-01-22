import sys
import warnings
import traceback
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
warnings.filterwarnings('ignore')

project_root = Path("../").resolve()
sys.path.append(str(project_root))

from pageindex import page_index_main, config

# 配置本地版本参数
local_config = config(
    model='deepseek-chat',
    toc_check_page_num=20,
    max_page_num_each_node=10,
    max_token_num_each_node=20000,
    if_add_node_id='yes',
    if_add_node_summary='yes',
    if_add_doc_description='yes',
    if_add_node_text='yes'
)

print("✓ 本地版本配置完成")
print(f"模型: {local_config.model}")


def process_document_local(pdf_path: Path, output_base_dir: Path = Path("./results"),
                          source_base_dir: Path = None) -> Dict[str, Any]:
    """使用本地版本处理单个PDF文档

    Args:
        pdf_path: PDF文件路径
        output_base_dir: 输出根目录
        source_base_dir: 源文件根目录，用于计算相对路径
    """
    try:
        start_time = time.time()

        # 处理PDF
        tree_structure = page_index_main(str(pdf_path), local_config)

        # 计算输出路径，保持目录结构
        if source_base_dir:
            # 获取相对于源目录的相对路径
            relative_path = pdf_path.relative_to(source_base_dir)
            # 在输出目录中重建相同的目录结构
            output_dir = output_base_dir / relative_path.parent
        else:
            output_dir = output_base_dir

        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)

        pdf_name = pdf_path.stem
        output_file = output_dir / f'{pdf_name}_structure.json'

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(tree_structure, f, indent=2, ensure_ascii=False)

        elapsed_time = time.time() - start_time

        return {
            'success': True,
            'file_name': pdf_path.name,
            'file_path': str(pdf_path),
            'tree': tree_structure,
            'output_file': str(output_file),
            'processing_time': elapsed_time
        }
    except Exception as e:
        tb = traceback.format_exc()
        return {
            'success': False,
            'file_name': pdf_path.name,
            'file_path': str(pdf_path),
            'error': str(e),
            'traceback': tb,
            'processing_time': 0
        }

def process_document_wrapper(pdf_file: Path, output_base_dir: Path, source_base_dir: Path) -> Dict[str, Any]:
    """包装函数，用于并行处理"""
    return process_document_local(pdf_file, output_base_dir, source_base_dir)


def check_existing_indices(pdf_files: List[Path], output_base_dir: Path,
                          source_base_dir: Path) -> Dict[str, Any]:
    """检查哪些文件已有索引，哪些缺失

    Args:
        pdf_files: PDF文件列表
        output_base_dir: 输出根目录
        source_base_dir: 源文件根目录

    Returns:
        包含已存在和缺失索引信息的字典
    """
    existing = []
    missing = []

    for pdf_file in pdf_files:
        if source_base_dir:
            relative_path = pdf_file.relative_to(source_base_dir)
            output_dir = output_base_dir / relative_path.parent
        else:
            output_dir = output_base_dir
        output_file = output_dir / f'{pdf_file.stem}_structure.json'

        if output_file.exists():
            existing.append({
                'source': pdf_file,
                'index': output_file,
                'size': output_file.stat().st_size,
                'modified': output_file.stat().st_mtime
            })
        else:
            missing.append({
                'source': pdf_file,
                'index': output_file
            })

    return {
        'existing': existing,
        'missing': missing,
        'total': len(pdf_files)
    }


def batch_process_documents(pdf_files: List[Path], output_base_dir: Path,
                            source_base_dir: Path, max_workers: int = 4,
                            overwrite: bool = False) -> Dict[str, Any]:
    """并行批量处理文档

    Args:
        pdf_files: PDF文件列表
        output_base_dir: 输出根目录
        source_base_dir: 源文件根目录
        max_workers: 最大并行工作进程数
        overwrite: 是否覆盖已存在的索引文件，False表示跳过已存在的
    """
    # 过滤已处理的文件（如果不覆盖）
    if not overwrite:
        files_to_process = []
        skipped_count = 0
        for pdf_file in pdf_files:
            if source_base_dir:
                relative_path = pdf_file.relative_to(source_base_dir)
                output_dir = output_base_dir / relative_path.parent
            else:
                output_dir = output_base_dir
            output_file = output_dir / f'{pdf_file.stem}_structure.json'

            if output_file.exists():
                skipped_count += 1
            else:
                files_to_process.append(pdf_file)

        if skipped_count > 0:
            print(f"⏭ 跳过已存在索引的文件: {skipped_count} 个")
        pdf_files = files_to_process
    else:
        print("⚠️  覆盖模式：将重新生成所有索引文件")

    if not pdf_files:
        print("没有需要处理的文件")
        return {
            'results': {},
            'total_time': 0,
            'success_count': 0,
            'total_count': 0
        }

    print(f"\n使用 {max_workers} 个进程并行处理 {len(pdf_files)} 个文件...")

    results = {}
    completed = 0
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_file = {
            executor.submit(process_document_wrapper, pdf_file, output_base_dir, source_base_dir): pdf_file
            for pdf_file in pdf_files
        }

        # 处理完成的任务
        for future in as_completed(future_to_file):
            pdf_file = future_to_file[future]
            completed += 1

            try:
                result = future.result()
                results[pdf_file.name] = result

                if result['success']:
                    print(f"[{completed}/{len(pdf_files)}] ✓ {pdf_file.name} - {result['processing_time']:.2f}秒")
                else:
                    print(f"[{completed}/{len(pdf_files)}] ✗ {pdf_file.name} - 失败: {result['error']}")
            except Exception as e:
                print(f"[{completed}/{len(pdf_files)}] ✗ {pdf_file.name} - 异常: {str(e)}")
                results[pdf_file.name] = {
                    'success': False,
                    'file_name': pdf_file.name,
                    'error': str(e),
                    'processing_time': 0
                }

    total_time = time.time() - start_time
    success_count = sum(1 for r in results.values() if r['success'])

    return {
        'results': results,
        'total_time': total_time,
        'success_count': success_count,
        'total_count': len(pdf_files)
    }


def main():
    """主函数，支持命令行参数"""
    parser = argparse.ArgumentParser(
        description='并行处理PDF文档，生成索引结构',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 检查索引状态
  python local_document_process.py --check

  # 处理缺失索引的文件（默认跳过已存在的）
  python local_document_process.py

  # 强制覆盖所有索引文件
  python local_document_process.py --overwrite

  # 使用8个进程并行处理
  python local_document_process.py -w 8

  # 指定输入输出目录
  python local_document_process.py /path/to/docs -o /path/to/output
        """
    )
    parser.add_argument('input_dir', type=str, nargs='?',
                       default='知识库测试集_pdf',
                       help='输入文档目录路径')
    parser.add_argument('-o', '--output', type=str, default='./doc_index_results',
                       help='输出目录路径 (默认: ./doc_index_results)')
    parser.add_argument('-w', '--workers', type=int, default=4,
                       help='并行工作进程数 (默认: 4)')
    parser.add_argument('-n', '--max-files', type=int, default=None,
                       help='限制处理文件数量，用于测试 (默认: 处理所有文件)')
    parser.add_argument('--overwrite', action='store_true',
                       help='覆盖已存在的索引文件（默认: 跳过已存在的）')
    parser.add_argument('--check', action='store_true',
                       help='只检查索引状态，不进行处理')
    parser.add_argument('-f', '--formats', type=str, nargs='+',
                       default=['.pdf', '.docx', '.doc'],
                       help='支持的文档格式 (默认: .pdf .docx .doc)')

    args = parser.parse_args()

    # 配置参数
    knowledge_base_path = Path(args.input_dir)
    output_base_dir = Path(args.output)
    max_workers = args.workers
    max_files = args.max_files
    overwrite = args.overwrite
    check_only = args.check
    supported_formats = args.formats

    # 检查输入目录
    if not knowledge_base_path.exists():
        print(f"❌ 错误: 输入目录不存在: {knowledge_base_path}")
        return

    # 扫描知识库文档
    pdf_files = []
    for ext in supported_formats:
        pdf_files.extend(list(knowledge_base_path.rglob(f'*{ext}')))

    # 排除隐藏文件和系统文件
    pdf_files = [f for f in pdf_files if not any(part.startswith('.') for part in f.parts)]

    print(f"✓ 扫描知识库完成")
    print(f"找到文档数量: {len(pdf_files)}")
    print(f"\n文档格式分布:")
    for ext in supported_formats:
        count = sum(1 for f in pdf_files if f.suffix.lower() == ext)
        print(f"  {ext}: {count} 个")

    if len(pdf_files) > 0:
        print(f"\n前5个文档示例:")
        for i, doc in enumerate(pdf_files[:5], 1):
            print(f"  {i}. {doc.name}")

    # 限制处理数量（用于测试）
    if max_files is not None:
        pdf_files = pdf_files[:max_files]
        print(f"\n⚠ 限制处理数量: {len(pdf_files)} 个文件")

    if len(pdf_files) == 0:
        print("\n❌ 没有找到需要处理的文档")
        return

    # 检查模式
    if check_only:
        print("\n" + "=" * 60)
        print("检查索引状态")
        print("=" * 60)

        check_result = check_existing_indices(pdf_files, output_base_dir, knowledge_base_path)

        print(f"\n📊 索引状态统计:")
        print(f"  总文档数: {check_result['total']}")
        print(f"  已有索引: {len(check_result['existing'])} ({len(check_result['existing'])/check_result['total']*100:.1f}%)")
        print(f"  缺失索引: {len(check_result['missing'])} ({len(check_result['missing'])/check_result['total']*100:.1f}%)")

        if check_result['existing']:
            print(f"\n✓ 已有索引的文件 ({len(check_result['existing'])} 个):")
            for i, item in enumerate(check_result['existing'][:10], 1):
                size_kb = item['size'] / 1024
                print(f"  {i}. {item['source'].name} ({size_kb:.1f} KB)")
            if len(check_result['existing']) > 10:
                print(f"  ... 还有 {len(check_result['existing']) - 10} 个文件")

        if check_result['missing']:
            print(f"\n✗ 缺失索引的文件 ({len(check_result['missing'])} 个):")
            for i, item in enumerate(check_result['missing'][:10], 1):
                print(f"  {i}. {item['source'].name}")
            if len(check_result['missing']) > 10:
                print(f"  ... 还有 {len(check_result['missing']) - 10} 个文件")

        print(f"\n💡 提示:")
        if check_result['missing']:
            print(f"  运行 'python {sys.argv[0]}' 处理缺失索引的文件")
        if check_result['existing']:
            print(f"  运行 'python {sys.argv[0]} --overwrite' 重新生成所有索引")

        return

    # 并行批量处理
    print("=" * 60)
    print("开始并行文档处理")
    print("=" * 60)

    batch_result = batch_process_documents(
        pdf_files,
        output_base_dir,
        knowledge_base_path,
        max_workers=max_workers,
        overwrite=overwrite
    )

    # 输出统计信息
    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)
    print(f"成功: {batch_result['success_count']} / {batch_result['total_count']}")
    print(f"总耗时: {batch_result['total_time']:.2f}秒")
    if batch_result['success_count'] > 0:
        avg_time = sum(r['processing_time'] for r in batch_result['results'].values() if r['success']) / batch_result['success_count']
        print(f"平均耗时: {avg_time:.2f}秒/文档")
        print(f"输出目录: {output_base_dir.resolve()}")

    # 输出失败的文档
    failed_docs = [name for name, r in batch_result['results'].items() if not r['success']]
    if failed_docs:
        print(f"\n失败的文档 ({len(failed_docs)}):")
        for doc_name in failed_docs:
            result = batch_result['results'][doc_name]
            print(f"  - {doc_name}: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()