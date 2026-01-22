# 文档转换为PDF
# 提供多种方案：优先使用已有PDF，也可尝试转换doc/docx
from pathlib import Path
from typing import Dict, List, Optional
import subprocess
import platform
import argparse
import shutil

def convert_doc_to_pdf_libreoffice(doc_path: Path, output_dir: Path) -> Optional[Path]:
    """使用LibreOffice命令行工具转换doc/docx为PDF"""
    try:
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # 根据操作系统选择LibreOffice命令
        if platform.system() == 'Darwin':  # macOS
            cmd = '/Applications/LibreOffice.app/Contents/MacOS/soffice'
        elif platform.system() == 'Linux':
            cmd = 'libreoffice'
        elif platform.system() == 'Windows':
            cmd = 'soffice'
        else:
            raise Exception(f"不支持的操作系统: {platform.system()}")
        
        # 执行转换
        subprocess.run([
            cmd,
            '--headless',
            '--convert-to', 'pdf',
            '--outdir', str(output_dir),
            str(doc_path)
        ], check=True, capture_output=True, timeout=60)
        
        # 返回转换后的PDF路径
        pdf_name = doc_path.stem + '.pdf'
        return output_dir / pdf_name
    except Exception as e:
        print(f"    转换失败: {e}")
        return None

def is_office_temp_file(path: Path) -> bool:
    return path.name.startswith("~$")

def build_pdf_dataset(
    source_root: Path,
    output_root: Path,
    convert_docs: bool = True,
    copy_pdfs: bool = True,
    overwrite: bool = False,
) -> Dict[str, int]:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    counts = {
        "pdf_copied": 0,
        "pdf_skipped": 0,
        "doc_converted": 0,
        "doc_skipped": 0,
        "doc_failed": 0,
    }

    if copy_pdfs:
        for pdf_path in sorted(source_root.rglob("*.pdf")):
            if is_office_temp_file(pdf_path):
                continue
            rel_path = pdf_path.relative_to(source_root)
            target_path = output_root / rel_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if target_path.exists() and not overwrite:
                counts["pdf_skipped"] += 1
                continue
            shutil.copy2(pdf_path, target_path)
            counts["pdf_copied"] += 1

    if convert_docs:
        doc_suffixes = {".doc", ".docx"}
        for doc_path in sorted(source_root.rglob("*")):
            if doc_path.suffix.lower() not in doc_suffixes:
                continue
            if is_office_temp_file(doc_path):
                continue
            rel_path = doc_path.relative_to(source_root)
            target_pdf = (output_root / rel_path).with_suffix(".pdf")
            target_pdf.parent.mkdir(parents=True, exist_ok=True)
            if target_pdf.exists() and not overwrite:
                counts["doc_skipped"] += 1
                continue
            pdf_path = convert_doc_to_pdf_libreoffice(doc_path, target_pdf.parent)
            if pdf_path and pdf_path.exists():
                if pdf_path != target_pdf:
                    pdf_path.rename(target_pdf)
                counts["doc_converted"] += 1
            else:
                counts["doc_failed"] += 1

    return counts

def main() -> None:
    parser = argparse.ArgumentParser(description="Convert knowledge base docs to PDFs and build a full PDF dataset.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("labbook/知识库测试集"),
        help="Source dataset root path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("labbook/知识库测试集_pdf"),
        help="Output dataset root path for PDFs.",
    )
    parser.add_argument("--no-convert", action="store_true", help="Do not convert doc/docx files.")
    parser.add_argument("--no-copy", action="store_true", help="Do not copy existing PDFs.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing PDFs in output.")
    args = parser.parse_args()

    counts = build_pdf_dataset(
        source_root=args.source,
        output_root=args.output,
        convert_docs=not args.no_convert,
        copy_pdfs=not args.no_copy,
        overwrite=args.overwrite,
    )

    print("PDF数据集构建完成：")
    print(f"- 已复制PDF: {counts['pdf_copied']}")
    print(f"- 跳过PDF: {counts['pdf_skipped']}")
    print(f"- 已转换DOC/DOCX: {counts['doc_converted']}")
    print(f"- 跳过DOC/DOCX: {counts['doc_skipped']}")
    print(f"- 转换失败: {counts['doc_failed']}")

if __name__ == "__main__":
    main()
