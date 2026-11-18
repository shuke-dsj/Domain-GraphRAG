from PyPDF2 import PdfReader, PdfWriter
import os

def split_pdf_by_pages(input_pdf_path, pages_per_file=200):
    """
    按固定页数拆分PDF文件
    :param input_pdf_path: 原PDF文件路径
    :param pages_per_file: 每个拆分文件的页数，默认200页
    """
    # 读取PDF文件
    try:
        pdf_reader = PdfReader(input_pdf_path)
        total_pages = len(pdf_reader.pages)
        print(f"✅ 原PDF总页数：{total_pages}")

        if total_pages <= pages_per_file:
            print("⚠️ PDF总页数不足200页，无需拆分")
            return

        # 获取原文件名称（用于生成新文件名）
        file_dir = os.path.dirname(input_pdf_path)
        file_name = os.path.splitext(os.path.basename(input_pdf_path))[0]

        # 计算需要拆分的份数
        part_num = 1
        for start_page in range(0, total_pages, pages_per_file):
            # 结束页码
            end_page = min(start_page + pages_per_file, total_pages)
            
            # 创建新的PDF写入器
            pdf_writer = PdfWriter()
            
            # 添加指定页码范围的页面
            for page_num in range(start_page, end_page):
                pdf_writer.add_page(pdf_reader.pages[page_num])
            
            # 保存拆分后的文件
            output_filename = f"{file_name}_部分{part_num}.pdf"
            output_path = os.path.join(file_dir, output_filename)
            
            with open(output_path, "wb") as output_pdf:
                pdf_writer.write(output_pdf)
            
            print(f"📄 已生成：{output_filename}（第{start_page+1}页 - 第{end_page}页）")
            part_num += 1

        print(f"\n🎉 拆分完成！共生成 {part_num-1} 个文件")

    except Exception as e:
        print(f"❌ 处理失败：{str(e)}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="按页数拆分 PDF")
    parser.add_argument("pdf_path", help="待拆分的 PDF 路径")
    parser.add_argument("--pages-per-file", type=int, default=200, help="每个输出文件的页数")
    args = parser.parse_args()
    if args.pages_per_file < 1:
        parser.error("--pages-per-file 必须大于 0")
    split_pdf_by_pages(args.pdf_path, pages_per_file=args.pages_per_file)
