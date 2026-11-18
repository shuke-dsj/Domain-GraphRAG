from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path


def _require_module(module_name: str, install_hint: str) -> None:
    try:
        __import__(module_name)
    except Exception:  # pragma: no cover - import guard
        print(f"Missing dependency: {module_name}. {install_hint}")
        sys.exit(1)


def _resolve_tesseract_cmd(cli_value: str) -> str | None:
    if cli_value:
        return cli_value

    env_value = os.getenv("TESSERACT_CMD", "").strip()
    if env_value:
        return env_value

    return None


def _render_page_to_image(page, dpi: int):
    import fitz  # type: ignore
    from PIL import Image

    scale = dpi / 72.0
    matrix = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return image


def _extract_embedded_images(doc, page, out_dir: Path, page_index: int) -> list[str]:
    image_paths: list[str] = []

    image_list = page.get_images(full=True)
    if not image_list:
        return image_paths

    for img_index, img in enumerate(image_list, start=1):
        xref = img[0]
        base_image = doc.extract_image(xref)
        ext = base_image.get("ext", "bin")
        img_bytes = base_image.get("image")
        if not img_bytes:
            continue

        filename = f"page_{page_index:04d}_img_{img_index:03d}.{ext}"
        target = out_dir / filename
        target.write_bytes(img_bytes)
        image_paths.append(filename)

    return image_paths


def process_pdf(
    pdf_path: Path,
    out_dir: Path,
    dpi: int,
    lang: str,
    tesseract_cmd: str | None,
    extract_embedded: bool,
) -> None:
    import fitz  # type: ignore
    import pytesseract

    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    images_dir = out_dir / "images"
    text_dir = out_dir / "text"
    embedded_dir = out_dir / "embedded"

    images_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    if extract_embedded:
        embedded_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    pages_data: list[dict[str, object]] = []

    for page_index in range(len(doc)):
        page_number = page_index + 1
        page = doc[page_index]

        image = _render_page_to_image(page, dpi)
        image_name = f"page_{page_number:04d}.png"
        image_path = images_dir / image_name
        image.save(image_path, format="PNG")

        embedded_images: list[str] = []
        if extract_embedded:
            embedded_images = _extract_embedded_images(doc, page, embedded_dir, page_number)

        text = pytesseract.image_to_string(image, lang=lang)
        text_name = f"page_{page_number:04d}.txt"
        text_path = text_dir / text_name
        text_path.write_text(text, encoding="utf-8")

        pages_data.append(
            {
                "page": page_number,
                "image": f"images/{image_name}",
                "text": f"text/{text_name}",
                "embedded_images": [f"embedded/{p}" for p in embedded_images],
            }
        )

        print(f"Processed page {page_number}/{len(doc)}")

    output = {
        "source_pdf": str(pdf_path),
        "dpi": dpi,
        "lang": lang,
        "pages": pages_data,
    }

    output_path = out_dir / "output.json"
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done. Output saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract OCR text and images from a scanned PDF.")
    parser.add_argument("pdf", help="Path to the input PDF file")
    parser.add_argument("-o", "--out", default="output_pdf_ocr", help="Output directory")
    parser.add_argument("--dpi", type=int, default=300, help="Render DPI for page images")
    parser.add_argument("--lang", default="chi_sim+eng", help="Tesseract language(s)")
    parser.add_argument("--tesseract", default="", help="Path to tesseract.exe")
    parser.add_argument("--extract-embedded", action="store_true", help="Extract embedded images")

    args = parser.parse_args()
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    _require_module("fitz", "Run: pip install pymupdf")
    _require_module("PIL", "Run: pip install pillow")
    _require_module("pytesseract", "Run: pip install pytesseract")

    tesseract_cmd = _resolve_tesseract_cmd(args.tesseract)
    if not tesseract_cmd:
        print("Tesseract not found. Set --tesseract or TESSERACT_CMD, or install it.")
        sys.exit(1)

    process_pdf(
        pdf_path=pdf_path,
        out_dir=out_dir,
        dpi=args.dpi,
        lang=args.lang,
        tesseract_cmd=tesseract_cmd,
        extract_embedded=args.extract_embedded,
    )


if __name__ == "__main__":
    main()
