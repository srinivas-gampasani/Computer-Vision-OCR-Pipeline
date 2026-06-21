#!/usr/bin/env python3
"""
scripts/run_ocr.py

Standalone CLI for the Computer Vision OCR Pipeline — no API server required.

Usage:
    cd backend && python ../scripts/run_ocr.py --image data/sample_documents/patient_intake_form.png
    cd backend && python ../scripts/run_ocr.py --image path/to/scan.png --output result.json
    cd backend && python ../scripts/run_ocr.py --all-samples
"""
import argparse
import json
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))


def banner():
    print("\n" + "=" * 65)
    print("  Computer Vision OCR Pipeline")
    print("  Srinivas Gampasani · AI & ML Engineering")
    print("=" * 65 + "\n")


def process_one(pipeline, image_path: str, output_path: str = None, verbose: bool = True):
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return None

    if verbose:
        print(f"📄 Processing: {image_path}")

    t0 = time.time()
    result = pipeline.process_file(image_path)
    elapsed = time.time() - t0

    if verbose:
        print(f"   ✓ Document type    : {result.document.document_type}")
        print(f"   ✓ OCR engine        : {result.ocr_engine_used}")
        print(f"   ✓ Mean confidence   : {result.document.metadata['mean_ocr_confidence']:.1f}%")
        print(f"   ✓ Words extracted   : {result.document.metadata['total_words']}")
        print(f"   ✓ Key-value pairs   : {len(result.document.key_value_pairs)}")
        print(f"   ✓ Tables detected   : {result.table_regions_detected}")
        print(f"   ✓ Columns detected  : {result.columns_detected}")
        print(f"   ✓ Skew corrected    : {result.skew_corrected_degrees:.2f}°")
        print(f"   ✓ Pipeline time     : {result.total_pipeline_time_ms:.0f}ms")

        if result.document.key_value_pairs:
            print(f"\n   📋 Extracted fields:")
            for kv in result.document.key_value_pairs[:20]:
                print(f"      {kv.key:25s}: {kv.value}")

        if result.document.tables:
            print(f"\n   📊 Extracted tables:")
            for t_idx, table in enumerate(result.document.tables):
                print(f"      Table {t_idx + 1} ({len(table)} rows):")
                for row in table[:10]:
                    cells = " | ".join(c.text[:15] for c in row)
                    print(f"        {cells}")

    output_dict = result.to_dict()

    if output_path:
        with open(output_path, "w") as f:
            json.dump(output_dict, f, indent=2)
        if verbose:
            print(f"\n   💾 Full JSON saved to: {output_path}")

    return output_dict


def main():
    banner()
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Path to a single image to process")
    parser.add_argument("--output", help="Path to save JSON output")
    parser.add_argument("--all-samples", action="store_true", help="Process all bundled sample documents")
    parser.add_argument("--paddle", action="store_true", help="Use PaddleOCR instead of Tesseract (if installed)")
    parser.add_argument("--no-deskew", action="store_true", help="Disable deskew correction")
    parser.add_argument("--no-denoise", action="store_true", help="Disable denoising")
    args = parser.parse_args()

    from app.services.ocr_pipeline import OCRPipeline

    pipeline = OCRPipeline(
        prefer_paddle=args.paddle,
        deskew=not args.no_deskew,
        denoise=not args.no_denoise,
    )

    if args.all_samples:
        sample_dir = "data/sample_documents"
        files = sorted([f for f in os.listdir(sample_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        os.makedirs("outputs", exist_ok=True)

        results_summary = []
        for f in files:
            print("\n" + "-" * 65)
            output_dict = process_one(
                pipeline, os.path.join(sample_dir, f),
                output_path=os.path.join("outputs", f.rsplit(".", 1)[0] + ".json"),
            )
            if output_dict:
                results_summary.append({
                    "file": f,
                    "type": output_dict["document_type"],
                    "confidence": output_dict["metadata"]["mean_ocr_confidence"],
                    "words": output_dict["metadata"]["total_words"],
                })

        print("\n" + "=" * 65)
        print("  SUMMARY — All Sample Documents")
        print("=" * 65)
        for r in results_summary:
            print(f"  {r['file']:45s}  {r['type']:10s}  conf={r['confidence']:.1f}%  words={r['words']}")

        avg_conf = sum(r["confidence"] for r in results_summary) / len(results_summary) if results_summary else 0
        print(f"\n  Average confidence across {len(results_summary)} documents: {avg_conf:.1f}%")

    elif args.image:
        process_one(pipeline, args.image, args.output)
    else:
        print("Specify --image <path> or --all-samples")
        sys.exit(1)


if __name__ == "__main__":
    main()
