import ollama
import fitz  # PyMuPDF
from PIL import Image
import io
import base64
import argparse
from difflib import SequenceMatcher


def pdf_to_images(pdf_path, num_splits=None, overlap_ratio=0.1):
    """Convert PDF pages to PIL Images with optional splitting

    Args:
        pdf_path: Path to the PDF file
        num_splits: Number of parts to split each page into. If None, no splitting occurs
        overlap_ratio: Ratio of overlap between splits (0.0 to 1.0)
    """
    doc = fitz.open(pdf_path)
    images = []

    for page_num in range(doc.page_count):
        page = doc[page_num]
        # Higher DPI for better text recognition
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_data = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_data))

        if num_splits and num_splits > 1:
            # Calculate split height based on number of splits
            split_height = image.height // num_splits
            overlap_pixels = int(split_height * overlap_ratio)

            for i in range(num_splits):
                # Calculate start and end positions
                y_start = max(0, i * split_height - (overlap_pixels if i > 0 else 0))
                y_end = min(
                    (i + 1) * split_height
                    + (overlap_pixels if i < num_splits - 1 else 0),
                    image.height,
                )

                # For the last split, extend to the bottom of the image
                if i == num_splits - 1:
                    y_end = image.height

                # Crop the image section
                cropped = image.crop((0, y_start, image.width, y_end))
                images.append(cropped)
        else:
            images.append(image)

    doc.close()
    return images


def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def remove_overlapping_text(texts, similarity_threshold=0.7):
    """Remove overlapping text between consecutive page sections"""
    if len(texts) <= 1:
        return texts

    cleaned_texts = [texts[0]]  # Keep first text as-is

    for i in range(1, len(texts)):
        current_text = texts[i].strip()
        previous_text = cleaned_texts[-1].strip()

        # Split into lines for better comparison
        current_lines = current_text.split("\n")
        previous_lines = previous_text.split("\n")

        # Find overlapping content at the beginning of current text
        best_match_end = 0
        for j in range(min(10, len(current_lines))):  # Check first 10 lines
            current_segment = "\n".join(current_lines[: j + 1])

            # Check if this segment appears in the previous text
            for k in range(max(0, len(previous_lines) - 15), len(previous_lines)):
                previous_segment = "\n".join(previous_lines[k:])
                similarity = SequenceMatcher(
                    None, current_segment.lower(), previous_segment.lower()
                ).ratio()

                if similarity > similarity_threshold:
                    best_match_end = j + 1
                    break

        # Remove overlapping part from current text
        if best_match_end > 0:
            cleaned_current = "\n".join(current_lines[best_match_end:])
        else:
            cleaned_current = current_text

        cleaned_texts.append(cleaned_current)

    return cleaned_texts


def extract_text_from_pdf(
    pdf_path: str,
    model="qwen2.5vl:7b",
    stream=False,
    num_splits=4,
    overlap_ratio=0.1,
):
    """Extract text from PDF using Qwen2.5-VL via Ollama"""
    images = pdf_to_images(pdf_path, num_splits=num_splits, overlap_ratio=overlap_ratio)
    extracted_text = []

    for i, image in enumerate(images):
        print(f"Processing page {i + 1}/{len(images)}...")

        # Convert image to base64
        img_base64 = image_to_base64(image)

        print("Extracting text from page", i + 1)
        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "Extract all the text from this document page. Maintain the original formatting and structure as much as possible. Only return the extracted text, no additional commentary.",
                    "images": [img_base64],
                }
            ],
            stream=stream,
            options={
                "temperature": 0.1,  # Lower temperature
                "top_p": 0.8,  # Reduce randomness
                "top_k": 10,  # Limit token choices
                "repeat_penalty": 1.2,  # Penalize repetition
                # "num_predict": 1500,  # Limit output length
                # "stop": ["---", "END"],  # Stop tokens
            },
        )

        page_text = ""
        for chunk in response:
            if chunk["message"]["content"]:
                page_text += chunk["message"]["content"]
                print(chunk["message"]["content"], end="", flush=True)

        print()  # New line after streaming is complete

        extracted_text.append(page_text.strip())

    # Remove overlapping text between sections
    cleaned_texts = remove_overlapping_text(extracted_text)

    # Join all text without page markers
    return "\n\n".join(text.strip() for text in cleaned_texts if text.strip())


def main():
    """Main function to handle command line arguments and initiate text extraction"""
    parser = argparse.ArgumentParser(
        description="Extract text from PDF using Qwen2.5-VL"
    )

    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--model", default="qwen2.5vl:7b", help="Model name to use")
    parser.add_argument(
        "--num_splits", type=int, default=4, help="Number of splits per page "
    )
    parser.add_argument(
        "--overlap_ratio", type=float, default=0.1, help="Overlap ratio between splits"
    )
    parser.add_argument(
        "--stream", type=bool, default=True, help="Enable streaming output"
    )

    args = parser.parse_args()

    extracted_text = extract_text_from_pdf(
        args.pdf_path,
        model=args.model,
        stream=args.stream,
        num_splits=args.num_splits,
        overlap_ratio=args.overlap_ratio,
    )

    # Save to markdown file
    output_path = args.pdf_path.rsplit(".", 1)[0] + "_extracted.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(extracted_text)

    print(f"Text extraction complete! Check '{output_path}' for results.")


# Usage
if __name__ == "__main__":
    main()
