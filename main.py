import ollama
import fitz  # PyMuPDF
from PIL import Image
import io
import base64
import pretty_errors
import sys
import argparse


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


def extract_text_from_pdf(pdf_path, model_name="qwen2.5vl:7b"):
    """Extract text from PDF using Qwen2.5-VL via Ollama"""
    images = pdf_to_images(pdf_path, num_splits=4)
    extracted_text = []

    for i, image in enumerate(images):
        print(f"Processing page {i + 1}/{len(images)}...")

        # Convert image to base64
        img_base64 = image_to_base64(image)

        print("Extracting text from page", i + 1)
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": "Extract all the text from this document page. Maintain the original formatting and structure as much as possible. Only return the extracted text, no additional commentary.",
                    "images": [img_base64],
                }
            ],
            stream=True,
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

        # page_text = response["message"]["content"]
        extracted_text.append(f"--- Page {i + 1} ---\n{page_text}\n")

    return "\n".join(extracted_text)


def main():
    parser = argparse.ArgumentParser(
        description="Extract text from PDF using Qwen2.5-VL"
    )
    parser.add_argument(
        "pdf_path", default="your_document.pdf", help="Path to the PDF file"
    )
    parser.add_argument("--model", default="qwen2.5vl:7b", help="Model name to use")

    args = parser.parse_args()

    extracted_text = extract_text_from_pdf(args.pdf_path, args.model)

    # Save to file
    with open("extracted_text.txt", "w", encoding="utf-8") as f:
        f.write(extracted_text)

    print("Text extraction complete! Check 'extracted_text.txt'")


# Usage
if __name__ == "__main__":
    main()
