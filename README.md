# PDF to Text with Qwen

A Python project for extracting text from PDF files using the Qwen model.

## Features

- Extract text content from PDF documents
- Leverage Qwen AI model for enhanced text processing
- Fast and efficient PDF text extraction
- Easy-to-use command-line interface

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management. uv is an extremely fast Python package and project manager written in Rust.

### Install uv

First, install uv using the official standalone installer:

**macOS and Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Install Project Dependencies

```bash
# Clone the repository
git clone <repository-url>
cd pdf-to-text-qwen

# Install dependencies using uv
uv sync
```

## Usage

```bash
➜  pdf-to-text-qwen git:(main) ✗ uv run main.py -h

usage: main.py [-h] [--model MODEL]
               [--num_splits NUM_SPLITS]
               [--overlap_ratio OVERLAP_RATIO]
               [--stream STREAM]
               pdf_path

Extract text from PDF using Qwen2.5-VL

positional arguments:
  pdf_path              Path to the PDF file

options:
  -h, --help            show this help message and exit
  --model MODEL         Model name to use
  --num_splits NUM_SPLITS
                        Number of splits per page
  --overlap_ratio OVERLAP_RATIO
                        Overlap ratio between splits
  --stream STREAM       Enable streaming output
```

## Requirements

- Python 3.8+
- uv package manager

## License

This project is licensed under the MIT License.
