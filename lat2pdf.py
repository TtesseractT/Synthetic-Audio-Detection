#!/usr/bin/env python3

import subprocess
import os
import sys
import shutil

def compile_latex_to_pdf(latex_file):
    # Check if pdflatex is installed
    if shutil.which('pdflatex') is None:
        print("Error: pdflatex is not installed. Please install a LaTeX distribution (e.g., TeX Live, MiKTeX).")
        sys.exit(1)
    
    # Construct the command to compile the LaTeX file with pdflatex in nonstop mode.
    cmd = ['pdflatex', '-interaction=nonstopmode', latex_file]
    print(f"Compiling {latex_file} to PDF...")

    # Run the command and capture the output
    process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Check if compilation was successful
    if process.returncode != 0:
        print("LaTeX compilation failed. Output:")
        print(process.stdout)
        print(process.stderr)
        sys.exit(1)
    else:
        print("PDF compilation succeeded.")

    # The output PDF file name is expected to be the same as the LaTeX file with a .pdf extension.
    base_name = os.path.splitext(latex_file)[0]
    pdf_file = base_name + '.pdf'
    if os.path.exists(pdf_file):
        print(f"Generated PDF: {pdf_file}")
    else:
        print("PDF file was not found after compilation.")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python convert_to_pdf.py <latex_file.tex>")
        sys.exit(1)
    
    latex_file = sys.argv[1]
    if not os.path.exists(latex_file):
        print(f"Error: File '{latex_file}' does not exist.")
        sys.exit(1)
    
    compile_latex_to_pdf(latex_file)
