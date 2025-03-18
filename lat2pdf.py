'''#!/usr/bin/env python3

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

'''


#!/usr/bin/env python3

import subprocess
import os
import sys
import shutil

def try_compile(file_path):
    print(f"Trying to compile using file path: {file_path}")
    cmd = ['pdflatex', '-interaction=nonstopmode', file_path]
    process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return process

def compile_latex_to_pdf(latex_file):
    # Check if pdflatex is installed
    if shutil.which('pdflatex') is None:
        print("Error: pdflatex is not installed. Please install a LaTeX distribution (e.g., TeX Live, MiKTeX).")
        sys.exit(1)
    
    # Fallback 1: Replace backslashes with forward slashes
    method1 = latex_file.replace('\\', '/')
    process = try_compile(method1)
    if process.returncode == 0:
        print("PDF compilation succeeded using fallback 1.")
        base_name = os.path.splitext(method1)[0]
        pdf_file = base_name + '.pdf'
        if os.path.exists(pdf_file):
            print(f"Generated PDF: {pdf_file}")
            return
        else:
            print("PDF file was not found after compilation with fallback 1.")
    else:
        print("Fallback 1 failed. Output:")
        print(process.stdout)
        print(process.stderr)
    
    # Fallback 2: Use an absolute path with forward slashes
    method2 = os.path.abspath(latex_file).replace('\\', '/')
    process = try_compile(method2)
    if process.returncode == 0:
        print("PDF compilation succeeded using fallback 2.")
        base_name = os.path.splitext(method2)[0]
        pdf_file = base_name + '.pdf'
        if os.path.exists(pdf_file):
            print(f"Generated PDF: {pdf_file}")
            return
        else:
            print("PDF file was not found after compilation with fallback 2.")
    else:
        print("Fallback 2 failed. Output:")
        print(process.stdout)
        print(process.stderr)
    
    # Fallback 3: Change the working directory to the file's directory
    abs_path = os.path.abspath(latex_file)
    directory, filename = os.path.split(abs_path)
    print(f"Trying fallback 3: Changing working directory to {directory} and using filename {filename}")
    old_dir = os.getcwd()
    try:
        os.chdir(directory)
        process = try_compile(filename)
        if process.returncode == 0:
            print("PDF compilation succeeded using fallback 3.")
            pdf_file = os.path.splitext(filename)[0] + '.pdf'
            if os.path.exists(pdf_file):
                print(f"Generated PDF: {os.path.join(directory, pdf_file)}")
                return
            else:
                print("PDF file was not found after compilation with fallback 3.")
        else:
            print("Fallback 3 failed. Output:")
            print(process.stdout)
            print(process.stderr)
    finally:
        os.chdir(old_dir)
    
    print("All fallback methods failed. Exiting.")
    sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python convert_to_pdf.py <latex_file.tex>")
        sys.exit(1)
    
    latex_file = sys.argv[1]
    if not os.path.exists(latex_file):
        print(f"Error: File '{latex_file}' does not exist.")
        sys.exit(1)
    
    compile_latex_to_pdf(latex_file)
