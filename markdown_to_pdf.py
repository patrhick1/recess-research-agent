import markdown
from xhtml2pdf import pisa

def convert_markdown_to_pdf(markdown_file_path: str, pdf_file_path: str) -> None:
    """
    Converts a Markdown file to a PDF file using Python packages with support for tables.
    
    Args:
        markdown_file_path (str): The path to the Markdown file.
        pdf_file_path (str): The desired output path for the PDF.
    """
    # Read the Markdown file
    with open(markdown_file_path, "r", encoding="utf-8") as md_file:
        md_text = md_file.read()

    # Convert Markdown text to HTML using the 'tables' extension for proper table support
    html_content = markdown.markdown(md_text, extensions=['tables'])

    # Wrap the HTML content in a basic HTML document structure with CSS for table styling
    html = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <title>Converted Document</title>
        <style>
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            table, th, td {{
                border: 1px solid #000;
            }}
            th, td {{
                padding: 8px;
                text-align: left;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    # Convert the HTML to PDF using xhtml2pdf
    with open(pdf_file_path, "wb") as pdf_file:
        pisa_status = pisa.CreatePDF(html, dest=pdf_file)
    
    if pisa_status.err:
        print("An error occurred during the PDF conversion.")
    else:
        print(f"Conversion successful! PDF saved at: {pdf_file_path}")

# Example usage:
if __name__ == "__main__":
    md_path = "ai_advancements_in_business.md"   # Path to your markdown file
    pdf_path = "ai_advancements_in_business.pdf" # Desired PDF file output path
    convert_markdown_to_pdf(md_path, pdf_path)
