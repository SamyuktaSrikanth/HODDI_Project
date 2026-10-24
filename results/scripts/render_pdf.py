import os, subprocess
from markdown_it import MarkdownIt

md_path = r'c:\Users\DELL\Desktop\mini project\HODDI_Project\results\FINAL_CONSOLIDATED_RESULTS.md'
html_path = r'c:\Users\DELL\Desktop\mini project\HODDI_Project\results\results_temp.html'
pdf_path = r'c:\Users\DELL\Desktop\mini project\HODDI_Project\results\results.pdf'

with open(md_path, 'r', encoding='utf-8') as f:
    text = f.read()

md = MarkdownIt().enable('table')
html_body = md.render(text)

styled_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  @page {{
    size: letter;
    margin: 14mm 16mm 14mm 16mm;
  }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 8.5pt;
    line-height: 1.35;
    color: #24292e;
  }}
  h1 {{
    font-size: 15pt;
    border-bottom: 2px solid #0969da;
    padding-bottom: 4px;
    margin-top: 0;
    margin-bottom: 8px;
    color: #1f2328;
  }}
  h2 {{
    font-size: 11.5pt;
    border-bottom: 1px solid #d0d7de;
    padding-bottom: 3px;
    margin-top: 14px;
    margin-bottom: 6px;
    color: #0969da;
    page-break-after: avoid;
  }}
  h3 {{
    font-size: 9.5pt;
    margin-top: 10px;
    margin-bottom: 4px;
    color: #1f2328;
    page-break-after: avoid;
  }}
  table {{
    border-collapse: collapse;
    width: 100%;
    margin: 8px 0;
    font-size: 8pt;
    page-break-inside: avoid;
  }}
  th, td {{
    border: 1px solid #d0d7de;
    padding: 4px 6px;
    text-align: left;
  }}
  th {{
    background-color: #f6f8fa;
    font-weight: 600;
  }}
  tr:nth-child(even) {{
    background-color: #fcfcfc;
  }}
  code {{
    background-color: #f6f8fa;
    padding: 0.15em 0.3em;
    border-radius: 3px;
    font-size: 7.5pt;
    font-family: ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace;
  }}
  pre code {{
    padding: 0;
    background: transparent;
  }}
  pre {{
    background-color: #f6f8fa;
    padding: 8px;
    border-radius: 4px;
    overflow-x: auto;
    font-size: 7.5pt;
    page-break-inside: avoid;
  }}
  blockquote {{
    margin: 6px 0;
    padding: 4px 10px;
    color: #57606a;
    border-left: 3px solid #0969da;
    background-color: #f6f8fa;
  }}
  ul, ol {{
    padding-left: 18px;
    margin: 4px 0;
  }}
  li {{
    margin-bottom: 2px;
  }}
  hr {{
    height: 1px;
    background-color: #d0d7de;
    border: none;
    margin: 12px 0;
  }}
  strong {{
    font-weight: 600;
  }}
</style>
</head>
<body>
{html_body}
</body>
</html>"""

with open(html_path, 'w', encoding='utf-8') as f:
    f.write(styled_html)

chrome = r'C:\Program Files\Google\Chrome\Application\chrome.exe'
cmd = [chrome, '--headless', '--disable-gpu', '--no-pdf-header-footer', f'--print-to-pdf={pdf_path}', html_path]
res = subprocess.run(cmd, capture_output=True, text=True)
print('Chrome return code:', res.returncode)
if os.path.exists(html_path):
    os.remove(html_path)
print('New PDF size:', os.path.getsize(pdf_path))
