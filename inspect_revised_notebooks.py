import json
import glob
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

notebooks = sorted(glob.glob("notebooks/*.ipynb"))

for nb_file in notebooks:
    print(f"\n{'='*70}")
    print(f"NOTEBOOK: {os.path.basename(nb_file)}")
    print(f"{'='*70}")
    
    with open(nb_file, 'r', encoding='utf-8') as f:
        nb_json = json.load(f)
        
    cells = nb_json.get('cells', [])
    print(f"Total cells: {len(cells)}")
    
    md_headers = []
    code_summary = []
    
    for i, c in enumerate(cells):
        src = "".join(c.get('source', []))
        if c.get('cell_type') == 'markdown':
            lines = [l.strip() for l in src.splitlines() if l.strip().startswith('#')]
            if lines:
                md_headers.extend(lines)
        elif c.get('cell_type') == 'code':
            exec_count = c.get('execution_count', None)
            outputs = c.get('outputs', [])
            has_error = any(o.get('output_type') == 'error' for o in outputs)
            out_sample = ""
            for o in outputs:
                if o.get('output_type') == 'stream':
                    out_sample = "".join(o.get('text', []))[:120].replace('\n', ' ')
                    break
                elif o.get('output_type') == 'execute_result':
                    out_sample = str(o.get('data', {}).get('text/plain', ''))[:120].replace('\n', ' ')
                    break
            code_summary.append((i, exec_count, len(src.splitlines()), has_error, out_sample))
            
    print("Markdown Structure:")
    for h in md_headers[:10]:
        print(f"  {h}")
    if len(md_headers) > 10:
        print(f"  ... (+ {len(md_headers)-10} more headers)")
        
    print(f"\nCode Cells ({len(code_summary)} total):")
    for idx, ec, n_lines, err, out_txt in code_summary[:6]:
        err_str = " [ERROR]" if err else ""
        out_prev = f" | Output: {out_txt[:70]}..." if out_txt else ""
        print(f"  Cell {idx}: [Exec {ec}] {n_lines} lines{err_str}{out_prev}")
    if len(code_summary) > 6:
        print(f"  ... (+ {len(code_summary)-6} more code cells)")
