import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open("notebooks/08_kohort_final.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    src = "".join(cell['source'])
    print(f"\n{'='*20} Cell {i} ({cell['cell_type']}) {'='*20}")
    print(src)
    outputs = cell.get('outputs', [])
    if outputs:
        print("\n--- OUTPUTS ---")
        for o in outputs:
            if o.get('output_type') == 'stream':
                print("".join(o.get('text', [])))
            elif o.get('output_type') == 'execute_result':
                print(o.get('data', {}).get('text/plain', ''))
