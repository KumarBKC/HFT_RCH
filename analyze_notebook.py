import json

with open('Algorithmic_Trading_Machine_Learning_Quant_Strategies.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

with open('notebook_analysis.txt', 'w', encoding='utf-8') as out:
    out.write(f"Total cells: {len(nb['cells'])}\n\n")
    out.write("="*80 + "\n")

    for i, cell in enumerate(nb['cells']):
        cell_type = cell.get('cell_type', '')
        source = ''.join(cell.get('source', []))
        
        if cell_type == 'markdown':
            out.write(f"\n[SECTION {i}] MARKDOWN:\n")
            out.write(source[:500])
            out.write("\n...\n")
        elif cell_type == 'code':
            lines = source.split('\n')[:8]
            preview = '\n'.join(lines)
            out.write(f"\n[Cell {i}] CODE:\n")
            out.write(preview[:400])
            out.write("\n---\n")

print("Done! Check notebook_analysis.txt")
