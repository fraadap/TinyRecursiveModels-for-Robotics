#!/usr/bin/env python3
"""
Script per convertire un file Python con marker speciali in un Jupyter Notebook.

Questo script legge un file .py con commenti speciali che delimitano le celle
(# %% per code cells, # %% [markdown] per markdown cells) e crea un notebook .ipynb.

Usage:
    python convert_to_notebook.py input.py output.ipynb
"""

import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple


def parse_py_file(py_content: str) -> List[Dict]:
    """
    Parse un file Python e estrae le celle
    
    Args:
        py_content: contenuto del file Python
    
    Returns:
        Lista di dizionari rappresentanti celle Jupyter
    """
    cells = []
    
    # Split per marker delle celle
    # Pattern: # %% [markdown] o # %%
    cell_pattern = r'^# %%(?:\s*\[markdown\])?'
    
    lines = py_content.split('\n')
    current_cell = None
    current_cell_type = 'code'
    current_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check se è un marker di cella
        match = re.match(cell_pattern, line)
        
        if match:
            # Salva cella precedente se esiste
            if current_lines:
                cells.append({
                    'cell_type': current_cell_type,
                    'source': current_lines
                })
            
            # Determina tipo di nuova cella
            if '[markdown]' in line:
                current_cell_type = 'markdown'
            else:
                current_cell_type = 'code'
            
            current_lines = []
        else:
            # Aggiungi linea alla cella corrente
            # Per markdown, rimuovi # iniziale
            if current_cell_type == 'markdown':
                # Rimuovi # iniziale e spazio
                if line.startswith('# '):
                    line = line[2:]
                elif line.startswith('#'):
                    line = line[1:]
            
            current_lines.append(line)
        
        i += 1
    
    # Aggiungi ultima cella
    if current_lines:
        cells.append({
            'cell_type': current_cell_type,
            'source': current_lines
        })
    
    return cells


def create_notebook_dict(cells: List[Dict]) -> Dict:
    """
    Crea il dizionario che rappresenta un notebook Jupyter
    
    Args:
        cells: lista di celle parsed
    
    Returns:
        Dizionario in formato notebook Jupyter
    """
    nb_cells = []
    
    for cell in cells:
        cell_type = cell['cell_type']
        source = cell['source']
        
        # Rimuovi linee vuote iniziali/finali
        while source and not source[0].strip():
            source.pop(0)
        while source and not source[-1].strip():
            source.pop()
        
        # Aggiungi newline alla fine di ogni linea (tranne l'ultima)
        source_with_newlines = [line + '\n' for line in source[:-1]]
        if source:
            source_with_newlines.append(source[-1])
        
        if cell_type == 'markdown':
            nb_cell = {
                'cell_type': 'markdown',
                'metadata': {},
                'source': source_with_newlines
            }
        else:  # code
            nb_cell = {
                'cell_type': 'code',
                'execution_count': None,
                'metadata': {},
                'outputs': [],
                'source': source_with_newlines
            }
        
        nb_cells.append(nb_cell)
    
    # Crea struttura notebook
    notebook = {
        'cells': nb_cells,
        'metadata': {
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3'
            },
            'language_info': {
                'codemirror_mode': {
                    'name': 'ipython',
                    'version': 3
                },
                'file_extension': '.py',
                'mimetype': 'text/x-python',
                'name': 'python',
                'nbconvert_exporter': 'python',
                'pygments_lexer': 'ipython3',
                'version': '3.8.0'
            }
        },
        'nbformat': 4,
        'nbformat_minor': 4
    }
    
    return notebook


def convert_py_to_notebook(input_path: str, output_path: str):
    """
    Converte un file Python in un notebook Jupyter
    
    Args:
        input_path: path al file .py di input
        output_path: path al file .ipynb di output
    """
    print(f"Converting {input_path} to {output_path}...")
    
    # Leggi file Python
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"File not found: {input_path}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        py_content = f.read()
    
    # Parse celle
    print("Parsing cells...")
    cells = parse_py_file(py_content)
    print(f"Found {len(cells)} cells")
    
    # Conta tipi di celle
    markdown_cells = sum(1 for c in cells if c['cell_type'] == 'markdown')
    code_cells = sum(1 for c in cells if c['cell_type'] == 'code')
    print(f"  - {markdown_cells} markdown cells")
    print(f"  - {code_cells} code cells")
    
    # Crea struttura notebook
    print("Creating notebook structure...")
    notebook = create_notebook_dict(cells)
    
    # Salva notebook
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Notebook created successfully: {output_path}")
    print(f"  Size: {output_file.stat().st_size} bytes")


def main():
    """Main function"""    
    input_path = "TinyRecursiveModels-for-Robotics\\notebook.py"
    output_path = (Path(input_path).with_suffix('.ipynb'))
    try:
        convert_py_to_notebook(input_path, output_path)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
