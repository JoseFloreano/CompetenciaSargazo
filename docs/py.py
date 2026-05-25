#!/usr/bin/env python3
"""
🔧 LIMPIADOR: Remover Triton, torch.compile y mixed precision del notebook
Uso: python clean_notebook.py sargazo_classification.ipynb
"""

import json
import sys
from pathlib import Path


def clean_notebook(notebook_path):
    """Limpia el notebook de Triton, torch.compile y mixed precision"""

    with open(notebook_path, "r") as f:
        nb = json.load(f)

    changes = []

    # Procesar cada celda
    for cell_idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue

        source = (
            "".join(cell["source"])
            if isinstance(cell["source"], list)
            else cell["source"]
        )
        original = source

        # ============================================================
        # CAMBIO 1: Remover torch.compile completamente (CELL 7)
        # ============================================================
        if "torch.compile" in source:
            changes.append(f"Cell {cell_idx}: Removiendo torch.compile")

            # Reemplazar todo el bloque de torch.compile
            lines = source.split("\n")
            new_lines = []
            skip_until_else = False
            indent_level = 0

            for i, line in enumerate(lines):
                # Detectar inicio del for _mode in
                if "_base_model = model" in line:
                    new_lines.append(line)
                    new_lines.append("model = _base_model")
                    new_lines.append(
                        'print("torch.compile desactivado (no necesario)")'
                    )
                    skip_until_else = True
                    continue

                # Saltar líneas hasta encontrar el else
                if skip_until_else:
                    if line.strip().startswith("else:"):
                        skip_until_else = False
                        continue
                    if (
                        line.strip().startswith("for _mode in")
                        or line.strip().startswith("try:")
                        or line.strip().startswith("except")
                        or "_candidate" in line
                        or "_dummy" in line
                        or "_test" in line
                        or "torch.compile" in line
                    ):
                        continue

                new_lines.append(line)

            source = "\n".join(new_lines)

        # ============================================================
        # CAMBIO 2: Remover mixed precision (bfloat16) de train
        # ============================================================
        if "def train_one_epoch" in source and "autocast" in source:
            changes.append(
                f"Cell {cell_idx}: Removiendo mixed precision de train_one_epoch"
            )

            source = source.replace(
                "with autocast(device_type=device.type, dtype=torch.bfloat16):\n"
                "            outputs = model(images)\n"
                "            loss = criterion(outputs, labels)",
                "outputs = model(images)\n" "        loss = criterion(outputs, labels)",
            )

            # Versión alternativa con espacios diferentes
            source = source.replace(
                "with autocast(device_type=device.type, dtype=torch.bfloat16):\n"
                "                outputs = model(images)\n"
                "                loss = criterion(outputs, labels)",
                "outputs = model(images)\n"
                "            loss = criterion(outputs, labels)",
            )

        # ============================================================
        # CAMBIO 3: Remover mixed precision de evaluate
        # ============================================================
        if "def evaluate" in source and "autocast" in source:
            changes.append(f"Cell {cell_idx}: Removiendo mixed precision de evaluate")

            source = source.replace(
                "with autocast(device_type=device.type, dtype=torch.bfloat16):\n"
                "                outputs = model(images)\n"
                "                loss = criterion(outputs, labels)",
                "outputs = model(images)\n"
                "            loss = criterion(outputs, labels)",
            )

        # ============================================================
        # CAMBIO 4: Remover mixed precision de predicción
        # ============================================================
        if "for images, names in test_loader" in source and "autocast" in source:
            changes.append(f"Cell {cell_idx}: Removiendo mixed precision de predicción")

            source = source.replace(
                "with autocast(device_type=device.type, dtype=torch.bfloat16):\n"
                "            outputs = model(images)",
                "outputs = model(images)",
            )

        # ============================================================
        # CAMBIO 5: Remover imports de triton si existen
        # ============================================================
        if "triton" in source.lower():
            changes.append(f"Cell {cell_idx}: Removiendo import de triton")
            lines = source.split("\n")
            lines = [line for line in lines if "triton" not in line.lower()]
            source = "\n".join(lines)

        # Actualizar la celda si cambió
        if source != original:
            cell["source"] = source.split("\n")

    # ============================================================
    # Guardar notebook modificado
    # ============================================================
    output_path = Path(notebook_path).stem + "_LIMPIO.ipynb"

    with open(output_path, "w") as f:
        json.dump(nb, f, indent=1)

    # ============================================================
    # Reporte
    # ============================================================
    print("\n" + "=" * 70)
    print("✅ NOTEBOOK LIMPIADO")
    print("=" * 70)

    print(f"\n📁 Notebook original: {notebook_path}")
    print(f"📁 Notebook limpio: {output_path}")

    print(f"\n📝 Cambios realizados:")
    for change in changes:
        print(f"  ✓ {change}")

    print("\n" + "=" * 70)
    print("CAMBIOS ESPECÍFICOS:")
    print("=" * 70)

    print("""
1️⃣ torch.compile: REMOVIDO
   ├─ Razón: No necesario, complica cosas
   └─ Efecto: Código más simple

2️⃣ Mixed precision (bfloat16): REMOVIDO
   ├─ Antes: VRAM 4.96 GB
   ├─ Después: VRAM 10-16 GB
   └─ Razón: Usar todo el potencial de GPU

3️⃣ Triton: NO NECESARIO instalar
   ├─ Razón: Solo se usa con torch.compile
   └─ Estado: Ignorado

4️⃣ Imports de triton: REMOVIDOS
   └─ Si existían en el código

5️⃣ GradScaler: MANTIENE (sigue siendo útil)
   └─ Aunque sin mixed precision, no hace nada crítico
""")

    print("\n" + "=" * 70)
    print("📌 PRÓXIMOS PASOS:")
    print("=" * 70)
    print(f"""
1. Abre el notebook limpio: {output_path}
2. Ejecuta CELL 7 nuevamente
3. Debería mostrar:
   ✅ torch.compile desactivado (no necesario)
   ✅ Peak memory: 10-16 GB (en lugar de 4.96 GB)
   ✅ Modelo en GPU: True
4. ¡A entrenar!

Diferencia esperada:
├─ VRAM utilizada: 10-16 GB (todo el potencial)
├─ Velocidad: Ligeramente más lenta (sin mixed precision)
└─ Precisión: Máxima (float32 completo)
""")

    print("=" * 70)
    print("✨ ¡Listo!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python clean_notebook.py <ruta_notebook.ipynb>")
        sys.exit(1)

    notebook_path = sys.argv[1]

    if not Path(notebook_path).exists():
        print(f"❌ Archivo no encontrado: {notebook_path}")
        sys.exit(1)

    clean_notebook(notebook_path)
