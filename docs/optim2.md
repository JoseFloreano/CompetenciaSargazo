# 🧹 SCRIPT DE LIMPIEZA - Remover Triton y torch.compile

## ¿Qué hace?

```
Remueve:
  ❌ torch.compile (complejo, no necesario)
  ❌ Mixed precision bfloat16 (reduce VRAM)
  ❌ Triton (no lo necesitas instalar)

Mantiene:
  ✅ Toda la funcionalidad del modelo
  ✅ Batch size 256
  ✅ num_workers 8
  ✅ GPU en CUDA
  ✅ Entrenamiento normal
```

---

## 📥 Descargar script

El archivo está en: `clean_notebook.py`

---

## 🚀 OPCIÓN 1: Usar en Claude Code (Recomendado)

### Paso 1: Abre Claude Code

```bash
claude code
```

### Paso 2: En Claude Code, ejecuta:

```bash
python clean_notebook.py sargazo_classification.ipynb
```

**Reemplaza `sargazo_classification.ipynb` por tu ruta exacta si es diferente.**

### Paso 3: ¡Listo!

Se crea: `sargazo_classification_LIMPIO.ipynb`

---

## 🚀 OPCIÓN 2: Command Prompt (Si Claude Code no funciona)

### Paso 1: Navega a la carpeta donde está el script

```bash
cd C:\ruta\donde\esta\clean_notebook.py
```

### Paso 2: Ejecuta:

```bash
python clean_notebook.py sargazo_classification.ipynb
```

---

## 📋 Lo que hace exactamente

### 1. CELL 7: Remove torch.compile

**ANTES:**

```python
for _mode in ["reduce-overhead", "default"]:
    try:
        _candidate = torch.compile(_base_model, mode=_mode)
        _dummy = torch.randn(BATCH_SIZE, 3, 224, 224, device=device)
        with torch.no_grad():
            _ = _candidate(_dummy)
        del _dummy
        model = _candidate
        print(f"Modelo compilado con torch.compile (modo: {_mode})")
        break
    except Exception:
        continue
else:
    model = _base_model
    print("torch.compile no disponible...")
```

**DESPUÉS:**

```python
model = _base_model
print("torch.compile desactivado (no necesario)")
```

### 2. CELL 10-14: Remove mixed precision

**ANTES:**

```python
with autocast(device_type=device.type, dtype=torch.bfloat16):
    outputs = model(images)
    loss = criterion(outputs, labels)
```

**DESPUÉS:**

```python
outputs = model(images)
loss = criterion(outputs, labels)
```

### 3. General: Remove triton imports

```python
# ❌ ANTES
import triton

# ✅ DESPUÉS
# (línea removida)
```

---

## 📊 Resultado esperado

Después de correr el script:

```
======================================================================
✅ NOTEBOOK LIMPIADO
======================================================================

📁 Notebook original: sargazo_classification.ipynb
📁 Notebook limpio: sargazo_classification_LIMPIO.ipynb

📝 Cambios realizados:
  ✓ Cell 7: Removiendo torch.compile
  ✓ Cell 10: Removiendo mixed precision de train_one_epoch
  ✓ Cell 11: Removiendo mixed precision de evaluate
  ✓ Cell 14: Removiendo mixed precision de predicción

======================================================================
```

---

## ✅ Próximos pasos

1. **Espera a que termine el script** (toma 5-10 segundos)

2. **Abre el nuevo notebook:**
   - `sargazo_classification_LIMPIO.ipynb`

3. **Ejecuta CELL 7:**

   ```
   Debería mostrar:
   ✅ torch.compile desactivado (no necesario)
   ✅ Peak memory: 10-16 GB ← (en lugar de 4.96 GB)
   ✅ Modelo en GPU: True
   ```

4. **¡A entrenar!**
   - Ejecuta las celdas de entrenamiento normalmente

---

## 📊 Diferencias después

```
ANTES (con mixed precision + torch.compile fallando):
├─ Peak memory: 4.96 GB (bajo)
├─ torch.compile: ❌ No funciona
└─ Complejidad: Alta

DESPUÉS (sin mixed precision, sin torch.compile):
├─ Peak memory: 10-16 GB ✅ (MÁXIMO de tu GPU)
├─ torch.compile: ❌ No (no lo necesitas)
├─ Complejidad: Baja (código más simple)
└─ Funcionalidad: 100% igual
```

---

## 🎯 Si algo falla

### Si el script no encuentra el notebook:

```bash
# Ver archivos en la carpeta actual
dir

# Luego ejecuta con el nombre correcto:
python clean_notebook.py nombre_correcto.ipynb
```

### Si sale error de JSON:

```bash
# Asegúrate de que el notebook es válido
# Intenta abrir el notebook original en Jupyter
# Guarda y cierra
# Luego corre el script
```

---

## ✨ Resumen

```
Una línea:
  python clean_notebook.py sargazo_classification.ipynb

Resultado:
  ✅ Notebook limpio
  ✅ VRAM: 10-16 GB
  ✅ Listo para entrenar
```

**¡Hazlo ahora!** 🚀
