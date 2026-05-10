# Grupo Alfa - Backend Extractor

Backend Flask para procesar catálogos PDF.

## Requisitos del sistema

```bash
# macOS
brew install poppler

# Ubuntu / Debian
sudo apt-get install -y poppler-utils
```

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> La primera ejecución de PaddleOCR descarga ~100 MB de modelos.

## Ejecutar

```bash
export ALLOWED_ORIGINS="https://TU-DOMINIO-LOVABLE.lovable.app,http://localhost:8080"
python app.py
# -> http://localhost:5000
```

## Endpoints

| Método | Ruta                       | Descripción                              |
|--------|----------------------------|------------------------------------------|
| POST   | `/process-pdf`             | Sube un PDF (`multipart/form-data: file`)|
| GET    | `/progress/<job_id>`       | SSE con progreso en tiempo real          |
| GET    | `/result/<job_id>`         | JSON final con piezas y URLs             |
| GET    | `/download/zip/<job_id>`   | ZIP de imágenes recortadas               |
| GET    | `/download/csv/<job_id>`   | CSV con metadata                         |
| GET    | `/download/xlsx/<job_id>`  | Excel con metadata                       |
| GET    | `/image/<job_id>/<file>`   | Imagen individual                        |

## Despliegue rápido

- **Render / Railway / Fly.io**: subir como Web Service Python, comando:
  `gunicorn -w 2 -k gthread --threads 4 -b 0.0.0.0:$PORT app:app`
- Asegúrate de instalar `poppler-utils` en el contenedor (Docker o buildpack).
