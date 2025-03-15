# Proyecto de Base de Datos Electoral

## Descripción
Este proyecto tiene como objetivo la normalización, importación y administración de datos electorales en México, incluyendo procesos de elecciones locales y federales. La estructura del sistema está diseñada para facilitar la consulta y manipulación de datos mediante una base de datos SQLite y una API REST desarrollada con FastAPI.

## Tecnologías Utilizadas
- **Python 3.9.21**
- **FastAPI [standard]** para el desarrollo de la API
- **SQLite** como sistema de base de datos

## Estructura del Proyecto

```
├── electoral_normalizada.sqlite3    # Base de datos SQLite normalizada
├── db.py                             # Módulo de conexión y operaciones con la base de datos
├── models.py                         # Definición de modelos de datos
├── import_data_estados.py            # Script de importación de datos de estados
├── import_data_partidos.py           # Script de importación de datos de partidos políticos
├── import_data_congresos.py          # Script de importación de datos de congresos locales
├── import_data_diputaciones.py       # Script de importación de datos de diputaciones federales
├── import_data_senado.py             # Script de importación de datos de senadores
├── import_data_financiamiento_candidaturas.py  # Importación de financiamiento de candidaturas
├── import_data_financiamiento_partidos.py      # Importación de financiamiento de partidos políticos
├── app/                              # Carpeta principal de la API
│   ├── main.py                       # Archivo principal de la aplicación FastAPI
│   ├── routers/                      # Carpeta de enrutadores para modularizar la API
│   │   ├── __init__.py               # Inicialización del módulo de enrutadores
│   │   ├── congresos.py              # Endpoints para congresos
│   │   ├── estados.py                # Endpoints para estados
│   │   ├── partidos.py               # Endpoints para partidos políticos
│   │   ├── diputaciones.py           # Endpoints para diputaciones
│   │   ├── senado.py                 # Endpoints para senadores
│   │   ├── financiamiento_candidaturas.py  # Endpoints para financiamiento de candidaturas
│   │   ├── financiamiento_partidos.py      # Endpoints para financiamiento de partidos
```

## Instalación y Configuración

1. Clonar el repositorio:
   ```sh
   git clone https://github.com/usuario/proyecto-electoral.git
   cd proyecto-electoral
   ```

2. Crear un entorno virtual y activarlo:
   ```sh
   python3 -m venv venv
   source venv/bin/activate  # En Linux/macOS
   venv\Scripts\activate     # En Windows
   ```

3. Instalar las dependencias del proyecto:
   ```sh
   pip install "fastapi[standard]" sqlite3
   ```

4. Ejecutar la aplicación FastAPI:
   ```sh
   uvicorn app.main:app --reload
   ```

5. La API estará disponible en `http://127.0.0.1:8000` y la documentación en `http://127.0.0.1:8000/docs`.

## Uso

- Los scripts de importación (`import_data_*.py`) deben ejecutarse en orden para poblar la base de datos con la información electoral.
- La API permite realizar consultas estructuradas sobre los datos normalizados, accediendo a información de estados, congresos, partidos, financiamiento y cargos de elección popular.

## Contribuciones
Se agradecen contribuciones al proyecto. Para ello, abre un issue o envía un pull request en GitHub.

## Licencia
Este proyecto está bajo la Licencia MIT. Para más información, consulta el archivo `LICENSE`.

## Contacto

Para cualquier consulta o colaboración, puedes contactar al autor en: **eduardobareapoot@outlook.com**.

