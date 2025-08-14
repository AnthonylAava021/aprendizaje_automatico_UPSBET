# UPSBet - Predicción de Partidos con Machine Learning

Sistema de predicción de partidos de fútbol ecuatoriano utilizando machine learning y PostgreSQL.

## 🚀 Características

- **Predicción de resultados**: Gana local, empate, gana visita
- **Predicción de córners**: Total de córners por equipo
- **Predicción de tarjetas**: Tarjetas amarillas y rojas
- **Base de datos PostgreSQL**: Almacenamiento persistente
- **API REST**: Endpoints para predicciones y estadísticas
- **Interfaz web moderna**: Diseño responsive y atractivo

## 📋 Requisitos

- Python 3.8+
- PostgreSQL 12+
- pip (gestor de paquetes de Python)

## 🛠️ Instalación

### 1. Clonar el repositorio
```bash
git clone <url-del-repositorio>
cd aprendizaje_automatico_UPSBET
```

### 2. Crear entorno virtual
```bash
python -m venv venv
# En Windows:
venv\Scripts\activate
# En macOS/Linux:
source venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Configurar base de datos PostgreSQL

#### Crear la base de datos:
```sql
CREATE DATABASE "UPS_BET"
    WITH
    OWNER = postgres
    ENCODING = 'UTF8'
    LOCALE_PROVIDER = 'libc'
    CONNECTION LIMIT = -1
    IS_TEMPLATE = False;
```

#### Configurar credenciales:
Las credenciales están configuradas en `config.py`:
- Usuario: `postgres`
- Contraseña: `anthony`
- Base de datos: `UPS_BET`
- Host: `localhost`
- Puerto: `5432`

### 5. Inicializar la base de datos
```bash
python database_init.py
```

## 🚀 Ejecutar la aplicación

### Opción 1: Usando run.py
```bash
python run.py
```

### Opción 2: Usando app.py directamente
```bash
python app.py
```

### Opción 3: Usando Flask CLI
```bash
export FLASK_APP=app.py
export FLASK_ENV=development
flask run
```

La aplicación estará disponible en: http://localhost:5000

## 📊 API Endpoints

### Predicciones
- `POST /api/predict` - Realizar predicción de partido
- `GET /api/predicciones` - Obtener historial de predicciones

### Equipos
- `GET /api/equipos` - Listar todos los equipos

### Partidos
- `GET /api/partidos` - Obtener historial de partidos
- `POST /api/partidos` - Crear nuevo partido

### Estadísticas
- `GET /api/stats` - Estadísticas generales del sistema

## 🔧 Estructura del proyecto

```
aprendizaje_automatico_UPSBET/
├── app/
│   ├── static/          # Archivos estáticos (CSS, JS, imágenes)
│   ├── templates/       # Plantillas HTML
│   ├── models/          # Modelos de ML entrenados
│   └── data/           # Datos de entrenamiento
├── app.py              # Aplicación principal Flask
├── models.py           # Modelos de base de datos
├── ml_models.py        # Lógica de machine learning
├── config.py           # Configuración de la aplicación
├── database_init.py    # Script de inicialización de BD
├── requirements.txt    # Dependencias de Python
└── run.py             # Script de ejecución
```

## 🤖 Modelos de Machine Learning

El sistema utiliza:
- **RandomForest** para predicción de resultados
- **StandardScaler** para normalización de datos
- **Modelos pre-entrenados** en `app/models/`

## 📝 Ejemplo de uso de la API

### Realizar predicción:
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "home_name": "Emelec",
    "away_name": "Barcelona SC",
    "home_code": 4,
    "away_code": 0
  }'
```

### Respuesta:
```json
{
  "home_win": 0.45,
  "draw": 0.28,
  "away_win": 0.27,
  "score": {
    "home": 2,
    "away": 1
  },
  "corners": {
    "home": 6,
    "away": 4
  },
  "cards": {
    "home": 2,
    "away": 3
  }
}
```

## 🔒 Seguridad

- Las credenciales de base de datos están en `config.py`
- En producción, usar variables de entorno
- CORS habilitado para desarrollo

## 📈 Próximas mejoras

- [ ] Autenticación de usuarios
- [ ] Dashboard de estadísticas
- [ ] Más modelos de ML
- [ ] API para resultados reales
- [ ] Notificaciones en tiempo real

## 🤝 Contribuir

1. Fork el proyecto
2. Crear una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abrir un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT.

---

**Desarrollado con ❤️ para UPS**
