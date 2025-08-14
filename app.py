from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from models import db, Equipo, Partido, Prediccion
from ml_models import predictor
from config import config
import os
import psycopg2
import numpy as np

def create_app(config_name='default'):
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    
    # Inicializar extensiones
    db.init_app(app)
    CORS(app)
    
    # Configurar carpeta de archivos estáticos
    app.static_folder = 'app/static'
    app.template_folder = 'app/templates'
    
    # Configurar URL para archivos estáticos
    app.static_url_path = '/static'
    
    @app.route('/')
    def index():
        """Página principal"""
        return render_template('index.html')
    
    @app.route('/api/equipos')
    def get_equipos():
        """Obtener lista de equipos"""
        try:
            equipos = Equipo.query.all()
            return jsonify([equipo.to_dict() for equipo in equipos])
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/predict', methods=['POST'])
    def predict_match():
        """Predice el resultado de un partido"""
        try:
            data = request.get_json()
            
            if not data:
                return jsonify({'error': 'No se recibieron datos'}), 400
            
            home_name = data.get('home_name')
            away_name = data.get('away_name')
            home_code = data.get('home_code')
            away_code = data.get('away_code')
            
            if not all([home_name, away_name, home_code is not None, away_code is not None]):
                return jsonify({'error': 'Faltan datos requeridos'}), 400
            
            # Obtener equipos de la base de datos
            home_team = Equipo.query.filter_by(nombre=home_name).first()
            away_team = Equipo.query.filter_by(nombre=away_name).first()
            
            if not home_team or not away_team:
                return jsonify({'error': 'Equipos no encontrados'}), 404
            
            # Obtener datos de corners_tabla para el modelo específico
            corners_data = get_corners_data(home_team.id, away_team.id)
            
            # Obtener datos de ganador_resultado_tabla para el modelo de marcador
            ganador_data = get_ganador_resultado_data(home_team.id, away_team.id)
            
            # Calcular corners totales usando el modelo pre-entrenado
            corners_total = calculate_corners_total(corners_data, home_code, away_code)
            
            # Calcular marcador usando el modelo pre-entrenado
            score_prediction = calculate_score_prediction(ganador_data, home_code, away_code)
            
            # Realizar predicción básica (probabilidades de resultado)
            prediction_result = predictor.predict_match(home_code, away_code)
            
            # Reemplazar el marcador mock con el real
            prediction_result['score'] = score_prediction
            
            # Agregar corners_total al resultado
            prediction_result['corners_total'] = corners_total
            
            # DEBUG: Imprimir información para verificar que usa los modelos reales
            print(f"🔍 DEBUG - Equipos: {home_name} ({home_code}) vs {away_name} ({away_code})")
            print(f"🔍 DEBUG - Datos de corners_tabla obtenidos: {len(corners_data)} campos")
            print(f"🔍 DEBUG - Datos de ganador_resultado_tabla obtenidos: {len(ganador_data)} campos")
            print(f"🔍 DEBUG - Corners totales calculados por tu modelo: {corners_total}")
            print(f"🔍 DEBUG - Marcador calculado por tu modelo: {score_prediction['home']}-{score_prediction['away']}")
            print(f"🔍 DEBUG - Modelo de corners usado: {type(predictor.corners_model).__name__}")
            print(f"🔍 DEBUG - Modelo de marcador usado: {type(predictor.score_model).__name__}")
            print(f"🔍 DEBUG - Escalador usado: {type(predictor.corners_scaler).__name__}")
            print("=" * 50)
            
            # Guardar predicción en la base de datos
            prediccion = Prediccion(
                equipo_local_id=home_team.id,
                equipo_visita_id=away_team.id,
                prob_local_win=prediction_result['home_win'],
                prob_draw=prediction_result['draw'],
                prob_visita_win=prediction_result['away_win'],
                goles_pred_local=score_prediction['home'],
                goles_pred_visita=score_prediction['away'],
                corners_pred_local=prediction_result['corners']['home'],
                corners_pred_visita=prediction_result['corners']['away'],
                tarjetas_pred_local=prediction_result['yellow_cards']['home'] + prediction_result['red_cards']['home'],
                tarjetas_pred_visita=prediction_result['yellow_cards']['away'] + prediction_result['red_cards']['away'],
                modelo_usado='XGBoost + MultiOutputRegressor'
            )
            
            db.session.add(prediccion)
            db.session.commit()
            
            return jsonify(prediction_result)
            
        except Exception as e:
            db.session.rollback()
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/partidos', methods=['GET'])
    def get_partidos():
        """Obtener lista de partidos"""
        try:
            partidos = Partido.query.order_by(Partido.fecha.desc()).limit(50).all()
            return jsonify([partido.to_dict() for partido in partidos])
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/partidos', methods=['POST'])
    def create_partido():
        """Crear un nuevo partido"""
        try:
            data = request.get_json()
            
            if not data:
                return jsonify({'error': 'No se recibieron datos'}), 400
            
            # Validar datos requeridos
            required_fields = ['equipo_local_id', 'equipo_visita_id', 'fecha']
            for field in required_fields:
                if field not in data:
                    return jsonify({'error': f'Campo requerido: {field}'}), 400
            
            # Crear partido
            partido = Partido(**data)
            db.session.add(partido)
            db.session.commit()
            
            return jsonify(partido.to_dict()), 201
            
        except Exception as e:
            db.session.rollback()
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/predicciones', methods=['GET'])
    def get_predicciones():
        """Obtener lista de predicciones"""
        try:
            predicciones = Prediccion.query.order_by(Prediccion.created_at.desc()).limit(50).all()
            return jsonify([prediccion.to_dict() for prediccion in predicciones])
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/stats')
    def get_stats():
        """Obtener estadísticas generales"""
        try:
            total_equipos = Equipo.query.count()
            total_partidos = Partido.query.count()
            total_predicciones = Prediccion.query.count()
            
            # Estadísticas de resultados
            partidos_con_resultado = Partido.query.filter(Partido.resultado.isnot(None)).count()
            if partidos_con_resultado > 0:
                victorias_local = Partido.query.filter_by(resultado='L').count()
                empates = Partido.query.filter_by(resultado='E').count()
                victorias_visita = Partido.query.filter_by(resultado='V').count()
            else:
                victorias_local = empates = victorias_visita = 0
            
            return jsonify({
                'total_equipos': total_equipos,
                'total_partidos': total_partidos,
                'total_predicciones': total_predicciones,
                'partidos_con_resultado': partidos_con_resultado,
                'victorias_local': victorias_local,
                'empates': empates,
                'victorias_visita': victorias_visita
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Recurso no encontrado'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        db.session.rollback()
        return jsonify({'error': 'Error interno del servidor'}), 500
    
    return app

def get_corners_data(equipo_local_id, equipo_visitante_id):
    """Obtiene el último registro de corners_tabla para los equipos especificados"""
    try:
        # Conectar a la base de datos PostgreSQL
        conn = psycopg2.connect(
            host="localhost",
            database="UPS_BET",
            user="user",
            password="upsbet05",
            port="5432"
        )
        
        cursor = conn.cursor()
        
        # Buscar el último registro por fecha para estos equipos
        query = """
        SELECT * FROM corners_tabla 
        WHERE equipo_local_id = %s AND equipo_visitante_id = %s
        ORDER BY fecha DESC 
        LIMIT 1
        """
        
        cursor.execute(query, (equipo_local_id, equipo_visitante_id))
        result = cursor.fetchone()
        
        if result:
            # Obtener nombres de columnas
            columns = [desc[0] for desc in cursor.description]
            corners_data = dict(zip(columns, result))
        else:
            # Si no hay datos, buscar con equipos invertidos
            query_inverted = """
            SELECT * FROM corners_tabla 
            WHERE equipo_local_id = %s AND equipo_visitante_id = %s
            ORDER BY fecha DESC 
            LIMIT 1
            """
            
            cursor.execute(query_inverted, (equipo_visitante_id, equipo_local_id))
            result = cursor.fetchone()
            
            if result:
                columns = [desc[0] for desc in cursor.description]
                corners_data = dict(zip(columns, result))
            else:
                # Si no hay datos históricos, usar valores por defecto
                corners_data = {
                    'corners_vs_rival_hist': 8.0,
                    'last3_vs_media_liga': 1.0,
                    'local_avg_last3': 5.0,
                    'local_avg_last5': 5.0,
                    'visitante_avg_last3': 4.0,
                    'local_corner_category': 1,
                    'diff_last3_vs_last5_local': 0.0,
                    'visitante_avg_last5': 4.0,
                    'visitante_corner_category': 1,
                    'diff_last3_vs_last5_visitante': 0.0,
                    'consistencia_corners_local': 0.8,
                    'tiros_bloqueados_local': 2.0,
                    'corners_por_ataque_peligroso': 0.1,
                    'diff_corners_equipo': 1.0,
                    'diff_corners_local': 0.5,
                    'diff_corners_visitante': 0.5
                }
        
        cursor.close()
        conn.close()
        
        return corners_data
        
    except Exception as e:
        print(f"Error obteniendo datos de corners_tabla: {e}")
        # Retornar valores por defecto en caso de error
        return {
            'corners_vs_rival_hist': 8.0,
            'last3_vs_media_liga': 1.0,
            'local_avg_last3': 5.0,
            'local_avg_last5': 5.0,
            'visitante_avg_last3': 4.0,
            'local_corner_category': 1,
            'diff_last3_vs_last5_local': 0.0,
            'visitante_avg_last5': 4.0,
            'visitante_corner_category': 1,
            'diff_last3_vs_last5_visitante': 0.0,
            'consistencia_corners_local': 0.8,
            'tiros_bloqueados_local': 2.0,
            'corners_por_ataque_peligroso': 0.1,
            'diff_corners_equipo': 1.0,
            'diff_corners_local': 0.5,
            'diff_corners_visitante': 0.5
        }

def calculate_corners_total(corners_data, home_code, away_code):
    """Calcula los corners totales usando el modelo pre-entrenado"""
    try:
        # Generar features para el escalador (16 features)
        features_for_scaler = np.array([
            corners_data['corners_vs_rival_hist'],
            corners_data['last3_vs_media_liga'],
            corners_data['local_avg_last3'],
            corners_data['local_avg_last5'],
            corners_data['visitante_avg_last3'],
            corners_data['local_corner_category'],
            corners_data['diff_last3_vs_last5_local'],
            corners_data['visitante_avg_last5'],
            corners_data['visitante_corner_category'],
            corners_data['diff_last3_vs_last5_visitante'],
            corners_data['consistencia_corners_local'],
            corners_data['tiros_bloqueados_local'],
            corners_data['corners_por_ataque_peligroso'],
            corners_data['diff_corners_equipo'],
            corners_data['diff_corners_local'],
            corners_data['diff_corners_visitante']
        ]).reshape(1, -1)
        
        print(f"🔍 DEBUG - Features para escalador: {features_for_scaler.shape}")
        print(f"🔍 DEBUG - Primeras 5 features: {features_for_scaler[0][:5]}")
        
        # Escalar features
        features_scaled = predictor.corners_scaler.transform(features_for_scaler)
        
        # Generar features para el modelo (18 features: IDs + 16 escaladas)
        features_for_model = np.concatenate([
            np.array([[home_code, away_code]]),
            features_scaled
        ], axis=1)
        
        print(f"🔍 DEBUG - Features para modelo: {features_for_model.shape}")
        print(f"🔍 DEBUG - IDs de equipos: [{home_code}, {away_code}]")
        
        # Predicción con el modelo
        prediction = predictor.corners_model.predict(features_for_model)[0]
        
        print(f"🔍 DEBUG - Predicción raw del modelo: {prediction}")
        
        # Asegurar que sea un número entero positivo
        corners_total = max(1, int(round(prediction)))
        
        print(f"🔍 DEBUG - Corners totales finales: {corners_total}")
        
        return corners_total
        
    except Exception as e:
        print(f"Error calculando corners totales: {e}")
        # Fallback: suma de corners históricos o valor por defecto
        return max(1, int(corners_data.get('corners_vs_rival_hist', 8)))

def get_ganador_resultado_data(equipo_local_id, equipo_visitante_id):
    """Obtiene el último registro de ganador_resultado_tabla para los equipos especificados"""
    try:
        # Conectar a la base de datos PostgreSQL
        conn = psycopg2.connect(
            host="localhost",
            database="UPS_BET",
            user="user",
            password="upsbet05",
            port="5432"
        )
        
        cursor = conn.cursor()
        
        # Buscar el último registro por año para estos equipos
        query = """
        SELECT * FROM ganador_resultado_tabla 
        WHERE equipo_local_id = %s AND equipo_visitante_id = %s
        ORDER BY anio DESC 
        LIMIT 1
        """
        
        cursor.execute(query, (equipo_local_id, equipo_visitante_id))
        result = cursor.fetchone()
        
        if result:
            # Obtener nombres de columnas
            columns = [desc[0] for desc in cursor.description]
            ganador_data = dict(zip(columns, result))
            print(f"🔍 DEBUG - Columnas disponibles en ganador_resultado_tabla: {list(ganador_data.keys())[:10]}...")
        else:
            # Si no hay datos, buscar con equipos invertidos
            query_inverted = """
            SELECT * FROM ganador_resultado_tabla 
            WHERE equipo_local_id = %s AND equipo_visitante_id = %s
            ORDER BY anio DESC 
            LIMIT 1
            """
            
            cursor.execute(query_inverted, (equipo_visitante_id, equipo_local_id))
            result = cursor.fetchone()
            
            if result:
                columns = [desc[0] for desc in cursor.description]
                ganador_data = dict(zip(columns, result))
                print(f"🔍 DEBUG - Columnas disponibles en ganador_resultado_tabla (invertido): {list(ganador_data.keys())[:10]}...")
            else:
                # Si no hay datos históricos, usar valores por defecto
                ganador_data = {
                    'goles_local_avg_last3': 1.5,
                    'goles_local_avg_last5': 1.4,
                    'goles_visitante_avg_last3': 1.2,
                    'goles_visitante_avg_last5': 1.3,
                    'goles_vs_rival_hist': 1.8,
                    'goles_por_ataque_peligroso_local': 0.15,
                    'goles_por_ataque_peligroso_visitante': 0.12,
                    'eficiencia_ataque_local': 0.25,
                    'eficiencia_ataque_visitante': 0.22,
                    'defensa_local_avg_last3': 0.8,
                    'defensa_local_avg_last5': 0.9,
                    'defensa_visitante_avg_last3': 1.1,
                    'defensa_visitante_avg_last5': 1.0,
                    'form_local': 0.6,
                    'form_visitante': 0.5,
                    'momentum_local': 0.7,
                    'momentum_visitante': 0.6
                }
        
        cursor.close()
        conn.close()
        
        return ganador_data
        
    except Exception as e:
        print(f"Error obteniendo datos de ganador_resultado_tabla: {e}")
        # Retornar valores por defecto en caso de error
        return {
            'goles_local_avg_last3': 1.5,
            'goles_local_avg_last5': 1.4,
            'goles_visitante_avg_last3': 1.2,
            'goles_visitante_avg_last5': 1.3,
            'goles_vs_rival_hist': 1.8,
            'goles_por_ataque_peligroso_local': 0.15,
            'goles_por_ataque_peligroso_visitante': 0.12,
            'eficiencia_ataque_local': 0.25,
            'eficiencia_ataque_visitante': 0.22,
            'defensa_local_avg_last3': 0.8,
            'defensa_local_avg_last5': 0.9,
            'defensa_visitante_avg_last3': 1.1,
            'defensa_visitante_avg_last5': 1.0,
            'form_local': 0.6,
            'form_visitante': 0.5,
            'momentum_local': 0.7,
            'momentum_visitante': 0.6
        }

def calculate_score_prediction(ganador_data, home_code, away_code):
    """Calcula el marcador usando el modelo pre-entrenado"""
    try:
        print(f"🔍 DEBUG - Claves disponibles en ganador_data: {list(ganador_data.keys())}")
        
        # Verificar si las columnas necesarias existen
        required_columns = [
            'goles_local_avg_last3', 'goles_local_avg_last5', 'goles_visitante_avg_last3',
            'goles_visitante_avg_last5', 'goles_vs_rival_hist', 'goles_por_ataque_peligroso_local',
            'goles_por_ataque_peligroso_visitante', 'eficiencia_ataque_local', 'eficiencia_ataque_visitante',
            'defensa_local_avg_last3', 'defensa_local_avg_last5', 'defensa_visitante_avg_last3',
            'defensa_visitante_avg_last5', 'form_local', 'form_visitante', 'momentum_local', 'momentum_visitante'
        ]
        
        missing_columns = [col for col in required_columns if col not in ganador_data]
        if missing_columns:
            print(f"🔍 DEBUG - Columnas faltantes: {missing_columns}")
            print(f"🔍 DEBUG - Usando valores por defecto para columnas faltantes")
        
        # Generar features para el modelo de marcador con valores por defecto para columnas faltantes
        features_for_model = np.array([
            ganador_data.get('goles_local_avg_last3', 1.5),
            ganador_data.get('goles_local_avg_last5', 1.4),
            ganador_data.get('goles_visitante_avg_last3', 1.2),
            ganador_data.get('goles_visitante_avg_last5', 1.3),
            ganador_data.get('goles_vs_rival_hist', 1.8),
            ganador_data.get('goles_por_ataque_peligroso_local', 0.15),
            ganador_data.get('goles_por_ataque_peligroso_visitante', 0.12),
            ganador_data.get('eficiencia_ataque_local', 0.25),
            ganador_data.get('eficiencia_ataque_visitante', 0.22),
            ganador_data.get('defensa_local_avg_last3', 0.8),
            ganador_data.get('defensa_local_avg_last5', 0.9),
            ganador_data.get('defensa_visitante_avg_last3', 1.1),
            ganador_data.get('defensa_visitante_avg_last5', 1.0),
            ganador_data.get('form_local', 0.6),
            ganador_data.get('form_visitante', 0.5),
            ganador_data.get('momentum_local', 0.7),
            ganador_data.get('momentum_visitante', 0.6),
            home_code,
            away_code
        ]).reshape(1, -1)
        
        print(f"🔍 DEBUG - Features para modelo de marcador: {features_for_model.shape}")
        print(f"🔍 DEBUG - Primeras 5 features: {features_for_model[0][:5]}")
        print(f"🔍 DEBUG - IDs de equipos: [{home_code}, {away_code}]")
        
        # Verificar que el modelo esté cargado correctamente
        if predictor.score_model is None:
            print("🔍 DEBUG - ERROR: predictor.score_model es None")
            raise Exception("Modelo de marcador no está cargado")
        
        print(f"🔍 DEBUG - Tipo de predictor.score_model: {type(predictor.score_model)}")
        print(f"🔍 DEBUG - predictor.score_model: {predictor.score_model}")
        
        # Predicción con el modelo de marcador
        prediction = predictor.score_model.predict(features_for_model)
        
        # Si es MultiOutputRegressor, tomar el primer resultado
        if hasattr(prediction, 'shape') and len(prediction.shape) > 1:
            prediction = prediction[0]
        else:
            prediction = prediction
        
        print(f"🔍 DEBUG - Predicción raw del modelo de marcador: {prediction}")
        
        # El modelo devuelve [goles_local, goles_visitante]
        goles_local = max(0, int(round(prediction[0])))
        goles_visitante = max(0, int(round(prediction[1])))
        
        print(f"🔍 DEBUG - Marcador final: {goles_local}-{goles_visitante}")
        
        return {
            'home': goles_local,
            'away': goles_visitante
        }
        
    except Exception as e:
        print(f"Error calculando marcador: {e}")
        # Fallback: valores por defecto
        return {
            'home': max(0, int(ganador_data.get('goles_local_avg_last3', 1.5))),
            'away': max(0, int(ganador_data.get('goles_visitante_avg_last3', 1.2)))
        }

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
