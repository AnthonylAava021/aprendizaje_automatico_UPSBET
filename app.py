from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from models import db, Equipo, Partido, Prediccion
from ml_models import predictor
from config import config
import os

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
            
            # Realizar predicción
            prediction_result = predictor.predict_match(home_code, away_code)
            
            # Guardar predicción en la base de datos
            prediccion = Prediccion(
                equipo_local_id=home_team.id,
                equipo_visita_id=away_team.id,
                prob_local_win=prediction_result['home_win'],
                prob_draw=prediction_result['draw'],
                prob_visita_win=prediction_result['away_win'],
                goles_pred_local=prediction_result['score']['home'],
                goles_pred_visita=prediction_result['score']['away'],
                corners_pred_local=prediction_result['corners']['home'],
                corners_pred_visita=prediction_result['corners']['away'],
                tarjetas_pred_local=prediction_result['cards']['home'],
                tarjetas_pred_visita=prediction_result['cards']['away'],
                modelo_usado='RandomForest'
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

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
