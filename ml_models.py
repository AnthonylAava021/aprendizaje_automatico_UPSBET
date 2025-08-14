import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import os
from models import db, Partido, Equipo
from datetime import datetime
from feature_generator import FeatureGenerator
from custom_models import safe_load_model

class MLPredictor:
    def __init__(self):
        self.corners_model = None
        self.corners_scaler = None
        self.cards_model = None
        self.result_model = None
        self.feature_generator = FeatureGenerator()
        self.load_models()
    
    def load_models(self):
        """Carga los modelos entrenados desde archivos"""
        try:
            # Cargar modelo de córners (usa escalador)
            if os.path.exists('app/models/prediccion_corners_totales.pkl'):
                self.corners_model = safe_load_model('app/models/prediccion_corners_totales.pkl')
                if self.corners_model:
                    print("✅ Modelo de córners cargado correctamente")
            
            if os.path.exists('app/models/escalador_corners.pkl'):
                self.corners_scaler = joblib.load('app/models/escalador_corners.pkl')
                print("✅ Escalador de córners cargado correctamente")
            
            # Cargar modelo de tarjetas
            if os.path.exists('app/models/modelo_amarillas.pkl'):
                try:
                    self.cards_model = joblib.load('app/models/modelo_amarillas.pkl')
                    print("✅ Modelo de tarjetas amarillas cargado correctamente")
                except Exception as e:
                    print(f"⚠️  Error cargando modelo de tarjetas: {e}")
                    self.cards_model = None
            
            # Cargar modelo de marcador
            if os.path.exists('app/models/modelo_marcador.pkl'):
                self.result_model = safe_load_model('app/models/modelo_marcador.pkl')
                if self.result_model:
                    print("✅ Modelo de marcador cargado correctamente")
                
        except Exception as e:
            print(f"Error cargando modelos: {e}")
            # Si no hay modelos, crear fallbacks
            self.create_fallback_models()
    
    def create_fallback_models(self):
        """Crea modelos básicos como fallback"""
        self.corners_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.corners_scaler = StandardScaler()
        self.cards_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.result_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    def get_historical_data(self, home_code, away_code):
        """
        Busca datos históricos entre dos equipos
        
        Args:
            home_code (int): Código del equipo local
            away_code (int): Código del equipo visita
            
        Returns:
            dict: Datos históricos del enfrentamiento
        """
        try:
            # Obtener IDs de equipos por código
            home_team = Equipo.query.filter_by(codigo=home_code).first()
            away_team = Equipo.query.filter_by(codigo=away_code).first()
            
            if not home_team or not away_team:
                return None
            
            # Buscar el último partido entre estos equipos (local vs visita)
            last_match = Partido.query.filter_by(
                equipo_local_id=home_team.id,
                equipo_visita_id=away_team.id
            ).order_by(Partido.fecha.desc()).first()
            
            if last_match:
                return {
                    'home_code': home_code,
                    'away_code': away_code,
                    'goles_local': last_match.goles_local,
                    'goles_visita': last_match.goles_visita,
                    'corners_local': last_match.corners_local,
                    'corners_visita': last_match.corners_visita,
                    'tarjetas_local': last_match.tarjetas_amarillas_local + last_match.tarjetas_rojas_local,
                    'tarjetas_visita': last_match.tarjetas_amarillas_visita + last_match.tarjetas_rojas_visita,
                    'resultado': last_match.resultado
                }
            
            # Si no hay datos, buscar con equipos invertidos
            last_match_inverted = Partido.query.filter_by(
                equipo_local_id=away_team.id,
                equipo_visita_id=home_team.id
            ).order_by(Partido.fecha.desc()).first()
            
            if last_match_inverted:
                return {
                    'home_code': home_code,
                    'away_code': away_code,
                    'goles_local': last_match_inverted.goles_visita,  # Invertir
                    'goles_visita': last_match_inverted.goles_local,  # Invertir
                    'corners_local': last_match_inverted.corners_visita,  # Invertir
                    'corners_visita': last_match_inverted.corners_local,  # Invertir
                    'tarjetas_local': last_match_inverted.tarjetas_amarillas_visita + last_match_inverted.tarjetas_rojas_visita,  # Invertir
                    'tarjetas_visita': last_match_inverted.tarjetas_amarillas_local + last_match_inverted.tarjetas_rojas_local,  # Invertir
                    'resultado': self.invert_result(last_match_inverted.resultado)
                }
            
            # Si no hay datos históricos, retornar valores por defecto
            return None
            
        except Exception as e:
            print(f"Error obteniendo datos históricos: {e}")
            return None
    
    def invert_result(self, result):
        """Invierte el resultado del partido"""
        if result == 'L':
            return 'V'
        elif result == 'V':
            return 'L'
        else:
            return 'E'  # Empate se mantiene
    
    def predict_match(self, home_code, away_code):
        """
        Predice el resultado de un partido usando datos históricos
        
        Args:
            home_code (int): Código del equipo local
            away_code (int): Código del equipo visita
            
        Returns:
            dict: Predicciones del partido
        """
        try:
            # Obtener datos históricos
            historical_data = self.get_historical_data(home_code, away_code)
            
            if historical_data:
                # Usar datos históricos para predicciones
                return self.predict_with_historical_data(historical_data)
            else:
                # Usar predicciones por defecto si no hay datos históricos
                return self.predict_without_historical_data(home_code, away_code)
                
        except Exception as e:
            print(f"Error en predicción: {e}")
            return self.get_default_prediction()
    
    def predict_with_historical_data(self, historical_data):
        """Predice usando datos históricos"""
        # Predicción de córners usando el modelo específico
        corners_prediction = self.predict_corners(historical_data)
        
        # Predicción de tarjetas
        cards_prediction = self.predict_cards(historical_data)
        
        # Predicción de resultado basada en goles históricos
        result_prediction = self.predict_result(historical_data)
        
        return {
            'home_win': result_prediction['home_win'],
            'draw': result_prediction['draw'],
            'away_win': result_prediction['away_win'],
            'score': {
                'home': result_prediction['score_home'],
                'away': result_prediction['score_away']
            },
            'corners': {
                'home': corners_prediction['home'],
                'away': corners_prediction['away']
            },
            'cards': {
                'home': cards_prediction['home'],
                'away': cards_prediction['away']
            }
        }
    
    def predict_corners(self, historical_data):
        """Predice córners usando el modelo específico"""
        try:
            if self.corners_model and self.corners_scaler:
                # Generar features correctas para el escalador (16 features)
                features = self.feature_generator.generate_corners_features(
                    historical_data.get('home_code', 0),
                    historical_data.get('away_code', 1),
                    historical_data
                )
                
                # Escalar features
                features_scaled = self.corners_scaler.transform(features)
                
                # Generar features para el modelo XGBoost (18 features)
                model_features = self.feature_generator.generate_corners_model_features(
                    historical_data.get('home_code', 0),
                    historical_data.get('away_code', 1),
                    historical_data
                )
                
                # Predicción
                prediction = self.corners_model.predict(model_features)[0]
                
                # Distribuir la predicción entre local y visita
                corners_home = max(2, int(prediction * 0.6))
                corners_away = max(2, int(prediction * 0.4))
                
                return {'home': corners_home, 'away': corners_away}
            else:
                # Fallback basado en datos históricos
                corners_home = max(3, historical_data['corners_local'] + np.random.randint(-1, 2))
                corners_away = max(2, historical_data['corners_visita'] + np.random.randint(-1, 2))
                
                return {'home': corners_home, 'away': corners_away}
        except Exception as e:
            print(f"Error prediciendo córners: {e}")
            return {'home': 5, 'away': 4}
    
    def predict_cards(self, historical_data):
        """Predice tarjetas usando el modelo entrenado"""
        try:
            if self.cards_model:
                # Crear features para el modelo de tarjetas
                features = np.array([[
                    historical_data['tarjetas_local'],
                    historical_data['tarjetas_visita'],
                    historical_data['goles_local'],
                    historical_data['goles_visita']
                ]])
                
                # Predicción
                prediction = self.cards_model.predict(features)[0]
                
                # Distribuir la predicción entre local y visita
                cards_home = max(1, int(prediction * 0.5))
                cards_away = max(1, int(prediction * 0.5))
                
                return {'home': cards_home, 'away': cards_away}
            else:
                # Fallback basado en datos históricos con variación
                cards_home = max(1, historical_data['tarjetas_local'] + np.random.randint(-1, 2))
                cards_away = max(1, historical_data['tarjetas_visita'] + np.random.randint(-1, 2))
                
                return {'home': cards_home, 'away': cards_away}
        except Exception as e:
            print(f"Error prediciendo tarjetas: {e}")
            return {'home': 2, 'away': 2}
    
    def predict_result(self, historical_data):
        """Predice resultado del partido usando el modelo entrenado"""
        try:
            if self.result_model and isinstance(self.result_model, dict) and 'model' in self.result_model:
                # Usar el modelo dentro del diccionario
                model = self.result_model['model']
                
                # Generar features para el modelo de marcador (477 features)
                features = self.feature_generator.generate_score_model_features(
                    historical_data.get('home_code', 0),
                    historical_data.get('away_code', 1),
                    historical_data
                )
                
                # Predicción de goles
                score_prediction = model.predict(features)[0]
                
                # Calcular probabilidades basadas en la predicción
                if isinstance(score_prediction, (list, np.ndarray)) and len(score_prediction) >= 2:
                    score_home = int(score_prediction[0])
                    score_away = int(score_prediction[1])
                else:
                    # Si el modelo devuelve un solo valor, distribuirlo
                    total_goals = int(score_prediction)
                    score_home = max(0, total_goals // 2)
                    score_away = max(0, total_goals - score_home)
                
                # Calcular probabilidades basadas en goles predichos
                total_goals = score_home + score_away
                
                if total_goals == 0:
                    home_win = 0.3
                    draw = 0.4
                    away_win = 0.3
                else:
                    home_goals_ratio = score_home / total_goals
                    away_goals_ratio = score_away / total_goals
                    
                    home_win = min(0.6, max(0.2, home_goals_ratio + 0.1))
                    away_win = min(0.6, max(0.2, away_goals_ratio + 0.1))
                    draw = max(0.1, 1 - home_win - away_win)
                    
                    # Normalizar
                    total = home_win + draw + away_win
                    home_win /= total
                    draw /= total
                    away_win /= total
                
                return {
                    'home_win': float(home_win),
                    'draw': float(draw),
                    'away_win': float(away_win),
                    'score_home': score_home,
                    'score_away': score_away
                }
            else:
                # Fallback usando datos históricos
                total_goals = historical_data['goles_local'] + historical_data['goles_visita']
                
                if total_goals == 0:
                    home_win = 0.3
                    draw = 0.4
                    away_win = 0.3
                else:
                    home_goals_ratio = historical_data['goles_local'] / total_goals
                    away_goals_ratio = historical_data['goles_visita'] / total_goals
                    
                    home_win = min(0.6, max(0.2, home_goals_ratio + 0.1))
                    away_win = min(0.6, max(0.2, away_goals_ratio + 0.1))
                    draw = max(0.1, 1 - home_win - away_win)
                    
                    total = home_win + draw + away_win
                    home_win /= total
                    draw /= total
                    away_win /= total
                
                score_home = max(0, historical_data['goles_local'] + np.random.randint(-1, 2))
                score_away = max(0, historical_data['goles_visita'] + np.random.randint(-1, 2))
                
                return {
                    'home_win': float(home_win),
                    'draw': float(draw),
                    'away_win': float(away_win),
                    'score_home': int(score_home),
                    'score_away': int(score_away)
                }
            
        except Exception as e:
            print(f"Error prediciendo resultado: {e}")
            return {
                'home_win': 0.33,
                'draw': 0.33,
                'away_win': 0.34,
                'score_home': 1,
                'score_away': 1
            }
    
    def predict_without_historical_data(self, home_code, away_code):
        """Predice sin datos históricos"""
        # Predicciones por defecto basadas en códigos de equipos
        home_win = 0.4 + (home_code % 10) * 0.01
        away_win = 0.3 + (away_code % 10) * 0.01
        draw = 1 - home_win - away_win
        
        # Normalizar
        total = home_win + draw + away_win
        home_win /= total
        draw /= total
        away_win /= total
        
        return {
            'home_win': float(home_win),
            'draw': float(draw),
            'away_win': float(away_win),
            'score': {
                'home': np.random.randint(0, 3),
                'away': np.random.randint(0, 3)
            },
            'corners': {
                'home': np.random.randint(3, 8),
                'away': np.random.randint(2, 7)
            },
            'cards': {
                'home': np.random.randint(1, 4),
                'away': np.random.randint(1, 4)
            }
        }
    
    def get_default_prediction(self):
        """Retorna predicción por defecto en caso de error"""
        return {
            'home_win': 0.33,
            'draw': 0.33,
            'away_win': 0.34,
            'score': {'home': 1, 'away': 1},
            'corners': {'home': 5, 'away': 4},
            'cards': {'home': 2, 'away': 2}
        }

# Instancia global del predictor
predictor = MLPredictor()
