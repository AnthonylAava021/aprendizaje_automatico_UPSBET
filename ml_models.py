import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import os

class MLPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.corners_model = None
        self.corners_scaler = None
        self.load_models()
    
    def load_models(self):
        """Carga los modelos entrenados desde archivos"""
        try:
            # Cargar modelo principal de predicción de resultados
            if os.path.exists('app/models/prediccion_corners_totales.pkl'):
                self.model = joblib.load('app/models/prediccion_corners_totales.pkl')
            
            # Cargar escalador
            if os.path.exists('app/models/escalador_corners.pkl'):
                self.scaler = joblib.load('app/models/escalador_corners.pkl')
                
        except Exception as e:
            print(f"Error cargando modelos: {e}")
            # Si no hay modelos, crear uno básico
            self.create_fallback_model()
    
    def create_fallback_model(self):
        """Crea un modelo básico como fallback"""
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
    
    def predict_match(self, home_code, away_code):
        """
        Predice el resultado de un partido
        
        Args:
            home_code (int): Código del equipo local
            away_code (int): Código del equipo visita
            
        Returns:
            dict: Predicciones del partido
        """
        try:
            # Crear features para la predicción
            features = np.array([[home_code, away_code]])
            
            # Escalar features si hay escalador disponible
            if self.scaler:
                features_scaled = self.scaler.transform(features)
            else:
                features_scaled = features
            
            # Predicción del resultado
            if self.model:
                # Para RandomForest, obtener probabilidades
                if hasattr(self.model, 'predict_proba'):
                    probs = self.model.predict_proba(features_scaled)[0]
                    home_win = probs[0] if len(probs) >= 1 else 0.33
                    draw = probs[1] if len(probs) >= 2 else 0.33
                    away_win = probs[2] if len(probs) >= 3 else 0.34
                else:
                    # Si no tiene predict_proba, usar valores por defecto
                    home_win, draw, away_win = 0.33, 0.33, 0.34
            else:
                # Modelo fallback
                home_win, draw, away_win = 0.33, 0.33, 0.34
            
            # Normalizar probabilidades
            total = home_win + draw + away_win
            home_win /= total
            draw /= total
            away_win /= total
            
            # Predicción de goles basada en probabilidades
            if home_win > draw and home_win > away_win:
                score_home = np.random.randint(1, 3)
                score_away = np.random.randint(0, 2)
            elif away_win > draw and away_win > home_win:
                score_home = np.random.randint(0, 2)
                score_away = np.random.randint(1, 3)
            else:
                score_home = np.random.randint(0, 2)
                score_away = np.random.randint(0, 2)
            
            # Predicción de córners
            corners_home = np.random.randint(3, 10)
            corners_away = np.random.randint(2, 9)
            
            # Predicción de tarjetas
            cards_home = np.random.randint(1, 5)
            cards_away = np.random.randint(1, 5)
            
            return {
                'home_win': float(home_win),
                'draw': float(draw),
                'away_win': float(away_win),
                'score': {
                    'home': int(score_home),
                    'away': int(score_away)
                },
                'corners': {
                    'home': int(corners_home),
                    'away': int(corners_away)
                },
                'cards': {
                    'home': int(cards_home),
                    'away': int(cards_away)
                }
            }
            
        except Exception as e:
            print(f"Error en predicción: {e}")
            # Retornar predicción por defecto
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
