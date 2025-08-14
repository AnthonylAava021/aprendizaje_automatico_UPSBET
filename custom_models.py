#!/usr/bin/env python3
"""
Clases personalizadas necesarias para cargar los modelos entrenados
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor

class YellowEnsemble(BaseEstimator, RegressorMixin):
    """
    Clase personalizada para el modelo de tarjetas amarillas
    """
    def __init__(self, models=None):
        self.models = models if models is not None else []
    
    def fit(self, X, y):
        """Entrenar el modelo"""
        if self.models:
            for model in self.models:
                model.fit(X, y)
        return self
    
    def predict(self, X):
        """Realizar predicción"""
        if not self.models:
            # Fallback si no hay modelos
            return np.array([2] * len(X))  # Predicción por defecto
        
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Promedio de predicciones
        return np.mean(predictions, axis=0)

class MultiOutputModel(BaseEstimator, RegressorMixin):
    """
    Clase personalizada para modelos de múltiples salidas
    """
    def __init__(self, model=None):
        self.model = model if model is not None else MultiOutputRegressor(RandomForestRegressor())
    
    def fit(self, X, y):
        """Entrenar el modelo"""
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """Realizar predicción"""
        return self.model.predict(X)

class DictModel(BaseEstimator, RegressorMixin):
    """
    Clase para manejar modelos guardados como diccionarios
    """
    def __init__(self, model_dict=None):
        self.model_dict = model_dict if model_dict is not None else {}
        self.model = None
    
    def fit(self, X, y):
        """Entrenar el modelo"""
        # Crear un modelo básico como fallback
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """Realizar predicción"""
        if self.model:
            return self.model.predict(X)
        else:
            # Predicción por defecto
            return np.array([1] * len(X))

# Función para cargar modelos con manejo de errores
def safe_load_model(filepath, custom_classes=None):
    """
    Carga un modelo de forma segura con manejo de clases personalizadas
    """
    import joblib
    
    try:
        # Intentar cargar con clases personalizadas
        if custom_classes:
            model = joblib.load(filepath, custom_objects=custom_classes)
        else:
            model = joblib.load(filepath)
        return model
    except Exception as e:
        print(f"Error cargando {filepath}: {e}")
        return None
