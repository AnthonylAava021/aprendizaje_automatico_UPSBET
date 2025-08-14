#!/usr/bin/env python3
"""
Script para inspeccionar la estructura de los modelos
"""

import joblib
import numpy as np
import os
from custom_models import YellowEnsemble, safe_load_model

def inspect_model_structure():
    """Inspecciona la estructura de cada modelo"""
    print("=== INSPECCIÓN DE ESTRUCTURA DE MODELOS ===")
    
    models_path = 'app/models/'
    
    # 1. Inspeccionar escalador de córners
    print("\n1. 🔍 Inspeccionando escalador de córners...")
    try:
        if os.path.exists(f'{models_path}escalador_corners.pkl'):
            scaler = joblib.load(f'{models_path}escalador_corners.pkl')
            print(f"   📊 Tipo: {type(scaler)}")
            print(f"   📏 Número de features: {scaler.n_features_in_}")
            print(f"   📋 Feature names: {getattr(scaler, 'feature_names_in_', 'No disponible')}")
            print(f"   📊 Atributos: {[attr for attr in dir(scaler) if not attr.startswith('_')]}")
            
            # Probar con diferentes números de features
            for n_features in [4, 8, 12, 16]:
                try:
                    test_features = np.random.rand(1, n_features)
                    scaled = scaler.transform(test_features)
                    print(f"   ✅ {n_features} features: OK - Shape: {scaled.shape}")
                except Exception as e:
                    print(f"   ❌ {n_features} features: {e}")
        else:
            print("   ❌ Archivo no encontrado")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 2. Inspeccionar modelo de córners
    print("\n2. 🔍 Inspeccionando modelo de córners...")
    try:
        if os.path.exists(f'{models_path}prediccion_corners_totales.pkl'):
            model = safe_load_model(f'{models_path}prediccion_corners_totales.pkl')
            if model:
                print(f"   📊 Tipo: {type(model)}")
                print(f"   📏 Número de features: {getattr(model, 'n_features_in_', 'No disponible')}")
                print(f"   📋 Feature names: {getattr(model, 'feature_names_in_', 'No disponible')}")
                print(f"   📊 Atributos: {[attr for attr in dir(model) if not attr.startswith('_')]}")
            else:
                print("   ❌ No se pudo cargar")
        else:
            print("   ❌ Archivo no encontrado")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 3. Inspeccionar modelo de tarjetas
    print("\n3. 🔍 Inspeccionando modelo de tarjetas...")
    try:
        if os.path.exists(f'{models_path}modelo_amarillas.pkl'):
            custom_classes = {'YellowEnsemble': YellowEnsemble}
            model = safe_load_model(f'{models_path}modelo_amarillas.pkl', custom_classes)
            if model:
                print(f"   📊 Tipo: {type(model)}")
                print(f"   📊 Atributos: {[attr for attr in dir(model) if not attr.startswith('_')]}")
                
                # Verificar si tiene modelos
                if hasattr(model, 'models'):
                    print(f"   📋 Número de modelos: {len(model.models) if model.models else 0}")
                else:
                    print("   ⚠️  No tiene atributo 'models'")
                    
                # Verificar si tiene predict
                if hasattr(model, 'predict'):
                    print("   ✅ Tiene método predict")
                else:
                    print("   ❌ No tiene método predict")
            else:
                print("   ❌ No se pudo cargar")
        else:
            print("   ❌ Archivo no encontrado")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 4. Inspeccionar modelo de marcador
    print("\n4. 🔍 Inspeccionando modelo de marcador...")
    try:
        if os.path.exists(f'{models_path}modelo_marcador.pkl'):
            model = safe_load_model(f'{models_path}modelo_marcador.pkl')
            if model:
                print(f"   📊 Tipo: {type(model)}")
                
                if isinstance(model, dict):
                    print(f"   📋 Claves del diccionario: {list(model.keys())}")
                    for key, value in model.items():
                        print(f"      {key}: {type(value)}")
                        
                        # Si es un modelo sklearn
                        if hasattr(value, 'predict'):
                            print(f"         ✅ Tiene método predict")
                            print(f"         📏 Número de features: {getattr(value, 'n_features_in_', 'No disponible')}")
                        else:
                            print(f"         ❌ No tiene método predict")
                else:
                    print(f"   📊 Atributos: {[attr for attr in dir(model) if not attr.startswith('_')]}")
            else:
                print("   ❌ No se pudo cargar")
        else:
            print("   ❌ Archivo no encontrado")
    except Exception as e:
        print(f"   ❌ Error: {e}")

def test_feature_combinations():
    """Prueba diferentes combinaciones de features"""
    print("\n=== PRUEBA DE COMBINACIONES DE FEATURES ===")
    
    # Probar diferentes combinaciones de features para el escalador
    print("\n🔍 Probando escalador con diferentes features...")
    try:
        scaler = joblib.load('app/models/escalador_corners.pkl')
        
        # Crear features de ejemplo con diferentes longitudes
        feature_combinations = [
            # 4 features básicas
            ['corners_local', 'corners_visita', 'goles_local', 'goles_visita'],
            # 8 features (agregando más estadísticas)
            ['corners_local', 'corners_visita', 'goles_local', 'goles_visita', 
             'tarjetas_local', 'tarjetas_visita', 'posesion_local', 'posesion_visita'],
            # 12 features
            ['corners_local', 'corners_visita', 'goles_local', 'goles_visita',
             'tarjetas_local', 'tarjetas_visita', 'posesion_local', 'posesion_visita',
             'tiros_local', 'tiros_visita', 'faltas_local', 'faltas_visita'],
            # 16 features
            ['corners_local', 'corners_visita', 'goles_local', 'goles_visita',
             'tarjetas_local', 'tarjetas_visita', 'posesion_local', 'posesion_visita',
             'tiros_local', 'tiros_visita', 'faltas_local', 'faltas_visita',
             'offsides_local', 'offsides_visita', 'saves_local', 'saves_visita']
        ]
        
        for i, features in enumerate(feature_combinations):
            n_features = len(features)
            try:
                test_data = np.random.rand(1, n_features)
                scaled = scaler.transform(test_data)
                print(f"   ✅ {n_features} features: OK - Shape: {scaled.shape}")
                print(f"      Features: {features}")
            except Exception as e:
                print(f"   ❌ {n_features} features: {e}")
                
    except Exception as e:
        print(f"   ❌ Error: {e}")

if __name__ == '__main__':
    print("🔍 INSPECCIÓN DE MODELOS")
    print("=" * 50)
    
    inspect_model_structure()
    test_feature_combinations()
    
    print("\n" + "=" * 50)
    print("✅ INSPECCIÓN COMPLETADA")
