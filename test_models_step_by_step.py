#!/usr/bin/env python3
"""
Script para probar cada modelo individualmente y diagnosticar problemas
"""

import joblib
import numpy as np
import os
from app import create_app
from models import db, Partido, Equipo
from custom_models import YellowEnsemble, MultiOutputModel, DictModel, safe_load_model

def test_model_loading():
    """Prueba la carga de cada modelo individualmente"""
    print("=== PRUEBA DE CARGA DE MODELOS ===")
    
    models_path = 'app/models/'
    
    # 1. Probar modelo de córners
    print("\n1. 🔍 Probando modelo de córners...")
    try:
        if os.path.exists(f'{models_path}prediccion_corners_totales.pkl'):
            corners_model = safe_load_model(f'{models_path}prediccion_corners_totales.pkl')
            if corners_model:
                print(f"   ✅ Modelo cargado: {type(corners_model)}")
                print(f"   📊 Atributos del modelo: {dir(corners_model)}")
                
                # Probar predicción con datos de ejemplo
                test_features = np.array([[5, 4, 2, 1]])  # corners_local, corners_visita, goles_local, goles_visita
                prediction = corners_model.predict(test_features)
                print(f"   🎯 Predicción de prueba: {prediction}")
                print(f"   📏 Forma de predicción: {prediction.shape}")
            else:
                print("   ❌ No se pudo cargar el modelo")
        else:
            print("   ❌ Archivo no encontrado")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 2. Probar escalador de córners
    print("\n2. 🔍 Probando escalador de córners...")
    try:
        if os.path.exists(f'{models_path}escalador_corners.pkl'):
            corners_scaler = joblib.load(f'{models_path}escalador_corners.pkl')
            print(f"   ✅ Escalador cargado: {type(corners_scaler)}")
            print(f"   📊 Atributos del escalador: {dir(corners_scaler)}")
            
            # Probar escalado con datos de ejemplo
            test_features = np.array([[5, 4, 2, 1]])
            scaled_features = corners_scaler.transform(test_features)
            print(f"   🎯 Datos escalados: {scaled_features}")
        else:
            print("   ❌ Archivo no encontrado")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 3. Probar modelo de tarjetas amarillas
    print("\n3. 🔍 Probando modelo de tarjetas amarillas...")
    try:
        if os.path.exists(f'{models_path}modelo_amarillas.pkl'):
            custom_classes = {'YellowEnsemble': YellowEnsemble}
            cards_model = safe_load_model(f'{models_path}modelo_amarillas.pkl', custom_classes)
            if cards_model:
                print(f"   ✅ Modelo cargado: {type(cards_model)}")
                print(f"   📊 Atributos del modelo: {dir(cards_model)}")
                
                # Probar predicción con datos de ejemplo
                test_features = np.array([[2, 3, 2, 1]])  # tarjetas_local, tarjetas_visita, goles_local, goles_visita
                prediction = cards_model.predict(test_features)
                print(f"   🎯 Predicción de prueba: {prediction}")
                print(f"   📏 Forma de predicción: {prediction.shape}")
            else:
                print("   ❌ No se pudo cargar el modelo")
        else:
            print("   ❌ Archivo no encontrado")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 4. Probar modelo de marcador
    print("\n4. 🔍 Probando modelo de marcador...")
    try:
        if os.path.exists(f'{models_path}modelo_marcador.pkl'):
            result_model = safe_load_model(f'{models_path}modelo_marcador.pkl')
            if result_model:
                print(f"   ✅ Modelo cargado: {type(result_model)}")
                print(f"   📊 Atributos del modelo: {dir(result_model)}")
                
                # Probar predicción con datos de ejemplo
                test_features = np.array([[2, 1, 5, 4]])  # goles_local, goles_visita, corners_local, corners_visita
                prediction = result_model.predict(test_features)
                print(f"   🎯 Predicción de prueba: {prediction}")
                print(f"   📏 Forma de predicción: {prediction.shape}")
            else:
                print("   ❌ No se pudo cargar el modelo")
        else:
            print("   ❌ Archivo no encontrado")
    except Exception as e:
        print(f"   ❌ Error: {e}")

def test_database_queries():
    """Prueba las consultas a la base de datos"""
    print("\n=== PRUEBA DE CONSULTAS A BASE DE DATOS ===")
    
    app = create_app()
    with app.app_context():
        try:
            # 1. Verificar equipos
            print("\n1. 🔍 Verificando equipos...")
            equipos = Equipo.query.all()
            print(f"   ✅ Total equipos: {len(equipos)}")
            
            # Mostrar algunos equipos con sus códigos
            for equipo in equipos[:5]:
                print(f"   📋 {equipo.nombre} (ID: {equipo.id}, Código: {equipo.codigo})")
            
            # 2. Verificar partidos
            print("\n2. 🔍 Verificando partidos...")
            partidos = Partido.query.all()
            print(f"   ✅ Total partidos: {len(partidos)}")
            
            if partidos:
                # Mostrar algunos partidos
                for partido in partidos[:3]:
                    local = partido.equipo_local.nombre if partido.equipo_local else "N/A"
                    visita = partido.equipo_visita.nombre if partido.equipo_visita else "N/A"
                    print(f"   📋 {local} vs {visita}")
                    print(f"      Goles: {partido.goles_local}-{partido.goles_visita}")
                    print(f"      Córners: {partido.corners_local}-{partido.corners_visita}")
                    print(f"      Tarjetas: {partido.tarjetas_amarillas_local}-{partido.tarjetas_amarillas_visita}")
            
            # 3. Probar búsqueda específica
            print("\n3. 🔍 Probando búsqueda específica...")
            
            # Buscar Emelec y Barcelona SC
            emelec = Equipo.query.filter_by(codigo=4).first()
            barcelona = Equipo.query.filter_by(codigo=0).first()
            
            if emelec and barcelona:
                print(f"   ✅ Emelec encontrado: ID {emelec.id}")
                print(f"   ✅ Barcelona SC encontrado: ID {barcelona.id}")
                
                # Buscar partido entre ellos
                partido = Partido.query.filter_by(
                    equipo_local_id=emelec.id,
                    equipo_visita_id=barcelona.id
                ).order_by(Partido.fecha.desc()).first()
                
                if partido:
                    print(f"   ✅ Partido encontrado: {partido.goles_local}-{partido.goles_visita}")
                else:
                    print("   ⚠️  No se encontró partido directo")
                    
                    # Buscar partido invertido
                    partido_invertido = Partido.query.filter_by(
                        equipo_local_id=barcelona.id,
                        equipo_visita_id=emelec.id
                    ).order_by(Partido.fecha.desc()).first()
                    
                    if partido_invertido:
                        print(f"   ✅ Partido invertido encontrado: {partido_invertido.goles_local}-{partido_invertido.goles_visita}")
                    else:
                        print("   ❌ No se encontró ningún partido entre estos equipos")
            else:
                print("   ❌ No se pudieron encontrar los equipos")
                
        except Exception as e:
            print(f"   ❌ Error en consultas: {e}")

def test_prediction_flow():
    """Prueba el flujo completo de predicción"""
    print("\n=== PRUEBA DE FLUJO DE PREDICCIÓN ===")
    
    app = create_app()
    with app.app_context():
        try:
            # Simular datos históricos
            historical_data = {
                'goles_local': 2,
                'goles_visita': 1,
                'corners_local': 6,
                'corners_visita': 4,
                'tarjetas_local': 2,
                'tarjetas_visita': 3,
                'resultado': 'L'
            }
            
            print(f"📊 Datos históricos de prueba: {historical_data}")
            
            # 1. Probar predicción de córners
            print("\n1. 🔮 Probando predicción de córners...")
            try:
                if os.path.exists('app/models/prediccion_corners_totales.pkl') and os.path.exists('app/models/escalador_corners.pkl'):
                    corners_model = joblib.load('app/models/prediccion_corners_totales.pkl')
                    corners_scaler = joblib.load('app/models/escalador_corners.pkl')
                    
                    features = np.array([[
                        historical_data['corners_local'],
                        historical_data['corners_visita'],
                        historical_data['goles_local'],
                        historical_data['goles_visita']
                    ]])
                    
                    features_scaled = corners_scaler.transform(features)
                    prediction = corners_model.predict(features_scaled)[0]
                    
                    print(f"   ✅ Predicción de córners: {prediction}")
                    
                    # Distribuir predicción
                    corners_home = max(2, int(prediction * 0.6))
                    corners_away = max(2, int(prediction * 0.4))
                    print(f"   📊 Distribución: Local {corners_home} - Visita {corners_away}")
                else:
                    print("   ❌ Modelos de córners no encontrados")
            except Exception as e:
                print(f"   ❌ Error en predicción de córners: {e}")
            
            # 2. Probar predicción de tarjetas
            print("\n2. 🔮 Probando predicción de tarjetas...")
            try:
                if os.path.exists('app/models/modelo_amarillas.pkl'):
                    cards_model = joblib.load('app/models/modelo_amarillas.pkl')
                    
                    features = np.array([[
                        historical_data['tarjetas_local'],
                        historical_data['tarjetas_visita'],
                        historical_data['goles_local'],
                        historical_data['goles_visita']
                    ]])
                    
                    prediction = cards_model.predict(features)[0]
                    print(f"   ✅ Predicción de tarjetas: {prediction}")
                    
                    # Distribuir predicción
                    cards_home = max(1, int(prediction * 0.5))
                    cards_away = max(1, int(prediction * 0.5))
                    print(f"   📊 Distribución: Local {cards_home} - Visita {cards_away}")
                else:
                    print("   ❌ Modelo de tarjetas no encontrado")
            except Exception as e:
                print(f"   ❌ Error en predicción de tarjetas: {e}")
            
            # 3. Probar predicción de resultado
            print("\n3. 🔮 Probando predicción de resultado...")
            try:
                if os.path.exists('app/models/modelo_marcador.pkl'):
                    result_model = joblib.load('app/models/modelo_marcador.pkl')
                    
                    features = np.array([[
                        historical_data['goles_local'],
                        historical_data['goles_visita'],
                        historical_data['corners_local'],
                        historical_data['corners_visita']
                    ]])
                    
                    prediction = result_model.predict(features)[0]
                    print(f"   ✅ Predicción de marcador: {prediction}")
                    print(f"   📏 Tipo de predicción: {type(prediction)}")
                    
                    # Interpretar predicción
                    if isinstance(prediction, (list, np.ndarray)) and len(prediction) >= 2:
                        score_home = int(prediction[0])
                        score_away = int(prediction[1])
                        print(f"   📊 Marcador predicho: {score_home} - {score_away}")
                    else:
                        total_goals = int(prediction)
                        score_home = max(0, total_goals // 2)
                        score_away = max(0, total_goals - score_home)
                        print(f"   📊 Total goles: {total_goals}, Distribución: {score_home} - {score_away}")
                else:
                    print("   ❌ Modelo de marcador no encontrado")
            except Exception as e:
                print(f"   ❌ Error en predicción de resultado: {e}")
                
        except Exception as e:
            print(f"❌ Error en flujo de predicción: {e}")

if __name__ == '__main__':
    print("🚀 INICIANDO PRUEBAS PASO A PASO")
    print("=" * 50)
    
    # Ejecutar todas las pruebas
    test_model_loading()
    test_database_queries()
    test_prediction_flow()
    
    print("\n" + "=" * 50)
    print("✅ PRUEBAS COMPLETADAS")
