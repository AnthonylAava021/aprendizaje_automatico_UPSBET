#!/usr/bin/env python3
"""
Script para probar el sistema de predicción
"""

from app import create_app
from ml_models import predictor
from models import db, Equipo

def test_prediction():
    app = create_app()
    with app.app_context():
        print("=== PRUEBA DE PREDICCIÓN ===")
        
        # Probar predicción Emelec vs Barcelona SC
        print("\n🔮 Predicción: Emelec vs Barcelona SC")
        print("   Códigos: 4 vs 0")
        
        prediction = predictor.predict_match(4, 0)
        
        print(f"   Probabilidades:")
        print(f"     - Gana Emelec: {prediction['home_win']:.1%}")
        print(f"     - Empate: {prediction['draw']:.1%}")
        print(f"     - Gana Barcelona: {prediction['away_win']:.1%}")
        
        print(f"   Marcador sugerido: {prediction['score']['home']} - {prediction['score']['away']}")
        print(f"   Córners: {prediction['corners']['home']} - {prediction['corners']['away']}")
        print(f"   Tarjetas: {prediction['cards']['home']} - {prediction['cards']['away']}")
        
        # Probar predicción Barcelona SC vs Emelec
        print("\n🔮 Predicción: Barcelona SC vs Emelec")
        print("   Códigos: 0 vs 4")
        
        prediction2 = predictor.predict_match(0, 4)
        
        print(f"   Probabilidades:")
        print(f"     - Gana Barcelona: {prediction2['home_win']:.1%}")
        print(f"     - Empate: {prediction2['draw']:.1%}")
        print(f"     - Gana Emelec: {prediction2['away_win']:.1%}")
        
        print(f"   Marcador sugerido: {prediction2['score']['home']} - {prediction2['score']['away']}")
        print(f"   Córners: {prediction2['corners']['home']} - {prediction2['corners']['away']}")
        print(f"   Tarjetas: {prediction2['cards']['home']} - {prediction2['cards']['away']}")
        
        # Probar predicción sin datos históricos
        print("\n🔮 Predicción: LDU vs Mushuc Runa (sin datos históricos)")
        print("   Códigos: 5 vs 6")
        
        prediction3 = predictor.predict_match(5, 6)
        
        print(f"   Probabilidades:")
        print(f"     - Gana LDU: {prediction3['home_win']:.1%}")
        print(f"     - Empate: {prediction3['draw']:.1%}")
        print(f"     - Gana Mushuc Runa: {prediction3['away_win']:.1%}")
        
        print(f"   Marcador sugerido: {prediction3['score']['home']} - {prediction3['score']['away']}")
        print(f"   Córners: {prediction3['corners']['home']} - {prediction3['corners']['away']}")
        print(f"   Tarjetas: {prediction3['cards']['home']} - {prediction3['cards']['away']}")

if __name__ == '__main__':
    test_prediction()
