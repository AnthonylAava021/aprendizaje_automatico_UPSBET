#!/usr/bin/env python3
"""
Script para poblar datos de partidos históricos de ejemplo
"""

from app import create_app
from models import db, Partido, Equipo
from datetime import datetime, timedelta
import random

def populate_matches():
    app = create_app()
    with app.app_context():
        print("=== POBLANDO DATOS DE PARTIDOS ===")
        
        # Obtener equipos
        equipos = Equipo.query.all()
        if len(equipos) < 2:
            print("❌ Necesitas al menos 2 equipos")
            return
        
        # Crear partidos históricos
        partidos_data = [
            # Emelec vs Barcelona SC
            {
                'equipo_local_id': 3,  # Emelec
                'equipo_visita_id': 1,  # Barcelona SC
                'fecha': datetime.now() - timedelta(days=30),
                'goles_local': 2, 'goles_visita': 1,
                'corners_local': 6, 'corners_visita': 4,
                'tarjetas_amarillas_local': 2, 'tarjetas_amarillas_visita': 3,
                'tarjetas_rojas_local': 0, 'tarjetas_rojas_visita': 1,
                'resultado': 'L'
            },
            # Barcelona SC vs Emelec (partido de vuelta)
            {
                'equipo_local_id': 1,  # Barcelona SC
                'equipo_visita_id': 3,  # Emelec
                'fecha': datetime.now() - timedelta(days=15),
                'goles_local': 1, 'goles_visita': 1,
                'corners_local': 5, 'corners_visita': 5,
                'tarjetas_amarillas_local': 1, 'tarjetas_amarillas_visita': 2,
                'tarjetas_rojas_local': 0, 'tarjetas_rojas_visita': 0,
                'resultado': 'E'
            },
            # LDU vs Independiente del Valle
            {
                'equipo_local_id': 4,  # LDU
                'equipo_visita_id': 6,  # Independiente del Valle
                'fecha': datetime.now() - timedelta(days=20),
                'goles_local': 3, 'goles_visita': 0,
                'corners_local': 8, 'corners_visita': 3,
                'tarjetas_amarillas_local': 1, 'tarjetas_amarillas_visita': 2,
                'tarjetas_rojas_local': 0, 'tarjetas_rojas_visita': 0,
                'resultado': 'L'
            },
            # Independiente del Valle vs LDU
            {
                'equipo_local_id': 6,  # Independiente del Valle
                'equipo_visita_id': 4,  # LDU
                'fecha': datetime.now() - timedelta(days=10),
                'goles_local': 2, 'goles_visita': 2,
                'corners_local': 7, 'corners_visita': 6,
                'tarjetas_amarillas_local': 2, 'tarjetas_amarillas_visita': 1,
                'tarjetas_rojas_local': 0, 'tarjetas_rojas_visita': 0,
                'resultado': 'E'
            },
            # Aucas vs Deportivo Cuenca
            {
                'equipo_local_id': 10,  # Aucas
                'equipo_visita_id': 9,  # Deportivo Cuenca
                'fecha': datetime.now() - timedelta(days=25),
                'goles_local': 1, 'goles_visita': 2,
                'corners_local': 4, 'corners_visita': 7,
                'tarjetas_amarillas_local': 3, 'tarjetas_amarillas_visita': 1,
                'tarjetas_rojas_local': 0, 'tarjetas_rojas_visita': 0,
                'resultado': 'V'
            },
            # Deportivo Cuenca vs Aucas
            {
                'equipo_local_id': 9,  # Deportivo Cuenca
                'equipo_visita_id': 10,  # Aucas
                'fecha': datetime.now() - timedelta(days=5),
                'goles_local': 0, 'goles_visita': 1,
                'corners_local': 3, 'corners_visita': 5,
                'tarjetas_amarillas_local': 2, 'tarjetas_amarillas_visita': 2,
                'tarjetas_rojas_local': 0, 'tarjetas_rojas_visita': 0,
                'resultado': 'V'
            }
        ]
        
        # Agregar partidos
        for partido_data in partidos_data:
            partido = Partido(**partido_data)
            db.session.add(partido)
            # Obtener nombres de equipos
            local = Equipo.query.get(partido_data['equipo_local_id'])
            visita = Equipo.query.get(partido_data['equipo_visita_id'])
            print(f"✅ Agregado: {local.nombre} vs {visita.nombre}")
        
        db.session.commit()
        print(f"\n🎉 Se agregaron {len(partidos_data)} partidos históricos")

if __name__ == '__main__':
    populate_matches()
