#!/usr/bin/env python3
"""
Script para verificar datos de partidos en la base de datos
"""

from app import create_app
from models import db, Partido, Equipo
from datetime import datetime

def check_data():
    app = create_app()
    with app.app_context():
        print("=== VERIFICACIÓN DE DATOS ===")
        
        # Verificar equipos
        equipos = Equipo.query.all()
        print(f"\n📊 Total equipos: {len(equipos)}")
        for equipo in equipos:
            print(f"   - {equipo.nombre} (ID: {equipo.id}, Código: {equipo.codigo})")
        
        # Verificar partidos
        partidos = Partido.query.all()
        print(f"\n⚽ Total partidos: {len(partidos)}")
        
        if partidos:
            print("\n📋 Últimos 5 partidos:")
            for partido in Partido.query.order_by(Partido.fecha.desc()).limit(5).all():
                local = partido.equipo_local.nombre if partido.equipo_local else "N/A"
                visita = partido.equipo_visita.nombre if partido.equipo_visita else "N/A"
                print(f"   - {local} vs {visita}")
                print(f"     Fecha: {partido.fecha}")
                print(f"     Goles: {partido.goles_local}-{partido.goles_visita}")
                print(f"     Córners: {partido.corners_local}-{partido.corners_visita}")
                print(f"     Tarjetas: {partido.tarjetas_amarillas_local}-{partido.tarjetas_amarillas_visita}")
                print()
        else:
            print("   ⚠️  No hay partidos registrados")
            print("   💡 Necesitas agregar datos históricos de partidos para hacer predicciones")

if __name__ == '__main__':
    check_data()
