#!/usr/bin/env python3
"""
Script para ejecutar la aplicación UPSBet
"""

from app import create_app

if __name__ == '__main__':
    app = create_app()
    print("🚀 Iniciando UPSBet...")
    print("📊 Base de datos: PostgreSQL")
    print("🤖 Modelo: Machine Learning")
    print("🌐 Servidor: http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)
