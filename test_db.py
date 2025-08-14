#!/usr/bin/env python3
"""
Script para probar la conexión a PostgreSQL
"""

import psycopg2
from config import config

def test_connection():
    """Prueba la conexión a PostgreSQL"""
    try:
        # Obtener configuración
        db_config = config['default']
        connection_string = db_config.SQLALCHEMY_DATABASE_URI
        
        print("🔍 Probando conexión a PostgreSQL...")
        print(f"📊 URL: {connection_string}")
        print("-" * 50)
        
        # Conectar a PostgreSQL
        conn = psycopg2.connect(connection_string)
        cursor = conn.cursor()
        
        # Probar consulta simple
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        
        print("✅ Conexión exitosa!")
        print(f"🐘 PostgreSQL version: {version[0]}")
        
        # Verificar si la base de datos existe
        cursor.execute("SELECT current_database();")
        db_name = cursor.fetchone()
        print(f"📁 Base de datos actual: {db_name[0]}")
        
        # Verificar tablas existentes
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        tables = cursor.fetchall()
        
        if tables:
            print("📋 Tablas existentes:")
            for table in tables:
                print(f"   - {table[0]}")
        else:
            print("📋 No hay tablas creadas aún")
        
        cursor.close()
        conn.close()
        
        print("-" * 50)
        print("🎉 Prueba de conexión completada exitosamente!")
        
    except psycopg2.OperationalError as e:
        print("❌ Error de conexión:")
        print(f"   {e}")
        print("\n💡 Posibles soluciones:")
        print("   1. Verificar que PostgreSQL esté ejecutándose")
        print("   2. Verificar credenciales en config.py")
        print("   3. Verificar que la base de datos 'UPS_BET' exista")
        print("   4. Verificar que el usuario 'postgres' tenga permisos")
        
    except Exception as e:
        print(f"❌ Error inesperado: {e}")

if __name__ == '__main__':
    test_connection()
