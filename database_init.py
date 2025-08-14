from models import db, Equipo, Partido, Prediccion
from config import config
import os

def init_db():
    """Inicializa la base de datos y crea las tablas"""
    from app import create_app
    app = create_app()
    
    with app.app_context():
        # Crear todas las tablas
        db.create_all()
        print("Tablas creadas exitosamente")
        
        # Poblar equipos si no existen
        populate_equipos()
        
def populate_equipos():
    """Pobla la tabla de equipos con los datos iniciales"""
    equipos_data = [
        {'nombre': 'Barcelona SC', 'codigo': 0, 'logo_url': 'img/Barcelona_Sporting_Club_Logo.png'},
        {'nombre': 'El Nacional', 'codigo': 2, 'logo_url': 'img/Nacional.png'},
        {'nombre': 'Emelec', 'codigo': 4, 'logo_url': 'img/EscudoCSEmelec.png'},
        {'nombre': 'LDU de Quito', 'codigo': 5, 'logo_url': 'img/Liga_Deportiva_Universitaria_de_Quito.png'},
        {'nombre': 'Mushuc Runa SC', 'codigo': 6, 'logo_url': 'img/MushucRuna.png'},
        {'nombre': 'Independiente del Valle', 'codigo': 7, 'logo_url': 'img/Independiente_del_Valle_Logo_2022.png'},
        {'nombre': 'CD Tecnico Universitario', 'codigo': 8, 'logo_url': 'img/Técnico_Universitario.png'},
        {'nombre': 'Delfin', 'codigo': 9, 'logo_url': 'img/Delfín_SC_logo.png'},
        {'nombre': 'Deportivo Cuenca', 'codigo': 10, 'logo_url': 'img/Depcuenca.png'},
        {'nombre': 'Aucas', 'codigo': 12, 'logo_url': 'img/SD_Aucas_logo.png'},
        {'nombre': 'Universidad Catolica', 'codigo': 13, 'logo_url': 'img/Ucatólica.png'},
        {'nombre': 'CSD Macara', 'codigo': 14, 'logo_url': 'img/Macara_6.png'},
        {'nombre': 'Orense SC', 'codigo': 15, 'logo_url': 'img/Orense_SC_logo.png'},
        {'nombre': 'Manta FC', 'codigo': 17, 'logo_url': 'img/Manta_F.C.png'},
        {'nombre': 'Libertad', 'codigo': 20, 'logo_url': 'img/Libertad_FC_Ecuador.png'},
        {'nombre': 'Vinotinto', 'codigo': 22, 'logo_url': 'img/Vinotinto.png'}
    ]
    
    for equipo_data in equipos_data:
        # Verificar si el equipo ya existe
        existing = Equipo.query.filter_by(nombre=equipo_data['nombre']).first()
        if not existing:
            equipo = Equipo(**equipo_data)
            db.session.add(equipo)
            print(f"Agregado equipo: {equipo_data['nombre']}")
    
    db.session.commit()
    print("Equipos poblados exitosamente")

if __name__ == '__main__':
    init_db()
