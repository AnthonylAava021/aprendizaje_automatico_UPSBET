from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Equipo(db.Model):
    __tablename__ = 'equipos'
    
    id = db.Column(db.Integer, primary_key=True)
    nombre = db.Column(db.String(100), nullable=False, unique=True)
    codigo = db.Column(db.Integer, unique=True)
    logo_url = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'nombre': self.nombre,
            'codigo': self.codigo,
            'logo_url': self.logo_url
        }

class Partido(db.Model):
    __tablename__ = 'partidos'
    
    id = db.Column(db.Integer, primary_key=True)
    equipo_local_id = db.Column(db.Integer, db.ForeignKey('equipos.id'), nullable=False)
    equipo_visita_id = db.Column(db.Integer, db.ForeignKey('equipos.id'), nullable=False)
    fecha = db.Column(db.DateTime, nullable=False)
    goles_local = db.Column(db.Integer)
    goles_visita = db.Column(db.Integer)
    corners_local = db.Column(db.Integer)
    corners_visita = db.Column(db.Integer)
    tarjetas_amarillas_local = db.Column(db.Integer)
    tarjetas_amarillas_visita = db.Column(db.Integer)
    tarjetas_rojas_local = db.Column(db.Integer)
    tarjetas_rojas_visita = db.Column(db.Integer)
    resultado = db.Column(db.String(10))  # 'L', 'E', 'V' (Local, Empate, Visita)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relaciones
    equipo_local = db.relationship('Equipo', foreign_keys=[equipo_local_id])
    equipo_visita = db.relationship('Equipo', foreign_keys=[equipo_visita_id])
    
    def to_dict(self):
        return {
            'id': self.id,
            'equipo_local': self.equipo_local.nombre if self.equipo_local else None,
            'equipo_visita': self.equipo_visita.nombre if self.equipo_visita else None,
            'fecha': self.fecha.isoformat() if self.fecha else None,
            'goles_local': self.goles_local,
            'goles_visita': self.goles_visita,
            'corners_local': self.corners_local,
            'corners_visita': self.corners_visita,
            'tarjetas_amarillas_local': self.tarjetas_amarillas_local,
            'tarjetas_amarillas_visita': self.tarjetas_amarillas_visita,
            'tarjetas_rojas_local': self.tarjetas_rojas_local,
            'tarjetas_rojas_visita': self.tarjetas_rojas_visita,
            'resultado': self.resultado
        }

class Prediccion(db.Model):
    __tablename__ = 'predicciones'
    
    id = db.Column(db.Integer, primary_key=True)
    equipo_local_id = db.Column(db.Integer, db.ForeignKey('equipos.id'), nullable=False)
    equipo_visita_id = db.Column(db.Integer, db.ForeignKey('equipos.id'), nullable=False)
    prob_local_win = db.Column(db.Float, nullable=False)
    prob_draw = db.Column(db.Float, nullable=False)
    prob_visita_win = db.Column(db.Float, nullable=False)
    goles_pred_local = db.Column(db.Integer)
    goles_pred_visita = db.Column(db.Integer)
    corners_pred_local = db.Column(db.Integer)
    corners_pred_visita = db.Column(db.Integer)
    tarjetas_pred_local = db.Column(db.Integer)
    tarjetas_pred_visita = db.Column(db.Integer)
    modelo_usado = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relaciones
    equipo_local = db.relationship('Equipo', foreign_keys=[equipo_local_id])
    equipo_visita = db.relationship('Equipo', foreign_keys=[equipo_visita_id])
    
    def to_dict(self):
        return {
            'id': self.id,
            'equipo_local': self.equipo_local.nombre if self.equipo_local else None,
            'equipo_visita': self.equipo_visita.nombre if self.equipo_visita else None,
            'prob_local_win': self.prob_local_win,
            'prob_draw': self.prob_draw,
            'prob_visita_win': self.prob_visita_win,
            'goles_pred_local': self.goles_pred_local,
            'goles_pred_visita': self.goles_pred_visita,
            'corners_pred_local': self.corners_pred_local,
            'corners_pred_visita': self.corners_pred_visita,
            'tarjetas_pred_local': self.tarjetas_pred_local,
            'tarjetas_pred_visita': self.tarjetas_pred_visita,
            'modelo_usado': self.modelo_usado,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
