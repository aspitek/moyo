import os
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
from app.config.settings import settings

DATABASE_URL = settings.database_url

engine = create_engine(DATABASE_URL, poolclass=NullPool)


def get_connection():
    """Retourne une connexion à la DB"""
    return engine.connect()
