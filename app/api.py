from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import text

from app.config.db import get_connection
from app.config.settings import settings
from app.services.indexer import index_new_artworks
from app.routes import health, moyo


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Événements de démarrage et arrêt"""
    # Démarrage
    print("🚀 Démarrage de l'API Moyo RAG...")
    
    # Vérifier la connexion DB
    try:
        with get_connection() as conn:
            result = conn.execute(text("SELECT version();"))
            version = result.scalar()
            print(f"✅ PostgreSQL connecté: {version[:60]}...")
    except Exception as e:
        print(f"❌ Erreur connexion DB: {e}")
        raise
    
    # Indexer les nouveaux artworks au démarrage
    if settings.index_on_startup:
        try:
            index_new_artworks()
        except Exception as e:
            print(f"⚠️  Erreur lors de l'indexation au démarrage: {e}")
    
    yield
    
    # Arrêt
    print("🛑 Arrêt de l'API...")


def create_app() -> FastAPI:
    """Factory pour créer l'application FastAPI"""
    app = FastAPI(
        title="Moyo RAG API",
        version="1.0.0",
        description="API de recherche sémantique pour artworks",
        lifespan=lifespan
    )
    
    # Enregistrer les routes
    app.include_router(health.router, tags=["Health"])
    app.include_router(moyo.router, prefix="/moyo/v1", tags=["Search"])
    
    @app.get("/")
    def root():
        return {
            "name": "Moyo RAG API",
            "version": "1.0.0",
            "status": "ok",
            "docs": "/docs"
        }
    
    return app


app = create_app()
