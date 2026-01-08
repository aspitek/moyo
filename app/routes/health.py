from fastapi import APIRouter

from app.services.indexer import get_indexing_stats

router = APIRouter()


@router.get("/health")
def health():
    """Healthcheck avec statistiques d'indexation"""
    stats = get_indexing_stats()
    return {
        "status": "ok",
        "database": "connected",
        **stats
    }


@router.get("/stats")
def stats():
    """Statistiques détaillées d'indexation"""
    return get_indexing_stats()
