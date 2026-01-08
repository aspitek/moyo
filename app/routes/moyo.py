from typing import Optional
import json

from fastapi import APIRouter, Query
from sqlalchemy import text

from app.config.db import get_connection
from app.services.embeddings import generate_embedding
from app.services.indexer import (
    index_new_artworks,
    reindex_all,
    get_indexing_stats
)

router = APIRouter()


@router.post("/search")
def search_artworks(
    query: str,
    top_k: int = Query(default=5, ge=1, le=50),
    min_price: Optional[float] = Query(default=None),
    max_price: Optional[float] = Query(default=None),
    style: Optional[str] = Query(default=None),
    location: Optional[str] = Query(default=None),
    ambiance: Optional[str] = Query(default=None),
    emotion: Optional[str] = Query(default=None)
):
    """
    Recherche sémantique dans les artworks avec filtres optionnels
    """
    # Générer embedding de la requête
    query_embedding = generate_embedding(query)
    
    # Convertir embedding en string format PostgreSQL array
    embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
    
    # Construire les filtres dynamiques
    filters = []
    params = {"limit": top_k}
    
    if min_price is not None:
        filters.append("(metadata->>'price')::float >= :min_price")
        params["min_price"] = min_price
    
    if max_price is not None:
        filters.append("(metadata->>'price')::float <= :max_price")
        params["max_price"] = max_price
    
    if style:
        filters.append("metadata->>'style' ILIKE :style")
        params["style"] = f"%{style}%"
    
    if location:
        filters.append("metadata->>'location' ILIKE :location")
        params["location"] = f"%{location}%"
    
    if ambiance:
        filters.append("metadata->>'ambiance' ILIKE :ambiance")
        params["ambiance"] = f"%{ambiance}%"
    
    if emotion:
        filters.append("metadata->>'emotion' ILIKE :emotion")
        params["emotion"] = f"%{emotion}%"
    
    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
    
    # Injecter le vector directement dans la query (pas en paramètre)
    with get_connection() as conn:
        query_sql = f"""
            SELECT 
                artwork_id,
                content,
                metadata,
                1 - (embedding <=> '{embedding_str}'::vector) AS similarity
            FROM moyo.documents
            {where_clause}
            ORDER BY embedding <=> '{embedding_str}'::vector
            LIMIT :limit
        """
        
        result = conn.execute(text(query_sql), params)
        
        results = [
            {
                "artwork_id": str(row[0]),
                "content": row[1][:200] + "..." if len(row[1]) > 200 else row[1],
                "metadata": row[2],
                "similarity": round(float(row[3]), 4)
            }
            for row in result
        ]
        
        return {
            "query": query,
            "total_results": len(results),
            "filters_applied": {
                "min_price": min_price,
                "max_price": max_price,
                "style": style,
                "location": location,
                "ambiance": ambiance,
                "emotion": emotion
            },
            "results": results
        }


@router.get("/similar/{artwork_id}")
def find_similar(
    artwork_id: str,
    top_k: int = Query(default=5, ge=1, le=20)
):
    """
    Trouve des artworks similaires à partir d'un artwork existant
    """
    with get_connection() as conn:
        # Récupérer l'embedding de l'artwork source
        result = conn.execute(text("""
            SELECT embedding::text
            FROM moyo.documents
            WHERE artwork_id = :artwork_id
        """), {"artwork_id": artwork_id})
        
        row = result.fetchone()
        if not row:
            return {"error": "Artwork not found", "artwork_id": artwork_id}
        
        source_embedding_str = row[0]
        
        # Trouver les similaires (exclure l'artwork source)
        query_sql = f"""
            SELECT 
                artwork_id,
                content,
                metadata,
                1 - (embedding <=> '{source_embedding_str}'::vector) AS similarity
            FROM moyo.documents
            WHERE artwork_id != :source_id
            ORDER BY embedding <=> '{source_embedding_str}'::vector
            LIMIT :limit
        """
        
        result = conn.execute(text(query_sql), {
            "source_id": artwork_id,
            "limit": top_k
        })
        
        return {
            "source_artwork_id": artwork_id,
            "similar_artworks": [
                {
                    "artwork_id": str(row[0]),
                    "content": row[1][:200] + "..." if len(row[1]) > 200 else row[1],
                    "metadata": row[2],
                    "similarity": round(float(row[3]), 4)
                }
                for row in result
            ]
        }


# ========== ROUTES D'INDEXATION ==========

@router.post("/index")
def index_new():
    """Indexe uniquement les nouveaux artworks non encore indexés"""
    count = index_new_artworks()
    return {
        "status": "success",
        "indexed": count,
        "message": f"{count} nouveaux artworks indexés"
    }


@router.post("/reindex")
def full_reindex():
    """Réindexe TOUS les artworks (opération lourde!)"""
    count = reindex_all()
    return {
        "status": "success",
        "reindexed": count,
        "message": f"Tous les artworks ({count}) ont été réindexés"
    }


@router.get("/stats")
def indexing_stats():
    """Statistiques détaillées sur l'indexation"""
    stats = get_indexing_stats()
    
    coverage_percent = 0
    if stats["total_artworks"] > 0:
        coverage_percent = round(
            (stats["indexed_documents"] / stats["total_artworks"]) * 100,
            2
        )
    
    return {
        **stats,
        "coverage_percent": coverage_percent,
        "status": "complete" if stats["not_indexed"] == 0 else "incomplete"
    }
