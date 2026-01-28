from typing import Optional, List
import json
from fastapi import APIRouter, Query, UploadFile, File, Form
from sqlalchemy import text
import base64
from google import genai
from google.genai import types
import asyncio

from app.config.db import get_connection
from app.services.embeddings import generate_embedding
from app.services.indexer import (
    index_new_artworks,
    reindex_all,
    get_indexing_stats
)
from app.config.settings import settings

router = APIRouter()

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB par image
MAX_IMAGES = 5

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




@router.post("/compose-scene")
async def compose_3d_scene(
    object_images: List[UploadFile] = File(
        ..., description="Images de l'objet sous différents angles (max 5)"
    ),
    space_images: List[UploadFile] = File(
        ..., description="Images de l'espace/environnement (max 5)"
    ),
    prompt: str = Form(..., description="Description de la composition désirée"),
    duration: int = Form(default=8, description="Durée en secondes (max 8)"),
):
    """
    Compose une scène 3D en générant une vidéo avec Veo 3.1
    """

    # Validation
    if len(object_images) > MAX_IMAGES or len(space_images) > MAX_IMAGES:
        raise HTTPException(
            status_code=400, detail=f"Maximum {MAX_IMAGES} images par catégorie"
        )

    if duration > 8:
        raise HTTPException(status_code=400, detail="Durée maximale: 8 secondes")

    # Configuration du client
    client = genai.Client(api_key=settings.google_genai_api_key)

    try:
        # Étape 1: Lire et valider les images
        object_data = []
        for img in object_images:
            content = await img.read()

            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=400, detail=f"Image {img.filename} trop grande (>10MB)"
                )

            object_data.append(
                types.Part.from_bytes(data=content, mime_type=img.content_type)
            )

        space_data = []
        for img in space_images:
            content = await img.read()

            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=400, detail=f"Image {img.filename} trop grande (>10MB)"
                )

            space_data.append(
                types.Part.from_bytes(data=content, mime_type=img.content_type)
            )

        # Étape 2: Générer image composite avec Imagen 3
        composition_prompt = f"""
        Analyse ces images et crée une composition 3D réaliste:
        - {len(object_images)} vues d'un objet 
        - {len(space_images)} vues d'un espace
        
        Intègre l'objet naturellement dans l'espace avec perspective correcte.
        
        Instructions: {prompt}
        """

        # Générer l'image composite
        response = await asyncio.to_thread(
            client.models.generate_content,
            model="gemini-2.0-flash",
            contents=[composition_prompt] + object_data + space_data,
        )

        # Extraire l'image générée
        if not response.candidates:
            raise HTTPException(
                status_code=500, detail="Aucune image générée par Gemini"
            )

        # Récupérer la première image générée
        composite_part = None
        for part in response.candidates[0].content.parts:
            if hasattr(part, "inline_data") and part.inline_data.mime_type.startswith(
                "image/"
            ):
                composite_part = part
                break

        if not composite_part:
            raise HTTPException(
                status_code=500, detail="Aucune image trouvée dans la réponse"
            )

        # Étape 3: Générer vidéo avec Veo 3.1
        video_prompt = f"""
        Crée une animation 3D fluide de cette scène.
        Mouvement de caméra cinématique autour de l'objet.
        {prompt}
        """

        operation = await asyncio.to_thread(
            client.models.generate_videos,
            model="veo-3.1-generate-preview",
            prompt=video_prompt,
            image=types.Part.from_bytes(
                data=composite_part.inline_data.data,
                mime_type=composite_part.inline_data.mime_type,
            ),
            config=types.GenerateVideosConfig(
                aspect_ratio="16:9",
            ),
        )

        # Polling avec timeout (max 3 minutes)
        max_wait = 360  # 6 minutes
        start_time = time.time()

        while not operation.done:
            if time.time() - start_time > max_wait:
                raise HTTPException(
                    status_code=504, detail="Timeout: génération vidéo trop longue"
                )

            await asyncio.sleep(10)  # Check toutes les 10s
            operation = await asyncio.to_thread(
                client.operations.get_videos_operation, operation=operation
            )

        # Récupérer la vidéo
        if not operation.response or not operation.response.generated_videos:
            raise HTTPException(
                status_code=500, detail="Aucune vidéo générée par Veo"
            )

        generated_video = operation.response.generated_videos[0]

        # Télécharger la vidéo
        video_bytes = await asyncio.to_thread(
            client.files.download, file=generated_video.video
        )

        return {
            "status": "success",
            "output_type": "video",
            "video_base64": base64.b64encode(video_bytes).decode(),
            "duration": duration,
            "resolution": "720p",
            "format": "16:9",
            "metadata": {
                "object_images_count": len(object_images),
                "space_images_count": len(space_images),
                "prompt": prompt,
                "generation_time_seconds": int(time.time() - start_time),
            },
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erreur génération: {str(e)}"
        ) from e