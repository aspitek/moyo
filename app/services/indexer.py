import uuid
import json
from sqlalchemy import text
from app.config.db import get_connection
from app.services.embeddings import generate_embedding


def index_new_artworks() -> int:
    print("🔄 Vérification des nouveaux artworks...")
    
    with get_connection() as conn:
        conn.execute(text("REFRESH MATERIALIZED VIEW moyo.artwork_documents;"))
        conn.commit()
        
        # COUNT retourne un seul scalaire
        result = conn.execute(text("""
            SELECT COUNT(*) 
            FROM moyo.artwork_documents ad
            LEFT JOIN moyo.documents d ON ad.id = d.artwork_id
            WHERE d.id IS NULL
        """))
        count = result.scalar()  # scalar() pas fetchall()
        
        if count == 0:
            print("✅ Tous les artworks sont déjà indexés")
            return 0
        
        print(f"📦 {count} nouveaux artworks détectés, indexation de tous...")
        
        # Récupérer les artworks non indexés
        result = conn.execute(text("""
            SELECT ad.id, ad.combined_text, ad.metadata
            FROM moyo.artwork_documents ad
            LEFT JOIN moyo.documents d ON ad.id = d.artwork_id
            WHERE d.id IS NULL
            ORDER BY ad.created_at DESC
        """))
        
        artworks = result.fetchall()  # OK ici, plusieurs lignes
        indexed_count = 0
        
        for idx, (artwork_id, combined_text, metadata) in enumerate(artworks, 1):
            try:
                embedding = generate_embedding(combined_text or "")
                
                # Embedding: passer une liste Python directement
                embedding_param = embedding
                
                # Metadata: convertir en JSON natif pour psycopg
                if isinstance(metadata, dict):
                    metadata_param = json.dumps(metadata)
                elif metadata is None:
                    metadata_param = '{}'
                else:
                    metadata_param = metadata
                
                # IMPORTANT: pas de ::vector / ::jsonb
                conn.execute(text("""
                    INSERT INTO moyo.documents (id, artwork_id, content, embedding, metadata)
                    VALUES (:id, :artwork_id, :content, :embedding, :metadata)
                """), {
                    "id": str(uuid.uuid4()),
                    "artwork_id": str(artwork_id),
                    "content": combined_text or "",
                    "embedding": embedding_param,
                    "metadata": metadata_param,
                })
                
                indexed_count += 1
                if idx % 10 == 0:
                    conn.commit()
                    print(f"  ✓ {idx}/{len(artworks)} indexés")
                    
            except Exception as e:
                print(f"  ✗ Erreur pour artwork {artwork_id}: {e}")
                continue
        
        conn.commit()
        print(f"✅ Indexation terminée! {indexed_count} artworks indexés")
        return indexed_count


def reindex_all() -> int:
    print("🔄 Réindexation complète en cours...")
    
    with get_connection() as conn:
        conn.execute(text("TRUNCATE TABLE moyo.documents CASCADE;"))
        conn.commit()
        print("🗑️  Table documents vidée")
        
        conn.execute(text("REFRESH MATERIALIZED VIEW moyo.artwork_documents;"))
        conn.commit()
        
        result = conn.execute(text("""
            SELECT id, combined_text, metadata
            FROM moyo.artwork_documents
            ORDER BY created_at DESC
        """))
        
        artworks = result.fetchall()
        total = len(artworks)
        print(f"📦 {total} artworks à réindexer...")
        
        indexed_count = 0
        
        for idx, (artwork_id, combined_text, metadata) in enumerate(artworks, 1):
            try:
                embedding = generate_embedding(combined_text or "")
                embedding_param = embedding
                
                if isinstance(metadata, dict):
                    metadata_param = json.dumps(metadata)
                elif metadata is None:
                    metadata_param = '{}'
                else:
                    metadata_param = metadata
                
                conn.execute(text("""
                    INSERT INTO moyo.documents (id, artwork_id, content, embedding, metadata)
                    VALUES (:id, :artwork_id, :content, :embedding, :metadata)
                """), {
                    "id": str(uuid.uuid4()),
                    "artwork_id": str(artwork_id),
                    "content": combined_text or "",
                    "embedding": embedding_param,
                    "metadata": metadata_param,
                })
                
                indexed_count += 1
                if idx % 10 == 0:
                    conn.commit()
                    print(f"  ✓ {idx}/{total} réindexés ({idx*100//total}%)")
                    
            except Exception as e:
                print(f"  ✗ Erreur pour artwork {artwork_id}: {e}")
                continue
        
        conn.commit()
        print(f"✅ Réindexation complète terminée! {indexed_count} artworks indexés")
        return indexed_count


def get_indexing_stats() -> dict:
    """Retourne les statistiques d'indexation"""
    with get_connection() as conn:
        # Tous scalar()
        artworks_count = conn.execute(
            text("SELECT COUNT(*) FROM moyo.artwork_documents")
        ).scalar()
        
        documents_count = conn.execute(
            text("SELECT COUNT(*) FROM moyo.documents")
        ).scalar()
        
        return {
            "total_artworks": artworks_count,
            "indexed_documents": documents_count,
            "not_indexed": artworks_count - documents_count
        }
