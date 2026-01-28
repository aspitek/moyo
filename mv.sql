-- 1. Créer le schéma moyo
CREATE SCHEMA IF NOT EXISTS moyo;

-- 2. Activer l'extension pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- 3. Créer la vue matérialisée
DROP MATERIALIZED VIEW IF EXISTS moyo.artwork_documents CASCADE;

CREATE MATERIALIZED VIEW moyo.artwork_documents AS
SELECT 
    a.id,
    a.title,
    a.description,
    a.price,
    a.stock,
    a.height,
    a.width,
    a.size,
    a.couleur_dominante,
    a.theme,
    a.ambiance,
    a.emotion,
    a.is_featured,
    
    -- Infos artiste
    ap.bio as artist_bio,
    ap.location as artist_location,
    
    -- Relations (noms lisibles)
    pt.name as painting_type,
    s.name as style_name,
    s.description as style_description,
    
    -- Texte combiné pour embedding
    CONCAT_WS(' | ',
        COALESCE(a.title, ''),
        COALESCE(a.description, ''),
        COALESCE(a.theme, ''),
        COALESCE(a.couleur_dominante, ''),
        COALESCE(a.ambiance, ''),
        COALESCE(a.emotion, ''),
        COALESCE(pt.name, ''),
        COALESCE(s.name, ''),
        COALESCE(ap.bio, ''),
        COALESCE(ap.location, '')
    ) as combined_text,
    
    -- Métadonnées JSON pour filtrage
    jsonb_build_object(
        'artist_id', a.artist_id,
        'price', a.price,
        'size', a.size,
        'height', a.height,
        'width', a.width,
        'is_featured', a.is_featured,
        'stock', a.stock,
        'location', ap.location,
        'painting_type', pt.name,
        'style', s.name,
        'ambiance', a.ambiance,
        'emotion', a.emotion,
        'couleur', a.couleur_dominante,
        'theme', a.theme
    ) as metadata,
    
    a.created_at,
    a.updated_at
FROM public.art_works a
LEFT JOIN public.artist_profiles ap 
    ON a.artist_id = ap.id
LEFT JOIN public.painting_type pt 
    ON a.painting_type_id = pt.id
LEFT JOIN public.styles s 
    ON a.style = s.id
WHERE a.is_deleted = false AND a.is_active = true;

-- Index unique pour refresh rapide
CREATE UNIQUE INDEX artwork_documents_id_idx ON moyo.artwork_documents(id);

-- Index pour recherche par updated_at
CREATE INDEX artwork_documents_updated_idx ON moyo.artwork_documents(updated_at DESC);

-- Refresh initial
REFRESH MATERIALIZED VIEW moyo.artwork_documents;

-- 4. Créer la table documents
DROP TABLE IF EXISTS moyo.documents CASCADE;

CREATE TABLE moyo.documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    artwork_id VARCHAR NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1536),  -- 1536 pour text-embedding-3-small
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Index vectoriel pour recherche de similarité (IVFFlat)
CREATE INDEX documents_embedding_ivfflat_idx 
ON moyo.documents 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Index pour recherche par artwork_id
CREATE INDEX documents_artwork_id_idx ON moyo.documents(artwork_id);

-- Index GIN pour filtrage metadata
CREATE INDEX documents_metadata_idx ON moyo.documents USING GIN (metadata);

-- Index pour recherche temporelle
CREATE INDEX documents_updated_at_idx ON moyo.documents(updated_at DESC);

-- Contrainte de clé étrangère vers art_works
ALTER TABLE moyo.documents 
ADD CONSTRAINT fk_documents_artwork 
FOREIGN KEY (artwork_id) REFERENCES public.art_works(id) 
ON DELETE CASCADE;

-- 5. Fonction trigger pour updated_at automatique
CREATE OR REPLACE FUNCTION moyo.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER documents_updated_at_trigger
BEFORE UPDATE ON moyo.documents
FOR EACH ROW
EXECUTE FUNCTION moyo.update_updated_at_column();

