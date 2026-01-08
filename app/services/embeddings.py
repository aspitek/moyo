from openai import OpenAI
from app.config.settings import settings

openai_client = OpenAI(api_key=settings.openai_api_key)


def generate_embedding(text: str) -> list[float]:
    """Génère un embedding avec OpenAI"""
    if not text or not text.strip():
        # Retourner un vecteur zéro pour texte vide
        return [0.0] * settings.embedding_dimension
    
    try:
        response = openai_client.embeddings.create(
            model=settings.embedding_model,
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"❌ Erreur génération embedding: {e}")
        raise
