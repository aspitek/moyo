from openai import OpenAI
from app.config.settings import settings
from typing import Optional


openai_client = OpenAI(api_key=settings.openai_api_key)


def generate_llm_response(
    prompt: str, 
    model: Optional[str] = None,
    max_tokens: int = 150,
    temperature: float = 0.7
) -> str:
    """
    Génère une réponse LLM avec OpenAI
    
    Args:
        prompt: Le prompt à envoyer au LLM
        model: Modèle à utiliser (défaut: settings.llm_model)
        max_tokens: Tokens max de sortie
        temperature: Créativité (0= déterministe, 1=créatif)
    
    Returns:
        Réponse du LLM (string nettoyée)
    """
    if not prompt or not prompt.strip():
        return "Prompt vide - aucune réponse générée."
    
    # Modèle par défaut depuis settings
    model = model or getattr(settings, 'llm_model', 'gpt-4o-mini')
    
    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system", 
                    "content": getattr(settings, 'llm_system_prompt', 'Tu es un assistant utile.')
                },
                {
                    "role": "user", 
                    "content": prompt.strip()
                }
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9
        )
        
        # Nettoyage de la réponse
        content = response.choices[0].message.content.strip()
        return content or "Réponse vide du LLM."
        
    except Exception as e:
        print(f"❌ Erreur génération LLM: {e}")
        # Fallback selon le contexte
        if "insufficient_quota" in str(e).lower():
            return "❌ Quota OpenAI épuisé - vérifiez votre plan."
        elif "invalid_api_key" in str(e).lower():
            return "❌ Clé API OpenAI invalide."
        else:
            return f"❌ Erreur LLM temporaire: {str(e)[:100]}"
