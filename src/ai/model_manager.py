# src/ai/model_manager.py (Continuazione)
"""
Manager per modelli AI - supporta Ollama, OpenAI, Gemini, Claude
"""
import logging
from typing import Dict, Any, Optional, List
from enum import Enum
from .ollama_client import OllamaClient
from .cloud_clients import OpenAIClient, GeminiClient, ClaudeClient
from utils.config import Config, AIProvider

logger = logging.getLogger(__name__)


class ModelManager:
    """Manager centralizzato per tutti i modelli AI"""

    def __init__(self, config: Config):
        self.config = config
        self.clients: Dict[AIProvider, Any] = {}
        self._initialize_clients()

    def _initialize_clients(self):
        """Inizializza i client AI disponibili"""

        try:
            # Ollama (sempre disponibile per locale)
            self.clients[AIProvider.OLLAMA] = OllamaClient(
                url=self.config.ollama_url,
                model=self.config.ollama_model
            )
            logger.info("✅ Ollama client inizializzato")
        except Exception as e:
            logger.warning(f"⚠️  Ollama non disponibile: {e}")

        # OpenAI
        if self.config.openai_api_key:
            try:
                self.clients[AIProvider.OPENAI] = OpenAIClient(
                    api_key=self.config.openai_api_key,
                    model=self.config.openai_model
                )
                logger.info("✅ OpenAI client inizializzato")
            except Exception as e:
                logger.warning(f"⚠️  OpenAI non disponibile: {e}")

        # Gemini
        if self.config.gemini_api_key:
            try:
                self.clients[AIProvider.GEMINI] = GeminiClient(
                    api_key=self.config.gemini_api_key,
                    model=self.config.gemini_model
                )
                logger.info("✅ Gemini client inizializzato")
            except Exception as e:
                logger.warning(f"⚠️  Gemini non disponibile: {e}")

        # Claude
        if self.config.anthropic_api_key:
            try:
                self.clients[AIProvider.CLAUDE] = ClaudeClient(
                    api_key=self.config.anthropic_api_key,
                    model=self.config.anthropic_model
                )
                logger.info("✅ Claude client inizializzato")
            except Exception as e:
                logger.warning(f"⚠️  Claude non disponibile: {e}")

    async def generate_response(
            self,
            prompt: str,
            provider: Optional[AIProvider] = None,
            system_prompt: Optional[str] = None,
            image_data: Optional[bytes] = None
    ) -> str:
        """Genera risposta utilizzando il provider specificato o quello di default"""

        if provider is None:
            provider = self.config.default_ai_provider

        if provider not in self.clients:
            # Fallback al primo client disponibile
            available_providers = list(self.clients.keys())
            if not available_providers:
                raise Exception("Nessun provider AI disponibile")

            provider = available_providers[0]
            logger.warning(f"Provider richiesto non disponibile, fallback a {provider}")

        try:
            client = self.clients[provider]

            # Genera la risposta
            response = await client.generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                image_data=image_data
            )

            logger.info(f"Risposta generata con {provider}")
            return response

        except Exception as e:
            logger.error(f"Errore generazione risposta con {provider}: {e}")

            # Prova con un altro provider se disponibile
            other_providers = [p for p in self.clients.keys() if p != provider]
            if other_providers:
                logger.info(f"Tentativo fallback con {other_providers[0]}")
                return await self.generate_response(
                    prompt=prompt,
                    provider=other_providers[0],
                    system_prompt=system_prompt,
                    image_data=image_data
                )

            raise Exception(f"Tutti i provider AI non disponibili: {e}")

    async def get_available_providers(self) -> List[AIProvider]:
        """Ottieni lista provider disponibili"""
        available = []
        for provider, client in self.clients.items():
            try:
                if provider == AIProvider.OLLAMA:
                    if await client.health_check():
                        available.append(provider)
                else:
                    # Per i provider cloud assumiamo disponibilità se inizializzati
                    available.append(provider)
            except Exception as e:
                logger.debug(f"Provider {provider} non disponibile: {e}")

        return available

    def get_provider_info(self) -> Dict[str, Any]:
        """Ottieni informazioni sui provider configurati"""
        info = {
            "default_provider": self.config.default_ai_provider,
            "available_providers": list(self.clients.keys()),
            "models": {}
        }

        for provider, client in self.clients.items():
            if provider == AIProvider.OLLAMA:
                info["models"][provider.value] = self.config.ollama_model
            elif provider == AIProvider.OPENAI:
                info["models"][provider.value] = self.config.openai_model
            elif provider == AIProvider.GEMINI:
                info["models"][provider.value] = self.config.gemini_model
            elif provider == AIProvider.CLAUDE:
                info["models"][provider.value] = self.config.anthropic_model

        return info

    async def close(self):
        """Chiudi tutti i client"""
        for client in self.clients.values():
            if hasattr(client, 'close'):
                await client.close()