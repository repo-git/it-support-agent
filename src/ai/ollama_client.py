# src/ai/ollama_client.py
"""
Client per Ollama - Modelli AI locali
"""
import logging
import requests
import json
from typing import Optional, AsyncGenerator, Dict, Any
import asyncio
import aiohttp

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client per comunicare con Ollama locale"""

    def __init__(self, url: str = "http://localhost:11434", model: str = "llama3:latest"):
        self.url = url.rstrip('/')
        self.model = model
        self.session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Ottieni session HTTP riutilizzabile"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=300)  # 5 minuti timeout
            )
        return self.session

    async def health_check(self) -> bool:
        """Verifica se Ollama è raggiungibile"""
        try:
            session = await self._get_session()
            async with session.get(f"{self.url}/api/tags") as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False

    async def list_models(self) -> list:
        """Lista modelli disponibili"""
        try:
            session = await self._get_session()
            async with session.get(f"{self.url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    return [model['name'] for model in data.get('models', [])]
                return []
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

    async def generate_response(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
            stream: bool = False
    ) -> str:
        """Genera risposta con Ollama"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": stream
            }

            if system_prompt:
                payload["system"] = system_prompt

            session = await self._get_session()

            if stream:
                return await self._generate_streaming(payload)
            else:
                return await self._generate_single(payload)

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    async def _generate_single(self, payload: Dict[str, Any]) -> str:
        """Genera risposta singola (non streaming)"""
        session = await self._get_session()

        async with session.post(
                f"{self.url}/api/generate",
                json=payload
        ) as response:
            if response.status != 200:
                raise Exception(f"Ollama API error: {response.status}")

            data = await response.json()
            return data.get('response', '')

    async def _generate_streaming(self, payload: Dict[str, Any]) -> str:
        """Genera risposta streaming"""
        session = await self._get_session()
        full_response = ""

        async with session.post(
                f"{self.url}/api/generate",
                json=payload
        ) as response:
            if response.status != 200:
                raise Exception(f"Ollama API error: {response.status}")

            async for line in response.content:
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        if 'response' in data:
                            full_response += data['response']
                        if data.get('done', False):
                            break
                    except json.JSONDecodeError:
                        continue

        return full_response

    async def analyze_image(self, prompt: str, image_data: bytes) -> str:
        """Analizza immagine (se il modello supporta vision)"""
        # Per ora usiamo solo testo, in futuro implementeremo llava
        logger.warning("Image analysis not yet implemented for Ollama")
        return f"Analisi immagine non ancora supportata. Prompt: {prompt}"

    async def close(self):
        """Chiudi connessioni"""
        if self.session and not self.session.closed:
            await self.session.close()