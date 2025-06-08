"""
Client per provider cloud: OpenAI, Gemini, Claude
"""
import logging
from typing import Optional, Dict, Any
import base64

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Client per OpenAI GPT-4 Vision"""

    def __init__(self, api_key: str, model: str = "gpt-4-vision-preview"):
        self.api_key = api_key
        self.model = model

        try:
            import openai
            self.client = openai.AsyncOpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai package required: pip install openai")

    async def generate_response(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
            image_data: Optional[bytes] = None
    ) -> str:
        """Genera risposta con GPT-4"""
        try:
            messages = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            # Messaggio utente
            user_content = [{"type": "text", "text": prompt}]

            # Aggiungi immagine se presente
            if image_data:
                image_b64 = base64.b64encode(image_data).decode('utf-8')
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}"
                    }
                })

            messages.append({"role": "user", "content": user_content})

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise


class GeminiClient:
    """Client per Google Gemini"""

    def __init__(self, api_key: str, model: str = "gemini-pro-vision"):
        self.api_key = api_key
        self.model = model

        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(model)
        except ImportError:
            raise ImportError("google-generativeai package required")

    async def generate_response(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
            image_data: Optional[bytes] = None
    ) -> str:
        """Genera risposta con Gemini"""
        try:
            # Combina system prompt e user prompt
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\nUser: {prompt}"

            if image_data:
                # Gemini supporta immagini
                from PIL import Image
                import io
                image = Image.open(io.BytesIO(image_data))

                response = await self.client.generate_content_async([full_prompt, image])
            else:
                response = await self.client.generate_content_async(full_prompt)

            return response.text

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise


class ClaudeClient:
    """Client per Anthropic Claude"""

    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        self.api_key = api_key
        self.model = model

        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(api_key=api_key)
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")

    async def generate_response(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
            image_data: Optional[bytes] = None
    ) -> str:
        """Genera risposta con Claude"""
        try:
            messages = []

            # Claude gestisce il system prompt separatamente
            user_content = [{"type": "text", "text": prompt}]

            # Aggiungi immagine se presente
            if image_data:
                image_b64 = base64.b64encode(image_data).decode('utf-8')
                user_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_b64
                    }
                })

            messages.append({"role": "user", "content": user_content})

            kwargs = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 1000
            }

            if system_prompt:
                kwargs["system"] = system_prompt

            response = await self.client.messages.create(**kwargs)

            return response.content[0].text

        except Exception as e:
            logger.error(f"Claude API error: {e}")
            raise