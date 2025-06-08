# src/main.py
"""
IT Support Agent - Entry Point
Agente AI per supporto tecnico con LiveKit
"""
import asyncio
import logging
import os
from dotenv import load_dotenv
from agents.it_support_agent import ITSupportAgent
from utils.logging_setup import setup_logging
from utils.config import Config

# Carica variabili ambiente
load_dotenv()


async def main():
    """Entry point principale dell'applicazione"""

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("🚀 Avvio IT Support Agent")

    try:
        # Carica configurazione
        config = Config()

        # Crea e avvia l'agente
        agent = ITSupportAgent(config)
        await agent.start()

    except KeyboardInterrupt:
        logger.info("🛑 Arresto richiesto dall'utente")
    except Exception as e:
        logger.error(f"❌ Errore critico: {e}")
        raise
    finally:
        logger.info("👋 IT Support Agent terminato")


if __name__ == "__main__":
    asyncio.run(main())
