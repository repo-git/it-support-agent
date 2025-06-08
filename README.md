# it-support-agent

Agente AI per il supporto IT capace di fornire assistenza remota guidata condividendo lo schermo dell'utente.

## Requisiti

- Python 3.11 o superiore
- [Docker](https://docs.docker.com/get-docker/) per l'infrastruttura LiveKit e Redis
- Git

## Installazione

1. Clona il repository e crea un ambiente virtuale:
   ```bash
   git clone <repo-url>
   cd it-support-agent
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Installa le dipendenze principali:
   ```bash
   pip install -r requirements.txt
   ```
3. Copia il file di esempio delle variabili d'ambiente e personalizzalo:
   ```bash
   cp .env.example .env
   # modifica .env con le tue credenziali
   ```

## Configurazione Docker

Per avviare LiveKit e Redis utilizza `docker-compose`:

```bash
cd docker
docker compose up -d
```

Questi servizi devono essere in esecuzione affinché l'agente possa connettersi correttamente.

## Avvio dell'agente

### Da terminale

Assicurati di aver attivato l'ambiente virtuale e di avere il file `.env` configurato, quindi esegui:

```bash
python src/main.py
```

### Tramite Docker

In alternativa puoi eseguire l'agente in un container Python:

```bash
docker run --rm -it --env-file .env -v $(pwd):/app -w /app python:3.11-slim \
    bash -c "pip install -r requirements.txt && python src/main.py"
```

## Uso

Una volta avviato l'agente, questo si collegherà a LiveKit e attenderà connessioni degli utenti per fornire supporto. Premi `CTRL+C` per interrompere l'esecuzione.

## Risorse utili

- [Documentazione LiveKit](https://docs.livekit.io/)
- [Modelli Ollama](https://github.com/ollama/ollama)
- [OpenAI API](https://platform.openai.com/docs)
- [Google Gemini](https://ai.google.dev/)
- [Anthropic Claude](https://www.anthropic.com/product)
