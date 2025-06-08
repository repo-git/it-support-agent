"""
Base Agent per IT Support
Implementazione core dell'agente LiveKit
"""
import asyncio
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime, timedelta

from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli
)
from livekit.agents.llm import LLM, ChatContext, ChatMessage
from livekit.agents.stt import STT, SpeechEvent, SpeechEventType
from livekit.agents.tts import TTS
from livekit.agents.vad import VAD

from ..ai.model_manager import ModelManager
from ..vision.screenshot_analyzer import ScreenshotAnalyzer
from ..remote.control_handler import ControlHandler
from ..utils.config import Config
from ..utils.logging_setup import setup_logging


@dataclass
class SessionState:
    """Stato della sessione utente"""
    user_id: str
    session_start: datetime
    last_activity: datetime
    problem_description: str = ""
    current_step: int = 0
    solution_steps: List[str] = None
    screenshot_count: int = 0
    remote_control_active: bool = False
    user_satisfaction: Optional[int] = None

    def is_expired(self, timeout: int = 300) -> bool:
        """Verifica se la sessione è scaduta"""
        return (datetime.now() - self.last_activity).seconds > timeout

    def update_activity(self):
        """Aggiorna timestamp ultima attività"""
        self.last_activity = datetime.now()


class BaseAgent:
    """
    Agente base per supporto IT con LiveKit
    """

    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging(__name__)

        # Componenti core
        self.model_manager = ModelManager(config)
        # Passa l'istanza di ModelManager allo ScreenshotAnalyzer
        self.screenshot_analyzer = ScreenshotAnalyzer(self.model_manager)
        self.control_handler = ControlHandler(config)

        # Stato sessioni
        self.sessions: Dict[str, SessionState] = {}

        # Componenti LiveKit
        self.llm: Optional[LLM] = None
        self.stt: Optional[STT] = None
        self.tts: Optional[TTS] = None
        self.vad: Optional[VAD] = None

        # Chat context per conversazione
        self.chat_ctx = ChatContext()

        # Flag di stato
        self.is_ready = False
        self.is_speaking = False

    async def initialize(self):
        """Inizializza tutti i componenti dell'agente"""
        try:
            self.logger.info("Inizializzazione agente IT Support...")

            # Inizializza model manager
            await self.model_manager.initialize()

            # Ottieni componenti AI
            self.llm = await self.model_manager.get_llm("general_conversation")
            self.stt = await self.model_manager.get_stt()
            self.tts = await self.model_manager.get_tts()
            self.vad = await self.model_manager.get_vad()

            # Inizializza screenshot analyzer
            await self.screenshot_analyzer.initialize()

            # Inizializza control handler
            await self.control_handler.initialize()

            # Setup sistema di prompt
            self._setup_system_prompts()

            self.is_ready = True
            self.logger.info("Agente inizializzato con successo")

        except Exception as e:
            self.logger.error(f"Errore inizializzazione agente: {e}")
            raise

    def _setup_system_prompts(self):
        """Configura i prompt di sistema"""
        system_prompt = self.config.get("prompts.system_prompts.it_support", "")
        if system_prompt:
            self.chat_ctx.messages.append(
                ChatMessage.create(role="system", text=system_prompt)
            )

    async def handle_job(self, job: JobProcess):
        """Gestisce una nuova connessione/job"""
        if not self.is_ready:
            await self.initialize()

        self.logger.info(f"Nuova sessione: {job.room.name}")

        # Crea stato sessione
        session_id = job.room.name
        self.sessions[session_id] = SessionState(
            user_id=session_id,
            session_start=datetime.now(),
            last_activity=datetime.now(),
            solution_steps=[]
        )

        try:
            # Entra nella room
            await job.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

            # Setup handlers eventi
            job.room.on("participant_connected", self._on_participant_connected)
            job.room.on("participant_disconnected", self._on_participant_disconnected)
            job.room.on("track_published", self._on_track_published)
            job.room.on("track_unpublished", self._on_track_unpublished)

            # Avvia task principali
            await asyncio.gather(
                self._audio_processing_task(job),
                self._session_management_task(session_id),
                self._periodic_screenshot_task(session_id),
                return_exceptions=True
            )

        except Exception as e:
            self.logger.error(f"Errore gestione job: {e}")
        finally:
            # Cleanup sessione
            if session_id in self.sessions:
                del self.sessions[session_id]
            self.logger.info(f"Sessione terminata: {session_id}")

    async def _audio_processing_task(self, job: JobProcess):
        """Task per elaborazione audio"""
        audio_source = rtc.AudioSource(sample_rate=16000, num_channels=1)
        track = rtc.LocalAudioTrack.create_audio_track("agent-mic", audio_source)
        options = rtc.TrackPublishOptions()
        options.source = rtc.TrackSource.SOURCE_MICROPHONE

        await job.room.local_participant.publish_track(track, options)

        async def _speech_event_handler(event: SpeechEvent):
            if event.type == SpeechEventType.FINAL_TRANSCRIPT:
                await self._handle_user_speech(job.room.name, event.alternatives[0].text)

        # Setup STT stream
        stt_stream = self.stt.stream()
        stt_stream.on("speech_event", _speech_event_handler)

        # Process audio from participants
        for participant in job.room.participants.values():
            for track_pub in participant.track_publications.values():
                if track_pub.track and track_pub.track.kind == rtc.TrackKind.KIND_AUDIO:
                    audio_stream = rtc.AudioStream(track_pub.track)
                    async for frame in audio_stream:
                        stt_stream.push_frame(frame)

    async def _handle_user_speech(self, session_id: str, text: str):
        """Gestisce il parlato dell'utente"""
        if not text.strip():
            return

        self.logger.info(f"Utente ha detto: {text}")

        # Aggiorna stato sessione
        if session_id in self.sessions:
            self.sessions[session_id].update_activity()
            if not self.sessions[session_id].problem_description:
                self.sessions[session_id].problem_description = text

        # Aggiungi al contesto chat
        self.chat_ctx.messages.append(
            ChatMessage.create(role="user", text=text)
        )

        # Analizza se richiede screenshot
        if await self._should_take_screenshot(text):
            await self._take_and_analyze_screenshot(session_id)

        # Genera risposta
        response = await self._generate_response(session_id, text)

        # Invia risposta vocale
        await self._speak_response(response)

        # Aggiungi risposta al contesto
        self.chat_ctx.messages.append(
            ChatMessage.create(role="assistant", text=response)
        )

    async def _should_take_screenshot(self, text: str) -> bool:
        """Determina se serve uno screenshot basandosi sul testo"""
        screenshot_keywords = [
            "vedi", "guarda", "schermo", "errore", "finestra",
            "problema", "schermata", "desktop", "applicazione",
            "messaggio", "popup", "dialog", "screenshot"
        ]

        text_lower = text.lower()
        return any(keyword in text_lower for keyword in screenshot_keywords)

    async def _take_and_analyze_screenshot(self, session_id: str):
        """Cattura e analizza screenshot"""
        try:
            screenshot_path = await self.screenshot_analyzer.capture_screenshot()
            analysis = await self.screenshot_analyzer.analyze_screenshot(screenshot_path)

            # Aggiorna contesto chat con analisi
            analysis_text = f"[SCREENSHOT ANALYSIS] {analysis['description']}"
            if analysis.get('detected_issues'):
                analysis_text += f" Problemi rilevati: {', '.join(analysis['detected_issues'])}"

            self.chat_ctx.messages.append(
                ChatMessage.create(role="system", text=analysis_text)
            )

            # Aggiorna stato sessione
            if session_id in self.sessions:
                self.sessions[session_id].screenshot_count += 1

            self.logger.info(f"Screenshot analizzato per sessione {session_id}")

        except Exception as e:
            self.logger.error(f"Errore cattura/analisi screenshot: {e}")

    async def _generate_response(self, session_id: str, user_input: str) -> str:
        """Genera risposta usando LLM"""
        try:
            # Ottieni contesto sessione
            session = self.sessions.get(session_id)

            # Prepara prompt contestuale
            context_info = ""
            if session:
                context_info = f"""
                Sessione attiva da: {(datetime.now() - session.session_start).seconds // 60} minuti
                Problema principale: {session.problem_description}
                Step corrente: {session.current_step + 1}
                Screenshot catturati: {session.screenshot_count}
                """

            # Genera risposta
            response = await self.llm.chat(
                chat_ctx=self.chat_ctx,
                fnc_ctx=None
            )

            return response.choices[0].message.content

        except Exception as e:
            self.logger.error(f"Errore generazione risposta: {e}")
            return "Mi dispiace, ho avuto un problema tecnico. Puoi ripetere per favore?"

    async def _speak_response(self, text: str):
        """Converte testo in parlato"""
        if not text or self.is_speaking:
            return

        try:
            self.is_speaking = True

            # Genera audio TTS
            tts_stream = self.tts.synthesize(text)

            # Qui dovresti inviare l'audio alla room LiveKit
            # Implementazione specifica dipende dalla configurazione LiveKit

            self.logger.info(f"Risposta vocale inviata: {text[:50]}...")

        except Exception as e:
            self.logger.error(f"Errore sintesi vocale: {e}")
        finally:
            self.is_speaking = False

    async def _session_management_task(self, session_id: str):
        """Task per gestione sessione"""
        while session_id in self.sessions:
            session = self.sessions[session_id]

            # Verifica timeout sessione
            if session.is_expired(self.config.get("agent.session.idle_timeout", 300)):
                self.logger.info(f"Sessione {session_id} scaduta per inattività")
                break

            # Verifica durata massima
            max_duration = self.config.get("agent.session.max_duration", 3600)
            if (datetime.now() - session.session_start).seconds > max_duration:
                self.logger.info(f"Sessione {session_id} terminata per durata massima")
                break

            await asyncio.sleep(30)  # Check ogni 30 secondi

    async def _periodic_screenshot_task(self, session_id: str):
        """Task per screenshot periodici"""
        interval = self.config.get("agent.session.auto_screenshot_interval", 30)

        while session_id in self.sessions:
            try:
                await asyncio.sleep(interval)

                # Screenshot automatico solo se sessione attiva
                session = self.sessions.get(session_id)
                if session and not session.is_expired(60):  # Attiva negli ultimi 60s
                    await self._take_and_analyze_screenshot(session_id)

            except Exception as e:
                self.logger.error(f"Errore screenshot periodico: {e}")

    # Event handlers LiveKit
    async def _on_participant_connected(self, participant: rtc.RemoteParticipant):
        """Nuovo partecipante connesso"""
        self.logger.info(f"Partecipante connesso: {participant.identity}")
        await self._speak_response("Ciao! Sono il tuo assistente IT. Come posso aiutarti oggi?")

    async def _on_participant_disconnected(self, participant: rtc.RemoteParticipant):
        """Partecipante disconnesso"""
        self.logger.info(f"Partecipante disconnesso: {participant.identity}")

    async def _on_track_published(self, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
        """Track pubblicato"""
        self.logger.debug(f"Track pubblicato: {publication.sid}")

    async def _on_track_unpublished(self, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
        """Track non pubblicato"""
        self.logger.debug(f"Track rimosso: {publication.sid}")

    async def shutdown(self):
        """Shutdown agente"""
        self.logger.info("Shutdown agente...")

        # Chiudi tutte le sessioni
        for session_id in list(self.sessions.keys()):
            del self.sessions[session_id]

        # Cleanup componenti
        if self.model_manager:
            await self.model_manager.shutdown()

        if self.screenshot_analyzer:
            await self.screenshot_analyzer.shutdown()

        if self.control_handler:
            await self.control_handler.shutdown()

        self.logger.info("Agente terminato")


# Entry point per LiveKit CLI
async def entrypoint(ctx: JobContext):
    """Entry point per LiveKit Agent"""
    config = Config()
    agent = BaseAgent(config)

    try:
        await agent.handle_job(ctx.job)
    finally:
        await agent.shutdown()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))