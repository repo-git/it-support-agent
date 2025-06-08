# src/agents/it_support_agent.py
"""
IT Support Agent - Agente principale per supporto tecnico
Versione completa con integrazione LiveKit, AI multimodale e controllo remoto
"""
import logging
import asyncio
import json
import base64
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm, stt, tts
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import openai, silero

from ai.model_manager import ModelManager
from vision.screenshot_analyzer import ScreenshotAnalyzer
from remote.control_handler import RemoteControlHandler
from utils.config import Config, AIProvider

logger = logging.getLogger(__name__)

class CustomLLMAdapter(llm.LLM):
    """Adapter personalizzato per integrare ModelManager con LiveKit"""
    
    def __init__(self, model_manager: ModelManager, config: Config):
        super().__init__()
        self.model_manager = model_manager
        self.config = config
        self.conversation_history: List[Dict[str, Any]] = []
    
    async def agenerate(
        self,
        *,
        chat_ctx: llm.ChatContext,
        conn_options: llm.LLMOptions = llm.LLMOptions(),
        fnc_ctx: Optional[llm.FunctionContext] = None,
    ) -> "llm.LLMStream":
        """Genera risposta utilizzando ModelManager"""
        try:
            # Estrai l'ultimo messaggio dell'utente
            messages = chat_ctx.messages
            if not messages:
                raise ValueError("Nessun messaggio nel contesto")
            
            last_message = messages[-1]
            user_prompt = last_message.content
            
            # Estrai system prompt se presente
            system_prompt = None
            for msg in messages:
                if msg.role == "system":
                    system_prompt = msg.content
                    break
            
            # Genera risposta
            response = await self.model_manager.generate_response(
                prompt=user_prompt,
                provider=self.config.default_ai_provider,
                system_prompt=system_prompt
            )
            
            # Crea stream response per LiveKit
            return CustomLLMStream(response)
            
        except Exception as e:
            logger.error(f"Errore generazione risposta LLM: {e}")
            error_response = "Mi dispiace, ho riscontrato un problema tecnico. Puoi ripetere la domanda?"
            return CustomLLMStream(error_response)

class CustomLLMStream(llm.LLMStream):
    """Stream personalizzato per le risposte"""
    
    def __init__(self, response: str):
        super().__init__()
        self.response = response
        self._sent = False
    
    async def __anext__(self) -> llm.ChatChunk:
        if self._sent:
            raise StopAsyncIteration
        
        self._sent = True
        return llm.ChatChunk(
            choices=[
                llm.Choice(
                    delta=llm.ChoiceDelta(content=self.response),
                    index=0
                )
            ]
        )

class ITSupportAgent:
    """Agente AI per supporto IT con capacità vocali e visive"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model_manager = ModelManager(config)
        self.screenshot_analyzer = ScreenshotAnalyzer(config)
        self.remote_control = RemoteControlHandler(config)
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.active_rooms: Dict[str, rtc.Room] = {}
        
        # Prompt di sistema per IT Support
        self.system_prompt = """
Sei un assistente IT esperto e amichevole specializzato nel supporto tecnico per sistemi Windows aziendali.

COMPETENZE PRINCIPALI:
- Risoluzione problemi Windows (errori, crash, performance)
- Supporto applicazioni Office e software aziendale
- Diagnosi problemi di rete e connettività
- Assistenza configurazione hardware e periferiche
- Sicurezza informatica e antivirus
- Backup e recupero dati
- Installazione e aggiornamento software

MODALITÀ OPERATIVE:
1. ASCOLTO ATTIVO: Comprendi il problema dell'utente attraverso domande mirate
2. DIAGNOSI VISIVA: Analizza screenshot per identificare errori e anomalie
3. GUIDA STEP-BY-STEP: Fornisci istruzioni chiare e dettagliate
4. CONTROLLO REMOTO: Quando necessario, richiedi permesso per intervenire direttamente
5. VERIFICA RISOLUZIONE: Conferma che il problema sia stato risolto

STILE COMUNICATIVO:
- Usa un tono professionale ma amichevole
- Spiega i passaggi in modo semplice e comprensibile
- Fornisci sempre il motivo dietro ogni azione
- Chiedi conferma prima di eseguire azioni critiche
- Sii paziente e pronto a ripetere le istruzioni

SICUREZZA:
- Non eseguire mai azioni che potrebbero danneggiare il sistema
- Richiedi sempre autorizzazione per controllo remoto
- Documenta tutte le azioni eseguite
- Rispetta la privacy dell'utente

COMANDI SPECIALI:
- Se l'utente dice "screenshot" o "condividi schermo", richiedi la condivisione dello schermo
- Se dici "controllo remoto", richiedi autorizzazione per il controllo diretto
- Se l'utente sembra confuso, offri di guidarlo passo-passo
- Mantieni sempre un approccio step-by-step per le soluzioni

Rispondi sempre in italiano e mantieni un approccio orientato alla risoluzione dei problemi.
"""
    
    async def start(self):
        """Avvia l'agente IT Support"""
        logger.info("🚀 Avvio IT Support Agent...")
        
        # Verifica provider AI disponibili
        available_providers = await self.model_manager.get_available_providers()
        if not available_providers:
            raise Exception("Nessun provider AI disponibile")
        
        logger.info(f"Provider AI disponibili: {available_providers}")
        
        # Avvia il worker LiveKit
        await cli.run_app(
            WorkerOptions(
                entrypoint_fnc=self.entrypoint,
                api_key=self.config.livekit_api_key,
                api_secret=self.config.livekit_api_secret,
                ws_url=self.config.livekit_url,
            )
        )
    
    async def entrypoint(self, ctx: JobContext):
        """Entry point per nuove sessioni LiveKit"""
        session_id = ctx.room.name
        logger.info(f"🔗 Nuova sessione: {session_id}")
        
        try:
            # Inizializza sessione
            await self._initialize_session(ctx, session_id)
            
            # Configura assistente vocale
            assistant = await self._create_voice_assistant()
            
            # Configura handlers eventi room
            await self._setup_room_handlers(ctx, assistant, session_id)
            
            # Avvia assistente
            assistant.start(ctx.room)
            self.active_rooms[session_id] = ctx.room
            
            # Avvia monitoraggio screenshot se abilitato
            if self.config.screenshot_interval > 0:
                asyncio.create_task(self._screenshot_monitor(session_id))
            
            # Messaggio di benvenuto
            await asyncio.sleep(1)  # Attendi connessione
            await assistant.say(
                "Ciao! Sono il tuo assistente IT. Come posso aiutarti oggi? "
                "Puoi descrivermi il problema o condividere il tuo schermo per mostramelo."
            )
            
            # Mantieni la sessione attiva
            await self._keep_session_alive(ctx, session_id)
            
        except Exception as e:
            logger.error(f"Errore nella sessione {session_id}: {e}")
        finally:
            # Cleanup sessione
            await self._cleanup_session(session_id)
    
    async def _initialize_session(self, ctx: JobContext, session_id: str):
        """Inizializza una nuova sessione"""
        self.sessions[session_id] = {
            "started_at": datetime.now(),
            "participant_count": 0,
            "last_screenshot": None,
            "conversation_history": [],
            "remote_control_active": False,
            "pending_approvals": [],
            "issues_detected": [],
            "resolution_steps": [],
            "user_context": {},
            "session_stats": {
                "messages_exchanged": 0,
                "screenshots_analyzed": 0,
                "problems_resolved": 0
            }
        }
        logger.info(f"✅ Sessione {session_id} inizializzata")
    
    async def _create_voice_assistant(self) -> VoiceAssistant:
        """Crea e configura l'assistente vocale"""
        
        # Configura STT (Speech-to-Text)
        stt_adapter = stt.StreamAdapter(
            openai.STT(
                model="whisper-1",
                api_key=self.config.openai_api_key
            ) if self.config.openai_api_key else None,
            vad=rtc.VAD.for_speaking_detection()
        )
        
        # Configura TTS (Text-to-Speech)
        tts_adapter = tts.StreamAdapter(
            openai.TTS(
                model="tts-1",
                voice="alloy",
                api_key=self.config.openai_api_key
            ) if self.config.openai_api_key else silero.TTS(),
            sentence_tokenizer=tts.basic.SentenceTokenizer(),
        )
        
        # Crea assistente con il nostro LLM personalizzato
        assistant = VoiceAssistant(
            vad=rtc.VAD.for_speaking_detection(),
            stt=stt_adapter,
            llm=self._create_llm_adapter(),
            tts=tts_adapter,
            chat_ctx=llm.ChatContext().append(
                role="system",
                text=self.system_prompt
            )
        )
        
        return assistant
    
    def _create_llm_adapter(self):
        """Crea adapter LLM personalizzato"""
        return CustomLLMAdapter(self.model_manager, self.config)
    
    async def _setup_room_handlers(self, ctx: JobContext, assistant: VoiceAssistant, session_id: str):
        """Configura handlers per eventi room"""
        
        @ctx.room.on("participant_connected")
        def on_participant_connected(participant: rtc.RemoteParticipant):
            logger.info(f"👤 Partecipante connesso: {participant.identity}")
            self.sessions[session_id]["participant_count"] += 1
            
            # Aggiorna contesto utente se disponibile
            if participant.metadata:
                try:
                    metadata = json.loads(participant.metadata)
                    self.sessions[session_id]["user_context"].update(metadata)
                except json.JSONDecodeError:
                    pass
        
        @ctx.room.on("participant_disconnected")
        def on_participant_disconnected(participant: rtc.RemoteParticipant):
            logger.info(f"👋 Partecipante disconnesso: {participant.identity}")
            if self.sessions[session_id]["participant_count"] > 0:
                self.sessions[session_id]["participant_count"] -= 1
        
        @ctx.room.on("track_published")
        def on_track_published(publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
            logger.info(f"📹 Track pubblicato: {publication.kind} da {participant.identity}")
            
            if publication.kind == rtc.TrackKind.KIND_VIDEO:
                # Subscribe al video track per screenshot
                asyncio.create_task(self._handle_video_track(publication, session_id))
        
        @ctx.room.on("data_received")
        async def on_data_received(data: bytes, participant: Optional[rtc.RemoteParticipant]):
            """Gestisce dati ricevuti (screenshot, comandi, etc.)"""
            try:
                message = data.decode('utf-8')
                await self._handle_data_message(session_id, message, data)
            except Exception as e:
                logger.error(f"Errore gestione dati: {e}")
    
    async def _handle_video_track(self, publication: rtc.RemoteTrackPublication, session_id: str):
        """Gestisce il track video per catturare screenshot"""
        try:
            # Subscribe al track
            track = await publication.set_subscribed(True)
            if not track:
                return
            
            logger.info(f"📺 Sottoscritto al video track per sessione {session_id}")
            
            # Cattura frame periodicamente
            while session_id in self.sessions:
                try:
                    # Attendi frame
                    frame = await track.recv()
                    if frame:
                        # Converti frame in immagine
                        image_data = await self._frame_to_image(frame)
                        if image_data:
                            await self._handle_screenshot(session_id, image_data)
                    
                    # Attendi prima del prossimo frame
                    await asyncio.sleep(self.config.screenshot_interval)
                    
                except Exception as e:
                    logger.error(f"Errore cattura frame: {e}")
                    await asyncio.sleep(5)  # Pausa prima di riprovare
                    
        except Exception as e:
            logger.error(f"Errore setup video track: {e}")
    
    async def _frame_to_image(self, frame) -> Optional[bytes]:
        """Converte un frame video in dati immagine"""
        try:
            # Implementazione conversione frame->immagine
            # Questo dipende dal formato del frame LiveKit
            # Per ora ritorniamo None come placeholder
            return None
        except Exception as e:
            logger.error(f"Errore conversione frame: {e}")
            return None
    
    async def _handle_data_message(self, session_id: str, message: str, raw_data: bytes):
        """Gestisce messaggi dati ricevuti"""
        try:
            if message.startswith("SCREENSHOT:"):
                # Screenshot inviato come base64
                screenshot_b64 = message[11:]  # Rimuovi prefisso
                screenshot_data = base64.b64decode(screenshot_b64)
                await self._handle_screenshot(session_id, screenshot_data)
                
            elif message.startswith("REMOTE_REQUEST:"):
                # Richiesta controllo remoto
                request_data = message[15:]
                await self._handle_remote_request(session_id, request_data)
                
            elif message.startswith("USER_ACTION:"):
                # Azione utente documentata
                action_data = message[12:]
                await self._log_user_action(session_id, action_data)
                
            elif message.startswith("PROBLEM_RESOLVED:"):
                # Problema risolto
                resolution_data = message[17:]
                await self._handle_problem_resolved(session_id, resolution_data)
                
        except Exception as e:
            logger.error(f"Errore gestione messaggio dati: {e}")
    
    async def _handle_screenshot(self, session_id: str, image_data: bytes):
        """Gestisce ricezione e analisi screenshot"""
        try:
            logger.info(f"📸 Screenshot ricevuto per sessione {session_id}")
            
            # Salva screenshot nella sessione
            self.sessions[session_id]["last_screenshot"] = image_data
            self.sessions[session_id]["session_stats"]["screenshots_analyzed"] += 1
            
            # Analizza screenshot
            analysis = await self.screenshot_analyzer.analyze_screenshot(image_data)
            
            if analysis and analysis.get("issues_found"):
                # Problemi trovati nell'analisi
                issues = analysis.get("issues", [])
                description = analysis.get("description", "")
                
                logger.info(f"🔍 Problemi rilevati: {len(issues)}")
                
                # Salva problemi nella sessione
                self.sessions[session_id]["issues_detected"].extend(issues)
                
                # Genera risposta basata sull'analisi
                prompt = f"""
Ho analizzato lo screenshot e ho trovato questi problemi:
{description}

Problemi specifici rilevati:
{json.dumps(issues, indent=2)}

Come assistente IT, fornisci una spiegazione chiara del problema e i passaggi per risolverlo.
Sii specifico e usa un linguaggio semplice.
"""
                
                response = await self.model_manager.generate_response(
                    prompt=prompt,
                    image_data=image_data,
                    provider=self.config.default_ai_provider
                )
                
                # Invia risposta all'utente
                await self._send_message_to_room(session_id, response)
                
            else:
                logger.info("✅ Nessun problema rilevato nello screenshot")
                
        except Exception as e:
            logger.error(f"Errore gestione screenshot: {e}")
    
    async def _handle_remote_request(self, session_id: str, request_data: str):
        """Gestisce richiesta controllo remoto"""
        try:
            if not self.config.enable_remote_control:
                await self._send_message_to_room(
                    session_id, 
                    "Mi dispiace, il controllo remoto non è attualmente abilitato."
                )
                return
            
            logger.info(f"🔧 Richiesta controllo remoto per sessione {session_id}")
            
            if self.config.require_approval:
                # Aggiungi richiesta alle approvazioni pending
                approval_request = {
                    "timestamp": datetime.now(),
                    "request_data": request_data,
                    "status": "pending"
                }
                self.sessions[session_id]["pending_approvals"].append(approval_request)
                
                # Richiedi approvazione
                await self._send_message_to_room(
                    session_id,
                    "Ho ricevuto la richiesta di controllo remoto. Per procedere, "
                    "ho bisogno della tua autorizzazione esplicita. "
                    "Confermi che posso prendere il controllo del tuo computer? "
                    "Rispondi 'Sì, autorizzato' per procedere."
                )
                
                # Imposta timeout per l'approvazione
                asyncio.create_task(self._handle_approval_timeout(session_id, len(self.sessions[session_id]["pending_approvals"]) - 1))
                
            else:
                # Avvia controllo remoto direttamente
                await self._start_remote_control(session_id, request_data)
                
        except Exception as e:
            logger.error(f"Errore richiesta controllo remoto: {e}")
    
    async def _start_remote_control(self, session_id: str, request_data: str):
        """Avvia sessione di controllo remoto"""
        try:
            logger.info(f"🚀 Avvio controllo remoto per sessione {session_id}")
            
            # Avvia controllo remoto
            success = await self.remote_control.start_session(session_id, request_data)
            
            if success:
                self.sessions[session_id]["remote_control_active"] = True
                await self._send_message_to_room(
                    session_id,
                    "✅ Controllo remoto attivato. Sto analizzando il problema e procederò con la risoluzione. "
                    "Puoi vedere le mie azioni sullo schermo."
                )
                
                # Imposta timeout per controllo remoto
                asyncio.create_task(self._handle_remote_control_timeout(session_id))
                
            else:
                await self._send_message_to_room(
                    session_id,
                    "❌ Non sono riuscito ad attivare il controllo remoto. "
                    "Procediamo con le istruzioni guidate."
                )
                
        except Exception as e:
            logger.error(f"Errore avvio controllo remoto: {e}")
            await self._send_message_to_room(
                session_id,
                "Si è verificato un errore durante l'avvio del controllo remoto. "
                "Procediamo con le istruzioni manuali."
            )
    
    async def _handle_approval_timeout(self, session_id: str, approval_index: int):
        """Gestisce timeout per approvazione controllo remoto"""
        await asyncio.sleep(60)  # Timeout di 1 minuto
        
        try:
            approvals = self.sessions[session_id]["pending_approvals"]
            if approval_index < len(approvals) and approvals[approval_index]["status"] == "pending":
                approvals[approval_index]["status"] = "timeout"
                await self._send_message_to_room(
                    session_id,
                    "⏰ Timeout per l'approvazione del controllo remoto. "
                    "Procediamo con le istruzioni guidate."
                )
        except KeyError:
            pass  # Sessione terminata
    
    async def _handle_remote_control_timeout(self, session_id: str):
        """Gestisce timeout per controllo remoto attivo"""
        await asyncio.sleep(self.config.control_timeout)
        
        try:
            if self.sessions[session_id]["remote_control_active"]:
                await self._stop_remote_control(session_id)
                await self._send_message_to_room(
                    session_id,
                    "⏰ Sessione di controllo remoto terminata per timeout. "
                    "Il controllo è tornato a te."
                )
        except KeyError:
            pass  # Sessione terminata
    
    async def _stop_remote_control(self, session_id: str):
        """Ferma il controllo remoto"""
        try:
            if session_id in self.sessions:
                self.sessions[session_id]["remote_control_active"] = False
            
            await self.remote_control.stop_session(session_id)
            logger.info(f"🛑 Controllo remoto fermato per sessione {session_id}")
            
        except Exception as e:
            logger.error(f"Errore stop controllo remoto: {e}")
    
    async def _log_user_action(self, session_id: str, action_data: str):
        """Registra azione utente"""
        try:
            timestamp = datetime.now()
            action_log = {
                "timestamp": timestamp,
                "action": action_data
            }
            
            # Aggiungi alla cronologia sessione
            if "action_history" not in self.sessions[session_id]:
                self.sessions[session_id]["action_history"] = []
            
            self.sessions[session_id]["action_history"].append(action_log)
            logger.info(f"📝 Azione utente registrata: {action_data}")
            
        except Exception as e:
            logger.error(f"Errore log azione utente: {e}")
    
    async def _handle_problem_resolved(self, session_id: str, resolution_data: str):
        """Gestisce risoluzione problema"""
        try:
            self.sessions[session_id]["session_stats"]["problems_resolved"] += 1
            
            resolution_log = {
                "timestamp": datetime.now(),
                "resolution": resolution_data,
                "session_duration": datetime.now() - self.sessions[session_id]["started_at"]
            }
            
            self.sessions[session_id]["resolution_steps"].append(resolution_log)
            
            await self._send_message_to_room(
                session_id,
                "🎉 Ottimo! Sono contento che il problema sia stato risolto. "
                "C'è altro con cui posso aiutarti?"
            )
            
            logger.info(f"✅ Problema risolto per sessione {session_id}")
            
        except Exception as e:
            logger.error(f"Errore gestione risoluzione: {e}")
    
    async def _send_message_to_room(self, session_id: str, message: str):
        """Invia messaggio alla room"""
        try:
            if session_id in self.active_rooms:
                room = self.active_rooms[session_id]
                # Invia messaggio come data
                await room.local_participant.publish_data(
                    message.encode('utf-8'),
                    destination_identities=None
                )
                
                # Aggiorna statistiche
                if session_id in self.sessions:
                    self.sessions[session_id]["session_stats"]["messages_exchanged"] += 1
                
        except Exception as e:
            logger.error(f"Errore invio messaggio: {e}")
    
    async def _screenshot_monitor(self, session_id: str):
        """Monitora e analizza screenshot periodicamente"""
        logger.info(f"🔍 Avvio monitoraggio screenshot per sessione {session_id}")
        
        while session_id in self.sessions:
            try:
                # Attendi intervallo
                await asyncio.sleep(self.config.screenshot_interval)
                
                # Verifica se abbiamo uno screenshot recente
                session_data = self.sessions[session_id]
                if session_data.get("last_screenshot"):
                    # Analizza solo se non analizzato di recente
                    last_analysis = session_data.get("last_screenshot_analysis", datetime.min)
                    if datetime.now() - last_analysis > timedelta(seconds=self.config.screenshot_interval):
                        
                        analysis = await self.screenshot_analyzer.analyze_screenshot(
                            session_data["last_screenshot"]
                        )
                        
                        session_data["last_screenshot_analysis"] = datetime.now()
                        
                        # Se trovati nuovi problemi, notifica
                        if analysis and analysis.get("issues_found"):
                            await self._notify_proactive_issues(session_id, analysis)
                
            except Exception as e:
                logger.error(f"Errore monitoraggio screenshot: {e}")
                await asyncio.sleep(30)  # Pausa prima di riprovare
    
    async def _notify_proactive_issues(self, session_id: str, analysis: Dict[str, Any]):
        """Notifica problemi rilevati proattivamente"""
        try:
            issues = analysis.get("issues", [])
            critical_issues = [issue for issue in issues if issue.get("severity") == "critical"]
            
            if critical_issues:
                await self._send_message_to_room(
                    session_id,
                    f"⚠️ Ho notato {len(critical_issues)} problema/i critico/i sul tuo schermo. "
                    "Vuoi che ti aiuti a risolverlo/i?"
                )
                
        except Exception as e:
            logger.error(f"Errore notifica problemi proattivi: {e}")
    
    async def _keep_session_alive(self, ctx: JobContext, session_id: str):
        """Mantiene la sessione attiva"""
        try:
            # Attendi fino a quando la room è aperta
            await ctx.room.aclose()
        except Exception as e:
            logger.error(f"Errore mantenimento sessione: {e}")
    
    async def _cleanup_session(self, session_id: str):
        """Cleanup sessione alla disconnessione"""
        try:
            logger.info(f"🧹 Cleanup sessione {session_id}")
            
            # Ferma controllo remoto se attivo
            if session_id in self.sessions and self.sessions[session_id].get("remote_control_active"):
                await self._stop_remote_control(session_id)
            
            # Rimuovi sessione
            if session_id in self.sessions:
                session_data = self.sessions.pop(session_id)
                
                # Log statistiche sessione
                stats = session_data.get("session_stats", {})
                duration = datetime.now() - session_data["started_at"]
                
                logger.info(f"📊 Statistiche sessione {session_id}:")
                logger.info(f"   Durata: {duration}")
                logger.info(f"   Messaggi: {stats.get('messages_exchanged', 0)}")
                logger.info(f"   Screenshot: {stats.get('screenshots_analyzed', 0)}")
                logger.info(f"   Problemi risolti: {stats.get('problems_resolved', 0)}")
            
            # Rimuovi room attiva
            if session_id in self.active_rooms:
                del self.active_rooms[session_id]
            
            logger.info(f"✅ Cleanup sessione {session_id} completato")
            
        except Exception as e:
            logger.error(f"Errore cleanup sessione: {e}")
    
    async def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Ottieni informazioni su una sessione"""
        if session_id in self.sessions:
            session_data = self.sessions[session_id].copy()
            # Rimuovi dati binari per la serializzazione
            if "last_screenshot" in session_data:
                session_data["last_screenshot"] = f"<screenshot_{len(session_data['last_screenshot'])}_bytes>"
            return session_data
        return None
    
    async def get_active_sessions(self) -> List[str]:
        """Ottieni lista sessioni attive"""
        return list(self.sessions.keys())
    
    async def force_disconnect_session(self, session_id: str) -> bool:
        """Forza disconnessione di una sessione"""
        try:
            if session_id in self.active_rooms:
                room = self.active_rooms[session_id]
                await room.disconnect()
                return True
            return False
        except Exception as e:
            logger.error(f"Errore disconnessione forzata: {e}")
            return False
