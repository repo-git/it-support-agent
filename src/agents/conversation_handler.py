"""
Conversation Handler per gestione intelligente delle conversazioni
Gestisce il flusso conversazionale, context switching e decision making
"""
import asyncio
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum


@dataclass
class ConversationStep:
    """Singolo step di una conversazione"""
    step_id: int
    title: str
    description: str
    action_type: str  # "instruction", "screenshot", "remote_control", "verification"
    content: str
    completed: bool = False
    timestamp: Optional[datetime] = None
    user_feedback: Optional[str] = None


class ConversationState(Enum):
    """Stati possibili della conversazione"""
    GREETING = "greeting"
    PROBLEM_IDENTIFICATION = "problem_identification"
    ANALYSIS = "analysis"
    SOLUTION_PLANNING = "solution_planning"
    STEP_BY_STEP_GUIDANCE = "step_by_step_guidance"
    VERIFICATION = "verification"
    COMPLETION = "completion"
    ERROR_HANDLING = "error_handling"


@dataclass
class ProblemContext:
    """Contesto del problema identificato"""
    category: str  # "windows", "mac", "linux", "network", "software", "hardware"
    severity: str  # "low", "medium", "high", "critical"
    description: str
    symptoms: List[str]
    user_skill_level: str  # "beginner", "intermediate", "advanced"
    estimated_resolution_time: int  # minuti
    requires_remote_control: bool = False
    requires_admin_rights: bool = False


class ConversationHandler:
    """
    Gestisce il flusso conversazionale intelligente per supporto IT
    """

    def __init__(self, config, model_manager):
        self.config = config
        self.model_manager = model_manager
        self.logger = logging.getLogger(__name__)

        # Stato conversazione
        self.conversations: Dict[str, Dict] = {}

        # Knowledge base per problemi comuni
        self.common_problems = self._load_common_problems()

        # Template di risposta
        self.response_templates = self._load_response_templates()

    def _load_common_problems(self) -> Dict[str, Any]:
        """Carica database problemi comuni"""
        return {
            "windows_update_failed": {
                "category": "windows",
                "severity": "medium",
                "keywords": ["windows update", "aggiornamento", "errore 0x", "windows defender"],
                "solution_steps": [
                    "Verifichiamo lo stato del Windows Update",
                    "Controlliamo i servizi Windows Update",
                    "Eseguiamo il troubleshooter automatico",
                    "Se necessario, resettiamo i componenti Update"
                ],
                "estimated_time": 15,
                "requires_admin": True
            },
            "wifi_connection_issues": {
                "category": "network",
                "severity": "medium",
                "keywords": ["wifi", "connessione", "internet", "rete", "router"],
                "solution_steps": [
                    "Verifichiamo la connessione WiFi nelle impostazioni",
                    "Testiamo la connessione con altri dispositivi",
                    "Riavviamo l'adattatore di rete",
                    "Se necessario, resettiamo la configurazione TCP/IP"
                ],
                "estimated_time": 10,
                "requires_admin": False
            },
            "printer_not_working": {
                "category": "hardware",
                "severity": "low",
                "keywords": ["stampante", "printer", "stampa", "spool", "driver"],
                "solution_steps": [
                    "Verifichiamo lo stato della stampante",
                    "Controlliamo i driver installati",
                    "Verifichiamo lo spooler di stampa",
                    "Se necessario, reinstalliamo i driver"
                ],
                "estimated_time": 12,
                "requires_admin": True
            },
            "software_crash": {
                "category": "software",
                "severity": "medium",
                "keywords": ["crash", "errore applicazione", "si chiude", "non risponde", "blocco"],
                "solution_steps": [
                    "Identifichiamo l'applicazione problematica",
                    "Verifichiamo i log degli errori",
                    "Proviamo a riavviare in modalità sicura",
                    "Se necessario, reinstalliamo l'applicazione"
                ],
                "estimated_time": 20,
                "requires_admin": False
            },
            "slow_performance": {
                "category": "performance",
                "severity": "medium",
                "keywords": ["lento", "rallentato", "performance", "velocità", "lag"],
                "solution_steps": [
                    "Verifichiamo l'utilizzo CPU e memoria",
                    "Controlliamo i processi in background",
                    "Eseguiamo una scansione malware",
                    "Ottimizziamo l'avvio del sistema"
                ],
                "estimated_time": 25,
                "requires_admin": True
            }
        }

    def _load_response_templates(self) -> Dict[str, str]:
        """Carica template di risposta"""
        return {
            "greeting": "Ciao! Sono il tuo assistente IT. Descrivi il problema che stai riscontrando e ti aiuterò a risolverlo passo dopo passo.",

            "problem_clarification": "Ho capito che hai un problema con {topic}. Puoi dirmi più dettagli? Ad esempio: quando è iniziato, cosa stavi facendo quando è successo?",

            "analysis_start": "Perfetto, ora analizzerò la situazione. Prenderò uno screenshot del tuo schermo per vedere cosa sta succedendo.",

            "solution_intro": "Ho identificato il problema: {problem_description}. Ti guiderò attraverso {steps_count} passaggi per risolverlo. Tempo stimato: circa {estimated_time} minuti.",

            "step_instruction": "Passaggio {step_number} di {total_steps}: {instruction}. Dimmi quando hai completato questo passaggio.",

            "remote_control_request": "Per questo passaggio, sarebbe più efficace se potessi controllare direttamente il tuo computer. Posso avere il permesso per il controllo remoto?",

            "verification": "Ottimo! Ora verifichiamo che tutto funzioni correttamente. {verification_instruction}",

            "completion_success": "Perfetto! Abbiamo risolto il problema con successo. Il tuo {problem_type} ora dovrebbe funzionare correttamente. Hai altre domande?",

            "completion_partial": "Abbiamo fatto progressi significativi. {status_update}. Vuoi che continuiamo con ulteriori passaggi?",

            "error_handling": "Mi dispiace, sembra che ci sia stato un intoppo. {error_description}. Proviamo un approccio alternativo.",

            "need_more_info": "Per aiutarti meglio, ho bisogno di alcune informazioni aggiuntive: {questions}"
        }

    async def start_conversation(self, session_id: str, user_input: str) -> str:
        """Inizia una nuova conversazione"""
        self.conversations[session_id] = {
            "state": ConversationState.PROBLEM_IDENTIFICATION,
            "problem_context": None,
            "steps": [],
            "current_step": 0,
            "user_input_history": [user_input],
            "start_time": datetime.now(),
            "user_skill_level": "intermediate"  # Default, verrà aggiornato
        }

        # Analizza input iniziale per identificare il problema
        problem_context = await self._identify_problem(user_input)
        self.conversations[session_id]["problem_context"] = problem_context

        if problem_context:
            return await self._generate_problem_clarification(session_id, problem_context)
        else:
            return self.response_templates["need_more_info"].format(
                questions="Puoi descrivermi più dettagliatamente il problema? Ad esempio, quale dispositivo/software è coinvolto?"
            )

    async def process_user_input(self, session_id: str, user_input: str, screenshot_analysis: Optional[Dict] = None) -> \
    Tuple[str, Dict]:
        """Elabora input utente e restituisce risposta + azioni"""
        if session_id not in self.conversations:
            return await self.start_conversation(session_id, user_input)

        conversation = self.conversations[session_id]
        conversation["user_input_history"].append(user_input)

        # Aggiorna skill level utente basandosi sul linguaggio
        await self._update_user_skill_level(session_id, user_input)

        current_state = conversation["state"]

        # State machine per gestire diversi stati conversazione
        if current_state == ConversationState.PROBLEM_IDENTIFICATION:
            return await self._handle_problem_identification(session_id, user_input, screenshot_analysis)

        elif current_state == ConversationState.ANALYSIS:
            return await self._handle_analysis(session_id, user_input, screenshot_analysis)

        elif current_state == ConversationState.SOLUTION_PLANNING:
            return await self._handle_solution_planning(session_id, user_input)

        elif current_state == ConversationState.STEP_BY_STEP_GUIDANCE:
            return await self._handle_step_guidance(session_id, user_input)

        elif current_state == ConversationState.VERIFICATION:
            return await self._handle_verification(session_id, user_input)

        elif current_state == ConversationState.COMPLETION:
            return await self._handle_completion(session_id, user_input)

        else:
            return "Mi dispiace, c'è stato un errore nel flusso conversazionale.", {}

    async def _identify_problem(self, user_input: str) -> Optional[ProblemContext]:
        """Identifica il problema dall'input utente"""
        user_input_lower = user_input.lower()

        # Cerca match con problemi comuni
        for problem_key, problem_data in self.common_problems.items():
            for keyword in problem_data["keywords"]:
                if keyword.lower() in user_input_lower:
                    return ProblemContext(
                        category=problem_data["category"],
                        severity=problem_data["severity"],
                        description=f"Possibile problema: {problem_key.replace('_', ' ').title()}",
                        symptoms=[user_input],
                        user_skill_level="intermediate",
                        estimated_resolution_time=problem_data["estimated_time"],
                        requires_admin_rights=problem_data.get("requires_admin", False)
                    )

        # Se non trova match specifico, usa LLM per classificare
        try:
            llm = await self.model_manager.get_llm("problem_classification")
            classification_prompt = f"""
            Classifica questo problema IT e fornisci dettagli in formato JSON:
            Problema utente: "{user_input}"

            Formato risposta:
            {{
                "category": "windows|mac|linux|network|software|hardware|security",
                "severity": "low|medium|high|critical", 
                "description": "breve descrizione problema",
                "estimated_time": numero_minuti,
                "requires_admin": true|false
            }}
            """

            response = await llm.chat(classification_prompt)
            result = json.loads(response.choices[0].message.content)

            return ProblemContext(
                category=result["category"],
                severity=result["severity"],
                description=result["description"],
                symptoms=[user_input],
                user_skill_level="intermediate",
                estimated_resolution_time=result["estimated_time"],
                requires_admin_rights=result.get("requires_admin", False)
            )

        except Exception as e:
            self.logger.error(f"Errore classificazione problema: {e}")
            return None

    async def _generate_problem_clarification(self, session_id: str, problem_context: ProblemContext) -> str:
        """Genera domande di chiarimento"""
        conversation = self.conversations[session_id]
        conversation["state"] = ConversationState.ANALYSIS

        category_questions = {
            "windows": "Quale versione di Windows stai usando? Quando è iniziato il problema?",
            "network": "Il problema riguarda tutti i dispositivi o solo questo computer? Da quando non riesci a connetterti?",
            "software": "Quale programma specifico ha problemi? Riesci ad aprirlo o si chiude subito?",
            "hardware": "Il dispositivo è acceso e collegato correttamente? Ci sono luci o suoni particolari?",
            "performance": "Da quanto tempo il computer è diventato lento? In quali situazioni è più evidente?"
        }

        clarification = category_questions.get(problem_context.category, "Puoi darmi più dettagli sul problema?")

        return self.response_templates["problem_clarification"].format(
            topic=problem_context.category
        ) + " " + clarification

    async def _handle_problem_identification(self, session_id: str, user_input: str,
                                             screenshot_analysis: Optional[Dict]) -> Tuple[str, Dict]:
        """Gestisce identificazione problema"""
        conversation = self.conversations[session_id]

        # Se abbiamo screenshot analysis, incorporalo
        if screenshot_analysis:
            conversation["screenshot_analysis"] = screenshot_analysis

        # Aggiorna contesto problema con nuove info
        if conversation["problem_context"]:
            conversation["problem_context"].symptoms.append(user_input)

        # Transizione ad analisi
        conversation["state"] = ConversationState.SOLUTION_PLANNING

        actions = {"take_screenshot": True}

        return self.response_templates["analysis_start"], actions

    async def _handle_analysis(self, session_id: str, user_input: str, screenshot_analysis: Optional[Dict]) -> Tuple[
        str, Dict]:
        """Gestisce fase di analisi"""
        conversation = self.conversations[session_id]

        if screenshot_analysis:
            conversation["screenshot_analysis"] = screenshot_analysis
            # Usa screenshot per affinare la diagnosi
            await self._refine_problem_diagnosis(session_id, screenshot_analysis)

        # Transizione a pianificazione soluzione
        conversation["state"] = ConversationState.SOLUTION_PLANNING
        return await self._handle_solution_planning(session_id, user_input)

    async def _handle_solution_planning(self, session_id: str, user_input: str) -> Tuple[str, Dict]:
        """Pianifica la soluzione step-by-step"""
        conversation = self.conversations[session_id]
        problem_context = conversation["problem_context"]

        # Genera steps soluzione
        solution_steps = await self._generate_solution_steps(session_id, problem_context)
        conversation["steps"] = solution_steps
        conversation["current_step"] = 0
        conversation["state"] = ConversationState.STEP_BY_STEP_GUIDANCE

        intro_message = self.response_templates["solution_intro"].format(
            problem_description=problem_context.description,
            steps_count=len(solution_steps),
            estimated_time=problem_context.estimated_resolution_time
        )

        # Inizia primo step
        first_step_message = await self._get_current_step_instruction(session_id)

        return f"{intro_message}\n\n{first_step_message}", {}

    async def _generate_solution_steps(self, session_id: str, problem_context: ProblemContext) -> List[
        ConversationStep]:
        """Genera steps di soluzione personalizzati"""
        conversation = self.conversations[session_id]

        # Cerca soluzione in problemi comuni
        matching_problem = None
        for problem_key, problem_data in self.common_problems.items():
            if problem_data["category"] == problem_context.category:
                # Verifica se i sintomi matchano
                user_symptoms = " ".join(problem_context.symptoms).lower()
                if any(keyword.lower() in user_symptoms for keyword in problem_data["keywords"]):
                    matching_problem = problem_data
                    break

        if matching_problem:
            # Usa soluzione predefinita
            steps = []
            for i, step_desc in enumerate(matching_problem["solution_steps"]):
                steps.append(ConversationStep(
                    step_id=i + 1,
                    title=f"Step {i + 1}",
                    description=step_desc,
                    action_type="instruction",
                    content=step_desc
                ))
            return steps

        # Altrimenti genera con LLM
        try:
            llm = await self.model_manager.get_llm("solution_generation")

            context_info = {
                "problem": problem_context.description,
                "category": problem_context.category,
                "symptoms": problem_context.symptoms,
                "user_level": conversation["user_skill_level"],
                "screenshot_info": conversation.get("screenshot_analysis", {})
            }

            solution_prompt = f"""
            Genera una soluzione step-by-step per questo problema IT:

            Contesto: {json.dumps(context_info, indent=2)}

            Fornisci la risposta in formato JSON:
            {{
                "steps": [
                    {{
                        "title": "Titolo passo",
                        "description": "Descrizione dettagliata",
                        "action_type": "instruction|screenshot|remote_control|verification",
                        "content": "Istruzioni specifiche per l'utente"
                    }}
                ]
            }}

            Regole:
            - Usa linguaggio semplice per utenti {conversation["user_skill_level"]}
            - Massimo 6 steps
            - Ogni step deve essere chiaro e specifico
            - Includi verifiche intermedie
            """

            response = await llm.chat(solution_prompt)
            solution_data = json.loads(response.choices[0].message.content)

            steps = []
            for i, step_data in enumerate(solution_data["steps"]):
                steps.append(ConversationStep(
                    step_id=i + 1,
                    title=step_data["title"],
                    description=step_data["description"],
                    action_type=step_data["action_type"],
                    content=step_data["content"]
                ))

            return steps

        except Exception as e:
            self.logger.error(f"Errore generazione soluzione: {e}")
            # Fallback con steps generici
            return [
                ConversationStep(
                    step_id=1,
                    title="Diagnosi iniziale",
                    description="Analizziamo il problema più in dettaglio",
                    action_type="screenshot",
                    content="Prenderò uno screenshot per analizzare la situazione"
                ),
                ConversationStep(
                    step_id=2,
                    title="Applicazione soluzione",
                    description="Applichiamo la correzione",
                    action_type="instruction",
                    content="Ti guiderò attraverso i passaggi necessari"
                ),
                ConversationStep(
                    step_id=3,
                    title="Verifica finale",
                    description="Confermiamo che tutto funzioni",
                    action_type="verification",
                    content="Testiamo che il problema sia risolto"
                )
            ]

    async def _handle_step_guidance(self, session_id: str, user_input: str) -> Tuple[str, Dict]:
        """Gestisce guida step-by-step"""
        conversation = self.conversations[session_id]
        current_step_idx = conversation["current_step"]
        steps = conversation["steps"]

        if current_step_idx >= len(steps):
            # Tutti gli step completati
            conversation["state"] = ConversationState.VERIFICATION
            return await self._handle_verification(session_id, user_input)

        current_step = steps[current_step_idx]

        # Analizza risposta utente per capire se step è completato
        if await self._is_step_completed(user_input):
            current_step.completed = True
            current_step.timestamp = datetime.now()
            current_step.user_feedback = user_input

            # Passa al prossimo step
            conversation["current_step"] += 1

            if conversation["current_step"] >= len(steps):
                conversation["state"] = ConversationState.VERIFICATION
                return self.response_templates["verification"].format(
                    verification_instruction="Ora testiamo che tutto funzioni correttamente. Prova a riprodurre la situazione che causava il problema."
                ), {}
            else:
                next_instruction = await self._get_current_step_instruction(session_id)
                return f"Perfetto! {next_instruction}", {}

        else:
            # L'utente ha bisogno di aiuto con lo step corrente
            clarification = await self._provide_step_clarification(session_id, user_input, current_step)
            return clarification, {}

    async def _handle_verification(self, session_id: str, user_input: str) -> Tuple[str, Dict]:
        """Gestisce verifica finale"""
        conversation = self.conversations[session_id]

        if await self._is_problem_resolved(user_input):
            conversation["state"] = ConversationState.COMPLETION
            problem_type = conversation["problem_context"].category
            return self.response_templates["completion_success"].format(
                problem_type=problem_type
            ), {"session_complete": True}

        else:
            # Problema non completamente risolto
            alternative_response = await self._suggest_alternative_solution(session_id, user_input)
            return alternative_response, {}

    async def _handle_completion(self, session_id: str, user_input: str) -> Tuple[str, Dict]:
        """Gestisce completamento conversazione"""
        if "grazie" in user_input.lower() or "ok" in user_input.lower():
            return "Prego! Sono contento di aver risolto il problema. Per qualsiasi altra questione IT, sono sempre qui per aiutarti!", {
                "session_complete": True}

        else:
            # L'utente ha un altro problema
            return await self.start_conversation(session_id, user_input)

    async def _get_current_step_instruction(self, session_id: str) -> str:
        """Ottiene istruzione per step corrente"""
        conversation = self.conversations[session_id]
        current_step_idx = conversation["current_step"]
        steps = conversation["steps"]

        if current_step_idx >= len(steps):
            return "Abbiamo completato tutti i passaggi!"

        current_step = steps[current_step_idx]

        return self.response_templates["step_instruction"].format(
            step_number=current_step_idx + 1,
            total_steps=len(steps),
            instruction=current_step.content
        )

    async def _is_step_completed(self, user_input: str) -> bool:
        """Determina se l'utente ha completato lo step"""
        completion_indicators = [
            "fatto", "completato", "finito", "ok", "sì", "si", "yes",
            "done", "pronto", "fatto tutto", "ho fatto", "completato",
            "funziona", "risolto", "a posto"
        ]

        user_input_lower = user_input.lower()
        return any(indicator in user_input_lower for indicator in completion_indicators)

    async def _is_problem_resolved(self, user_input: str) -> bool:
        """Determina se il problema è risolto"""
        resolution_indicators = [
            "funziona", "risolto", "a posto", "perfetto", "tutto ok",
            "tutto bene", "problema risolto", "ora va", "adesso funziona"
        ]

        user_input_lower = user_input.lower()
        return any(indicator in user_input_lower for indicator in resolution_indicators)

    async def _update_user_skill_level(self, session_id: str, user_input: str):
        """Aggiorna valutazione skill level utente"""
        conversation = self.conversations[session_id]

        technical_terms = ["cmd", "prompt", "registry", "regedit", "service", "driver", "tcp/ip", "dns"]
        beginner_terms = ["non capisco", "cosa significa", "come faccio", "non so dove"]

        user_input_lower = user_input.lower()

        if any(term in user_input_lower for term in technical_terms):
            conversation["user_skill_level"] = "advanced"
        elif any(term in user_input_lower for term in beginner_terms):
            conversation["user_skill_level"] = "beginner"
        # Altrimenti rimane "intermediate"

    async def _provide_step_clarification(self, session_id: str, user_input: str,
                                          current_step: ConversationStep) -> str:
        """Fornisce chiarimenti per step corrente"""
        try:
            llm = await self.model_manager.get_llm("general_conversation")

            clarification_prompt = f"""
            L'utente ha difficoltà con questo passaggio:
            Step: {current_step.content}
            Domanda utente: {user_input}

            Fornisci una spiegazione più dettagliata e semplice in italiano.
            Se necessario, suggerisci alternative o passaggi intermedi.
            """

            response = await llm.chat(clarification_prompt)
            return response.choices[0].message.content

        except Exception as e:
            self.logger.error(f"Errore generazione chiarimento: {e}")
            return f"Capisco che hai difficoltà con questo passaggio. Proviamo insieme: {current_step.content}. Dimmi esattamente cosa non riesci a fare."

    async def _suggest_alternative_solution(self, session_id: str, user_input: str) -> str:
        """Suggerisce soluzioni alternative"""
        conversation = self.conversations[session_id]
        problem_context = conversation["problem_context"]

        return f"Ho capito che il problema persiste. Proviamo un approccio diverso per {problem_context.category}. Puoi descrivermi esattamente cosa succede adesso?"

    async def _refine_problem_diagnosis(self, session_id: str, screenshot_analysis: Dict):
        """Affina diagnosi basandosi su screenshot"""
        conversation = self.conversations[session_id]
        problem_context = conversation["problem_context"]

        # Incorpora info screenshot nella diagnosi
        if screenshot_analysis.get("detected_issues"):
            problem_context.symptoms.extend(screenshot_analysis["detected_issues"])

        # Aggiorna severità se screenshot mostra errori critici
        if any("error" in issue.lower() for issue in screenshot_analysis.get("detected_issues", [])):
            if problem_context.severity == "low":
                problem_context.severity = "medium"

    def get_conversation_summary(self, session_id: str) -> Dict:
        """Ottiene riassunto conversazione"""
        if session_id not in self.conversations:
            return {}

        conversation = self.conversations[session_id]
        completed_steps = [step for step in conversation.get("steps", []) if step.completed]

        return {
            "session_id": session_id,
            "state": conversation["state"].value,
            "problem_context": asdict(conversation["problem_context"]) if conversation["problem_context"] else None,
            "total_steps": len(conversation.get("steps", [])),
            "completed_steps": len(completed_steps),
            "current_step": conversation.get("current_step", 0),
            "start_time": conversation["start_time"].isoformat(),
            "user_skill_level": conversation.get("user_skill_level", "intermediate"),
            "session_duration": (datetime.now() - conversation["start_time"]).seconds
        }

    def cleanup_conversation(self, session_id: str):
        """Pulisce conversazione completata"""
        if session_id in self.conversations:
            del self.conversations[session_id]
            self.logger.info(f"Conversazione {session_id} pulita")