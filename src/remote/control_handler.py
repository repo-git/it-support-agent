"""
Remote Control Handler per IT Support Agent
Gestisce il controllo remoto sicuro del desktop dell'utente
"""
import pyautogui
import time
import logging
from typing import Optional, Tuple, Dict, Any, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import json
from pynput import mouse, keyboard
import psutil
import os

logger = logging.getLogger(__name__)


class ControlAction(Enum):
    """Tipi di azioni di controllo remoto"""
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    DRAG = "drag"
    TYPE = "type"
    KEY_PRESS = "key_press"
    KEY_COMBINATION = "key_combination"
    SCROLL = "scroll"
    MOVE_MOUSE = "move_mouse"


class PermissionLevel(Enum):
    """Livelli di permessi per il controllo remoto"""
    NONE = "none"
    VIEW_ONLY = "view_only"
    ASSISTED = "assisted"  # Richiede approvazione per ogni azione
    FULL_CONTROL = "full_control"


@dataclass
class PendingAction:
    """Azione in attesa di approvazione"""
    action_type: ControlAction
    parameters: Dict[str, Any]
    timestamp: datetime
    timeout: int = 30  # secondi
    approved: Optional[bool] = None
    description: str = ""


@dataclass
class ControlSession:
    """Sessione di controllo remoto"""
    session_id: str
    permission_level: PermissionLevel
    start_time: datetime
    last_activity: datetime
    timeout_minutes: int = 30
    actions_log: List[Dict[str, Any]] = field(default_factory=list)
    pending_actions: List[PendingAction] = field(default_factory=list)
    is_active: bool = True


class RemoteControlHandler:
    """Gestisce il controllo remoto sicuro del desktop"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.current_session: Optional[ControlSession] = None
        self.approval_callback: Optional[Callable] = None
        self.activity_callback: Optional[Callable] = None

        # Configurazioni di sicurezza
        self.max_session_duration = self.config.get('max_session_duration', 3600)  # 1 ora
        self.require_approval = self.config.get('require_approval', True)
        self.allowed_apps = self.config.get('allowed_apps', [])
        self.blocked_apps = self.config.get('blocked_apps', ['cmd.exe', 'powershell.exe', 'regedit.exe'])

        # Configurazioni PyAutoGUI
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.1

        # Monitoring thread
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()

    def start_session(self, session_id: str, permission_level: PermissionLevel = PermissionLevel.ASSISTED) -> bool:
        """Avvia una nuova sessione di controllo remoto"""
        try:
            if self.current_session and self.current_session.is_active:
                logger.warning("Session already active, terminating previous session")
                self.end_session()

            self.current_session = ControlSession(
                session_id=session_id,
                permission_level=permission_level,
                start_time=datetime.now(),
                last_activity=datetime.now(),
                timeout_minutes=self.max_session_duration // 60
            )

            # Avvia monitoring
            self._start_monitoring()

            logger.info(f"Remote control session started: {session_id} with level {permission_level.value}")
            return True

        except Exception as e:
            logger.error(f"Failed to start control session: {e}")
            return False

    def end_session(self) -> bool:
        """Termina la sessione di controllo remoto"""
        try:
            if self.current_session:
                self.current_session.is_active = False
                self._stop_monitoring.set()

                if self._monitoring_thread and self._monitoring_thread.is_alive():
                    self._monitoring_thread.join(timeout=5)

                logger.info(f"Remote control session ended: {self.current_session.session_id}")
                self.current_session = None

            return True

        except Exception as e:
            logger.error(f"Failed to end control session: {e}")
            return False

    def _start_monitoring(self):
        """Avvia il thread di monitoring della sessione"""
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(target=self._monitor_session)
        self._monitoring_thread.daemon = True
        self._monitoring_thread.start()

    def _monitor_session(self):
        """Monitora la sessione per timeout e sicurezza"""
        while not self._stop_monitoring.is_set() and self.current_session and self.current_session.is_active:
            try:
                now = datetime.now()

                # Verifica timeout di inattività
                if (now - self.current_session.last_activity).total_seconds() > (
                        self.current_session.timeout_minutes * 60):
                    logger.warning("Session timeout due to inactivity")
                    self.end_session()
                    break

                # Verifica durata massima sessione
                if (now - self.current_session.start_time).total_seconds() > self.max_session_duration:
                    logger.warning("Session timeout due to maximum duration")
                    self.end_session()
                    break

                # Pulisci azioni scadute
                self._cleanup_pending_actions()

                # Verifica processi sospetti
                self._check_security()

                time.sleep(5)  # Check ogni 5 secondi

            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(5)

    def _cleanup_pending_actions(self):
        """Rimuove le azioni in attesa scadute"""
        if not self.current_session:
            return

        now = datetime.now()
        expired_actions = []

        for action in self.current_session.pending_actions:
            if (now - action.timestamp).total_seconds() > action.timeout:
                expired_actions.append(action)

        for action in expired_actions:
            self.current_session.pending_actions.remove(action)
            logger.info(f"Pending action expired: {action.action_type.value}")

    def _check_security(self):
        """Verifica condizioni di sicurezza"""
        try:
            # Verifica processi attivi
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    proc_name = proc.info['name'].lower()
                    if any(blocked in proc_name for blocked in self.blocked_apps):
                        logger.warning(f"Blocked application detected: {proc_name}")
                        # Potremmo terminare la sessione o limitare i permessi

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

        except Exception as e:
            logger.error(f"Security check failed: {e}")

    def execute_action(self, action_type: ControlAction, **kwargs) -> bool:
        """Esegue un'azione di controllo remoto"""
        if not self._validate_session():
            return False

        # Crea l'azione
        action = PendingAction(
            action_type=action_type,
            parameters=kwargs,
            timestamp=datetime.now(),
            description=self._generate_action_description(action_type, kwargs)
        )

        # Verifica permessi
        if not self._check_permission(action):
            return False

        # Esegui l'azione
        success = self._perform_action(action)

        # Log dell'azione
        self._log_action(action, success)

        return success

    def _validate_session(self) -> bool:
        """Valida la sessione corrente"""
        if not self.current_session or not self.current_session.is_active:
            logger.error("No active control session")
            return False

        if self.current_session.permission_level == PermissionLevel.NONE:
            logger.error("No control permissions")
            return False

        return True

    def _check_permission(self, action: PendingAction) -> bool:
        """Verifica i permessi per un'azione"""
        if not self.current_session:
            return False

        permission_level = self.current_session.permission_level

        if permission_level == PermissionLevel.VIEW_ONLY:
            logger.error("View-only mode, action not allowed")
            return False

        if permission_level == PermissionLevel.FULL_CONTROL:
            return True

        if permission_level == PermissionLevel.ASSISTED:
            # Richiede approvazione
            return self._request_approval(action)

        return False

    def _request_approval(self, action: PendingAction) -> bool:
        """Richiede approvazione per un'azione"""
        if not self.approval_callback:
            logger.error("No approval callback configured")
            return False

        self.current_session.pending_actions.append(action)

        try:
            # Chiama il callback per richiedere approvazione
            approved = self.approval_callback(action.description, action.timeout)
            action.approved = approved

            return approved

        except Exception as e:
            logger.error(f"Approval request failed: {e}")
            return False

    def _perform_action(self, action: PendingAction) -> bool:
        """Esegue fisicamente l'azione"""
        try:
            action_type = action.action_type
            params = action.parameters

            if action_type == ControlAction.CLICK:
                x, y = params.get('x', 0), params.get('y', 0)
                button = params.get('button', 'left')
                pyautogui.click(x, y, button=button)

            elif action_type == ControlAction.DOUBLE_CLICK:
                x, y = params.get('x', 0), params.get('y', 0)
                pyautogui.doubleClick(x, y)

            elif action_type == ControlAction.RIGHT_CLICK:
                x, y = params.get('x', 0), params.get('y', 0)
                pyautogui.rightClick(x, y)

            elif action_type == ControlAction.DRAG:
                start_x, start_y = params.get('start_x', 0), params.get('start_y', 0)
                end_x, end_y = params.get('end_x', 0), params.get('end_y', 0)
                duration = params.get('duration', 0.5)
                pyautogui.drag(end_x - start_x, end_y - start_y, duration, start=(start_x, start_y))

            elif action_type == ControlAction.TYPE:
                text = params.get('text', '')
                interval = params.get('interval', 0.01)
                pyautogui.typewrite(text, interval=interval)

            elif action_type == ControlAction.KEY_PRESS:
                key = params.get('key', '')
                presses = params.get('presses', 1)
                interval = params.get('interval', 0.1)
                pyautogui.press(key, presses=presses, interval=interval)

            elif action_type == ControlAction.KEY_COMBINATION:
                keys = params.get('keys', [])
                pyautogui.hotkey(*keys)

            elif action_type == ControlAction.SCROLL:
                x, y = params.get('x', 0), params.get('y', 0)
                clicks = params.get('clicks', 1)
                pyautogui.scroll(clicks, x=x, y=y)

            elif action_type == ControlAction.MOVE_MOUSE:
                x, y = params.get('x', 0), params.get('y', 0)
                duration = params.get('duration', 0.25)
                pyautogui.moveTo(x, y, duration=duration)

            else:
                logger.error(f"Unknown action type: {action_type}")
                return False

            # Aggiorna last activity
            if self.current_session:
                self.current_session.last_activity = datetime.now()

            return True

        except Exception as e:
            logger.error(f"Failed to perform action {action_type}: {e}")
            return False

    def _generate_action_description(self, action_type: ControlAction, params: Dict[str, Any]) -> str:
        """Genera una descrizione leggibile dell'azione"""
        if action_type == ControlAction.CLICK:
            return f"Click at position ({params.get('x', 0)}, {params.get('y', 0)})"
        elif action_type