"""
Modulo per l'analisi delle screenshot condivise dall'utente.
Utilizza modelli di visione AI per comprendere il contenuto dello schermo.
"""

import logging
import base64
import io
import json
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from datetime import datetime

from ..ai.model_manager import ModelManager
from ..utils.config import Config

logger = logging.getLogger(__name__)


class ScreenshotAnalyzer:
    """Analizza screenshot per identificare problemi e elementi UI."""

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.config = Config()
        self.analysis_history = []

    async def analyze_screenshot(self, image_data: bytes, context: str = "") -> Dict[str, Any]:
        """
        Analizza una screenshot per identificare problemi IT e elementi UI.

        Args:
            image_data: Dati dell'immagine in formato bytes
            context: Contesto opzionale fornito dall'utente

        Returns:
            Dizionario con risultati dell'analisi
        """
        try:
            # Converti l'immagine
            image = Image.open(io.BytesIO(image_data))

            # Prepara il prompt per l'analisi
            analysis_prompt = self._create_analysis_prompt(context)

            # Analizza con il modello vision
            analysis_result = await self._analyze_with_vision_model(image, analysis_prompt)

            # Estrai informazioni strutturate
            structured_analysis = self._parse_analysis_result(analysis_result)

            # Aggiungi metadati
            structured_analysis.update({
                'timestamp': datetime.now().isoformat(),
                'image_size': image.size,
                'context': context,
                'analysis_id': f"analysis_{len(self.analysis_history)}"
            })

            # Salva nella cronologia
            self.analysis_history.append(structured_analysis)

            logger.info(f"Screenshot analizzata: {len(structured_analysis.get('issues', []))} problemi identificati")

            return structured_analysis

        except Exception as e:
            logger.error(f"Errore nell'analisi screenshot: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'success': False
            }

    def _create_analysis_prompt(self, context: str) -> str:
        """Crea il prompt per l'analisi dell'immagine."""
        base_prompt = """
        Analizza questa screenshot come un esperto di supporto IT. Identifica:

        1. PROBLEMI EVIDENTI:
           - Messaggi di errore (descrivi il testo esatto)
           - Finestre di dialogo di errore
           - Problemi di prestazioni visibili
           - Problemi di interfaccia

        2. ELEMENTI UI CHIAVE:
           - Pulsanti importanti e loro posizioni
           - Menu e opzioni disponibili
           - Campi di input e form
           - Elementi cliccabili

        3. SISTEMA OPERATIVO E APPLICAZIONI:
           - Sistema operativo rilevato
           - Applicazioni aperte e loro stato
           - Versioni software visibili

        4. RACCOMANDAZIONI:
           - Passi specifici per risolvere problemi
           - Elementi UI da utilizzare
           - Azioni consigliate

        Fornisci una risposta strutturata in formato JSON con le seguenti sezioni:
        - "system_info": informazioni sul sistema
        - "issues": array di problemi identificati
        - "ui_elements": elementi dell'interfaccia utente
        - "recommendations": raccomandazioni specifiche
        - "severity": livello di gravità (low/medium/high/critical)
        """

        if context:
            base_prompt += f"\n\nContesto aggiuntivo dall'utente: {context}"

        return base_prompt

    async def _analyze_with_vision_model(self, image: Image.Image, prompt: str) -> str:
        """Utilizza il modello vision per analizzare l'immagine."""
        try:
            # Converti l'immagine in base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()

            # Utilizza il model manager per l'analisi
            if self.model_manager.current_vision_model:
                result = await self.model_manager.analyze_image(img_base64, prompt)
                return result
            else:
                # Fallback analysis senza AI
                return self._fallback_analysis(image)

        except Exception as e:
            logger.error(f"Errore nell'analisi con modello vision: {e}")
            return self._fallback_analysis(image)

    def _fallback_analysis(self, image: Image.Image) -> str:
        """Analisi di base senza modelli AI."""
        width, height = image.size

        # Converti in OpenCV per analisi base
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Rileva finestre di errore (colori rossi)
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        red_lower = np.array([0, 50, 50])
        red_upper = np.array([10, 255, 255])
        red_mask = cv2.inRange(hsv, red_lower, red_upper)

        red_areas = cv2.countNonZero(red_mask)
        has_red_elements = red_areas > (width * height * 0.01)  # >1% dell'immagine

        fallback_result = {
            "system_info": {
                "screen_resolution": f"{width}x{height}",
                "analysis_method": "fallback"
            },
            "issues": [
                {
                    "type": "possible_error",
                    "description": "Rilevati elementi rossi che potrebbero indicare errori",
                    "confidence": 0.3
                }
            ] if has_red_elements else [],
            "ui_elements": [
                {
                    "type": "screen",
                    "description": f"Schermo {width}x{height}",
                    "position": {"x": 0, "y": 0, "width": width, "height": height}
                }
            ],
            "recommendations": [
                "Condividi una descrizione del problema per un'analisi più accurata",
                "Verifica la presenza di messaggi di errore specifici"
            ],
            "severity": "medium" if has_red_elements else "low"
        }

        return json.dumps(fallback_result)

    def _parse_analysis_result(self, analysis_text: str) -> Dict[str, Any]:
        """Parsing del risultato dell'analisi in formato strutturato."""
        try:
            # Cerca di parsare come JSON
            if analysis_text.strip().startswith('{'):
                return json.loads(analysis_text)

            # Se non è JSON, crea una struttura base
            return {
                'system_info': {'raw_analysis': True},
                'issues': [{'description': analysis_text, 'type': 'general'}],
                'ui_elements': [],
                'recommendations': ['Analisi completata - verifica i dettagli'],
                'severity': 'medium',
                'success': True
            }

        except json.JSONDecodeError:
            logger.warning("Impossibile parsare il risultato come JSON")
            return {
                'system_info': {'parsing_error': True},
                'issues': [{'description': analysis_text, 'type': 'unparsed'}],
                'ui_elements': [],
                'recommendations': ['Verifica manualmente il contenuto'],
                'severity': 'low',
                'success': True
            }

    async def compare_screenshots(self, image1_data: bytes, image2_data: bytes) -> Dict[str, Any]:
        """
        Confronta due screenshot per identificare cambiamenti.

        Args:
            image1_data: Prima immagine
            image2_data: Seconda immagine

        Returns:
            Analisi delle differenze
        """
        try:
            img1 = Image.open(io.BytesIO(image1_data))
            img2 = Image.open(io.BytesIO(image2_data))

            # Ridimensiona le immagini alla stessa dimensione se necessario
            if img1.size != img2.size:
                min_width = min(img1.width, img2.width)
                min_height = min(img1.height, img2.height)
                img1 = img1.resize((min_width, min_height))
                img2 = img2.resize((min_width, min_height))

            # Converti in array numpy
            arr1 = np.array(img1)
            arr2 = np.array(img2)

            # Calcola differenze
            diff = cv2.absdiff(arr1, arr2)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

            # Trova contorni delle differenze
            _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Analizza le differenze significative
            significant_changes = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Solo cambiamenti significativi
                    x, y, w, h = cv2.boundingRect(contour)
                    significant_changes.append({
                        'position': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                        'area': int(area),
                        'type': 'visual_change'
                    })

            total_diff_pixels = np.sum(gray_diff > 30)
            total_pixels = gray_diff.shape[0] * gray_diff.shape[1]
            change_percentage = (total_diff_pixels / total_pixels) * 100

            return {
                'changes_detected': len(significant_changes) > 0,
                'change_percentage': round(change_percentage, 2),
                'significant_changes': significant_changes,
                'total_changes': len(significant_changes),
                'analysis_timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Errore nel confronto screenshot: {e}")
            return {
                'error': str(e),
                'changes_detected': False,
                'change_percentage': 0
            }

    def get_analysis_history(self) -> List[Dict[str, Any]]:
        """Restituisce la cronologia delle analisi."""
        return self.analysis_history.copy()

    def clear_history(self):
        """Pulisce la cronologia delle analisi."""
        self.analysis_history.clear()
        logger.info("Cronologia analisi pulita")

    async def generate_step_by_step_guide(self, analysis: Dict[str, Any]) -> List[str]:
        """
        Genera una guida passo-passo basata sull'analisi.

        Args:
            analysis: Risultato dell'analisi screenshot

        Returns:
            Lista di passi da seguire
        """
        try:
            if not analysis.get('success', True):
                return ["Impossibile generare guida: analisi fallita"]

            steps = []

            # Passi basati sui problemi identificati
            for issue in analysis.get('issues', []):
                issue_type = issue.get('type', 'unknown')
                description = issue.get('description', '')

                if 'error' in issue_type.lower() or 'errore' in description.lower():
                    steps.extend([
                        f"1. Leggi attentamente il messaggio di errore: '{description}'",
                        "2. Annota il codice errore se presente",
                        "3. Chiudi la finestra di errore se possibile"
                    ])

                elif 'performance' in issue_type.lower():
                    steps.extend([
                        "1. Controlla l'utilizzo CPU nel Task Manager",
                        "2. Verifica lo spazio disponibile su disco",
                        "3. Chiudi le applicazioni non necessarie"
                    ])

            # Passi basati sulle raccomandazioni
            for i, recommendation in enumerate(analysis.get('recommendations', []), 1):
                if recommendation not in [step.split('. ', 1)[-1] for step in steps]:
                    steps.append(f"{len(steps) + 1}. {recommendation}")

            # Se non ci sono passi specifici, aggiungi passi generali
            if not steps:
                steps = [
                    "1. Verifica che non ci siano messaggi di errore visibili",
                    "2. Controlla se tutte le applicazioni rispondono correttamente",
                    "3. Riavvia l'applicazione se necessario",
                    "4. Contatta il supporto se il problema persiste"
                ]

            return steps

        except Exception as e:
            logger.error(f"Errore nella generazione della guida: {e}")
            return [
                "1. Verifica la presenza di errori evidenti",
                "2. Riavvia l'applicazione problematica",
                "3. Contatta il supporto tecnico se necessario"
            ]