"""
UI Detection Module per IT Support Agent
Rileva e identifica elementi dell'interfaccia utente negli screenshot
"""
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import pytesseract
from PIL import Image
import logging

logger = logging.getLogger(__name__)


@dataclass
class UIElement:
    """Rappresenta un elemento UI rilevato"""
    type: str  # button, text, window, menu, etc.
    bounds: Tuple[int, int, int, int]  # x, y, width, height
    text: str
    confidence: float
    attributes: Dict[str, Any]


class UIDetector:
    """Rileva elementi UI negli screenshot per guidare l'utente"""

    def __init__(self):
        self.cascade_classifiers = self._load_classifiers()
        self.ocr_config = '--oem 3 --psm 6'

    def _load_classifiers(self) -> Dict[str, cv2.CascadeClassifier]:
        """Carica i classificatori per il riconoscimento UI"""
        classifiers = {}

        # Proviamo a caricare i classificatori standard di OpenCV
        try:
            # Questi sono classificatori standard che potrebbero essere utili
            classifier_files = {
                'face': 'haarcascade_frontalface_default.xml',
                'eye': 'haarcascade_eye.xml'
            }

            for name, filename in classifier_files.items():
                try:
                    classifier = cv2.CascadeClassifier(cv2.data.haarcascades + filename)
                    if not classifier.empty():
                        classifiers[name] = classifier
                except Exception as e:
                    logger.warning(f"Could not load classifier {name}: {e}")

        except Exception as e:
            logger.error(f"Error loading classifiers: {e}")

        return classifiers

    def detect_elements(self, image: np.ndarray) -> List[UIElement]:
        """Rileva tutti gli elementi UI nell'immagine"""
        elements = []

        # Converti in grayscale per alcuni algoritmi
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Rileva bottoni e elementi rettangolari
        elements.extend(self._detect_buttons(image, gray))

        # Rileva testo
        elements.extend(self._detect_text(image))

        # Rileva finestre
        elements.extend(self._detect_windows(image, gray))

        # Rileva menu e dropdown
        elements.extend(self._detect_menus(image, gray))

        return elements

    def _detect_buttons(self, image: np.ndarray, gray: np.ndarray) -> List[UIElement]:
        """Rileva bottoni e elementi clickabili"""
        elements = []

        # Usa edge detection per trovare forme rettangolari
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Trova contorni
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Approssima il contorno
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Se è un rettangolo o quadrato
            if len(approx) >= 4:
                x, y, w, h = cv2.boundingRect(contour)

                # Filtra per dimensioni ragionevoli per un bottone
                if 20 <= w <= 300 and 15 <= h <= 100:
                    # Estrai la regione
                    roi = image[y:y + h, x:x + w]

                    # Prova a leggere il testo nel bottone
                    text = self._extract_text_from_region(roi)

                    # Calcola confidence basato su vari fattori
                    aspect_ratio = w / h
                    area = w * h

                    # I bottoni tendono ad avere certe proporzioni
                    confidence = 0.5
                    if 1.5 <= aspect_ratio <= 8:  # Bottoni tipicamente larghi
                        confidence += 0.2
                    if 400 <= area <= 15000:  # Area ragionevole
                        confidence += 0.2
                    if text and len(text.strip()) > 0:  # Ha testo
                        confidence += 0.3

                    if confidence > 0.6:
                        elements.append(UIElement(
                            type="button",
                            bounds=(x, y, w, h),
                            text=text.strip(),
                            confidence=confidence,
                            attributes={
                                "area": area,
                                "aspect_ratio": aspect_ratio
                            }
                        ))

        return elements

    def _detect_text(self, image: np.ndarray) -> List[UIElement]:
        """Rileva elementi di testo usando OCR"""
        elements = []

        try:
            # Usa pytesseract per rilevare testo con bounding boxes
            data = pytesseract.image_to_data(image, config=self.ocr_config, output_type=pytesseract.Output.DICT)

            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                conf = int(data['conf'][i])

                if text and conf > 30:  # Solo testo con confidence ragionevole
                    x = data['left'][i]
                    y = data['top'][i]
                    w = data['width'][i]
                    h = data['height'][i]

                    elements.append(UIElement(
                        type="text",
                        bounds=(x, y, w, h),
                        text=text,
                        confidence=conf / 100.0,
                        attributes={
                            "word_level": True,
                            "block_num": data['block_num'][i],
                            "par_num": data['par_num'][i]
                        }
                    ))

        except Exception as e:
            logger.error(f"OCR detection failed: {e}")

        return elements

    def _detect_windows(self, image: np.ndarray, gray: np.ndarray) -> List[UIElement]:
        """Rileva finestre e dialoghi"""
        elements = []

        # Cerca rettangoli grandi che potrebbero essere finestre
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        height, width = image.shape[:2]

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Finestre sono tipicamente grandi
            if w > width * 0.2 and h > height * 0.2:
                # Verifica se ha una title bar (area superiore con testo)
                title_bar_roi = image[y:y + min(30, h // 4), x:x + w]
                title_text = self._extract_text_from_region(title_bar_roi)

                confidence = 0.3
                if title_text and len(title_text.strip()) > 0:
                    confidence += 0.4
                if w > width * 0.4 and h > height * 0.4:  # Finestra grande
                    confidence += 0.3

                if confidence > 0.5:
                    elements.append(UIElement(
                        type="window",
                        bounds=(x, y, w, h),
                        text=title_text.strip(),
                        confidence=confidence,
                        attributes={
                            "is_main_window": w > width * 0.6 and h > height * 0.6
                        }
                    ))

        return elements

    def _detect_menus(self, image: np.ndarray, gray: np.ndarray) -> List[UIElement]:
        """Rileva menu e dropdown"""
        elements = []

        # Cerca aree che potrebbero essere menu (liste verticali di elementi)
        height, width = image.shape[:2]

        # Usa template matching per pattern comuni di menu
        # Questo è un approccio semplificato

        # Cerca righe orizzontali che potrebbero separare elementi di menu
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)

        contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Raggruppa linee vicine che potrebbero formare un menu
        menu_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 50 and h < 5:  # Linea orizzontale
                menu_regions.append((x, y, w, h))

        # Analizza le regioni per trovare possibili menu
        for i, (x, y, w, h) in enumerate(menu_regions):
            # Cerca altre linee vicine
            nearby_lines = []
            for j, (x2, y2, w2, h2) in enumerate(menu_regions):
                if i != j and abs(x - x2) < 20 and abs(y - y2) < 100:
                    nearby_lines.append((x2, y2, w2, h2))

            if len(nearby_lines) >= 2:  # Almeno 3 linee totali = possibile menu
                # Calcola bounds del menu
                all_lines = [(x, y, w, h)] + nearby_lines
                min_x = min(line[0] for line in all_lines)
                min_y = min(line[1] for line in all_lines)
                max_x = max(line[0] + line[2] for line in all_lines)
                max_y = max(line[1] + line[3] for line in all_lines)

                menu_w = max_x - min_x
                menu_h = max_y - min_y

                # Estrai testo dalla regione del menu
                menu_roi = image[min_y:max_y, min_x:max_x]
                menu_text = self._extract_text_from_region(menu_roi)

                elements.append(UIElement(
                    type="menu",
                    bounds=(min_x, min_y, menu_w, menu_h),
                    text=menu_text.strip(),
                    confidence=0.6,
                    attributes={
                        "line_count": len(all_lines)
                    }
                ))

        return elements

    def _extract_text_from_region(self, roi: np.ndarray) -> str:
        """Estrae testo da una regione specifica"""
        try:
            if roi.size == 0:
                return ""

            # Migliora il contrasto per OCR
            if len(roi.shape) == 3:
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                roi_gray = roi

            # Applica threshold per migliorare OCR
            _, roi_thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # OCR con configurazione ottimizzata per piccole regioni
            text = pytesseract.image_to_string(roi_thresh,
                                               config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 ')

            return text.strip()

        except Exception as e:
            logger.debug(f"Text extraction failed: {e}")
            return ""

    def find_element_by_text(self, elements: List[UIElement], search_text: str,
                             element_type: Optional[str] = None) -> Optional[UIElement]:
        """Trova un elemento specifico per testo"""
        search_text = search_text.lower().strip()

        best_match = None
        best_score = 0

        for element in elements:
            if element_type and element.type != element_type:
                continue

            element_text = element.text.lower().strip()

            # Exact match
            if element_text == search_text:
                return element

            # Partial match
            if search_text in element_text:
                score = len(search_text) / len(element_text)
                if score > best_score:
                    best_score = score
                    best_match = element

            # Reverse partial match
            elif element_text in search_text:
                score = len(element_text) / len(search_text)
                if score > best_score:
                    best_score = score
                    best_match = element

        return best_match if best_score > 0.3 else None

    def get_element_center(self, element: UIElement) -> Tuple[int, int]:
        """Calcola il centro di un elemento UI"""
        x, y, w, h = element.bounds
        return (x + w // 2, y + h // 2)

    def annotate_elements(self, image: np.ndarray, elements: List[UIElement]) -> np.ndarray:
        """Annota gli elementi rilevati sull'immagine"""
        annotated = image.copy()

        colors = {
            'button': (0, 255, 0),  # Verde
            'text': (255, 0, 0),  # Rosso
            'window': (0, 0, 255),  # Blu
            'menu': (255, 255, 0)  # Giallo
        }

        for i, element in enumerate(elements):
            x, y, w, h = element.bounds
            color = colors.get(element.type, (128, 128, 128))

            # Disegna il rettangolo
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            # Aggiungi label
            label = f"{element.type}_{i + 1}"
            if element.text:
                label += f": {element.text[:20]}"

            # Background per il testo
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x, y - text_height - 10), (x + text_width, y), color, -1)

            # Testo
            cv2.putText(annotated, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return annotated

    def describe_layout(self, elements: List[UIElement]) -> str:
        """Genera una descrizione testuale del layout"""
        if not elements:
            return "No UI elements detected in the image."

        # Raggruppa per tipo
        by_type = {}
        for element in elements:
            if element.type not in by_type:
                by_type[element.type] = []
            by_type[element.type].append(element)

        description = "Detected UI elements:\n"

        for element_type, element_list in by_type.items():
            description += f"\n{element_type.title()}s ({len(element_list)}):\n"

            for i, element in enumerate(element_list[:5]):  # Limite a 5 per tipo
                x, y, w, h = element.bounds
                text_part = f" - '{element.text}'" if element.text else ""
                description += f"  {i + 1}. At position ({x}, {y}), size {w}x{h}{text_part}\n"

            if len(element_list) > 5:
                description += f"  ... and {len(element_list) - 5} more\n"

        return description