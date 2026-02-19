from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

@dataclass
class ModelResult:
    score_0_1: float
    details: Dict[str, Any]

class OptionalModelHooks:
    def detect_voice_clone(self, wav_path: Path) -> Optional[ModelResult]:
        return None

    def detect_image_synthetic(self, image_path: Path) -> Optional[ModelResult]:
        return None

    def detect_document_synthetic(self, pdf_path: Path) -> Optional[ModelResult]:
        return None
