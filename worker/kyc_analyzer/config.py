from pydantic import BaseModel, Field

class Weights(BaseModel):
    integrity: float = 0.162
    metadata: float = 0.108
    ocr_semantic: float = 0.162
    tamper: float = 0.162
    print_recapture: float = 0.090
    synthetic: float = 0.108
    correlation: float = 0.108
    liveness: float = 0.100

class Thresholds(BaseModel):
    pass_max: int = 30
    review_max: int = 60

class AnalyzerConfig(BaseModel):
    weights: Weights = Field(default_factory=Weights)
    thresholds: Thresholds = Field(default_factory=Thresholds)

    tesseract_lang: str = "eng"
    ela_quality: int = 90
    audio_sr: int = 16000

    phash_size: int = 32
    phash_highfreq: int = 8
