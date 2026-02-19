from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple
import json
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

from ..utils import which, run_cmd, sha256_file, ensure_dir
from ..models import OptionalModelHooks

def _flag(code, severity, message, category):
    return {"code": code, "severity": severity, "message": message, "category": category}

def normalize_audio(audio_path: Path, artifacts_dir: Path, target_sr: int = 16000) -> Tuple[Path, List[Dict[str, Any]]]:
    flags=[]; out_wav = artifacts_dir/"normalized.wav"
    if which("ffmpeg"):
        rc,out,err = run_cmd(["ffmpeg","-y","-i",str(audio_path),"-ac","1","-ar",str(target_sr),"-vn",str(out_wav)], timeout=180)
        (artifacts_dir/"ffmpeg_normalize.log").write_text(out+"\n"+err, encoding="utf-8")
        if rc==0 and out_wav.exists():
            return out_wav, flags
        flags.append(_flag("ffmpeg_normalize_fail","medium","FFmpeg normalization failed; using librosa fallback.","integrity"))
    else:
        flags.append(_flag("ffmpeg_missing","skipped","ffmpeg not installed; using librosa fallback.","integrity"))

    try:
        y,sr = librosa.load(str(audio_path), sr=target_sr, mono=True)
        sf.write(str(out_wav), y, target_sr)
        return out_wav, flags
    except Exception as e:
        flags.append(_flag("audio_decode_fail","high",f"Could not decode audio: {e}","integrity"))
        return audio_path, flags

def audio_metadata(audio_path: Path, artifacts_dir: Path) -> Dict[str, Any]:
    flags=[]; meta={"sha256": sha256_file(audio_path), "size_bytes": audio_path.stat().st_size}
    if which("ffprobe"):
        rc,out,err=run_cmd(["ffprobe","-v","error","-print_format","json","-show_format","-show_streams",str(audio_path)], timeout=30)
        (artifacts_dir/"ffprobe.json").write_text(out if out else "{}", encoding="utf-8")
        if rc==0 and out:
            try: meta["ffprobe"]=json.loads(out)
            except Exception: meta["ffprobe"]={}
        else:
            flags.append(_flag("ffprobe_fail","low","ffprobe failed to extract metadata.","metadata"))
    else:
        flags.append(_flag("ffprobe_missing","skipped","ffprobe not installed; skipped audio metadata.","metadata"))
    return {"metadata": meta, "flags": flags}

def silence_and_boundary(y: np.ndarray, sr: int) -> Dict[str, Any]:
    flags=[]
    frame=int(0.02*sr); hop=int(0.01*sr)

    # If the waveform is basically all zeros -> hard silence
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak < 1e-4:  # very safe threshold for "no signal"
        flags.append(_flag("audio_no_signal","high","Audio contains no measurable signal (silent/empty).","integrity"))
        return {"metrics":{"silent_ratio":1.0,"silence_transitions":0,"rms_thr":0.0,"peak_abs":peak}, "flags": flags}

    rms=librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    if rms.size == 0:
        flags.append(_flag("rms_empty","high","Could not compute RMS frames.","integrity"))
        return {"metrics":{"silent_ratio":1.0,"silence_transitions":0,"rms_thr":None,"peak_abs":peak}, "flags": flags}

    # Robust threshold: never allow thr to be 0
    base = float(np.percentile(rms, 10))
    thr = max(base * 1.2, 1e-6)

    # IMPORTANT: use <= so true-zero RMS counts as silent
    silent = (rms <= thr).astype(np.int32)
    silent_ratio=float(np.mean(silent))

    transitions=int(np.sum(np.abs(np.diff(silent))>0))

    if silent_ratio > 0.90:
        flags.append(_flag("mostly_silence","high","Audio is mostly silent.","integrity"))
    elif silent_ratio > 0.60:
        flags.append(_flag("high_silence_ratio","medium","High proportion of near-silence.","integrity"))

    if transitions > 40:
        flags.append(_flag("many_silence_transitions","low","Many silence transitions; could indicate edits or prompted speech.","tamper"))

    return {"metrics":{
        "silent_ratio":silent_ratio,
        "silence_transitions":transitions,
        "rms_thr":thr,
        "peak_abs":peak
    }, "flags": flags}



def splicing_edit_detection(y: np.ndarray, sr: int) -> Dict[str, Any]:
    flags=[]
    frame=int(0.02*sr); hop=int(0.01*sr)
    rms=librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    dr=np.abs(np.diff(rms))
    jump_ratio=float(np.mean(dr > np.percentile(dr, 95))) if dr.size else 0.0
    if jump_ratio>0.10:
        flags.append(_flag("many_energy_jumps","medium","Many abrupt energy jumps; possible splicing/edits (heuristic).","tamper"))
    return {"metrics":{"energy_jump_ratio":jump_ratio}, "flags": flags}

def background_noise_profile(y: np.ndarray, sr: int) -> Dict[str, Any]:
    flags=[]
    frame=int(0.02*sr); hop=int(0.01*sr)
    rms=librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    q = float(np.percentile(rms, 10))
    noise_floor=q
    if noise_floor > float(np.percentile(rms, 50))*0.8:
        flags.append(_flag("high_noise_floor","low","Noise floor is high relative to median energy.","integrity"))
    return {"metrics":{"noise_floor_rms":noise_floor}, "flags": flags}

def spectral_analysis(y: np.ndarray, sr: int) -> Dict[str, Any]:
    flags=[]
    centroid=librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloff=librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    flatness=librosa.feature.spectral_flatness(y=y)[0]
    zcr=librosa.feature.zero_crossing_rate(y)[0]
    metrics={
        "centroid_mean": float(np.mean(centroid)),
        "rolloff_mean": float(np.mean(rolloff)),
        "flatness_mean": float(np.mean(flatness)),
        "zcr_mean": float(np.mean(zcr)),
    }
    if metrics["centroid_mean"] < 800:
        flags.append(_flag("low_bandwidth_audio","low","Low spectral centroid; narrowband/filtered audio.","integrity"))
    return {"metrics": metrics, "flags": flags}

def compression_reencoding(y: np.ndarray, sr: int) -> Dict[str, Any]:
    flags=[]
    mfcc=librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    d=np.diff(mfcc, axis=1)
    var=float(np.mean(np.var(d, axis=1))) if d.size else 0.0
    if var < 5.0:
        flags.append(_flag("possible_heavy_compression","low","Low MFCC-delta variance; could indicate heavy compression.","integrity"))
    return {"metrics":{"mfcc_delta_var":var}, "flags": flags}

def replay_rerecord(y: np.ndarray, sr: int) -> Dict[str, Any]:
    flags=[]; metrics={}
    S=np.abs(librosa.stft(y, n_fft=1024, hop_length=256)) + 1e-9
    logS=np.log(S)
    cep=np.fft.irfft(logS, axis=0)
    q1=int(0.002*sr); q2=int(0.020*sr)
    if q2 < cep.shape[0]:
        peak=float(np.max(cep[q1:q2,:]))
        metrics["cepstrum_peak"]=peak
        if peak > 1.5:
            flags.append(_flag("possible_replay_echo","low","Cepstrum indicates echo/coloration; possible replay/re-record.","print_recapture"))
    else:
        metrics["cepstrum_peak"]=None
    return {"metrics": metrics, "flags": flags}

def voice_embedding(y: np.ndarray, sr: int) -> np.ndarray:
    mfcc=librosa.feature.mfcc(y=y, sr=sr, n_mfcc=24)
    return np.mean(mfcc, axis=1)

def ai_voice_clone(y: np.ndarray, sr: int, wav_path: Path, hooks: OptionalModelHooks | None) -> Dict[str, Any]:
    flags=[]; metrics={}
    try:
        f0=librosa.yin(y, fmin=50, fmax=400, sr=sr)
        f0=f0[np.isfinite(f0)]
        if f0.size:
            f0_std=float(np.std(f0)); f0_range=float(np.percentile(f0,95)-np.percentile(f0,5))
            metrics["f0_std"]=f0_std; metrics["f0_range"]=f0_range
            if f0_std < 5:
                flags.append(_flag("flat_pitch","medium","Very flat pitch; can indicate TTS/voice-clone (or monotone speech).","synthetic"))
    except Exception:
        pass

    if hooks:
        res=hooks.detect_voice_clone(wav_path)
        if res is None:
            flags.append(_flag("voice_clone_model_missing","skipped","No voice anti-spoof model configured; used heuristics only.","synthetic"))
        else:
            metrics["model_score"]=float(res.score_0_1)
            if res.score_0_1 > 0.7:
                flags.append(_flag("voice_clone_model_high","high","Model indicates high likelihood of AI voice/clone.","synthetic"))
    else:
        flags.append(_flag("voice_clone_hooks_none","skipped","No model hooks provided; used heuristics only.","synthetic"))
    return {"metrics": metrics, "flags": flags}

def prosody_speech_dynamics(y: np.ndarray, sr: int) -> Dict[str, Any]:
    flags=[]; metrics={}
    env=librosa.onset.onset_strength(y=y, sr=sr)
    peaks=int(np.sum((env[1:-1] > env[:-2]) & (env[1:-1] > env[2:]) & (env[1:-1] > np.percentile(env, 75))))
    dur=float(len(y)/sr)
    rate=peaks/max(1e-6, dur)
    metrics["onset_peak_rate"]=float(rate)
    if rate < 1.0:
        flags.append(_flag("slow_speech_rate","low","Very low onset peak rate; could be slow speech or synthetic pacing.","synthetic"))
    return {"metrics": metrics, "flags": flags}

def language_semantic_coherence() -> Dict[str, Any]:
    flags=[_flag("asr_not_implemented","skipped","ASR not bundled. Add an ASR engine (e.g., Whisper/Vosk) to enable transcript coherence checks.","ocr_semantic")]
    return {"metrics": {}, "flags": flags}

def save_spectrogram(y: np.ndarray, sr: int, artifacts_dir: Path) -> str:
    S = librosa.stft(y, n_fft=1024, hop_length=256)
    D = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    plt.figure(figsize=(10,4))
    import librosa.display
    librosa.display.specshow(D, sr=sr, hop_length=256, x_axis="time", y_axis="hz")
    plt.colorbar()
    plt.tight_layout()
    out=artifacts_dir/"spectrogram.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return str(out)

def analyze_audio(audio_path: Path, out_dir: Path, target_sr: int = 16000, hooks: OptionalModelHooks | None = None) -> Dict[str, Any]:
    artifacts = ensure_dir(out_dir/"artifacts"/f"audio_{audio_path.stem}")
    meta = audio_metadata(audio_path, artifacts)
    norm_path, norm_flags = normalize_audio(audio_path, artifacts, target_sr=target_sr)

    flags=[]
    flags.extend(meta.get("flags", []))
    flags.extend(norm_flags)

    try:
        y,sr = librosa.load(str(norm_path), sr=target_sr, mono=True)
    except Exception as e:
        flags.append(_flag("audio_load_fail","high",f"Could not load audio: {e}","integrity"))
        return {"path": str(audio_path), "metadata": meta, "analysis": {}, "flags": flags}

    silence = silence_and_boundary(y, sr)
    splice = splicing_edit_detection(y, sr)
    noise = background_noise_profile(y, sr)
    spectral = spectral_analysis(y, sr)
    compress = compression_reencoding(y, sr)
    replay = replay_rerecord(y, sr)
    voice_ai = ai_voice_clone(y, sr, norm_path, hooks)
    prosody = prosody_speech_dynamics(y, sr)
    coherence = language_semantic_coherence()

    try:
        spec_path = save_spectrogram(y, sr, artifacts)
    except Exception:
        spec_path = None

    embed = voice_embedding(y, sr).tolist()

    return {
        "path": str(audio_path),
        "metadata": meta,
        "analysis": {
            "normalized_path": str(norm_path),
            "spectrogram_path": spec_path,
            "silence_boundary": silence,
            "splicing_editing": splice,
            "noise_profile": noise,
            "spectral": spectral,
            "compression": compress,
            "replay_rerecord": replay,
            "voice_clone": voice_ai,
            "prosody": prosody,
            "language_semantic": coherence,
            "embedding_mfcc_mean": embed,
        },
        "flags": flags,
    }
