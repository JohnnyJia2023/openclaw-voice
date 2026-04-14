"""
Text-to-Speech module — Kokoro (local) → Chatterbox (local) → ElevenLabs → OpenAI → mock.
"""

import asyncio
import os
import re
from typing import Optional, AsyncGenerator

import numpy as np
from loguru import logger


class ChatterboxTTS:
    """TTS with cascading backends: Kokoro → Chatterbox → ElevenLabs → OpenAI → mock."""

    def __init__(
        self,
        voice_sample: Optional[str] = None,
        device: str = "auto",
        voice_id: Optional[str] = None,  # ElevenLabs voice ID
    ):
        self.voice_sample = voice_sample
        self.device = device
        self.voice_id = voice_id or "cgSgspJ2msm6clMCkdW9"  # Jessica (ElevenLabs)
        self.model = None
        self._backend = "mock"
        self._elevenlabs_client = None
        self._openai_client = None
        self._kokoro = None
        self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        tts_pref = os.environ.get("OPENCLAW_TTS_MODEL", "").lower()
        openai_key = os.environ.get("OPENAI_API_KEY")
        elevenlabs_key = os.environ.get("ELEVENLABS_API_KEY")

        # 1. OpenAI TTS — only when explicitly requested
        if tts_pref == "openai" and openai_key:
            try:
                from openai import OpenAI
                self._openai_client = OpenAI(api_key=openai_key)
                self._backend = "openai"
                logger.info("✅ OpenAI TTS ready (tts-1, alloy)")
                return
            except Exception as e:
                logger.warning(f"OpenAI TTS init failed: {e}")

        # 2. ElevenLabs — only when explicitly requested OR no local pref set
        if elevenlabs_key and tts_pref not in ("kokoro", "chatterbox", "xtts"):
            try:
                from elevenlabs import ElevenLabs
                self._elevenlabs_client = ElevenLabs(api_key=elevenlabs_key)
                self._backend = "elevenlabs"
                logger.info("✅ ElevenLabs TTS ready")
                return
            except Exception as e:
                logger.warning(f"ElevenLabs failed: {e}")

        # 3. Kokoro — fast local ONNX, no PyTorch required (~500 MB)
        if tts_pref in ("", "kokoro"):
            try:
                import pathlib
                import urllib.request
                from kokoro_onnx import Kokoro as KokoroModel
                logger.info("Loading Kokoro TTS (ONNX)…")
                _BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
                _CACHE = pathlib.Path.home() / ".cache" / "kokoro-onnx"
                _CACHE.mkdir(parents=True, exist_ok=True)
                model_path = _CACHE / "kokoro-v1.0.onnx"
                voices_path = _CACHE / "voices-v1.0.bin"
                if not model_path.exists():
                    logger.info("Downloading kokoro-v1.0.onnx (~310 MB)…")
                    urllib.request.urlretrieve(f"{_BASE}/kokoro-v1.0.onnx", model_path)
                if not voices_path.exists():
                    logger.info("Downloading voices-v1.0.bin (~14 MB)…")
                    urllib.request.urlretrieve(f"{_BASE}/voices-v1.0.bin", voices_path)
                self._kokoro = KokoroModel(str(model_path), str(voices_path))
                self._backend = "kokoro"
                logger.info("✅ Kokoro TTS ready (24 kHz, 54 voices)")
                return
            except ImportError:
                logger.warning("kokoro-onnx not installed — falling back to Chatterbox")
            except Exception as e:
                logger.warning(f"Kokoro failed: {e} — falling back to Chatterbox")

        # 4. Chatterbox — good quality, MIT, runs on MPS
        try:
            from chatterbox.tts import ChatterboxTTS as CBModel
            logger.info("Loading Chatterbox TTS…")
            self.model = CBModel.from_pretrained(device=self._get_device())
            self._backend = "chatterbox"
            logger.info("✅ Chatterbox TTS ready")
            return
        except ImportError:
            logger.warning("Chatterbox not installed")
        except Exception as e:
            logger.warning(f"Chatterbox failed: {e}")

        # 5. XTTS (Coqui)
        try:
            from TTS.api import TTS
            logger.info("Loading Coqui XTTS…")
            self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            self._backend = "xtts"
            logger.info("✅ XTTS ready")
            return
        except ImportError:
            logger.warning("Coqui TTS not installed")
        except Exception as e:
            logger.warning(f"XTTS failed: {e}")

        # 6. Mock
        logger.warning("⚠️  No TTS backend available — using mock (silence)")
        self._backend = "mock"

    def _get_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def synthesize(self, text: str) -> np.ndarray:
        """Synthesize full text → float32 numpy array."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._synthesize_sync, text)

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Stream synthesized audio as raw PCM chunks (24 kHz, 16-bit signed).

        Kokoro and Chatterbox split on sentence boundaries so the first audio
        chunk arrives after the first sentence rather than after the full text.
        """
        if self._backend == "kokoro":
            sentences = _split_sentences(text)
            loop = asyncio.get_event_loop()
            for sentence in sentences:
                if not sentence.strip():
                    continue
                try:
                    samples, _ = await loop.run_in_executor(
                        None,
                        lambda s=sentence: self._kokoro.create(
                            s, voice="af_heart", speed=1.0, lang="en-us"
                        ),
                    )
                    yield _float32_to_pcm16(samples)
                except Exception as e:
                    logger.error(f"Kokoro stream error: {e}")

        elif self._backend == "openai":
            try:
                loop = asyncio.get_event_loop()
                def _call():
                    return self._openai_client.audio.speech.create(
                        model="tts-1", voice="alloy", input=text,
                        response_format="pcm",
                    ).content
                audio_bytes = await loop.run_in_executor(None, _call)
                for i in range(0, len(audio_bytes), 4096):
                    yield audio_bytes[i:i + 4096]
            except Exception as e:
                logger.error(f"OpenAI TTS stream error: {e}")

        elif self._backend == "elevenlabs":
            try:
                gen = self._elevenlabs_client.text_to_speech.convert(
                    voice_id=self.voice_id, text=text,
                    model_id="eleven_turbo_v2_5", output_format="pcm_24000",
                )
                for chunk in gen:
                    yield chunk
            except Exception as e:
                logger.error(f"ElevenLabs stream error: {e}")

        else:
            # Chatterbox / XTTS / mock — synthesize sentence-by-sentence
            sentences = _split_sentences(text)
            loop = asyncio.get_event_loop()
            for sentence in sentences:
                if not sentence.strip():
                    continue
                audio = await loop.run_in_executor(
                    None, self._synthesize_sync, sentence
                )
                yield _float32_to_pcm16(audio)

    # ------------------------------------------------------------------
    # Synchronous synthesis
    # ------------------------------------------------------------------

    def _synthesize_sync(self, text: str) -> np.ndarray:
        if self._backend == "kokoro":
            try:
                samples, _ = self._kokoro.create(
                    text, voice="af_heart", speed=1.0, lang="en-us"
                )
                return np.array(samples, dtype=np.float32)
            except Exception as e:
                logger.error(f"Kokoro synthesis error: {e}")
                return np.zeros(12000, dtype=np.float32)

        elif self._backend == "openai":
            try:
                response = self._openai_client.audio.speech.create(
                    model="tts-1", voice="alloy", input=text,
                    response_format="pcm",
                )
                arr = np.frombuffer(response.content, dtype=np.int16)
                return arr.astype(np.float32) / 32768.0
            except Exception as e:
                logger.error(f"OpenAI TTS error: {e}")
                return np.zeros(16000, dtype=np.float32)

        elif self._backend == "elevenlabs":
            try:
                gen = self._elevenlabs_client.text_to_speech.convert(
                    voice_id=self.voice_id, text=text,
                    model_id="eleven_turbo_v2_5", output_format="pcm_24000",
                )
                audio_bytes = b"".join(gen)
                arr = np.frombuffer(audio_bytes, dtype=np.int16)
                return arr.astype(np.float32) / 32768.0
            except Exception as e:
                logger.error(f"ElevenLabs TTS error: {e}")
                return np.zeros(16000, dtype=np.float32)

        elif self._backend == "chatterbox":
            try:
                if self.voice_sample:
                    audio = self.model.generate(text, audio_prompt=self.voice_sample)
                else:
                    audio = self.model.generate(text)
                return audio.cpu().numpy().astype(np.float32)
            except Exception as e:
                logger.error(f"Chatterbox error: {e}")
                return np.zeros(12000, dtype=np.float32)

        elif self._backend == "xtts":
            try:
                if self.voice_sample:
                    wav = self.model.tts(text=text, speaker_wav=self.voice_sample, language="en")
                else:
                    wav = self.model.tts(text=text, language="en")
                return np.array(wav, dtype=np.float32)
            except Exception as e:
                logger.error(f"XTTS error: {e}")
                return np.zeros(12000, dtype=np.float32)

        else:
            logger.debug(f"Mock TTS: '{text[:50]}'")
            return np.zeros(12000, dtype=np.float32)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """Split text on sentence boundaries for low-latency streaming."""
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]


def _float32_to_pcm16(samples: np.ndarray) -> bytes:
    """Convert float32 [-1, 1] audio to 16-bit signed PCM bytes."""
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16).tobytes()
