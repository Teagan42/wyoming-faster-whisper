"""Event handler for clients of the server."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import faster_whisper

from .const import Transcriber

_LOGGER = logging.getLogger(__name__)


class FasterWhisperTranscriber(Transcriber):
    """Event handler for clients."""

    def __init__(
        self,
        model_id: str,
        cache_dir: Union[str, Path],
        device: str = "cpu",
        compute_type: str = "default",
        cpu_threads: int = 4,
        vad_parameters: Optional[Dict[str, Any]] = None,
        task: Optional[str] = None,
    ) -> None:
        self.vad_filter = vad_parameters is not None
        self.vad_parameters = vad_parameters
        self.task = task

        self.model = faster_whisper.WhisperModel(
            model_id,
            download_root=str(cache_dir),
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
        )

    def transcribe(
        self,
        wav_path: Union[str, Path],
        language: Optional[str],
        beam_size: int = 5,
        initial_prompt: Optional[str] = None,
    ) -> str:

        kwargs = {
            "beam_size": beam_size,
            "language": language,
            "initial_prompt": initial_prompt,
            "vad_filter": self.vad_filter,
            "vad_parameters": self.vad_parameters,
        }
        if self.task:
            kwargs["task"] = self.task
            self._wav_file.writeframes(chunk.audio)
            return True

        if AudioStop.is_type(event.type):
            _LOGGER.debug(
                "Audio stopped. Transcribing with initial prompt=%s",
                self.initial_prompt,
            )
            assert self._wav_file is not None

            self._wav_file.close()
            self._wav_file = None

            async with self.model_lock:
                segments, _info = self.model.transcribe(
                    self._wav_path,
                    beam_size=self.cli_args.beam_size,
                    language=self._language,
                    initial_prompt=self.initial_prompt,
                    best_of=self.cli_args.best_of,
                    vad_filter=self.cli_args.vad_filter,
                    without_timestamps=self.cli_args.without_timestamps,
                )

            text = " ".join(segment.text for segment in segments)
            _LOGGER.info(text)

            await self.write_event(Transcript(text=text).event())
            _LOGGER.debug("Completed request")

            # Reset
            self._language = self.cli_args.language

            return False

        if Transcribe.is_type(event.type):
            transcribe = Transcribe.from_event(event)
            if transcribe.language:
                self._language = transcribe.language
                _LOGGER.debug("Language set to %s", transcribe.language)
            return True

        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        return True
