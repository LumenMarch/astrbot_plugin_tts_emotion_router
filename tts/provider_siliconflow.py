import asyncio
import hashlib
import json
import logging
import os
import uuid
import weakref
from pathlib import Path
from typing import Optional

import aiohttp

from ..utils.audio import append_audio_silence, validate_audio_file
from ..core.diagnostics import check_decoder, decode_duration, DecoderUnavailable
from .errors import http_failure, set_failure


COSYVOICE_TAIL_PADDING_MS = 350
CACHE_SCHEMA_VERSION = 3


class SiliconFlowTTS:
    def __init__(
        self,
        api_url: str,
        api_key: str,
        model: str,
        fmt: str = "mp3",
        speed: float = 1.0,
        max_retries: int = 2,
        timeout: int = 30,
        *,
        gain: float = 5.0,
        sample_rate: Optional[int] = None,
    ):
        self.api_url = (api_url or "").rstrip("/")
        self.api_key = api_key or ""
        self.model = model
        self.format = fmt
        self.speed = speed
        self.max_retries = max_retries
        self.timeout = timeout
        self.gain = gain
        self.sample_rate = sample_rate
        self._session: Optional[aiohttp.ClientSession] = None
        self._key_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )

    async def close(self):
        """关闭 HTTP 会话"""
        if self._session:
            await self._session.close()
            self._session = None

    def _is_audio_response(self, content_type: str) -> bool:
        ct = content_type.lower()
        return ct.startswith("audio/") or ct.startswith("application/octet-stream")

    async def synth(
        self,
        text: str,
        voice: str,
        out_dir: Path,
        speed: Optional[float] = None,
        *,
        emotion: Optional[str] = None,
    ) -> Optional[Path]:
        _ = emotion
        out_dir.mkdir(parents=True, exist_ok=True)

        if not self.api_url or not self.api_key:
            logging.error("SiliconFlowTTS: 缺少 api_url 或 api_key")
            return None

        request_text = str(text or "")
        is_cosyvoice = "cosyvoice" in str(self.model or "").lower()
        tail_padding_ms = COSYVOICE_TAIL_PADDING_MS if is_cosyvoice else 0
        if is_cosyvoice and request_text:
            request_text = request_text.strip()
            closing_chars = "\"'”’」』）》】）)]}"
            suffix = ""
            while request_text and request_text[-1] in closing_chars:
                suffix = request_text[-1] + suffix
                request_text = request_text[:-1].rstrip()
            if request_text and request_text[-1] not in "。！？!?；;.…":
                request_text += "。"
            request_text += suffix

        eff_speed = float(speed) if speed is not None else float(self.speed)

        key = hashlib.sha256(
            json.dumps(
                {
                    "cache_schema": CACHE_SCHEMA_VERSION,
                    "endpoint": self.api_url,
                    "t": request_text,
                    "v": voice,
                    "m": self.model,
                    "s": eff_speed,
                    "f": self.format,
                    "g": self.gain,
                    "sr": self.sample_rate,
                    "tail_padding_ms": tail_padding_ms,
                },
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()[:16]
        out_path = out_dir / f"{key}.{self.format}"
        key_lock = self._key_locks.get(key)
        if key_lock is None:
            key_lock = asyncio.Lock()
            self._key_locks[key] = key_lock

        async with key_lock:
            if out_path.exists() and out_path.stat().st_size > 0:
                if await validate_audio_file(
                    out_path,
                    expected_format=self.format,
                    strict_format=True,
                ):
                    out_path.touch()
                    return out_path
                await asyncio.to_thread(out_path.unlink, missing_ok=True)

            pending_path = out_dir / '.pending' / out_path.name

            async def finish_pending():
                padded = out_dir / f'.{key}.{uuid.uuid4().hex}.padded.{self.format}'
                try:
                    if self.format == 'wav' and not await validate_audio_file(
                        pending_path, expected_format='wav', strict_format=True
                    ):
                        await asyncio.to_thread(pending_path.unlink, missing_ok=True)
                        set_failure('audio', '供应商音频损坏，请重试或检查模型与音频格式')
                        return None
                    if self.format in ('mp3', 'opus'):
                        try:
                            await asyncio.to_thread(decode_duration, pending_path, self.format)
                        except DecoderUnavailable:
                            raise
                        except ValueError:
                            await asyncio.to_thread(pending_path.unlink, missing_ok=True)
                            set_failure('audio', '供应商音频损坏，请重试或检查模型与音频格式')
                            return None
                    ready = await append_audio_silence(
                        pending_path, padded, tail_padding_ms,
                        audio_format=self.format, sample_rate=self.sample_rate)
                    if not await validate_audio_file(ready, expected_format=self.format, strict_format=True):
                        raise ValueError('invalid padded audio')
                    await asyncio.to_thread(os.replace, ready, out_path)
                    await asyncio.to_thread(pending_path.unlink, missing_ok=True)
                    return out_path
                except DecoderUnavailable as exc:
                    set_failure('decoder', str(exc))
                    return None
                except Exception:
                    set_failure('padding', 'CosyVoice 尾音保护失败，已保留原音频；请检查 ffmpeg、格式和采样率后重试')
                    return None
                finally:
                    await asyncio.to_thread(padded.unlink, missing_ok=True)

            if tail_padding_ms:
                if self.format not in ('wav', 'mp3', 'opus', 'pcm') or (self.format == 'pcm' and not self.sample_rate):
                    set_failure('padding', 'CosyVoice 请使用 wav、mp3、opus 或配置采样率的 pcm 格式')
                    return None
                if self.format != 'wav':
                    try:
                        await asyncio.to_thread(check_decoder)
                    except DecoderUnavailable as exc:
                        set_failure('decoder', str(exc))
                        return None
                if pending_path.exists():
                    ready = await finish_pending()
                    if ready is not None or pending_path.exists():
                        return ready

            url = f"{self.api_url}/audio/speech"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model,
                "voice": voice,
                "input": request_text,
                "response_format": self.format,
                "speed": eff_speed,
                "gain": self.gain,
                "stream": False,
            }
            if self.sample_rate:
                payload["sample_rate"] = int(self.sample_rate)

            last_err = None
            backoff = 1.0

            if self._session is None or self._session.closed:
                client_timeout = aiohttp.ClientTimeout(total=self.timeout)
                self._session = aiohttp.ClientSession(timeout=client_timeout)

            for attempt in range(1, self.max_retries + 2):
                raw_path = out_dir / (f".{key}.{uuid.uuid4().hex}.raw.{self.format}")
                padded_path = out_dir / (
                    f".{key}.{uuid.uuid4().hex}.padded.{self.format}"
                )
                try:
                    async with self._session.post(
                        url,
                        headers=headers,
                        json=payload,
                    ) as response:
                        if 200 <= response.status < 300:
                            content_type = response.headers.get("Content-Type", "")
                            if not self._is_audio_response(content_type):
                                try:
                                    err = await response.json()
                                except Exception:
                                    text_content = await response.text()
                                    err = {"error": text_content[:200]}
                                logging.error(
                                    "SiliconFlowTTS: non-audio response, code=%s, detail=%s",
                                    response.status,
                                    "invalid audio response",
                                )
                                last_err = err
                                break

                            content = await response.read()
                            await asyncio.to_thread(raw_path.write_bytes, content)
                            if not await validate_audio_file(
                                raw_path,
                                expected_format=self.format,
                                strict_format=True,
                            ):
                                last_err = {
                                    "error": "Generated audio file validation failed"
                                }
                                continue

                            ready_path = raw_path
                            if tail_padding_ms > 0:
                                pending_path.parent.mkdir(exist_ok=True)
                                await asyncio.to_thread(os.replace, raw_path, pending_path)
                                ready = await finish_pending()
                                if ready is not None or pending_path.exists():
                                    return ready
                                continue

                            await asyncio.to_thread(os.replace, ready_path, out_path)
                            logging.info(
                                "SiliconFlowTTS: generated audio file: %s (%d bytes, tail_padding_ms=%d)",
                                out_path,
                                out_path.stat().st_size,
                                tail_padding_ms,
                            )
                            return out_path

                        try:
                            err_detail = await response.json()
                        except Exception:
                            text_content = await response.text()
                            err_detail = {"error": text_content[:200]}

                        logging.warning(
                            "SiliconFlowTTS: request failed (%s) attempt=%s, detail=%s",
                            response.status,
                            attempt,
                            "request rejected",
                        )
                        http_failure(response.status)
                        last_err = err_detail
                        if response.status in (429,) or 500 <= response.status < 600:
                            if attempt <= self.max_retries:
                                await asyncio.sleep(backoff)
                                backoff = min(backoff * 2, 8)
                                continue
                        break
                except Exception as e:
                    logging.warning(
                        "SiliconFlowTTS: request or audio processing failed attempt=%s, err=%s",
                        attempt,
                        type(e).__name__,
                    )
                    set_failure("network", "连接或处理超时，请检查服务地址、网络和 ffmpeg", True)
                    last_err = type(e).__name__
                    if attempt <= self.max_retries:
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2, 8)
                        continue
                    break
                finally:
                    await asyncio.to_thread(raw_path.unlink, missing_ok=True)
                    await asyncio.to_thread(padded_path.unlink, missing_ok=True)

            await asyncio.to_thread(out_path.unlink, missing_ok=True)
            logging.error(
                "SiliconFlowTTS: synthesis failed after retries, last_error=%s",
                "request/audio failure",
            )
            return None
