"""Process-local single-flight and atomic, decoded audio caches."""
import asyncio
import functools
import hashlib
import inspect
import json
import os
import tempfile
import weakref
from pathlib import Path

from ..core.diagnostics import DecoderUnavailable, check_decoder, decode_duration
from .errors import failure, set_failure


def audio_cache(method):
    signature = inspect.signature(method)

    @functools.wraps(method)
    async def wrapped(self, *args, **kwargs):
        failure.set(None)
        try:
            await asyncio.to_thread(check_decoder)
        except DecoderUnavailable as exc:
            set_failure("decoder", str(exc))
            return None
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        arguments.pop("self")
        root = Path(arguments.pop("out_dir"))
        # Snapshot all persistent synthesis options, including style/director fields.
        settings = {k: v for k, v in vars(self).items()
                    if not k.startswith("_") and k not in ("api_key", "last_response_meta")}
        key = hashlib.sha256(json.dumps(
            [type(self).__name__, settings, arguments], sort_keys=True,
            ensure_ascii=False, default=str).encode("utf-8")).hexdigest()
        root.mkdir(parents=True, exist_ok=True)
        target = root / f"v3-{key}.{self.format}"
        if not hasattr(self, "_audio_locks"):
            self._audio_locks = weakref.WeakValueDictionary()
        lock = self._audio_locks.get(str(target))
        if lock is None:
            lock = asyncio.Lock()
            self._audio_locks[str(target)] = lock

        async def valid(path):
            try:
                await asyncio.to_thread(decode_duration, path, self.format,
                                        getattr(self, "sample_rate", 0), getattr(self, "channel", 1))
                return True
            except DecoderUnavailable as exc:
                set_failure("decoder", str(exc))
                return None
            except OSError:
                set_failure("cache_io", "无法读取音频缓存，请检查目录权限和磁盘状态")
                return None
            except ValueError:
                set_failure("audio_invalid", "音频无法完整解码，请检查服务商格式与采样率")
                return False

        async with lock:
            if target.exists():
                state = await valid(target)
                if state is None:
                    return None
                if state:
                    target.touch()
                    return target
                target.unlink(missing_ok=True)
            with tempfile.TemporaryDirectory(prefix=".synth-", dir=root) as directory:
                bound.arguments["out_dir"] = Path(directory)
                result = await method(*bound.args, **bound.kwargs)
                if result is None:
                    return None
                state = await valid(Path(result))
                if state is False:
                    return None
                # Keep unverified audio on local decoder failure to avoid billing again.
                os.replace(result, target)
                if state is None:
                    return None
                failure.set(None)
                return target
    return wrapped
