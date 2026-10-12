import asyncio
import io
import wave

import pytest

from provider_siliconflow_test import _FakeSession, _wav_bytes
from astrbot_plugin_tts_emotion_router.tts import provider_siliconflow as sf


@pytest.mark.parametrize('damage', ['none', 'truncated', 'header'])
@pytest.mark.parametrize('bad_response', [False, True])
def test_pending_wav_recovery(tmp_path, monkeypatch, damage, bad_response):
    good = _wav_bytes()
    session = _FakeSession(good)
    cache = tmp_path / 'cache'
    append = sf.append_audio_silence

    def client():
        obj = sf.SiliconFlowTTS(
            'https://example.test/v1', 'test', 'CosyVoice',
            fmt='wav', max_retries=1)
        obj._session = session
        return obj

    async def unavailable(*args, **kwargs):
        raise RuntimeError('temporary padding failure')

    async def run():
        monkeypatch.setattr(sf, 'append_audio_silence', unavailable)
        assert await client().synth('text', 'voice', cache) is None
        pending = next((cache / '.pending').iterdir())
        assert pending.read_bytes() == good
        assert len(session.posts) == 1
        if damage == 'truncated':
            pending.write_bytes(good[:-len(good) // 3])
        elif damage == 'header':
            pending.write_bytes(b'BROKEN!!' + good[8:])
        if bad_response:
            from provider_siliconflow_test import _FakeResponse
            original = session.post

            def post(*args, **kwargs):
                original(*args, **kwargs)
                return _FakeResponse(good[:-len(good) // 3])

            session.post = post
        monkeypatch.setattr(sf, 'append_audio_silence', append)
        output = await client().synth('text', 'voice', cache)
        assert not pending.exists()
        if damage != 'none' and bad_response:
            assert output is None
            assert len(session.posts) == 3
            assert not list(cache.rglob('*.wav'))
            return
        assert output is not None
        assert len(session.posts) == (1 if damage == 'none' else 2)
        assert await sf.validate_audio_file(output, expected_format='wav', strict_format=True)
        with wave.open(io.BytesIO(good), 'rb') as source:
            frames = source.readframes(source.getnframes())
            expected_frames = source.getnframes() + round(source.getframerate() * .35)
        with wave.open(str(output), 'rb') as result:
            assert result.getnframes() == expected_frames
            assert result.readframes(result.getnframes()) == frames + bytes(
                round(result.getframerate() * .35) * result.getnchannels() * result.getsampwidth())
        assert await client().synth('text', 'voice', cache) == output
        assert len(session.posts) == (1 if damage == 'none' else 2)

    asyncio.run(run())
