import asyncio
import subprocess

import pytest

from astrbot_plugin_tts_emotion_router.core.diagnostics import DecoderUnavailable
from astrbot_plugin_tts_emotion_router.tts import provider_siliconflow as sf
from provider_siliconflow_test import _FakeSession, _FakeResponse, _wav_bytes


@pytest.fixture(params=['mp3', 'opus'])
def audio(request, tmp_path):
    fmt = request.param
    source = tmp_path / 'source.wav'
    source.write_bytes(_wav_bytes())
    target = tmp_path / ('source.' + fmt)
    subprocess.run([sf.check_decoder(), '-v', 'error', '-y', '-i', str(source),
                    str(target)], check=True, capture_output=True)
    broken = (b'ID3' if fmt == 'mp3' else b'OggS') + bytes(4093)
    return fmt, target.read_bytes(), broken


def client(session, fmt, retries=1):
    obj = sf.SiliconFlowTTS('https://example.test/v1', 'test', 'CosyVoice',
                          fmt=fmt, max_retries=retries)
    obj._session = session
    return obj


def test_bad_response_retries_then_recovers(audio, tmp_path):
    fmt, good, bad = audio
    session = _FakeSession(bad)
    original = session.post
    def post(*args, **kwargs):
        original(*args, **kwargs)
        return _FakeResponse(bad if len(session.posts) == 1 else good)
    session.post = post
    output = asyncio.run(client(session, fmt).synth('text', 'voice', tmp_path / 'cache'))
    assert output is not None
    assert sf.decode_duration(output, fmt) > .35
    assert len(session.posts) == 2
    assert not list((output.parent / '.pending').iterdir())


def test_bad_responses_have_bounded_retries(audio, tmp_path):
    fmt, _, bad = audio
    session = _FakeSession(bad)
    cache = tmp_path / 'cache'
    assert asyncio.run(client(session, fmt).synth('text', 'voice', cache)) is None
    assert len(session.posts) == 2
    assert not list(cache.rglob('*.' + fmt))


@pytest.mark.parametrize('corrupt_pending', [False, True])
def test_pending_decoder_failure_preserves_then_recovers(audio, tmp_path, monkeypatch, corrupt_pending):
    fmt, good, bad = audio
    session = _FakeSession(good)
    cache = tmp_path / 'cache'
    decode = sf.decode_duration
    def unavailable(*args):
        raise DecoderUnavailable('decoder temporarily unavailable')
    monkeypatch.setattr(sf, 'decode_duration', unavailable)
    async def run():
        assert await client(session, fmt).synth('text', 'voice', cache) is None
        pending = next((cache / '.pending').iterdir())
        assert pending.read_bytes() == good
        if corrupt_pending:
            pending.write_bytes(bad)
        assert await client(session, fmt).synth('text', 'voice', cache) is None
        assert len(session.posts) == 1
        monkeypatch.setattr(sf, 'decode_duration', decode)
        output = await client(session, fmt).synth('text', 'voice', cache)
        assert output is not None
        assert decode(output, fmt) > .35
        assert len(session.posts) == (2 if corrupt_pending else 1)
        assert not pending.exists()
    asyncio.run(run())
