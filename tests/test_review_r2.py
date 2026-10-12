import asyncio
import importlib.util
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from astrbot_plugin_tts_emotion_router.core.segment_delivery import pace_records
from astrbot_plugin_tts_emotion_router.core.diagnostics import DecoderUnavailable
from astrbot_plugin_tts_emotion_router.tts import provider_siliconflow as sf
from provider_siliconflow_test import _FakeSession, _wav_bytes
from test_review_r1 import plugin_module


@pytest.mark.parametrize('cancelled', [False, True])
def test_delivery_cleanup_on_cancellation_or_fallback_failure(plugin_module, cancelled):
    class Chain:
        def __init__(self, chain):
            self.chain = chain
        def derive(self, chain):
            return Chain(chain)
    record = object()
    error = asyncio.CancelledError if cancelled else RuntimeError
    sender = AsyncMock(side_effect=error())
    event = SimpleNamespace(send=sender)
    state = SimpleNamespace(last_sent_ts=0)
    pace_records(event, [(record, 0, 'fallback text')], state)
    with pytest.raises(error):
        asyncio.run(event.send(Chain([record])))
    assert event.send is sender
    assert event._tts_segment_delivery['texts'] == []
    assert state.last_sent_ts == 0
    assert sender.await_count == (1 if cancelled else 2)
    if not cancelled:
        assert sender.call_args.args[0].chain[0].text == 'fallback text'


@pytest.mark.parametrize('fault', ['exception', 'invalid'])
def test_pending_audio_survives_retries_and_new_client(monkeypatch, tmp_path, fault):
    session = _FakeSession(_wav_bytes())
    def client():
        obj = sf.SiliconFlowTTS('https://example.test/v1', 'test', 'CosyVoice', fmt='wav')
        obj._session = session
        return obj
    real = sf.append_audio_silence
    async def broken(source, target, *args, **kwargs):
        if fault == 'exception':
            raise OSError('local failure')
        target.write_bytes(b'invalid')
        return target
    monkeypatch.setattr(sf, 'append_audio_silence', broken)
    async def run():
        for _ in range(2):
            assert await client().synth('text', 'voice', tmp_path) is None
        assert len(session.posts) == 1
        assert not list(tmp_path.glob('*.wav'))
        monkeypatch.setattr(sf, 'append_audio_silence', real)
        assert await client().synth('text', 'voice', tmp_path)
        assert len(session.posts) == 1
        assert not list((tmp_path / '.pending').iterdir())
    asyncio.run(run())


def test_missing_decoder_never_requests(monkeypatch, tmp_path):
    def missing():
        raise DecoderUnavailable('missing')
    monkeypatch.setattr(sf, 'check_decoder', missing)
    client = sf.SiliconFlowTTS('https://example.test/v1', 'test', 'CosyVoice', fmt='mp3')
    session = _FakeSession(b'')
    client._session = session
    async def run():
        for _ in range(2):
            assert await client.synth('text', 'voice', tmp_path) is None
        assert session.posts == []
    asyncio.run(run())


@pytest.mark.parametrize('segmented', [False, True])
@pytest.mark.parametrize('fault,failed', [('delivery', None), ('delivery', 0),
    ('delivery', 1), ('delivery', 2), ('synthesis', 0), ('synthesis', 1), ('synthesis', 2)])
def test_actual_host_respond_stage(monkeypatch, plugin_module, segmented, fault, failed):
    # Execute the complete host stage, replacing only its surrounding API dependencies.
    host = Path(__file__).resolve().parents[2] / '../_astrbot/astrbot/core/pipeline/respond/stage.py'
    assert host.is_file(), 'AstrBot source checkout required for lifecycle integration'
    names = 'Plain Face Record Video At Image Reply Poke Node Nodes File Json Share Music Forward Location Contact Shake Dice RPS Unknown'.split()
    class Component:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    comps = {name: type(name, (Component,), {'type': name}) for name in names}
    kinds = SimpleNamespace(**{name: name for name in names})
    class Chain:
        def __init__(self, chain):
            self.chain = chain
            self.result_content_type = 'llm'
        def derive(self, chain):
            return Chain(chain)
        def get_plain_text(self):
            return ''.join(c.text for c in self.chain if c.type == 'Plain')
        def is_model_result(self):
            return True
        def is_llm_result(self):
            return True
    hooks = []
    plugin = SimpleNamespace(_remember_spoken_assistant_text=AsyncMock(),
                             _get_umo=lambda event: 'test', _session_state={})
    async def hook(event, kind):
        hooks.append(kind)
        await plugin_module.TTSEmotionRouter.after_message_sent(plugin, event)
        return False
    modules = {
        'astrbot.core': {'logger': logging.getLogger(__name__)},
        'astrbot.core.message': {},
        'astrbot.core.message.components': dict(comps, BaseMessageComponent=Component, ComponentType=kinds),
        'astrbot.core.message.message_event_result': {'MessageChain': Chain, 'ResultContentType': SimpleNamespace(STREAMING_FINISH='finish', STREAMING_RESULT='stream')},
        'astrbot.core.platform.astr_message_event': {'AstrMessageEvent': object},
        'astrbot.core.star.star_handler': {'EventType': SimpleNamespace(OnAfterMessageSentEvent='after')},
        'astrbot.core.utils.path_util': {'path_Mapping': lambda mappings, path: path},
        'astrbot.core.pipeline': {},
        'astrbot.core.pipeline.context': {'PipelineContext': object, 'call_event_hook': hook},
        'astrbot.core.pipeline.stage': {'Stage': object, 'register_stage': lambda cls: cls},
    }
    for name, attrs in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules['astrbot.core.message'].components = sys.modules['astrbot.core.message.components']
    spec = importlib.util.spec_from_file_location('astrbot.core.pipeline.respond.stage', host)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(plugin_module, 'Plain', comps['Plain'])
    monkeypatch.setattr(plugin_module, 'Record', comps['Record'])
    result = Chain([comps['Plain'](text='0 1 2')])
    sent = []
    async def send(message):
        if fault == 'delivery' and failed is not None and any(
            c.type == 'Record' and c.file == f'{failed}.wav' for c in message.chain
        ):
            raise RuntimeError('delivery failed')
        sent.extend(message.chain)
    event = SimpleNamespace(send=send, get_result=lambda: result, get_extra=lambda k, d=None: d,
        get_sender_name=lambda: 'test', get_sender_id=lambda: 'test',
        _outline_chain=lambda c: '', get_platform_name=lambda: 'test', clear_result=lambda: hooks.append('clear'))
    state = SimpleNamespace(text_voice_enabled=None, consume_pending_direct_speak_text=lambda: '')
    parts = [SimpleNamespace(success=not (fault == 'synthesis' and i == failed),
        audio_path=Path(f'{i}.wav'), segment=SimpleNamespace(text=str(i))) for i in range(3)]
    seg_result = SimpleNamespace(segments=parts, successful_segments=[p for p in parts if p.success])
    plugin.__dict__.update(
        _start_background_tasks=AsyncMock(), _cleanup_stale_inflight=lambda: None,
        _get_session_state=lambda _: state, _normalize_text=lambda t: t,
        _strip_emo_head_many=lambda t: (t, None), _strip_any_visible_markers=lambda t: t,
        _prepare_text_for_tts=lambda t: SimpleNamespace(tts_text=t, display_text=t, links=[], codes=[]),
        _append_references_to_text=lambda t, **kw: t,
        _build_fallback_result_chain=lambda chain, t, **kw: [comps['Plain'](text=t)],
        config=SimpleNamespace(is_voice_output_enabled_for_umo=lambda _: True,
            is_probability_output_enabled_for_umo=lambda _: False,
            is_text_voice_output_enabled_for_umo=lambda _: False,
            is_segmented_output_enabled_for_umo=lambda _: True, validation_errors=lambda: []),
        condition_checker=SimpleNamespace(check_all=lambda *a, **kw: SimpleNamespace(passed=True)),
        _build_inflight_sig=lambda *a: 'sig', _inflight_sigs={}, _recent_reasons={},
        segmented_tts_enabled=True, segmented_min_chars=1,
        segmented_tts_processor=SimpleNamespace(should_use_segmented=lambda *a: True,
            process_only=AsyncMock(return_value=seg_result), _calculate_interval=lambda _: .1),
        tts_processor=SimpleNamespace(normalize_audio_path=str, process=AsyncMock()),
        _get_current_conversation_id=AsyncMock(return_value=None))
    asyncio.run(plugin_module.TTSEmotionRouter.on_decorating_result(plugin, event))
    assert sent == []
    # A later decorator sees all unsent components and can still alter them.
    result.chain.insert(0, comps['Plain'](text='later decorator'))
    stage = module.RespondStage()
    stage.platform_settings = {}
    stage.enable_seg = segmented
    stage.only_llm_result = False
    stage._calc_comp_interval = AsyncMock(return_value=0)
    sleep = AsyncMock()
    monkeypatch.setattr(asyncio, 'sleep', sleep)
    asyncio.run(stage.process(event))
    assert hooks == ['after', 'clear']
    assert event.send == send
    assert not hasattr(event, '_tts_segment_delivery')
    if fault == 'synthesis':
        plugin._remember_spoken_assistant_text.assert_not_called()
        expected = [('Plain', '0 1 2')]
    else:
        plugin._remember_spoken_assistant_text.assert_awaited_once_with(
            event, ' '.join(str(i) for i in range(3) if i != failed), conversation_id=None)
        expected = [('Plain', str(i)) if i == failed else ('Record', f'{i}.wav') for i in range(3)]
    decoration = [('Plain', 'later decorator')]
    assert [(c.type, c.text if c.type == 'Plain' else c.file) for c in sent] == (
        decoration + expected if segmented or fault == 'synthesis' else expected + decoration)
    if failed is None:
        assert any(call.args[0] > 0 for call in sleep.call_args_list)
