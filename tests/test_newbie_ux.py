import ast
import asyncio
import copy
import io
import json
import logging
import time
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from astrbot_plugin_tts_emotion_router.core.diagnostics import check_provider, decode_duration
from astrbot_plugin_tts_emotion_router.core.text_splitter import TextSplitter
from astrbot_plugin_tts_emotion_router.tts.cache import audio_cache

ROOT = Path(__file__).resolve().parents[1]


def method(name, **extra):
    tree = ast.parse((ROOT / "main.py").read_text(encoding="utf-8"))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == name)
    node.decorator_list = []
    namespace = dict(AstrMessageEvent=object, GreedyStr=str, Optional=__import__("typing").Optional,
                     __package__='astrbot_plugin_tts_emotion_router',
                     DEFAULT_TEST_TEXT="测试句", filter=SimpleNamespace(after_message_sent=True),
                     copy=copy, time=time, **extra)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / "main.py"), "exec"), namespace)
    return namespace[name]


async def collect(generator):
    return [item async for item in generator]


@pytest.mark.parametrize("text,expected", [("hello world", "hello world"), ("123", "123"),
                                             ("0", "0"), ("", "测试句"), ("  ", "测试句"),
                                             ("中文  123\n测试", "中文  123\n测试")])
def test_say_whole_text(text, expected):
    build = AsyncMock(return_value=(False, [], "stop"))
    event = SimpleNamespace(plain_result=lambda x: x)
    asyncio.run(collect(method("tts_say")(SimpleNamespace(_build_manual_tts_chain=build), event, text)))
    assert build.call_args.args[1] == expected


@pytest.mark.parametrize("name", ["tts_all_on", "tts_all_off"])
def test_global_switch_denied(name):
    event = SimpleNamespace(is_admin=lambda: False, plain_result=lambda x: x)
    assert "管理员" in asyncio.run(collect(method(name)(object(), event)))[0]


def test_session_enable_does_not_reopen_global():
    config = SimpleNamespace(get_feature_policy=lambda _: {"enable": False})
    obj = SimpleNamespace(config=config, _get_umo=lambda _: "session")
    with pytest.raises(ValueError, match="总开关"):
        asyncio.run(method("_switch_voice_output_for_current_umo")(obj, object(), enable=True))


def test_bad_regex_falls_back():
    splitter = TextSplitter(split_pattern="[")
    assert splitter.split("第一句。第二句。")
    assert splitter.split_pattern == splitter.DEFAULT_SPLIT_PATTERN


def wav_bytes():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\x01\x00" * 1600)
    return buffer.getvalue()


def test_decode_duration_rejects_fake_header(tmp_path):
    path = tmp_path / "sample.wav"
    path.write_bytes(wav_bytes())
    assert decode_duration(path, "wav") == pytest.approx(0.1)
    path.write_bytes(b"RIFFxxxxWAVE" + b"x" * 200)
    with pytest.raises(ValueError):
        decode_duration(path, "wav")


def test_selfcheck_synthesizes_and_closes():
    class Client:
        closed = False
        async def synth(self, text, voice, out_dir):
            path = out_dir / "sample.wav"
            path.write_bytes(wav_bytes())
            return path
        async def close(self):
            self.closed = True
    client = Client()
    cfg = dict(provider="siliconflow", key="test", model="test", default_voice="test",
               url="https://example.com/v1", timeout=30, max_retries=0, speed=1, format="wav")
    lines = asyncio.run(check_provider(cfg, lambda _: client))
    assert len(lines) == 4 and all(line.startswith("✅") for line in lines)
    assert client.closed
    cfg["key"] = ""
    lines = asyncio.run(check_provider(cfg, lambda _: pytest.fail("must not request")))
    assert all(line.startswith("❌") for line in lines)


@pytest.mark.parametrize("mode", ["url", "factory", "factory_value", "synth_value", "close"])
def test_selfcheck_isolates_provider_errors(mode):
    cfg = dict(provider="siliconflow", key="test", model="test", default_voice="test",
               url="https://example.com/v1", timeout=30, max_retries=0, speed=1, format="wav")
    if mode == "url":
        cfg["url"] = "https://[invalid"
    class Client:
        async def synth(self, text, voice, out_dir):
            if mode == "synth_value":
                raise ValueError("secret-synth-detail")
            path = out_dir / "sample.wav"
            path.write_bytes(wav_bytes())
            return path
        async def close(self):
            raise RuntimeError("secret-close-detail")
    def factory(_):
        if mode == "factory_value":
            raise ValueError("secret-factory-detail")
        if mode == "factory":
            raise RuntimeError("secret-factory-detail")
        return Client()
    lines = asyncio.run(check_provider(cfg, factory))
    assert any(line.startswith("❌") for line in lines)
    assert "secret" not in "\n".join(lines)


def test_cache_singleflight_style_and_corruption(tmp_path):
    class Client:
        format = "wav"
        def __init__(self):
            self.style = "gentle"
            self._calls = 0
        @audio_cache
        async def synth(self, text, voice, out_dir):
            self._calls += 1
            await asyncio.sleep(0.01)
            path = out_dir / "sample.wav"
            path.write_bytes(wav_bytes())
            return path
    async def run():
        client = Client()
        one, two = await asyncio.gather(client.synth("text", "voice", tmp_path),
                                        client.synth("text", "voice", tmp_path))
        assert one == two and client._calls == 1
        client.style = "serious"
        three = await client.synth("text", "voice", tmp_path)
        assert three != one and client._calls == 2
        three.write_bytes(b"bad")
        assert await client.synth("text", "voice", tmp_path) == three
        assert client._calls == 3
        assert not list(tmp_path.glob(".synth-*"))
    asyncio.run(run())


def test_corrupt_history_is_never_overwritten():
    manager = SimpleNamespace(get_conversation=AsyncMock(return_value=SimpleNamespace(history="broken")),
                              update_conversation=AsyncMock())
    obj = SimpleNamespace(context=SimpleNamespace(conversation_manager=manager), _get_umo=lambda _: "sid")
    result = asyncio.run(method("_append_assistant_text_to_history", json=json, logger=logging.getLogger(__name__))(
        obj, object(), "new text", conversation_id="captured"))
    assert result is False
    manager.update_conversation.assert_not_called()
    manager.get_conversation.assert_awaited_once_with("sid", "captured")


def test_disabled_tool_copy_is_request_local():
    class Tools:
        def __init__(self):
            self.tools = ["tts_speak", "other"]
        def remove_tool(self, name):
            self.tools = [t for t in self.tools if t != name]
    shared = Tools()
    req = SimpleNamespace(func_tool=shared, prompt="")
    obj = SimpleNamespace(config=SimpleNamespace(is_llm_tool_enabled=lambda: False),
                          _inject_recent_spoken_assistant_context=AsyncMock(),
                          publish_output_marker_mode=lambda _: None,
                          _should_inject_minimax_prompt=lambda _: False)
    asyncio.run(method("on_llm_request")(obj, object(), req))
    assert req.func_tool.tools == ["other"]
    assert shared.tools == ["tts_speak", "other"]


def test_new_defaults_and_existing_policy():
    from astrbot_plugin_tts_emotion_router.core.config import ConfigManager
    fresh = ConfigManager({})
    assert not fresh.is_voice_output_enabled_for_umo("sid")
    assert not fresh.is_probability_output_enabled_for_umo("sid")
    old = ConfigManager({"feature_policies": {"voice_output": {"enable": True, "mode": "blacklist"}}})
    assert old.is_voice_output_enabled_for_umo("sid")


def test_actual_host_greedy_parser():
    host = ROOT.parents[1] / "_astrbot/astrbot/core/star/filter/command.py"
    tree = ast.parse(host.read_text(encoding="utf-8"))
    converter = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                     and n.name == "validate_and_convert_params")
    namespace = dict(Any=object, GreedyStr=str, typing=__import__("typing"),
                     types=__import__("types"), inspect=__import__("inspect"))
    exec(compile(ast.Module(body=[converter], type_ignores=[]), str(host), "exec"), namespace)
    for tokens in ([], ["0"], ["123"], ["hello", "world"], ["中文", "测试"]):
        result = namespace[converter.name](None, tokens, {"text": str})
        assert result == {"text": " ".join(tokens)}


@pytest.mark.parametrize("failed", [0, 1, 2, None])
def test_segment_delivery_preserves_order_and_failures(failed):
    class Plain:
        def __init__(self, text):
            self.text = text
    class Record:
        def __init__(self, file):
            self.file = file
    state = SimpleNamespace(text_voice_enabled=None, consume_pending_direct_speak_text=lambda: "")
    parts = [SimpleNamespace(success=i != failed, audio_path=Path(f"{i}.wav") if i != failed else None,
                             segment=SimpleNamespace(text=f"段{i}")) for i in range(3)]
    seg_result = SimpleNamespace(segments=parts, successful_segments=[p for p in parts if p.success])
    result = SimpleNamespace(chain=[Plain("全文")], is_llm_result=lambda: True)
    sent = []
    async def send(chain):
        sent.extend(chain)
    event = SimpleNamespace(get_result=lambda: result, send=send,
                            plain_result=lambda t: [Plain(t)], chain_result=lambda c: c)
    config = SimpleNamespace(is_voice_output_enabled_for_umo=lambda _: True,
                             is_probability_output_enabled_for_umo=lambda _: False,
                             is_text_voice_output_enabled_for_umo=lambda _: False,
                             is_segmented_output_enabled_for_umo=lambda _: True,
                             validation_errors=lambda: [])
    obj = SimpleNamespace(_start_background_tasks=AsyncMock(), _cleanup_stale_inflight=lambda: None,
        _get_umo=lambda _: "sid", _session_state={}, _get_session_state=lambda _: state,
        _normalize_text=lambda t: t, _strip_emo_head_many=lambda t: (t, None),
        _strip_any_visible_markers=lambda t: t,
        _prepare_text_for_tts=lambda t: SimpleNamespace(tts_text=t, display_text=t, links=[], codes=[]),
        _append_references_to_text=lambda t, **kw: t,
        _build_fallback_result_chain=lambda chain, t, **kw: [Plain(t)], config=config,
        condition_checker=SimpleNamespace(check_all=lambda *a, **kw: SimpleNamespace(passed=True)),
        _build_inflight_sig=lambda *a: "sig", _inflight_sigs={}, _recent_reasons={},
        segmented_tts_enabled=True, segmented_min_chars=1,
        segmented_tts_processor=SimpleNamespace(should_use_segmented=lambda *a: True,
            process_only=AsyncMock(return_value=seg_result), _calculate_interval=lambda _: 0),
        tts_processor=SimpleNamespace(normalize_audio_path=str, process=AsyncMock()),
        _get_current_conversation_id=AsyncMock(return_value="captured"),
        _remember_spoken_assistant_text=AsyncMock())
    run = method("on_decorating_result", Plain=Plain, Record=Record, asyncio=asyncio, logger=logging.getLogger(__name__))
    asyncio.run(run(obj, event))
    assert sent == []
    assert [getattr(x, "text", getattr(x, "file", "")) for x in result.chain] == (
        ["全文"] if failed is not None else [f"{i}.wav" for i in range(3)])
    assert not obj._inflight_sigs
    obj.tts_processor.process.assert_not_called()
    obj._remember_spoken_assistant_text.assert_not_called()
