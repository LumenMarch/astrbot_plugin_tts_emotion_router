"""Pace this event's records without bypassing the host response pipeline."""
import asyncio
import time


def pace_records(event, records, state):
    original_send = event.send
    pending = {id(record): (interval, text) for record, interval, text in records}
    delivery = {'texts': [], 'restore': lambda: setattr(event, 'send', original_send)}
    event._tts_segment_delivery = delivery
    next_send = 0.0

    async def send(message):
        nonlocal next_send
        matched = [comp for comp in message.chain if id(comp) in pending]
        try:
            if matched:
                await asyncio.sleep(max(0.0, next_send - time.monotonic()))
            result = await original_send(message)
            for comp in matched:
                interval, text = pending[id(comp)]
                next_send = time.monotonic() + interval
                delivery['texts'].append(text)
                state.last_sent_ts = time.time()
            return result
        except asyncio.CancelledError:
            event.send = original_send
            raise
        except Exception:
            if not matched:
                raise
            from astrbot.core.message.components import Plain

            # Use the underlying sender so fallback cannot recurse or count as speech.
            fallback = [
                Plain(text=pending[id(comp)][1]) if id(comp) in pending else comp
                for comp in message.chain
            ]
            return await original_send(message.derive(fallback))
        finally:
            for comp in matched:
                pending.pop(id(comp), None)
            if not pending:
                event.send = original_send

    event.send = send
