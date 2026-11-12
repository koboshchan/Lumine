"""Authenticated, single-frame OpenAI-compatible serving of full LFM3 checkpoints.

The renderer is imported from training. Stateless requests intentionally have no
invented memory or unverified previous actions. Use an encrypted private tunnel.
"""
import asyncio
import argparse
import base64
import binascii
import hmac
import io
import json
import logging
import math
import os
import re
import threading
import time
import uuid
from pathlib import Path

from train import ActionCollator, ROOT

MAX_BODY = 12 * 1024 * 1024


def request_sample(body):
    """Adapt the existing runner transport, not its obsolete Qwen prompt."""
    from PIL import Image
    messages = body.get('messages')
    if not isinstance(messages, list) or len(messages) != 1:
        raise ValueError('Expected exactly one image-only user message')
    message = messages[0]
    content = message.get('content')
    if message.get('role') != 'user' or not isinstance(content, list) or len(content) != 1:
        raise ValueError('Expected exactly one image-only user message')
    part = content[0]
    if part.get('type') != 'image_url':
        raise ValueError('Expected image_url data URI')
    url = part.get('image_url', {}).get('url', '')
    prefix = 'data:image/png;base64,'
    if not isinstance(url, str) or not url.startswith(prefix):
        raise ValueError('Only inline PNG is supported; remote URLs are never fetched')
    try:
        raw = base64.b64decode(url[len(prefix):], validate=True)
        image = Image.open(io.BytesIO(raw))
        if image.format != 'PNG' or image.size not in ((1280, 736), (1280, 720), (640, 360)):
            raise ValueError('Expected PNG 1280x720, legacy-padded 1280x736 or 640x360')
        image.load()
        image = image.convert('RGB')
        if image.size == (1280, 736):
            if image.crop((0, 720, 1280, 736)).getbbox() is not None:
                raise ValueError('Legacy padding must be entirely black')
            image = image.crop((0, 0, 1280, 720))
        if image.size != (640, 360):
            image = image.resize((640, 360), Image.Resampling.LANCZOS)
    except (binascii.Error, OSError) as exc:
        raise ValueError('Invalid PNG data') from exc
    # There is no trusted execution/timestamp history in the old runner protocol.
    return dict(current=dict(image=image, action='', timestamp_s=0.0), recent=[], middle=[], notes=[])


def encode_prompt(processor, config, sample):
    messages = ActionCollator(processor, config).messages(sample)
    assert messages[-1]['role'] == 'assistant'
    def encode(value):
        return processor.apply_chat_template(
            ActionCollator(processor, config).messages(value)[:-1], tokenize=True,
            return_dict=True, return_tensors='pt', add_generation_prompt=True,
            enable_thinking=False)
    sample = dict(sample, recent=list(sample['recent']), middle=list(sample['middle']),
                  notes=list(sample.get('notes', [])))
    baseline = encode(dict(sample, recent=[], middle=[], notes=[]))['input_ids'].numel()
    while True:
        batch = encode(sample)
        if batch['input_ids'].numel() - baseline <= config['history_tokens']:
            return batch
        for tier in ('middle', 'recent', 'notes'):
            if sample[tier]:
                sample[tier].pop(0)
                break
        else:
            raise ValueError('Cannot satisfy history token budget')


class Backend:
    def __init__(self, checkpoint, device='cuda'):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        checkpoint = Path(checkpoint).resolve()
        self.config = json.loads((checkpoint / 'training-policy.json').read_text())
        expected_policy = json.loads((ROOT / 'config.json').read_text())
        if self.config.get('action_special_tokens') is True:
            expected_policy['action_special_tokens'] = True
        if self.config != expected_policy:
            raise ValueError('Checkpoint training policy does not match this code')
        self.processor = AutoProcessor.from_pretrained(str(checkpoint), trust_remote_code=True)
        if self.processor.chat_template != (ROOT / 'chat_template.jinja').read_text():
            raise ValueError('Checkpoint template differs from exact training template')
        if self.config.get('action_special_tokens'):
            from action_tokens import ACTION_TOKENS
            audit = json.loads((checkpoint / 'action-token-audit.json').read_text())
            for token in ACTION_TOKENS:
                if self.processor.tokenizer.encode(token, add_special_tokens=False) != [audit[token]['id']]:
                    raise ValueError('Checkpoint action token identity differs')
        self.device = device
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(checkpoint), trust_remote_code=True,
            dtype=torch.bfloat16 if device.startswith('cuda') else torch.float32,
            attn_implementation='sdpa').to(device).eval()
        self.model.config.use_cache = True
        self.lock = threading.Lock()

    def complete(self, sample, max_tokens):
        import torch
        with self.lock, torch.inference_mode():
            batch = encode_prompt(self.processor, self.config, sample).to(self.device)
            prompt_tokens = batch['input_ids'].shape[-1]
            output = self.model.generate(**batch, max_new_tokens=max_tokens,
                                         do_sample=False, use_cache=True)
            ids = output[0, prompt_tokens:].tolist()
            # Strip only terminal EOS/padding, never action delimiters or vocabulary.
            eos = self.model.generation_config.eos_token_id
            end = set(eos if isinstance(eos, list) else [eos])
            end.add(self.processor.tokenizer.pad_token_id)
            count = len(ids)
            while ids and ids[-1] in end:
                ids.pop()
            text = self.processor.tokenizer.decode(ids, skip_special_tokens=False)
            return text, prompt_tokens, count


def create_app(backend, api_key, model_name='lumine-mini-2b'):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from starlette.concurrency import run_in_threadpool
    if not api_key:
        raise ValueError('LUMINE_SERVE_KEY must be nonempty')
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    sessions = {}
    gate = asyncio.Lock()
    id_pattern = re.compile(r'[0-9a-f]{32}\Z')

    async def session_json(request, keys):
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > MAX_BODY:
                raise ValueError('Request too large')
        body = json.loads(raw)
        if not isinstance(body, dict) or set(body) != keys:
            raise ValueError('Invalid session request fields')
        if not isinstance(body['session_id'], str) or not id_pattern.fullmatch(body['session_id']):
            raise ValueError('Invalid session id')
        return body

    @app.post('/v1/session/bootstrap')
    async def bootstrap(request: Request):
        try:
            body = await session_json(request, {'session_id', 'frames', 'timestamps_s', 'acknowledge_no_inputs'})
            if body['acknowledge_no_inputs'] is not True:
                raise ValueError('Explicit no-input acknowledgement required')
            if not isinstance(body['frames'], list) or not 1 <= len(body['frames']) <= 8:
                raise ValueError('Provide 1..8 actual chronological captured frames')
            times = body['timestamps_s']
            if not isinstance(times, list) or len(times) != len(body['frames']) or any(
                    type(t) not in (int, float) or not math.isfinite(t) or t < 0 for t in times):
                raise ValueError('Actual finite capture timestamps required')
            if times[-1] - times[0] > 60:
                raise ValueError('Bootstrap history must span at most 60 seconds')
            if any(b <= a for a, b in zip(times, times[1:])):
                raise ValueError('Capture timestamps must increase')
            rows = []
            for i, frame in enumerate(body['frames']):
                sample = request_sample({'messages': [{'role': 'user', 'content': [
                    {'type': 'image_url', 'image_url': {'url': frame}}]}]})
                row = sample['current']
                row.update(action='<|action_start|>0 0 0 ;  ;  ;  ;  ;  ; <|action_end|>',
                           timestamp_s=times[i] - times[0])
                rows.append(row)
        except (ValueError, TypeError, AttributeError, KeyError):
            return JSONResponse({'error': 'Invalid bootstrap'}, status_code=400)
        if gate.locked() or body['session_id'] in sessions:
            return JSONResponse({'error': 'Bootstrap requires a fresh idle session'}, status_code=409)
        if len(sessions) >= 8:
            return JSONResponse({'error': 'Session capacity'}, status_code=503)
        now = time.monotonic()
        sessions[body['session_id']] = dict(started=now-(times[-1]-times[0]), touched=now,
            records=rows, pending=None, requests=set(), bootstrap_noinputs=True)
        return {'session_id': body['session_id'], 'bootstrapped': len(rows),
                'protocol': 'lumine-session-v1'}

    @app.post('/v1/session/reset')
    async def reset(request: Request):
        try:
            body = await session_json(request, {'session_id'})
        except (ValueError, TypeError):
            return JSONResponse({'error': 'Invalid reset'}, status_code=400)
        if gate.locked():
            return JSONResponse({'error': 'Inference busy'}, status_code=409)
        session = sessions.get(body['session_id'])
        if session:
            session.update(records=[], pending=None, started=time.monotonic(), touched=time.monotonic(), bootstrap_noinputs=False)
        return {'reset': True}

    @app.post('/v1/session/step')
    async def step(request: Request):
        try:
            body = await session_json(request, {'session_id', 'request_id', 'frame',
                                                'last_executed_action', 'executed_action_id'})
            rid = body['request_id']
            if not isinstance(rid, str) or not id_pattern.fullmatch(rid):
                raise ValueError('Invalid request id')
            sample = request_sample({'messages': [{'role': 'user', 'content': [
                {'type': 'image_url', 'image_url': {'url': body['frame']}}]}]})
        except (ValueError, TypeError, AttributeError, KeyError):
            return JSONResponse({'error': 'Invalid step'}, status_code=400)
        if gate.locked():
            return JSONResponse({'error': 'Inference busy'}, status_code=409)
        async with gate:
            now = time.monotonic()
            for sid in list(sessions):
                if now - sessions[sid]['touched'] > 900:
                    del sessions[sid]
            sid = body['session_id']
            session = sessions.get(sid)
            ack, aid = body['last_executed_action'], body['executed_action_id']
            pending = session['pending'] if session else None
            if session and session.get('bootstrap_noinputs') and ack is not None:
                return JSONResponse({'error': 'Dry-run bootstrap sessions forbid executed input; reset first'}, status_code=409)
            if (ack is None) != (aid is None) or (ack is not None and
                    (not pending or ack != pending['action'] or aid != pending['action_id'])):
                return JSONResponse({'error': 'Execution acknowledgement mismatch'}, status_code=409)
            if session is None:
                if len(sessions) >= 2:
                    return JSONResponse({'error': 'Session capacity'}, status_code=503)
                session = dict(started=now, touched=now, records=[], pending=None, requests=set())
                sessions[sid] = session
            if rid in session['requests'] or len(session['requests']) >= 8192:
                return JSONResponse({'error': 'Request replay or limit'}, status_code=409)
            session['requests'].add(rid)  # Burn before inference, even if it fails.
            session['touched'] = now
            current = sample['current']
            current['timestamp_s'] = now - session['started']
            records = list(session['records'])
            if ack is None and pending is not None and session.get('bootstrap_noinputs'):
                records.append(dict(pending['row'],
                    action='<|action_start|>0 0 0 ;  ;  ;  ;  ;  ; <|action_end|>'))
            if ack is not None:
                records.append(dict(pending['row'], action=pending['canonical_action']))
            records = [r for r in records if current['timestamp_s'] - r['timestamp_s'] <= 60]
            recent = records[-backend.config['recent_frames']:]
            cutoff = recent[0]['timestamp_s'] if recent else current['timestamp_s']
            middle, last = [], None
            for row in records:
                ts = row['timestamp_s']
                if ts >= cutoff:
                    continue
                if last is None or ts - last >= 5:
                    middle.append(row)
                    last = ts
            sample.update(recent=recent, middle=middle)
            try:
                text, _, _ = await run_in_threadpool(backend.complete, sample, 192)
                # Use the existing runner proxy's strict action grammar and safety checks.
                import sys
                root = str(ROOT.parents[1])
                if root not in sys.path:
                    sys.path.insert(0, root)
                from SessionProxy.service import parse_completion
                action, _ = parse_completion(text)
                if await request.is_disconnected():
                    return JSONResponse({'error': 'Client disconnected'}, status_code=409)
            except Exception:
                logging.exception('Session inference or canonical action validation failed')
                return JSONResponse({'error': 'Inference or action validation failed'}, status_code=502)
            # Preserve every acknowledged observation in the 60s source window.
            # Older-than-recent8 pixels only need their training thumbnail tier.
            from PIL import Image
            retained = []
            recent_ids = {id(r) for r in recent}
            for row in records:
                row = dict(row)
                if 'thumbnail' not in row:
                    row['thumbnail'] = row['image'].resize((320, 180), Image.Resampling.LANCZOS)
                if id(records[len(retained)]) not in recent_ids:
                    row.pop('image', None)
                retained.append(row)
            if len(retained) > 400:
                return JSONResponse({'error': '60s observation capacity'}, status_code=503)
            aid = uuid.uuid4().hex
            runner_action = action.replace('<use_absolute>', '<|use_abs|>')
            session.update(records=retained, pending={'row': current, 'action': runner_action,
                           'canonical_action': action, 'action_id': aid})
            return {'session_id': sid, 'action': runner_action, 'action_id': aid, 'protocol': 'lumine-session-v1'}


    @app.middleware('http')
    async def authenticate(request, call_next):
        if not hmac.compare_digest(request.headers.get('authorization', ''), 'Bearer ' + api_key):
            return JSONResponse({'error': {'message': 'Unauthorized'}}, status_code=401)
        return await call_next(request)

    @app.get('/health')
    def health():
        return {'status': 'ready', 'model': model_name, 'context': 'single-frame'}

    @app.get('/v1/models')
    def models():
        return {'object': 'list', 'data': [{'id': model_name, 'object': 'model', 'owned_by': 'local'}]}

    @app.post('/v1/chat/completions')
    async def complete(request: Request):
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > MAX_BODY:
                return JSONResponse({'error': {'message': 'Request too large'}}, status_code=413)
        try:
            body = json.loads(raw)
            if not isinstance(body, dict):
                raise ValueError('Expected JSON object')
            if body.get('stream', False) or body.get('n', 1) != 1:
                raise ValueError('Only nonstreaming n=1 requests are supported')
            if body.get('model') != model_name:
                raise ValueError('Unknown model')
            if body.get('temperature', 0) != 0:
                raise ValueError('Only greedy temperature=0 is supported')
            max_tokens = body.get('max_tokens', 192)
            if type(max_tokens) is not int or not 1 <= max_tokens <= 512:
                raise ValueError('max_tokens must be an integer in 1..512')
            sample = request_sample(body)
        except (ValueError, TypeError, AttributeError, KeyError) as exc:
            return JSONResponse({'error': {'message': str(exc)}}, status_code=400)
        text, prompt, generated = await run_in_threadpool(backend.complete, sample, max_tokens)
        return {'id': 'chatcmpl-' + uuid.uuid4().hex, 'object': 'chat.completion',
                'created': int(time.time()), 'model': model_name,
                'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': text},
                             'finish_reason': 'length' if generated >= max_tokens else 'stop'}],
                'usage': {'prompt_tokens': prompt, 'completion_tokens': generated,
                          'total_tokens': prompt + generated}}
    return app


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='Extracted full model checkpoint directory')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--model-name', default='lumine-mini-2b')
    args = parser.parse_args()
    key = os.environ.get('LUMINE_SERVE_KEY', '')
    if not key:
        parser.error('Set LUMINE_SERVE_KEY in the environment')
    backend = Backend(args.checkpoint, args.device)
    uvicorn.run(create_app(backend, key, args.model_name), host=args.host, port=args.port,
                access_log=False)


if __name__ == '__main__':
    main()
