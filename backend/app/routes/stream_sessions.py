import json
import os
import time
import wave
import io
import struct
import asyncio
import sys
import re
import numpy as np
import sounddevice as sd
import threading
from datetime import datetime
from flask import Blueprint, Response, request, stream_with_context
from collections import deque
from .sessions import _session_path
from ..transcription_service import WhisperTranscriptionService
from ..settings import read_settings

# Import ai_core orchestrator (main model)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from ai_core.main import JanusAI, JanusConfig, PersuasionObjective
from ai_core.core.sentiment_analyzer import ConversationAnalysis
from ai_core.core.persuasion_engine import AlignmentAnalysis
from ..vectorstore import get_collection

bp = Blueprint('stream_sessions', __name__)

# Store active session audio buffers
# Format: {session_id: deque of PCM16 audio chunks}
_session_audio_buffers = {}
_session_states = {}  # Track session streaming state

# Track upload counts per session for logging
_upload_counts = {}

# Track packet directories for each session
_session_packet_dirs = {}

# Track audio output streams for playback
_session_audio_streams = {}

# Track latency statistics per session
_latency_stats = {}  # {session_id: {'min': ms, 'max': ms, 'sum': ms, 'count': n}}

# Track transcription services per session
_transcription_services = {}  # {session_id: WhisperTranscriptionService}

# Outgoing audio queue per session (producer: upload handler; consumer: stream generator)
_session_output_queues = {}
_session_output_conds = {}

def _ensure_output_queue(session_id: str):
    if session_id not in _session_output_queues:
        _session_output_queues[session_id] = deque()
        _session_output_conds[session_id] = threading.Condition()

def _enqueue_audio(session_id: str, audio_bytes: bytes):
    if not audio_bytes:
        return
    _ensure_output_queue(session_id)
    with _session_output_conds[session_id]:
        _session_output_queues[session_id].append(audio_bytes)
        _session_output_conds[session_id].notify_all()

async def _synthesize_for_session(session_id: str, verbose: bool) -> bytes:
    api_key = os.getenv('BOSON_API_KEY') or ''
    cfg = JanusConfig(api_key=api_key)
    janus = JanusAI(cfg)
    print(f"[JANUS] init gen={cfg.generation_model} reason={cfg.reasoning_model}")

    # Load session objective if available
    try:
        sess_json_path = _session_path(session_id)
        if os.path.exists(sess_json_path):
            with open(sess_json_path, 'r') as f:
                sess = json.load(f)
            obj_text = (sess.get('objective') or '').strip()
            if obj_text:
                objective = PersuasionObjective(
                    main_goal=obj_text,
                    key_points=[],
                    audience_triggers=["value", "trust", "results"]
                )
                await janus.set_persuasion_objective(objective)
                print(f"[JANUS] objective='{obj_text[:48]}{'…' if len(obj_text)>48 else ''}'")
    except Exception as e:
        print(f"[SS] obj_warn {type(e).__name__}: {str(e)[:80]}")

    # Use current session transcript from file
    transcript_text = _read_session_transcript(session_id).strip() or ""
    print(f"[JANUS] input len={len(transcript_text)} '{transcript_text[:48]}{'…' if len(transcript_text)>48 else ''}'")

    # Detect last question sentence (activate only if a real question exists)
    q_idx = transcript_text.rfind('?')
    has_question = q_idx != -1
    if not has_question:
        return b""
    start = max(transcript_text.rfind('.', 0, q_idx), transcript_text.rfind('!', 0, q_idx), transcript_text.rfind('?', 0, q_idx))
    start = 0 if start == -1 else start + 1
    end = q_idx + 1
    question_sentence = transcript_text[start:end].strip()
    print(f"[JANUS] question='{question_sentence[:64]}{'…' if len(question_sentence)>64 else ''}'")

    # Retrieve session-linked documents from Chroma for grounding
    retrieved_context = ""
    try:
        sess_json_path = _session_path(session_id)
        file_ids = []
        if os.path.exists(sess_json_path):
            with open(sess_json_path, 'r') as f:
                sess = json.load(f)
            file_ids = sess.get('fileIds') or []
        if file_ids:
            col = get_collection()
            query_text = question_sentence or transcript_text[-200:]
            res = col.query(query_texts=[query_text], n_results=4, where={})
            docs = (res.get('documents') or [[]])[0]
            metas = (res.get('metadatas') or [[]])[0]
            pairs = []
            for d, m in zip(docs, metas):
                if not isinstance(m, dict) or (m.get('sessionId') and m.get('sessionId') != session_id):
                    continue
                pairs.append(d)
            if pairs:
                retrieved_context = "\n".join(pairs[:4])
                print(f"[SS] ctx {len(pairs)} docs")
    except Exception as e:
        print(f"[SS] ctx_err {type(e).__name__}: {str(e)[:100]}")

    # Highlight question and build grounded input
    highlighted = transcript_text.replace(question_sentence, f"[QUESTION]{question_sentence}[/QUESTION]")
    grounded_input = highlighted if not retrieved_context else f"[CONTEXT]\n{retrieved_context}\n[/CONTEXT]\n\n{highlighted}"

    # Minimal analysis/alignment
    analysis = ConversationAnalysis(
        sentiment="interested",
        is_question=True,
        question_type="clarification",
        detected_concerns=[],
        emotional_state="engaged",
        requires_response=True,
        is_complex_question=False,
        key_topics=[]
    )

    alignment = AlignmentAnalysis(
        alignment_score=0.7,
        addressed_points=[],
        remaining_points=[],
        detected_opportunities=["address directly"],
        suggested_pivot=None,
        urgency_level="medium",
        next_best_action="respond persuasively"
    )

    # Generate persuasive response
    response = await janus.response_generator.generate(
        transcript=grounded_input,
        analysis=analysis,
        alignment=alignment,
        objective=janus.current_objective,
        history=janus.conversation_history
    )
    reply_text = response.prosody_text if verbose else response.text
    if not reply_text or len(reply_text) < 4:
        reply_text = "Got it."

    # Synthesize audio
    audio_bytes = await janus.audio_generator.generate(
        reply_text,
        response.prosody_tokens,
        response.voice_profile
    )
    return audio_bytes or b""

# Outgoing audio queue per session (producer: upload handler; consumer: stream generator)
_session_output_queues = {}
_session_output_conds = {}

def _ensure_output_queue(session_id: str):
    if session_id not in _session_output_queues:
        _session_output_queues[session_id] = deque()
        _session_output_conds[session_id] = threading.Condition()

def _enqueue_audio(session_id: str, audio_bytes: bytes):
    if not audio_bytes:
        return
    _ensure_output_queue(session_id)
    with _session_output_conds[session_id]:
        _session_output_queues[session_id].append(audio_bytes)
        _session_output_conds[session_id].notify_all()

async def _synthesize_for_session(session_id: str, verbose: bool) -> bytes:
    api_key = os.getenv('BOSON_API_KEY') or ''
    cfg = JanusConfig(api_key=api_key)
    janus = JanusAI(cfg)
    print(f"[JANUS] init gen={cfg.generation_model} reason={cfg.reasoning_model}")

    # Load session objective if available
    try:
        sess_json_path = _session_path(session_id)
        if os.path.exists(sess_json_path):
            with open(sess_json_path, 'r') as f:
                sess = json.load(f)
            obj_text = (sess.get('objective') or '').strip()
            if obj_text:
                objective = PersuasionObjective(
                    main_goal=obj_text,
                    key_points=[],
                    audience_triggers=["value", "trust", "results"]
                )
                await janus.set_persuasion_objective(objective)
                print(f"[JANUS] objective='{obj_text[:48]}{'…' if len(obj_text)>48 else ''}'")
    except Exception as e:
        print(f"[SS] obj_warn {type(e).__name__}: {str(e)[:80]}")

    # Use current session transcript from file
    transcript_text = _read_session_transcript(session_id).strip() or ""
    print(f"[JANUS] input len={len(transcript_text)} '{transcript_text[:48]}{'…' if len(transcript_text)>48 else ''}'")

    # Detect last question sentence (activate only if a real question exists)
    q_idx = transcript_text.rfind('?')
    has_question = q_idx != -1
    if not has_question:
        return b""
    start = max(transcript_text.rfind('.', 0, q_idx), transcript_text.rfind('!', 0, q_idx), transcript_text.rfind('?', 0, q_idx))
    start = 0 if start == -1 else start + 1
    end = q_idx + 1
    question_sentence = transcript_text[start:end].strip()
    print(f"[JANUS] question='{question_sentence[:64]}{'…' if len(question_sentence)>64 else ''}'")

    # Retrieve session-linked documents from Chroma for grounding
    retrieved_context = ""
    try:
        sess_json_path = _session_path(session_id)
        file_ids = []
        if os.path.exists(sess_json_path):
            with open(sess_json_path, 'r') as f:
                sess = json.load(f)
            file_ids = sess.get('fileIds') or []
        if file_ids:
            col = get_collection()
            query_text = question_sentence or transcript_text[-200:]
            res = col.query(query_texts=[query_text], n_results=4, where={})
            docs = (res.get('documents') or [[]])[0]
            metas = (res.get('metadatas') or [[]])[0]
            pairs = []
            for d, m in zip(docs, metas):
                if not isinstance(m, dict) or (m.get('sessionId') and m.get('sessionId') != session_id):
                    continue
                pairs.append(d)
            if pairs:
                retrieved_context = "\n".join(pairs[:4])
                print(f"[SS] ctx {len(pairs)} docs")
    except Exception as e:
        print(f"[SS] ctx_err {type(e).__name__}: {str(e)[:100]}")

    # Highlight question and build grounded input
    highlighted = transcript_text.replace(question_sentence, f"[QUESTION]{question_sentence}[/QUESTION]")
    grounded_input = highlighted if not retrieved_context else f"[CONTEXT]\n{retrieved_context}\n[/CONTEXT]\n\n{highlighted}"

    # Minimal analysis/alignment
    analysis = ConversationAnalysis(
        sentiment="interested",
        is_question=True,
        question_type="clarification",
        detected_concerns=[],
        emotional_state="engaged",
        requires_response=True,
        is_complex_question=False,
        key_topics=[]
    )

    alignment = AlignmentAnalysis(
        alignment_score=0.7,
        addressed_points=[],
        remaining_points=[],
        detected_opportunities=["address directly"],
        suggested_pivot=None,
        urgency_level="medium",
        next_best_action="respond persuasively"
    )

    # Generate persuasive response
    response = await janus.response_generator.generate(
        transcript=grounded_input,
        analysis=analysis,
        alignment=alignment,
        objective=janus.current_objective,
        history=janus.conversation_history
    )
    reply_text = response.prosody_text if verbose else response.text
    if not reply_text or len(reply_text) < 4:
        reply_text = "Got it."

    # Synthesize audio
    audio_bytes = await janus.audio_generator.generate(
        reply_text,
        response.prosody_tokens,
        response.voice_profile
    )
    return audio_bytes or b""

# Outgoing audio queues per session (producer: upload handler; consumer: stream generator)
_session_output_queues = {}
_session_output_conds = {}

def _ensure_output_queue(session_id: str):
    if session_id not in _session_output_queues:
        _session_output_queues[session_id] = deque()
        _session_output_conds[session_id] = threading.Condition()

def _enqueue_audio(session_id: str, audio_bytes: bytes):
    if not audio_bytes:
        return
    _ensure_output_queue(session_id)
    with _session_output_conds[session_id]:
        _session_output_queues[session_id].append(audio_bytes)
        _session_output_conds[session_id].notify_all()

async def _synthesize_for_session(session_id: str, verbose: bool) -> bytes:
    api_key = os.getenv('BOSON_API_KEY') or ''
    cfg = JanusConfig(api_key=api_key)
    janus = JanusAI(cfg)
    print(f"[JANUS] init gen={cfg.generation_model} reason={cfg.reasoning_model}")

    # Load session objective if available
    try:
        sess_json_path = _session_path(session_id)
        if os.path.exists(sess_json_path):
            with open(sess_json_path, 'r') as f:
                sess = json.load(f)
            obj_text = (sess.get('objective') or '').strip()
            if obj_text:
                objective = PersuasionObjective(
                    main_goal=obj_text,
                    key_points=[],
                    audience_triggers=["value", "trust", "results"]
                )
                await janus.set_persuasion_objective(objective)
                print(f"[JANUS] objective='{obj_text[:48]}{'…' if len(obj_text)>48 else ''}'")
    except Exception as e:
        print(f"[SS] obj_warn {type(e).__name__}: {str(e)[:80]}")

    # Use current session transcript from file
    transcript_text = _read_session_transcript(session_id).strip() or ""
    print(f"[JANUS] input len={len(transcript_text)} '{transcript_text[:48]}{'…' if len(transcript_text)>48 else ''}'")

    # Detect last question sentence (activate only if a real question exists)
    q_idx = transcript_text.rfind('?')
    has_question = q_idx != -1
    if not has_question:
        return b""
    start = max(transcript_text.rfind('.', 0, q_idx), transcript_text.rfind('!', 0, q_idx), transcript_text.rfind('?', 0, q_idx))
    start = 0 if start == -1 else start + 1
    end = q_idx + 1
    question_sentence = transcript_text[start:end].strip()
    print(f"[JANUS] question='{question_sentence[:64]}{'…' if len(question_sentence)>64 else ''}'")

    # Retrieve session-linked documents from Chroma for grounding
    retrieved_context = ""
    try:
        sess_json_path = _session_path(session_id)
        file_ids = []
        if os.path.exists(sess_json_path):
            with open(sess_json_path, 'r') as f:
                sess = json.load(f)
            file_ids = sess.get('fileIds') or []
        if file_ids:
            col = get_collection()
            query_text = question_sentence or transcript_text[-200:]
            res = col.query(query_texts=[query_text], n_results=4, where={})
            docs = (res.get('documents') or [[]])[0]
            metas = (res.get('metadatas') or [[]])[0]
            pairs = []
            for d, m in zip(docs, metas):
                if not isinstance(m, dict) or (m.get('sessionId') and m.get('sessionId') != session_id):
                    continue
                pairs.append(d)
            if pairs:
                retrieved_context = "\n".join(pairs[:4])
                print(f"[SS] ctx {len(pairs)} docs")
    except Exception as e:
        print(f"[SS] ctx_err {type(e).__name__}: {str(e)[:100]}")

    # Highlight question and build grounded input
    highlighted = transcript_text.replace(question_sentence, f"[QUESTION]{question_sentence}[/QUESTION]")
    grounded_input = highlighted if not retrieved_context else f"[CONTEXT]\n{retrieved_context}\n[/CONTEXT]\n\n{highlighted}"

    # Minimal analysis/alignment
    analysis = ConversationAnalysis(
        sentiment="interested",
        is_question=True,
        question_type="clarification",
        detected_concerns=[],
        emotional_state="engaged",
        requires_response=True,
        is_complex_question=False,
        key_topics=[]
    )

    alignment = AlignmentAnalysis(
        alignment_score=0.7,
        addressed_points=[],
        remaining_points=[],
        detected_opportunities=["address directly"],
        suggested_pivot=None,
        urgency_level="medium",
        next_best_action="respond persuasively"
    )

    # Generate persuasive response
    response = await janus.response_generator.generate(
        transcript=grounded_input,
        analysis=analysis,
        alignment=alignment,
        objective=janus.current_objective,
        history=janus.conversation_history
    )
    reply_text = response.prosody_text if verbose else response.text
    if not reply_text or len(reply_text) < 4:
        reply_text = "Got it."

    # Synthesize audio
    audio_bytes = await janus.audio_generator.generate(
        reply_text,
        response.prosody_tokens,
        response.voice_profile
    )
    return audio_bytes or b""

# --- Event-driven TTS state (24 kHz end-to-end) ---
_janus_instances = {}
_last_answered_qpos = {}
_tts_workers_active = {}
_session_voice_profiles = {}

def _get_janus_for_session(session_id: str) -> JanusAI:
    api_key = os.getenv('BOSON_API_KEY') or ''
    cfg = JanusConfig(api_key=api_key)
    if session_id in _janus_instances:
        return _janus_instances[session_id]
    janus = JanusAI(cfg)
    # Attempt to apply session objective
    try:
        sess_json_path = _session_path(session_id)
        if os.path.exists(sess_json_path):
            with open(sess_json_path, 'r') as f:
                sess = json.load(f)
            obj_text = (sess.get('objective') or '').strip()
            if obj_text:
                objective = PersuasionObjective(
                    main_goal=obj_text,
                    key_points=[],
                    audience_triggers=["value", "trust", "results"]
                )
                # Run async setter in a temp loop
                asyncio.run(janus.set_persuasion_objective(objective))
    except Exception as e:
        print(f"[SS] janus_obj_warn {type(e).__name__}: {str(e)[:80]}")
    _janus_instances[session_id] = janus
    # Attach voice sample reference if available
    try:
        voice_path = None
        # Prefer path saved on session JSON
        sess_json_path = _session_path(session_id)
        if os.path.exists(sess_json_path):
            with open(sess_json_path, 'r') as f:
                sess = json.load(f)
            voice_path = sess.get('voiceSamplePath')
        # Fallback: pick latest converted recording in backend/uploads
        if not voice_path:
            uploads_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'uploads'))
            if os.path.isdir(uploads_dir):
                try:
                    candidates = [
                        os.path.join(uploads_dir, fn) for fn in os.listdir(uploads_dir)
                        if fn.startswith('recording-') and fn.endswith('.wav')
                    ]
                    if candidates:
                        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                        voice_path = candidates[0]
                except Exception:
                    pass
        if voice_path and os.path.exists(voice_path):
            with open(voice_path, 'rb') as vf:
                vbytes = vf.read()
            # Cache under a session-specific profile key
            profile_key = f'session:{session_id}'
            janus.audio_generator.reference_cache[profile_key] = (vbytes, "User reference sample")
            _session_voice_profiles[session_id] = profile_key
            print(f"[HIGGS] voice_ref {session_id} -> {os.path.basename(voice_path)}")
        else:
            print(f"[HIGGS] voice_ref {session_id} none")
    except Exception as e:
        print(f"[HIGGS] voice_ref_err {type(e).__name__}: {str(e)[:100]}")
    return janus

def _transcript_path_for_session(session_id: str) -> str:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_dir = os.path.join(project_root, 'data', 'sessions')
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, f'{session_id}_transcript.txt')

def _read_session_transcript(session_id: str) -> str:
    try:
        path = _transcript_path_for_session(session_id)
        if not os.path.exists(path):
            return ""
        with open(path, 'r', encoding='utf-8') as f:
            return f.read() or ""
    except Exception as e:
        print(f"[SS] read_trx_err {type(e).__name__}: {str(e)[:100]}")
        return ""

def _extract_last_question_and_context(text: str) -> tuple[str, str, int] | tuple[None, None, int]:
    """
    Returns (context_including_question, question_text, question_end_index) for the last question found.
    If none found, returns (None, None, -1).
    Pattern matches sentence segments ending with '?' and captures the question.
    """
    if not text:
        return None, None, -1
    pattern = r"(?:^|[.!?]\s*)([^.!?]*\?)"
    matches = list(re.finditer(pattern, text, flags=re.MULTILINE | re.DOTALL))
    if not matches:
        return None, None, -1
    last = matches[-1]
    # Extract the question substring captured by the group
    question = last.group(1)
    q_end = last.end()  # index after '?'
    context = text[:q_end]
    return context, question, q_end

def _split_and_enqueue(session_id: str, audio_bytes: bytes, samples_per_chunk: int = 2048):
    if not audio_bytes:
        return
    bytes_per_sample = 2
    chunk_size = samples_per_chunk * bytes_per_sample
    pos = 0
    total = len(audio_bytes)
    while pos < total:
        end = min(pos + chunk_size, total)
        _enqueue_audio(session_id, audio_bytes[pos:end])
        pos = end

def _maybe_trigger_tts(session_id: str, verbose: bool = False):

    # Avoid concurrent TTS workers per session
    if _tts_workers_active.get(session_id):
        return
    transcript_text = _read_session_transcript(session_id).strip() or ""
    if not transcript_text:
        return
    # Extract last question and context using regex pattern
    ctx, question, q_end = _extract_last_question_and_context(transcript_text)
    if q_end != -1:
        try:
            preview = (question or "")[:120]
            print(f"[SS] q_detect {session_id} end={q_end} ctx_chars={len(ctx or '')} q='{preview}'")
        except Exception:
            pass
    if q_end == -1:
        return
    last_pos = _last_answered_qpos.get(session_id, -1)
    if q_end <= last_pos:
        return

    def worker():
        try:
            _tts_workers_active[session_id] = True
            janus = _get_janus_for_session(session_id)
            try:
                q_prev = (question or "")[:160]
                print(f"[HIGGS] start {session_id} q='{q_prev}'")
            except Exception:
                pass

            # Prepare grounded input (retrieve docs if any)
            retrieved_context = ""
            try:
                sess_json_path = _session_path(session_id)
                file_ids = []
                if os.path.exists(sess_json_path):
                    with open(sess_json_path, 'r') as f:
                        sess = json.load(f)
                    file_ids = sess.get('fileIds') or []
                if file_ids:
                    col = get_collection()
                    # Query with detected question or transcript tail
                    query_text = (question or "").strip() or transcript_text[-200:]
                    res = col.query(query_texts=[query_text], n_results=4, where={})
                    docs = (res.get('documents') or [[]])[0]
                    metas = (res.get('metadatas') or [[]])[0]
                    pairs = []
                    for d, m in zip(docs, metas):
                        if not isinstance(m, dict) or (m.get('sessionId') and m.get('sessionId') != session_id):
                            continue
                        pairs.append(d)
                    if pairs:
                        retrieved_context = "\n".join(pairs[:4])
                        print(f"[SS] ctx {len(pairs)} docs")
            except Exception as e:
                print(f"[SS] ctx_err {type(e).__name__}: {str(e)[:100]}")

            # Build prompt: all transcript up to and including the question
            prompt_context = ctx or transcript_text
            highlighted = prompt_context
            if question and question in prompt_context:
                highlighted = prompt_context.replace(question, f"[QUESTION]{question}[/QUESTION]")
            grounded_input = highlighted if not retrieved_context else f"[CONTEXT]\n{retrieved_context}\n[/CONTEXT]\n\n{highlighted}"

            analysis = ConversationAnalysis(
                sentiment="interested",
                is_question=True,
                question_type="clarification",
                detected_concerns=[],
                emotional_state="engaged",
                requires_response=True,
                is_complex_question=False,
                key_topics=[]
            )
            alignment = AlignmentAnalysis(
                alignment_score=0.7,
                addressed_points=[],
                remaining_points=[],
                detected_opportunities=["address directly"],
                suggested_pivot=None,
                urgency_level="medium",
                next_best_action="respond persuasively"
            )

            async def _generate_audio():
                response = await janus.response_generator.generate(
                    transcript=grounded_input,
                    analysis=analysis,
                    alignment=alignment,
                    objective=janus.current_objective,
                    history=janus.conversation_history
                )
                reply_text = response.prosody_text if verbose else response.text
                if not reply_text or len(reply_text) < 4:
                    reply_text = "Got it."
                # Select session voice if available
                voice_profile = _session_voice_profiles.get(session_id) or response.voice_profile
                audio_bytes = await janus.audio_generator.generate(
                    reply_text,
                    response.prosody_tokens,
                    voice_profile
                )
                return audio_bytes or b""

            audio = asyncio.run(_generate_audio())
            if audio:
                dur = len(audio) / (24000 * 2)
                print(f"[SS] tts bytes={len(audio)} dur={dur:.2f}s")
                _split_and_enqueue(session_id, audio)
            else:
                print("[SS] tts_empty")
            _last_answered_qpos[session_id] = q_end
        except Exception as e:
            print(f"[SS] tts_err {type(e).__name__}: {str(e)[:120]}")
        finally:
            _tts_workers_active[session_id] = False

    # Launch worker
    try:
        _tts_workers_active[session_id] = True
        threading.Thread(target=worker, name=f"tts-{session_id}", daemon=True).start()
    except Exception as e:
        _tts_workers_active[session_id] = False
        print(f"[SS] tts_thread_err {type(e).__name__}: {str(e)[:80]}")

@bp.route('/sessions/<session_id>/upload_audio', methods=['POST'])
def upload_audio_chunk(session_id):
    """
    Receives audio chunks from the client (PCM16, mono, 16kHz).
    Saves them to a WAV file for the session.
    """
    # Validate session exists
    if not os.path.exists(_session_path(session_id)):
        return {"error": "session not found"}, 404
    
    # Get raw data from request body (timestamp + PCM16 data)
    raw_data = request.get_data()
    
    if len(raw_data) < 8:
        return {"error": "invalid packet: too small"}, 400
    
    # Extract timestamp (first 8 bytes, Int64 little-endian, milliseconds)
    timestamp_ms = struct.unpack('<q', raw_data[:8])[0]
    
    # Calculate latency
    current_time_ms = int(time.time() * 1000)
    latency_ms = current_time_ms - timestamp_ms
    
    # Extract PCM audio data (remaining bytes)
    pcm_data = raw_data[8:]
    
    if len(pcm_data) == 0:
        return {"error": "empty audio data"}, 400
    
    # Initialize buffer and packet directory for this session if needed
    if session_id not in _session_audio_buffers:
        _session_audio_buffers[session_id] = deque(maxlen=200)  # Keep last ~4 seconds at 50 chunks/sec
        _upload_counts[session_id] = 0
        _latency_stats[session_id] = {'min': float('inf'), 'max': 0, 'sum': 0, 'count': 0}
        _ensure_output_queue(session_id)
        _last_answered_qpos[session_id] = -1
        
        # Create packets directory for this session
        uploads_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
        timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        packet_dir = os.path.join(uploads_dir, f'packets-{session_id}-{timestamp}')
        os.makedirs(packet_dir, exist_ok=True)
        _session_packet_dirs[session_id] = packet_dir
        
        print(f"[SS] dir {os.path.basename(packet_dir)}")
        
        # Initialize transcription service for this session
        # Ensure transcripts are stored under backend/data/sessions
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # .../backend
        data_dir = os.path.join(project_root, 'data', 'sessions')
        os.makedirs(data_dir, exist_ok=True)
        # path: backend/data/sessions/sess_<id>_transcript.txt
        transcript_path = os.path.join(data_dir, f'{session_id}_transcript.txt')
        
        try:
            transcription_service = WhisperTranscriptionService(
                session_id=session_id,
                packet_dir=packet_dir,
                transcript_path=transcript_path,
                packets_per_transcription=75,  # kept for compatibility
                chunks_per_file=50  # write 50 packets per WAV, transcribe, delete, loop
            )
            _transcription_services[session_id] = transcription_service
            print(f"[SS] trx_ready {session_id}")
        except Exception as e:
            print(f"[stream_sessions] Error initializing transcription service: {e}")
        
        # Initialize audio output stream for playback through speakers
        try:
            audio_stream = sd.OutputStream(
                samplerate=24000,
                channels=1,
                dtype='int16',
                blocksize=0  # Use variable block size
            )
            audio_stream.start()
            _session_audio_streams[session_id] = audio_stream
            print(f"[SS] spk_ready {session_id}")
        except Exception as e:
            print(f"[stream_sessions] Error starting audio playback: {e}")
    
    # Increment upload count
    _upload_counts[session_id] += 1
    count = _upload_counts[session_id]
    
    # Add to buffer
    _session_audio_buffers[session_id].append(pcm_data)
    
    # Add packet to transcription service
    if session_id in _transcription_services:
        try:
            transcript = _transcription_services[session_id].add_packet(pcm_data)
            if transcript:
                print(f"[SS] trx_upd '{transcript[:48]}{'…' if len(transcript)>48 else ''}'")
        except Exception as e:
            print(f"[stream_sessions] Error adding packet to transcription: {e}")
    
    # Update latency statistics
    stats = _latency_stats[session_id]
    stats['min'] = min(stats['min'], latency_ms)
    stats['max'] = max(stats['max'], latency_ms)
    stats['sum'] += latency_ms
    stats['count'] += 1
    avg_latency = stats['sum'] / stats['count']
    
    # Log first few and periodic chunks with latency
    if count <= 3 or count % 50 == 0:
        bytes_preview = " ".join(f"{b:02x}" for b in pcm_data[:16])
        print(f"[SS] pkt {count} lat={latency_ms}ms bytes={len(pcm_data)} avg={avg_latency:.1f}ms")
    
    # Save packet to individual file
    packet_filename = None
    try:
        if session_id in _session_packet_dirs:
            packet_filename = os.path.join(
                _session_packet_dirs[session_id],
                f'packet_{count:06d}_latency{latency_ms}ms_ts{timestamp_ms}.pcm'
            )
            with open(packet_filename, 'wb') as f:
                f.write(pcm_data)
            if count <= 3 or count % 50 == 0:
                print(f"[SS] pkt_save {os.path.basename(packet_filename)}")
    except Exception as e:
        print(f"[stream_sessions] Error saving packet #{count}: {e}")
    
    # Play audio through speakers by reading the saved file
    # try:
    #     if session_id in _session_audio_streams and packet_filename and os.path.exists(packet_filename):
    #         # Read the packet file we just saved
    #         with open(packet_filename, 'rb') as f:
    #             file_pcm_data = f.read()
            
    #         # Convert PCM bytes to numpy array for sounddevice
    #         audio_array = np.frombuffer(file_pcm_data, dtype=np.int16)
    #         _session_audio_streams[session_id].write(audio_array)
    #         if count <= 3 or count % 50 == 0:
    #             print(f"[SS] spk_play {count}")
    # except Exception as e:
    #     print(f"[stream_sessions] Error playing audio chunk #{count}: {e}")
    
    # Trigger TTS if a fresh question is detected (event-driven)
    try:
        settings = read_settings()
        verbose = bool(settings.get('verbose', False))
    except Exception:
        verbose = False
    _maybe_trigger_tts(session_id, verbose=verbose)

    return {"status": "ok", "bytes_received": len(pcm_data)}, 200


@bp.route('/sessions/<session_id>/stream_audio')
def stream_audio(session_id):
    """
    Streams audio back to the client as a continuous WAV stream.
    The client can play this with AVPlayer or AVAudioEngine.
    """
    # Validate session exists
    if not os.path.exists(_session_path(session_id)):
        return {"error": "session not found"}, 404
    
    print(f"[SS] stream_start {session_id}")
    
    def generate_wav_stream():
        """
        Generator that yields WAV header followed by PCM16 chunks.
        This creates a continuous audio stream.
        """
        # Send WAV header first (44 bytes for a basic WAV with unknown size)
        # We'll use a "streaming" WAV format with max size
        sample_rate = 24000
        num_channels = 1
        bytes_per_sample = 2
        
        # Create WAV header
        wav_header = io.BytesIO()
        with wave.open(wav_header, 'wb') as wav:
            wav.setnchannels(num_channels)
            wav.setsampwidth(bytes_per_sample)
            wav.setframerate(sample_rate)
            # Write a dummy frame to create header, we'll stream the rest
            wav.writeframes(b'\x00\x00')
        
        # Get the header (modify to support streaming)
        header_bytes = wav_header.getvalue()
        # Modify the RIFF chunk size to maximum (0xFFFFFFFF for streaming)
        header_bytes = bytearray(header_bytes)
        header_bytes[4:8] = (0xFFFFFFFF).to_bytes(4, 'little')
        header_bytes[40:44] = (0xFFFFFFFF).to_bytes(4, 'little')
        
        print(f"[SS] wav_hdr 44B {session_id}")
        yield bytes(header_bytes[:44])  # Send just the 44-byte header
        
        # Mark session as streaming
        _session_states[session_id] = {'streaming': True, 'start_time': time.time()}
        
        # Pure consumer loop: pop from outgoing queue and stream
        try:
            print(f"[SS] stream_consume {session_id}")
            # Initial prime: small silence to start playback promptly
            yield b"\x00\x00" * int(24000 * 0.1)
            frame_count = 0
            while _session_states.get(session_id, {}).get('streaming', False):
                chunk = None
                _ensure_output_queue(session_id)
                cond = _session_output_conds[session_id]
                q = _session_output_queues[session_id]
                with cond:
                    if not q:
                        # Wait briefly for new data
                        cond.wait(timeout=0.25)
                    if q:
                        chunk = q.popleft()
                if chunk:
                    yield chunk
                    frame_count += 1
                    if frame_count % 200 == 0:
                        print(f"[SS] stream_chunks {frame_count}")
                    # Pace approximately to real-time
                    time.sleep(max(0.0, (len(chunk) / bytes_per_sample) / 24000.0))
                else:
                    # Keepalive silence to avoid client timeouts
                    keepalive = b"\x00\x00" * int(24000 * 0.05)
                    yield keepalive
                    time.sleep(0.05)
        except GeneratorExit:
            print(f"[SS] stream_client_disconnected {session_id}")
        finally:
            # Cleanup
            print(f"[SS] stream_stop {session_id}")
            if session_id in _session_states:
                _session_states[session_id]['streaming'] = False
    
    return Response(
        stream_with_context(generate_wav_stream()),
        mimetype='audio/wav',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',  # Disable nginx buffering if behind nginx
        }
    )


@bp.route('/sessions/<session_id>/stop_stream', methods=['POST'])
def stop_stream(session_id):
    """
    Stops the audio stream for a session and closes the recording file.
    """
    print(f"[SS] stop_req {session_id}")
    
    if session_id in _session_states:
        _session_states[session_id]['streaming'] = False
    
    # Finalize transcription (transcribe remaining audio)
    if session_id in _transcription_services:
        try:
            transcription_service = _transcription_services[session_id]
            final_transcript = transcription_service.finalize()
            if final_transcript:
                print(f"[SS] trx_final '{final_transcript[:48]}{'…' if len(final_transcript)>48 else ''}'")
            
            stats = transcription_service.get_stats()
            print(f"[SS] trx_done {stats['duration_seconds']:.2f}s -> {stats['transcript_path']}")
            
            del _transcription_services[session_id]
        except Exception as e:
            print(f"[stream_sessions] Error finalizing transcription: {e}")
    
    # Clean up packet directory reference
    if session_id in _session_packet_dirs:
        packet_dir = _session_packet_dirs[session_id]
        print(f"[SS] pkt_dir {packet_dir}")
        del _session_packet_dirs[session_id]
    
    # Stop and close audio playback stream
    if session_id in _session_audio_streams:
        try:
            _session_audio_streams[session_id].stop()
            _session_audio_streams[session_id].close()
            print(f"[SS] spk_close {session_id}")
        except Exception as e:
            print(f"[stream_sessions] Error closing audio playback: {e}")
        del _session_audio_streams[session_id]
    
    # Clean up buffers
    if session_id in _session_audio_buffers:
        del _session_audio_buffers[session_id]
    
    # Clean up upload counts and print latency stats
    if session_id in _upload_counts:
        total_chunks = _upload_counts[session_id]
        duration_seconds = (total_chunks * 3200) / (16000 * 2)  # 3200 bytes / (16000 Hz * 2 bytes/sample)
        print(f"[SS] stats chunks={total_chunks} dur={duration_seconds:.1f}s")
        del _upload_counts[session_id]
    
    # Print final latency statistics
    if session_id in _latency_stats:
        stats = _latency_stats[session_id]
        if stats['count'] > 0:
            avg_latency = stats['sum'] / stats['count']
            print(f"[SS] latency avg={avg_latency:.1f} min={stats['min']} max={stats['max']} n={stats['count']}")
        del _latency_stats[session_id]

    return {"status": "stopped"}, 200

