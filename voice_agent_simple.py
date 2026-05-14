# -*- coding: utf-8 -*-
"""
Hands-Free Voice Agent
======================
Listens continuously -> detects speech automatically -> stops when you go silent
-> Whisper STT -> Ollama Cloud LLM -> gTTS TTS -> VideoReTalking lip-sync

No buttons. No Enter. Just speak.

Run:
  $env:KMP_DUPLICATE_LIB_OK="TRUE"
  $env:TORCHDYNAMO_DISABLE="1"
  & "C:\\Users\\DESK0030\\conda_envs\\video_retalking\\python.exe" voice_agent_simple.py
"""

import os, sys, time, wave, subprocess, platform, traceback, io
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

# Force UTF-8 output on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

def _safe_excepthook(exc_type, exc_value, exc_tb):
    msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    safe = msg.encode("ascii", errors="replace").decode("ascii")
    print(safe, file=sys.stderr, flush=True)
sys.excepthook = _safe_excepthook

import numpy as np

# ── CONFIG ─────────────────────────────────────────────────────────────────────
FACE_VIDEO      = os.path.abspath("examples/face/women.mp4")
OUT_DIR         = os.path.abspath("outputs")

# Ollama Cloud (https://docs.ollama.com/cloud)
OLLAMA_API_KEY  = "20e8614648a846a78aad0eb90beeddb3.Iu3v9IluH-v48Vhj803osISB"
OLLAMA_HOST     = "https://ollama.com"
OLLAMA_MODEL    = "gpt-oss:120b"

SYSTEM_PROMPT   = "You are a concise voice assistant. Reply in 1-3 short sentences."
WHISPER_MODEL   = "base"

IMG_SIZE        = 256
LNET_BATCH_SIZE = 4

# VAD / recording settings
SAMPLE_RATE       = 16000
VAD_CHUNK_SAMPLES = 512          # exactly 512 samples required by Silero VAD v6 at 16kHz (32ms)
VAD_CHUNK_MS      = VAD_CHUNK_SAMPLES * 1000 // SAMPLE_RATE   # = 32 ms
SILENCE_AFTER_SEC = 1.2          # stop recording this many seconds after last speech
MIN_SPEECH_SEC    = 0.4          # ignore clips shorter than this
MAX_RECORD_SEC    = 30           # hard cap to avoid runaway recording
# ───────────────────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)


# ── VAD: record until silence ──────────────────────────────────────────────────
def listen_until_silence() -> np.ndarray:
    """
    Streams mic audio in real-time. Uses Silero VAD to detect speech.
    Returns int16 numpy array at 16kHz once the speaker goes silent.
    """
    import torch
    from silero_vad import load_silero_vad
    import pyaudio

    vad_model = load_silero_vad()

    chunk_samples = VAD_CHUNK_SAMPLES   # 512 samples @ 32ms — required by Silero VAD v6
    pa = pyaudio.PyAudio()
    stream = pa.open(
        rate=SAMPLE_RATE,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=chunk_samples,
    )

    print("[VAD] Listening... speak now.", flush=True)

    all_chunks  = []       # raw int16 bytes per chunk
    speech_buf  = []       # indices of chunks where speech was detected
    recording   = False    # True once we hear first speech
    silence_start = None   # time when speech last ended
    started_at  = None

    try:
        while True:
            raw = stream.read(chunk_samples, exception_on_overflow=False)
            all_chunks.append(raw)

            # Convert to float32 tensor for Silero
            audio_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            audio_t  = torch.from_numpy(audio_np)

            speech_prob = vad_model(audio_t, SAMPLE_RATE).item()
            is_speech   = speech_prob > 0.5

            now = time.time()

            if is_speech:
                if not recording:
                    print("[VAD] Speech detected, recording...", flush=True)
                    recording    = True
                    started_at   = now
                silence_start = None          # reset silence timer
                speech_buf.append(len(all_chunks) - 1)

            elif recording:
                if silence_start is None:
                    silence_start = now
                elif (now - silence_start) >= SILENCE_AFTER_SEC:
                    print("[VAD] Silence detected, processing...", flush=True)
                    break

            # Hard cap
            if recording and started_at and (now - started_at) >= MAX_RECORD_SEC:
                print("[VAD] Max recording length reached.", flush=True)
                break

    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

    if not speech_buf:
        return np.array([], dtype=np.int16)

    # Include a small buffer around detected speech for natural feel
    pad = int(0.3 * SAMPLE_RATE / VAD_CHUNK_SAMPLES)   # ~300ms padding in chunks
    start_idx = max(0, speech_buf[0] - pad)
    end_idx   = min(len(all_chunks) - 1, speech_buf[-1] + pad)
    audio_bytes = b"".join(all_chunks[start_idx:end_idx + 1])
    return np.frombuffer(audio_bytes, dtype=np.int16)


# ── save WAV ───────────────────────────────────────────────────────────────────
def save_wav(path: str, samples: np.ndarray, sr: int = SAMPLE_RATE):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.astype(np.int16).tobytes())


# ── STT ────────────────────────────────────────────────────────────────────────
def transcribe(wav_path: str) -> str:
    import whisper
    print("[STT] Transcribing...", flush=True)
    try:
        model = whisper.load_model(WHISPER_MODEL)
        result = model.transcribe(wav_path, language="en", fp16=False)
        text = (result.get("text") or "").strip()
        safe = text.encode("ascii", errors="replace").decode("ascii")
        print(f"[STT] You said: {safe}", flush=True)
        return text
    except Exception:
        err = traceback.format_exc().encode("ascii", errors="replace").decode("ascii")
        print(f"[STT ERROR]\n{err}", flush=True)
        return ""


# ── LLM ────────────────────────────────────────────────────────────────────────
def ask_ollama(user_text: str, history: list) -> str:
    from ollama import Client
    print("[LLM] Asking Ollama Cloud...", flush=True)
    client = Client(
        host=OLLAMA_HOST,
        headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"},
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})

    reply = ""
    for chunk in client.chat(OLLAMA_MODEL, messages=messages, stream=True):
        piece = chunk["message"]["content"] or ""
        sys.stdout.write(piece)
        sys.stdout.flush()
        reply += piece
    print("\n", flush=True)
    return reply.strip()


# ── TTS ────────────────────────────────────────────────────────────────────────
def tts_gtts(text: str, out_wav: str):
    from gtts import gTTS
    print("[TTS] Synthesizing speech...", flush=True)
    mp3_path = out_wav.replace(".wav", ".mp3")
    gTTS(text=text, lang="en").save(mp3_path)
    subprocess.check_call([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", mp3_path,
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        out_wav
    ])
    os.remove(mp3_path)


# ── Lip-sync ───────────────────────────────────────────────────────────────────
def run_lipsync(audio_wav: str, out_mp4: str):
    print("[LIPSYNC] Running VideoReTalking (~3-5 min)...", flush=True)
    python_exe = os.path.join(
        r"C:\Users\DESK0030\conda_envs\video_retalking", "python.exe"
    )
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["TORCHDYNAMO_DISABLE"]  = "1"
    result = subprocess.run(
        [python_exe, "inference.py",
         "--face",    FACE_VIDEO,
         "--audio",   audio_wav,
         "--outfile", out_mp4,
         "--img_size",        str(IMG_SIZE),
         "--LNet_batch_size", str(LNET_BATCH_SIZE),
         "--fast"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"inference.py failed (exit {result.returncode})")
    print(f"[LIPSYNC] Saved: {out_mp4}", flush=True)


def play_video(mp4_path: str):
    try:
        os.startfile(mp4_path)
    except Exception:
        subprocess.Popen(["start", mp4_path], shell=True)


# ── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    # Startup check
    try:
        from ollama import Client
        c = Client(host=OLLAMA_HOST, headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"})
        c.chat(OLLAMA_MODEL, messages=[{"role":"user","content":"hi"}], stream=False)
        print(f"[OK] Ollama Cloud ready. Model: {OLLAMA_MODEL}", flush=True)
    except Exception as e:
        print(f"[ERROR] Ollama Cloud not reachable: {e}")
        print("  Get a key at: https://ollama.com/settings/keys")
        sys.exit(1)

    if not os.path.isfile(FACE_VIDEO):
        print(f"[ERROR] Face video not found: {FACE_VIDEO}")
        sys.exit(1)

    print("\n" + "="*60)
    print("  Hands-Free Voice Agent -- VideoReTalking + Ollama Cloud")
    print("="*60)
    print(f"  LLM    : {OLLAMA_MODEL} (cloud)")
    print(f"  STT    : Whisper ({WHISPER_MODEL})")
    print(f"  TTS    : gTTS")
    print(f"  Face   : {os.path.basename(FACE_VIDEO)}")
    print(f"  Silence: {SILENCE_AFTER_SEC}s to stop recording")
    print("  Just SPEAK. Stop talking to trigger the pipeline.")
    print("  Ctrl+C to quit.")
    print("="*60 + "\n")

    history = []
    turn    = 0

    while True:
        print(f"\n--- Turn {turn+1} ---  (waiting for your voice...)", flush=True)

        # 1. Listen until silence
        samples = listen_until_silence()

        if len(samples) < int(MIN_SPEECH_SEC * SAMPLE_RATE):
            print("[VAD] Too short, ignoring.", flush=True)
            continue

        mic_wav = os.path.join(OUT_DIR, f"mic_{turn}.wav")
        save_wav(mic_wav, samples)

        # 2. STT
        user_text = transcribe(mic_wav)
        if not user_text:
            print("[WARN] Nothing transcribed, listening again...", flush=True)
            continue

        # 3. LLM
        assistant_text = ask_ollama(user_text, history)
        if not assistant_text:
            print("[WARN] LLM returned empty, listening again...", flush=True)
            continue

        history.append({"role": "user",      "content": user_text})
        history.append({"role": "assistant", "content": assistant_text})

        # 4. TTS
        tts_wav = os.path.join(OUT_DIR, f"tts_{turn}.wav")
        tts_gtts(assistant_text, tts_wav)

        # 5. Lip-sync
        out_mp4 = os.path.join(OUT_DIR, f"response_{turn}.mp4")
        t0 = time.time()
        run_lipsync(tts_wav, out_mp4)
        print(f"[DONE] Render time: {time.time()-t0:.1f}s", flush=True)

        # 6. Play
        print(f"[PLAY] Opening: {out_mp4}", flush=True)
        play_video(out_mp4)

        # Free GPU memory before next turn
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        turn += 1


if __name__ == "__main__":
    main()
