import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import random
import subprocess
import shutil
import wave
import tempfile

import numpy as np
import gradio as gr

# ── dirs ──────────────────────────────────────────────────────────────────────
os.makedirs("temp/video",   exist_ok=True)
os.makedirs("temp/audio",   exist_ok=True)
os.makedirs("outputs",      exist_ok=True)
os.makedirs("results",      exist_ok=True)

# ── helpers ───────────────────────────────────────────────────────────────────
def write_wav(path: str, audio_tuple):
    """Write (sample_rate, samples_int16) from gr.Audio to a wav file."""
    sr, data = audio_tuple
    if data.ndim > 1:
        data = data[:, 0]
    data = data.astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data.tobytes())


def resample_wav(src: str, dst: str, target_sr: int = 16000):
    subprocess.check_call([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", src, "-ac", "1", "-ar", str(target_sr), "-c:a", "pcm_s16le", dst
    ])


# ── STT ───────────────────────────────────────────────────────────────────────
def transcribe(audio_path: str, model_size: str = "base") -> str:
    """Transcribe audio using OpenAI Whisper."""
    try:
        import whisper
        model = whisper.load_model(model_size)
        result = model.transcribe(audio_path)
        return result["text"].strip()
    except ImportError:
        raise gr.Error("whisper not installed. Run: pip install openai-whisper")


# ── LLM ───────────────────────────────────────────────────────────────────────
def llm_respond(user_text: str, history: list, system_prompt: str, model: str) -> str:
    """Call Ollama and return assistant reply."""
    try:
        from ollama import chat
        messages = [{"role": "system", "content": system_prompt}]
        for h in history:
            messages.append({"role": "user",      "content": h[0]})
            messages.append({"role": "assistant", "content": h[1]})
        messages.append({"role": "user", "content": user_text})

        reply = ""
        for chunk in chat(model=model, messages=messages, stream=True):
            reply += chunk["message"]["content"]
        return reply.strip()
    except ImportError:
        raise gr.Error("ollama not installed. Run: pip install ollama")
    except Exception as e:
        raise gr.Error(f"Ollama error (is it running?): {e}")


# ── TTS ───────────────────────────────────────────────────────────────────────
def tts_synthesize(text: str, engine: str, piper_model: str) -> str:
    """Synthesize text → wav file path."""
    out_wav = os.path.join("outputs", f"tts_{random.randint(0,99999)}.wav")

    if engine == "gTTS (online)":
        try:
            from gtts import gTTS
            tts = gTTS(text=text, lang="en")
            mp3 = out_wav.replace(".wav", ".mp3")
            tts.save(mp3)
            subprocess.check_call([
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", mp3, "-ac", "1", "-ar", "22050", out_wav
            ])
            os.remove(mp3)
        except ImportError:
            raise gr.Error("gTTS not installed. Run: pip install gtts")

    elif engine == "pyttsx3 (offline)":
        try:
            import pyttsx3
            engine_obj = pyttsx3.init()
            engine_obj.save_to_file(text, out_wav)
            engine_obj.runAndWait()
        except ImportError:
            raise gr.Error("pyttsx3 not installed. Run: pip install pyttsx3")

    elif engine == "Piper (offline, high quality)":
        if not piper_model or not os.path.isfile(piper_model):
            raise gr.Error(f"Piper model not found: {piper_model}")
        try:
            from piper import PiperVoice
            voice = PiperVoice.load(piper_model)
            pcm = bytearray()
            sr = None
            for chunk in voice.synthesize(text):
                sr = chunk.sample_rate
                pcm.extend(chunk.audio_int16_bytes)
            with wave.open(out_wav, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(bytes(pcm))
        except ImportError:
            raise gr.Error("piper-tts not installed. Run: pip install piper-tts")

    return out_wav


# ── Lip Sync ──────────────────────────────────────────────────────────────────
def run_lipsync(face_video: str, audio_wav: str, img_size: int) -> str:
    out_mp4 = os.path.join("outputs", f"lipsync_{random.randint(0,99999)}.mp4")
    cmd = [
        "python", "inference.py",
        "--face",    face_video,
        "--audio",   audio_wav,
        "--outfile", out_mp4,
        "--img_size", str(img_size),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise gr.Error(f"inference.py failed:\n{result.stderr[-1000:]}")
    return out_mp4


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Batch Lip Sync
# ─────────────────────────────────────────────────────────────────────────────
def batch_lipsync(segment_length, video, audio, progress=gr.Progress()):
    if segment_length is None:
        segment_length = 0

    if segment_length != 0:
        video_segments = cut_video_segments(video, segment_length)
        audio_segments = cut_audio_segments(audio, segment_length)
    else:
        video_path = os.path.join("temp/video", os.path.basename(video))
        shutil.move(video, video_path)
        video_segments = [video_path]
        audio_path = os.path.join("temp/audio", os.path.basename(audio))
        shutil.move(audio, audio_path)
        audio_segments = [audio_path]

    processed_segments = []
    for i, (video_seg, audio_seg) in progress.tqdm(
        enumerate(zip(video_segments, audio_segments)), desc="Processing"
    ):
        out = process_segment(video_seg, audio_seg, i)
        processed_segments.append(out)

    output_file = f"results/output_{random.randint(0,1000)}.mp4"
    concatenate_videos(processed_segments, output_file)
    cleanup_temp_files(video_segments + audio_segments)
    return output_file


def cleanup_temp_files(file_list):
    for f in file_list:
        if os.path.isfile(f):
            os.remove(f)


def cut_video_segments(video_file, segment_length):
    d = "temp/audio"
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    tpl = f"{d}/{random.randint(0,1000)}_%03d.mp4"
    subprocess.run(["ffmpeg", "-i", video_file, "-c", "copy", "-f",
                    "segment", "-segment_time", str(segment_length), tpl], check=True)
    return [tpl % i for i in range(len(os.listdir(d)))]


def cut_audio_segments(audio_file, segment_length):
    d = "temp/video"
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    tpl = f"{d}/{random.randint(0,1000)}_%03d.mp3"
    subprocess.run(["ffmpeg", "-i", audio_file, "-f", "segment",
                    "-segment_time", str(segment_length), tpl], check=True)
    return [tpl % i for i in range(len(os.listdir(d)))]


def process_segment(video_seg, audio_seg, i):
    out = f"results/{random.randint(10,100000)}_{i}.mp4"
    subprocess.run(["python", "inference.py",
                    "--face", video_seg, "--audio", audio_seg, "--outfile", out], check=True)
    return out


def concatenate_videos(segments, output_file):
    with open("segments.txt", "w") as f:
        for s in segments:
            f.write(f"file '{s}'\n")
    subprocess.run(["ffmpeg", "-f", "concat", "-i", "segments.txt", "-c", "copy", output_file], check=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Voice Agent
# ─────────────────────────────────────────────────────────────────────────────
def voice_agent_pipeline(
    audio_input,
    face_video,
    chat_history,
    system_prompt,
    ollama_model,
    tts_engine,
    piper_model_path,
    whisper_size,
    img_size,
    progress=gr.Progress(),
):
    if audio_input is None:
        raise gr.Error("Please record your voice first.")
    if not face_video:
        raise gr.Error("Please select a face video.")

    # ── Step 1: Save mic audio ──
    progress(0.1, desc="Saving audio...")
    raw_wav = os.path.join("outputs", f"mic_{random.randint(0,99999)}.wav")
    if isinstance(audio_input, tuple):
        write_wav(raw_wav, audio_input)
    else:
        shutil.copy(audio_input, raw_wav)

    # ── Step 2: STT ──
    progress(0.2, desc="Transcribing speech...")
    wav_16k = raw_wav.replace(".wav", "_16k.wav")
    resample_wav(raw_wav, wav_16k, 16000)
    user_text = transcribe(wav_16k, model_size=whisper_size)
    if not user_text:
        raise gr.Error("Could not transcribe audio. Please speak more clearly.")

    # ── Step 3: LLM ──
    progress(0.4, desc="Generating response...")
    assistant_text = llm_respond(user_text, chat_history, system_prompt, ollama_model)

    # ── Step 4: TTS ──
    progress(0.6, desc="Synthesizing voice...")
    tts_wav = tts_synthesize(assistant_text, tts_engine, piper_model_path)
    tts_wav_16k = tts_wav.replace(".wav", "_16k.wav")
    resample_wav(tts_wav, tts_wav_16k, 16000)

    # ── Step 5: Lip Sync ──
    progress(0.75, desc="Generating lip-sync video (this takes ~30–60s)...")
    out_video = run_lipsync(face_video, tts_wav_16k, img_size)

    # ── Update history ──
    chat_history = chat_history + [[user_text, assistant_text]]

    progress(1.0, desc="Done!")
    return chat_history, out_video, user_text, assistant_text


# ─────────────────────────────────────────────────────────────────────────────
# BUILD UI
# ─────────────────────────────────────────────────────────────────────────────
FACE_EXAMPLES = [
    os.path.join(os.path.dirname(__file__), f"examples/face/{n}.mp4")
    for n in ["1", "2", "3", "4", "5"]
    if os.path.exists(os.path.join(os.path.dirname(__file__), f"examples/face/{n}.mp4"))
]

with gr.Blocks(
    title="VideoReTalking",
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.green,
        font=["Source Sans Pro", "Arial", "sans-serif"],
        font_mono=["JetBrains Mono", "Consolas", "Courier New"],
    ),
) as demo:

    gr.Markdown("# 🎭 VideoReTalking — Audio-based Lip Synchronization")

    with gr.Tabs():

        # ── TAB 1 ──────────────────────────────────────────────────────────
        with gr.Tab("📂 Batch Lip Sync"):
            gr.Markdown("Upload a face video and an audio file to generate a lip-synced output.")
            with gr.Row():
                with gr.Column():
                    seg_len = gr.Number(label="Segment length (seconds) — 0 = no segmentation", value=0)
                    with gr.Row():
                        face_input = gr.Video(label="Face Video")
                        audio_input = gr.Audio(type="filepath", label="Target Audio")
                    run_btn = gr.Button("▶ Synthesize", variant="primary")
                    gr.Examples(label="Face Examples", examples=FACE_EXAMPLES, inputs=[face_input])
                    gr.Examples(
                        label="Audio Examples",
                        examples=[
                            os.path.join(os.path.dirname(__file__), "examples/audio/1.wav"),
                            os.path.join(os.path.dirname(__file__), "examples/audio/2.wav"),
                        ],
                        inputs=[audio_input],
                    )
                with gr.Column():
                    batch_output = gr.Video(label="Output Video")

            run_btn.click(fn=batch_lipsync, inputs=[seg_len, face_input, audio_input], outputs=[batch_output])

        # ── TAB 2 ──────────────────────────────────────────────────────────
        with gr.Tab("🎤 Voice Agent"):
            gr.Markdown(
                "Speak into the microphone → your speech is transcribed → "
                "an AI responds → the response is spoken and lip-synced onto the face video."
            )

            with gr.Row():
                # ── Left column: inputs & settings ────────────────────────
                with gr.Column(scale=1):
                    gr.Markdown("### 🎬 Face Video")
                    agent_face = gr.Video(label="Select Face Video")
                    gr.Examples(label="Examples", examples=FACE_EXAMPLES, inputs=[agent_face])

                    gr.Markdown("### 🎙️ Your Voice")
                    mic_input = gr.Audio(sources=["microphone"], type="numpy", label="Record")

                    send_btn = gr.Button("🚀 Send & Generate", variant="primary", size="lg")

                    with gr.Accordion("⚙️ Settings", open=False):
                        system_prompt = gr.Textbox(
                            label="System Prompt",
                            value="You are a concise voice assistant. Always answer the user's query in 1-3 sentences.",
                            lines=3,
                        )
                        ollama_model = gr.Textbox(label="Ollama Model", value="gemma3")
                        tts_engine = gr.Dropdown(
                            label="TTS Engine",
                            choices=["gTTS (online)", "pyttsx3 (offline)", "Piper (offline, high quality)"],
                            value="gTTS (online)",
                        )
                        piper_model_path = gr.Textbox(
                            label="Piper ONNX model path (only if Piper selected)",
                            value="./models/en_US-lessac-medium.onnx",
                        )
                        whisper_size = gr.Dropdown(
                            label="Whisper Model Size",
                            choices=["tiny", "base", "small", "medium", "large"],
                            value="base",
                        )
                        img_size = gr.Slider(
                            label="Image Size (larger = slower but better quality)",
                            minimum=256, maximum=512, step=64, value=384,
                        )

                # ── Right column: outputs ──────────────────────────────────
                with gr.Column(scale=1):
                    gr.Markdown("### 💬 Conversation")
                    chatbot = gr.Chatbot(label="Chat History", height=300)

                    gr.Markdown("### 📝 Last Turn")
                    with gr.Row():
                        user_text_box  = gr.Textbox(label="You said",         interactive=False)
                        agent_text_box = gr.Textbox(label="Agent replied",    interactive=False)

                    gr.Markdown("### 🎬 Lip-Synced Response")
                    agent_video = gr.Video(label="Generated Video")

                    clear_btn = gr.Button("🗑️ Clear History", variant="secondary")

            send_btn.click(
                fn=voice_agent_pipeline,
                inputs=[
                    mic_input, agent_face, chatbot,
                    system_prompt, ollama_model,
                    tts_engine, piper_model_path,
                    whisper_size, img_size,
                ],
                outputs=[chatbot, agent_video, user_text_box, agent_text_box],
            )
            clear_btn.click(fn=lambda: ([], None, "", ""), outputs=[chatbot, agent_video, user_text_box, agent_text_box])


demo.queue().launch()
