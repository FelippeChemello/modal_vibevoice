import modal

model_name = "vibevoice/VibeVoice-7B"

def download_model():
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
    from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
    import torch
    print("Downloading VibeVoice model...")
    VibeVoiceProcessor.from_pretrained(model_name, trust_remote_code=True)
    VibeVoiceForConditionalGenerationInference.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
        trust_remote_code=True
    )
    print("VibeVoice model downloaded.")

hf_cache = modal.Volume.from_name("hf_hub_cache", create_if_missing=True)
CACHE_DIR = "/cache"

image = (
    modal.Image.debian_slim(python_version="3.10")
        .apt_install("git")
        .uv_pip_install(
            "fastapi",
            "uvicorn[standard]",
            "python-dotenv",
            "requests",
            "numpy",
            "rich",
            "tensorboardX",
            "silero_vad",
            "librosa",
            "soundfile",
            "torchvision",
            "torchaudio",
            "torch",
            "transformers",
            "huggingface-hub",
            "tokenizers",
        )
        .uv_pip_install("vibevoice @ git+https://github.com/vibevoice-community/VibeVoice/")
        .run_commands("mkdir -p data logs training_runs")
        .env({ "HF_HOME": CACHE_DIR, "HF_HUB_CACHE": CACHE_DIR })
        .run_function(
            download_model,
            gpu="A10",
            volumes={ CACHE_DIR: hf_cache },
        )
        .add_local_dir("./voices", remote_path="/app/voices")
)

app = modal.App('vibevoice')

with image.imports():
    import os, tempfile, torch, time, re, librosa, soundfile as sf, numpy as np
    from abc import ABC, abstractmethod
    from dataclasses import dataclass
    from typing import List
    from fastapi import Header
    from fastapi.responses import Response, JSONResponse
    from pydantic import BaseModel
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
    from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference

@dataclass
class GenerationResult:
    audio_path: str
    audio_duration: float
    generation_time: float
    rtf: float
    input_tokens: int
    generated_tokens: int

class TTSModel(ABC):
    @abstractmethod
    def generate(self, text: str, voice_samples: List[str], output_path: str, **kwargs) -> GenerationResult: 
        pass

class TTSRequest(BaseModel):
    text: str
    voices: List[str]

@app.cls(
    gpu='A10',
    image=image,
    timeout=180,
    secrets=[modal.Secret.from_name("vibevoice-secret")],
    # **mesmo volume, mesmo path**
    volumes={CACHE_DIR: hf_cache},
)
class Model:
    @modal.enter()
    def load_model(self):
        os.environ.setdefault("HF_HOME", CACHE_DIR)
        os.environ.setdefault("HF_HUB_CACHE", CACHE_DIR)

        self.processor = VibeVoiceProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="sdpa",
            trust_remote_code=True,
            local_files_only=True,
        )
        self.voices_dir = "/app/voices"

    def generate(self, text: str, num_speakers: int, voices: list[str]):
        speakers = voices[:num_speakers]
        print(f"Using speakers: {speakers}")

        script = text.replace("'", "'")
        
        lines = script.strip().split("\n")
        formatted_script_lines = []
        speaker_ids_used = set()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("Speaker") and ":" in line:
                match = re.match(r"^Speaker\s+(\d+)\s*:", line, re.IGNORECASE)
                if match:
                    speaker_id = int(match.group(1))
                    speaker_ids_used.add(speaker_id)

        if speaker_ids_used:
            sorted_speakers = sorted(speaker_ids_used)
            speaker_id_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted_speakers)}

            if len(sorted_speakers) > num_speakers:
                raise ValueError(f"Number of unique speakers in text ({len(sorted_speakers)}) exceeds num_speakers ({num_speakers})")
        else:
            speaker_id_mapping = {}

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("Speaker ") and ":" in line:
                match = re.match(r"^Speaker\s+(\d+)\s*:\s*(.*)$", line, re.IGNORECASE)
                if match:
                    old_speaker_id = int(match.group(1))
                    text_content = match.group(2)
                    new_speaker_id = speaker_id_mapping.get(old_speaker_id, old_speaker_id)
                    formatted_script_lines.append(f"Speaker {new_speaker_id}: {text_content}")
                else:
                    formatted_script_lines.append(line)
            else:
                speaker_id = len(formatted_script_lines) % num_speakers
                formatted_script_lines.append(f"Speaker {speaker_id}: {line}")

        formatted_script = "\n".join(formatted_script_lines)

        voice_samples = []
        for speaker_name in speakers:
            voice_path = os.path.join(self.voices_dir, f"{speaker_name}.wav")
            if not os.path.isfile(voice_path):
                raise ValueError(f"Voice file for speaker '{speaker_name}' not found at {voice_path}")
            
            wav, sr = sf.read(voice_path)
            if len(wav.shape) > 1:
                wav = np.mean(wav, axis=1)
            if sr != 24000:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=24000)

            voice_samples.append(wav)

        print(f"Using {num_speakers} speakers with voice samples: {speakers}")

        self.voices = voice_samples

        generation = self.inference(formatted_script)

        print(f"Generated audio saved at: {generation.audio_path}")
        print(f"Audio duration: {generation.audio_duration:.2f} seconds")
        print(f"Generation time: {generation.generation_time:.2f} seconds")
        print(f"Real-time factor (RTF): {generation.rtf:.4f}")
        print(f"Input tokens: {generation.input_tokens}")
        print(f"Generated tokens: {generation.generated_tokens}")

        with open(generation.audio_path, "rb") as f:
            audio_bytes = f.read()
        
        return audio_bytes
            

    def inference(self, text: str) -> GenerationResult:
        print(f"Generating text to speech for '{text}'")

        processor_kwargs = {
            "text": text,
            "padding": True,
            "return_tensors": "pt",
            "return_attention_mask": True,
            "voice_samples": self.voices
        }

        inputs = self.processor(**processor_kwargs)

        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to("cuda")

        start_time = time.time()
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=2.1,
            tokenizer=self.processor.tokenizer,
            generation_config={"do_sample": False},
            verbose=True,
            is_prefill=True, # VOICE CLONING ENABLED
        )
        generation_time = time.time() - start_time

        sample_rate = 24000
        audio_samples = (
            outputs.speech_outputs[0].shape[-1]
            if len(outputs.speech_outputs[0].shape) > 0
            else len(outputs.speech_outputs[0])
        )
        audio_duration = audio_samples / sample_rate
        rtf = generation_time / audio_duration if audio_duration > 0 else float("inf")

        input_tokens = inputs["input_ids"].shape[1]
        output_tokens = outputs.sequences.shape[1]
        generated_tokens = output_tokens - input_tokens

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            self.processor.save_audio(outputs.speech_outputs[0], output_path=tmpfile.name)
        
        return GenerationResult(
            audio_path=tmpfile.name,
            audio_duration=audio_duration,
            generation_time=generation_time,
            rtf=rtf,
            input_tokens=input_tokens,
            generated_tokens=generated_tokens,
        )
    
    @modal.method()
    def _inference(self, text: str):
        return self.generate(text, num_speakers=2, voices=["Cody", "Felippe"])
    
    @modal.fastapi_endpoint(docs=True, method="POST")
    def web_inference(self, request: TTSRequest, x_api_key: str = Header(None)):
        api_key = os.getenv("API_KEY")
        if x_api_key != api_key:
            return JSONResponse(status_code=401, content={"message": "Unauthorized"})

        print("Received TTS request via API")
        audio = self.generate(request.text, num_speakers=len(request.voices), voices=request.voices)
        
        return Response(
            content=audio,
            media_type="audio/wav",
            headers={
                "Content-Disposition": 'attachment; filename="output.wav"'
            }
        )
    
@app.local_entrypoint()
def main():
    text = """
    Speaker 0: Olá! Bem-vindo ao VibeVoice!
    Speaker 1: Obrigado! Estou animado para experimentar essa incrível tecnologia de síntese de voz.
    Speaker 0: Com o VibeVoice, você pode clonar vozes e gerar fala realista com facilidade.
    Speaker 1: Isso é impressionante! Mal posso esperar para ver como funciona na prática
    """

    audio = Model()._inference.remote(text)
    with open("output.wav", "wb") as f:
        f.write(audio)

