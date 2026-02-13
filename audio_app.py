import asyncio
import os
import sounddevice as sd
from scipy.io.wavfile import write
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Load API key
load_dotenv()

# Record audio function
def record_audio(filename="input.wav", duration=5, fs=44100):
    print(" Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, recording)
    print(" Recording saved!")

async def main():
    # Record 5 seconds audio
    record_audio()

    # Create model client
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Create assistant
    assistant = AssistantAgent(
        name="audio_assistant",
        model_client=model_client,
        system_message="You are a helpful AI assistant."
    )

    # Ask AI (for now we send text, audio-to-text needs Whisper model)
    result = await assistant.run(
        task="User recorded audio. Respond politely."
    )

    print("\n=== AI Response ===\n")
    print(result.messages[-1].content)

asyncio.run(main())
