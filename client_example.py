#!/usr/bin/env python3
"""
Example client for streaming microphone audio to the transcription server
"""

import asyncio
import json
import pyaudio
import websockets
import numpy as np
import threading
import queue
import argparse
import sys
from typing import Optional

class StreamingClient:
    def __init__(
        self,
        server_url: str = "ws://localhost:8765",
        sample_rate: int = 16000,
        chunk_duration: float = 0.1,  # Send audio in 100ms chunks
        device_index: Optional[int] = None
    ):
        """
        Initialize the streaming client
        
        Args:
            server_url: WebSocket server URL
            sample_rate: Audio sample rate in Hz
            chunk_duration: Duration of audio chunks to send
            device_index: Audio input device index (None for default)
        """
        self.server_url = server_url
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_duration)
        self.device_index = device_index
        
        # Audio setup
        self.audio = pyaudio.PyAudio()
        self.audio_queue = queue.Queue()
        self.running = False
        
        # Find and display available audio devices
        self.list_audio_devices()
    
    def list_audio_devices(self):
        """List available audio input devices"""
        print("\nAvailable audio input devices:")
        print("-" * 50)
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"Device {i}: {info['name']} (Channels: {info['maxInputChannels']})")
        print("-" * 50)
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream"""
        if status:
            print(f"Audio callback status: {status}")
        
        # Add audio data to queue
        self.audio_queue.put(in_data)
        
        return (in_data, pyaudio.paContinue if self.running else pyaudio.paComplete)
    
    async def send_audio(self, websocket):
        """Send audio data to the server"""
        print("Starting audio streaming...")
        
        # Open audio stream
        stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.chunk_size,
            stream_callback=self.audio_callback
        )
        
        self.running = True
        stream.start_stream()
        
        try:
            while self.running:
                try:
                    # Get audio data from queue
                    audio_data = self.audio_queue.get(timeout=0.1)
                    
                    # Send to server
                    await websocket.send(audio_data)
                    
                except queue.Empty:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    print("Connection to server lost")
                    break
                except Exception as e:
                    print(f"Error sending audio: {e}")
                    break
        
        finally:
            stream.stop_stream()
            stream.close()
    
    async def receive_transcriptions(self, websocket):
        """Receive transcriptions from the server"""
        print("Waiting for transcriptions...")
        print("=" * 50)
        
        try:
            async for message in websocket:
                data = json.loads(message)
                
                if data["type"] == "partial":
                    # Real-time partial transcription
                    print(f"\r[{data['language']}] {data['text']}", end="", flush=True)
                    print()  # New line after each transcription
                
                elif data["type"] == "full_transcript":
                    # Full session transcript
                    print("\n" + "=" * 50)
                    print("Full Transcript:")
                    print(data["text"])
                    print("=" * 50)
        
        except websockets.exceptions.ConnectionClosed:
            print("\nConnection closed")
        except Exception as e:
            print(f"\nError receiving transcriptions: {e}")
    
    async def handle_commands(self, websocket):
        """Handle user commands"""
        print("\nCommands:")
        print("  'lang <code>' - Set language (e.g., 'lang en' for English)")
        print("  'full' - Get full session transcript")
        print("  'clear' - Clear session")
        print("  'quit' - Exit")
        print()
        
        loop = asyncio.get_event_loop()
        
        while self.running:
            try:
                # Non-blocking input using executor
                command = await loop.run_in_executor(None, input, "")
                
                if command.lower() == "quit":
                    self.running = False
                    break
                elif command.lower() == "full":
                    await websocket.send(json.dumps({"command": "get_full_transcript"}))
                elif command.lower() == "clear":
                    await websocket.send(json.dumps({"command": "clear_session"}))
                elif command.lower().startswith("lang "):
                    language = command.split()[1]
                    await websocket.send(json.dumps({
                        "command": "set_language",
                        "language": language
                    }))
                    print(f"Language set to: {language}")
            
            except Exception:
                await asyncio.sleep(0.1)
    
    async def run(self):
        """Run the streaming client"""
        print(f"Connecting to {self.server_url}...")
        
        try:
            async with websockets.connect(self.server_url) as websocket:
                print("Connected to transcription server")
                print("Speak into your microphone. Press Ctrl+C to stop.")
                print("=" * 50)
                
                # Create tasks for sending and receiving
                send_task = asyncio.create_task(self.send_audio(websocket))
                receive_task = asyncio.create_task(self.receive_transcriptions(websocket))
                command_task = asyncio.create_task(self.handle_commands(websocket))
                
                # Wait for any task to complete
                done, pending = await asyncio.wait(
                    [send_task, receive_task, command_task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Cancel remaining tasks
                self.running = False
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
        
        except ConnectionRefusedError:
            print(f"Failed to connect to server at {self.server_url}")
            print("Make sure the server is running.")
        except KeyboardInterrupt:
            print("\nStopping...")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.running = False
            self.audio.terminate()

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Streaming audio client for transcription")
    parser.add_argument(
        "--server",
        default="ws://localhost:8765",
        help="WebSocket server URL (default: ws://localhost:8765)"
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Audio input device index (use --list-devices to see available devices)"
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Audio sample rate in Hz (default: 16000)"
    )
    
    args = parser.parse_args()
    
    # If listing devices, just list and exit
    if args.list_devices:
        audio = pyaudio.PyAudio()
        print("\nAvailable audio input devices:")
        print("-" * 50)
        for i in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"Device {i}: {info['name']} (Channels: {info['maxInputChannels']})")
        print("-" * 50)
        audio.terminate()
        return
    
    # Create and run client
    client = StreamingClient(
        server_url=args.server,
        sample_rate=args.sample_rate,
        device_index=args.device
    )
    
    await client.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)