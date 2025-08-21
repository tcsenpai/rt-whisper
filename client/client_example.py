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
import time
from typing import Optional
from scipy import signal

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
        self.actual_sample_rate = sample_rate  # Will be updated based on device capabilities
        self.target_sample_rate = 16000  # Server expects this
        self.resample_needed = False
        
        # Adaptive noise gating
        self.noise_calibration = True
        self.noise_samples = []
        self.noise_floor = 0.01  # Default
        self.calibration_start_time = None
        self.calibration_duration = 3.0  # Seconds
        
        # Find and display available audio devices
        self.list_audio_devices()
    
    def list_audio_devices(self):
        """List available audio input devices"""
        print("\nAvailable audio input devices:")
        print("-" * 50)
        devices = []
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                devices.append((i, info))
                print(f"Device {i}: {info['name']} (Channels: {info['maxInputChannels']})")
        print("-" * 50)
        return devices
    
    def test_sample_rate(self, device_index, sample_rate):
        """Test if a device supports a specific sample rate"""
        try:
            # Just test if we can create the stream
            test_stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=1024
            )
            test_stream.close()
            return True
        except OSError:
            return False
    
    def find_best_sample_rate(self, device_index):
        """Find the best supported sample rate for a device"""
        # Try sample rates in order of preference
        rates_to_try = [16000, 44100, 48000, 22050, 8000]
        
        for rate in rates_to_try:
            if self.test_sample_rate(device_index, rate):
                return rate
        
        # If none work, get the device's default
        try:
            device_info = self.audio.get_device_info_by_index(device_index)
            return int(device_info['defaultSampleRate'])
        except:
            return 44100  # Last resort
    
    def select_audio_device(self):
        """Interactive device selection"""
        devices = self.list_audio_devices()
        
        if not devices:
            print("No audio input devices found!")
            return None, None
        
        # Get default device
        try:
            default_info = self.audio.get_default_input_device_info()
            default_index = default_info['index']
        except:
            default_index = devices[0][0]
        
        print(f"\nDefault device is: {default_index}")
        print("Enter device number to use (or press Enter for default): ", end="")
        
        try:
            choice = input().strip()
            if choice == "":
                device_index = default_index
            else:
                device_id = int(choice)
                # Verify it's a valid input device
                device_index = None
                for dev_id, info in devices:
                    if dev_id == device_id:
                        device_index = device_id
                        break
                
                if device_index is None:
                    print(f"Invalid device number. Using default device {default_index}")
                    device_index = default_index
            
            # Find best sample rate for this device
            best_rate = self.find_best_sample_rate(device_index)
            print(f"Best sample rate for this device: {best_rate} Hz")
            
            return device_index, best_rate
            
        except (ValueError, KeyboardInterrupt):
            print(f"\nUsing default device {default_index}")
            best_rate = self.find_best_sample_rate(default_index)
            print(f"Best sample rate for this device: {best_rate} Hz")
            return default_index, best_rate
    
    def resample_audio(self, audio_data):
        """Resample audio to target sample rate if needed"""
        if not self.resample_needed:
            return audio_data
        
        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        
        # Resample
        num_samples_target = int(len(audio_array) * self.target_sample_rate / self.actual_sample_rate)
        resampled = signal.resample(audio_array, num_samples_target)
        
        # Convert back to int16 bytes
        resampled_int16 = resampled.astype(np.int16)
        return resampled_int16.tobytes()
    
    def update_noise_floor(self, audio_array):
        """Update adaptive noise floor during calibration"""
        if not self.noise_calibration:
            return
            
        if self.calibration_start_time is None:
            self.calibration_start_time = time.time()
            print(f"\n🔊 Calibrating noise floor... Please stay quiet for {self.calibration_duration} seconds.")
            
        elapsed = time.time() - self.calibration_start_time
        if elapsed >= self.calibration_duration:
            # Calculate adaptive noise floor
            if self.noise_samples:
                noise_levels = np.array(self.noise_samples)
                self.noise_floor = np.percentile(noise_levels, 95) * 1.5  # 95th percentile + margin
                print(f"\n✅ Noise calibration complete! Noise floor: {self.noise_floor:.4f}")
                print("You can now speak normally - only audio above noise floor will be sent.")
            else:
                print("\n⚠️  No noise samples collected, using default threshold")
                
            self.noise_calibration = False
            self.noise_samples = []
        else:
            # Collect noise samples
            max_amp = np.max(np.abs(audio_array.astype(np.float32) / 32768.0))
            self.noise_samples.append(max_amp)
            
            # Show progress
            remaining = self.calibration_duration - elapsed
            if int(elapsed * 2) != int((elapsed - 0.05) * 2):  # Update twice per second
                print(f"\rCalibrating... {remaining:.1f}s remaining (level: {max_amp:.4f})", end="", flush=True)

    def should_send_audio(self, audio_array):
        """Check if audio should be sent based on noise gate"""
        if self.noise_calibration:
            return False  # Don't send during calibration
            
        max_amp = np.max(np.abs(audio_array.astype(np.float32) / 32768.0))
        return max_amp > self.noise_floor

    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream"""
        if status:
            print(f"Audio callback status: {status}")
        
        # Convert to numpy for processing
        audio_array = np.frombuffer(in_data, dtype=np.int16)
        
        # Update noise floor during calibration
        self.update_noise_floor(audio_array)
        
        # Show audio levels during calibration or when above threshold
        max_amp = np.max(np.abs(audio_array))
        if self.noise_calibration or max_amp > (self.noise_floor * 32768 * 0.8):
            bars = min(50, int(max_amp/500))
            threshold_marker = int((self.noise_floor * 32768) / 500) if not self.noise_calibration else 0
            bar_display = '█' * bars + '│' if bars > threshold_marker else '█' * bars
            print(f"\rAudio: {bar_display}{' ' * (50-bars)} [{max_amp:5d}] {'(calibrating)' if self.noise_calibration else ''}", end="", flush=True)
        
        # Only queue audio if it should be sent
        if self.should_send_audio(audio_array):
            # Resample if needed
            processed_data = self.resample_audio(in_data)
            self.audio_queue.put(processed_data)
        
        return (in_data, pyaudio.paContinue if self.running else pyaudio.paComplete)
    
    async def send_audio(self, websocket):
        """Send audio data to the server"""
        print("Starting audio streaming...")
        
        # Show which device we're using
        if self.device_index is not None:
            device_info = self.audio.get_device_info_by_index(self.device_index)
            print(f"Using audio device {self.device_index}: {device_info['name']}")
        else:
            default_device = self.audio.get_default_input_device_info()
            print(f"Using default audio device: {default_device['name']}")
        
        # Setup resampling if needed
        if self.actual_sample_rate != self.target_sample_rate:
            self.resample_needed = True
            print(f"Will resample from {self.actual_sample_rate} Hz to {self.target_sample_rate} Hz")
        else:
            self.resample_needed = False
            print("No resampling needed")
        
        # Calculate chunk size based on actual sample rate
        actual_chunk_size = int(self.actual_sample_rate * 0.1)  # 100ms chunks
        
        # Open audio stream
        stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.actual_sample_rate,  # Use actual device sample rate
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=actual_chunk_size,
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
    
    # Create client
    client = StreamingClient(
        server_url=args.server,
        sample_rate=args.sample_rate,
        device_index=args.device
    )
    
    # If no device specified, let user select one
    if args.device is None:
        device_index, sample_rate = client.select_audio_device()
        client.device_index = device_index
        client.actual_sample_rate = sample_rate
        print()
    
    await client.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)