#!/usr/bin/env python3
"""
Real-time streaming transcription server using faster-whisper
Receives audio streams via WebSocket and returns transcripts in real-time
"""

import asyncio
import json
import logging
import numpy as np
import time
from collections import deque
from typing import Optional, Dict, Any
import websockets
from faster_whisper import WhisperModel
import webrtcvad
import soundfile as sf
import io
import torch

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for troubleshooting
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TranscriptionServer:
    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        vad_aggressiveness: int = 1,  # Less aggressive for noisy environments
        sample_rate: int = 16000,
        buffer_duration: float = 0.5,  # Buffer duration in seconds
        silence_duration: float = 1.0,  # Silence duration to trigger transcription
    ):
        """
        Initialize the transcription server
        
        Args:
            model_size: Whisper model size (large-v3 recommended for 4070 Ti Super)
            device: Device to run the model on (cuda for GPU)
            compute_type: Computation type (float16 for faster inference)
            vad_aggressiveness: Voice Activity Detection aggressiveness (0-3)
            sample_rate: Audio sample rate in Hz
            buffer_duration: Duration of audio buffer in seconds
            silence_duration: Duration of silence before processing buffered audio
        """
        self.sample_rate = sample_rate
        self.buffer_duration = buffer_duration
        self.silence_duration = silence_duration
        self.frame_duration_ms = 30  # VAD frame duration in milliseconds
        self.frame_size = int(sample_rate * self.frame_duration_ms / 1000)
        
        # Check CUDA availability
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device = "cpu"
            compute_type = "float32"
        
        logger.info(f"Initializing Whisper model: {model_size} on {device} with {compute_type}")
        
        # Initialize Whisper model
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            num_workers=4,
            download_root="./models"
        )
        
        # Initialize VAD with less aggressive setting for noisy environments
        logger.info(f"Setting VAD aggressiveness to {vad_aggressiveness} (0=least aggressive, 3=most aggressive)")
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        
        # Client connections
        self.connections: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Transcription server initialized successfully")
    
    async def handle_connection(self, websocket, path):
        """Handle WebSocket connection from a client"""
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"Client connected: {client_id}")
        
        # Initialize client state
        self.connections[client_id] = {
            "websocket": websocket,
            "audio_buffer": deque(),
            "transcription_buffer": [],
            "last_speech_time": time.time(),
            "is_speaking": False,
            "language": None,  # Auto-detect language
            "session_text": []  # Keep track of full session transcription
        }
        
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    # Audio data received
                    await self.process_audio_chunk(client_id, message)
                else:
                    # Control message (JSON)
                    await self.handle_control_message(client_id, message)
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {client_id}")
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}")
        finally:
            # Clean up client state
            if client_id in self.connections:
                del self.connections[client_id]
    
    async def handle_control_message(self, client_id: str, message: str):
        """Handle control messages from client"""
        try:
            data = json.loads(message)
            command = data.get("command")
            
            if command == "set_language":
                self.connections[client_id]["language"] = data.get("language")
                logger.info(f"Language set to {data.get('language')} for {client_id}")
                
            elif command == "get_full_transcript":
                # Send the full session transcript
                full_text = " ".join(self.connections[client_id]["session_text"])
                await self.send_transcription(client_id, {
                    "type": "full_transcript",
                    "text": full_text,
                    "timestamp": time.time()
                })
                
            elif command == "clear_session":
                self.connections[client_id]["session_text"] = []
                logger.info(f"Session cleared for {client_id}")
        
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON from {client_id}: {message}")
    
    async def process_audio_chunk(self, client_id: str, audio_data: bytes):
        """Process incoming audio chunk"""
        client = self.connections[client_id]
        
        # Convert audio data to numpy array
        try:
            # Assume audio is in 16-bit PCM format
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Debug: Log audio chunk info
            logger.debug(f"Received audio chunk: {len(audio_data)} bytes, {len(audio_array)} samples, max amplitude: {np.max(np.abs(audio_array)):.4f}")
            
            # Add to buffer (client has already done noise gating)
            client["audio_buffer"].extend(audio_array)
            
            # Check if we have enough audio for VAD
            frame_size = self.frame_size
            
            while len(client["audio_buffer"]) >= frame_size:
                # Extract frame for VAD
                frame = np.array(list(client["audio_buffer"])[:frame_size])
                
                # Convert back to int16 for VAD
                frame_int16 = (frame * 32768).astype(np.int16)
                
                # Since client has pre-filtered audio, use VAD more liberally
                frame_volume = np.max(np.abs(frame))
                
                # Check for speech with VAD (client has already done noise gating)
                is_speech = self.vad.is_speech(frame_int16.tobytes(), self.sample_rate)
                
                if is_speech:
                    if not client["is_speaking"]:
                        logger.info(f"Speech detected for {client_id} (volume: {frame_volume:.4f})")
                    client["last_speech_time"] = time.time()
                    client["is_speaking"] = True
                    # Add frame to transcription buffer
                    client["transcription_buffer"].extend(frame)
                elif frame_volume > 0.005:
                    # Audio present but VAD says no speech - still add for context
                    if len(client["transcription_buffer"]) > 0:
                        client["transcription_buffer"].extend(frame[:len(frame)//4])  # Add some context
                else:
                    # Check if silence duration exceeded
                    silence_time = time.time() - client["last_speech_time"]
                    
                    if client["is_speaking"] and silence_time > self.silence_duration:
                        logger.info(f"Silence detected for {client_id}, triggering transcription. Buffer size: {len(client['transcription_buffer'])} samples")
                        # Process buffered audio
                        await self.transcribe_buffer(client_id)
                        client["is_speaking"] = False
                    elif not client["is_speaking"]:
                        # Add a small amount of silence to buffer for context
                        if len(client["transcription_buffer"]) > 0:
                            client["transcription_buffer"].extend(frame[:len(frame)//4])
                
                # Remove processed frame from buffer
                for _ in range(frame_size):
                    client["audio_buffer"].popleft()
            
            # Check buffer size limit (prevent memory issues)
            max_buffer_size = self.sample_rate * 30  # 30 seconds max
            if len(client["transcription_buffer"]) > max_buffer_size:
                await self.transcribe_buffer(client_id)
        
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
    
    async def transcribe_buffer(self, client_id: str):
        """Transcribe the buffered audio and send results"""
        client = self.connections[client_id]
        
        if len(client["transcription_buffer"]) == 0:
            return
        
        try:
            # Convert buffer to numpy array
            audio_array = np.array(client["transcription_buffer"])
            
            # Skip if audio is too short
            if len(audio_array) < self.sample_rate * 0.1:  # Less than 100ms
                client["transcription_buffer"] = []
                return
            
            # Transcribe audio
            segments, info = self.model.transcribe(
                audio_array,
                beam_size=5,
                best_of=5,
                language=client["language"],
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=400,
                    window_size_samples=1024,
                    threshold=0.5
                )
            )
            
            # Process segments
            transcription = ""
            for segment in segments:
                transcription += segment.text
            
            if transcription.strip():
                # Add to session text
                client["session_text"].append(transcription.strip())
                
                # Send transcription to client
                await self.send_transcription(client_id, {
                    "type": "partial",
                    "text": transcription.strip(),
                    "language": info.language,
                    "timestamp": time.time(),
                    "duration": len(audio_array) / self.sample_rate
                })
                
                logger.info(f"Transcribed for {client_id}: {transcription.strip()}")
            
            # Clear transcription buffer
            client["transcription_buffer"] = []
        
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            client["transcription_buffer"] = []
    
    async def send_transcription(self, client_id: str, data: Dict[str, Any]):
        """Send transcription data to client"""
        if client_id in self.connections:
            websocket = self.connections[client_id]["websocket"]
            try:
                await websocket.send(json.dumps(data))
            except Exception as e:
                logger.error(f"Error sending transcription to {client_id}: {e}")
    
    async def start_server(self, host: str = "0.0.0.0", port: int = 8765):
        """Start the WebSocket server"""
        logger.info(f"Starting WebSocket server on {host}:{port}")
        logger.info("Server will be accessible from any network interface")
        logger.info("Make sure port 8765 is open in your firewall")
        async with websockets.serve(self.handle_connection, host, port):
            logger.info(f"Server listening on ws://{host}:{port}")
            logger.info(f"Clients can connect using ws://YOUR_SERVER_IP:8765")
            await asyncio.Future()  # Run forever

async def main():
    """Main entry point"""
    server = TranscriptionServer(
        model_size="large-v3",  # Use large model for best accuracy
        device="cuda",  # Use GPU
        compute_type="float16",  # Use float16 for faster inference on RTX 4070 Ti Super
        vad_aggressiveness=1,  # Less aggressive for noisy environments
        sample_rate=16000,
        buffer_duration=0.5,
        silence_duration=1.0  # Longer silence before triggering transcription
    )
    
    await server.start_server()

if __name__ == "__main__":
    asyncio.run(main())