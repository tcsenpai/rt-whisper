# Real-Time Whisper Streaming Transcription Service

A high-performance real-time transcription service using faster-whisper, optimized for NVIDIA RTX 4070 Ti Super GPU. Receives audio streams via WebSocket and returns transcripts in real-time.

## Features

- **Real-time streaming transcription** using faster-whisper large-v3 model
- **WebSocket-based API** for low-latency bidirectional communication
- **Voice Activity Detection (VAD)** for efficient processing
- **Automatic language detection** or manual language selection
- **GPU acceleration** with CUDA and float16 precision
- **Buffered processing** for optimal transcription quality
- **Session management** with full transcript retrieval
- **Docker support** for easy deployment

## Architecture

```
[Microphone] → [Client] → [WebSocket] → [Server] → [faster-whisper] → [Transcription]
                   ↑                         ↓
                   └──── Real-time Text ─────┘
```

## Installation

### Local Installation

1. Install system dependencies:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install portaudio19-dev python3-dev ffmpeg

# For CUDA support, ensure you have CUDA 11.8+ and cuDNN 8+ installed
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

### Docker Installation

```bash
# Build the Docker image
docker-compose build

# Run the service
docker-compose up -d
```

## Usage

### Starting the Server

#### Local:
```bash
python3 streaming_server.py
```

#### Docker:
```bash
docker-compose up
```

The server will start on `ws://localhost:8765` by default.

### Using the Client

1. List available audio devices:
```bash
python3 client_example.py --list-devices
```

2. Connect and start streaming:
```bash
# Use default microphone
python3 client_example.py

# Use specific audio device
python3 client_example.py --device 2

# Connect to remote server
python3 client_example.py --server ws://192.168.1.100:8765
```

### Client Commands

While the client is running, you can use these commands:
- `lang <code>` - Set transcription language (e.g., `lang en` for English)
- `full` - Get the full session transcript
- `clear` - Clear the current session
- `quit` - Exit the client

## API Protocol

### WebSocket Connection
Connect to `ws://server:8765`

### Sending Audio
Send raw PCM audio data as binary messages:
- Format: 16-bit PCM
- Sample rate: 16000 Hz
- Channels: Mono

### Control Messages (JSON)
```json
{
  "command": "set_language",
  "language": "en"
}
```

Available commands:
- `set_language` - Set transcription language
- `get_full_transcript` - Request full session transcript
- `clear_session` - Clear current session

### Receiving Transcriptions (JSON)

Partial transcription:
```json
{
  "type": "partial",
  "text": "Hello world",
  "language": "en",
  "timestamp": 1234567890.123,
  "duration": 2.5
}
```

Full transcript:
```json
{
  "type": "full_transcript",
  "text": "Complete session transcript...",
  "timestamp": 1234567890.123
}
```

## Performance Optimization

The service is optimized for RTX 4070 Ti Super:
- Uses `large-v3` model for best accuracy
- `float16` computation for faster inference
- CUDA acceleration enabled
- Batch processing with VAD for efficiency
- 4 worker threads for parallel processing

### Expected Performance
- Model loading: ~5-10 seconds
- Transcription latency: 100-500ms (depending on audio length)
- Real-time factor: >5x (processes 5 seconds of audio in <1 second)

## Configuration

Modify these parameters in `streaming_server.py`:

```python
TranscriptionServer(
    model_size="large-v3",        # Model size
    device="cuda",                 # Device (cuda/cpu)
    compute_type="float16",        # Precision
    vad_aggressiveness=2,          # VAD sensitivity (0-3)
    sample_rate=16000,             # Audio sample rate
    buffer_duration=0.5,           # Buffer size in seconds
    silence_duration=0.5           # Silence before transcription
)
```

## Custom Client Implementation

To create your own client, implement this protocol:

1. Connect to WebSocket server
2. Send audio data as binary messages (16-bit PCM, 16kHz, mono)
3. Receive JSON transcription messages
4. Handle connection lifecycle

Example in Python:
```python
import websockets
import asyncio

async def stream_audio():
    async with websockets.connect("ws://localhost:8765") as ws:
        # Send audio
        await ws.send(audio_bytes)
        
        # Receive transcription
        response = await ws.recv()
        data = json.loads(response)
        print(data["text"])
```

## Troubleshooting

### CUDA Not Available
- Ensure NVIDIA drivers are installed: `nvidia-smi`
- Check CUDA installation: `python3 -c "import torch; print(torch.cuda.is_available())"`
- The server will automatically fall back to CPU if CUDA is not available

### Audio Issues
- Check microphone permissions
- Verify audio device with `--list-devices`
- Ensure sample rate matches (16000 Hz)

### Connection Issues
- Check firewall settings for port 8765
- Verify server is running: `netstat -an | grep 8765`
- Check WebSocket connection with browser developer tools

## License

MIT License

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.