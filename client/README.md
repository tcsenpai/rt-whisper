# Whisper Streaming Clients

These are the **client components** that can run on any machine to send audio to the transcription server.

## 📍 Where to Run This

**Run this on**: Any machine with a microphone (your laptop, desktop, Raspberry Pi, etc.)

## 🎯 Available Clients

### 1. Python Client (`client_example.py`)

A command-line client with full features.

#### Installation
```bash
pip install -r requirements.txt
```

#### Usage
```bash
# Connect to server (replace with your server's IP)
python3 client_example.py --server ws://192.168.1.100:8765

# List available microphones
python3 client_example.py --list-devices

# Use specific microphone (device ID from list)
python3 client_example.py --server ws://192.168.1.100:8765 --device 2
```

#### Interactive Commands
While running, you can type:
- `lang en` - Set language (en, es, fr, de, it, pt, ru, zh, ja, ko)
- `full` - Get full session transcript
- `clear` - Clear current session
- `quit` - Exit

### 2. Web Client (`web_client.html`)

A browser-based client with a nice UI. No installation required!

#### Usage
1. Open `web_client.html` in any modern web browser
2. Enter server URL: `ws://YOUR_SERVER_IP:8765`
3. Click "Connect"
4. Click "Start Recording"
5. Speak into your microphone
6. Transcriptions appear in real-time

#### Features
- Visual volume meter
- Language selection dropdown
- Download transcript as text file
- Clear transcript
- Works on mobile browsers too!

## 🔧 Configuration

### Python Client Options

```bash
python3 client_example.py --help

Options:
  --server URL       WebSocket server URL (default: ws://localhost:8765)
  --device ID        Audio input device ID
  --list-devices     List available audio devices and exit
  --sample-rate HZ   Audio sample rate (default: 16000)
```

### Web Client Configuration

Edit the server URL directly in the input field, or modify the default in `web_client.html`:

```javascript
value="ws://localhost:8765"  // Change this default
```

## 🎤 Microphone Setup

### Windows
1. Check microphone in Settings → System → Sound
2. Allow microphone access for Python/Browser

### macOS
1. System Preferences → Security & Privacy → Microphone
2. Allow Terminal/Browser to access microphone

### Linux
```bash
# Test microphone
arecord -l

# Adjust microphone volume
alsamixer
# or
pavucontrol
```

## 🌐 Network Requirements

- Must be able to reach the server machine
- Port 8765 must be accessible
- Works over LAN or WAN (if properly configured)

## 🔍 Testing Connection

### Test Server Reachability
```bash
# Ping server
ping YOUR_SERVER_IP

# Test port (requires netcat)
nc -zv YOUR_SERVER_IP 8765

# Or with telnet
telnet YOUR_SERVER_IP 8765
```

### Test with Simple WebSocket
```python
import websockets
import asyncio

async def test():
    async with websockets.connect("ws://YOUR_SERVER_IP:8765") as ws:
        print("Connected!")

asyncio.run(test())
```

## 📱 Mobile Support

The web client works on mobile devices:

1. Open your phone's browser
2. Navigate to `http://YOUR_CLIENT_PC_IP/web_client.html`
   (You may need to serve it with a simple HTTP server)
3. Or use a file sharing service to transfer the HTML file

### Serve the web client:
```bash
# Python 3
python3 -m http.server 8000

# Then access from mobile:
# http://YOUR_CLIENT_PC_IP:8000/web_client.html
```

## 🐛 Troubleshooting

### No Audio Input

**Python Client:**
```bash
# Install PortAudio (Linux)
sudo apt-get install portaudio19-dev

# Install PortAudio (macOS)
brew install portaudio

# Reinstall PyAudio
pip uninstall pyaudio
pip install pyaudio
```

**Web Client:**
- Check browser permissions
- Ensure HTTPS or localhost (some browsers require secure context)
- Try different browser

### Connection Refused
- Check server is running
- Verify IP address and port
- Check firewall settings on both machines
- Ensure on same network or routing is configured

### Poor Quality
- Check microphone quality and positioning
- Reduce background noise
- Adjust VAD aggressiveness on server
- Ensure stable network connection

## 🎯 Use Cases

1. **Meeting Transcription**: Run client on meeting room computer
2. **Remote Dictation**: Dictate from anywhere on network
3. **Multi-Room Setup**: Multiple clients to one server
4. **Mobile Transcription**: Use phone as wireless microphone
5. **Accessibility**: Real-time captions for hearing impaired

## 📊 Performance

- **Network Usage**: ~32 KB/s upstream (audio)
- **CPU Usage**: Minimal (<5%)
- **RAM Usage**: <100MB for Python client
- **Latency**: Typically 100-500ms depending on network

## 🔄 Advanced Usage

### Custom Integration

```python
import websockets
import asyncio
import json

async def stream_audio_file(filename, server_url):
    async with websockets.connect(server_url) as ws:
        # Read and send audio file
        with open(filename, 'rb') as f:
            audio_data = f.read(4096)  # Read in chunks
            while audio_data:
                await ws.send(audio_data)
                audio_data = f.read(4096)
                await asyncio.sleep(0.1)  # Simulate real-time
        
        # Receive transcription
        response = await ws.recv()
        data = json.loads(response)
        print(f"Transcription: {data['text']}")

# Use it
asyncio.run(stream_audio_file("audio.pcm", "ws://server:8765"))
```