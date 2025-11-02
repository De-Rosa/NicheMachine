import serial
import threading
from flask import Flask
from flask_socketio import SocketIO

SERIAL_PORT = "/dev/tty.usbmodemF412FA9FE0602"  
BAUD_RATE = 9600
FLASK_PORT = 5000

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route("/")
def index():
    return "Arduino USB Serial Bridge Running"

def read_from_arduino():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Connected to Arduino on {SERIAL_PORT}")
    except Exception as e:
        print(f"Failed to connect to Arduino: {e}")
        return

    while True:
        try:
            line = ser.readline().decode().strip()
            if not line:
                continue

            print(f"Received from Arduino: {line}")

            if line == "TRIGGER":
                import random
                number = random.randint(1, 10)
                socketio.emit("trigger", {
                    "song": f"Song Name {number}",
                    "artist": "Artist Name"
                })

        except Exception as e:
            print(f"Error reading serial: {e}")

socketio.start_background_task(read_from_arduino)

if __name__ == "__main__":
    print(f"Flask-SocketIO server running on http://0.0.0.0:{FLASK_PORT}")
    socketio.run(app, host="0.0.0.0", port=FLASK_PORT)

