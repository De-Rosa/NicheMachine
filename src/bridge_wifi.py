from flask import Flask, request
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route("/")
def index():
    return "Arduino Wi-Fi Bridge Running"

@app.route("/trigger", methods=["POST"])
def trigger():
    data = request.get_json()
    if data and data.get("trigger"):
        import random
        number = random.randint(1, 10)
        socketio.emit("trigger", {
            "song": f"Song Name {number}",
            "artist": "Artist Name"
        })
        return {"status": "ok"}, 200
    return {"status": "error"}, 400

if __name__ == "__main__":
    print("Flask-SocketIO server running on http://0.0.0.0:5000")
    socketio.run(app, host="0.0.0.0", port=5000)

