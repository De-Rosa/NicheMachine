#include <WiFi.h>

const char* ssid = "ssid";
const char* password = "password";

const char* serverAddress = "192.0.0.2";
const int serverPort = 5000;

const int irPin = 2;

WiFiClient wifiClient;

void setup() {
  Serial.begin(9600);
  delay(1000);

  pinMode(irPin, INPUT);

  WiFi.begin(ssid, password);
  Serial.print("Connecting to Wi-Fi");
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 30) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  Serial.println();
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("✅ Connected!");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("❌ Failed to connect to Wi-Fi");
  }
}

void loop() {
  int sensorState = digitalRead(irPin);
  if (sensorState == HIGH) {
    Serial.println("TRIGGER");

    if (wifiClient.connect(serverAddress, serverPort)) {
      wifiClient.println("POST /trigger HTTP/1.1");
      wifiClient.println("Host: " + String(serverAddress));
      wifiClient.println("Content-Type: application/json");
      wifiClient.println("Content-Length: 17");
      wifiClient.println();
      wifiClient.println("{\"trigger\":true}");
      wifiClient.stop();
      Serial.println("HTTP POST sent");
    } else {
      Serial.println("❌ Connection failed");
    }

    delay(500); // prevent multiple triggers
  }
}
