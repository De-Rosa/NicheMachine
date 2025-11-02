const int irPin = 2;
const int ledPin = LED_BUILTIN;

void setup() {
  pinMode(irPin, INPUT);
  pinMode(ledPin, OUTPUT);
}

void loop() {
  int sensorState = digitalRead(irPin);
  if (sensorState == HIGH) {  // Object detected 
    digitalWrite(ledPin, LOW);
    delay(100);
    digitalWrite(ledPin, HIGH);
    delay(100);
  } else {
  digitalWrite(ledPin, HIGH);  // LED on
  }
  delay(100);
}
