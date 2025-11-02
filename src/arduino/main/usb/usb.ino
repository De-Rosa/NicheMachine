const int irPin = 2;
const int ledPin = LED_BUILTIN;

void setup() {
  Serial.begin(9600);
  pinMode(irPin, INPUT);
  pinMode(ledPin, OUTPUT);
}

void loop() {
  int sensorState = digitalRead(irPin);
  if (sensorState == HIGH) {  // Object detected 
    Serial.println("TRIGGER");
    digitalWrite(ledPin, LOW);
    delay(100);
    digitalWrite(ledPin, HIGH);
    delay(100);
  } else {
  digitalWrite(ledPin, HIGH);  // LED on
  }
  delay(100);
}