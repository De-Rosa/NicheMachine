const int irPin = 2;
const int ledPin = LED_BUILTIN;

bool triggered = false;  // Keeps track of trigger state

void setup() {
  Serial.begin(9600);
  pinMode(irPin, INPUT);
  pinMode(ledPin, OUTPUT);
}

void loop() {
  int sensorState = digitalRead(irPin);

  if (sensorState == HIGH && !triggered) {
    // Object just detected
    Serial.println("TRIGGER");
    triggered = true;
    digitalWrite(ledPin, LOW);  // LED off (optional visual cue)
  } 
  else if (sensorState == LOW && triggered) {
    // Object no longer detected, reset trigger
    triggered = false;
    digitalWrite(ledPin, HIGH); // LED on
  }

  delay(50); // small debounce delay
}

