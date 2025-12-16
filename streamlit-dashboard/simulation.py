import paho.mqtt.client as mqtt
import json, time, random

BROKER = "mqtt-broker"
TOPIC = "fleet/data"

client = mqtt.Client()
client.connect(BROKER, 1883, 60)

def r(b,j): return round(b + random.uniform(-j,j), 2)

while True:
    for i in range(1,7):
        msg = {
            "Vehicle_ID": f"Truck_{i}",
            "Engine rpm": r(1500, 500),
            "Lub oil pressure": r(3, 1.5),
            "Fuel pressure": r(9, 2.5),
            "Coolant pressure": r(3, 1.2),
            "lub oil temp": r(85, 15),
            "Coolant temp": r(88, 15)
        }
        client.publish(TOPIC, json.dumps(msg))
        print("Sent:", msg)
        time.sleep(0.8)
