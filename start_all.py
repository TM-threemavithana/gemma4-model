import subprocess
import time

print("Starting server.py...")
server = subprocess.Popen(["python", "server.py"], stdout=open("server.log", "w"), stderr=subprocess.STDOUT)
time.sleep(2)

print("Starting Asterisk...")
subprocess.run(["wsl", "-u", "root", "service", "asterisk", "start"])
time.sleep(1)

print("Starting bridge...")
bridge = subprocess.Popen(["wsl", "bash", "-c", "./start_bridge.sh --debug"], stdout=open("bridge.log", "w"), stderr=subprocess.STDOUT)
time.sleep(2)

print("All services started in background.")
