"""Local poller: pull each (pod, exp) with pull_fast.py once every queue tag of that exp on that pod reports finished.
Usage: python experiments/common/autopull.py   (writes experiments/common/logs/autopull.log)"""
import json, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, ROOT)
from core import remote as R
PODS = {"P1": ("213.173.110.151", 13256), "P2": ("157.157.221.29", 32812), "P3": ("213.173.111.15", 10853), "P4": ("213.173.110.226", 39505),
        "P5": ("47.47.180.58", 14713), "P6": ("47.47.180.58", 16409), "P8": ("149.36.1.192", 22337)}
TAGS = {("P1", "E1"): ["P1s0"], ("P1", "E6"): ["P1s1"], ("P2", "E1"): ["P2s0"], ("P3", "E1"): ["P3s0"], ("P4", "E7"): ["P4s0"], ("P4", "E1"): ["P4s1"],
        ("P5", "E3"): ["P5as0", "P5bs0"], ("P5", "E1"): ["P5cs0", "P5ds0"], ("P6", "E3"): ["P6as0", "P6bs0", "P6cs0"], ("P6", "E1"): ["P6ds0"],
        ("P8", "E3"): ["P8as0", "P8bs0"], ("P8", "E8"): ["P8as1", "P8cs0", "P8ds0", "P8es0"]}
done = set(json.load(open(os.path.join(HERE, "logs", "autopull_done.json")))) if os.path.exists(os.path.join(HERE, "logs", "autopull_done.json")) else set()
log = open(os.path.join(HERE, "logs", "autopull.log"), "a")
while len(done) < len(TAGS):
    for (pod, exp), tags in TAGS.items():
        key = f"{pod}:{exp}"
        if key in done:
            continue
        host, port = PODS[pod]
        try:
            out = R._ssh(host, port, " ; ".join(f"cat {R.REMOTE_ROOT}/experiments/{exp}/queue_status_{t}.json 2>/dev/null; echo" for t in tags), check=False).stdout
        except Exception as e:
            continue
        fin = sum('"finished": true' in line for line in out.splitlines())
        if fin == len(tags):
            r = subprocess.run([sys.executable, os.path.join(HERE, "pull_fast.py"), pod, host, str(port), exp], capture_output=True, text=True)
            log.write(time.strftime("%H:%M:%S ") + key + " rc=" + str(r.returncode) + " " + r.stdout.strip() + " " + r.stderr.strip()[-500:] + "\n"); log.flush()
            if r.returncode == 0:
                done.add(key); json.dump(sorted(done), open(os.path.join(HERE, "logs", "autopull_done.json"), "w"))
    time.sleep(20)
log.write(time.strftime("%H:%M:%S ") + "ALL PULLED\n")
