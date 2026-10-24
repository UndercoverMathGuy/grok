jobs = []
for p in [5, 7, 11]:
    for n, bs in [(2, [2.0, 2.5]), (3, [2.0, 3.0])]:
        for b in bs:
            big = (p == 11 and n == 3)
            jobs.append(f"hunt_b2.py {p} {n} {b} {4 if big else 10} 61")
            for mode in ['struct', 'pen', 'sf']:
                jobs.append(f"hunt_b.py {p} {n} {b} {6 if big else 14} 51 {mode}")
for p in [5, 7]:
    for n in [3, 4, 5]:
        for b in [2.0, 2.6]:
            nr = {3: 18, 4: 12, 5: 8}[n] if p == 5 else {3: 12, 4: 8, 5: 4}[n]
            jobs.append(f"hunt_c.py {p} {n} {b} {nr} 71")
for p in [11, 13, 17, 19, 23]:
    for b in [2.0, 2.5]:
        jobs.append(f"hunt_straddle.py {p} {b} 12 111")
for p in [11, 13, 17, 23, 31, 43]:
    jobs.append(f"detect_lp.py {p} 2.0 Sstar 10 91")
    jobs.append(f"detect_lp.py {p} 2.0 AP1_6_1 10 92")
for p in [13, 23, 31]:
    jobs.append(f"detect_lp.py {p} 2.5 Sstar 10 91")
for p in [23, 43]:
    jobs.append(f"detect_lp.py {p} 2.0 AP1_6_1 8 95 2 single")
for p in [5, 7, 11]:
    jobs.append(f"detect_lp.py {p} 2.0 Sstar 6 93 3")
for p in [13, 19, 23, 31]:
    F = (p - 1) // 2
    jobs.append(f"hunt_a.py {p} 2.0 AP1_{min(F, 6)}_1 20 21")
    jobs.append(f"hunt_a.py {p} 2.5 AP1_{min(F, 7)}_2 20 21")
for p in [37, 43]:
    for b in [2.0, 2.5]:
        jobs.append(f"hunt_a.py {p} {b} F 10 31")
for p in [11, 13, 17, 19, 23]:
    jobs.append(f"hunt_safety.py {p} 2 2.0 F 3 41")
for p in [13, 19]:
    jobs.append(f"hunt_safety.py {p} 2 2.5 F 3 41")
for p in [11, 17, 23, 31]:
    F = (p - 1) // 2
    jobs.append(f"hunt_fejer.py {p} 2.0 Sstar 24 81")
    jobs.append(f"hunt_fejer.py {p} 2.0 AP1_{min(F, 6)}_1 24 82")
for p in [17, 31]:
    jobs.append(f"hunt_fejer.py {p} 2.5 Sstar 24 81")
open('jobs_rest.txt', 'w').write('\n'.join(jobs) + '\n'); print(len(jobs))
