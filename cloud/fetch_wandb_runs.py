"""Pull the SEMIFINAL v2 run directories back out of wandb into runs/.

Credentials are NEVER passed on the command line or stored here: wandb.Api()
reads them from `wandb login` (~/.netrc) or $WANDB_API_KEY. If neither is
set the script says so and exits.

Each per-run wandb run carries one artifact of type "grok-run" holding that
run's config.json, metrics.json, spectra.npz and checkpoints/ — the exact
layout semifinal/analysis/ expects. Artifact names are the run path with
"/" -> "__" (orthWE__p-113__seed4811__seed61001); this reverses that.

Idempotent: a destination that already has spectra.npz is skipped, so an
interrupted pull resumes. Every download is verified for the three required
files before it counts as complete.

DO NOT pull into a --dest that already holds runs from another backend.
Cloud run paths collide exactly with the MLX ones (runs/p-113/seed4811/
seed61001 exists in both), and "skip if spectra.npz exists" would then keep
the MLX run and silently drop the cloud one — a backend-mixed cohort, which
every cohort-level claim forbids. Use a fresh directory per backend.

Run:  uv run --with wandb python cloud/fetch_wandb_runs.py
      ... --project grok-semifinal-v2 --dest runs --entity <org>
      ... --list        (show what's there, download nothing)
"""
import argparse
import os
import sys
from pathlib import Path

REQUIRED = ("config.json", "metrics.json", "spectra.npz")


def resolve_credentials():
    """Confirm wandb can authenticate WITHOUT us touching the secret."""
    import netrc
    if os.environ.get("WANDB_API_KEY"):
        return "$WANDB_API_KEY"
    try:
        hosts = netrc.netrc(os.path.expanduser("~/.netrc")).hosts
        for h in hosts:
            if "wandb" in h:
                return f"~/.netrc ({h})"
    except (FileNotFoundError, netrc.NetrcParseError):
        pass
    sys.exit("No wandb credentials found. Either:\n"
             "  wandb login                  (stores in ~/.netrc)\n"
             "  export WANDB_API_KEY=...     (this shell only)\n"
             "Do not put the key in this file or on the command line.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=os.environ.get("WANDB_PROJECT",
                                                        "grok-semifinal-v2"))
    ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    ap.add_argument("--dest", default="runs", type=Path)
    ap.add_argument("--list", action="store_true", help="show, download nothing")
    args = ap.parse_args()

    import wandb
    src = resolve_credentials()
    print(f"auth via {src}", flush=True)
    try:
        api = wandb.Api()
    except Exception as e:
        # A credential being PRESENT is not the same as it being VALID: a
        # rotated/revoked key sits in ~/.netrc forever and 401s.
        sys.exit(f"wandb rejected the credential in {src}:\n  {e}\n"
                 "Get a fresh key at https://wandb.ai/authorize, then:\n"
                 "  wandb login --relogin\n"
                 "(or export WANDB_API_KEY=... , which takes precedence)")
    path = f"{args.entity}/{args.project}" if args.entity else args.project

    try:
        runs = list(api.runs(path))
    except Exception as e:
        sys.exit(f"could not list runs in '{path}': {e}\n"
                 "Pass --entity <org> if your key spans multiple entities.")

    jobs = []
    for r in runs:
        for art in r.logged_artifacts():
            if art.type == "grok-run":
                jobs.append((art.name.split(":")[0].replace("__", "/"), art))
    jobs.sort()
    print(f"{len(jobs)} grok-run artifacts in {path} "
          f"({len(runs)} wandb runs incl. driver)", flush=True)
    if args.list:
        for name, art in jobs:
            size = getattr(art, "size", None)
            print(f"  {name:48s} "
                  f"{f'{size / 1e6:8.1f} MB' if size else '     ? MB'}")
        return

    # Loud warning if --dest already holds runs we are NOT about to fetch:
    # those are from some other source, and mixing backends in one directory
    # is how a cohort quietly becomes uninterpretable.
    want = {n for n, _ in jobs}
    stray = sorted({str(p.parent.relative_to(args.dest))
                    for p in args.dest.rglob("spectra.npz")} - want)
    if stray:
        print(f"\n!! {args.dest}/ already holds {len(stray)} run(s) that are NOT "
              f"in this project, e.g. {stray[:3]} — if those came from another "
              f"backend, pull into a fresh --dest instead.\n", flush=True)

    done = skipped = failed = 0
    for name, art in jobs:
        dest = args.dest / name
        if (dest / "spectra.npz").exists():
            print(f"  skip {name} (have it)", flush=True)
            skipped += 1
            continue
        try:
            art.download(root=str(dest))
        except Exception as e:
            print(f"  FAIL {name}: {e}", flush=True)
            failed += 1
            continue
        missing = [f for f in REQUIRED if not (dest / f).exists()]
        if missing:
            print(f"  FAIL {name}: missing {missing}", flush=True)
            failed += 1
            continue
        n_ck = len(list((dest / "checkpoints").glob("*.safetensors")))
        print(f"  ok   {name}  ({n_ck} checkpoints)", flush=True)
        done += 1

    print(f"\ndownloaded {done}, already had {skipped}, failed {failed}"
          f"  ->  {args.dest}/", flush=True)
    if failed:
        sys.exit(1)
    print("re-run to retry anything missing; semifinal/analysis/ reads "
          f"{args.dest}/ directly.", flush=True)


if __name__ == "__main__":
    main()
