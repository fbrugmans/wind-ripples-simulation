"""Run the wind ripple model with the settings of the article (Brugmans, in prep.).

Usage:
    python3 run_model.py <seed> [series] [NAME=value ...]

Examples:
    python3 run_model.py 271                      # the article model (series SR0), 4 hours of model time, seed 271
    python3 run_model.py 271 SG0                  # series SG0: the article model without wind-driven sliding
    python3 run_model.py 271 SR0 TEIND=60 NAAMSUF=_test   # 60 s of model time, output c23_z271_test.npz

The output (<name>.npz) and a manifest (<name>.manifest.json: settings, source hashes, versions, output hash)
are written to the folder runs/. The C library is built on first use with clang (-O3, no -ffast-math).
Settings are passed to the model code as environment variables; extra NAME=value pairs override the article
settings. The series and their settings are listed in series.py; the output is named c23_z<seed>_<series>.npz
unless NAAMSUF is given.
"""
import hashlib, json, os, platform, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "model")

from series import SERIES, ARTICLE   # settings per run series; ARTICLE = SERIES['SR0'], the article model


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    if len(sys.argv) < 2 or not sys.argv[1].isdigit():
        print(__doc__); sys.exit(1)
    seed = sys.argv[1]; args = sys.argv[2:]
    series = args.pop(0) if args and "=" not in args[0] else "SR0"
    if series not in SERIES: print("unknown series", series, "- known:", " ".join(SERIES)); sys.exit(1)
    extra = dict(a.split("=", 1) for a in args)
    env = dict(os.environ); env.update(SERIES[series]); env.setdefault("NAAMSUF", "_" + series); env.update(extra)
    runs = os.path.join(HERE, "runs"); os.makedirs(runs, exist_ok=True)
    name = f"{env['CVAR'].lower()}_z{seed}{env.get('NAAMSUF', '')}"
    sources = sorted(f for f in os.listdir(MODEL) if f.endswith((".c", ".py")) or f.startswith("tabel_"))
    rec = dict(name=name, seed=int(seed), series=series, command=" ".join(["python3", "run_model.py"] + sys.argv[1:]),
               settings={k: env[k] for k in sorted(set(SERIES[series]) | set(extra))}, extra=extra,
               sources={f: sha(os.path.join(MODEL, f)) for f in sources},
               python=sys.version.split()[0], platform=platform.platform(), started=time.strftime("%Y-%m-%d %H:%M:%S"))
    try:
        import numpy; rec["numpy"] = numpy.__version__
    except ImportError:
        pass
    rc = subprocess.call([sys.executable, os.path.join(MODEL, "run_fetch.py"), "ref", seed], cwd=runs, env=env)
    out = os.path.join(runs, name + ".npz")
    rec.update(finished=time.strftime("%Y-%m-%d %H:%M:%S"), returncode=rc, output={name + ".npz": sha(out)} if os.path.exists(out) else {})
    with open(os.path.join(runs, name + ".manifest.json"), "w") as f:
        json.dump(rec, f, indent=1)
    sys.exit(rc)


if __name__ == "__main__":
    main()
