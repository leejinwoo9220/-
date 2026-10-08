"""rvt - road-view time-lapse pipeline.

  python -m rvt init     CAPTURES_DIR [--scene "..."]   # write project.json from dated capture files
  python -m rvt run      project.json [--execute]       # prep -> plan -> generate -> finish
  python -m rvt prep     project.json          # common framing of the stills (no lens warp by default)
  python -m rvt plan     project.json          # H3 prompts + sogni-agent commands (free)
  python -m rvt generate project.json [--execute] [--only SEG ...]
  python -m rvt seam     project.json SEG      # rebuild a seam frame from a hold clip
  python -m rvt finish   project.json          # lock signs, fix frozen subjects, ramp, assemble
  python -m rvt demo     OUT_DIR               # offline end-to-end run on synthetic data
"""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rvt", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prep", "plan", "finish"):
        sub.add_parser(name).add_argument("project")
    g = sub.add_parser("generate")
    g.add_argument("project")
    g.add_argument("--execute", action="store_true", help="actually call sogni-agent (otherwise dry-run)")
    g.add_argument("--only", nargs="*", help="segment ids to run")
    s = sub.add_parser("seam")
    s.add_argument("project")
    s.add_argument("segment", help="hold segment id, e.g. hold_2009")
    d = sub.add_parser("demo")
    d.add_argument("out_dir")
    i = sub.add_parser("init")
    i.add_argument("captures_dir")
    i.add_argument("--out", default="project.json")
    i.add_argument("--scene", help="one English line describing the street (used in the H3 prompts)")
    r = sub.add_parser("run")
    r.add_argument("project")
    r.add_argument("--execute", action="store_true", help="actually call sogni-agent (otherwise dry-run)")
    args = ap.parse_args(argv)

    from . import project as P

    if args.cmd == "init":
        out = P.scaffold(args.captures_dir, args.out, args.scene)
        print(f"wrote {out} - check ids/dates, add `scene`, then: python -m rvt run {out.name} --execute")
        return 0
    if args.cmd == "demo":
        from .demo import run_demo
        run_demo(args.out_dir)
        return 0
    p = P.load(args.project)
    if args.cmd == "prep":
        from . import prep
        prep.run(p)
    elif args.cmd == "plan":
        from . import sogni
        plan = sogni.build_plan(p)
        est = plan["estimate"]
        print(f"{len(plan['segments'])} segments, {est['generated_seconds']}s of H3 {plan['tier']} "
              f"(~{est['spark_if_not_covered']} Spark / ${est['usd_if_not_covered']} if not covered by Unlimited)")
    elif args.cmd == "generate":
        from . import sogni
        sogni.run(p, execute=args.execute, only=args.only)
    elif args.cmd == "seam":
        from . import finish
        from .util import read_json
        finish.make_seam(p, read_json(P.work(p, "plan.json")), args.segment)
    elif args.cmd == "finish":
        from . import finish
        finish.run(p)
    elif args.cmd == "run":
        from . import finish, prep, sogni
        from .util import log, read_json
        prep.run(p)
        sogni.build_plan(p)
        sogni.run(p, execute=args.execute)
        plan = read_json(P.work(p, "plan.json"))
        missing = [s["id"] for s in plan["segments"] if not (p["_root"] / s["out"]).exists()]
        if missing:
            log(f"not finished: clips missing {missing}" + ("" if args.execute else " (dry-run: add --execute)"))
            return 1
        finish.run(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
