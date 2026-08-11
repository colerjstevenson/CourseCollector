from __future__ import annotations

import argparse
import os
from pathlib import Path

from .config import load_config
from .runtime import RunContext, mark_stage_done, setup_logging, stage_done
from .stages import run_collect, run_combine, run_match, run_scrape


STAGE_ORDER = ["collect", "scrape", "combine", "match"]


def _select_stages(args: argparse.Namespace) -> list[str]:
    if args.only_stage:
        return [args.only_stage]

    stages = list(STAGE_ORDER)

    if args.from_stage:
        start_idx = STAGE_ORDER.index(args.from_stage)
        stages = stages[start_idx:]

    if args.to_stage:
        end_idx = STAGE_ORDER.index(args.to_stage)
        stages = stages[: end_idx + 1]

    return stages


def _target_block(config: dict, target: str) -> dict:
    targets = config.get("targets", {})
    if target not in targets:
        raise ValueError(f"Unknown target '{target}'. Known targets: {', '.join(sorted(targets.keys()))}")
    return targets[target]


def _run_stage(stage: str, ctx: RunContext, config: dict, args: argparse.Namespace, logger) -> None:
    if ctx.resume and not ctx.force and stage_done(ctx.checkpoint_path, stage):
        logger.info("Skipping stage '%s' because checkpoint indicates completion.", stage)
        return

    if ctx.dry_run:
        logger.info("[dry-run] Would run stage: %s", stage)
        return

    target_cfg = _target_block(config, ctx.target)

    if stage == "collect":
        regions_file = Path(args.regions_file or target_cfg.get("regions_file", "states_list.txt"))
        metadata = run_collect(ctx.target, ctx.repo_root / regions_file)
    elif stage == "scrape":
        scrape_cfg = config.get("scrape", {})
        metadata = run_scrape(
            enable_golflink=bool(scrape_cfg.get("golflink", True)),
            enable_golfcanada=bool(scrape_cfg.get("golfcanada", False)),
            enable_golfdigest=bool(scrape_cfg.get("golfdigest", False)),
        )
    elif stage == "combine":
        data_dir = ctx.repo_root / Path(target_cfg["data_dir"])
        combined_csv = ctx.repo_root / Path(target_cfg["combined_csv"])
        metadata = run_combine(data_dir, combined_csv)
    elif stage == "match":
        combined_csv = ctx.repo_root / Path(target_cfg["combined_csv"])
        postal_csv = ctx.repo_root / Path(target_cfg["postal_csv"])
        matched_csv = ctx.repo_root / Path(target_cfg["matched_csv"])
        golflink_csv = ctx.repo_root / Path(config.get("inputs", {}).get("golflink_csv", "data/golfLinkData.csv"))
        metadata = run_match(combined_csv, postal_csv, golflink_csv, matched_csv)
    else:
        raise ValueError(f"Unsupported stage: {stage}")

    mark_stage_done(ctx.checkpoint_path, stage, metadata)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="CourseCollector",
        description="Unified non-map pipeline orchestrator for collection, scraping, combining, and matching.",
    )

    parser.add_argument("command", nargs="?", default="run", choices=["run", "collect", "scrape", "combine", "match"])
    parser.add_argument("--target", default="world", choices=["usa", "world"], help="Target profile to execute.")
    parser.add_argument("--config", help="Path to YAML config file relative to repository root.")
    parser.add_argument("--regions-file", help="Override regions/states file for collect stage.")

    parser.add_argument("--dry-run", action="store_true", help="Print what would run without executing stages.")
    parser.add_argument("--resume", action="store_true", help="Skip stages already marked complete in checkpoint.")
    parser.add_argument("--force", action="store_true", help="Ignore completion checkpoint and rerun requested stages.")

    parser.add_argument("--from-stage", choices=STAGE_ORDER, help="Start execution from this stage.")
    parser.add_argument("--to-stage", choices=STAGE_ORDER, help="Stop execution after this stage.")
    parser.add_argument("--only-stage", choices=STAGE_ORDER, help="Execute only one stage.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]
    os.chdir(repo_root)
    config = load_config(repo_root, args.config)

    log_path = repo_root / Path(config.get("logging", {}).get("path", "golf_course_collection.log"))
    checkpoint_path = repo_root / Path(config.get("checkpoint", {}).get("path", ".course_collector/checkpoint.json"))

    ctx = RunContext(
        repo_root=repo_root,
        target=args.target,
        dry_run=bool(args.dry_run),
        force=bool(args.force),
        resume=bool(args.resume),
        log_path=log_path,
        checkpoint_path=checkpoint_path,
    )

    logger = setup_logging(ctx.log_path)
    logger.info("Starting CourseCollector command=%s target=%s", args.command, ctx.target)

    if args.command == "run":
        stages = _select_stages(args)
    else:
        if args.from_stage or args.to_stage or args.only_stage:
            parser.error("--from-stage/--to-stage/--only-stage are only valid with command 'run'")
        stages = [args.command]

    if args.only_stage and args.command != "run" and args.command != args.only_stage:
        parser.error("--only-stage cannot conflict with explicit command")

    for stage in stages:
        logger.info("Running stage: %s", stage)
        _run_stage(stage, ctx, config, args, logger)
        logger.info("Finished stage: %s", stage)

    logger.info("CourseCollector completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
