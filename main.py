import argparse
import json
import os
import runpy
import sys
import signal
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

# Set before OpenCV opens an RTSP stream.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

from src.pipeline import Pipeline  # noqa: E402


PROJECT_DIR = Path(__file__).resolve().parent
VIDEO_EXTENSIONS = "*.mp4 *.avi *.mov *.mkv *.wmv *.m4v"


def choose_video(data_dir: Path) -> str:
    """Open a file picker in the project's data directory."""
    root = tk.Tk()
    root.withdraw()
    try:
        video_path = filedialog.askopenfilename(
            title="Select a video",
            initialdir=str(data_dir),
            filetypes=[("Video files", VIDEO_EXTENSIONS), ("All files", "*.*")],
        )
    finally:
        root.destroy()

    if not video_path:
        print("No video selected. Program stopped.")
        raise SystemExit(0)
    print(f"\nSelected video: {video_path}\n")
    return video_path


def _resolve_config_paths(config: dict, base_dir: Path) -> None:
    """Resolve local file paths relative to the config file, not the shell CWD."""
    path_fields = (
        ("input", "video_path"),
        ("detection", "model_path"),
        ("storage", "db_path"),
        ("storage", "logs_dir"),
        ("storage", "log_file"),
        ("output", "annotated_video_path"),
    )
    for section, key in path_fields:
        value = config.get(section, {}).get(key)
        if value:
            path = Path(value).expanduser()
            if not path.is_absolute():
                config[section][key] = str((base_dir / path).resolve())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Intelligent Face Tracker & Visitor Counter"
    )
    parser.add_argument("--config", default=str(PROJECT_DIR / "config.json"))
    parser.add_argument("--mode", choices=["video", "rtsp"], help="Override input mode")
    parser.add_argument("--source", help="Video path or RTSP URL")
    parser.add_argument(
        "--preserve-data", action="store_true",
        help="Append this run to the existing database and event log instead of resetting them",
    )
    parser.add_argument("--stop-file", help=argparse.SUPPRESS)
    parser.add_argument("--annotated-output", help=argparse.SUPPRESS)
    args = parser.parse_args()

    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = (PROJECT_DIR / config_path).resolve()
    with config_path.open(encoding="utf-8") as file:
        config = json.load(file)

    _resolve_config_paths(config, config_path.parent)

    if args.preserve_data:
        config["storage"]["reset_on_start"] = False
    if args.stop_file:
        stop_path = Path(args.stop_file).expanduser()
        config["_stop_file"] = str(
            stop_path if stop_path.is_absolute() else (PROJECT_DIR / stop_path).resolve()
        )
    if args.annotated_output:
        output_path = Path(args.annotated_output).expanduser()
        config["output"]["annotated_video_path"] = str(
            output_path if output_path.is_absolute() else (PROJECT_DIR / output_path).resolve()
        )

    if args.mode:
        config["input"]["mode"] = args.mode

    if args.source:
        if config["input"]["mode"] == "rtsp":
            config["input"]["rtsp_url"] = args.source
        else:
            source = Path(args.source).expanduser()
            config["input"]["video_path"] = str(
                source if source.is_absolute() else (Path.cwd() / source).resolve()
            )
    elif config["input"]["mode"] == "video":
        config["input"]["video_path"] = choose_video(PROJECT_DIR / "data")

    # Flush active tracks and close SQLite cleanly when terminated by the OS.
    signal.signal(
        signal.SIGTERM,
        lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    total = Pipeline(config).run()
    print(f"\nUnique visitors: {total}")
        # Print the detailed database report in the same IDLE Shell.
    report_path = PROJECT_DIR / "report.py"
    saved_argv = sys.argv[:]
    try:
        sys.argv = [
            str(report_path),
            config["storage"]["db_path"],
            config["storage"]["log_file"],
        ]
        runpy.run_path(str(report_path), run_name="__main__")
    finally:
        sys.argv = saved_argv



if __name__ == "__main__":
    main()
