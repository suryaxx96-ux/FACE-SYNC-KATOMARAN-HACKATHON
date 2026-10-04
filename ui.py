"""Small Tkinter dashboard for running and monitoring the face tracker.

Run with: python ui.py
Tkinter is part of the standard Windows Python installation; no web server is used.
"""
import json
import os
import re
import csv
import sqlite3
import subprocess
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_DIR / "config.json"
ICON_PNG = PROJECT_DIR / "assets" / "face_sync_icon.png"
ICON_ICO = PROJECT_DIR / "assets" / "face_sync_icon.ico"
BRAND_PNG = PROJECT_DIR / "assets" / "face_sync_reference.png"


class TrackerDashboard:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("FACE SYNC · VISITOR ANALYTICS")
        self.root.geometry("1020x720")
        self.root.minsize(850, 600)
        self.process = None
        self.console_file = None
        self.started_at = None
        self.stop_path = None
        self.close_when_stopped = False
        self.stats_log_offset = 0
        self.last_output_path = None
        self.run_started_at = None
        self.database_poll_after = 0.0
        self.active_source_name = None
        self.registration_values = []
        self.last_frame = 0
        self.last_fps = None
        self.pulse_phase = False
        self.config_path = DEFAULT_CONFIG
        self.config = self._load_config()
        self._set_app_icon()
        try:
            # Tk does not scale Canvas images automatically; fit the supplied
            # logo to the sidebar slot without cropping it.
            source_logo = tk.PhotoImage(file=str(BRAND_PNG))
            logo_scale = max(1, (source_logo.width() + 45) // 46)
            self.brand_image = source_logo.subsample(logo_scale, logo_scale)
        except (tk.TclError, OSError):
            self.brand_image = None

        self.mode_var = tk.StringVar(value="video")
        self.source_var = tk.StringVar(value=self._default_video())
        self.preserve_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Ready")
        self.count_var = tk.StringVar(value="0")
        self.entry_count_var = tk.StringVar(value="0")
        self.exit_count_var = tk.StringVar(value="0")
        self.frames_var = tk.StringVar(value="0")
        self.elapsed_var = tk.StringVar(value="00:00")
        self.fps_var = tk.StringVar(value="Waiting for a run")
        self.health_var = tk.StringVar(value="Ready · choose a video or camera stream")
        self.detection_var = tk.StringVar(value="0 face boxes in latest detector pass")
        self.database_health_var = tk.StringVar(value="SQLite ready · no session started")
        self.latest_events_var = tk.StringVar(value="No event records yet · start a run to collect evidence")
        for display_var in (self.status_var, self.fps_var, self.health_var,
                            self.detection_var, self.database_health_var,
                            self.latest_events_var):
            display_var.trace_add("write", lambda *_args, item=display_var: self._uppercase_var(item))
            self._uppercase_var(display_var)
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._refresh()

    def _load_config(self):
        try:
            return json.loads(self.config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            messagebox.showerror("Config error", f"Could not read config.json:\n{exc}")
            return {}

    @staticmethod
    def _uppercase_var(variable):
        value = variable.get()
        if value and value != value.upper():
            variable.set(value.upper())

    def _default_video(self):
        configured = self.config.get("input", {}).get("video_path", "data")
        path = Path(configured)
        return str(path if path.is_absolute() else PROJECT_DIR / path)

    def _set_app_icon(self):
        """Apply the Face Sync face-scan mark to the window and popups."""
        try:
            self._icon_image = tk.PhotoImage(file=str(ICON_PNG))
            self.root.iconphoto(True, self._icon_image)
        except (tk.TclError, OSError):
            self._icon_image = None
        try:
            self.root.iconbitmap(default=str(ICON_ICO))
        except (tk.TclError, OSError):
            pass

    def _build(self):
        style = ttk.Style()
        style.theme_use("clam")
        self.colors = {
            "page": "#0D1222", "sidebar": "#11182B", "panel": "#171F35",
            "panel2": "#11192D", "stroke": "#2B3855", "text": "#F7F8FF",
            "muted": "#AEBBD4", "accent": "#27E6C1", "blue": "#39BDF8",
            "warning": "#FF55A5", "danger": "#FF6B78", "green": "#FFAD45",
        }
        c = self.colors
        # Deep navy surfaces with vivid teal, cyan, pink, violet and amber accents.
        self.root.configure(bg="#080C18")
        self.root.title("FACE SYNC · VISITOR ANALYTICS")
        self.root.geometry("1560x980")
        self.root.minsize(1200, 820)
        style.configure("Dark.TCombobox", fieldbackground=c["panel2"], background=c["panel2"],
                        foreground=c["text"], arrowcolor=c["text"], bordercolor=c["stroke"],
                        padding=8)
        style.map("Dark.TCombobox", fieldbackground=[("readonly", c["panel2"])],
                  foreground=[("readonly", c["text"])])
        style.configure("Dark.Horizontal.TProgressbar", troughcolor=c["panel2"],
                        background=c["accent"], bordercolor=c["panel2"], lightcolor=c["accent"],
                        darkcolor=c["accent"])
        style.configure("Dark.Treeview", background=c["panel"], fieldbackground=c["panel"],
                        foreground=c["text"], rowheight=34, borderwidth=0,
                        font=("Segoe UI", 9))
        style.configure("Dark.Treeview.Heading", background=c["panel2"], foreground=c["text"],
                        relief="flat", padding=(10, 9), font=("Segoe UI", 9, "bold"))
        style.map("Dark.Treeview", background=[("selected", "#267568")],
                  foreground=[("selected", c["text"])])

        shell = tk.Frame(self.root, bg=c["page"])
        shell.pack(fill="both", expand=True, padx=7, pady=7)
        sidebar = tk.Frame(shell, bg=c["sidebar"], width=220)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        brand = tk.Frame(sidebar, bg=c["sidebar"])
        brand.pack(fill="x", padx=19, pady=(26, 34))
        mark = tk.Canvas(brand, width=46, height=46, bg=c["sidebar"], highlightthickness=0)
        mark.pack(side="left")
        if self.brand_image:
            mark.create_image(23, 23, image=self.brand_image)
        else:
            mark.create_oval(4, 4, 42, 42, fill="#0A1020", outline=c["accent"], width=2)
            mark.create_line(11, 16, 17, 16, 11, 22, fill=c["text"], width=2)
            mark.create_line(35, 16, 29, 16, 35, 22, fill=c["text"], width=2)
            mark.create_oval(19, 14, 27, 22, fill=c["accent"], outline="")
            mark.create_oval(15, 22, 31, 32, fill=c["accent"], outline="")
            mark.create_line(12, 24, 34, 24, fill=c["text"], width=2)
        brand_text = tk.Frame(brand, bg=c["sidebar"])
        brand_text.pack(side="left", padx=10)
        tk.Label(brand_text, text="FACE SYNC", bg=c["sidebar"], fg=c["text"], font=("Segoe UI", 12, "bold")).pack(anchor="w")
        tk.Label(brand_text, text="VISITOR ANALYTICS", bg=c["sidebar"], fg=c["muted"], font=("Segoe UI", 7, "bold")).pack(anchor="w")

        tk.Label(sidebar, text="WORKSPACE", bg=c["sidebar"], fg=c["muted"], font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=24, pady=(0, 10))
        self._nav_item(sidebar, "▦", "Dashboard", active=True)
        self._nav_item(sidebar, "◉", "Visitors", command=self._open_visitors)
        self._nav_item(sidebar, "▤", "Reports", command=self._open_report)
        tk.Label(sidebar, text="LIBRARY", bg=c["sidebar"], fg=c["muted"], font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=24, pady=(28, 10))
        self._nav_item(sidebar, "▶", "Annotated video", command=self._open_video)
        self._nav_item(sidebar, "▣", "Photo gallery", command=self._open_photo_gallery)

        side_bottom = tk.Frame(sidebar, bg=c["sidebar"])
        side_bottom.pack(side="bottom", fill="x", padx=20, pady=20)
        tk.Frame(side_bottom, bg=c["stroke"], height=1).pack(fill="x", pady=(0, 14))
        tk.Label(side_bottom, text="LOCAL PROCESSING", bg=c["sidebar"], fg=c["muted"], font=("Segoe UI", 8, "bold")).pack(anchor="w")
        tk.Label(side_bottom, text="●  SQLITE  ·  FACE RECOGNITION", bg=c["sidebar"], fg=c["accent"], font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(7, 0))

        main = tk.Frame(shell, bg=c["page"])
        main.pack(side="left", fill="both", expand=True, padx=27, pady=22)
        header = tk.Frame(main, bg=c["page"])
        header.pack(fill="x", pady=(0, 22))
        head_left = tk.Frame(header, bg=c["page"])
        head_left.pack(side="left", fill="x", expand=True)
        tk.Label(head_left, text="FACE SYNC", bg=c["page"], fg=c["text"], font=("Segoe UI", 25, "bold")).pack(anchor="w")
        tk.Label(head_left, text="VISITOR INTELLIGENCE  ·  LIVE TRACKING  ·  REGISTRATION", bg=c["page"], fg=c["muted"], font=("Segoe UI", 10)).pack(anchor="w", pady=(4, 0))
        state = tk.Frame(header, bg=c["panel"], highlightbackground=c["stroke"], highlightthickness=1)
        state.pack(side="right", padx=(10, 0), ipadx=13, ipady=8)
        self.status_dot = tk.Canvas(state, width=14, height=14, bg=c["panel"], highlightthickness=0)
        self.status_dot.pack(side="left", padx=(0, 8))
        self.status_dot_id = self.status_dot.create_oval(3, 3, 11, 11, fill=c["muted"], outline="")
        tk.Label(state, textvariable=self.status_var, bg=c["panel"], fg=c["text"], font=("Segoe UI", 9, "bold")).pack(side="left")

        source_card = self._panel(main, "START A TRACKING SESSION", "Choose a video or connect to a live camera")
        source_card.pack(fill="x", pady=(0, 14))
        source_row = tk.Frame(source_card, bg=c["panel"])
        source_row.pack(fill="x", padx=18)
        tk.Label(source_row, text="MODE", bg=c["panel"], fg=c["muted"], font=("Segoe UI", 8, "bold")).pack(side="left", padx=(0, 10))
        mode = ttk.Combobox(source_row, textvariable=self.mode_var, values=("video", "rtsp"), width=10,
                            state="readonly", style="Dark.TCombobox")
        mode.pack(side="left", padx=(0, 16))
        mode.bind("<<ComboboxSelected>>", self._mode_changed)
        self.source_entry = tk.Entry(source_row, textvariable=self.source_var, bg=c["panel2"], fg=c["text"],
                                     insertbackground=c["accent"], relief="flat", font=("Segoe UI", 10),
                                     highlightthickness=1, highlightbackground=c["stroke"], highlightcolor=c["accent"])
        self.source_entry.pack(side="left", fill="x", expand=True, ipady=10, padx=(0, 9))
        self.browse_button = self._action_button(source_row, "Browse video", self._browse, c["panel2"], c["text"], c["stroke"])
        self.browse_button.pack(side="left")
        source_bottom = tk.Frame(source_card, bg=c["panel"])
        source_bottom.pack(fill="x", padx=18, pady=(11, 16))
        self.start_button = self._action_button(source_bottom, "▶   START TRACKING", self._start, c["accent"], "#102A2A", "#67DFC5", bold=True)
        self.start_button.pack(side="left")
        self.stop_button = self._action_button(source_bottom, "■   Stop safely", self._stop, "#713D4A", "#FFF1F2", "#8B4B58")
        self.stop_button.pack(side="left", padx=9)
        self.stop_button.configure(state="disabled")
        self.preserve_check = tk.Checkbutton(source_bottom, text="APPEND TO PREVIOUS RESULTS", variable=self.preserve_var,
                                             bg=c["panel"], fg=c["muted"], activebackground=c["panel"],
                                             activeforeground=c["text"], selectcolor=c["panel2"],
                                             font=("Segoe UI", 9), relief="flat", bd=0)
        self.preserve_check.pack(side="left", padx=(18, 0))
        tk.Label(source_bottom, text="OFF = CLEAN RUN   ·   ON = RETAIN PRIOR VISITOR DATA", bg=c["panel"], fg=c["muted"], font=("Segoe UI", 8)).pack(side="right")
        self.activity_bar = ttk.Progressbar(source_card, mode="indeterminate", style="Dark.Horizontal.TProgressbar")
        self.activity_bar.pack(fill="x", padx=18, pady=(0, 13))

        metrics = tk.Frame(main, bg=c["page"])
        metrics.pack(fill="x", pady=(0, 14))
        self._metric_card(metrics, 0, "UNIQUE VISITORS", self.count_var, "REGISTERED FACE IDENTITIES", c["accent"], "◉")
        self._metric_card(metrics, 1, "ENTRY EVENTS", self.entry_count_var, "CAPTURED ARRIVALS", c["blue"], "⌂")
        self._metric_card(metrics, 2, "EXIT EVENTS", self.exit_count_var, "CAPTURED DEPARTURES", c["warning"], "▣")
        self._metric_card(metrics, 3, "FRAMES PROCESSED", self.frames_var, "CURRENT SESSION", c["green"], "▥")
        for column in range(4):
            metrics.grid_columnconfigure(column, weight=1, uniform="metric")

        health_panel = self._panel(
            main, "SYSTEM HEALTH", "Run state · processing speed · detector activity · database status",
        )
        health_panel.pack(fill="x", pady=(0, 14))
        health_row = tk.Frame(health_panel, bg=c["panel"])
        health_row.pack(fill="x", padx=16, pady=(0, 14))
        for column, (title, variable, color) in enumerate((
            ("RUN STATUS", self.health_var, c["accent"]),
            ("PROCESSING SPEED", self.fps_var, c["warning"]),
            ("LATEST DETECTIONS", self.detection_var, c["green"]),
            ("DATABASE", self.database_health_var, c["blue"]),
        )):
            cell = tk.Frame(health_row, bg=c["panel2"])
            cell.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 8, 0))
            health_row.grid_columnconfigure(column, weight=1, uniform="health")
            tk.Label(cell, text=title, bg=c["panel2"], fg=color,
                     font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=12, pady=(9, 3))
            tk.Label(cell, textvariable=variable, bg=c["panel2"], fg=c["text"],
                     font=("Segoe UI", 8), anchor="w", justify="left", wraplength=215,
                     height=2).pack(fill="x", padx=12, pady=(0, 9))

        overview = tk.Frame(main, bg=c["page"])
        overview.pack(fill="both", expand=True, pady=(0, 14))
        trend_panel = self._panel(overview, "VISITOR REGISTRATION TREND", "Cumulative identities registered during processing")
        trend_panel.pack(side="left", fill="both", expand=True, padx=(0, 12))
        self.trend_canvas = tk.Canvas(trend_panel, height=170, bg=c["panel"], highlightthickness=0)
        self.trend_canvas.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.trend_canvas.bind("<Configure>", lambda _event: self._draw_trend())
        self._draw_trend()

        live = tk.Frame(main, bg=c["page"])
        live.pack(fill="both", expand=True)
        event_panel = self._panel(live, "LATEST VISITOR EVENTS", "One row per saved arrival/departure; double-click a row to open its evidence photo")
        event_panel.pack(fill="both", expand=True)
        tk.Label(event_panel, textvariable=self.latest_events_var, bg=c["panel"], fg=c["accent"],
                 font=("Segoe UI", 8, "bold"), anchor="w").pack(fill="x", padx=17, pady=(0, 8))
        table_wrap = tk.Frame(event_panel, bg=c["panel"])
        table_wrap.pack(fill="both", expand=True, padx=14, pady=(2, 12))
        columns = ("event_id", "time", "type", "face", "track", "image")
        self.events = ttk.Treeview(table_wrap, columns=columns, show="headings", height=7, style="Dark.Treeview")
        self.events.heading("event_id", text="")
        self.events.column("event_id", width=0, stretch=False)
        for col, title, width in (("time", "TIMESTAMP", 190), ("type", "EVENT", 90), ("face", "FACE ID", 110), ("track", "TRACK", 65), ("image", "SAVED IMAGE", 410)):
            self.events.heading(col, text=title)
            self.events.column(col, width=width, anchor="w")
        self.events.tag_configure("ENTRY", foreground=c["green"])
        self.events.tag_configure("EXIT", foreground=c["warning"])
        scrollbar = ttk.Scrollbar(table_wrap, orient="vertical", command=self.events.yview)
        self.events.configure(yscrollcommand=scrollbar.set)
        self.events.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.events.bind("<Double-1>", lambda _event: self._open_selected_image())
        self.events_empty_label = tk.Label(
            table_wrap,
            text="NO VISITOR EVENTS YET. START A VIDEO RUN TO RECORD ENTRIES, EXITS AND PHOTOS.",
            bg=c["panel"], fg=c["muted"], font=("Segoe UI", 10),
        )
        self.events_empty_label.place(relx=0.5, rely=0.55, anchor="center")

        footer = tk.Frame(overview, bg=c["page"], width=255)
        footer.pack(side="right", fill="y")
        footer.pack_propagate(False)
        console_panel = self._panel(footer, "PIPELINE LOG", "Technical messages from the current run")
        console_panel.pack(fill="both", expand=True)
        console_inner = tk.Frame(console_panel, bg="#151E2E", highlightbackground=c["stroke"], highlightthickness=1)
        console_inner.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.console = tk.Text(console_inner, height=6, wrap="word", state="disabled", bg="#151E2E", fg="#DCE7FF",
                               insertbackground=c["accent"], relief="flat", font=("Cascadia Code", 8), padx=10, pady=8)
        self.console.pack(fill="both", expand=True)
        self.console.configure(state="normal")
        self.console.insert("end", "DASHBOARD READY. SELECT A SOURCE AND PRESS START TRACKING TO BEGIN A RUN.\n")
        self.console.configure(state="disabled")
        self._animate_status()

    def _panel(self, parent, title, subtitle):
        c = self.colors
        panel = tk.Frame(parent, bg=c["panel"], highlightbackground=c["stroke"], highlightthickness=1)
        head = tk.Frame(panel, bg=c["panel"])
        head.pack(fill="x", padx=17, pady=(13, 11))
        tk.Label(head, text=title.upper(), bg=c["panel"], fg=c["text"], font=("Segoe UI", 9, "bold")).pack(anchor="w")
        tk.Label(head, text=subtitle.upper(), bg=c["panel"], fg=c["muted"], font=("Segoe UI", 8)).pack(anchor="w", pady=(3, 0))
        return panel

    def _metric_card(self, parent, column, title, variable, caption, accent, icon):
        c = self.colors
        tile_color = {
            c["accent"]: "#087F72", c["blue"]: "#176EAA",
            c["warning"]: "#7941B8", c["green"]: "#B96925",
        }.get(accent, c["panel"])
        card = tk.Frame(parent, bg=tile_color, highlightbackground=tile_color, highlightthickness=1)
        card.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 8, 0))
        top = tk.Frame(card, bg=tile_color)
        top.pack(fill="x", padx=16, pady=(13, 3))
        tk.Label(top, text=title.upper(), bg=tile_color, fg="#F3F3F7", font=("Segoe UI", 8, "bold")).pack(side="left")
        tk.Label(top, text=icon, bg=tile_color, fg="#FFFFFF", font=("Segoe UI Symbol", 14, "bold")).pack(side="right")
        tk.Label(card, textvariable=variable, bg=tile_color, fg="#FFFFFF", font=("Segoe UI", 25, "bold")).pack(anchor="w", padx=16, pady=(0, 0))
        tk.Label(card, text=caption.upper(), bg=tile_color, fg="#E2E3EA", font=("Segoe UI", 8)).pack(anchor="w", padx=16, pady=(0, 13))

    def _draw_trend(self):
        canvas = getattr(self, "trend_canvas", None)
        if canvas is None:
            return
        c = self.colors
        canvas.delete("all")
        width = max(320, canvas.winfo_width())
        height = max(130, canvas.winfo_height())
        left, right, top, bottom = 38, width - 18, 22, height - 28
        for idx in range(4):
            y = top + (bottom - top) * idx / 3
            canvas.create_line(left, y, right, y, fill=c["stroke"], dash=(2, 4))
            canvas.create_text(4, y, anchor="w", text=str(max(0, len(self.registration_values) * (3 - idx) // 3)), fill=c["muted"], font=("Segoe UI", 7))
        canvas.create_text(left, height - 5, anchor="sw", text="REGISTRATION ORDER", fill=c["muted"], font=("Segoe UI", 7, "bold"))
        canvas.create_text(right, 8, anchor="ne", text=f"TOTAL  {len(self.registration_values)}", fill=c["accent"], font=("Segoe UI", 8, "bold"))
        if not self.registration_values:
            canvas.create_text((left + right) / 2, (top + bottom) / 2, text="VISITOR REGISTRATIONS WILL APPEAR HERE", fill=c["muted"], font=("Segoe UI", 9))
            return
        total = len(self.registration_values)
        points = []
        for index in range(total):
            x = left + (right - left) * index / max(1, total - 1)
            y = bottom - (bottom - top) * (index + 1) / max(1, total)
            points.extend((x, y))
        if total > 1:
            area = [left, bottom, *points, right, bottom]
            canvas.create_polygon(*area, fill="#24484B", outline="", smooth=True, splinesteps=18)
            canvas.create_line(*points, fill=c["accent"], width=3, smooth=True, splinesteps=20, capstyle="round")
        x, y = points[-2:]
        canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill=c["accent"], outline="#D3FFF5", width=1)

    def _action_button(self, parent, text, command, bg, fg, hover, bold=False):
        button = tk.Button(parent, text=text.upper(), command=command, bg=bg, fg=fg,
                           activebackground=hover, activeforeground=fg, relief="flat", bd=0,
                           cursor="hand2", padx=15, pady=9,
                           font=("Segoe UI", 9, "bold" if bold else "normal"))
        button.bind("<Enter>", lambda _event: button.configure(bg=hover) if str(button["state"]) == "normal" else None)
        button.bind("<Leave>", lambda _event: button.configure(bg=bg) if str(button["state"]) == "normal" else None)
        return button

    def _nav_item(self, parent, icon, label, command=None, active=False):
        c = self.colors
        bg = "#202B46" if active else c["sidebar"]
        fg = c["accent"] if active else c["muted"]
        item = tk.Frame(parent, bg=bg, cursor="hand2")
        item.pack(fill="x", padx=12, pady=3)
        if active:
            tk.Frame(item, bg=c["accent"], width=3).pack(side="left", fill="y")
        tk.Label(item, text=icon, bg=bg, fg=fg, font=("Segoe UI Symbol", 13), width=3, pady=9).pack(side="left")
        label_widget = tk.Label(item, text=label.upper(), bg=bg, fg=c["text"] if active else c["muted"], font=("Segoe UI", 9, "bold" if active else "normal"), anchor="w", pady=9)
        label_widget.pack(side="left", fill="x", expand=True)
        if command:
            for widget in (item, *item.winfo_children()):
                widget.bind("<Button-1>", lambda _event, fn=command: fn())

    def _focus_events(self):
        if hasattr(self, "events"):
            self.events.focus_set()

    def _read_all_data(self):
        """Read the current visitor directory and saved event evidence."""
        db_path, _ = self._paths()
        if not db_path.exists():
            return [], []
        conn = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True, timeout=1.0)
        try:
            faces = conn.execute(
                "SELECT face_id, COALESCE(label, ''), first_seen FROM faces ORDER BY face_id"
            ).fetchall()
            events = conn.execute(
                "SELECT event_id, timestamp, event_type, face_id, track_id, frame_idx, image_path "
                "FROM events ORDER BY timestamp DESC, event_id DESC"
            ).fetchall()
            return faces, events
        except sqlite3.Error as exc:
            messagebox.showerror("Could not read visitor database", f"{exc}\n\nDatabase:\n{db_path}")
            return [], []
        finally:
            conn.close()

    def _popup(self, title, geometry):
        window = tk.Toplevel(self.root)
        window.title(f"FACE SYNC · {title.upper()}")
        window.geometry(geometry)
        window.minsize(760, 520)
        window.configure(bg=self.colors["page"])
        if getattr(self, "_icon_image", None):
            try:
                window.iconphoto(True, self._icon_image)
            except tk.TclError:
                pass
        return window

    def _window_header(self, parent, title, subtitle):
        c = self.colors
        tk.Label(parent, text=title.upper(), bg=c["page"], fg=c["text"],
                 font=("Segoe UI", 22, "bold")).pack(anchor="w")
        tk.Label(parent, text=subtitle.upper(), bg=c["page"], fg=c["muted"],
                 font=("Segoe UI", 10)).pack(anchor="w", pady=(3, 14))

    def _popup_refresh(self, parent, window, callback):
        row = tk.Frame(parent, bg=self.colors["page"])
        row.pack(fill="x", pady=(0, 10))
        self._action_button(
            row, "↻  Refresh data", lambda: (window.destroy(), callback()),
            self.colors["panel"], self.colors["text"], self.colors["stroke"],
        ).pack(side="right")

    def _make_table(self, parent, columns, headings, widths, height=10):
        c = self.colors
        frame = tk.Frame(parent, bg=c["panel"], highlightbackground=c["stroke"], highlightthickness=1)
        frame.pack(fill="both", expand=True)
        tree = ttk.Treeview(frame, columns=columns, show="headings", height=height, style="Dark.Treeview")
        for column, heading, width in zip(columns, headings, widths):
            tree.heading(column, text=heading)
            tree.column(column, width=width, anchor="w", stretch=True)
        scroll_y = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        scroll_x = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        tree.grid(row=0, column=0, sticky="nsew")
        scroll_y.grid(row=0, column=1, sticky="ns")
        scroll_x.grid(row=1, column=0, sticky="ew")
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        return tree

    def _open_visitors(self):
        c = self.colors
        window = self._popup("Visitors", "1080x720")
        body = tk.Frame(window, bg=c["page"])
        body.pack(fill="both", expand=True, padx=24, pady=22)
        self._window_header(body, "Visitor directory", "Registered identities and their recorded entry/exit history")
        self._popup_refresh(body, window, self._open_visitors)
        search_row = tk.Frame(body, bg=c["page"])
        search_row.pack(fill="x", pady=(0, 10))
        tk.Label(search_row, text="SEARCH VISITORS", bg=c["page"], fg=c["muted"],
                 font=("Segoe UI", 8, "bold")).pack(side="left", padx=(0, 10))
        search_var = tk.StringVar()
        search_entry = tk.Entry(search_row, textvariable=search_var, bg=c["panel2"], fg=c["text"],
                                insertbackground=c["accent"], relief="flat", font=("Segoe UI", 9),
                                highlightthickness=1, highlightbackground=c["stroke"])
        search_entry.pack(side="left", fill="x", expand=True, ipady=6)
        faces, events = self._read_all_data()
        by_face = {}
        for event in events:
            by_face.setdefault(event[3], []).append(event)
        tree = self._make_table(
            body, ("face", "first", "entries", "exits", "last"),
            ("FACE ID", "FIRST SEEN", "ENTRIES", "EXITS", "LAST EVENT"),
            (130, 220, 100, 100, 220), height=11,
        )
        empty_var = tk.StringVar(value="")
        empty = tk.Label(body, textvariable=empty_var, bg=c["panel"], fg=c["muted"],
                         font=("Segoe UI", 10))
        def refill_visitors(*_args):
            query = search_var.get().strip().lower()
            for item in tree.get_children(""):
                tree.delete(item)
            for face_id, label, first_seen in faces:
                records = by_face.get(face_id, [])
                entry_n = sum(event[2].lower() == "entry" for event in records)
                exit_n = sum(event[2].lower() == "exit" for event in records)
                last_seen = records[0][1] if records else "No entry/exit event yet"
                face_label = label or f"FACE_{face_id:04d}"
                searchable = f"{face_label} {first_seen} {last_seen}".lower()
                if query and query not in searchable:
                    continue
                tree.insert("", "end", iid=str(face_id),
                            values=(face_label, first_seen, entry_n, exit_n, last_seen))
            if not tree.get_children(""):
                empty_var.set(
                    "No visitors are stored yet. Run a video first; new identities will appear here."
                    if not faces else "No visitor matches that search. Try a FACE ID or first-seen timestamp."
                )
                if not empty.winfo_manager():
                    empty.pack(fill="x", pady=(10, 0))
            elif empty.winfo_manager():
                empty.pack_forget()

        search_var.trace_add("write", refill_visitors)
        refill_visitors()
        details_var = tk.StringVar(value="Select a visitor above to view the exact event times and saved images.")
        tk.Label(body, textvariable=details_var, bg=c["page"], fg=c["accent"],
                 font=("Segoe UI", 9), anchor="w").pack(fill="x", pady=(12, 7))
        timeline = self._make_table(
            body, ("time", "event", "track", "frame", "image"),
            ("TIMESTAMP", "EVENT", "TRACK", "FRAME", "SAVED PHOTO PATH"),
            (220, 90, 80, 80, 440), height=6,
        )

        def show_timeline(_event=None):
            selected = tree.selection()
            if not selected:
                return
            face_id = int(selected[0])
            label = f"FACE_{face_id:04d}"
            records = [row for row in events if row[3] == face_id]
            details_var.set(f"{label}  ·  {len(records)} event(s). Double-click a photo row to open its saved crop.")
            for item in timeline.get_children(""):
                timeline.delete(item)
            for event_id, timestamp, kind, _fid, track_id, frame_idx, image_path in sorted(records, key=lambda row: row[1]):
                timeline.insert("", "end", iid=f"event-{event_id}", values=(timestamp, kind.upper(), track_id, frame_idx, image_path or ""))

        def open_timeline_photo(_event=None):
            selection = timeline.selection()
            if not selection:
                return
            path_text = timeline.item(selection[0], "values")[-1]
            path = Path(path_text)
            if path_text and path.exists():
                os.startfile(path)
            elif path_text:
                messagebox.showerror("Photo missing", f"Saved photo not found:\n{path}", parent=window)

        tree.bind("<<TreeviewSelect>>", show_timeline)
        timeline.bind("<Double-1>", open_timeline_photo)

    def _open_photo_gallery(self):
        c = self.colors
        window = self._popup("Photo gallery", "1180x760")
        body = tk.Frame(window, bg=c["page"])
        body.pack(fill="both", expand=True, padx=24, pady=22)
        self._window_header(body, "Entry & exit photo gallery", "Choose a saved event to see its visitor, timestamp and original crop")
        self._popup_refresh(body, window, self._open_photo_gallery)
        faces, events = self._read_all_data()
        filter_row = tk.Frame(body, bg=c["page"])
        filter_row.pack(fill="x", pady=(0, 10))
        tk.Label(filter_row, text="SHOW", bg=c["page"], fg=c["muted"], font=("Segoe UI", 9, "bold")).pack(side="left", padx=(0, 10))
        filter_var = tk.StringVar(value="All event photos")
        picker = ttk.Combobox(filter_row, textvariable=filter_var,
                              values=("All event photos", "Entry photos", "Exit photos"),
                              state="readonly", width=20, style="Dark.TCombobox")
        picker.pack(side="left")
        content = tk.Frame(body, bg=c["page"])
        content.pack(fill="both", expand=True)
        listing = tk.Frame(content, bg=c["page"])
        listing.pack(side="left", fill="both", expand=True, padx=(0, 14))
        gallery = self._make_table(
            listing, ("id", "kind", "face", "time", "track", "path"),
            ("#", "EVENT", "FACE ID", "TIMESTAMP", "TRACK", "SAVED PHOTO"),
            (55, 90, 100, 205, 70, 300), height=16,
        )
        gallery.tag_configure("ENTRY", foreground=c["warning"])
        gallery.tag_configure("EXIT", foreground=c["green"])
        right = tk.Frame(content, bg=c["panel"], highlightbackground=c["stroke"], highlightthickness=1, width=340)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)
        photo_preview = tk.Label(right, text="Select an entry or exit row\nto preview its crop", bg=c["panel2"],
                                 fg=c["muted"], font=("Segoe UI", 10), width=36, height=14)
        photo_preview.pack(fill="both", expand=True, padx=14, pady=(14, 8))
        detail_var = tk.StringVar(value="No event selected")
        tk.Label(right, textvariable=detail_var, bg=c["panel"], fg=c["text"], font=("Segoe UI", 9),
                 justify="left", wraplength=300).pack(fill="x", padx=14, pady=8)
        open_button = self._action_button(right, "Open original photo", lambda: open_selected(),
                                          c["accent"], "#1B1028", "#D3A2FF", bold=True)
        open_button.pack(fill="x", padx=14, pady=(2, 14))

        def refill(_event=None):
            for item in gallery.get_children(""):
                gallery.delete(item)
            selection = filter_var.get()
            kept = [row for row in events if selection == "All event photos"
                    or (selection == "Entry photos" and row[2].lower() == "entry")
                    or (selection == "Exit photos" and row[2].lower() == "exit")]
            for event_id, timestamp, kind, face_id, track_id, frame_idx, image_path in kept:
                image_name = Path(image_path).name if image_path else "Image path unavailable"
                gallery.insert("", "end", iid=str(event_id),
                               values=(event_id, kind.upper(), f"FACE_{face_id:04d}", timestamp, track_id, image_name),
                               tags=(kind.upper(),))
            if not kept:
                detail_var.set("No saved entry/exit photos are in the database yet. Run a video; saved evidence will appear here.")
                photo_preview.configure(image="", text="No saved photos yet")
                photo_preview.image = None
            else:
                detail_var.set(f"Showing {len(kept)} saved photo event(s). Select a row to preview details.")

        def show_preview(_event=None):
            selected = gallery.selection()
            if not selected:
                return
            event_id = int(selected[0])
            row = next((item for item in events if item[0] == event_id), None)
            if row is None:
                return
            _eid, timestamp, kind, face_id, track_id, frame_idx, image_path = row
            path = Path(image_path) if image_path else None
            exists = bool(path and path.is_file())
            detail_var.set(
                f"{kind.upper()}  ·  FACE_{face_id:04d}\nTimestamp: {timestamp}\nTrack: {track_id}  ·  Frame: {frame_idx}\n"
                f"Photo: {path if path else 'not recorded'}\nStatus: {'saved' if exists else 'file unavailable'}"
            )
            if exists:
                try:
                    image = tk.PhotoImage(file=str(path))
                    factor = max(1, (image.width() + 309) // 310, (image.height() + 290) // 290)
                    if factor > 1:
                        image = image.subsample(factor, factor)
                    photo_preview.configure(image=image, text="")
                    photo_preview.image = image
                except tk.TclError:
                    photo_preview.configure(image="", text="Preview unavailable\nUse Open original photo")
                    photo_preview.image = None
            else:
                photo_preview.configure(image="", text="Saved image file is unavailable")
                photo_preview.image = None

        def open_selected():
            selected = gallery.selection()
            if not selected:
                messagebox.showinfo("Choose a photo", "Select an entry or exit row in the gallery first.", parent=window)
                return
            event_id = int(selected[0])
            row = next((item for item in events if item[0] == event_id), None)
            path = Path(row[6]) if row and row[6] else None
            if path and path.is_file():
                os.startfile(path)
            else:
                messagebox.showerror("Photo unavailable", f"The saved crop could not be found:\n{path or 'No image path recorded'}", parent=window)

        picker.bind("<<ComboboxSelected>>", refill)
        gallery.bind("<<TreeviewSelect>>", show_preview)
        gallery.bind("<Double-1>", lambda _event: open_selected())
        refill()

    def _animate_status(self):
        active = self.process is not None and self.process.poll() is None
        c = self.colors
        if active:
            self.pulse_phase = not self.pulse_phase
            fill = c["accent"] if self.pulse_phase else "#FF72B6"
        elif self.status_var.get() == "Completed successfully":
            fill = c["green"]
        elif "error" in self.status_var.get().lower():
            fill = c["danger"]
        else:
            fill = c["muted"]
        self.status_dot.itemconfigure(self.status_dot_id, fill=fill)
        self.root.after(450, self._animate_status)

    def _mode_changed(self, _event=None):
        is_video = self.mode_var.get() == "video"
        self.browse_button.configure(state="normal" if is_video else "disabled")
        if is_video and self.source_var.get().startswith("rtsp://"):
            self.source_var.set(self._default_video())
        elif not is_video and not self.source_var.get().startswith("rtsp://"):
            self.source_var.set("rtsp://username:password@camera-ip:554/stream")

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select a video", initialdir=str(PROJECT_DIR / "data"),
            filetypes=(("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.m4v"), ("All files", "*.*")),
        )
        if path:
            self.source_var.set(path)

    def _start(self):
        if self.process and self.process.poll() is None:
            return
        source = self.source_var.get().strip()
        if not source:
            messagebox.showwarning("Input required", "Choose a video or enter an RTSP URL.")
            return
        if self.mode_var.get() == "video" and not Path(source).is_file():
            messagebox.showerror("Video not found", f"The selected file does not exist:\n{source}")
            return
        log_path = self.config.get("storage", {}).get("logs_dir", "logs")
        log_dir = Path(log_path)
        if not log_dir.is_absolute():
            log_dir = PROJECT_DIR / log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        console_path = log_dir / "ui_console.log"
        self.stop_path = log_dir / "ui_stop.request"
        try:
            self.stop_path.unlink(missing_ok=True)
        except OSError as exc:
            messagebox.showerror("Could not prepare run", str(exc))
            return
        source_stem = "rtsp_stream" if self.mode_var.get() == "rtsp" else Path(source).stem
        source_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", source_stem).strip("._") or "video"
        self.active_source_name = source_stem
        output_dir = PROJECT_DIR / "output" / "annotated"
        self.last_output_path = output_dir / f"{source_stem}_annotated.mp4"
        command = [sys.executable, str(PROJECT_DIR / "main.py"), "--config", str(self.config_path), "--mode", self.mode_var.get(), "--source", source]
        command.extend(["--stop-file", str(self.stop_path)])
        command.extend(["--annotated-output", str(self.last_output_path)])
        if self.preserve_var.get():
            command.append("--preserve-data")
        try:
            self.console_file = console_path.open("w", encoding="utf-8", errors="replace")
            self.console_file.write(f"\n\n--- UI run started {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            self.console_file.flush()
            self.process = subprocess.Popen(command, cwd=PROJECT_DIR, stdout=self.console_file, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError as exc:
            if self.console_file:
                self.console_file.close()
                self.console_file = None
            messagebox.showerror("Could not start tracker", str(exc))
            return
        self.started_at = time.monotonic()
        self.run_started_at = self.started_at
        self.database_poll_after = self.started_at + 2.0
        self.last_frame, self.last_fps = 0, None
        self.frames_var.set("0")
        self.elapsed_var.set("00:00")
        self.count_var.set("…" if self.preserve_var.get() else "0")
        self.fps_var.set("Preparing the run…")
        self.health_var.set("●  Starting a clean run" if not self.preserve_var.get() else "●  Loading previous visitor gallery")
        self.detection_var.set("Waiting for the first detector pass")
        self.database_health_var.set("Opening SQLite for this run…")
        self.latest_events_var.set("Waiting for the first confirmed entry/exit event")
        for item in self.events.get_children(""):
            self.events.delete(item)
        self.events_empty_label.place(relx=0.5, rely=0.55, anchor="center")
        _, event_log_path = self._paths()
        try:
            self.stats_log_offset = event_log_path.stat().st_size
        except OSError:
            self.stats_log_offset = 0
        self.status_var.set("Starting…")
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.activity_bar.start(14)
        self.health_var.set("●  Initializing detector and face models")

    def _stop(self):
        if self.process and self.process.poll() is None:
            self.status_var.set("Stopping safely…")
            try:
                self.stop_path.write_text("stop\n", encoding="utf-8")
            except OSError as exc:
                messagebox.showerror("Could not request stop", str(exc))

    def _paths(self):
        storage = self.config.get("storage", {})
        def resolve(value):
            path = Path(value)
            return path if path.is_absolute() else PROJECT_DIR / path
        return resolve(storage.get("db_path", "data/visitors.db")), resolve(storage.get("log_file", "logs/events.log"))

    def _refresh(self):
        running = self.process is not None and self.process.poll() is None
        if running and self.started_at is not None:
            seconds = int(time.monotonic() - self.started_at)
            self.elapsed_var.set(f"{seconds // 60:02d}:{seconds % 60:02d}")
            self.status_var.set(
                "Stopping safely…" if self.stop_path and self.stop_path.exists()
                else "Processing video / stream…"
            )
            self.health_var.set("●  Graceful shutdown requested" if self.stop_path and self.stop_path.exists() else "●  Detection and recognition active")
        elif self.process is not None:
            code = self.process.poll()
            self.status_var.set("Completed successfully" if code == 0 else f"Stopped / error (exit code {code})")
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            self.activity_bar.stop()
            self.health_var.set("●  Run complete" if code == 0 else "●  Check the activity console for details")
            self.process = None
            if self.console_file:
                self.console_file.close()
                self.console_file = None
            if self.stop_path:
                self.stop_path.unlink(missing_ok=True)
            if self.close_when_stopped:
                self.root.destroy()
                return
        if self.run_started_at is not None:
            run_finished = self.process is None
            if time.monotonic() >= self.database_poll_after or run_finished:
                self._read_database()
                self._read_progress()
            self._read_console()
        self.root.after(1200, self._refresh)

    def _read_database(self):
        db_path, _ = self._paths()
        if not db_path.exists():
            self.count_var.set("0")
            self.database_health_var.set("Database not created yet · start a run")
            return
        try:
            conn = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True, timeout=0.25)
            try:
                count = conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
                registrations = conn.execute("SELECT first_seen FROM faces ORDER BY face_id").fetchall()
                source = (self.active_source_name or "").lower()
                slash_pattern = f"%/{source}/%"
                backslash_pattern = f"%\\{source}\\%"
                if source:
                    event_counts = dict(conn.execute(
                        "SELECT event_type, COUNT(*) FROM events "
                        "WHERE lower(COALESCE(image_path,'')) LIKE ? "
                        "OR lower(COALESCE(image_path,'')) LIKE ? GROUP BY event_type",
                        (slash_pattern, backslash_pattern),
                    ).fetchall())
                    total_events = sum(event_counts.values())
                else:
                    event_counts = dict(conn.execute(
                        "SELECT event_type, COUNT(*) FROM events GROUP BY event_type"
                    ).fetchall())
                    total_events = sum(event_counts.values())
                rows = conn.execute(
                    "SELECT event_id, timestamp, event_type, face_id, track_id, image_path "
                    "FROM events WHERE lower(image_path) LIKE ? OR lower(image_path) LIKE ? "
                    "ORDER BY event_id DESC LIMIT 12",
                    (slash_pattern, backslash_pattern),
                ).fetchall()
            finally:
                conn.close()
            self.count_var.set(str(count))
            self.entry_count_var.set(str(event_counts.get("entry", 0)))
            self.exit_count_var.set(str(event_counts.get("exit", 0)))
            self.database_health_var.set(f"Connected · {count} visitor(s) · {total_events} event(s)")
            self.latest_events_var.set(
                f"Showing {min(total_events, 12)} recent record(s) · "
                f"{event_counts.get('entry', 0)} entries · {event_counts.get('exit', 0)} exits"
                if total_events else "No saved events for this video yet · confirmed entries and exits appear here"
            )
            self.registration_values = [row[0] for row in registrations]
            self._draw_trend()
            desired = [
                tuple(str(value) if value is not None else "" for value in (
                    event_id, timestamp, kind.upper(), f"FACE_{face_id:04d}", track_id, image
                ))
                for event_id, timestamp, kind, face_id, track_id, image in reversed(rows)
            ]
            current = [self.events.item(item, "values") for item in self.events.get_children("")]
            if current != desired:
                selected = self.events.selection()
                selected_event_id = self.events.item(selected[0], "values")[0] if selected else None
                for item in self.events.get_children(""):
                    self.events.delete(item)
                for values in desired:
                    self.events.insert("", "end", values=values, tags=(values[2],))
                if selected_event_id is not None:
                    matches = [item for item in self.events.get_children("") if self.events.item(item, "values")[0] == selected_event_id]
                    if matches:
                        self.events.selection_set(matches)
            if desired:
                self.events_empty_label.place_forget()
            else:
                self.events_empty_label.place(relx=0.5, rely=0.55, anchor="center")
        except (sqlite3.Error, OSError):
            # SQLite can briefly be busy while the processing pipeline commits.
            pass

    def _read_progress(self):
        _, log_path = self._paths()
        if not log_path.exists():
            return
        try:
            with log_path.open("rb") as file:
                file.seek(0, os.SEEK_END)
                end = file.tell()
                if end < self.stats_log_offset:
                    # The user chose a fresh run, so the event log was truncated.
                    self.stats_log_offset = 0
                file.seek(max(self.stats_log_offset, end - 64000))
                lines = file.read().decode("utf-8", errors="replace").splitlines()
                self.stats_log_offset = end
            for line in reversed(lines):
                match = re.search(
                    r"STATS \| frame=(\d+) fps=([\d.]+) detections=(\d+).*unique=(\d+)", line
                )
                if match:
                    self.last_frame = int(match.group(1))
                    self.last_fps = float(match.group(2))
                    self.frames_var.set(str(self.last_frame))
                    self.fps_var.set(f"{self.last_fps:.1f} FPS  ·  live pipeline rate")
                    self.detection_var.set(f"{match.group(3)} face boxes in latest detector pass")
                    break
                legacy = re.search(r"STATS \| frame=(\d+) fps=([\d.]+).*unique=(\d+)", line)
                if legacy:
                    self.last_frame = int(legacy.group(1))
                    self.last_fps = float(legacy.group(2))
                    self.frames_var.set(str(self.last_frame))
                    self.fps_var.set(f"{self.last_fps:.1f} FPS  ·  live pipeline rate")
                    self.detection_var.set("No detection count in this run’s older log format")
                    break
        except OSError:
            pass

    def _read_console(self):
        path = Path(self.config.get("storage", {}).get("logs_dir", "logs"))
        if not path.is_absolute():
            path = PROJECT_DIR / path
        path = path / "ui_console.log"
        if not path.exists():
            return
        try:
            with path.open("rb") as file:
                file.seek(0, os.SEEK_END)
                end = file.tell()
                file.seek(max(0, end - 12000))
                tail = file.read().decode("utf-8", errors="replace").splitlines(keepends=True)[-18:]
            tail = "".join(tail)
            self.console.configure(state="normal")
            self.console.delete("1.0", "end")
            self.console.insert("end", tail)
            self.console.see("end")
            self.console.configure(state="disabled")
        except OSError:
            pass

    def _open_video(self):
        value = self.last_output_path or Path(self.config.get("output", {}).get("annotated_video_path", "output/annotated.mp4"))
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_DIR / path
        if not path.exists():
            messagebox.showinfo("Not ready", "The annotated video appears after the run finishes and closes the video writer.")
            return
        os.startfile(path)

    def _open_selected_image(self):
        selected = self.events.selection()
        if not selected:
            messagebox.showinfo("Select an event", "Select an entry or exit row first.")
            return
        path = Path(self.events.item(selected[0], "values")[-1])
        if not path.is_absolute():
            path = PROJECT_DIR / path
        if path.exists():
            os.startfile(path)
        else:
            messagebox.showerror("Image unavailable", f"Image file not found:\n{path}")

    def _open_report(self):
        c = self.colors
        window = self._popup("Reports", "1160x760")
        body = tk.Frame(window, bg=c["page"])
        body.pack(fill="both", expand=True, padx=24, pady=22)
        self._window_header(body, "Visitor report", "Structured visitor totals, entry/exit times and database locations")
        self._popup_refresh(body, window, self._open_report)
        faces, events = self._read_all_data()
        entries = sum(row[2].lower() == "entry" for row in events)
        exits = sum(row[2].lower() == "exit" for row in events)
        summary = tk.Frame(body, bg=c["page"])
        summary.pack(fill="x", pady=(0, 12))
        for index, (title, value, color) in enumerate((
            ("UNIQUE VISITORS", len(faces), c["accent"]),
            ("ENTRY EVENTS", entries, c["warning"]),
            ("EXIT EVENTS", exits, c["green"]),
        )):
            card = tk.Frame(summary, bg=c["panel"], highlightbackground=color, highlightthickness=1)
            card.pack(side="left", fill="x", expand=True, padx=(0 if index == 0 else 10, 0))
            tk.Label(card, text=title, bg=c["panel"], fg=c["muted"], font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=14, pady=(11, 0))
            tk.Label(card, text=str(value), bg=c["panel"], fg=color, font=("Segoe UI", 22, "bold")).pack(anchor="w", padx=14, pady=(0, 10))
        db_path, log_path = self._paths()
        tk.Label(body, text=f"Database: {db_path}     ·     Event log: {log_path}", bg=c["page"],
                 fg=c["muted"], font=("Segoe UI", 8), anchor="w").pack(fill="x", pady=(0, 12))
        by_face = {}
        for event in events:
            by_face.setdefault(event[3], []).append(event)
        report_actions = tk.Frame(body, bg=c["page"])
        report_actions.pack(fill="x", pady=(0, 10))

        def export_csv():
            output_path = filedialog.asksaveasfilename(
                parent=window, title="Save visitor report", defaultextension=".csv",
                initialfile="face_sync_visitor_report.csv",
                filetypes=(("CSV spreadsheet", "*.csv"), ("All files", "*.*")),
            )
            if not output_path:
                return
            try:
                with open(output_path, "w", newline="", encoding="utf-8-sig") as output_file:
                    writer = csv.writer(output_file)
                    writer.writerow(("Face ID", "First seen", "Entry timestamps", "Exit timestamps"))
                    for face_id, label, first_seen in faces:
                        records = sorted(by_face.get(face_id, []), key=lambda row: (row[1], row[0]))
                        entry_times = [row[1] for row in records if row[2].lower() == "entry"]
                        exit_times = [row[1] for row in records if row[2].lower() == "exit"]
                        writer.writerow((label or f"FACE_{face_id:04d}", first_seen,
                                         "; ".join(entry_times), "; ".join(exit_times)))
                messagebox.showinfo("Report exported", f"CSV report saved to:\n{output_path}", parent=window)
            except OSError as exc:
                messagebox.showerror("Export failed", str(exc), parent=window)

        self._action_button(report_actions, "Export report as CSV", export_csv,
                            c["warning"], "#241226", "#FF91BF", bold=True).pack(side="left")
        self._action_button(
            report_actions, "Show database in folder",
            lambda: subprocess.Popen(["explorer.exe", "/select,", str(db_path)]),
            c["panel"], c["text"], c["stroke"],
        ).pack(side="left", padx=10)
        report = self._make_table(
            body, ("face", "first", "entries", "exits", "last"),
            ("FACE ID", "FIRST SEEN", "ENTRY COUNT", "EXIT COUNT", "LAST EVENT"),
            (125, 220, 110, 110, 230), height=11,
        )
        for face_id, label, first_seen in faces:
            records = by_face.get(face_id, [])
            entry_n = sum(item[2].lower() == "entry" for item in records)
            exit_n = sum(item[2].lower() == "exit" for item in records)
            last_seen = records[0][1] if records else "No event recorded"
            report.insert("", "end", iid=str(face_id),
                          values=(label or f"FACE_{face_id:04d}", first_seen, entry_n, exit_n, last_seen))
        empty_var = tk.StringVar(value=(
            "No report rows yet. Start tracking a video; each confirmed face and saved event will appear here."
            if not faces else "Select a visitor to see every entry and exit timestamp below."
        ))
        tk.Label(body, textvariable=empty_var, bg=c["page"], fg=c["accent"],
                 font=("Segoe UI", 9), anchor="w").pack(fill="x", pady=(10, 6))
        timeline = self._make_table(
            body, ("time", "event", "track", "frame", "photo"),
            ("TIMESTAMP", "EVENT", "TRACK", "FRAME", "SAVED IMAGE PATH"),
            (220, 90, 80, 80, 440), height=6,
        )

        def show_report_detail(_event=None):
            selected = report.selection()
            if not selected:
                return
            face_id = int(selected[0])
            records = sorted(by_face.get(face_id, []), key=lambda item: (item[1], item[0]))
            empty_var.set(f"FACE_{face_id:04d}  ·  {len(records)} recorded entry/exit event(s). Double-click a row to open its evidence image.")
            for item in timeline.get_children(""):
                timeline.delete(item)
            for event_id, timestamp, kind, _fid, track_id, frame_idx, image_path in records:
                timeline.insert("", "end", iid=f"report-event-{event_id}",
                                values=(timestamp, kind.upper(), track_id, frame_idx, image_path or ""))

        def open_report_image(_event=None):
            selected = timeline.selection()
            if not selected:
                return
            path_text = timeline.item(selected[0], "values")[-1]
            path = Path(path_text)
            if path_text and path.is_file():
                os.startfile(path)
            elif path_text:
                messagebox.showerror("Image missing", f"Saved image not found:\n{path_text}", parent=window)

        report.bind("<<TreeviewSelect>>", show_report_detail)
        timeline.bind("<Double-1>", open_report_image)

    def _on_close(self):
        if self.process and self.process.poll() is None:
            if not messagebox.askyesno("Tracker is running", "Stop the tracker safely and close the dashboard?"):
                return
            self.close_when_stopped = True
            self._stop()
            self.status_var.set("Stopping safely; dashboard will close when cleanup finishes…")
            return
        self.root.destroy()


def main():
    root = tk.Tk()
    TrackerDashboard(root)
    root.mainloop()


if __name__ == "__main__":
    main()
