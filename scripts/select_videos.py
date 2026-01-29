#!/usr/bin/env python3
"""
Interactive TUI for selecting WhisperX videos for speaker labeling.

Usage: python scripts/select_whisperx_videos.py

Controls:
  ↑/↓ or j/k  Navigate rows
  Space       Toggle selection
  x           Mark as excluded (explainer/non-call video)
  Enter       Open video in browser
  /           Focus filter
  Escape      Clear filter
  s           Save selections and exclusions
  q           Quit
"""

import csv
import webbrowser
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Footer, Header, Input, Static
from textual.coordinate import Coordinate

# Paths
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RANKINGS_PATH = DATA_DIR / "whisperx_video_rankings.csv"
SELECTED_PATH = DATA_DIR / "selected_videos.txt"
EXCLUDED_PATH = DATA_DIR / "excluded_videos.txt"
TARGET_COUNT = 50


class VideoSelector(App):
    """TUI app for selecting videos."""

    CSS = """
    #filter-container {
        height: 3;
        padding: 0 1;
    }
    #filter-input {
        width: 50;
    }
    #status {
        text-align: right;
        width: 100%;
        padding-right: 2;
    }
    #counter {
        color: $success;
        text-style: bold;
    }
    DataTable {
        height: 1fr;
    }
    .warning {
        color: $warning;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("s", "save", "Save"),
        Binding("x", "exclude", "Exclude"),
        Binding("enter", "open_video", "Open"),
        Binding("/", "focus_filter", "Filter"),
        Binding("escape", "clear_filter", "Clear", show=False),
        Binding("space", "toggle_select", "Toggle", show=False),
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
    ]

    def __init__(self):
        super().__init__()
        self.videos: list[dict] = []
        self.selected: set[str] = set()  # video IDs
        self.excluded: set[str] = set()  # video IDs marked as non-call/explainer
        self.filtered_indices: list[int] = []  # indices into self.videos
        self.current_filter = ""

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            with Horizontal(id="filter-container"):
                yield Input(placeholder="Type to filter titles...", id="filter-input")
                yield Static("", id="status")
            yield DataTable(id="video-table", cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "WhisperX Video Selector"
        self.load_videos()
        self.load_existing_selections()
        self.load_existing_exclusions()
        self.setup_table()
        self.apply_filter("")
        self.update_counter()

    def load_videos(self) -> None:
        """Load videos from rankings CSV."""
        if not RANKINGS_PATH.exists():
            self.notify(f"Rankings not found: {RANKINGS_PATH}", severity="error")
            return

        with open(RANKINGS_PATH) as f:
            reader = csv.DictReader(f)
            self.videos = list(reader)

    def load_existing_selections(self) -> None:
        """Load previously saved selections."""
        if SELECTED_PATH.exists():
            with open(SELECTED_PATH) as f:
                self.selected = {line.strip() for line in f if line.strip()}
            if self.selected:
                self.notify(f"Loaded {len(self.selected)} previous selections")

    def load_existing_exclusions(self) -> None:
        """Load previously saved exclusions."""
        if EXCLUDED_PATH.exists():
            with open(EXCLUDED_PATH) as f:
                self.excluded = {line.strip() for line in f if line.strip()}
            if self.excluded:
                self.notify(f"Loaded {len(self.excluded)} previous exclusions")

    def setup_table(self) -> None:
        """Configure the data table columns."""
        table = self.query_one("#video-table", DataTable)
        table.add_column("✓", width=3, key="selected")
        table.add_column("#", width=4, key="rank")
        table.add_column("Title", width=50, key="title")
        table.add_column("Channel", width=20, key="channel")
        table.add_column("Views", width=8, key="views")
        table.add_column("Dur", width=5, key="duration")
        table.add_column("Score", width=6, key="score")

    def apply_filter(self, filter_text: str) -> None:
        """Filter videos and refresh table."""
        self.current_filter = filter_text.lower()
        table = self.query_one("#video-table", DataTable)
        table.clear()

        self.filtered_indices = []
        for i, video in enumerate(self.videos):
            title = video.get("title", "").lower()
            channel = video.get("channel", "").lower()
            if self.current_filter in title or self.current_filter in channel:
                self.filtered_indices.append(i)
                self.add_video_row(table, video)

    def add_video_row(self, table: DataTable, video: dict) -> None:
        """Add a single video row to the table."""
        vid = video.get("video_id", "")
        is_selected = vid in self.selected
        is_excluded = vid in self.excluded
        # ☑ = selected, ✗ = excluded, ☐ = neither
        check = "✗" if is_excluded else ("☑" if is_selected else "☐")

        views = int(video.get("view_count", 0))
        if views >= 1_000_000:
            views_str = f"{views / 1_000_000:.1f}M"
        elif views >= 1_000:
            views_str = f"{views / 1_000:.0f}K"
        else:
            views_str = str(views)

        table.add_row(
            check,
            video.get("rank", ""),
            video.get("title", "")[:48],
            video.get("channel", "")[:18],
            views_str,
            f"{float(video.get('duration_min', 0)):.0f}m",
            video.get("total_score", ""),
            key=vid,
        )

    def update_counter(self) -> None:
        """Update the selection counter display."""
        count = len(self.selected)
        excluded_count = len(self.excluded)
        status = self.query_one("#status", Static)

        excluded_text = f" | [dim]Excluded: {excluded_count}[/]" if excluded_count else ""

        if count > TARGET_COUNT:
            status.update(f"[bold red]Selected: {count}/{TARGET_COUNT} (over limit!)[/]{excluded_text}")
        elif count == TARGET_COUNT:
            status.update(f"[bold green]Selected: {count}/{TARGET_COUNT} ✓[/]{excluded_text}")
        else:
            status.update(f"Selected: [bold]{count}[/]/{TARGET_COUNT}{excluded_text}")

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle filter input changes."""
        if event.input.id == "filter-input":
            self.apply_filter(event.value)

    def action_focus_filter(self) -> None:
        """Focus the filter input."""
        self.query_one("#filter-input", Input).focus()

    def action_clear_filter(self) -> None:
        """Clear filter and return to table."""
        filter_input = self.query_one("#filter-input", Input)
        filter_input.value = ""
        self.apply_filter("")
        self.query_one("#video-table", DataTable).focus()

    def action_toggle_select(self) -> None:
        """Toggle selection on current row."""
        table = self.query_one("#video-table", DataTable)
        if table.cursor_row is None or table.cursor_row >= len(self.filtered_indices):
            return

        # Get video ID from current row key
        try:
            keys = list(table.rows.keys())
            if table.cursor_row >= len(keys):
                return
            vid = str(keys[table.cursor_row].value)
        except Exception:
            return

        # If excluded, pressing space removes exclusion and selects
        if vid in self.excluded:
            self.excluded.discard(vid)
            self.selected.add(vid)
            new_check = "☑"
            if len(self.selected) > TARGET_COUNT:
                self.notify(f"Over {TARGET_COUNT} limit!", severity="warning")
        elif vid in self.selected:
            self.selected.discard(vid)
            new_check = "☐"
        else:
            self.selected.add(vid)
            new_check = "☑"
            if len(self.selected) > TARGET_COUNT:
                self.notify(f"Over {TARGET_COUNT} limit!", severity="warning")

        # Update the checkbox cell
        table.update_cell_at(Coordinate(table.cursor_row, 0), new_check)
        self.update_counter()

    def action_exclude(self) -> None:
        """Mark current video as excluded (explainer/non-call video)."""
        table = self.query_one("#video-table", DataTable)
        if table.cursor_row is None or table.cursor_row >= len(self.filtered_indices):
            return

        # Get video ID from current row key
        try:
            keys = list(table.rows.keys())
            if table.cursor_row >= len(keys):
                return
            vid = str(keys[table.cursor_row].value)
        except Exception:
            return

        # If already excluded, pressing x removes exclusion
        if vid in self.excluded:
            self.excluded.discard(vid)
            new_check = "☐"
        else:
            # Remove from selected if present, then exclude
            self.selected.discard(vid)
            self.excluded.add(vid)
            new_check = "✗"

        # Update the checkbox cell
        table.update_cell_at(Coordinate(table.cursor_row, 0), new_check)
        self.update_counter()

    def action_cursor_down(self) -> None:
        """Move cursor down (vim style)."""
        table = self.query_one("#video-table", DataTable)
        table.action_cursor_down()

    def action_cursor_up(self) -> None:
        """Move cursor up (vim style)."""
        table = self.query_one("#video-table", DataTable)
        table.action_cursor_up()

    def get_current_video_id(self) -> str | None:
        """Get video ID of currently selected row."""
        table = self.query_one("#video-table", DataTable)
        if table.cursor_row is None:
            return None
        try:
            keys = list(table.rows.keys())
            if table.cursor_row >= len(keys):
                return None
            return str(keys[table.cursor_row].value)
        except Exception:
            return None

    def action_open_video(self) -> None:
        """Open current video in browser."""
        vid = self.get_current_video_id()
        if vid:
            url = f"https://www.youtube.com/watch?v={vid}"
            webbrowser.open(url)
            self.notify(f"Opening {vid} in browser")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row click - open video in browser."""
        if event.row_key:
            vid = str(event.row_key.value)
            url = f"https://www.youtube.com/watch?v={vid}"
            webbrowser.open(url)
            self.notify(f"Opening {vid} in browser")

    def action_save(self) -> None:
        """Save selected and excluded video IDs to files."""
        saved_parts = []

        if self.selected:
            with open(SELECTED_PATH, "w") as f:
                for vid in sorted(self.selected):
                    f.write(f"{vid}\n")
            saved_parts.append(f"{len(self.selected)} selected")

        if self.excluded:
            with open(EXCLUDED_PATH, "w") as f:
                for vid in sorted(self.excluded):
                    f.write(f"{vid}\n")
            saved_parts.append(f"{len(self.excluded)} excluded")

        if saved_parts:
            self.notify(f"Saved {' and '.join(saved_parts)}")
        else:
            self.notify("Nothing to save!", severity="warning")

    def action_quit(self) -> None:
        """Quit the app."""
        self.exit()


def main():
    app = VideoSelector()
    app.run()


if __name__ == "__main__":
    main()
