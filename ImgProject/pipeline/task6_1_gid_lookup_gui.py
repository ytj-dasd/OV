from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from tkinter import BOTH, END, LEFT, RIGHT, TOP, X, Button, Entry, Frame, Label, StringVar, TclError, Tk
from tkinter import messagebox
from tkinter.scrolledtext import ScrolledText

PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from task6_1_gid_lookup import build_gid_index, load_front_records, query_by_global_gid


BG_ROOT = "#EAF0F7"
BG_HEADER = "#1F3B57"
BG_CARD = "#F7FAFF"
BG_PANEL = "#EDF4FD"
TEXT_PRIMARY = "#1D2A3A"
TEXT_MUTED = "#5A6B80"
ACCENT = "#2D7FF9"
ACCENT_HOVER = "#4D97FF"
BORDER = "#C9D8EB"
FONT_FAMILY = "Times New Roman"
FONT_TITLE = (FONT_FAMILY, 20, "bold")
FONT_SUBTITLE = (FONT_FAMILY, 12)
FONT_LABEL_BOLD = (FONT_FAMILY, 14, "bold")
FONT_LABEL = (FONT_FAMILY, 12)
FONT_ENTRY = (FONT_FAMILY, 15)
FONT_BUTTON = (FONT_FAMILY, 12, "bold")
FONT_RESULT = (FONT_FAMILY, 13)
FONT_STATUS = (FONT_FAMILY, 11)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task6.1 GUI: query Task6 attributes by global_gid.")
    parser.add_argument(
        "--front-attrs",
        required=True,
        help="Path to Task6 merged front attrs (.json or .npz).",
    )
    return parser.parse_args()


def _apply_window_alpha(root: Tk, alpha: float) -> None:
    try:
        root.attributes("-alpha", alpha)
    except TclError:
        pass


def _make_button(
    parent: Frame,
    *,
    text: str,
    command,
    width: int,
    bg: str,
    hover_bg: str,
    fg: str = "#FFFFFF",
) -> Button:
    btn = Button(
        parent,
        text=text,
        command=command,
        width=width,
        bg=bg,
        fg=fg,
        activebackground=hover_bg,
        activeforeground=fg,
        relief="flat",
        bd=0,
        padx=10,
        pady=6,
        font=FONT_BUTTON,
        cursor="hand2",
    )
    btn.bind("<Enter>", lambda _event: btn.configure(bg=hover_bg))
    btn.bind("<Leave>", lambda _event: btn.configure(bg=bg))
    return btn


def main() -> None:
    args = parse_args()
    front_attrs_path = Path(args.front_attrs).expanduser().resolve()
    records = load_front_records(front_attrs_path)
    gid_index = build_gid_index(records)

    root = Tk()
    root.title("Task6.1 | global_gid Lookup")
    root.geometry("1120x800")
    root.minsize(1024, 720)
    root.configure(bg=BG_ROOT)
    _apply_window_alpha(root, 0.98)

    header = Frame(root, bg=BG_HEADER)
    header.pack(side=TOP, fill=X)
    Label(
        header,
        text="Task6.1 global_gid Lookup",
        bg=BG_HEADER,
        fg="#F3F8FF",
        font=FONT_TITLE,
        anchor="w",
    ).pack(side=TOP, fill=X, padx=18, pady=(12, 2))
    Label(
        header,
        text="Scene-Level Attributes Retrieval for Scientific Verification",
        bg=BG_HEADER,
        fg="#B8CCE3",
        font=FONT_SUBTITLE,
        anchor="w",
    ).pack(side=TOP, fill=X, padx=18, pady=(0, 12))

    content = Frame(root, bg=BG_ROOT)
    content.pack(side=TOP, fill=BOTH, expand=True, padx=18, pady=14)

    card = Frame(content, bg=BG_CARD, highlightthickness=1, highlightbackground=BORDER, bd=0)
    card.pack(side=TOP, fill=BOTH, expand=True)

    top = Frame(card, bg=BG_CARD)
    top.pack(side=TOP, fill=X, padx=16, pady=(16, 8))

    left_panel = Frame(top, bg=BG_CARD)
    left_panel.pack(side=LEFT, fill=X, expand=True)

    Label(
        left_panel,
        text="global_gid",
        bg=BG_CARD,
        fg=TEXT_PRIMARY,
        font=FONT_LABEL_BOLD,
    ).pack(side=TOP, anchor="w")

    gid_entry = Entry(
        left_panel,
        width=26,
        relief="flat",
        bd=0,
        highlightthickness=2,
        highlightbackground="#B5C9DF",
        highlightcolor=ACCENT,
        bg="#FFFFFF",
        fg=TEXT_PRIMARY,
        font=FONT_ENTRY,
        insertbackground=TEXT_PRIMARY,
    )
    gid_entry.pack(side=TOP, anchor="w", pady=(6, 2), ipady=6)

    Label(
        left_panel,
        text="Press Enter to query; Ctrl+L / Esc to clear",
        bg=BG_CARD,
        fg=TEXT_MUTED,
        font=FONT_LABEL,
    ).pack(side=TOP, anchor="w")

    button_panel = Frame(top, bg=BG_CARD)
    button_panel.pack(side=RIGHT, padx=(12, 0))

    def _default_info_text() -> str:
        return f"records={len(records)} indexed={len(gid_index)}"

    info_var = StringVar(value=_default_info_text())
    info_panel = Frame(card, bg=BG_PANEL, highlightthickness=1, highlightbackground="#D6E3F2")
    info_panel.pack(side=TOP, fill=X, padx=16, pady=(2, 10))
    info_label = Label(
        info_panel,
        textvariable=info_var,
        bg=BG_PANEL,
        fg=TEXT_PRIMARY,
        anchor="w",
        justify=LEFT,
        font=FONT_LABEL,
        padx=12,
        pady=8,
    )
    info_label.pack(side=TOP, fill=X)

    result_panel = Frame(card, bg=BG_CARD)
    result_panel.pack(side=TOP, fill=BOTH, expand=True, padx=16, pady=(0, 16))

    Label(
        result_panel,
        text="Query Result JSON",
        bg=BG_CARD,
        fg=TEXT_PRIMARY,
        font=FONT_LABEL_BOLD,
        anchor="w",
    ).pack(side=TOP, fill=X, pady=(0, 6))

    result_box = ScrolledText(
        result_panel,
        wrap="word",
        relief="flat",
        bd=0,
        highlightthickness=1,
        highlightbackground=BORDER,
        bg="#F9FBFF",
        fg=TEXT_PRIMARY,
        insertbackground=TEXT_PRIMARY,
        font=FONT_RESULT,
        padx=12,
        pady=12,
    )
    result_box.pack(side=TOP, fill=BOTH, expand=True)
    result_box.configure(state="normal")

    status_var = StringVar(value="Ready")
    status_bar = Frame(root, bg=BG_ROOT)
    status_bar.pack(side=TOP, fill=X, padx=18, pady=(0, 10))
    Label(
        status_bar,
        textvariable=status_var,
        bg=BG_ROOT,
        fg=TEXT_MUTED,
        font=FONT_STATUS,
        anchor="w",
    ).pack(side=TOP, fill=X)

    def _set_result(payload: dict) -> None:
        result_box.delete("1.0", END)
        result_box.insert(END, json.dumps(payload, ensure_ascii=False, indent=2))

    def _query() -> None:
        text = gid_entry.get().strip()
        if not text:
            messagebox.showwarning("Notice", "Please enter global_gid.")
            return
        try:
            gid = int(text)
        except Exception:
            messagebox.showerror("Error", f"Invalid global_gid: {text}")
            status_var.set("Input error: global_gid must be an integer.")
            return

        try:
            payload = query_by_global_gid(gid_index=gid_index, global_gid=gid)
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            status_var.set(f"Query failed: {exc}")
            return

        record = payload.get("record")
        if isinstance(record, dict):
            info_var.set(
                f"scene={record.get('scene_name')} "
                f"object_type={record.get('object_type')} "
                f"object_id={record.get('object_id')} "
                f"class_id={record.get('class_id')}"
            )
            status_var.set(f"Query completed: global_gid={gid}")
        else:
            decoded = payload.get("decoded", {})
            info_var.set(
                f"No record matched: scene_id={decoded.get('scene_id')} "
                f"object_type={decoded.get('object_type')} "
                f"object_id={decoded.get('object_id')}"
            )
            status_var.set(f"No match: global_gid={gid}")
        _set_result(payload)

    def _clear() -> None:
        gid_entry.delete(0, END)
        result_box.delete("1.0", END)
        info_var.set(_default_info_text())
        status_var.set("Cleared.")

    def _copy() -> None:
        text = result_box.get("1.0", END).strip()
        if not text:
            return
        root.clipboard_clear()
        root.clipboard_append(text)
        info_var.set("Result copied to clipboard.")
        status_var.set("Result copied to clipboard.")

    query_btn = _make_button(
        button_panel,
        text="Lookup",
        command=_query,
        width=10,
        bg=ACCENT,
        hover_bg=ACCENT_HOVER,
    )
    clear_btn = _make_button(
        button_panel,
        text="Clear",
        command=_clear,
        width=10,
        bg="#7089A3",
        hover_bg="#819AB4",
    )
    copy_btn = _make_button(
        button_panel,
        text="Copy JSON",
        command=_copy,
        width=10,
        bg="#2F5E8C",
        hover_bg="#3E6F9E",
    )
    query_btn.pack(side=LEFT, padx=4)
    clear_btn.pack(side=LEFT, padx=4)
    copy_btn.pack(side=LEFT, padx=4)

    gid_entry.bind("<Return>", lambda _event: _query())
    root.bind("<Control-l>", lambda _event: _clear())
    root.bind("<Escape>", lambda _event: _clear())
    gid_entry.focus_set()

    root.mainloop()


if __name__ == "__main__":
    main()
