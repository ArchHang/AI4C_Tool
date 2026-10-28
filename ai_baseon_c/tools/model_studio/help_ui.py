"""Accessible hover/focus help, without third-party GUI dependencies."""

import tkinter as tk


class HelpTip:
    def __init__(self, widget, text, delay=300):
        self.widget, self.text, self.delay = widget, text, delay
        self.timer = self.window = None
        for event in ("<Enter>", "<FocusIn>"):
            widget.bind(event, self.schedule, add="+")
        widget.bind("<Button-1>", self.show, add="+")
        widget.bind("<Return>", self.show, add="+")
        for event in ("<Leave>", "<FocusOut>", "<Escape>"):
            widget.bind(event, self.hide, add="+")
        widget.bind("<Destroy>", self.hide, add="+")

    def schedule(self, event=None):
        self.hide()
        self.timer = self.widget.after(self.delay, self.show)

    def show(self, event=None):
        self.hide()
        if not self.widget.winfo_exists():
            return
        text = self.text() if callable(self.text) else self.text
        self.window = tk.Toplevel(self.widget)
        self.window.wm_overrideredirect(True)
        self.window.attributes("-topmost", True)
        tk.Label(self.window, text=text, justify="left", wraplength=410, bg="#f4fbfc", fg="#24354b",
                 padx=14, pady=12, relief="solid", borderwidth=1,
                 font=("Microsoft YaHei UI", 10)).pack()
        self.window.update_idletasks()
        x, y = self.widget.winfo_rootx() + 20, self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        x = max(0, min(x, self.widget.winfo_screenwidth() - self.window.winfo_reqwidth() - 6))
        y = max(0, min(y, self.widget.winfo_screenheight() - self.window.winfo_reqheight() - 6))
        self.window.geometry(f"+{x}+{y}")

    def hide(self, event=None):
        if self.timer is not None:
            try:
                self.widget.after_cancel(self.timer)
            except tk.TclError:
                pass  # The application may already be destroying the widget.
            self.timer = None
        if self.window is not None:
            try:
                self.window.destroy()
            except tk.TclError:
                pass
            self.window = None


def help_icon(parent, text, background, foreground="#17818d"):
    icon = tk.Canvas(parent, width=21, height=21, bg=background, highlightthickness=0,
                     cursor="hand2", takefocus=1)
    icon.create_oval(3, 3, 18, 18, outline=foreground, width=1.5)
    icon.create_text(10.5, 10.5, text="!", fill=foreground, font=("Segoe UI", 10, "bold"))
    tip = HelpTip(icon, text)
    return icon, tip
