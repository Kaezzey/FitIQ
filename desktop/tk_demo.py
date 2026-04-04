from __future__ import annotations

import json
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox, ttk
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


@dataclass
class ApiClient:
    base_url: str
    timeout_seconds: float = 10.0

    def _get_json(self, path: str) -> dict[str, object]:
        request = Request(f"{self.base_url}{path}", method="GET")
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} for {path}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Failed to reach local API at {self.base_url}: {exc.reason}") from exc

    def get_health(self) -> dict[str, object]:
        return self._get_json("/health")

    def get_query_groups(self) -> dict[str, object]:
        return self._get_json("/query-groups")

    def get_rankings(self, query_group: str, limit: int, offset: int) -> dict[str, object]:
        return self._get_json(
            f"/rankings/{quote(query_group)}?limit={int(limit)}&offset={int(offset)}"
        )

    def get_product(self, asin: str) -> dict[str, object]:
        return self._get_json(f"/products/{quote(asin)}")


class FitIQDesktopApp(tk.Tk):
    BACKGROUND = "#f5f5f7"
    CARD_BACKGROUND = "#ffffff"
    BORDER = "#d1d1d6"
    TEXT = "#1c1c1e"
    MUTED = "#6e6e73"
    ACCENT = "#0a84ff"
    ACCENT_HOVER = "#238dff"
    ACCENT_PRESSED = "#0060df"
    TABLE_HEADER = "#f2f2f7"
    TABLE_SELECTED = "#dbeafe"
    TABLE_SELECTED_TEXT = "#0f172a"

    def __init__(self, client: ApiClient, on_close=None) -> None:
        super().__init__()
        self.client = client
        self.on_close_callback = on_close
        self.current_query_group: str | None = None
        self.current_offset = 0
        self.current_total = 0
        self.page_size = 25

        self.title("FitIQ Desktop Demo")
        self.geometry("1240x1050")
        self.minsize(980, 640)
        self.configure(bg=self.BACKGROUND)

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._handle_close)
        self.after(100, self.initialize)

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        default_font = ("Segoe UI", 10)
        title_font = ("Segoe UI", 18, "bold")
        value_font = ("Segoe UI", 14, "bold")
        inline_value_font = ("Segoe UI", 10, "bold")

        style.configure(".", background=self.BACKGROUND, foreground=self.TEXT, font=default_font)
        style.configure("App.TFrame", background=self.BACKGROUND)
        style.configure("Card.TFrame", background=self.CARD_BACKGROUND)
        style.configure(
            "Card.TLabelframe",
            background=self.CARD_BACKGROUND,
            bordercolor=self.BORDER,
            lightcolor=self.BORDER,
            darkcolor=self.BORDER,
            relief="solid",
            borderwidth=1,
        )
        style.configure(
            "Card.TLabelframe.Label",
            background=self.CARD_BACKGROUND,
            foreground=self.MUTED,
            font=("Segoe UI", 10, "bold"),
        )
        style.configure("Title.TLabel", background=self.BACKGROUND, foreground=self.TEXT, font=title_font)
        style.configure(
            "Subtitle.TLabel",
            background=self.BACKGROUND,
            foreground=self.MUTED,
            font=default_font,
        )
        style.configure(
            "Field.TLabel",
            background=self.CARD_BACKGROUND,
            foreground=self.MUTED,
            font=("Segoe UI", 9),
        )
        style.configure(
            "Value.TLabel",
            background=self.CARD_BACKGROUND,
            foreground=self.TEXT,
            font=("Segoe UI", 10, "bold"),
        )
        style.configure(
            "HealthValue.TLabel",
            background=self.CARD_BACKGROUND,
            foreground=self.TEXT,
            font=value_font,
        )
        style.configure(
            "InlineValue.TLabel",
            background=self.CARD_BACKGROUND,
            foreground=self.TEXT,
            font=inline_value_font,
        )
        style.configure(
            "Status.TLabel",
            background=self.BACKGROUND,
            foreground=self.MUTED,
            font=default_font,
        )
        style.configure(
            "Hint.TLabel",
            background=self.CARD_BACKGROUND,
            foreground=self.MUTED,
            font=default_font,
        )
        style.configure(
            "Accent.TButton",
            background=self.ACCENT,
            foreground="#ffffff",
            borderwidth=0,
            focuscolor=self.ACCENT,
            padding=(14, 8),
            font=("Segoe UI", 10, "bold"),
        )
        style.map(
            "Accent.TButton",
            background=[
                ("pressed", self.ACCENT_PRESSED),
                ("active", self.ACCENT_HOVER),
                ("disabled", "#c7d2e0"),
            ],
            foreground=[("disabled", "#f7f9fc")],
        )
        style.configure(
            "Modern.TEntry",
            fieldbackground=self.CARD_BACKGROUND,
            foreground=self.TEXT,
            bordercolor=self.BORDER,
            lightcolor=self.BORDER,
            darkcolor=self.BORDER,
            insertcolor=self.TEXT,
            padding=(10, 7),
        )
        style.configure(
            "Modern.TCombobox",
            fieldbackground=self.CARD_BACKGROUND,
            foreground=self.TEXT,
            bordercolor=self.BORDER,
            lightcolor=self.BORDER,
            darkcolor=self.BORDER,
            arrowsize=16,
            padding=(8, 6),
        )
        style.map(
            "Modern.TCombobox",
            fieldbackground=[("readonly", self.CARD_BACKGROUND)],
            foreground=[("readonly", self.TEXT)],
            selectbackground=[("readonly", self.CARD_BACKGROUND)],
            selectforeground=[("readonly", self.TEXT)],
        )
        style.configure(
            "Modern.Treeview",
            background=self.CARD_BACKGROUND,
            fieldbackground=self.CARD_BACKGROUND,
            foreground=self.TEXT,
            bordercolor=self.BORDER,
            relief="flat",
            rowheight=30,
            font=default_font,
        )
        style.map(
            "Modern.Treeview",
            background=[("selected", self.TABLE_SELECTED)],
            foreground=[("selected", self.TABLE_SELECTED_TEXT)],
        )
        style.configure(
            "Modern.Treeview.Heading",
            background=self.TABLE_HEADER,
            foreground=self.TEXT,
            bordercolor=self.BORDER,
            relief="flat",
            font=("Segoe UI", 10, "bold"),
            padding=(8, 8),
        )
        style.map(
            "Modern.Treeview.Heading",
            background=[("active", self.TABLE_HEADER)],
            foreground=[("active", self.TEXT)],
        )

    @staticmethod
    def format_query_group_label(query_group: str) -> str:
        return str(query_group).replace("_", " ").strip().capitalize()

    def _build_ui(self) -> None:
        self._configure_styles()

        container = ttk.Frame(self, padding=18, style="App.TFrame")
        container.pack(fill="both", expand=True)

        header = ttk.Frame(container, style="App.TFrame")
        header.pack(fill="x", pady=(0, 16))

        ttk.Label(header, text="FitIQ v1 Desktop Demo", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Local desktop client backed by the read-only FitIQ ranking API.",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        health_frame = ttk.LabelFrame(container, text="Service health", padding=14, style="Card.TLabelframe")
        health_frame.pack(fill="x", pady=(0, 16))

        self.health_status_var = tk.StringVar(value="Loading...")
        self.health_products_var = tk.StringVar(value="-")
        self.health_groups_var = tk.StringVar(value="-")
        self.health_score_var = tk.StringVar(value="-")

        for column_index in range(4):
            health_frame.columnconfigure(column_index, weight=1)

        self._add_stat(health_frame, 0, "Status", self.health_status_var)
        self._add_stat(health_frame, 1, "Ranked products", self.health_products_var)
        self._add_stat(health_frame, 2, "Query groups", self.health_groups_var)
        self._add_stat(health_frame, 3, "Score column", self.health_score_var)

        controls = ttk.LabelFrame(container, text="Rankings", padding=14, style="Card.TLabelframe")
        controls.pack(fill="x", pady=(0, 12))
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(4, weight=1)

        ttk.Label(controls, text="Query group", style="Field.TLabel").grid(row=0, column=0, sticky="w")
        self.query_group_var = tk.StringVar()
        self.query_group_combo = ttk.Combobox(
            controls,
            textvariable=self.query_group_var,
            state="readonly",
            width=42,
            style="Modern.TCombobox",
        )
        self.query_group_combo.grid(row=0, column=1, sticky="ew", padx=(8, 16))
        self.query_group_combo.bind("<<ComboboxSelected>>", self._on_query_group_changed)

        ttk.Label(controls, text="Total products", style="Field.TLabel").grid(row=0, column=2, sticky="w")
        self.total_products_var = tk.StringVar(value="-")
        ttk.Label(controls, textvariable=self.total_products_var, style="InlineValue.TLabel").grid(
            row=0, column=3, sticky="w", padx=(8, 16)
        )

        ttk.Label(controls, text="Showing", style="Field.TLabel").grid(row=0, column=4, sticky="w")
        self.showing_var = tk.StringVar(value="0")
        ttk.Label(controls, textvariable=self.showing_var, style="InlineValue.TLabel").grid(
            row=0, column=5, sticky="w", padx=(8, 0)
        )

        table_frame = ttk.Frame(container, style="Card.TFrame", padding=2)
        table_frame.pack(fill="both", expand=True)

        self.rankings_tree = ttk.Treeview(
            table_frame,
            columns=("rank", "asin", "subcategory", "score", "rating", "reviews"),
            show="headings",
            height=18,
            style="Modern.Treeview",
        )
        self.rankings_tree.pack(side="left", fill="both", expand=True)
        self.rankings_tree.heading("rank", text="Rank")
        self.rankings_tree.heading("asin", text="ASIN")
        self.rankings_tree.heading("subcategory", text="Subcategory")
        self.rankings_tree.heading("score", text="FitIQ score")
        self.rankings_tree.heading("rating", text="Mean rating")
        self.rankings_tree.heading("reviews", text="Reviews")
        self.rankings_tree.column("rank", width=72, anchor="center")
        self.rankings_tree.column("asin", width=170, anchor="w")
        self.rankings_tree.column("subcategory", width=180, anchor="w")
        self.rankings_tree.column("score", width=120, anchor="e")
        self.rankings_tree.column("rating", width=120, anchor="e")
        self.rankings_tree.column("reviews", width=120, anchor="e")
        self.rankings_tree.bind("<Double-1>", self._on_tree_double_click)

        tree_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.rankings_tree.yview)
        tree_scroll.pack(side="right", fill="y")
        self.rankings_tree.configure(yscrollcommand=tree_scroll.set)

        actions = ttk.Frame(container, style="App.TFrame")
        actions.pack(fill="x", pady=(10, 14))
        self.load_more_button = ttk.Button(
            actions,
            text="Load more",
            command=self.load_more,
            style="Accent.TButton",
        )
        self.load_more_button.pack(side="right")

        lookup = ttk.LabelFrame(container, text="ASIN lookup", padding=14, style="Card.TLabelframe")
        lookup.pack(fill="x")
        lookup.columnconfigure(1, weight=1)

        ttk.Label(lookup, text="ASIN", style="Field.TLabel").grid(row=0, column=0, sticky="w")
        self.asin_entry = ttk.Entry(lookup, style="Modern.TEntry")
        self.asin_entry.grid(row=0, column=1, sticky="ew", padx=(8, 10))
        self.asin_entry.bind("<Return>", self._on_lookup_submit)
        ttk.Button(lookup, text="Lookup", command=self.lookup_asin, style="Accent.TButton").grid(
            row=0, column=2, sticky="e"
        )

        details = ttk.Frame(lookup, style="Card.TFrame")
        details.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        for column_index in range(2):
            details.columnconfigure(column_index, weight=1)

        self.product_detail_vars = {
            "asin": tk.StringVar(value="-"),
            "query_group_v2": tk.StringVar(value="-"),
            "ranking_group": tk.StringVar(value="-"),
            "subcategory": tk.StringVar(value="-"),
            "rank": tk.StringVar(value="-"),
            "score_fitiq_v1": tk.StringVar(value="-"),
            "mean_rating": tk.StringVar(value="-"),
            "review_count": tk.StringVar(value="-"),
        }

        detail_specs = [
            ("ASIN", "asin"),
            ("Query group", "query_group_v2"),
            ("Ranking group", "ranking_group"),
            ("Subcategory", "subcategory"),
            ("Rank", "rank"),
            ("FitIQ score", "score_fitiq_v1"),
            ("Mean rating", "mean_rating"),
            ("Review count", "review_count"),
        ]
        for index, (label_text, key) in enumerate(detail_specs):
            row_index = index // 2
            column_index = index % 2
            cell = ttk.Frame(details, padding=(0, 0, 18, 10), style="Card.TFrame")
            cell.grid(row=row_index, column=column_index, sticky="ew")
            ttk.Label(cell, text=label_text, style="Field.TLabel").pack(anchor="w")
            ttk.Label(cell, textvariable=self.product_detail_vars[key], style="Value.TLabel").pack(
                anchor="w",
                pady=(2, 0),
            )

        self.product_note_var = tk.StringVar(value="No product loaded yet.")
        ttk.Label(lookup, textvariable=self.product_note_var, style="Hint.TLabel").grid(
            row=2,
            column=0,
            columnspan=3,
            sticky="w",
            pady=(8, 0),
        )

        self.status_var = tk.StringVar(value="Initializing...")
        ttk.Label(container, textvariable=self.status_var, anchor="w", style="Status.TLabel").pack(
            fill="x",
            pady=(10, 0),
        )

    def _add_stat(self, parent: ttk.Frame, column: int, label: str, variable: tk.StringVar) -> None:
        cell = ttk.Frame(parent, style="Card.TFrame")
        cell.grid(row=0, column=column, sticky="ew", padx=(0, 12))
        ttk.Label(cell, text=label, style="Field.TLabel").pack(anchor="w")
        ttk.Label(cell, textvariable=variable, style="HealthValue.TLabel").pack(anchor="w", pady=(4, 0))

    def initialize(self) -> None:
        try:
            health = self.client.get_health()
            self.health_status_var.set(str(health.get("status", "-")))
            self.health_products_var.set(f"{int(health.get('total_ranked_products', 0)):,}")
            self.health_groups_var.set(f"{int(health.get('query_group_count', 0)):,}")
            self.health_score_var.set(str(health.get("score_column", "-")))

            payload = self.client.get_query_groups()
            items = payload.get("items", [])
            self.query_group_display_to_key = {
                self.format_query_group_label(str(item["query_group_v2"])): str(item["query_group_v2"])
                for item in items
            }
            display_values = list(self.query_group_display_to_key.keys())
            self.query_group_combo["values"] = display_values
            if display_values:
                self.query_group_var.set(display_values[0])
                self.current_query_group = self.query_group_display_to_key[display_values[0]]
                self.refresh_rankings(reset=True)
            else:
                self.status_var.set("No query groups were returned by the local API.")
        except Exception as exc:
            self.status_var.set(str(exc))
            messagebox.showerror("FitIQ demo", str(exc))

    def refresh_rankings(self, reset: bool) -> None:
        if not self.current_query_group:
            return

        try:
            if reset:
                self.current_offset = 0
                self.current_total = 0
                for row_id in self.rankings_tree.get_children():
                    self.rankings_tree.delete(row_id)

            payload = self.client.get_rankings(
                query_group=self.current_query_group,
                limit=self.page_size,
                offset=self.current_offset,
            )
            items = payload.get("items", [])
            self.current_total = int(payload.get("total_products", 0))
            for item in items:
                self.rankings_tree.insert(
                    "",
                    "end",
                    values=(
                        int(item["rank_in_query_group"]),
                        str(item["asin"]),
                        str(item["subcategory"]),
                        f"{float(item['score_fitiq_v1']):.4f}",
                        f"{float(item['mean_rating']):.2f}",
                        f"{int(item['review_count']):,}",
                    ),
                )
            self.current_offset += len(items)
            self.total_products_var.set(f"{self.current_total:,}")
            self.showing_var.set(f"{min(self.current_offset, self.current_total):,}")
            self.load_more_button.state(
                ["disabled"] if self.current_offset >= self.current_total else ["!disabled"]
            )
            self.status_var.set(
                f"Showing {min(self.current_offset, self.current_total):,} of {self.current_total:,} products in {self.format_query_group_label(self.current_query_group)}."
            )
        except Exception as exc:
            self.status_var.set(str(exc))
            messagebox.showerror("FitIQ demo", str(exc))

    def load_more(self) -> None:
        self.refresh_rankings(reset=False)

    def _on_query_group_changed(self, _event=None) -> None:
        selected_display = self.query_group_var.get().strip()
        self.current_query_group = self.query_group_display_to_key.get(selected_display) or None
        self.refresh_rankings(reset=True)

    def _on_lookup_submit(self, _event=None) -> None:
        self.lookup_asin()

    def _on_tree_double_click(self, _event=None) -> None:
        selected = self.rankings_tree.selection()
        if not selected:
            return
        values = self.rankings_tree.item(selected[0], "values")
        if len(values) < 2:
            return
        asin = str(values[1])
        self.asin_entry.delete(0, tk.END)
        self.asin_entry.insert(0, asin)
        self.lookup_asin()

    def lookup_asin(self) -> None:
        asin = self.asin_entry.get().strip()
        if not asin:
            messagebox.showinfo("FitIQ demo", "Enter an ASIN to look it up.")
            return
        try:
            product = self.client.get_product(asin)
            self.product_detail_vars["asin"].set(str(product["asin"]))
            self.product_detail_vars["query_group_v2"].set(
                self.format_query_group_label(str(product["query_group_v2"]))
            )
            self.product_detail_vars["ranking_group"].set(str(product["ranking_group"]))
            self.product_detail_vars["subcategory"].set(str(product["subcategory"]))
            self.product_detail_vars["rank"].set(
                f"{int(product['rank_in_query_group']):,} / {int(product['query_group_size']):,}"
            )
            self.product_detail_vars["score_fitiq_v1"].set(f"{float(product['score_fitiq_v1']):.4f}")
            self.product_detail_vars["mean_rating"].set(f"{float(product['mean_rating']):.2f}")
            self.product_detail_vars["review_count"].set(f"{int(product['review_count']):,}")
            self.product_note_var.set("Loaded product details from the local ranking artifact.")
            self.status_var.set(f"Loaded product {asin}.")
        except Exception as exc:
            self.status_var.set(str(exc))
            messagebox.showerror("FitIQ demo", str(exc))

    def _handle_close(self) -> None:
        if callable(self.on_close_callback):
            try:
                self.on_close_callback()
            except Exception:
                pass
        self.destroy()


def launch_desktop_demo(client: ApiClient, on_close=None) -> int:
    app = FitIQDesktopApp(client=client, on_close=on_close)
    app.mainloop()
    return 0
