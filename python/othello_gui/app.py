"""Play Othello against a minimax, MCTS or trained-network bot.

    uv run othello-gui

Pick your color and an opponent on the right, then press New game. Click a
highlighted square to move. The bot searches on a background thread, so the
window stays responsive while it thinks.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from othello_gui._othello_bot import (
    Board,
    Color,
    MCTSPlayer,
    MinimaxPlayer,
    NetworkPlayer,
    Player,
    square_name,
)

# geometry, in pixels
CELL = 68
LABEL = 44  # room for the a-h / 1-8 labels
BORDER = 3
GRID = 2
DISC_RADIUS = CELL * 0.38
HINT_RADIUS = CELL * 0.28
BAR_HEIGHT = 64
BAR_GAP = 14

# colors
GREEN = "#289b72"
HINT = "#4daf8a"
HINT_HOVER = "#74c3a4"
LAST_MOVE = "#d4524c"
LINE = "#111111"
BAR = "#b4b4b4"
BLACK = "#000000"
WHITE = "#ffffff"

BOT_DELAY_MS = 250  # pause before the bot starts, so your move is seen first
POLL_MS = 30

OPPONENTS = ("Minimax", "MCTS", "Model")


def other(color: Color) -> Color:
    return Color.WHITE if color == Color.BLACK else Color.BLACK


def color_name(color: Color) -> str:
    return "Black" if color == Color.BLACK else "White"


def find_default_model() -> str:
    """The newest runs/*/latest.ckpt, looked for from here and from the repo root."""
    roots = [Path.cwd(), Path(__file__).resolve().parents[2]]
    found = [p for root in roots for p in root.glob("runs/*/latest.ckpt")]
    if not found:
        return ""
    return str(max(found, key=lambda p: p.stat().st_mtime))


@dataclass
class BotResult:
    game: int
    square: int | None = None
    error: str | None = None


class OthelloApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("Othello")
        root.resizable(False, False)

        self.board = Board()
        self.human = Color.BLACK
        self.bot: Player | None = None
        self.bot_label = ""
        self.last_move: int | None = None
        self.hover: int | None = None
        # bumped on every new game so a search from an old game is ignored
        self.game = 0
        self.thinking = False
        self.results: queue.Queue[BotResult] = queue.Queue()

        self._build_board(root)
        self._build_controls(root)
        self.new_game()

    # ---------------------------------------------------------------- layout

    def _build_board(self, root: tk.Tk) -> None:
        size = LABEL + 8 * CELL + BORDER * 2 + 16
        height = size + BAR_GAP + BAR_HEIGHT
        self.canvas = tk.Canvas(root, width=size, height=height, bg=WHITE, highlightthickness=0)
        self.canvas.grid(row=0, column=0, padx=(12, 0), pady=12)
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Leave>", lambda _e: self._set_hover(None))

    def _build_controls(self, root: tk.Tk) -> None:
        panel = ttk.Frame(root, padding=16)
        panel.grid(row=0, column=1, sticky="ns")

        heading = ("Segoe UI", 11, "bold")

        ttk.Label(panel, text="You play", font=heading).grid(row=0, column=0, sticky="w")
        self.side = tk.StringVar(value="Black")
        sides = ttk.Frame(panel)
        sides.grid(row=1, column=0, sticky="w", pady=(2, 12))
        for name in ("Black", "White"):
            ttk.Radiobutton(sides, text=name, value=name, variable=self.side).pack(side="left", padx=(0, 12))

        ttk.Label(panel, text="Opponent", font=heading).grid(row=2, column=0, sticky="w")
        self.opponent = tk.StringVar(value="Minimax")
        opponents = ttk.Frame(panel)
        opponents.grid(row=3, column=0, sticky="w", pady=(2, 8))
        for name in OPPONENTS:
            ttk.Radiobutton(
                opponents, text=name, value=name, variable=self.opponent, command=self._show_options
            ).pack(side="left", padx=(0, 12))

        # one options frame per opponent, shown in the same grid cell
        self.options: dict[str, ttk.Frame] = {}

        f = ttk.Frame(panel)
        self.minimax_depth = tk.IntVar(value=6)
        self.minimax_heuristic = tk.StringVar(value="mobility")
        ttk.Label(f, text="Depth").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(f, from_=1, to=14, width=6, textvariable=self.minimax_depth).grid(row=0, column=1, sticky="w", padx=8, pady=2)
        ttk.Label(f, text="Heuristic").grid(row=1, column=0, sticky="w")
        ttk.Combobox(
            f, values=("mobility", "disc_difference"), state="readonly", width=16, textvariable=self.minimax_heuristic
        ).grid(row=1, column=1, sticky="w", padx=8, pady=2)
        self.options["Minimax"] = f

        f = ttk.Frame(panel)
        self.mcts_iterations = tk.IntVar(value=10_000)
        ttk.Label(f, text="Playouts per move").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(f, from_=100, to=1_000_000, increment=1000, width=10, textvariable=self.mcts_iterations).grid(
            row=0, column=1, sticky="w", padx=8, pady=2
        )
        self.options["MCTS"] = f

        f = ttk.Frame(panel)
        self.model_path = tk.StringVar(value=find_default_model())
        self.model_depth = tk.IntVar(value=4)
        ttk.Label(f, text="File (.ckpt / .mlfo)").grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Entry(f, width=30, textvariable=self.model_path).grid(row=1, column=0, sticky="we", pady=2)
        ttk.Button(f, text="Browse…", command=self._browse_model).grid(row=1, column=1, padx=(6, 0))
        depth_row = ttk.Frame(f)
        depth_row.grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Label(depth_row, text="Search depth").pack(side="left")
        ttk.Spinbox(depth_row, from_=1, to=10, width=6, textvariable=self.model_depth).pack(side="left", padx=8)
        self.options["Model"] = f

        for frame in self.options.values():
            frame.grid(row=4, column=0, sticky="nw")
        panel.rowconfigure(4, minsize=90)
        self._show_options()

        ttk.Button(panel, text="New game", command=self.new_game).grid(row=5, column=0, sticky="we", pady=(12, 16))

        ttk.Separator(panel).grid(row=6, column=0, sticky="we", pady=(0, 12))
        self.matchup = tk.StringVar()
        ttk.Label(panel, textvariable=self.matchup, wraplength=260).grid(row=7, column=0, sticky="w")
        self.status = tk.StringVar()
        ttk.Label(panel, textvariable=self.status, font=("Segoe UI", 12, "bold"), wraplength=260).grid(
            row=8, column=0, sticky="w", pady=(8, 0)
        )

    def _show_options(self) -> None:
        for name, frame in self.options.items():
            if name == self.opponent.get():
                frame.grid()
            else:
                frame.grid_remove()

    def _browse_model(self) -> None:
        current = Path(self.model_path.get())
        path = filedialog.askopenfilename(
            title="Choose a model",
            initialdir=current.parent if current.parent.exists() else Path.cwd(),
            filetypes=[("Models", "*.ckpt *.mlfo"), ("All files", "*.*")],
        )
        if path:
            self.model_path.set(path)

    # ------------------------------------------------------------- game flow

    def _make_bot(self) -> tuple[Player, str]:
        opponent = self.opponent.get()
        if opponent == "Minimax":
            depth = int(self.minimax_depth.get())
            heuristic = self.minimax_heuristic.get()
            return MinimaxPlayer(depth, heuristic), f"Minimax (depth {depth}, {heuristic})"
        if opponent == "MCTS":
            iterations = int(self.mcts_iterations.get())
            return MCTSPlayer(iterations), f"MCTS ({iterations:,} playouts)"
        path = self.model_path.get().strip()
        if not path:
            raise ValueError("choose a model file first")
        depth = int(self.model_depth.get())
        return NetworkPlayer.load(path, depth), f"Model {Path(path).parent.name}/{Path(path).name} (depth {depth})"

    def new_game(self) -> None:
        try:
            bot, label = self._make_bot()
        except Exception as e:  # bad numbers in a spinbox, or a model that won't load
            messagebox.showerror("Can't start a game", str(e))
            return
        self.game += 1
        self.thinking = False
        self.bot, self.bot_label = bot, label
        self.human = Color.BLACK if self.side.get() == "Black" else Color.WHITE
        self.board = Board()
        self.last_move = None
        self.matchup.set(f"You ({color_name(self.human)}) vs {label} ({color_name(other(self.human))})")
        self._advance(note="")

    def _play(self, square: int) -> None:
        mover = self.board.turn
        self.board.make_move(square)
        self.last_move = square
        note = ""
        if self.board.last_move_caused_pass:
            note = f"{color_name(other(mover))} has no moves and passes. "
        self._advance(note)

    def _advance(self, note: str) -> None:
        """Redraws, then either waits for the human or starts the bot."""
        self.hover = None
        self.render()
        if self.board.game_over:
            b, w = self.board.black_count, self.board.white_count
            if b == w:
                result = f"Draw, {b}-{w}."
            else:
                winner = Color.BLACK if b > w else Color.WHITE
                who = "You win" if winner == self.human else "You lose"
                result = f"{who}! {color_name(winner)} {max(b, w)}-{min(b, w)}."
            self.status.set(note + "Game over. " + result)
            return
        if self.board.turn == self.human:
            self.status.set(note + "Your move.")
            return
        self.status.set(note + "Opponent is thinking…")
        self.thinking = True
        game = self.game
        self.root.after(BOT_DELAY_MS, lambda: self._start_bot(game))

    def _start_bot(self, game: int) -> None:
        if game != self.game:
            return
        board, bot = Board(self.board), self.bot

        def search() -> None:
            try:
                self.results.put(BotResult(game, square=bot.get_move(board)))
            except Exception as e:
                self.results.put(BotResult(game, error=str(e)))

        threading.Thread(target=search, daemon=True).start()
        self.root.after(POLL_MS, self._poll_bot)

    def _poll_bot(self) -> None:
        try:
            result = self.results.get_nowait()
        except queue.Empty:
            self.root.after(POLL_MS, self._poll_bot)
            return
        if result.game != self.game:
            # an abandoned game's search; the current one may still be running
            self.root.after(POLL_MS, self._poll_bot)
            return
        self.thinking = False
        if result.error is not None:
            self.status.set(f"The bot failed: {result.error}")
            return
        self._play(result.square)

    # ----------------------------------------------------------------- input

    def _square_at(self, x: float, y: float) -> int | None:
        left = top = LABEL + BORDER
        col, row = int((x - left) // CELL), int((y - top) // CELL)
        if x < left or y < top or not (0 <= col < 8 and 0 <= row < 8):
            return None
        return row * 8 + col

    def _human_to_move(self) -> bool:
        return not self.thinking and not self.board.game_over and self.board.turn == self.human

    def _on_click(self, event: tk.Event) -> None:
        square = self._square_at(event.x, event.y)
        if square is None or not self._human_to_move():
            return
        if square in self.board.legal_moves():
            self._play(square)

    def _on_motion(self, event: tk.Event) -> None:
        square = self._square_at(event.x, event.y)
        if not self._human_to_move() or square not in self.board.legal_moves():
            square = None
        self._set_hover(square)

    def _set_hover(self, square: int | None) -> None:
        if square != self.hover:
            self.hover = square
            self.canvas.configure(cursor="hand2" if square is not None else "")
            self.render()

    # --------------------------------------------------------------- drawing

    def render(self) -> None:
        c = self.canvas
        c.delete("all")
        left = top = LABEL + BORDER
        size = 8 * CELL
        label_font = ("Arial", 17, "bold")

        for i in range(8):
            center = left + i * CELL + CELL / 2
            c.create_text(center, LABEL / 2, text="abcdefgh"[i], font=label_font)
            c.create_text(LABEL / 2, center, text=str(i + 1), font=label_font)

        c.create_rectangle(left, top, left + size, top + size, fill=GREEN, outline="")

        hints = set(self.board.legal_moves()) if self._human_to_move() else set()
        for square in range(64):
            row, col = divmod(square, 8)
            x0, y0 = left + col * CELL, top + row * CELL
            cx, cy = x0 + CELL / 2, y0 + CELL / 2
            if square == self.last_move:
                c.create_rectangle(x0, y0, x0 + CELL, y0 + CELL, fill=LAST_MOVE, outline="")
            disc = self.board.at(square)
            if disc is not None:
                self._disc(cx, cy, DISC_RADIUS, BLACK if disc == Color.BLACK else WHITE)
            elif square in hints:
                fill = HINT_HOVER if square == self.hover else HINT
                c.create_oval(cx - HINT_RADIUS, cy - HINT_RADIUS, cx + HINT_RADIUS, cy + HINT_RADIUS, fill=fill, outline="")

        for i in range(1, 8):
            offset = i * CELL
            c.create_line(left + offset, top, left + offset, top + size, fill=LINE, width=GRID)
            c.create_line(left, top + offset, left + size, top + offset, fill=LINE, width=GRID)
        c.create_rectangle(left, top, left + size, top + size, outline=LINE, width=BORDER)

        self._score_bar(left, top + size + BAR_GAP, size)

    def _disc(self, cx: float, cy: float, r: float, fill: str, outline: str = "") -> None:
        self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill=fill, outline=outline, width=2 if outline else 0)

    def _score_bar(self, x: float, y: float, width: float) -> None:
        c = self.canvas
        inset = width * 0.07
        c.create_rectangle(x + inset, y, x + width - inset, y + BAR_HEIGHT, fill=BAR, outline="")
        cy = y + BAR_HEIGHT / 2
        mid = x + width / 2
        r = BAR_HEIGHT * 0.33
        font = ("Arial", 22, "bold")

        black_x, white_x = mid - 120, mid + 120
        # ring the side to move while the game is on
        if not self.board.game_over:
            ring_x = black_x if self.board.turn == Color.BLACK else white_x
            self._disc(ring_x, cy, r + 5, BAR, outline=LAST_MOVE)
        self._disc(black_x, cy, r, BLACK)
        self._disc(white_x, cy, r, WHITE)
        c.create_text(mid - 55, cy, text=str(self.board.black_count), font=font)
        c.create_text(mid, cy, text="-", font=font)
        c.create_text(mid + 55, cy, text=str(self.board.white_count), font=font)

        if self.last_move is not None:
            c.create_text(x + width - inset - 8, y + BAR_HEIGHT - 6, text=f"last {square_name(self.last_move)}",
                          anchor="se", font=("Arial", 10))


def main() -> None:
    root = tk.Tk()
    OthelloApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
