# 2026-10-28 — tic-tac-toe: a watch-confirmed opponent move moves that piece in the belief (stream C)

- **What happened.** On the proxy replay of B_play_tic_tac_toe_breadth2 100004 (per-step frames, stream C), the
  opponent moved `opponent_piece0` from its start (-0.15, 0.20) to the board's centre around step 500.
  `play_board_game` registered the move from its watch read (`st["opp_cells"]`), but nothing moved the piece in the
  belief: the runtime only re-measures what our own jaws touched. For the last 5 events (steps 488-677) the planner's
  collision model and the grasp-time metric had no object on the centre cell ("neighbour opponent_piece0 not found").
  Stream M's table counts 13 such events across the tic-tac-toe rounds.
- **Change** (`interactive.opponent_into_belief`, `OPP_INTO_BELIEF = True`; `play_board_game(opp_into_belief=...)`):
  once the opponent's arm is at rest after a watch-confirmed landing in cell k, the opponent piece whose believed start
  place is now clearly empty in the head view moves there. "Clearly empty" means perception's `check` precision
  < 0.3 over >= 40 px, and it must hold for exactly one piece. The provider's `confirm_pose` second-look rule then has
  to prefer the cell before the belief moves. With no checker (G3) it does nothing.
- **Why it is default on.** In the head checks of every opponent start place, every 5th step of the 4 proxy
  tic-tac-toe replays (100000-02 + 100004; `~/dojo-data/bgperc/ttt_headcheck.json` on compute3):
  - a piece still at its place never read < 0.3: 0 of 675 checks, minimum 0.70;
  - the emptied place read 0.0 on 11 of 11 checks once visible;
  - the gate abstains when the place is hidden (n_vis 0 at steps 480-520), and when two places look empty.
  - Unit tests replay those values (`tests/dojo/test_longh_ttt.py`).
- **Not measured yet.** A rendered or proxy episode with the change. G3 mode A is unchanged by construction (no
  checker).
