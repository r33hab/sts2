"""Stage 3: the combat-prior loop (`rl/run_train/DESIGN.md` section 6).

Each iteration generates fights with the champion prior in the search, trains a new
combat network on the last three iterations' fights, plays the fixed fight set with the
champion and the challenger, and promotes the challenger only if it ends those fights
with more HP and no more deaths. `python -m rl.prior_loop run` runs it; see `cli.py`.
"""
