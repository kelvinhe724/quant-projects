"""Production order placement. Same book as paper.py, three locks on the door.

Runs only if LIVE_TRADING=true in .env, the confirmation phrase is typed at the
prompt, and every order fits under MAX_ORDER_DOLLARS and MAX_OPEN_DOLLARS.
Each order is written to live.log before it is sent.
"""

import os
import sys

import kalshi
import paper

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "live.log")
PHRASE = "place real money orders"


def armed(settings, prompt=input):
    """True only when the .env switch is on and the operator types the phrase."""
    if settings.get("LIVE_TRADING", "false").strip().lower() != "true":
        print("live trading is OFF (LIVE_TRADING is not true in .env). Nothing sent.")
        return False
    try:
        typed = prompt(f'Production. Real money. Type "{PHRASE}" to continue: ')
    except EOFError:  # no terminal (cron, closed stdin): a phrase cannot be typed
        typed = ""
    if typed.strip() != PHRASE:
        print("phrase did not match. Nothing sent.")
        return False
    return True


def log(line):
    with open(LOG, "a") as fh:
        fh.write(f"{paper.now_iso()} {line}\n")
    print(line)


def main(argv):
    settings = kalshi.load_env()
    if not armed(settings):
        sys.exit(1)
    n = int(argv[1]) if len(argv) > 1 else paper.DEFAULT_N
    client = kalshi.Client("prod", settings)
    db = paper.open_ledger()
    log(f"LIVE session, caps {settings['MAX_ORDER_DOLLARS']}/order "
        f"{settings['MAX_OPEN_DOLLARS']} open, n={n}")
    try:
        paper.place(client, db, n, settings, log=log)
    except kalshi.NoKey as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv)
