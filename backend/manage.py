"""
Admin CLI (run on the server)
=============================
  python backend/manage.py create-user <username> --role engineer|planner
  python backend/manage.py reset-password <username>
  python backend/manage.py list-users

Passwords are read interactively (never pass them on the command line - shell history).
Further users can be created by an engineer in the web UI (Users tab).
"""

import argparse
import getpass
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import auth
from db import get_conn, init_db


def _read_password() -> str:
    pw = getpass.getpass("Password: ")
    if pw != getpass.getpass("Repeat password: "):
        sys.exit("Passwords do not match.")
    if len(pw) < auth.MIN_PASSWORD_LENGTH:
        sys.exit(f"Password must be at least {auth.MIN_PASSWORD_LENGTH} characters.")
    return pw


def main():
    parser = argparse.ArgumentParser(description="Belt RUL admin")
    sub = parser.add_subparsers(dest="cmd", required=True)
    cu = sub.add_parser("create-user")
    cu.add_argument("username")
    cu.add_argument("--role", choices=["engineer", "planner"], required=True)
    rp = sub.add_parser("reset-password")
    rp.add_argument("username")
    sub.add_parser("list-users")
    args = parser.parse_args()

    init_db()
    if args.cmd == "create-user":
        try:
            auth.create_user(args.username, _read_password(), args.role)
        except Exception as e:
            sys.exit(f"Could not create user: {e}")
        print(f"Created {args.role} '{args.username}'.")
    elif args.cmd == "reset-password":
        pw = _read_password()
        with get_conn() as conn:
            cur = conn.execute("UPDATE users SET password_hash = ? WHERE username = ?", (auth.hash_password(pw), args.username))
            if cur.rowcount == 0:
                sys.exit("No such user.")
            conn.execute("DELETE FROM sessions WHERE user_id = (SELECT id FROM users WHERE username = ?)", (args.username,))
        print("Password reset; existing sessions signed out.")
    else:
        with get_conn() as conn:
            for r in conn.execute("SELECT username, role, active, created_at FROM users ORDER BY id"):
                print(f"{r['username']:24s} {r['role']:9s} {'active' if r['active'] else 'disabled':9s} {r['created_at']}")


if __name__ == "__main__":
    main()
