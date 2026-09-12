"""
Who is using Skim, and how they prove it.

The smallest identity that actually works: a name they pick and a
four-digit PIN. No email, no password reset, no profile. The goal is
that a friend can start using this in ten seconds and still only see
their own receipts.

WHY A PIN NEEDS LOCKOUT, and why this is not optional.

Four digits is 10,000 combinations. A script can try all of them in
well under a minute, so the PIN by itself protects nothing at all. What
makes it usable is refusing to answer quickly: after a handful of wrong
guesses the account stops accepting attempts for a while, which turns
"guessable in seconds" into "guessable over weeks, noisily". The
lockout is the security control here; the PIN is just the secret it
protects.

WHY THE PIN IS HASHED even though four digits is brute-forceable.

Anyone who steals the database can compute all 10,000 hashes and find
the PIN regardless -- so hashing does not make the PIN safe. It is done
anyway because it means the database never contains the literal thing a
person typed, which matters when people reuse PINs across apps that are
not this one. Storing a secret in plaintext because it was a weak
secret is how the habit gets lost.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

# PBKDF2 from the standard library. bcrypt or argon2 would be stronger
# and would both be new dependencies; against a 10,000-item keyspace the
# difference is academic, because the defence here is the lockout, not
# the cost of a single hash.
PBKDF2_ITERATIONS = 200_000

MAX_FAILED_ATTEMPTS = 5
LOCKOUT_MINUTES = 15

PIN_PATTERN = re.compile(r"^\d{4}$")
NAME_PATTERN = re.compile(r"^[A-Za-z0-9 ._'-]{2,24}$")


class ShopperError(Exception):
    """Something about the name or PIN was not acceptable."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def hash_pin(pin: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", pin.encode(), bytes.fromhex(salt), PBKDF2_ITERATIONS
    ).hex()


def validate_name(name: str) -> str:
    cleaned = (name or "").strip()
    if not NAME_PATTERN.match(cleaned):
        raise ShopperError(
            "Names can be 2-24 characters: letters, numbers, spaces, "
            "and . _ ' - only."
        )
    return cleaned


def validate_pin(pin: str) -> str:
    cleaned = (pin or "").strip()
    if not PIN_PATTERN.match(cleaned):
        raise ShopperError("Your PIN has to be exactly 4 digits.")
    return cleaned


def _lockout_remaining(row: sqlite3.Row) -> Optional[int]:
    """Seconds left on a lockout, or None if not locked."""
    if not row["locked_until"]:
        return None
    until = datetime.fromisoformat(row["locked_until"])
    remaining = (until - _now()).total_seconds()
    return int(remaining) if remaining > 0 else None


def sign_in(connection: sqlite3.Connection, name: str, pin: str) -> int:
    """Return the shopper id for this name and PIN, creating them if new.

    There is no separate "register" step. A name nobody has claimed is
    claimed by whoever types it first, with the PIN they chose. A name
    that already exists needs its PIN. This is the whole of the account
    system, and it is enough: a friend can be uploading within seconds
    and still cannot read anyone else's receipts.

    Raises ShopperError with a message meant to be shown to the person.
    """
    display_name = validate_name(name)
    pin = validate_pin(pin)
    key = display_name.lower()

    row = connection.execute(
        "SELECT * FROM shoppers WHERE name_key = ?", (key,)
    ).fetchone()

    if row is None:
        salt = os.urandom(16).hex()
        cursor = connection.execute(
            """
            INSERT INTO shoppers (name_key, display_name, pin_hash, pin_salt,
                                  created_at)
            VALUES (?,?,?,?,?)
            """,
            (key, display_name, hash_pin(pin, salt), salt,
             _now().isoformat(timespec="seconds")),
        )
        connection.commit()
        return int(cursor.lastrowid)

    locked_for = _lockout_remaining(row)
    if locked_for is not None:
        minutes = max(1, round(locked_for / 60))
        raise ShopperError(
            f"Too many wrong PINs for that name. Try again in {minutes} "
            f"minute{'s' if minutes != 1 else ''}."
        )

    # `compare_digest` rather than `==`: string comparison returns as soon
    # as two characters differ, and the time that takes leaks how much of
    # the guess was right. Constant-time comparison does not.
    if hmac.compare_digest(hash_pin(pin, row["pin_salt"]), row["pin_hash"]):
        connection.execute(
            "UPDATE shoppers SET failed_attempts = 0, locked_until = NULL "
            "WHERE shopper_id = ?", (row["shopper_id"],)
        )
        connection.commit()
        return int(row["shopper_id"])

    attempts = int(row["failed_attempts"]) + 1
    locked_until = None
    if attempts >= MAX_FAILED_ATTEMPTS:
        locked_until = (_now() + timedelta(minutes=LOCKOUT_MINUTES)).isoformat(
            timespec="seconds")
        attempts = 0  # the lockout replaces the count

    connection.execute(
        "UPDATE shoppers SET failed_attempts = ?, locked_until = ? "
        "WHERE shopper_id = ?", (attempts, locked_until, row["shopper_id"])
    )
    connection.commit()

    if locked_until:
        raise ShopperError(
            f"That PIN is wrong, and that was {MAX_FAILED_ATTEMPTS} tries. "
            f"This name is locked for {LOCKOUT_MINUTES} minutes."
        )
    # Deliberately does NOT say whether the name exists -- that would
    # tell an attacker which names are worth attacking.
    left = MAX_FAILED_ATTEMPTS - attempts
    raise ShopperError(
        f"That name and PIN don't match. {left} "
        f"{'tries' if left != 1 else 'try'} left before it locks."
    )


def get_shopper(connection: sqlite3.Connection, shopper_id: int) -> Optional[sqlite3.Row]:
    return connection.execute(
        "SELECT * FROM shoppers WHERE shopper_id = ?", (shopper_id,)
    ).fetchone()
