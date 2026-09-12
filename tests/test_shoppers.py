"""
Tests for skim.shoppers.

The lockout is the security control here, not the PIN -- four digits is
10,000 combinations and falls in under a minute to a script. So these
tests care most about the counting: that it locks when it should, that
it does not lock early, and that a correct PIN during a lockout is still
refused.
"""

import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from skim import storage
from skim.shoppers import (
    LOCKOUT_MINUTES,
    MAX_FAILED_ATTEMPTS,
    ShopperError,
    get_shopper,
    hash_pin,
    sign_in,
)


class ShopperTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.conn = storage.connect(Path(self.tmp.name) / "skim.db")
        self.addCleanup(self.conn.close)

    def _wrong(self, times, name="Priya"):
        last = None
        for _ in range(times):
            with self.assertRaises(ShopperError) as caught:
                sign_in(self.conn, name, "0000")
            last = str(caught.exception)
        return last

    # -- claiming a name --------------------------------------------

    def test_an_unclaimed_name_is_claimed_by_whoever_types_it(self):
        # There is no separate registration step; that is the point.
        self.assertIsInstance(sign_in(self.conn, "Priya", "4821"), int)

    def test_the_same_person_signs_back_in(self):
        first = sign_in(self.conn, "Priya", "4821")
        self.assertEqual(sign_in(self.conn, "Priya", "4821"), first)

    def test_names_are_case_insensitive(self):
        first = sign_in(self.conn, "Priya", "4821")
        self.assertEqual(sign_in(self.conn, "  priya ", "4821"), first)

    def test_two_people_are_two_shoppers(self):
        self.assertNotEqual(sign_in(self.conn, "Priya", "4821"),
                            sign_in(self.conn, "Arjun", "4821"))

    def test_the_display_name_keeps_its_capitals(self):
        shopper_id = sign_in(self.conn, "PriYa", "4821")
        self.assertEqual(get_shopper(self.conn, shopper_id)["display_name"], "PriYa")

    # -- the PIN ----------------------------------------------------

    def test_a_wrong_pin_is_refused(self):
        sign_in(self.conn, "Priya", "4821")
        with self.assertRaises(ShopperError):
            sign_in(self.conn, "Priya", "0000")

    def test_the_pin_is_never_stored_in_the_clear(self):
        sign_in(self.conn, "Priya", "4821")
        row = self.conn.execute("SELECT * FROM shoppers").fetchone()
        self.assertNotIn("4821", " ".join(str(v) for v in tuple(row)))

    def test_each_shopper_gets_their_own_salt(self):
        # Without per-shopper salts, two people choosing 1234 would have
        # identical hashes -- which tells anyone reading the database
        # that those two accounts share a PIN.
        sign_in(self.conn, "Priya", "1234")
        sign_in(self.conn, "Arjun", "1234")
        rows = self.conn.execute("SELECT pin_salt, pin_hash FROM shoppers").fetchall()
        self.assertNotEqual(rows[0]["pin_salt"], rows[1]["pin_salt"])
        self.assertNotEqual(rows[0]["pin_hash"], rows[1]["pin_hash"])

    def test_pins_must_be_four_digits(self):
        for bad in ("123", "12345", "abcd", "", "12 4"):
            with self.assertRaises(ShopperError):
                sign_in(self.conn, "Priya", bad)

    def test_names_have_to_be_reasonable(self):
        for bad in ("A", "", "x" * 25, "robert'); DROP TABLE--"):
            with self.assertRaises(ShopperError):
                sign_in(self.conn, bad, "4821")

    # -- the lockout, which is the actual protection -----------------

    def test_it_does_not_lock_before_the_limit(self):
        sign_in(self.conn, "Priya", "4821")
        self._wrong(MAX_FAILED_ATTEMPTS - 1)
        # Still the right PIN, still works.
        self.assertIsInstance(sign_in(self.conn, "Priya", "4821"), int)

    def test_it_locks_on_the_limit(self):
        sign_in(self.conn, "Priya", "4821")
        message = self._wrong(MAX_FAILED_ATTEMPTS)
        self.assertIn("locked", message.lower())

    def test_a_correct_pin_is_refused_while_locked(self):
        # The whole point. If the right PIN worked during a lockout, an
        # attacker would just include it in the guesses.
        sign_in(self.conn, "Priya", "4821")
        self._wrong(MAX_FAILED_ATTEMPTS)
        with self.assertRaises(ShopperError) as caught:
            sign_in(self.conn, "Priya", "4821")
        self.assertIn("try again", str(caught.exception).lower())

    def test_a_successful_sign_in_clears_the_count(self):
        sign_in(self.conn, "Priya", "4821")
        self._wrong(MAX_FAILED_ATTEMPTS - 1)
        sign_in(self.conn, "Priya", "4821")
        row = self.conn.execute("SELECT failed_attempts FROM shoppers").fetchone()
        self.assertEqual(row["failed_attempts"], 0)

    def test_the_lockout_expires(self):
        shopper_id = sign_in(self.conn, "Priya", "4821")
        self._wrong(MAX_FAILED_ATTEMPTS)
        # Wind the clock back rather than waiting fifteen minutes.
        past = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(
            timespec="seconds")
        self.conn.execute("UPDATE shoppers SET locked_until = ?", (past,))
        self.conn.commit()
        self.assertEqual(sign_in(self.conn, "Priya", "4821"), shopper_id)

    def test_locking_one_name_does_not_lock_another(self):
        sign_in(self.conn, "Priya", "4821")
        other = sign_in(self.conn, "Arjun", "7777")
        self._wrong(MAX_FAILED_ATTEMPTS)
        self.assertEqual(sign_in(self.conn, "Arjun", "7777"), other)

    def test_a_failure_does_not_reveal_whether_the_name_exists(self):
        # Saying "no such name" would tell an attacker which names are
        # worth spending guesses on.
        sign_in(self.conn, "Priya", "4821")
        with self.assertRaises(ShopperError) as known:
            sign_in(self.conn, "Priya", "0000")
        # An unclaimed name simply gets claimed, so the only observable
        # failure for a valid name+PIN shape is a wrong PIN -- and that
        # message never mentions the name's existence either way.
        self.assertNotIn("exist", str(known.exception).lower())
        self.assertNotIn("no such", str(known.exception).lower())


class HashTests(unittest.TestCase):
    def test_the_same_pin_and_salt_hash_the_same(self):
        salt = "a" * 32
        self.assertEqual(hash_pin("4821", salt), hash_pin("4821", salt))

    def test_a_different_salt_gives_a_different_hash(self):
        self.assertNotEqual(hash_pin("4821", "a" * 32), hash_pin("4821", "b" * 32))


if __name__ == "__main__":
    unittest.main()
