"""
Unit tests for the data manager and scoring engine.

These cover the pure-logic core of the project and need no GUI / screen /
OpenCV — run them with::

    cd mlbb_draft_assistant
    python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

# Make the project root importable when run directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src.data_manager import DataManager
from src.scoring_engine import DraftState, ScoringEngine


class DataManagerTests(unittest.TestCase):
    def setUp(self):
        self.dm = DataManager().load()

    def test_loads_heroes(self):
        self.assertGreater(len(self.dm.all_heroes()), 20)
        self.assertIsNotNone(self.dm.get_hero("Fanny"))

    def test_rank_filtering_changes_stats(self):
        fanny_mg = self.dm.get_hero("Fanny")
        self.assertAlmostEqual(fanny_mg.ban_rate, 0.72, places=2)
        self.dm.set_rank("all")
        fanny_all = self.dm.get_hero("Fanny")
        self.assertAlmostEqual(fanny_all.ban_rate, 0.40, places=2)
        self.assertNotEqual(fanny_mg.ban_rate, fanny_all.ban_rate)

    def test_available_excludes_taken(self):
        avail = self.dm.available_heroes({"Fanny", "Ling"})
        names = {h.name for h in avail}
        self.assertNotIn("Fanny", names)
        self.assertNotIn("Ling", names)

    def test_unknown_rank_raises(self):
        with self.assertRaises(ValueError):
            self.dm.set_rank("grandmaster")


class ScoringEngineTests(unittest.TestCase):
    def setUp(self):
        self.dm = DataManager().load()
        self.engine = ScoringEngine(self.dm)

    def _score(self, name, draft, **kw):
        return self.engine.score_hero(self.dm.get_hero(name), draft, **kw)

    def test_baseline_present(self):
        bd = self._score("Beatrix", DraftState())
        self.assertEqual(bd.components["baseline"], config.SCORING["baseline"])

    def test_higher_winrate_scores_higher(self):
        empty = DraftState()
        # Karrie (54% WR) should out-meta Miya (46% WR) on an empty board.
        karrie = self._score("Karrie", empty).total
        miya = self._score("Miya", empty).total
        self.assertGreater(karrie, miya)

    def test_countered_by_enemy_penalty(self):
        # Fanny is weak against Khufra; enemy has Khufra.
        draft = DraftState(enemy_picks=["Khufra"])
        bd = self._score("Fanny", draft)
        self.assertIn("countered_by_enemy", bd.components)
        self.assertLess(bd.components["countered_by_enemy"], 0)

    def test_counter_neutralised_bonus(self):
        # Khufra banned -> Fanny's threat removed -> positive component.
        draft = DraftState(bans=["Khufra"])
        bd = self._score("Fanny", draft)
        self.assertIn("counter_neutralised", bd.components)
        self.assertGreater(bd.components["counter_neutralised"], 0)

    def test_we_counter_enemy_bonus(self):
        # Fanny is strong against Layla; enemy picked Layla.
        draft = DraftState(enemy_picks=["Layla"])
        bd = self._score("Fanny", draft)
        self.assertIn("we_counter_enemy", bd.components)
        self.assertGreater(bd.components["we_counter_enemy"], 0)

    def test_synergy_bonus(self):
        # Angela synergises with Fanny.
        draft = DraftState(ally_picks=["Angela"])
        bd = self._score("Fanny", draft)
        self.assertIn("synergy", bd.components)

    def test_role_saturation_penalty(self):
        # We already have a roam (Tigreal); another roam should be penalised.
        draft = DraftState(ally_picks=["Tigreal"])
        bd = self._score("Atlas", draft)  # Atlas is roam
        self.assertIn("role_saturation", bd.components)
        self.assertLess(bd.components["role_saturation"], 0)

    def test_damage_balance_bonus(self):
        # All-physical team should boost a magic candidate.
        draft = DraftState(ally_picks=["Beatrix", "Chou"])  # both physical
        magic_bd = self._score("Pharsa", draft)  # magic
        self.assertIn("damage_balance", magic_bd.components)
        self.assertGreater(magic_bd.components["damage_balance"], 0)

    def test_lane_counter_mode_boosts_direct_counter(self):
        # Enemy gold = Layla. Brody (gold) is strong against Layla.
        draft = DraftState(enemy_picks=["Layla"])
        without = self._score("Brody", draft, lane="gold", lane_counter=False)
        with_lc = self._score("Brody", draft, lane="gold", lane_counter=True)
        self.assertIn("lane_counter", with_lc.components)
        self.assertGreater(with_lc.total, without.total)

    def test_recommend_excludes_taken_and_groups_by_lane(self):
        draft = DraftState(ally_picks=["Tigreal"], bans=["Fanny"])
        recs = self.engine.recommend(draft)
        self.assertEqual(set(recs.keys()), set(config.LANES))
        all_recommended = {b.hero for lane in recs.values() for b in lane}
        self.assertNotIn("Fanny", all_recommended)   # banned
        self.assertNotIn("Tigreal", all_recommended)  # already picked

    def test_dark_system_reverses_order(self):
        draft = DraftState()
        normal = self.engine.recommend(draft, dark_system=False)["gold"]
        dark = self.engine.recommend(draft, dark_system=True)["gold"]
        # Normal best should beat dark "best" (which is actually the worst).
        self.assertGreater(normal[0].total, dark[0].total)
        # Within dark mode, list is ascending (worst-first).
        self.assertLessEqual(dark[0].total, dark[-1].total)


if __name__ == "__main__":
    unittest.main(verbosity=2)
