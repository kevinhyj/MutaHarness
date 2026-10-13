"""Properties that matter for blind rank fusion, independent of archived results."""

import math
import unittest

from runtime.ranking import percentile_ranks, query_ids, rank_ensemble, validate_scores


class RankFusionTests(unittest.TestCase):
    def setUp(self):
        self.ids = ["a", "b", "c"]
        self.forward = {"a": 1, "b": 2, "c": 3}

    def test_ties_share_the_midpoint(self):
        self.assertEqual(percentile_ranks([7]), [0.5])
        self.assertEqual(percentile_ranks([7, 1, 1, 9]), [0.625, 0.25, 0.25, 0.875])
        self.assertEqual(percentile_ranks([7, 7, 7]), [0.5, 0.5, 0.5])

    def test_unanimous_programs_have_no_disagreement(self):
        rows = rank_ensemble(self.ids, [self.forward] * 3)
        for row, expected in zip(rows, [-2 / 3, 0, 2 / 3], strict=True):
            self.assertAlmostEqual(row["score"], expected)
            self.assertEqual(row["disagreement"], 0)

    def test_member_scales_do_not_change_the_ensemble(self):
        rescaled = {id_: 10_000 * score - 200 for id_, score in self.forward.items()}
        base = rank_ensemble(self.ids, [self.forward] * 3)
        scaled = rank_ensemble(self.ids, [self.forward, rescaled, self.forward])
        self.assertEqual([row["score"] for row in base], [row["score"] for row in scaled])
        self.assertEqual(
            [row["disagreement"] for row in base], [row["disagreement"] for row in scaled]
        )

    def test_balanced_disagreement_is_retained(self):
        reverse = {"a": 3, "b": 2, "c": 1}
        tied = dict.fromkeys(self.ids, 1)
        rows = rank_ensemble(self.ids, [self.forward, reverse, tied])
        self.assertEqual([row["score"] for row in rows], [0, 0, 0])
        self.assertAlmostEqual(rows[0]["disagreement"], math.sqrt(2 / 27))
        self.assertEqual(rows[1]["disagreement"], 0)

    def test_ids_not_dictionary_order_define_alignment(self):
        reordered = {"c": 3, "a": 1, "b": 2}
        ids = ["c", "a", "b"]
        rows = rank_ensemble(ids, [self.forward, reordered, self.forward])
        self.assertEqual([row["id"] for row in rows], ids)
        self.assertEqual([row["program_scores"] for row in rows], [[3, 3, 3], [1, 1, 1], [2, 2, 2]])

    def test_missing_or_extra_candidates_are_rejected(self):
        for malformed in [{"a": 1, "b": 2}, {**self.forward, "extra": 4}]:
            with self.subTest(scores=malformed), self.assertRaises(ValueError):
                rank_ensemble(self.ids, [self.forward, malformed, self.forward])

    def test_nonfinite_and_nonnumeric_scores_are_rejected(self):
        for value in [math.nan, math.inf, -math.inf, True, "1", None]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                rank_ensemble(self.ids, [self.forward, {**self.forward, "a": value}, self.forward])

    def test_exactly_three_members_are_required(self):
        for count in [0, 1, 2, 4]:
            with self.subTest(count=count), self.assertRaises(ValueError):
                rank_ensemble(self.ids, [self.forward] * count)

    def test_duplicate_and_empty_queries_are_rejected(self):
        for ids in [[], ["a", "a"], [""], [1]]:
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                rank_ensemble(ids, [self.forward] * 3)
        with self.assertRaises(ValueError):
            percentile_ranks([])

    def test_query_ids_must_be_unmeasured(self):
        clean = {"query": [{"id": "a"}], "support": [{"id": "b", "fitness": 1}]}
        self.assertEqual(query_ids(clean), ["a"])
        for case in [
            {"query": [{"id": "a"}, {"id": "a"}]},
            {"query": [{"id": "a", "fitness": 1}]},
            {"query": [{"id": "a"}], "support": [{"id": "a", "fitness": 1}]},
        ]:
            with self.subTest(case=case), self.assertRaises(ValueError):
                query_ids(case)

    def test_replay_score_bounds_are_enforced(self):
        validate_scores({"a": -1, "b": 1}, ["a", "b"], bounded=True)
        with self.assertRaises(ValueError):
            validate_scores(self.forward, self.ids, bounded=True)


if __name__ == "__main__":
    unittest.main()
