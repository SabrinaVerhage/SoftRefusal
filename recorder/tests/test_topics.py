"""Text-theme regressions, independent of GPU recordings and refusal labels."""
import unittest

from avoidance.topics import topic_layout


def rows(prompts, conversation="fixture"):
    return [dict(point_id=f"{conversation}-{i}", conversation_id=conversation, turn_id=i, prompt=prompt)
            for i, prompt in enumerate(prompts)]


class TopicTests(unittest.TestCase):
    def test_distinct_openings_and_short_followup_context(self):
        source = rows(["A romantic kiss and tender cuddling", "Go on", "Explain contraception and pregnancy"])
        points, themes = topic_layout(source)
        self.assertEqual([p["topic_id"] for p in points], ["romance", "romance", "health"])
        self.assertTrue(points[1]["topic_context"])
        self.assertNotEqual(points[0]["topic_angle"], points[2]["topic_angle"])
        self.assertEqual(sum(len(r["point_ids"]) for r in themes["regions"]), len(source))

    def test_generic_sex_is_not_automatically_health(self):
        points, _ = topic_layout(rows(["sex sexy desire pleasure fantasy"]))
        self.assertEqual(points[0]["topic_id"], "fantasy")

    def test_context_never_leaks_between_conversations(self):
        points, _ = topic_layout(rows(["Pregnancy contraception"], "a") + rows(["Go on"], "b"))
        self.assertEqual(points[1]["topic_id"], "other")
        self.assertFalse(points[1]["topic_context"])

    def test_layout_is_text_based_and_only_populated_regions_are_exported(self):
        a, _ = topic_layout(rows(["A romantic kiss"], "a"))
        b, themes = topic_layout(rows(["A romantic kiss"], "b"))
        self.assertEqual(a, b)
        self.assertEqual(len(themes["regions"]), 1)
        self.assertEqual(themes["regions"][0]["point_ids"], ["b-0"])


if __name__ == "__main__":
    unittest.main()
