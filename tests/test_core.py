import unittest
from inno_medical_records_nlp import core


class TestNlp(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(core.normalize("ديابت  ۲"), "دیابت 2")
        self.assertEqual(core.normalize("نارسایی‌قلبی"), "نارسایی قلبی")

    def test_extracts_arabic_variants_and_zwnj(self):
        e = core.extract("سابقه ديابت و نارسایی‌قلبی دارد")
        self.assertEqual({x["text"] for x in e if x["type"] == "condition"}, {"دیابت", "نارسایی قلبی"})

    def test_negation_before_and_after(self):
        e = {x["text"]: x for x in core.extract("بدون تب. سرفه ندارد. تنگی نفس دارد.")}
        self.assertTrue(e["تب"]["negated"]); self.assertTrue(e["سرفه"]["negated"]); self.assertFalse(e["تنگی نفس"]["negated"])

    def test_negated_condition_gets_no_code(self):
        self.assertEqual(core.suggest_codes(core.extract("دیابت رد شد.")), [])

    def test_icd_and_review_flag(self):
        s = core.suggest_codes(core.extract("سابقه دیابت دارد."))
        self.assertEqual(s[0]["code"], "E11"); self.assertFalse(s[0]["needs_review"])
        s = core.suggest_codes(core.extract("سابقه دیابتت دارد."))  # one-character typo
        self.assertEqual(s[0]["code"], "E11"); self.assertTrue(s[0]["needs_review"])

    def test_no_entities_in_unrelated_text(self):
        self.assertEqual(core.extract("امروز هوا خوب است."), [])
