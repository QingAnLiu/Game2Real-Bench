from __future__ import annotations

import unittest

from game2real.evaluator import evaluate_record


class RealWorldEvaluatorTests(unittest.TestCase):
    def record(self, evaluator_type: str, answer, accepted=None):
        return {
            "partition": "real_world",
            "evaluator_type": evaluator_type,
            "answer": answer,
            "accepted_answers": accepted or f'["{answer}"]',
        }

    def test_mcq_single(self):
        self.assertTrue(evaluate_record("The answer is C.", self.record("mcq_single", "C")).correct)

    def test_exact_short_text(self):
        self.assertTrue(evaluate_record("YES.", self.record("exact_short_text", "yes")).correct)

    def test_numeric(self):
        self.assertTrue(evaluate_record("1,200", self.record("numeric", "1200")).correct)

    def test_json_canonical(self):
        record = self.record("json_canonical", '{}', '[{"b":2,"a":"X"}]')
        self.assertTrue(evaluate_record('{"a":"x","b":2}', record).correct)

    def test_json_unordered_set(self):
        record = self.record("json_unordered_set", '[]', '[["A","B"]]')
        self.assertTrue(evaluate_record('["b","a"]', record).correct)

    def test_ranking(self):
        record = self.record("ranking", '[]', '[["A","B","C"]]')
        self.assertTrue(evaluate_record('["a","b","c"]', record).correct)

    def test_route_plan(self):
        route = '[{"route_name":"Blue","departure_stop":"East Station","arrival_stop":"West Station","num_via_stops":2,"via_stops":["A","B"]}]'
        record = self.record("route_plan", route, f'[{route}]')
        self.assertTrue(evaluate_record(route, record).correct)


class ToyEvaluatorTests(unittest.TestCase):
    def test_single_choice(self):
        record = {"partition": "toy_game", "question_type": "single_choice", "answer": "D"}
        self.assertTrue(evaluate_record("Answer: D", record).correct)

    def test_open_mapping_accepts_key_order_and_numeric_equivalence(self):
        record = {
            "partition": "toy_game",
            "question_type": "open",
            "answer": "R1C1=1; R2C2=2",
            "accepted_answers": '[{"R1C1":1,"R2C2":2}]',
            "evaluator_payload": '{"open_schema_id":"example_map"}',
        }
        self.assertTrue(evaluate_record("R2C2=2.0; R1C1=1", record).correct)


if __name__ == "__main__":
    unittest.main()
