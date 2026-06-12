import unittest

from router import route_intent


class RouterTests(unittest.TestCase):
    def test_negated_cancel_with_reschedule_routes_to_reschedule(self) -> None:
        intent, _ = route_intent("Please don't cancel my appointment, just reschedule it")

        self.assertEqual(intent, "reschedule")

    def test_cannot_make_with_reschedule_routes_to_reschedule(self) -> None:
        intent, _ = route_intent("I cannot make it tomorrow, please reschedule")

        self.assertEqual(intent, "reschedule")

    def test_plain_cancel_still_routes_to_cancel(self) -> None:
        intent, _ = route_intent("Please cancel my appointment tomorrow")

        self.assertEqual(intent, "cancel")

    def test_negated_cancel_synonyms_do_not_route_to_cancel(self) -> None:
        examples = (
            "Please don't call off my appointment",
            "Please do not drop appointment tomorrow",
            "Please never remove appointment from my calendar",
        )

        for text in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, "schedule")

    def test_remove_appointment_routes_to_cancel_not_reschedule(self) -> None:
        intent, _ = route_intent("Please remove appointment tomorrow")

        self.assertEqual(intent, "cancel")

    def test_move_inside_remove_does_not_route_to_reschedule(self) -> None:
        intent, _ = route_intent("Please do not remove appointment tomorrow")

        self.assertEqual(intent, "schedule")

    def test_coordinated_negated_cancel_synonyms_do_not_route_to_cancel(self) -> None:
        examples = (
            "Please do not cancel or remove appointment tomorrow",
            "Don't call off or remove appointment tomorrow",
            "Do not drop appointment or remove appointment tomorrow",
        )

        for text in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, "schedule")

    def test_negated_intent_to_cancel_does_not_route_to_cancel(self) -> None:
        examples = (
            "I don't want to cancel my appointment",
            "I do not want to remove appointment tomorrow",
            "I am not trying to cancel my appointment",
            "I never want to cancel my appointment",
        )

        for text in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, "schedule")


if __name__ == "__main__":
    unittest.main()
