import unittest

from router import route_intent


class RouterTests(unittest.TestCase):
    def test_negated_cancel_with_reschedule_routes_to_reschedule(self) -> None:
        examples = (
            "Please don't cancel my appointment, just reschedule it",
            "Please don't cancel my appointment, reschedule it instead",
        )

        for text in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, "reschedule")

    def test_cannot_make_with_reschedule_routes_to_reschedule(self) -> None:
        intent, _ = route_intent("I cannot make it tomorrow, please reschedule")

        self.assertEqual(intent, "reschedule")

    def test_cannot_make_with_negated_cancel_does_not_route_to_cancel(self) -> None:
        examples = (
            "I cannot make it tomorrow, please don't cancel",
            "I can't make it, do not cancel my appointment",
            "I cannot make it tomorrow, please don't remove appointment",
        )

        for text in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, "no_action")

    def test_cannot_make_with_positive_cancel_still_routes_to_cancel(self) -> None:
        intent, _ = route_intent("I cannot make it tomorrow, please cancel")

        self.assertEqual(intent, "cancel")

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

                self.assertEqual(intent, "no_action")

    def test_remove_appointment_routes_to_cancel_not_reschedule(self) -> None:
        intent, _ = route_intent("Please remove appointment tomorrow")

        self.assertEqual(intent, "cancel")

    def test_move_inside_remove_does_not_route_to_reschedule(self) -> None:
        intent, _ = route_intent("Please do not remove appointment tomorrow")

        self.assertEqual(intent, "no_action")

    def test_coordinated_negated_cancel_synonyms_do_not_route_to_cancel(self) -> None:
        examples = (
            "Please do not cancel or remove appointment tomorrow",
            "Don't call off or remove appointment tomorrow",
            "Do not drop appointment or remove appointment tomorrow",
        )

        for text in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, "no_action")

    def test_negated_intent_to_cancel_does_not_route_to_cancel(self) -> None:
        examples = (
            "I don't want to cancel my appointment",
            "I don't want you to cancel my appointment",
            "I do not want to remove appointment tomorrow",
            "I do not need anyone to remove appointment tomorrow",
            "I am not trying to cancel my appointment",
            "I never want to cancel my appointment",
            "Please don't want you to cancel or remove appointment",
        )

        for text in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, "no_action")

    def test_negated_reschedule_synonyms_do_not_route_to_reschedule(self) -> None:
        examples = (
            "Please don't reschedule my appointment",
            "I don't want you to move my appointment",
            "Please do not change time tomorrow",
            "I never want to rebook this appointment",
        )

        for text in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, "no_action")

    def test_cancel_with_negated_reschedule_routes_to_cancel(self) -> None:
        examples = (
            "Please cancel, don't reschedule",
            "I don't want to reschedule my appointment, please cancel it",
        )

        for text in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, "cancel")

    def test_negated_schedule_synonyms_do_not_route_to_schedule(self) -> None:
        examples = (
            "Please do not schedule an appointment",
            "I don't want you to book an appointment",
            "Never arrange a new appointment",
        )

        for text in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, "no_action")

    def test_coordinated_negated_mixed_actions_do_not_route_to_action(self) -> None:
        examples = (
            "I do not want to cancel or reschedule my appointment",
            "Please don't reschedule or cancel my appointment",
            "Please do not cancel my appointment or reschedule it",
            "Please do not reschedule my appointment or cancel it",
            "I do not want to cancel my appointment tomorrow or reschedule it",
            "Please do not schedule, cancel, or reschedule anything",
            "Please do not cancel, reschedule, nor book an appointment",
            "Please do not cancel and reschedule my appointment",
            "Please do not schedule and cancel anything",
        )

        for text in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, "no_action")

    def test_negated_cancel_with_reschedule_instead_still_routes_to_reschedule(self) -> None:
        intent, _ = route_intent("Please don't cancel my appointment, reschedule it instead")

        self.assertEqual(intent, "reschedule")

    def test_comma_only_negated_mixed_action_lists_do_not_route_to_action(self) -> None:
        examples = (
            "Please do not schedule, cancel, reschedule",
            "Please don't schedule, cancel, reschedule anything",
        )

        for text in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, "no_action")

    def test_empty_text_does_not_default_to_schedule(self) -> None:
        for text in ("", "   "):
            with self.subTest(text=repr(text)):
                intent, _ = route_intent(text)

                self.assertEqual(intent, "no_action")

    def test_unmatched_text_does_not_default_to_schedule(self) -> None:
        examples = (
            "What time is my appointment tomorrow?",
            "Hello, can you help me?",
            "I can not make it tomorrow",
            "I cant make it tomorrow",
        )

        for text in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, "no_action")


if __name__ == "__main__":
    unittest.main()
