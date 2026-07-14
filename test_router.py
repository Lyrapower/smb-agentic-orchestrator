import unittest
from unittest import mock

import router
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

    def test_absent_keywords_do_not_trigger_negation_scans(self) -> None:
        with mock.patch(
            "router._is_negated_keyword",
            wraps=router._is_negated_keyword,
        ) as negated_keyword:
            intent, _ = route_intent("Please cancel my appointment tomorrow")

        self.assertEqual(intent, "cancel")
        checked_keywords = [call.args[2] for call in negated_keyword.call_args_list]
        self.assertEqual(checked_keywords, ["cancel", "cancel"])

    def test_negated_cancel_synonyms_do_not_route_to_cancel(self) -> None:
        examples = (
            "Please don\u2019t cancel my appointment",
            "Please don\u2018t cancel my appointment",
            "Please don\u02bct cancel my appointment",
            "Please don\uff07t cancel my appointment",
            "Please don`t cancel my appointment",
            "Please don\u00b4t cancel my appointment",
            "Please don't ever cancel my appointment",
            "Please do not ever cancel my appointment",
            "Never ever cancel my appointment",
            "Please do not, under any circumstances, cancel my appointment",
            "Please don't, under any circumstances, cancel my appointment",
            "Please do not for any reason cancel my appointment",
            "Please don't ever, ever cancel my appointment",
            "Please never, ever cancel my appointment",
            "Please confirm you won't cancel my appointment",
            "Please confirm you wont cancel my appointment",
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
            "Please tell them not to cancel my appointment",
            "Please make sure not to cancel my appointment",
            "Please remind them never to cancel my appointment",
            "Please do not let anyone cancel my appointment",
            "Please do not allow the office to cancel my appointment",
            "Please do not allow my care team to cancel my appointment",
            "I do not want to remove appointment tomorrow",
            "I do not need anyone to remove appointment tomorrow",
            "I don't, under any circumstances, want to cancel my appointment",
            "I don't ever want to cancel my appointment",
            "I am not trying to cancel my appointment",
            "I'm not looking to cancel my appointment",
            "I am not seeking to cancel my appointment",
            "I never want to cancel my appointment",
            "Please don't want you to cancel or remove appointment",
            "Please don't ask the office to cancel my appointment",
            "Please do not tell the clinic to cancel my appointment",
            "Never authorize anyone to cancel my appointment",
            "Please don't ask Dr. Smith to cancel my appointment",
            "Please do not tell front-desk staff to cancel my appointment",
        )

        for text in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, "no_action")

    def test_no_intent_nouns_do_not_route_to_action(self) -> None:
        examples = (
            "I have no plans to cancel my appointment",
            "I have no plan to cancel my appointment",
            "I have no intention to cancel my appointment",
            "I have no intent to cancel my appointment",
            "I have no desire to cancel my appointment",
            "I have no need to cancel my appointment",
            "There is no plan to cancel my appointment",
            "There are no plans to reschedule my appointment",
            "I have no intention to reschedule my appointment",
            "I have no desire to book a new appointment",
            "No need to schedule an appointment",
            "I have no need for a new appointment",
            "No need for a new appointment",
            "No need to cancel my appointment",
            "No plans to cancel my appointment",
            "No reschedule is needed",
            "No new appointment is needed",
        )

        for text in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, "no_action")

    def test_negated_belief_or_need_do_not_route_to_action(self) -> None:
        examples = (
            "I don't think I need to cancel my appointment",
            "I do not believe I should cancel my appointment",
            "I don't think there is any reason to cancel my appointment",
            "I don't feel there is any need to reschedule my appointment",
            "I don't think I have to move my appointment",
            "I don't think I need to book a new appointment",
            "I do not believe a new appointment is needed",
        )

        for text in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, "no_action")

    def test_negated_reschedule_synonyms_do_not_route_to_reschedule(self) -> None:
        examples = (
            "Please don\u2019t reschedule my appointment",
            "Please don\u02bct reschedule my appointment",
            "Please don\uff07t reschedule my appointment",
            "Please don't reschedule my appointment",
            "Please don't ever reschedule my appointment",
            "Please do not under any circumstances reschedule my appointment",
            "Please don't for any reason reschedule my appointment",
            "I don't want you to move my appointment",
            "I asked not to reschedule my appointment",
            "I am not looking to reschedule my appointment",
            "I'm not seeking to move my appointment",
            "Please do not change time tomorrow",
            "Please do not let the front desk reschedule my appointment",
            "I never want to rebook this appointment",
            "Please don't instruct the office to reschedule my appointment",
            "Please do not request anyone to move my appointment",
            "Never direct the clinic to rebook this appointment",
            "Please don't ask the on-call office manager to reschedule my appointment",
            "Please do not request Nurse Jones, to move my appointment",
            "Please confirm you won't reschedule my appointment",
            "Please confirm you wont move my appointment",
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

    def test_informational_cancellation_questions_do_not_route_to_cancel(self) -> None:
        examples = (
            "What is your cancellation policy?",
            "Can you explain the cancellation fee?",
            "Where can I find the cancellation rules?",
            "What is your cancellation process?",
            "How does cancellation work?",
            "What is the cancellation deadline?",
            "Can you explain the cancellation procedure?",
            "Where can I find cancellation instructions?",
        )

        for text in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, "no_action")

    def test_informational_action_process_questions_do_not_route_to_action(self) -> None:
        examples = (
            "Where can I find instructions to cancel an appointment?",
            "Please tell me the steps to cancel an appointment",
            "What is the process to cancel an appointment?",
            "Can you tell me how to cancel my appointment?",
            "How do I reschedule my appointment?",
            "What are the steps to schedule a new appointment?",
        )

        for text in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, "no_action")

    def test_appointment_cancellation_noun_still_routes_to_cancel(self) -> None:
        examples = (
            "I need cancellation of my appointment",
            "Please process my appointment cancellation",
            "Please start the cancellation process for my appointment",
        )

        for text in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, "cancel")

    def test_negated_cancellation_processing_does_not_route_to_cancel(self) -> None:
        examples = (
            "Please do not process cancellation of my appointment",
            "Please do not process my appointment cancellation",
            "Please do not proceed with cancellation of my appointment",
            "Please do not initiate cancellation of my appointment",
            "Please do not submit a cancellation request for my appointment",
            "Please do not start the cancellation process for my appointment",
            "Please do not complete cancellation of my appointment",
            "Please do not handle my appointment cancellation",
        )

        for text in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, "no_action")

    def test_negated_processing_in_prior_clause_does_not_hide_cancel_request(self) -> None:
        examples = (
            "I do not process claims but please cancel my appointment",
            "I did not process the note, please start cancellation of my appointment",
            "I did not start the paperwork. Please process cancellation of my appointment",
            "I did not complete the form; please initiate cancellation of my appointment",
            "I did not handle that request; instead cancel my appointment",
        )

        for text in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, "cancel")

    def test_direct_action_questions_still_route_to_action(self) -> None:
        examples = (
            ("Can you cancel my appointment?", "cancel"),
            ("Can you reschedule my appointment?", "reschedule"),
            ("Can you schedule a new appointment?", "schedule"),
            ("I'm looking to cancel my appointment", "cancel"),
            ("I am seeking to reschedule my appointment", "reschedule"),
            ("I'm looking to book a new appointment", "schedule"),
            ("I want a new appointment", "schedule"),
            ("I need a new appointment", "schedule"),
            ("Please ask Dr. Smith to cancel my appointment", "cancel"),
            ("Please tell front-desk staff to reschedule my appointment", "reschedule"),
            ("Please request Nurse Jones, to book a new appointment", "schedule"),
            ("No, please cancel my appointment", "cancel"),
            ("No - please reschedule my appointment", "reschedule"),
            ("No, please schedule a new appointment", "schedule"),
        )

        for text, expected_intent in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, expected_intent)

    def test_negated_schedule_synonyms_do_not_route_to_schedule(self) -> None:
        examples = (
            "Please don\u2019t schedule an appointment",
            "Please don\u02bct schedule an appointment",
            "Please don\uff07t schedule an appointment",
            "Please do not schedule an appointment",
            "Please don't ever schedule an appointment",
            "Please don't under any circumstances schedule an appointment",
            "Please do not for any reason book an appointment",
            "I don't want you to book an appointment",
            "I don't want a new appointment",
            "I don't need a new appointment",
            "I do not wish for a new appointment",
            "I'm not looking to schedule an appointment",
            "I am not seeking to book a new appointment",
            "Please make sure not to schedule a new appointment",
            "Please do not permit the scheduling staff to book a new appointment",
            "Never arrange a new appointment",
            "Please don't ask the office to schedule an appointment",
            "Please do not tell anyone to book a new appointment",
            "Never authorize the clinic to arrange a new appointment",
            "Please don't ask Dr. Smith to schedule an appointment",
            "Please do not request Nurse Jones, to book a new appointment",
            "Please confirm you won't schedule an appointment",
            "Please confirm you wont book a new appointment",
        )

        for text in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, "no_action")

    def test_do_not_forget_positive_action_still_routes_to_cancel(self) -> None:
        examples = (
            "Please do not forget to cancel my appointment",
            "Please don't ever forget to cancel my appointment",
            "Please don't under any circumstances forget to cancel my appointment",
            "Please do not allow the office to forget to cancel my appointment",
            "Please don't ask the office to forget to cancel my appointment",
            "Please do not tell anyone to forget to cancel my appointment",
        )

        for text in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, "cancel")

    def test_coordinated_negated_mixed_actions_do_not_route_to_action(self) -> None:
        examples = (
            "I do not want to cancel or reschedule my appointment",
            "Please don't reschedule or cancel my appointment",
            "Please do not cancel my appointment or reschedule it",
            "Please do not reschedule my appointment or cancel it",
            "I do not want to cancel my appointment tomorrow or reschedule it",
            (
                "Please do not cancel my very important annual comprehensive "
                "checkup appointment or reschedule it"
            ),
            (
                "Please do not cancel my very important annual comprehensive "
                "checkup appointment and reschedule it"
            ),
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
        examples = (
            "Please don't cancel my appointment, reschedule it instead",
            "Please don\u2019t cancel my appointment, reschedule it instead",
        )

        for text in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, "reschedule")

    def test_unless_condition_actions_do_not_override_current_request(self) -> None:
        examples = (
            (
                "Please cancel my appointment unless I ask you to reschedule it",
                "cancel",
            ),
            (
                "Please schedule a new appointment unless I call to cancel",
                "schedule",
            ),
            (
                "Please reschedule my appointment unless I ask you to cancel it",
                "reschedule",
            ),
            (
                "Unless I ask you to reschedule, please cancel my appointment",
                "cancel",
            ),
        )

        for text, expected_intent in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, expected_intent)

    def test_negated_request_with_unless_condition_does_not_route_to_condition(self) -> None:
        examples = (
            "Do not cancel my appointment unless I ask you to reschedule it",
            "Do not reschedule my appointment unless I ask you to cancel it",
            "Do not schedule a new appointment unless I ask you to cancel it",
        )

        for text in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, "no_action")

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
            "I cannot make it tomorrow",
            "I can't make it tomorrow",
            "I can not make it tomorrow",
            "I cant make it tomorrow",
        )

        for text in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, "no_action")


if __name__ == "__main__":
    unittest.main()
