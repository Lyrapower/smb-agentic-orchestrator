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
            "I shouldn't cancel my appointment",
            "I shouldnt cancel my appointment",
            "I mustn't cancel my appointment",
            "I mustnt cancel my appointment",
            "I wouldn't cancel my appointment",
            "I wouldnt cancel my appointment",
            "I couldn't cancel my appointment",
            "I couldnt cancel my appointment",
            "I didn't cancel my appointment",
            "I didnt cancel my appointment",
            "I didn't want to cancel my appointment",
            "I didnt want to cancel my appointment",
            "I didn't ask you to cancel my appointment",
            "I didn't tell them to cancel my appointment",
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
            "I don't wanna cancel my appointment",
            "I do not wanna cancel my appointment",
            "I'm not gonna cancel my appointment",
            "I am not gonna cancel my appointment",
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
            "I don't wanna reschedule my appointment",
            "I'm not gonna reschedule my appointment",
            "I don't wanna move my appointment",
            "I'm not gonna move my appointment",
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
            "I shouldn't reschedule my appointment",
            "I mustn't move my appointment",
            "I wouldn't rebook this appointment",
            "I couldn't reschedule my appointment",
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

    def test_appositive_inside_negated_processing_does_not_route_to_action(self) -> None:
        examples = (
            "Please do not process my request, which is to cancel my appointment",
            "Please do not process my request, which is to reschedule my appointment",
            "Please do not process my request, which is to schedule a new appointment",
            "Please do not handle my request, that is to cancel my appointment",
            "Please do not process my request (which is to cancel my appointment)",
            "Please do not process my request (which is to reschedule my appointment)",
            "Please do not process my request - which is to cancel my appointment",
            "Please do not process my request – which is to reschedule my appointment",
            "Please do not process my request — which is to schedule a new appointment",
            "Please do not process my request -- which is to cancel my appointment",
            "Please do not process my request: which is to cancel my appointment",
            "Please do not process my request; which is to cancel my appointment",
            "Please do not process my request [which is to cancel my appointment]",
            "Please do not process my request {which is to cancel my appointment}",
            "Please do not process my request <which is to cancel my appointment>",
            "Please do not process my request... which is to cancel my appointment",
            "Please do not process my request… which is to cancel my appointment",
        )

        for text in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, "no_action")

    def test_negated_processing_in_prior_clause_does_not_hide_cancel_request(self) -> None:
        examples = (
            "I do not process claims but please cancel my appointment",
            "I do not process claims, please cancel my appointment",
            "I do not process claims, cancel my appointment",
            "I do not process claims - cancel my appointment",
            "I do not process claims — cancel my appointment",
            "I do not process claims -- cancel my appointment",
            "I do not process claims: cancel my appointment",
            "I do not process claims; cancel my appointment",
            "Please do not handle this, cancel my appointment",
            "Please do not handle this - cancel my appointment",
            "Please do not handle billing, cancel my appointment",
            "Do not process this request, cancel my appointment",
            "I did not process the note, please start cancellation of my appointment",
            "I did not start the paperwork. Please process cancellation of my appointment",
            "I did not complete the form; please initiate cancellation of my appointment",
            "I did not handle that request; instead cancel my appointment",
        )

        for text in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, "cancel")

    def test_negated_processing_in_prior_clause_does_not_hide_other_actions(self) -> None:
        examples = (
            ("Do not handle the claim, reschedule my appointment", "reschedule"),
            ("Do not process billing, schedule a new appointment", "schedule"),
            ("Please do not submit anything, book a new appointment", "schedule"),
        )

        for text, expected_intent in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, expected_intent)

    def test_negated_delegation_in_prior_clause_does_not_hide_action_request(self) -> None:
        examples = (
            ("I did not make the payment, please cancel my appointment", "cancel"),
            ("I did not ask the office, please cancel my appointment", "cancel"),
            ("I did not ask for help but please cancel my appointment", "cancel"),
            ("Please do not ask questions, cancel my appointment", "cancel"),
            ("Please do not ask Sarah, cancel my appointment", "cancel"),
            ("Please do not tell anyone, cancel my appointment", "cancel"),
            ("I did not tell them. Please reschedule my appointment", "reschedule"),
            ("Do not ask them anything, reschedule my appointment", "reschedule"),
            ("I do not allow cookies, please schedule a new appointment", "schedule"),
        )

        for text, expected_intent in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, expected_intent)

    def test_appositive_inside_negated_delegation_does_not_route_to_action(self) -> None:
        examples = (
            "Please do not ask Sarah, my assistant, to cancel my appointment",
            "Please do not ask Sarah, my assistant, to reschedule my appointment",
            "Please do not ask Sarah, my assistant, to book a new appointment",
            "Please do not tell Dr. Smith, my cardiologist, to cancel my appointment",
            "Please do not ask Sarah (my assistant) to cancel my appointment",
            "Please do not ask Sarah (my assistant) to reschedule my appointment",
            "Please do not tell Dr. Smith (my cardiologist) to cancel my appointment",
            "Please do not ask Sarah - my assistant - to cancel my appointment",
            "Please do not ask Sarah – my assistant – to reschedule my appointment",
            "Please do not ask Sarah — my assistant — to book a new appointment",
            "Please do not ask Sarah—my assistant—to cancel my appointment",
            "Please do not ask Sarah -- my assistant -- to cancel my appointment",
            "Please do not ask Sarah--my assistant--to cancel my appointment",
            "Please do not ask Sarah; my assistant; to cancel my appointment",
            "Please do not ask Sarah; my assistant; to reschedule my appointment",
            'Please do not ask "Sarah" to cancel my appointment',
            'Please do not tell the "front desk" to reschedule my appointment',
            "Please do not ask Sarah [my assistant] to cancel my appointment",
            "Please do not ask Sarah {my assistant} to cancel my appointment",
            "Please do not ask Sarah <my assistant> to cancel my appointment",
            "Please do not ask Sarah <my assistant> to reschedule my appointment",
            "Please do not ask Sarah <my assistant> to book a new appointment",
            "Please do not ask <Sarah> to cancel my appointment",
            "Please do not ask Sarah/my assistant to cancel my appointment",
            "Please do not ask Sarah / my assistant to cancel my appointment",
            "Please do not ask Sarah... to cancel my appointment",
            "Please do not ask Sarah… to cancel my appointment",
            "Please do not ask Mary-Jane to cancel my appointment",
        )

        for text in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, "no_action")

    def test_emphasis_inside_negated_delegation_does_not_route_to_action(self) -> None:
        examples = (
            "Please do not ask the office, under any circumstances, to cancel my appointment",
            "Please do not ask the office, under no circumstances, to cancel my appointment",
            "Please do not tell the office, for any reason, to reschedule my appointment",
            "Please do not tell the office, under no circumstances, to reschedule my appointment",
            "Please do not permit the clinic, ever, to book a new appointment",
            "Please do not permit the clinic, under no circumstances, to book a new appointment",
            "Please do not ask them please to cancel my appointment",
        )

        for text in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, "no_action")

    def test_markdown_emphasis_negation_does_not_route_to_action(self) -> None:
        examples = (
            "Please do *not* cancel my appointment",
            "Please do *not* reschedule my appointment",
            "Please do *not* schedule an appointment",
            "Please do **not** cancel my appointment",
            "Please do _not_ cancel my appointment",
            "Please do __not__ reschedule my appointment",
            "Please do `not` cancel my appointment",
            "Please do `not` reschedule my appointment",
            "Please do ``not`` cancel my appointment",
            "Please do ```not``` cancel my appointment",
            "Please do ```not``` reschedule my appointment",
            "Please do ```not``` schedule an appointment",
            "Please ```do not``` cancel my appointment",
            "Please do ````not```` cancel my appointment",
            "Please do ~~not~~ cancel my appointment",
            "Please do ~~not~~ reschedule my appointment",
            "Please do ~~not~~ schedule an appointment",
            "Please ~~do not~~ cancel my appointment",
            "Please do ~not~ cancel my appointment",
            "Please *do not* cancel my appointment",
            "Please _do not_ book a new appointment",
            "I do *not* want to cancel my appointment",
            "Please do *not* ask Sarah to cancel my appointment",
            "Please do not ask `Sarah` to cancel my appointment",
            "Please don`t cancel my appointment",
        )

        for text in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, "no_action")

    def test_html_emphasis_negation_does_not_route_to_action(self) -> None:
        examples = (
            "Please do <b>not</b> cancel my appointment",
            "Please do <b>not</b> reschedule my appointment",
            "Please do <b>not</b> schedule an appointment",
            "Please do <em>not</em> cancel my appointment",
            "Please do <i>not</i> reschedule my appointment",
            "Please do <strong>not</strong> cancel my appointment",
            "Please <b>do not</b> cancel my appointment",
            "Please <em>do not</em> schedule an appointment",
            "Please do<span>not</span> cancel my appointment",
            "Please do <span>not</span> cancel my appointment",
            "Please do <b class=\"x\">not</b> cancel my appointment",
            "Please do <B>NOT</B> cancel my appointment",
            "Please do not ask Sarah <b>my assistant</b> to cancel my appointment",
            "Please do not ask Sarah <em>my assistant</em> to reschedule my appointment",
            "I do <b>not</b> want to cancel my appointment",
        )

        for text in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, "no_action")

    def test_zero_width_glue_negation_does_not_route_to_action(self) -> None:
        examples = (
            "Please do not\u200bcancel my appointment",
            "Please do not\u200b cancel my appointment",
            "Please do\u200bnot cancel my appointment",
            "Please don't\u200bcancel my appointment",
            "Please don't\u200bwant to cancel my appointment",
            "Please never\u200bcancel my appointment",
            "Please do not ever\u200bcancel my appointment",
            "Please do not\u200breschedule my appointment",
            "Please do not\u200bschedule an appointment",
            "Please do not\u200ccancel my appointment",
            "Please do not\u200dcancel my appointment",
            "Please do not\ufeffcancel my appointment",
            "Please do not\u2060cancel my appointment",
            "Please do\u200b not\u200b\u200bcancel my appointment",
            # Bidirectional marks and isolates from email/chat RTL paste.
            "Please do not\u200ecancel my appointment",
            "Please do not\u200fcancel my appointment",
            "Please do not\u202acancel my appointment",
            "Please do not\u202bcancel my appointment",
            "Please do not\u202ccancel my appointment",
            "Please do not\u202dcancel my appointment",
            "Please do not\u202ecancel my appointment",
            "Please do not\u2066cancel my appointment",
            "Please do not\u2067cancel my appointment",
            "Please do not\u2068cancel my appointment",
            "Please do not\u2069cancel my appointment",
            "Please do not\u2061cancel my appointment",
            "Please do not\u2062cancel my appointment",
            "Please don't\u200ewant to cancel my appointment",
            "Please do not\u200ereschedule my appointment",
            "Please do not\u200eschedule an appointment",
        )

        for text in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, "no_action")

    def test_combining_mark_glue_negation_does_not_route_to_action(self) -> None:
        examples = (
            # Combining diacritics from mobile/OCR paste on negation tokens.
            "Please do not\u0301 cancel my appointment",
            "Please do not\u0301cancel my appointment",
            "Please do not\u0327 cancel my appointment",
            "Please don't\u0301 cancel my appointment",
            "Please never\u0301 cancel my appointment",
            # Emoji/text variation selectors (Mn) after negation.
            "Please do not\ufe0f cancel my appointment",
            "Please do not\ufe0fcancel my appointment",
            "Please don't\ufe0f want to cancel my appointment",
            # Same class of glue on other appointment actions.
            "Please do not\u0301 reschedule my appointment",
            "Please do not\ufe0f schedule an appointment",
            # Precomposed accented letters that NFD-fold into Mn marks.
            "Please do n\u00f3t cancel my appointment",
            "Please do n\u00f2t cancel my appointment",
        )

        for text in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, "no_action")

    def test_fullwidth_and_soft_hyphen_negation_does_not_route_to_action(self) -> None:
        examples = (
            # Fullwidth Latin from CJK/IME paste on negation tokens.
            "Please do \uff4e\uff4f\uff54 cancel my appointment",
            "Please do \uff4e\uff4f\uff54 reschedule my appointment",
            "Please do \uff4e\uff4f\uff54 schedule an appointment",
            "Please do not \uff43\uff41\uff4e\uff43\uff45\uff4c my appointment",
            "Please \uff44\uff4f\uff4e\uff07\uff54 cancel my appointment",
            # Soft-hyphen between do/not strips to "donot"; keep it negated.
            "Please do\u00adnot cancel my appointment",
            "Please do\u00adnot reschedule my appointment",
            "Please do\u00adnot schedule an appointment",
            "Please donot cancel my appointment",
            # Fullwidth punctuation appositives fold via NFKC.
            "Please do not ask Sarah\uff0c my assistant\uff0c to cancel my appointment",
            'Please do not ask \uff02Sarah\uff02 to cancel my appointment',
        )

        for text in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, "no_action")

    def test_guillemet_and_cjk_bracket_negation_does_not_route_to_action(self) -> None:
        examples = (
            # European guillemets / low quotes around delegated actors.
            "Please do not ask \u00abSarah\u00bb to cancel my appointment",
            "Please do not ask \u2039Sarah\u203a to cancel my appointment",
            "Please do not ask \u201eSarah\u201c to cancel my appointment",
            "Please do not ask \u201aSarah\u2018 to cancel my appointment",
            "Please do not ask \u201fSarah\u201f to cancel my appointment",
            "Please do not ask \u00abSarah, my assistant\u00bb to cancel my appointment",
            # CJK corner / lenticular / angle brackets from chat paste.
            "Please do not ask \u300cSarah\u300d to cancel my appointment",
            "Please do not ask \u300eSarah\u300f to cancel my appointment",
            "Please do not ask \u3010Sarah\u3011 to cancel my appointment",
            "Please do not ask \u300aSarah\u300b to cancel my appointment",
            "Please do not ask \u3008Sarah\u3009 to cancel my appointment",
            "Please do not ask \u3016Sarah\u3017 to cancel my appointment",
            "Please do not ask \u3014Sarah\u3015 to cancel my appointment",
            # Halfwidth corners NFKC-fold into CJK corner brackets.
            "Please do not ask \uff62Sarah\uff63 to cancel my appointment",
            "Please do not ask \u300cSarah\u300d to reschedule my appointment",
            "Please do not ask \u00abSarah\u00bb to schedule an appointment",
        )

        for text in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, "no_action")

    def test_direct_action_questions_still_route_to_action(self) -> None:
        examples = (
            ("Can you cancel my appointment?", "cancel"),
            ("Can you reschedule my appointment?", "reschedule"),
            ("Can you schedule a new appointment?", "schedule"),
            ("I'm looking to cancel my appointment", "cancel"),
            ("I am seeking to reschedule my appointment", "reschedule"),
            ("I'm looking to book a new appointment", "schedule"),
            ("I wanna cancel my appointment", "cancel"),
            ("I wanna reschedule my appointment", "reschedule"),
            ("I'm gonna cancel my appointment", "cancel"),
            ("I'm gonna schedule an appointment", "schedule"),
            ("I want a new appointment", "schedule"),
            ("I need a new appointment", "schedule"),
            ("Please ask Dr. Smith to cancel my appointment", "cancel"),
            ("Please tell front-desk staff to reschedule my appointment", "reschedule"),
            ("Please request Nurse Jones, to book a new appointment", "schedule"),
            ("No, please cancel my appointment", "cancel"),
            ("No - please reschedule my appointment", "reschedule"),
            ("No, please schedule a new appointment", "schedule"),
            ("Please <b>cancel</b> my appointment", "cancel"),
            ("Please <em>reschedule</em> my appointment", "reschedule"),
            ("I want to <strong>schedule</strong> an appointment", "schedule"),
            ("Please\u200bcancel my appointment", "cancel"),
            ("Please cancel\u200b my appointment", "cancel"),
            ("Please\u200breschedule my appointment", "reschedule"),
            ("Please\u200bschedule an appointment", "schedule"),
            ("Please\u200ecancel my appointment", "cancel"),
            ("Please cancel\u200e my appointment", "cancel"),
            ("Please\u2066reschedule my appointment", "reschedule"),
            ("Please\u202aschedule an appointment", "schedule"),
            ("Please can\u00adcel my appointment", "cancel"),
            ("Please can\u0301cel my appointment", "cancel"),
            ("Please ca\u0144cel my appointment", "cancel"),
            ("Please\ufe0f cancel my appointment", "cancel"),
            ("Please cancel\u0301 my appointment", "cancel"),
            ("Please reschedule\u0301 my appointment", "reschedule"),
            ("Please schedule\ufe0f an appointment", "schedule"),
            ("Please \uff43\uff41\uff4e\uff43\uff45\uff4c my appointment", "cancel"),
            ("Please \uff52\uff45\uff53\uff43\uff48\uff45\uff44\uff55\uff4c\uff45 my appointment", "reschedule"),
            ("Please \uff53\uff43\uff48\uff45\uff44\uff55\uff4c\uff45 an appointment", "schedule"),
            ('Please ask \uff02Sarah\uff02 to cancel my appointment', "cancel"),
            ("Please ask \u00abSarah\u00bb to cancel my appointment", "cancel"),
            ("Please ask \u300cSarah\u300d to reschedule my appointment", "reschedule"),
            ("Please ask \u3010Sarah\u3011 to schedule an appointment", "schedule"),
            ("Please ask \uff62Sarah\uff63 to cancel my appointment", "cancel"),
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
            "I don't wanna schedule an appointment",
            "I'm not gonna schedule an appointment",
            "I don't wanna book a new appointment",
            "I'm not gonna book a new appointment",
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
            "I shouldn't schedule an appointment",
            "I mustn't book a new appointment",
            "I wouldn't schedule an appointment right now",
            "I couldn't arrange a new appointment",
        )

        for text in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, "no_action")

    def test_modal_contraction_negations_do_not_route_to_action(self) -> None:
        examples = (
            "I shouldn't cancel my appointment",
            "I mustn't cancel my appointment",
            "I wouldn't cancel my appointment",
            "I couldn't cancel my appointment",
            "I shouldn't reschedule my appointment",
            "I wouldn't schedule an appointment right now",
        )

        for text in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, "no_action")

    def test_past_tense_didnt_negations_do_not_route_to_action(self) -> None:
        examples = (
            "I didn't cancel my appointment",
            "I didnt cancel my appointment",
            "I didn’t cancel my appointment",
            "I didn't want to cancel my appointment",
            "I didn't ask you to cancel my appointment",
            "I didn't tell them to reschedule",
            "I didn't want to schedule an appointment",
            "I didn't book an appointment",
        )

        for text in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, "no_action")

    def test_past_tense_wasnt_werent_negations_do_not_route_to_action(self) -> None:
        examples = (
            "I wasn't going to cancel my appointment",
            "I wasnt going to cancel my appointment",
            "I wasn’t going to cancel my appointment",
            "I wasn't gonna cancel my appointment",
            "I wasn't planning to cancel my appointment",
            "I wasn't trying to cancel my appointment",
            "We weren't going to cancel our appointment",
            "We werent going to cancel our appointment",
            "We weren’t going to reschedule",
            "I wasn't going to schedule an appointment",
            "I wasn't planning to book an appointment",
        )

        for text in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, "no_action")

    def test_present_tense_isnt_arent_negations_do_not_route_to_action(self) -> None:
        examples = (
            "He isn't going to cancel my appointment",
            "He isnt going to cancel my appointment",
            "He isn’t going to cancel my appointment",
            "He isn't gonna cancel my appointment",
            "She isn't planning to cancel my appointment",
            "She isn't trying to cancel my appointment",
            "They aren't going to cancel my appointment",
            "They arent going to cancel my appointment",
            "They aren’t going to reschedule",
            "They aren't gonna cancel my appointment",
            "The office isn't going to cancel my appointment",
            "He isn't going to schedule an appointment",
            "They aren't planning to book an appointment",
        )

        for text in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, "no_action")

    def test_dialect_aint_negations_do_not_route_to_action(self) -> None:
        examples = (
            "He ain't going to cancel my appointment",
            "He aint going to cancel my appointment",
            "He ain’t going to cancel my appointment",
            "He ain't gonna cancel my appointment",
            "She ain't planning to cancel my appointment",
            "She ain't trying to cancel my appointment",
            "They ain't going to cancel my appointment",
            "They aint going to cancel my appointment",
            "They ain’t going to reschedule",
            "They ain't gonna cancel my appointment",
            "The office ain't going to cancel my appointment",
            "He ain't going to schedule an appointment",
            "They ain't planning to book an appointment",
            "I ain't gonna cancel my appointment",
            "Please confirm you ain't going to cancel my appointment",
        )

        for text in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, "no_action")

    def test_progressive_delegated_gerund_negations_do_not_route_to_action(
        self,
    ) -> None:
        examples = (
            "I'm not asking you to cancel my appointment",
            "I am not asking you to cancel my appointment",
            "They aren't asking you to cancel my appointment",
            "They aren't asking you to cancel",
            "She isn't telling you to cancel my appointment",
            "I'm not telling you to cancel my appointment",
            "I'm not instructing you to cancel my appointment",
            "I'm not requesting you to cancel my appointment",
            "I'm not advising you to cancel my appointment",
            "I'm not asking you to reschedule my appointment",
            "I'm not asking you to schedule an appointment",
            "I wasn't asking you to cancel my appointment",
            "He isn't asking you to cancel my appointment",
            "He ain't asking you to cancel my appointment",
            "They aren't telling you to cancel my appointment",
            "Please confirm you are not asking me to cancel my appointment",
        )

        for text in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, "no_action")

    def test_allowed_to_negations_do_not_route_to_action(self) -> None:
        examples = (
            "I'm not allowed to cancel my appointment",
            "I am not allowed to cancel my appointment",
            "I'm not permitted to cancel my appointment",
            "I'm not allowed to reschedule my appointment",
            "I'm not allowed to schedule an appointment",
            "I wasn't allowed to cancel my appointment",
            "He isn't allowed to cancel my appointment",
            "He ain't allowed to cancel my appointment",
            "They aren't permitted to cancel my appointment",
            "I'm not really allowed to cancel my appointment",
            "I'm not even permitted to cancel my appointment",
            "Please confirm you are not allowed to cancel my appointment",
        )

        for text in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, "no_action")

    def test_supposed_to_negations_do_not_route_to_action(self) -> None:
        examples = (
            "I'm not supposed to cancel my appointment",
            "I am not supposed to cancel my appointment",
            "I'm not willing to cancel my appointment",
            "I'm not ready to cancel my appointment",
            "I'm not prepared to cancel my appointment",
            "I'm not supposed to reschedule my appointment",
            "I'm not supposed to schedule an appointment",
            "I wasn't supposed to cancel my appointment",
            "He isn't supposed to cancel my appointment",
            "He ain't supposed to cancel my appointment",
            "They aren't willing to cancel my appointment",
            "I'm not really supposed to cancel my appointment",
            "I'm not even willing to cancel my appointment",
            "Please confirm you are not supposed to cancel my appointment",
        )

        for text in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, "no_action")

    def test_about_to_negations_do_not_route_to_action(self) -> None:
        examples = (
            "I'm not about to cancel my appointment",
            "I am not about to cancel my appointment",
            "I'm not about to reschedule my appointment",
            "I'm not about to schedule an appointment",
            "I wasn't about to cancel my appointment",
            "He isn't about to cancel my appointment",
            "He ain't about to cancel my appointment",
            "They aren't about to cancel my appointment",
            "I'm not really about to cancel my appointment",
            "I'm not even about to cancel my appointment",
            "Please confirm you are not about to cancel my appointment",
        )

        for text in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, "no_action")

    def test_no_longer_negations_do_not_route_to_action(self) -> None:
        examples = (
            "I no longer want to cancel my appointment",
            "I no longer need to cancel my appointment",
            "I no longer wish to cancel my appointment",
            "I no longer plan to cancel my appointment",
            "I no longer want to reschedule my appointment",
            "I no longer want to schedule an appointment",
            "I'm no longer going to cancel my appointment",
            "I am no longer going to cancel my appointment",
            "I'm no longer gonna cancel my appointment",
            "I no longer want a cancellation",
            "Please confirm you no longer want to cancel my appointment",
        )

        for text in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, "no_action")

    def test_perfect_tense_havent_hadnt_hasnt_negations_do_not_route_to_action(
        self,
    ) -> None:
        examples = (
            "I haven't wanted to cancel my appointment",
            "I havent wanted to cancel my appointment",
            "I haven’t wanted to cancel my appointment",
            "I haven't planned to cancel my appointment",
            "I hadn't planned to cancel my appointment",
            "I hadnt planned to cancel my appointment",
            "I hadn’t wanted to cancel my appointment",
            "She hasn't asked to cancel my appointment",
            "She hasnt asked to cancel my appointment",
            "She hasn’t asked to cancel my appointment",
            "He hasn't told you to cancel my appointment",
            "I haven't wanted to reschedule my appointment",
            "I haven't planned to book an appointment",
        )

        for text in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, "no_action")

    def test_affirmative_past_tense_still_routes_to_action(self) -> None:
        examples = (
            ("I did cancel my appointment", "cancel"),
            ("Please cancel my appointment, I didn't want Tuesday", "cancel"),
            ("I did want to reschedule my appointment", "reschedule"),
            ("I did book an appointment", "schedule"),
            ("I was going to cancel my appointment", "cancel"),
            ("I was planning to cancel my appointment", "cancel"),
            ("Please cancel my appointment, I wasn't sure about Tuesday", "cancel"),
            ("Please cancel my appointment, I wasn't going to make it", "cancel"),
            ("I wasn't available Tuesday, please cancel", "cancel"),
            ("Please reschedule, I wasn't able to come", "reschedule"),
            ("Please cancel my appointment, I haven't been able to make it", "cancel"),
            ("I haven't been feeling well, please cancel", "cancel"),
            ("She hasn't confirmed yet, please cancel", "cancel"),
            ("I hadn't realized the conflict, please cancel", "cancel"),
            ("I haven't canceled yet, please cancel", "cancel"),
            ("I haven't heard back, please reschedule", "reschedule"),
            ("I have wanted to cancel my appointment", "cancel"),
            ("She has asked to cancel my appointment", "cancel"),
            ("He is going to cancel my appointment", "cancel"),
            ("They are going to cancel my appointment", "cancel"),
            ("Please cancel my appointment, he isn't available Tuesday", "cancel"),
            ("Please cancel my appointment, they aren't able to make it", "cancel"),
            ("They aren't available Tuesday, please cancel", "cancel"),
            ("He isn't coming, please cancel", "cancel"),
            ("This isn't working, please cancel", "cancel"),
            ("She isn't confirmed yet, please cancel", "cancel"),
            ("Please reschedule, they aren't able to come", "reschedule"),
            ("Please cancel my appointment, he ain't available Tuesday", "cancel"),
            ("Please cancel my appointment, they ain't able to make it", "cancel"),
            ("They ain't available Tuesday, please cancel", "cancel"),
            ("He ain't coming, please cancel", "cancel"),
            ("This ain't working, please cancel", "cancel"),
            ("Please reschedule, they ain't able to come", "reschedule"),
            ("I want to cancel my appointment", "cancel"),
            (
                "Please cancel my appointment, I no longer need Tuesday",
                "cancel",
            ),
            ("I no longer need Tuesday, please cancel", "cancel"),
            ("I no longer want Tuesday, please cancel", "cancel"),
            ("I no longer can make it, please cancel", "cancel"),
            ("Please reschedule, I no longer need that time", "reschedule"),
            ("I'm about to cancel my appointment", "cancel"),
            ("I am about to cancel my appointment", "cancel"),
            ("I'm about to reschedule my appointment", "reschedule"),
            ("Please cancel my appointment, I'm not about to make it", "cancel"),
            ("I'm not about to make it, please cancel", "cancel"),
            ("I'm not about Tuesday, please cancel", "cancel"),
            ("I'm supposed to cancel my appointment", "cancel"),
            ("I am willing to cancel my appointment", "cancel"),
            ("I'm ready to cancel my appointment", "cancel"),
            ("I'm prepared to reschedule my appointment", "reschedule"),
            (
                "Please cancel my appointment, I'm not supposed to make it",
                "cancel",
            ),
            ("I'm not supposed to make it, please cancel", "cancel"),
            ("I'm not ready Tuesday, please cancel", "cancel"),
            ("I'm not willing to wait, please cancel", "cancel"),
            ("I'm not prepared for Tuesday, please cancel", "cancel"),
            ("I'm asking you to cancel my appointment", "cancel"),
            ("They are asking you to cancel my appointment", "cancel"),
            ("Please ask them to cancel my appointment", "cancel"),
            (
                "Please cancel my appointment, I'm not asking about Tuesday",
                "cancel",
            ),
            ("I'm not asking about Tuesday, please cancel", "cancel"),
            ("I'm not telling you about Tuesday, please cancel", "cancel"),
            ("I'm allowed to cancel my appointment", "cancel"),
            ("I'm permitted to cancel my appointment", "cancel"),
            (
                "Please cancel my appointment, I'm not allowed to make it",
                "cancel",
            ),
            ("I'm not allowed to make it, please cancel", "cancel"),
            ("I'm not allowed Tuesday, please cancel", "cancel"),
            ("I'm not permitted to wait, please cancel", "cancel"),
            ("I'm not permitted Tuesday, please cancel", "cancel"),
        )

        for text, expected_intent in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, expected_intent)

    def test_adverb_bridge_negations_do_not_route_to_action(self) -> None:
        examples = (
            "I don't really want to cancel my appointment",
            "I don't actually want to cancel my appointment",
            "I don't even want to cancel my appointment",
            "I don't particularly want to reschedule my appointment",
            "I don't currently plan to cancel my appointment",
            "I'm not really going to cancel my appointment",
            "I'm not really gonna cancel my appointment",
            "I don't really wanna cancel my appointment",
            "I don't even wanna reschedule my appointment",
            "I don't really want a cancellation",
            "I don't really think I need to cancel my appointment",
            "I don't actually believe we should cancel my appointment",
            "I'm not really looking to schedule an appointment",
            "I don't particularly wish to schedule an appointment",
        )

        for text in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, "no_action")

    def test_affirmative_adverb_bridges_still_route_to_action(self) -> None:
        examples = (
            ("I really want to cancel my appointment", "cancel"),
            ("I actually want to reschedule my appointment", "reschedule"),
            ("I'm really going to cancel my appointment", "cancel"),
            ("I really wanna schedule an appointment", "schedule"),
            (
                "Please cancel my appointment, I don't really want Tuesday",
                "cancel",
            ),
            ("Please don't cancel, I really want to reschedule", "reschedule"),
            # Rejected slot preference must not negate a later explicit action.
            ("I don't want Tuesday, please cancel", "cancel"),
            ("I don't really want Tuesday, please cancel", "cancel"),
            (
                "I don't actually want Tuesday, please cancel my appointment",
                "cancel",
            ),
            ("I don't need Tuesday, please cancel", "cancel"),
            ("I don't want that time, please reschedule", "reschedule"),
        )

        for text, expected_intent in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, expected_intent)

    def test_affirmative_modals_still_route_to_action(self) -> None:
        examples = (
            ("I should cancel my appointment", "cancel"),
            ("I must cancel my appointment", "cancel"),
            ("I would like to cancel my appointment", "cancel"),
            ("I should reschedule my appointment", "reschedule"),
            ("I must schedule an appointment", "schedule"),
        )

        for text, expected_intent in examples:
            with self.subTest(text=text):
                intent, _ = route_intent(text)

                self.assertEqual(intent, expected_intent)

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
            "I don't wanna cancel or reschedule my appointment",
            "I'm not gonna cancel or reschedule my appointment",
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
