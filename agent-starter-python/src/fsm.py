"""
FSM for Arte de Viver Brasil Voice Agent.

Registration flow (field-by-field via save_field tool):
  START
    ↓ data["chosen_date"] (intent="book" triggers move to REG_ASK_DATE first)
  REG_ASK_DATE → REG_ASK_NAME → REG_ASK_PHONE → REG_ASK_EMAIL
    → REG_ASK_BIRTH_YEAR → REG_ASK_NEIGHBORHOOD → REG_ASK_CITY
    → REG_ASK_CONSENT → REG_CONFIRM → REG_DONE

Manage flows:
  START → MANAGE_ASK_PHONE → MANAGE_LIST | MANAGE_CANCEL_CONFIRM → START
"""

from enum import Enum, auto
from typing import Optional, Dict, Any, List
from datetime import datetime
import copy
import logging

_log = logging.getLogger("fsm")


class State(Enum):
    START                = auto()
    REG_ASK_DATE         = auto()
    REG_ASK_NAME         = auto()
    REG_ASK_PHONE        = auto()
    REG_ASK_EMAIL        = auto()
    REG_ASK_BIRTH_YEAR   = auto()
    REG_ASK_NEIGHBORHOOD = auto()
    REG_ASK_CITY         = auto()
    REG_ASK_CONSENT      = auto()
    REG_CONFIRM          = auto()
    REG_DONE             = auto()
    MANAGE_ASK_PHONE     = auto()
    MANAGE_LIST          = auto()
    MANAGE_CANCEL_CONFIRM= auto()
    MANAGE_RESCHEDULE_DATE = auto()
    MANAGE_RESCHEDULE_REASON = auto()
    MANAGE_RESCHEDULE_DONE = auto()


class ConversationContext:
    def __init__(self):
        self.chosen_date:      Optional[str]  = None
        self.chosen_class_iso: Optional[str]  = None
        # Reschedule-specific — never shared with registration fields
        self.reschedule_date:      Optional[str]  = None
        self.reschedule_class_iso: Optional[str]  = None
        # Manage flow (cancel / reschedule / list)
        self.manage_phone:     Optional[str]  = None
        self.pending_bookings: Optional[list] = None
        self.selected_booking: Optional[dict] = None
        self.full_name:        Optional[str]  = None
        self.phone:            Optional[str]  = None
        self.email:            Optional[str]  = None
        self.birth_year:       Optional[str]  = None
        self.neighborhood:     Optional[str]  = None
        self.city:             Optional[str]  = None
        self.privacy_consent:  bool           = False
        self.available_dates:  List[str]      = []
        self.booking_uid:      Optional[str]  = None
        self.intent:           Optional[str]  = None


class FSM:
    def __init__(self):
        self.state            = State.START
        self.ctx              = ConversationContext()
        self.completed_ctx    = None
        self._dropped_ctx_snapshot = None

    # ── System prompt prefix (injected by _rebuild_instructions) ─────────────

    def get_system_prompt(self) -> str:
        now  = datetime.now()
        base = (
            f"[STATE CONTEXT] Today: {now.strftime('%A, %d %B %Y')} | "
            f"Timezone: America/Sao_Paulo."
        )

        if self.state == State.START:
            return (
                base + " Welcome the user. Answer their questions about Arte de Viver. "
                "When they want to register, call get_available_dates immediately. "
                "If they want to check, cancel, or reschedule a registration, ask for their email."
            )

        if self.state == State.REG_ASK_DATE:
            return (
                base + " REGISTRATION — STEP 1 of 7: CLASS DATE. "
                "You have already fetched the available dates. "
                "Read them one by one. Ask which works best. "
                "The moment the user picks one, call save_field(field='chosen_date', value='<what they said>')."
            )

        if self.state == State.REG_ASK_NAME:
            return (
                base
                + f" REGISTRATION — STEP 2 of 7: FULL NAME. "
                f"Date saved: {self.ctx.chosen_date or '(set)'}. "
                "Ask for their full name. The moment they say it, "
                "call save_field(field='full_name', value='<their name>') immediately. "
                "Do NOT read it back or ask them to confirm."
            )

        if self.state == State.REG_ASK_PHONE:
            return (
                base
                + f" REGISTRATION — STEP 3 of 7: WHATSAPP NUMBER. "
                f"Name saved: {self.ctx.full_name or '(set)'}. "
                "Ask for WhatsApp with DDD area code. "
                "The moment they give any number, call save_field(field='phone', value='<digits>') immediately. "
                "Do NOT read digits back. Do NOT count digits or ask for more yourself — "
                "the system validates automatically and returns an error if the number is incomplete."
            )

        if self.state == State.REG_ASK_EMAIL:
            return (
                base
                + f" REGISTRATION — STEP 4 of 7: EMAIL ADDRESS. "
                f"Phone saved: {self.ctx.phone or '(set)'}. "
                "Ask the user for their email. They may spell it letter by letter, "
                "say words like 'at' or 'dot', use phonetic letters, or speak it naturally. "
                "Reconstruct the email from whatever they say. "
                "Read the reconstructed email back character by character to verify. "
                "ONLY after they confirm it is correct, "
                "call save_field(field='email', value='<their email>')."
            )

        if self.state == State.REG_ASK_BIRTH_YEAR:
            return (
                base
                + f" REGISTRATION — STEP 5 of 7: YEAR OF BIRTH. "
                f"Email saved: {self.ctx.email or '(set)'}. "
                "Ask for their 4-digit year of birth (not age). "
                "The moment they say it, call save_field(field='birth_year', value='<year>') immediately. "
                "Do NOT read it back or ask them to confirm."
            )

        if self.state == State.REG_ASK_NEIGHBORHOOD:
            return (
                base
                + f" REGISTRATION — STEP 6 of 7: NEIGHBORHOOD. "
                f"Birth year saved: {self.ctx.birth_year or '(set)'}. "
                "Ask for their neighborhood. "
                "The moment they say it, call save_field(field='neighborhood', value='<neighborhood>') immediately. "
                "Do NOT read it back or ask them to confirm."
            )

        if self.state == State.REG_ASK_CITY:
            return (
                base
                + f" REGISTRATION — STEP 7 of 7: CITY. "
                f"Neighborhood saved: {self.ctx.neighborhood or '(set)'}. "
                "Ask for their city. "
                "The moment they say it, call save_field(field='city', value='<city>') immediately. "
                "Do NOT read it back or ask them to confirm."
            )

        if self.state == State.REG_ASK_CONSENT:
            c = self.ctx
            return (
                base
                + " ALL 7 FIELDS SAVED. Read a summary of all fields back to the user in one turn: "
                f"date '{c.chosen_date}', name '{c.full_name}', WhatsApp '{c.phone}', "
                f"email '{c.email}', born '{c.birth_year}', neighborhood '{c.neighborhood}', city '{c.city}'. "
                "Then ask: 'Is everything correct?' "
                "If YES → immediately call register_for_class with all field values: "
                f"class_date='{c.chosen_date}', full_name='{c.full_name}', "
                f"whatsapp_number='{c.phone}', email='{c.email}', "
                f"birth_year='{c.birth_year}', neighborhood='{c.neighborhood}', city='{c.city}'. "
                "If NO → ask which field to correct, fix it using save_field, then re-read the full summary."
            )

        if self.state == State.REG_CONFIRM:
            return (
                base + " Registration was just completed successfully. "
                "Say: Tudo certo! The user is registered. "
                "Reminders will be sent 24h and 2h before the class."
            )

        if self.state == State.REG_DONE:
            return (
                base + " Registration complete. "
                "Ask if there is anything else you can help with."
            )

        if self.state == State.MANAGE_ASK_PHONE:
            if self.ctx.intent == "reschedule":
                return (
                    base + " RESCHEDULE — STEP 1: GET PHONE. "
                    "Ask: 'What WhatsApp number did you use when registering?' "
                    "When they give it, call start_reschedule(phone='<their number>') IMMEDIATELY. "
                    "Do NOT call save_field or get_available_dates yet."
                )
            if self.ctx.intent == "cancel":
                return (
                    base + " CANCEL — STEP 1: GET EMAIL. "
                    "Ask: 'What email address did you use when registering?' "
                    "When they give it, call start_cancel(email='<their email>') IMMEDIATELY. "
                    "Do NOT call start_reschedule, save_field, or cancel_registration yet."
                )
            return (
                base + " User wants to check their registration. "
                "Ask: 'What WhatsApp number did you use when registering?' "
                "When they give it, call list_registrations(phone='<their number>')."
            )

        if self.state == State.MANAGE_LIST:
            return (
                base + " Registrations presented. "
                "Ask if they want to cancel or if there's anything else you can help with."
            )

        if self.state == State.MANAGE_CANCEL_CONFIRM:
            email_hint = self.ctx.email or '<their email>'
            return (
                base + " User wants to cancel. "
                "Ask 'May I ask why?' — wait for their answer. "
                f"Then call cancel_registration(email='{email_hint}', cancellation_reason='<reason>'). "
                "NEVER call cancel_registration without a real reason."
            )

        if self.state == State.MANAGE_RESCHEDULE_DATE:
            return (
                base + " RESCHEDULE — STEP 2: PICK NEW DATE. "
                "Email is already saved. Call get_available_dates immediately. "
                "Read dates one by one. "
                "When user picks one, call save_reschedule_date(date='<what they said>'). "
                "Do NOT call save_field. Do NOT call register_for_class. "
                "Do NOT ask for name, phone, birth year, neighborhood, or city."
            )

        if self.state == State.MANAGE_RESCHEDULE_REASON:
            date_hint = self.ctx.reschedule_date or '<chosen date>'
            return (
                base + " RESCHEDULE — STEP 3: ASK REASON THEN CALL TOOL. "
                f"New date chosen: {date_hint}. "
                "Ask: 'May I ask why you would like to reschedule?' — wait for the answer. "
                f"Then call reschedule_registration(new_date='{date_hint}', reschedule_reason='<their reason>'). "
                "Do NOT call save_field. Do NOT call register_for_class."
            )

        if self.state == State.MANAGE_RESCHEDULE_DONE:
            return (
                base + " Reschedule complete. "
                "Tell the user their class has been moved to the new date. "
                "Remind them they will receive reminders 24h and 2h before the new class. "
                "Ask if there is anything else you can help with."
            )

        return base + " How can I help you today?"

    # ── Silence prompts ───────────────────────────────────────────────────────

    def get_silence_prompt(self) -> str:
        return {
            State.REG_ASK_DATE:         "Which date would you like to sign up for?",
            State.REG_ASK_NAME:         "Could you tell me your full name?",
            State.REG_ASK_PHONE:        "Just need your WhatsApp number with the area code!",
            State.REG_ASK_EMAIL:        "Go ahead and spell out your email address.",
            State.REG_ASK_BIRTH_YEAR:   "What year were you born?",
            State.REG_ASK_NEIGHBORHOOD: "Which neighborhood are you in?",
            State.REG_ASK_CITY:         "And which city?",
            State.REG_ASK_CONSENT:      "I've read out your details — does everything look correct?",
            State.REG_CONFIRM:              "Should I go ahead and complete your registration?",
            State.MANAGE_ASK_PHONE:         "What email address did you use when registering?",
            State.MANAGE_CANCEL_CONFIRM:    "Would you like me to go ahead with the cancellation?",
            State.MANAGE_RESCHEDULE_DATE:   "Which new date would you like to move your class to?",
            State.MANAGE_RESCHEDULE_REASON: "May I ask why you'd like to reschedule?",
        }.get(self.state, "Hello, are you still there?")

    # ── State transitions ─────────────────────────────────────────────────────

    def update_state(self, intent: str = None, data: Dict[str, Any] = None) -> None:
        """
        Advance FSM based on intent signal and/or newly saved field data.

        Intent signals:
          'book'          → user wants to register
          'list'          → user wants to see registrations
          'cancel'        → user wants to cancel
          'confirm'       → registration complete (from register_for_class)
          'cancel_confirm'→ cancellation done
          'consent_denied'→ user refused consent

        Data keys (from save_field tool):
          chosen_date, full_name, phone, email,
          birth_year, neighborhood, city,
          chosen_class_iso, available_dates, booking_uid
        """
        if data is None:
            data = {}

        old = self.state

        # ── Always write data to ctx ──────────────────────────────────────────
        ALL_KEYS = (
            "chosen_date", "chosen_class_iso", "full_name", "phone",
            "email", "birth_year", "neighborhood", "city",
            "available_dates", "booking_uid",
            "reschedule_date", "reschedule_class_iso",
            "manage_phone", "pending_bookings", "selected_booking",
        )
        for k in ALL_KEYS:
            if k in data and data[k] is not None:
                setattr(self.ctx, k, data[k])

        # ── START ─────────────────────────────────────────────────────────────
        if self.state == State.START:
            if intent == "book":
                self.ctx.intent = "book"
                self.state      = State.REG_ASK_DATE
            elif intent == "list":
                self.ctx.intent = "list"
                self.state      = State.MANAGE_ASK_PHONE
            elif intent == "cancel":
                self.ctx.intent = "cancel"
                self.state      = State.MANAGE_ASK_PHONE
            elif intent == "reschedule":
                self.ctx.intent = "reschedule"
                self.state      = State.MANAGE_ASK_PHONE

        # ── REGISTRATION — field-by-field ─────────────────────────────────────
        elif self.state == State.REG_ASK_DATE:
            if "chosen_date" in data:
                self.state = State.REG_ASK_NAME

        elif self.state == State.REG_ASK_NAME:
            if "full_name" in data:
                self.state = State.REG_ASK_PHONE

        elif self.state == State.REG_ASK_PHONE:
            if "phone" in data:
                self.state = State.REG_ASK_EMAIL

        elif self.state == State.REG_ASK_EMAIL:
            if "email" in data:
                self.state = State.REG_ASK_BIRTH_YEAR

        elif self.state == State.REG_ASK_BIRTH_YEAR:
            if "birth_year" in data:
                self.state = State.REG_ASK_NEIGHBORHOOD

        elif self.state == State.REG_ASK_NEIGHBORHOOD:
            if "neighborhood" in data:
                self.state = State.REG_ASK_CITY

        elif self.state == State.REG_ASK_CITY:
            if "city" in data:
                self.state = State.REG_ASK_CONSENT

        elif self.state == State.REG_ASK_CONSENT:
            if intent == "confirm":
                self.ctx.privacy_consent = True
                self.state               = State.REG_CONFIRM
            elif intent == "consent_denied":
                self.state = State.START
                self.ctx   = ConversationContext()

        elif self.state == State.REG_CONFIRM:
            if intent == "confirm":
                self._snapshot(keep_ctx=True)
                self.state = State.REG_DONE

        elif self.state == State.REG_DONE:
            if intent in ("book", "list", "cancel"):
                self.ctx   = ConversationContext()
                self.state = State.START
                self.update_state(intent=intent, data=data)

        # ── MANAGE ────────────────────────────────────────────────────────────
        elif self.state == State.MANAGE_ASK_PHONE:
            # Transitions driven by intent set by start_cancel / start_reschedule tools
            if intent == "reschedule":
                self.state = State.MANAGE_RESCHEDULE_DATE
            elif intent == "cancel":
                self.state = State.MANAGE_CANCEL_CONFIRM
            elif intent in ("list", "phone_saved"):
                self.state = State.MANAGE_LIST

        elif self.state == State.MANAGE_LIST:
            if intent == "cancel":
                self.ctx.intent = "cancel"
                self.state      = State.MANAGE_CANCEL_CONFIRM
            elif intent == "reschedule":
                self.ctx.intent = "reschedule"
                self.state      = State.MANAGE_RESCHEDULE_DATE
            elif intent in ("confirm", "list"):
                self.state = State.START

        elif self.state == State.MANAGE_CANCEL_CONFIRM:
            if intent == "cancel_confirm":
                self._snapshot(keep_ctx=False)

        elif self.state == State.MANAGE_RESCHEDULE_DATE:
            # Triggered by save_reschedule_date tool
            if intent == "reschedule_date_saved" or "reschedule_date" in data:
                self.state = State.MANAGE_RESCHEDULE_REASON

        elif self.state == State.MANAGE_RESCHEDULE_REASON:
            if intent == "reschedule_confirm":
                self._snapshot(keep_ctx=True)
                self.state = State.MANAGE_RESCHEDULE_DONE

        elif self.state == State.MANAGE_RESCHEDULE_DONE:
            if intent in ("book", "list", "cancel", "reschedule"):
                self.ctx   = ConversationContext()
                self.state = State.START
                self.update_state(intent=intent, data=data)

        # ── Log ───────────────────────────────────────────────────────────────
        if old != self.state:
            _log.info(f"FSM: {old.name} → {self.state.name}")
        safe = {k: v for k, v in data.items() if k != "available_dates"}
        if safe:
            _log.debug(f"FSM data: {safe}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _snapshot(self, keep_ctx: bool = False) -> None:
        self.completed_ctx = copy.copy(self.ctx)
        if not keep_ctx:
            self.ctx   = ConversationContext()
            self.state = State.START    