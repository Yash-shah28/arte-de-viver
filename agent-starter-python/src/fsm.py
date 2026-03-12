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
        self._user_language   = "pt"  # detectado dinamicamente: "pt" ou "en"

    # ── System prompt prefix (injected by _rebuild_instructions) ─────────────

    def get_system_prompt(self) -> str:
        now  = datetime.now()
        base = (
            f"[INTERNAL] Hoje: {now.strftime('%A, %d %B %Y')} | "
            f"Fuso: America/Sao_Paulo."
        )

        if self.state == State.START:
            return (
                base + " Responda perguntas sobre o Arte de Viver com calor e naturalidade. "
                "Quando a pessoa quiser se inscrever, chame get_available_dates imediatamente. "
                "Se quiser verificar, cancelar ou reagendar uma inscricao, peca o e-mail dela."
            )

        if self.state == State.REG_ASK_DATE:
            return (
                base + " [ESTADO: escolha de data] Voce esta mostrando as datas disponiveis. "
                "Leia cada data devagar, uma por vez — como quem esta contando opcoes para um amigo. "
                "Quando a pessoa escolher, chame save_field(field='chosen_date') imediatamente."
            )

        if self.state == State.REG_ASK_NAME:
            return (
                base + f" [ESTADO: coletando nome] Data escolhida: {self.ctx.chosen_date}. "
                "Peca o nome completo de forma calorosa e direta — voce esta feliz em fazer a inscricao. "
                "Ao ouvir o nome, chame save_field(field='full_name') imediatamente. Nao repita de volta."
            )

        if self.state == State.REG_ASK_PHONE:
            return (
                base + f" [ESTADO: coletando WhatsApp] Nome salvo: {self.ctx.full_name}. "
                "Peca o WhatsApp com DDD de forma natural — como quem esta anotando num papel. "
                "Ao ouvir qualquer numero, chame save_field(field='phone') imediatamente. Nao leia os digitos de volta."
            )

        if self.state == State.REG_ASK_EMAIL:
            return (
                base + f" [ESTADO: coletando e-mail] WhatsApp salvo. "
                "Peca o e-mail de forma leve. Reconstrua do que a pessoa disser e leia de volta para confirmar. "
                "So apos confirmacao chame save_field(field='email')."
            )

        if self.state == State.REG_ASK_BIRTH_YEAR:
            return (
                base + f" [ESTADO: coletando ano de nascimento] E-mail salvo: {self.ctx.email}. "
                "Peca o ano de nascimento de forma simples e descontraida. "
                "Ao ouvir, chame save_field(field='birth_year') imediatamente. Nao repita."
            )

        if self.state == State.REG_ASK_NEIGHBORHOOD:
            return (
                base + f" [ESTADO: coletando bairro] Ano salvo: {self.ctx.birth_year}. "
                "Peca o bairro de forma natural — rapido, sem fazer parecer um formulario. "
                "Ao ouvir, chame save_field(field='neighborhood') imediatamente. Nao repita."
            )

        if self.state == State.REG_ASK_CITY:
            return (
                base + f" [ESTADO: coletando cidade] Bairro salvo: {self.ctx.neighborhood}. "
                "Peca a cidade de forma simples. "
                "Ao ouvir, chame save_field(field='city') imediatamente. Nao repita."
            )

        if self.state == State.REG_ASK_CONSENT:
            c = self.ctx
            return (
                base + " [ESTADO: confirmacao final] Todos os campos foram coletados. "
                "Faca o resumo como pessoa encerrando — nao como impressora de recibo. "
                f"Dados: data={c.chosen_date}, nome={c.full_name}, WhatsApp={c.phone}, "
                f"e-mail={c.email}, nascimento={c.birth_year}, bairro={c.neighborhood}, cidade={c.city}. "
                "Se a pessoa confirmar (SIM) → chame register_for_class imediatamente com todos os valores. "
                "Se NAO → pergunte o que corrigir, use save_field, depois releia o resumo naturalmente."
            )

        if self.state == State.REG_CONFIRM:
            return (
                base + " [ESTADO: inscricao concluida] Inscricao acabou de ser confirmada. "
                "Reaja com genuina alegria pela pessoa — use o primeiro nome se souber. "
                "Mencione os lembretes naturalmente, nao como aviso. Seja breve e caloroso."
            )

        if self.state == State.REG_DONE:
            return (
                base + " [ESTADO: pos-inscricao] Inscricao completa. "
                "Pergunte se ha mais alguma coisa que a pessoa queira saber — como encerrando uma conversa real."
            )

        if self.state == State.MANAGE_ASK_PHONE:
            if self.ctx.intent == "reschedule":
                return (
                    base + " [ESTADO: reagendamento — aguardando e-mail] "
                    "Peca o e-mail de cadastro de forma natural. "
                    "Quando fornecerem, chame start_reschedule(email='<e-mail deles>') IMEDIATAMENTE. "
                    "Nao chame save_field nem get_available_dates ainda."
                )
            if self.ctx.intent == "cancel":
                return (
                    base + " [ESTADO: cancelamento — aguardando e-mail] "
                    "Peca o e-mail de cadastro de forma natural. "
                    "Quando fornecerem, chame start_cancel(email='<e-mail deles>') IMEDIATAMENTE. "
                    "Nao chame start_reschedule, save_field ou cancel_registration ainda."
                )
            return (
                base + " [ESTADO: consulta — aguardando e-mail] "
                "Peca o e-mail de cadastro. "
                "Quando fornecerem, chame list_registrations(email='<e-mail deles>')."
            )

        if self.state == State.MANAGE_LIST:
            return (
                base + " [ESTADO: inscricoes listadas] "
                "Pergunte se querem cancelar, reagendar ou se ha mais alguma coisa que pode ajudar."
            )

        if self.state == State.MANAGE_CANCEL_CONFIRM:
            email_hint = self.ctx.email or '<e-mail deles>'
            return (
                base + " [ESTADO: cancelamento — aguardando motivo] "
                "Pergunte o motivo do cancelamento de forma gentil — como quem quer entender, nao auditar. "
                f"Aguarde a resposta. Depois chame cancel_registration(email='{email_hint}', cancellation_reason='<motivo>'). "
                "NUNCA chame cancel_registration sem um motivo real."
            )

        if self.state == State.MANAGE_RESCHEDULE_DATE:
            return (
                base + " [ESTADO: reagendamento — escolha de nova data] "
                "E-mail ja esta salvo. Chame get_available_dates imediatamente. "
                "Leia as datas uma por uma de forma natural. "
                "Quando o usuario escolher, chame save_reschedule_date(date='<o que disseram>'). "
                "Nao chame save_field. Nao chame register_for_class. "
                "Nao peca nome, telefone, ano de nascimento, bairro ou cidade."
            )

        if self.state == State.MANAGE_RESCHEDULE_REASON:
            date_hint = self.ctx.reschedule_date or '<data escolhida>'
            return (
                base + f" [ESTADO: reagendamento — aguardando motivo] Nova data: {date_hint}. "
                "Pergunte o motivo de forma calorosa — nao e uma auditoria, e uma conversa. "
                f"Aguarde a resposta. Depois chame reschedule_registration(new_date='{date_hint}', reschedule_reason='<motivo>'). "
                "Nao chame save_field. Nao chame register_for_class."
            )

        if self.state == State.MANAGE_RESCHEDULE_DONE:
            return (
                base + " [ESTADO: reagendamento concluido] Reagendamento foi feito com sucesso. "
                "Reaja como pessoa que resolveu algo para um amigo — aliviada e feliz. "
                "Use o nome deles se souber. Mencione o lembrete naturalmente. "
                "Encerre como conversa real, nao como ticket de suporte."
            )

        return base + " Como posso te ajudar?"

    # ── Silence prompts ───────────────────────────────────────────────────────

    def get_silence_prompt(self) -> str:
        import random
        pt_prompts = {
            State.REG_ASK_DATE:         [
                "Voce ainda ta ai? Qual das datas funciona melhor pra voce?",
                "Oi, continuo aqui! Alguma das datas te agrada?",
            ],
            State.REG_ASK_NAME:         [
                "Oi, pode me dizer seu nome completo quando quiser!",
                "Ainda to aqui — pode falar seu nome?",
            ],
            State.REG_ASK_PHONE:        [
                "Pode falar o WhatsApp com DDD, pode ser!",
                "E o numero do WhatsApp com DDD?",
            ],
            State.REG_ASK_EMAIL:        [
                "E o e-mail? Pode soletrar se preferir.",
                "Pode me passar o e-mail?",
            ],
            State.REG_ASK_BIRTH_YEAR:   [
                "E o ano que voce nasceu?",
                "Pode me dizer o ano de nascimento?",
            ],
            State.REG_ASK_NEIGHBORHOOD: [
                "Pode me dizer o bairro?",
                "E o bairro?",
            ],
            State.REG_ASK_CITY:         [
                "E a cidade?",
                "Pode me dizer a cidade?",
            ],
            State.REG_ASK_CONSENT:      [
                "Ta tudo certinho nos seus dados?",
                "Posso confirmar as informacoes?",
            ],
            State.REG_CONFIRM:              [
                "Posso confirmar sua inscricao?",
                "Tudo certo pra confirmar?",
            ],
            State.MANAGE_ASK_PHONE:         [
                "Qual e-mail voce usou quando se inscreveu?",
                "Pode me passar o e-mail de cadastro?",
            ],
            State.MANAGE_CANCEL_CONFIRM:    [
                "Posso ir em frente com o cancelamento?",
                "Confirma o cancelamento?",
            ],
            State.MANAGE_RESCHEDULE_DATE:   [
                "Qual data nova funciona pra voce?",
                "Alguma das datas te funciona?",
            ],
            State.MANAGE_RESCHEDULE_REASON: [
                "Pode me contar o motivo do reagendamento?",
                "E o motivo, pode me dizer?",
            ],
        }
        en_prompts = {
            State.REG_ASK_DATE:         ["Still there? Which date works best for you?"],
            State.REG_ASK_NAME:         ["Hi, still here! Can you tell me your full name?"],
            State.REG_ASK_PHONE:        ["What's your WhatsApp number with area code?"],
            State.REG_ASK_EMAIL:        ["What's your email? Feel free to spell it out."],
            State.REG_ASK_BIRTH_YEAR:   ["And your year of birth?"],
            State.REG_ASK_NEIGHBORHOOD: ["What neighborhood are you in?"],
            State.REG_ASK_CITY:         ["And your city?"],
            State.REG_ASK_CONSENT:      ["Does everything look correct?"],
            State.REG_CONFIRM:          ["Ready to confirm your registration?"],
            State.MANAGE_ASK_PHONE:     ["Which email did you use when you registered?"],
            State.MANAGE_CANCEL_CONFIRM:["Shall I go ahead with the cancellation?"],
            State.MANAGE_RESCHEDULE_DATE:["Which new date works for you?"],
            State.MANAGE_RESCHEDULE_REASON:["Can you tell me why you'd like to reschedule?"],
        }
        prompts = en_prompts if self._user_language == "en" else pt_prompts
        options = prompts.get(self.state, ["Oi, voce ainda ta ai?"])
        return random.choice(options)

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