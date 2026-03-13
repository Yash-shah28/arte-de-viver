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
        if self._user_language == "en":
            base = (
                f"[INTERNAL] Today: {now.strftime('%A, %d %B %Y')} | "
                f"Timezone: America/Sao_Paulo."
            )
        else:
            base = (
                f"[INTERNAL] Hoje: {now.strftime('%A, %d %B %Y')} | "
                f"Fuso: America/Sao_Paulo."
            )

        if self.state == State.START:
            if self._user_language == "en":
                return (
                    base + " Answer questions about Arte de Viver warmly and naturally. "
                    "When the person wants to register, call get_available_dates immediately. "
                    "If they want to check, cancel or reschedule, ask for their email."
                )
            return (
                base + " Responda perguntas sobre o Arte de Viver com calor e naturalidade. "
                "Quando a pessoa quiser se inscrever, chame get_available_dates imediatamente. "
                "Se quiser verificar, cancelar ou reagendar uma inscricao, peca o e-mail dela."
            )

        if self.state == State.REG_ASK_DATE:
            if self._user_language == "en":
                return (
                    base + " [STATE: date selection] You are showing available class dates. "
                    "The tool already gave you a day-level summary and per-day time details. "
                    "Present how many days are available and name the days — do NOT list every time slot. "
                    "Ask which DAY works best. When they pick a day, tell them the times available on that day and ask which time they prefer. "
                    "Only after they confirm a specific time, call save_field(field='chosen_date') immediately."
                )
            return (
                base + " [ESTADO: escolha de data] Voce esta mostrando as datas disponiveis. "
                "A ferramenta ja te deu o resumo por dia e os horarios de cada dia. "
                "Apresente quantos dias ha e quais sao — NAO liste todos os horarios de uma vez. "
                "Pergunte qual DIA funciona melhor. Quando escolherem um dia, fale os horarios daquele dia e pergunte qual prefere. "
                "So apos confirmarem um horario especifico, chame save_field(field='chosen_date') imediatamente."
            )

        if self.state == State.REG_ASK_NAME:
            if self._user_language == "en":
                return (
                    base + f" [STATE: collecting name] Date chosen: {self.ctx.chosen_date}. "
                    "Ask for their full name warmly — you are happy to get them registered. "
                    "As soon as you hear the name, call save_field(field='full_name') immediately. Do not repeat it back."
                )
            return (
                base + f" [ESTADO: coletando nome] Data escolhida: {self.ctx.chosen_date}. "
                "Peca o nome completo de forma calorosa e direta — voce esta feliz em fazer a inscricao. "
                "Ao ouvir o nome, chame save_field(field='full_name') imediatamente. Nao repita de volta."
            )

        if self.state == State.REG_ASK_PHONE:
            if self._user_language == "en":
                phone_instr = (
                    f" [STATE: collecting WhatsApp] Name saved: {self.ctx.full_name}. "
                    "Ask for their WhatsApp number naturally. Do NOT mention area code. "
                    "As soon as they give any number, call save_field(field='phone') immediately. Do not read digits back."
                )
            else:
                phone_instr = (
                    f" [ESTADO: coletando WhatsApp] Nome salvo: {self.ctx.full_name}. "
                    "Peca o numero de WhatsApp de forma natural. NAO mencione DDD ou codigo de area. "
                    "Ao ouvir qualquer numero, chame save_field(field='phone') imediatamente. Nao leia os digitos de volta."
                )
            return base + phone_instr

        if self.state == State.REG_ASK_EMAIL:
            if self._user_language == "en":
                return (
                    base + f" [STATE: collecting email] WhatsApp saved. "
                    "Ask for their email lightly. When they give it, call save_field(field='email') IMMEDIATELY. "
                    "Do NOT read the email back. Do NOT ask them to confirm it. Save and move on."
                )
            return (
                base + f" [ESTADO: coletando e-mail] WhatsApp salvo. "
                "Peca o e-mail de forma leve. Quando a pessoa fornecer, chame save_field(field='email') IMEDIATAMENTE. "
                "NAO leia o e-mail de volta. NAO peca confirmacao do e-mail. Salve direto e siga para proximo campo."
            )

        if self.state == State.REG_ASK_BIRTH_YEAR:
            if self._user_language == "en":
                return (
                    base + f" [STATE: collecting birth year] Email saved: {self.ctx.email}. "
                    "Ask for their year of birth simply and casually. "
                    "As soon as you hear it, call save_field(field='birth_year') immediately. Do not repeat it."
                )
            return (
                base + f" [ESTADO: coletando ano de nascimento] E-mail salvo: {self.ctx.email}. "
                "Peca o ano de nascimento de forma simples e descontraida. "
                "Ao ouvir, chame save_field(field='birth_year') imediatamente. Nao repita."
            )

        if self.state == State.REG_ASK_NEIGHBORHOOD:
            if self._user_language == "en":
                return (
                    base + f" [STATE: collecting neighborhood] Birth year saved: {self.ctx.birth_year}. "
                    "Ask for their neighborhood naturally — quick, conversational. "
                    "As soon as you hear it, call save_field(field='neighborhood') immediately. Do not repeat it."
                )
            return (
                base + f" [ESTADO: coletando bairro] Ano salvo: {self.ctx.birth_year}. "
                "Peca o bairro de forma natural — rapido, sem fazer parecer um formulario. "
                "Ao ouvir, chame save_field(field='neighborhood') imediatamente. Nao repita."
            )

        if self.state == State.REG_ASK_CITY:
            if self._user_language == "en":
                return (
                    base + f" [STATE: collecting city] Neighborhood saved: {self.ctx.neighborhood}. "
                    "Ask for their city simply. "
                    "As soon as you hear it, call save_field(field='city') immediately. Do not repeat it."
                )
            return (
                base + f" [ESTADO: coletando cidade] Bairro salvo: {self.ctx.neighborhood}. "
                "Peca a cidade de forma simples. "
                "Ao ouvir, chame save_field(field='city') imediatamente. Nao repita."
            )

        if self.state == State.REG_ASK_CONSENT:
            c = self.ctx
            if self._user_language == "en":
                return (
                    base + " [STATE: final confirmation] All 7 fields are in. "
                    "Read them back like a friend catching up — warm, natural, no robotic list. "
                    "Use their first name and weave the details into real sentences, for example: "
                    "'Okay [name], let me just make sure I've got everything — you're coming on [date], right? "
                    "WhatsApp [phone], email [email], born in [birth_year], from [neighborhood] in [city]. "
                    "That all look good to you?' Then wait for their answer. "
                    f"Actual values: date={c.chosen_date}, name={c.full_name}, WhatsApp={c.phone}, "
                    f"email={c.email}, birth year={c.birth_year}, neighborhood={c.neighborhood}, city={c.city}. "
                    "If YES → call register_for_class immediately with all values. "
                    "If NO → ask warmly what needs fixing, correct it with save_field, re-read just the changed part — don't start over."
                )
            return (
                base + " [ESTADO: confirmacao final] Todos os 7 campos foram coletados. "
                "Faca um resumo caloroso e conversacional — como alguem lendo de volta pra um amigo, NAO como impressora de recibo. "
                "Use o primeiro nome. Conecte os campos de forma natural, por exemplo: "
                "'Entao [nome], deixa eu confirmar aqui — voce vai na aula do dia [data], ne? "
                "Tenho seu WhatsApp como [telefone], e-mail [email], nascido(a) em [ano], "
                "do [bairro] em [cidade]. Ta tudo certinho assim?' "
                f"Dados: data={c.chosen_date}, nome={c.full_name}, WhatsApp={c.phone}, "
                f"e-mail={c.email}, nascimento={c.birth_year}, bairro={c.neighborhood}, cidade={c.city}. "
                "Se SIM → chame register_for_class imediatamente com todos os valores. "
                "Se NAO → pergunte o que corrigir, use save_field, depois releia o resumo naturalmente."
            )

        if self.state == State.REG_CONFIRM:
            if self._user_language == "en":
                return (
                    base + " [STATE: registration confirmed] Registration just went through. "
                    "React with genuine joy — something like 'You're all set, [name]!' or 'Done! You're officially in!' "
                    "Use their first name. Mention reminders as a friendly heads-up, not a legal notice — "
                    "like 'You'll get a reminder before the class!' Be brief and warm. Max 2 sentences."
                )
            return (
                base + " [ESTADO: inscricao concluida] Inscricao acabou de ser confirmada. "
                "Reaja com genuina alegria — algo como 'Prontinho, [nome]! Tudo confirmado!' ou 'Feito! Voce ja esta inscrito(a)!' "
                "Use o primeiro nome. Mencione os lembretes como aviso amigavel — "
                "tipo 'Voce vai receber um lembrete antes da aula!' — nao como termo legal. Maximo 2 frases."
            )

        if self.state == State.REG_DONE:
            if self._user_language == "en":
                return (
                    base + " [STATE: post-registration] Registration is complete. "
                    "Do NOT bring up registering again — they're already in. "
                    "Close like a friend wrapping up a call — warm and natural. "
                    "Something like 'Is there anything else I can help you with?' or 'Any other questions about the class?'"
                )
            return (
                base + " [ESTADO: pos-inscricao] Inscricao ja foi concluida. "
                "NAO sugira nem pergunte sobre inscrever de novo — a pessoa JA esta inscrita. "
                "Encerre como alguem encerrando uma conversa de forma natural e calorosa. "
                "Algo como 'Tem mais alguma coisa que posso te ajudar?' ou 'Alguma duvida sobre a aula?'"
            )

        if self.state == State.MANAGE_ASK_PHONE:
            if self._user_language == "en":
                if self.ctx.intent == "reschedule":
                    return (
                        base + " [STATE: reschedule — waiting for email] "
                        "Ask naturally for their registration email. "
                        "When they give it, call start_reschedule(email='<their email>') IMMEDIATELY. "
                        "Do not call save_field or get_available_dates yet."
                    )
                if self.ctx.intent == "cancel":
                    return (
                        base + " [STATE: cancellation — waiting for email] "
                        "Ask naturally for their registration email. "
                        "When they give it, call start_cancel(email='<their email>') IMMEDIATELY. "
                        "Do not call start_reschedule, save_field or cancel_registration yet."
                    )
                return (
                    base + " [STATE: lookup — waiting for email] "
                    "Ask for their registration email. "
                    "When they give it, call list_registrations(email='<their email>')."
                )
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
            if self._user_language == "en":
                return (
                    base + " [STATE: registrations listed] "
                    "Ask if they want to cancel, reschedule, or if there is anything else you can help with."
                )
            return (
                base + " [ESTADO: inscricoes listadas] "
                "Pergunte se querem cancelar, reagendar ou se ha mais alguma coisa que pode ajudar."
            )

        if self.state == State.MANAGE_CANCEL_CONFIRM:
            email_hint = self.ctx.email or '<e-mail deles>'
            if self._user_language == "en":
                return (
                    base + " [STATE: cancellation — waiting for reason] "
                    "Ask for the cancellation reason kindly — like someone who wants to understand, not audit. "
                    f"Wait for the answer. Then call cancel_registration(email='{email_hint}', cancellation_reason='<reason>'). "
                    "NEVER call cancel_registration without a real reason."
                )
            return (
                base + " [ESTADO: cancelamento — aguardando motivo] "
                "Pergunte o motivo do cancelamento de forma gentil — como quem quer entender, nao auditar. "
                f"Aguarde a resposta. Depois chame cancel_registration(email='{email_hint}', cancellation_reason='<motivo>'). "
                "NUNCA chame cancel_registration sem um motivo real."
            )

        if self.state == State.MANAGE_RESCHEDULE_DATE:
            if self._user_language == "en":
                return (
                    base + " [STATE: reschedule — choosing new date] "
                    "Email is already saved. Call get_available_dates immediately. "
                    "The tool will give you a day-level summary and per-day time details. "
                    "Present how many days are available and name the days — do NOT list every time slot. "
                    "Ask which DAY works best. When they pick a day, tell them the available times and ask which they prefer. "
                    "Only after they confirm a specific time, call save_reschedule_date(date='<what they said>'). "
                    "Do not call save_field. Do not call register_for_class. "
                    "Do not ask for name, phone, birth year, neighborhood or city."
                )
            return (
                base + " [ESTADO: reagendamento — escolha de nova data] "
                "E-mail ja esta salvo. Chame get_available_dates imediatamente. "
                "A ferramenta vai te dar o resumo por dia e os horarios de cada dia. "
                "Apresente quantos dias ha e quais sao — NAO liste todos os horarios de uma vez. "
                "Pergunte qual DIA funciona melhor. Quando escolherem um dia, fale os horarios daquele dia e pergunte qual prefere. "
                "So apos confirmarem um horario especifico, chame save_reschedule_date(date='<o que disseram>'). "
                "Nao chame save_field. Nao chame register_for_class. "
                "Nao peca nome, telefone, ano de nascimento, bairro ou cidade."
            )

        if self.state == State.MANAGE_RESCHEDULE_REASON:
            date_hint = self.ctx.reschedule_date or '<data escolhida>'
            if self._user_language == "en":
                return (
                    base + f" [STATE: reschedule — waiting for reason] New date: {date_hint}. "
                    "Ask for the reason warmly — this is a conversation, not an audit. "
                    f"Wait for the answer. Then call reschedule_registration(new_date='{date_hint}', reschedule_reason='<reason>'). "
                    "Do not call save_field. Do not call register_for_class."
                )
            return (
                base + f" [ESTADO: reagendamento — aguardando motivo] Nova data: {date_hint}. "
                "Pergunte o motivo de forma calorosa — nao e uma auditoria, e uma conversa. "
                f"Aguarde a resposta. Depois chame reschedule_registration(new_date='{date_hint}', reschedule_reason='<motivo>'). "
                "Nao chame save_field. Nao chame register_for_class."
            )

        if self.state == State.MANAGE_RESCHEDULE_DONE:
            if self._user_language == "en":
                return (
                    base + " [STATE: reschedule complete] Reschedule went through successfully. "
                    "React like someone who just sorted something for a friend — genuinely relieved and happy. "
                    "Something like 'All sorted, [name]! You're on for [new date] now.' "
                    "Use their first name if you know it. Mention the reminder lightly. "
                    "Close warm and brief — no support-ticket language."
                )
            return (
                base + " [ESTADO: reagendamento concluido] Reagendamento foi feito com sucesso. "
                "Reaja como pessoa que resolveu algo pra um amigo — aliviada e genuinamente feliz. "
                "Algo como 'Prontinho, [nome]! Ja ta marcado pra [nova data]!' "
                "Use o primeiro nome se souber. Mencione o lembrete de forma leve. "
                "Encerre de forma calorosa e breve — sem linguagem de suporte tecnico."
            )

        return base + " Como posso te ajudar?"

    # ── Silence prompts ───────────────────────────────────────────────────────

    def get_silence_prompt(self) -> str:
        import random
        pt_prompts = {
            State.REG_ASK_DATE:         [
                "Oi, continua comigo? Qual dessas datas funciona melhor pra voce?",
                "Ainda to aqui! Alguma das datas te agrada?",
                "Sem pressa — me fala qual data fica melhor pra voce!",
            ],
            State.REG_ASK_NAME:         [
                "Oi, pode me dizer seu nome completo quando quiser!",
                "Ainda to aqui — pode falar o seu nome?",
                "Sem pressa! E o seu nome completo?",
            ],
            State.REG_ASK_PHONE:        [
                "Pode me passar o WhatsApp quando quiser!",
                "Ainda to aqui — e o seu WhatsApp?",
                "Sem pressa, me fala o numero do WhatsApp.",
            ],
            State.REG_ASK_EMAIL:        [
                "E o e-mail? Pode falar quando estiver pronto!",
                "Ainda to aqui — pode me passar o e-mail?",
                "Sem pressa! E o seu e-mail?",
            ],
            State.REG_ASK_BIRTH_YEAR:   [
                "E o ano que voce nasceu? Pode me dizer!",
                "Ainda to aqui — qual e o seu ano de nascimento?",
                "Sem pressa! E o ano de nascimento?",
            ],
            State.REG_ASK_NEIGHBORHOOD: [
                "Pode me dizer o bairro quando quiser!",
                "Ainda to aqui — e o seu bairro?",
                "E o bairro? Pode falar!",
            ],
            State.REG_ASK_CITY:         [
                "E a cidade? Pode me dizer!",
                "Ainda to aqui — e a sua cidade?",
                "Sem pressa, e a cidade?",
            ],
            State.REG_ASK_CONSENT:      [
                "Ta tudo certinho nos seus dados? E so me confirmar!",
                "Ainda to aqui — pode confirmar se ta tudo certo?",
                "Me fala se esta tudo certo com as informacoes!",
            ],
            State.REG_CONFIRM:          [
                "Posso finalizar a sua inscricao?",
                "Ainda to aqui — tudo certo pra confirmar?",
                "E so me dar o OK e ja confirmo!",
            ],
            State.MANAGE_ASK_PHONE:     [
                "Qual e-mail voce usou quando se inscreveu?",
                "Pode me passar o e-mail de cadastro quando quiser!",
                "Ainda to aqui — me fala o e-mail que voce usou.",
            ],
            State.MANAGE_CANCEL_CONFIRM: [
                "Pode me contar o motivo do cancelamento quando quiser?",
                "Ainda to aqui — me diz por que quer cancelar, pode ser?",
                "Sem pressa — qual seria o motivo do cancelamento?",
            ],
            State.MANAGE_RESCHEDULE_DATE: [
                "Qual data nova funciona melhor pra voce?",
                "Ainda to aqui — alguma das datas te funciona?",
                "Me fala qual data prefere e ja marco pra voce!",
            ],
            State.MANAGE_RESCHEDULE_REASON: [
                "Pode me contar o motivo do reagendamento?",
                "Ainda to aqui — me diz por que quer reagendar?",
                "Sem pressa — qual seria o motivo?",
            ],
        }
        en_prompts = {
            State.REG_ASK_DATE:         [
                "Still there? Which of those days works best for you?",
                "Hey, still here! Does any of those days work for you?",
                "No rush — which day would you prefer?",
            ],
            State.REG_ASK_NAME:         [
                "Still here! What's your full name?",
                "Hey, still with you — can you tell me your full name?",
                "No rush! And your full name is?",
            ],
            State.REG_ASK_PHONE:        [
                "What's your WhatsApp number whenever you're ready!",
                "Still here — and your WhatsApp number?",
                "No rush! What's your WhatsApp?",
            ],
            State.REG_ASK_EMAIL:        [
                "And your email? Just whenever you're ready!",
                "Still here — what's your email address?",
                "No rush! And your email?",
            ],
            State.REG_ASK_BIRTH_YEAR:   [
                "And what year were you born? Just whenever!",
                "Still here — what's your year of birth?",
                "No rush! And your birth year?",
            ],
            State.REG_ASK_NEIGHBORHOOD: [
                "What neighborhood are you in? Whenever you're ready!",
                "Still here — and your neighborhood?",
                "No rush! What's your neighborhood?",
            ],
            State.REG_ASK_CITY:         [
                "And your city? Just whenever you're ready!",
                "Still here — what city are you in?",
                "No rush! And your city?",
            ],
            State.REG_ASK_CONSENT:      [
                "Does everything look right? Just say yes and we're good to go!",
                "Still here — does all that information look correct?",
                "Just let me know if everything's right!",
            ],
            State.REG_CONFIRM:          [
                "Ready to lock in your registration?",
                "Still here — just say the word and I'll confirm you!",
                "Shall I go ahead and confirm everything?",
            ],
            State.MANAGE_ASK_PHONE:     [
                "Which email did you use when you registered?",
                "Still here — can you give me your registration email?",
                "No rush! What email did you sign up with?",
            ],
            State.MANAGE_CANCEL_CONFIRM: [
                "Can you tell me why you'd like to cancel? Whenever you're ready.",
                "Still here — what's the reason for cancelling?",
                "No rush — just let me know why you'd like to cancel.",
            ],
            State.MANAGE_RESCHEDULE_DATE: [
                "Which new day works best for you?",
                "Still here — does any of those days work for you?",
                "No rush! Which day would you prefer?",
            ],
            State.MANAGE_RESCHEDULE_REASON: [
                "Can you tell me why you'd like to reschedule?",
                "Still here — what's the reason for the change?",
                "No rush — just let me know why you'd like to reschedule.",
            ],
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