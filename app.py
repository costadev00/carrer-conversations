import base64
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from threading import Lock

import gradio as gr
import requests
from dotenv import load_dotenv
from openai import OpenAI
from PyPDF2 import PdfReader


load_dotenv(override=True)

CONTACT_EMAIL = "mcostamonteiro@usp.br"
LEGACY_CONTACT_EMAILS = {
    "matheuscostamonteiro.mc@gmail.com",
}

def push(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        }
    )


def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Recording {question}")
    return {"recorded": "ok"}


def _sanitize_contact_info(text: str) -> str:
    sanitized = text
    for legacy_email in LEGACY_CONTACT_EMAILS:
        sanitized = sanitized.replace(legacy_email, CONTACT_EMAIL)
    return sanitized

def _normalize_messages(history, user_message, assistant_response):
    """Return chronological list of dicts with role/content for current chat."""

    normalized = []
    base_messages = list(history or [])
    base_messages.append({"role": "user", "content": user_message})
    base_messages.append({"role": "assistant", "content": assistant_response})

    for entry in base_messages:
        role = None
        content = None
        if isinstance(entry, dict):
            role = entry.get("role")
            content = entry.get("content")
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            role, content = entry

        if role in {"user", "assistant"} and isinstance(content, str):
            normalized.append({"role": role, "content": content})

    return normalized


def _build_ordered_turns(normalized_messages):
    """Group chronological messages into user/assistant iterations."""

    turns = []
    iteration = 1
    pending_user = None

    for message in normalized_messages:
        role = message["role"]
        content = message["content"]

        if role == "user":
            pending_user = content
        elif role == "assistant" and pending_user is not None:
            turns.append({
                "iteration": iteration,
                "user": pending_user,
                "assistant": content,
            })
            iteration += 1
            pending_user = None

    if pending_user:
        turns.append({
            "iteration": iteration,
            "user": pending_user,
            "assistant": "",
        })

    return turns
def log_chat_interaction(user_message, assistant_response, history, session_path, current_sha):
    """Persist entire chat session to a single GitHub file, updating each turn."""

    owner = os.getenv("GITHUB_OWNER")
    repo = os.getenv("GITHUB_REPO")
    token = os.getenv("GITHUB_TOKEN")
    branch = os.getenv("GITHUB_BRANCH", "main")

    if not (owner and repo and token):
        return current_sha

    normalized_messages = _normalize_messages(history, user_message, assistant_response)
    turns = _build_ordered_turns(normalized_messages)

    content = json.dumps(
        {
            "messages": normalized_messages,
            "session": turns,
        },
        ensure_ascii=False,
        indent=2,
    )
    encoded_content = base64.b64encode(content.encode("utf-8")).decode("ascii")

    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{session_path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    data = {
        "message": f"Update chat session {session_path}",
        "content": encoded_content,
        "branch": branch,
    }

    if current_sha:
        data["sha"] = current_sha

    try:
        resp = requests.put(url, headers=headers, json=data, timeout=15)
        resp.raise_for_status()
        response_json = resp.json()
        new_sha = response_json.get("content", {}).get("sha", current_sha)
        return new_sha
    except requests.RequestException as exc:
        print(f"Failed to log chat interaction to GitHub repo: {exc}", flush=True)
        return current_sha


record_user_details_json = {
    "name": "record_user_details",
    "description": "Use esta ferramenta para registrar que um usuário está interessado em entrar em contato e forneceu um endereço de e-mail",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "O endereço de e-mail deste usuário"
            },
            "name": {
                "type": "string",
                "description": "O nome do usuário, se fornecido"
            },
            "notes": {
                "type": "string",
                "description": "Qualquer informação adicional sobre a conversa que seja relevante para registrar o contexto"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Sempre use esta ferramenta para registrar qualquer pergunta que não pôde ser respondida porque você não sabia a resposta",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "A pergunta que não pôde ser respondida"
            },
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [
    {"type": "function", "function": record_user_details_json},
    {"type": "function", "function": record_unknown_question_json},
]


class Me:

    def __init__(self):
        self.openai = OpenAI()
        self.name = "Matheus Costa"
        self.sessions_lock = Lock()
        self.sessions = {}
        base_dir = Path("me")
        self.linkedin = _sanitize_contact_info(
            self._load_document(base_dir / "linkedin.txt", base_dir / "linkedin.pdf")
        )
        self.lattes = _sanitize_contact_info(
            self._load_document(base_dir / "lattes.txt", base_dir / "lattes.pdf")
        )
        with open(base_dir / "summary.txt", "r", encoding="utf-8") as f:
            self.summary = _sanitize_contact_info(f.read())

    def _create_session_record(self):
        session_started_at = datetime.utcnow().isoformat()
        session_id = uuid.uuid4().hex[:8]
        session_stamp = session_started_at.replace(":", "-")
        return {
            "started_at": session_started_at,
            "session_id": session_id,
            "session_path": f"sessions/{session_stamp}_{session_id}.json",
            "session_sha": None,
        }

    def _get_session_record(self, session_key, reset=False):
        with self.sessions_lock:
            if reset or session_key not in self.sessions:
                self.sessions[session_key] = self._create_session_record()
            return dict(self.sessions[session_key])

    def _update_session_sha(self, session_key, session_sha):
        with self.sessions_lock:
            if session_key in self.sessions:
                self.sessions[session_key]["session_sha"] = session_sha

    def _load_document(self, txt_path: Path, pdf_path: Path) -> str:
        if txt_path.exists():
            return txt_path.read_text(encoding="utf-8")
        if pdf_path.exists():
            reader = PdfReader(str(pdf_path))
            chunks = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    chunks.append(text)
            return "".join(chunks)
        return ""

    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": tool_call.id,
            })
        return results

    def system_prompt(self):
        system_prompt = f"Você está atuando como {self.name}. Você está respondendo perguntas no site de {self.name}, \
    particularmente perguntas relacionadas à carreira, histórico, habilidades e experiência de {self.name}. \
    Sua responsabilidade é representar {self.name} nas interações no site da forma mais fiel possível. \
    Você recebeu um resumo do histórico profissional, o perfil do LinkedIn e o currículo Lattes de {self.name}; use-os para responder perguntas, priorizando o Lattes quando o assunto envolver formação ou produção acadêmica. \
    Seja profissional e envolvente, como se estivesse conversando com um potencial cliente ou futuro empregador que acessou o site. \
    O e-mail de contato correto de {self.name} é {CONTACT_EMAIL}. Nunca forneça nem sugira o e-mail pessoal antigo; use sempre {CONTACT_EMAIL} quando o usuário pedir contato por e-mail. \
    Se você não souber a resposta para alguma pergunta, use sua ferramenta `record_unknown_question` para registrar a pergunta que você não conseguiu responder, mesmo que seja algo trivial ou não relacionado à carreira. \
    Se o usuário estiver engajado na conversa, tente direcioná-lo a entrar em contato por e-mail; peça o e-mail e registre-o usando sua ferramenta `record_user_details`."

        system_prompt += f"\n\n## Resumo:\n{self.summary}\n\n## Perfil do LinkedIn:\n{self.linkedin}\n\n## Currículo Lattes (formação e produção acadêmica):\n{self.lattes}\n\n"
        system_prompt += f"Com esse contexto, por favor converse com o usuário, sempre mantendo o personagem de {self.name}."
        return system_prompt

    def chat(self, message, history, request: gr.Request):
        session_key = request.session_hash if request and request.session_hash else "default"
        session_record = self._get_session_record(session_key, reset=not history)

        user_message = message
        messages = (
            [{"role": "system", "content": self.system_prompt()}]
            + history
            + [{"role": "user", "content": user_message}]
        )
        done = False
        while not done:
            response = self.openai.chat.completions.create(
                model="gpt-5-nano-2025-08-07",
                messages=messages,
                tools=tools,
            )
            if response.choices[0].finish_reason == "tool_calls":
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(message)
                messages.extend(results)
            else:
                done = True
        assistant_reply = response.choices[0].message.content
        session_sha = log_chat_interaction(
            user_message,
            assistant_reply,
            history,
            session_record["session_path"],
            session_record["session_sha"],
        )
        self._update_session_sha(session_key, session_sha)
        return assistant_reply


if __name__ == "__main__":
    me = Me()
    gr.ChatInterface(me.chat, type="messages").launch(share=True)
