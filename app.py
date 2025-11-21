from dotenv import load_dotenv
from openai import OpenAI
import json
import os
from pathlib import Path
import requests
from PyPDF2 import PdfReader
import gradio as gr


load_dotenv(override=True)

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
            }
            ,
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

tools = [{"type": "function", "function": record_user_details_json},
        {"type": "function", "function": record_unknown_question_json}]


class Me:

    def __init__(self):
        self.openai = OpenAI()
        self.name = "Matheus Costa"
        base_dir = Path("me")
        self.linkedin = self._load_document(base_dir / "linkedin.txt", base_dir / "linkedin.pdf")
        self.lattes = self._load_document(base_dir / "lattes.txt", base_dir / "lattes.pdf")
        with open(base_dir / "summary.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()

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
            results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
        return results
    
    def system_prompt(self):
        system_prompt = f"Você está atuando como {self.name}. Você está respondendo perguntas no site de {self.name}, \
    particularmente perguntas relacionadas à carreira, histórico, habilidades e experiência de {self.name}. \
    Sua responsabilidade é representar {self.name} nas interações no site da forma mais fiel possível. \
    Você recebeu um resumo do histórico profissional, o perfil do LinkedIn e o currículo Lattes de {self.name}; use-os para responder perguntas, priorizando o Lattes quando o assunto envolver formação ou produção acadêmica. \
    Seja profissional e envolvente, como se estivesse conversando com um potencial cliente ou futuro empregador que acessou o site. \
    Se você não souber a resposta para alguma pergunta, use sua ferramenta `record_unknown_question` para registrar a pergunta que você não conseguiu responder, mesmo que seja algo trivial ou não relacionado à carreira. \
    Se o usuário estiver engajado na conversa, tente direcioná-lo a entrar em contato por e-mail; peça o e-mail e registre-o usando sua ferramenta `record_user_details`."

        system_prompt += f"\n\n## Resumo:\n{self.summary}\n\n## Perfil do LinkedIn:\n{self.linkedin}\n\n## Currículo Lattes (formação e produção acadêmica):\n{self.lattes}\n\n"
        system_prompt += f"Com esse contexto, por favor converse com o usuário, sempre mantendo o personagem de {self.name}."
        return system_prompt
    
    def chat(self, message, history):
        messages = [{"role": "system", "content": self.system_prompt()}] + history + [{"role": "user", "content": message}]
        done = False
        while not done:
            response = self.openai.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
            if response.choices[0].finish_reason=="tool_calls":
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(message)
                messages.extend(results)
            else:
                done = True
        return response.choices[0].message.content
    

if __name__ == "__main__":
    me = Me()
    gr.ChatInterface(me.chat, type="messages").launch(share=True)
