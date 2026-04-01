"""
Pipeline de pré-processamento textual para conversas.
Baseado no pré-processamento descrito em Panzariello (2022).
"""

import re
import os
from lxml import etree


def load_conversations(xml_path: str) -> list[dict]:
    """Carrega conversas de um arquivo XML no formato PAN 2012."""
    tree = etree.parse(xml_path)
    conversations = []

    for conv in tree.findall(".//conversation"):
        messages = []
        for msg in conv.findall("message"):
            messages.append({
                "line": int(msg.get("line", 0)),
                "author": msg.get("author", ""),
                "time": msg.get("time", ""),
                "text": (msg.text or "").strip(),
            })
        conversations.append({
            "id": conv.get("id", ""),
            "messages": messages,
        })

    return conversations


def preprocess_text(text: str) -> str:
    """
    Pré-processa uma mensagem de chat.
    Passos (conforme Panzariello, 2022):
    1. Converter para minúsculas
    2. Remover URLs
    3. Remover entidades HTML
    4. Remover caracteres especiais excessivos
    5. Normalizar repetições de letras (ex: 'aaaa' -> 'aa')
    6. Normalizar repetições de palavras
    7. Remover espaços extras
    """
    # 1. Lowercase
    text = text.lower()

    # 2. Remover URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # 3. Remover entidades HTML
    text = re.sub(r"&[a-zA-Z]+;", " ", text)

    # 4. Remover caracteres especiais (manter letras, números, espaços e pontuação básica)
    text = re.sub(r"[^\w\s.,!?]", " ", text)

    # 5. Normalizar repetições de letras (3+ -> 2)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # 6. Normalizar repetições de palavras
    text = re.sub(r"\b(\w+)(\s+\1){2,}\b", r"\1", text)

    # 7. Remover espaços extras
    text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_first_n_messages(conversations: list[dict], n: int) -> list[dict]:
    """Extrai as primeiras N mensagens de cada conversa."""
    result = []
    for conv in conversations:
        truncated = conv.copy()
        truncated["messages"] = conv["messages"][:n]
        result.append(truncated)
    return result


def conversation_to_text(conversation: dict) -> str:
    """Concatena todas as mensagens de uma conversa em um único texto."""
    texts = [preprocess_text(msg["text"]) for msg in conversation["messages"]]
    return " ".join(texts)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pré-processamento de conversas")
    parser.add_argument("--input", required=True, help="Diretório com XMLs de entrada")
    parser.add_argument("--output", required=True, help="Diretório de saída")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Processando arquivos em {args.input}...")
    # TODO: implementar pipeline completo
    print("Pipeline de pré-processamento em desenvolvimento.")
