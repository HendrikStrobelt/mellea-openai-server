"""Test client for the mellea OpenAI-compatible server.

Exercises /v1/chat/completions in both non-streaming and streaming modes
using the standard openai SDK — confirming full API compatibility.

Usage:
    # Start the server first:
    python main.py

    # Then run this script:
    python examples/test_chat.py [--model MODEL] [--url URL]
"""

import argparse

import openai

BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "granite-4.0-micro"  # replace with any model ID served by LMStudio
MESSAGES = [{"role": "user", "content": "In one sentence, what is the capital of France?"}]


def test_non_streaming(client: openai.OpenAI, model: str) -> None:
    print("=== Non-streaming ===")
    response = client.chat.completions.create(
        model=model,
        messages=MESSAGES,
        stream=False,
    )
    print(f"id:      {response.id}")
    print(f"model:   {response.model}")
    print(f"content: {response.choices[0].message.content}")
    print()


def test_streaming(client: openai.OpenAI, model: str) -> None:
    print("=== Streaming ===")
    stream = client.chat.completions.create(
        model=model,
        messages=MESSAGES,
        stream=True,
    )
    print("content: ", end="", flush=True)
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="-", flush=True)
    print("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test chat completions endpoint")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model ID to use")
    parser.add_argument("--url", default=BASE_URL, help="Server base URL")
    args = parser.parse_args()

    client = openai.OpenAI(api_key="not-needed", base_url=args.url)

    test_non_streaming(client, args.model)
    test_streaming(client, args.model)


if __name__ == "__main__":
    main()
