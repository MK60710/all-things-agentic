from types import SimpleNamespace

import pytest

from agent.general_chat import ChatTurn, GeneralChatAgent


def test_general_chat_sends_history_and_message_to_vertex_client():
    calls = []

    class FakeModels:
        def generate_content(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(text="A useful answer")

    agent = GeneralChatAgent(client=SimpleNamespace(models=FakeModels()))
    answer = agent.answer(
        "What should we explore next?",
        [
            ChatTurn(role="user", text="Help me study agents."),
            ChatTurn(role="assistant", text="Let's begin with memory."),
        ],
    )

    assert answer == "A useful answer"
    assert [content.role for content in calls[0]["contents"]] == [
        "user",
        "model",
        "user",
    ]
    assert calls[0]["contents"][-1].parts[0].text == "What should we explore next?"


def test_general_chat_without_vertex_configuration_fails_clearly():
    with pytest.raises(RuntimeError, match="not configured"):
        GeneralChatAgent().answer("hello")
