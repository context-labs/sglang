import json
import time
import unittest
from typing import Optional

import openai

from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)


def setup_class(cls, tool_call_parser: str, grammar_backend: str, tp: int):
    cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
    cls.tool_call_parser = tool_call_parser
    cls.tp = tp
    cls.base_url = DEFAULT_URL_FOR_TEST
    cls.api_key = "sk-123456"
    cls.grammar_backend = grammar_backend

    # Start the local OpenAI Server
    cls.process = popen_launch_server(
        cls.model,
        cls.base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        api_key=cls.api_key,
        other_args=[
            "--tool-call-parser",
            cls.tool_call_parser,
            "--tp",
            str(cls.tp),
            "--grammar-backend",
            cls.grammar_backend,
        ],
    )
    cls.base_url += "/v1"
    cls.tokenizer = get_tokenizer(cls.model)


class OpenAIServerFunctionCallingBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup_class(cls, tool_call_parser="llama3", grammar_backend="outlines", tp=1)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_function_calling_format_no_tool_choice_specified(self):
        """
        Test: Whether the function call format returned by the AI is correct.
        When returning a tool call, message.content should be None, and tool_calls should be a list.
        """
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        tools = [self.get_add_tool()]

        messages = [{"role": "user", "content": "Compute (3+5)"}]
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.8,
            top_p=0.8,
            stream=False,
            tools=tools,
        )

        self.assert_tool_call_format(response, expected_function_name="add", expected_function_arguments=["a", "b"])
    
    def test_function_calling_named_tool_choice(self):
        """
        Test: Whether the function call format returned by the AI is correct when using named function tool choice.
        When returning a tool call, message.content should be None, and tool_calls should be a list.
        """
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        tools = [self.get_add_tool()]

        messages = [{"role": "user", "content": "Compute (3+5)"}]
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.8,
            top_p=0.8,
            stream=False,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "add"}}
        )

        self.assert_tool_call_format(response, expected_function_name="add", expected_function_arguments=["a", "b"])

    def test_function_calling_required_tool_choice(self):
        """
        Test: Whether the function call format returned by the AI is correct when using required function tool choice.
        When returning a tool call, message.content should be None, and tool_calls should be a list.
        """
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        tools = [self.get_add_tool()]

        messages = [{"role": "user", "content": "Compute (3+5)"}]
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.8,
            top_p=0.8,
            stream=False,
            tools=tools,
            tool_choice={"type": "required"}
        )

        self.assert_tool_call_format(response, expected_function_name="add", expected_function_arguments=["a", "b"])

    def test_function_calling_auto_tool_choice(self):
        """
        Test: Whether the function call format returned by the AI is correct when using auto function tool choice.
        When returning a tool call, message.content should be None, and tool_calls should be a list.
        """
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        tools = [self.get_add_tool()]

        messages = [{"role": "user", "content": "Compute (3+5)"}]
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.8,
            top_p=0.8,
            stream=False,
            tools=tools,
            tool_choice={"type": "auto"}
        )

        self.assert_tool_call_format(response, expected_function_name="add", expected_function_arguments=["a", "b"])

    def test_function_calling_streaming_args_parsing(self):
        """
        Test: Whether the function call arguments returned in streaming mode can be correctly concatenated into valid JSON.
        - The user request requires multiple parameters.
        - AI may return the arguments in chunks that need to be concatenated.
        """
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        tools = [
            self.get_add_tool()
        ]

        messages = [
            {"role": "user", "content": "Please sum 5 and 7, just call the function."}
        ]

        response_stream = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.9,
            top_p=0.9,
            stream=True,
            tools=tools,
        )

        argument_fragments = []
        function_name = None
        for chunk in response_stream:
            choice = chunk.choices[0]
            if choice.delta.tool_calls:
                tool_call = choice.delta.tool_calls[0]
                # Record the function name on first occurrence
                function_name = tool_call.function.name or function_name
                # In case of multiple chunks, JSON fragments may need to be concatenated
                if tool_call.function.arguments:
                    argument_fragments.append(tool_call.function.arguments)

        self.assertEqual(function_name, "add", "Function name should be 'add'")
        joined_args = "".join(argument_fragments)
        self.assertTrue(
            len(joined_args) > 0,
            "No parameter fragments were returned in the function call",
        )

        # Check whether the concatenated JSON is valid
        try:
            args_obj = json.loads(joined_args)
        except json.JSONDecodeError:
            self.fail(
                "The concatenated tool call arguments are not valid JSON, parsing failed"
            )

        self.assertIn("a", args_obj, "Missing parameter 'a'")
        self.assertIn("b", args_obj, "Missing parameter 'b'")
        self.assertEqual(
            args_obj["a"],
            5,
            "Parameter a should be 5",
        )
        self.assertEqual(args_obj["b"], 7, "Parameter b should be 7")


    def assert_tool_call_format(self, response, expected_function_name : Optional[str] = None):
        content = response.choices[0].message.content
        tool_calls = response.choices[0].message.tool_calls

        assert content is None, (
            "When function call is successful, message.content should be None, "
            f"but got: {content}"
        )
        assert (
            isinstance(tool_calls, list) and len(tool_calls) > 0
        ), "tool_calls should be a non-empty list"

        function_name = tool_calls[0].function.name
        if expected_function_name is not None:
            assert function_name == expected_function_name, f"Function name should be '{expected_function_name}'"

    def get_add_tool(self):
        return {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Compute the sum of two numbers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {
                            "type": "int",
                            "description": "A number",
                        },
                        "b": {
                            "type": "int",
                            "description": "A number",
                        },
                    },
                    "required": ["a", "b"],
                },
            },
        }

    def get_weather_tool(self):
        return {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The city to find the weather for",
                        },
                        "unit": {
                            "type": "string",
                            "description": "Weather unit (celsius or fahrenheit)",
                            "enum": ["celsius", "fahrenheit"],
                        },
                        "required": ["city", "unit"],
                    },
                },
            }
        }


class MetaLlama_3_1_8BInstruct(OpenAIServerFunctionCallingBase):
    @classmethod
    def setUpClass(cls):
        setup_class(cls, tool_call_parser="llama3", grammar_backend="outlines", tp=1)


class MetaLlama_3_1_70BInstruct(OpenAIServerFunctionCallingBase):
    @classmethod
    def setUpClass(cls):
        setup_class(cls, tool_call_parser="llama3", grammar_backend="outlines", tp=2)


class MetaLlama_3_2_11BVisionInstruct(OpenAIServerFunctionCallingBase):
    @classmethod
    def setUpClass(cls):
        setup_class(cls, tool_call_parser="llama3", grammar_backend="outlines", tp=1)


class MetaLlama_3_3_70BInstruct(OpenAIServerFunctionCallingBase):
    @classmethod
    def setUpClass(cls):
        setup_class(cls, tool_call_parser="llama3", grammar_backend="outlines", tp=2)


class MistralNemo12BInstruct(OpenAIServerFunctionCallingBase):
    @classmethod
    def setUpClass(cls):
        setup_class(cls, tool_call_parser="mistral", grammar_backend="outlines", tp=1)


if __name__ == "__main__":
    unittest.main()
