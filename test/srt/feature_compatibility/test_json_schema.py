import json
import unittest
from concurrent.futures import ThreadPoolExecutor

import openai
import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)


def setup_class(cls, backend: str, model: str, tp: int):
    cls.model = model
    cls.base_url = DEFAULT_URL_FOR_TEST
    cls.json_schema = json.dumps(
        {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "population": {"type": "integer"},
            },
            "required": ["name", "population"],
            "additionalProperties": False,
        }
    )

    other_args = [
        "--grammar-backend",
        backend,
        "--tp",
        str(tp),
    ]

    cls.process = popen_launch_server(
        cls.model,
        cls.base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=other_args,
    )


class TestJSONSchemaBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup_class(cls, backend="outlines", model=DEFAULT_SMALL_MODEL_NAME_FOR_TEST)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_json_openai(self):
        client = openai.Client(api_key="EMPTY", base_url=f"{self.base_url}/v1")

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant"},
                {"role": "user", "content": "Introduce the capital of France."},
            ],
            temperature=0,
            max_tokens=128,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "foo", "schema": json.loads(self.json_schema)},
            },
        )
        text = response.choices[0].message.content

        try:
            js_obj = json.loads(text)
        except (TypeError, json.decoder.JSONDecodeError):
            print("JSONDecodeError", text)
            raise

        self.assertIsInstance(js_obj["name"], str)
        self.assertIsInstance(js_obj["population"], int)


# MetaLlama_3_1_8BInstruct

class MetaLlama_3_1_8BInstructOutlines(TestJSONSchemaBase):
    @classmethod
    def setUpClass(cls):
        setup_class(cls, backend="outlines", model = "meta-llama/Llama-3.1-8B-Instruct", tp = 1)

class MetaLlama_3_1_8BInstructXGrammar(TestJSONSchemaBase):
    @classmethod
    def setUpClass(cls):
        setup_class(cls, backend="xgrammar", model = "meta-llama/Llama-3.1-8B-Instruct", tp = 1)  

class MetaLlama_3_1_8BInstructLLGuidance(TestJSONSchemaBase):
    @classmethod
    def setUpClass(cls):
        setup_class(cls, backend="llguidance", model = "meta-llama/Llama-3.1-8B-Instruct", tp = 1)      

# MetaLlama_3_1_70BInstruct

class MetaLlama_3_1_70BInstructOutlines(TestJSONSchemaBase):
    @classmethod
    def setUpClass(cls):
        setup_class(cls, backend="outlines", model = "meta-llama/Llama-3.1-70B-Instruct", tp = 2)

class MetaLlama_3_1_70BInstructXGrammar(TestJSONSchemaBase):
    @classmethod
    def setUpClass(cls):
        setup_class(cls, backend="xgrammar", model = "meta-llama/Llama-3.1-70B-Instruct", tp = 2)

class MetaLlama_3_1_70BInstructLLGuidance(TestJSONSchemaBase):
    @classmethod
    def setUpClass(cls):
        setup_class(cls, backend="llguidance", model = "meta-llama/Llama-3.1-70B-Instruct", tp = 2)

# MetaLlama_3_2_11BVisionInstruct

class MetaLlama_3_2_11BVisionInstruct(TestJSONSchemaBase):
    @classmethod
    def setUpClass(cls):
        setup_class(cls, backend="outlines", model = "meta-llama/Llama-3.2-11B-Vision-Instruct", tp = 2)

class MetaLlama_3_2_11BVisionInstructXGrammar(TestJSONSchemaBase):
    @classmethod
    def setUpClass(cls):
        setup_class(cls, backend="xgrammar", model = "meta-llama/Llama-3.2-11B-Vision-Instruct", tp = 2)  

class MetaLlama_3_2_11BVisionInstructLLGuidance(TestJSONSchemaBase):
    @classmethod
    def setUpClass(cls):
        setup_class(cls, backend="llguidance", model = "meta-llama/Llama-3.2-11B-Vision-Instruct", tp = 2)                

# MetaLlama_3_3_70BInstruct

class MetaLlama_3_3_70BInstructOutlines(TestJSONSchemaBase):
    @classmethod
    def setUpClass(cls):
        setup_class(cls, backend="outlines", model = "meta-llama/Llama-3.3-70B-Instruct", tp = 2)

class MetaLlama_3_3_70BInstructXGrammar(TestJSONSchemaBase):
    @classmethod
    def setUpClass(cls):
        setup_class(cls, backend="xgrammar", model = "meta-llama/Llama-3.3-70B-Instruct", tp = 2)        

class MetaLlama_3_3_70BInstructLLGuidance(TestJSONSchemaBase):
    @classmethod
    def setUpClass(cls):
        setup_class(cls, backend="llguidance", model = "meta-llama/Llama-3.3-70B-Instruct", tp = 2)                

# MistralNemo12BInstruct

class MistralNemo12BInstructOutlines(TestJSONSchemaBase):
    @classmethod
    def setUpClass(cls):
        setup_class(cls, backend="outlines", model = "nvidia/Mistral-NeMo-12B-Instruct", tp = 1)

class MistralNemo12BInstructXGrammar(TestJSONSchemaBase):
    @classmethod
    def setUpClass(cls):
        setup_class(cls, backend="xgrammar", model = "nvidia/Mistral-NeMo-12B-Instruct", tp = 1)        

class MistralNemo12BInstructLLGuidance(TestJSONSchemaBase):
    @classmethod
    def setUpClass(cls):
        setup_class(cls, backend="llguidance", model = "nvidia/Mistral-NeMo-12B-Instruct", tp = 1)                

if __name__ == "__main__":
    unittest.main()
