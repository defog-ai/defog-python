import unittest
from defog.llm.utils import LLMResponse, LLM_COSTS_PER_TOKEN


class TestLLMResponse(unittest.TestCase):
    def test_token_costs(self):
        # Test data
        test_cases = [
            {
                "model_name": "gpt-4o",
                "input_tokens": 1000,
                "cached_input_tokens": 500,
                "expected_cost": (
                    LLM_COSTS_PER_TOKEN["gpt-4o"]["input_cost_per1k"]
                    + LLM_COSTS_PER_TOKEN["gpt-4o"]["cached_input_cost_per1k"] * 0.5
                    + LLM_COSTS_PER_TOKEN["gpt-4o"]["output_cost_per1k"]
                ),
            },
            {
                "model_name": "gpt-4o-mini",
                "input_tokens": 1000,
                "cached_input_tokens": 500,
                "expected_cost": (
                    LLM_COSTS_PER_TOKEN["gpt-4o-mini"]["input_cost_per1k"]
                    + LLM_COSTS_PER_TOKEN["gpt-4o-mini"]["cached_input_cost_per1k"]
                    * 0.5
                    + LLM_COSTS_PER_TOKEN["gpt-4o-mini"]["output_cost_per1k"]
                ),
            },
            {
                "model_name": "o1",
                "input_tokens": 1000,
                "cached_input_tokens": 500,
                "expected_cost": (
                    LLM_COSTS_PER_TOKEN["o1"]["input_cost_per1k"]
                    + LLM_COSTS_PER_TOKEN["o1"]["cached_input_cost_per1k"] * 0.5
                    + LLM_COSTS_PER_TOKEN["o1"]["output_cost_per1k"]
                ),
            },
            {
                "model_name": "o1-mini",
                "input_tokens": 1000,
                "cached_input_tokens": 500,
                "expected_cost": (
                    LLM_COSTS_PER_TOKEN["o1-mini"]["input_cost_per1k"]
                    + LLM_COSTS_PER_TOKEN["o1-mini"]["cached_input_cost_per1k"] * 0.5
                    + LLM_COSTS_PER_TOKEN["o1-mini"]["output_cost_per1k"]
                ),
            },
            {
                "model_name": "o3-mini",
                "input_tokens": 1000,
                "cached_input_tokens": 500,
                "expected_cost": (
                    LLM_COSTS_PER_TOKEN["o3-mini"]["input_cost_per1k"]
                    + LLM_COSTS_PER_TOKEN["o3-mini"]["cached_input_cost_per1k"] * 0.5
                    + LLM_COSTS_PER_TOKEN["o3-mini"]["output_cost_per1k"]
                ),
            },
            {
                "model_name": "deepseek-chat",
                "input_tokens": 1000,
                "cached_input_tokens": 500,
                "expected_cost": (
                    LLM_COSTS_PER_TOKEN["deepseek-chat"]["input_cost_per1k"]
                    + LLM_COSTS_PER_TOKEN["deepseek-chat"]["cached_input_cost_per1k"]
                    * 0.5
                    + LLM_COSTS_PER_TOKEN["deepseek-chat"]["output_cost_per1k"]
                ),
            },
            {
                "model_name": "deepseek-reasoner",
                "input_tokens": 1000,
                "cached_input_tokens": 500,
                "expected_cost": (
                    LLM_COSTS_PER_TOKEN["deepseek-reasoner"]["input_cost_per1k"]
                    + LLM_COSTS_PER_TOKEN["deepseek-reasoner"][
                        "cached_input_cost_per1k"
                    ]
                    * 0.5
                    + LLM_COSTS_PER_TOKEN["deepseek-reasoner"]["output_cost_per1k"]
                ),
            },
            {
                "model_name": "claude-3-5-sonnet",
                "input_tokens": 1000,
                "cached_input_tokens": 0,
                "expected_cost": (
                    LLM_COSTS_PER_TOKEN["claude-3-5-sonnet"]["input_cost_per1k"]
                    + LLM_COSTS_PER_TOKEN["claude-3-5-sonnet"]["output_cost_per1k"]
                ),
            },
        ]

        for case in test_cases:
            with self.subTest(case=case):
                response = LLMResponse(
                    model=case["model_name"],
                    time=0.0,
                    input_tokens=case["input_tokens"],
                    cached_input_tokens=case["cached_input_tokens"],
                    output_tokens=1000,
                    content="",
                )
                response.__post_init__()  # Ensure cost calculation
                expected_cost_in_cents = case["expected_cost"] * 100  # Convert to cents
                self.assertAlmostEqual(
                    response.cost_in_cents, expected_cost_in_cents, places=10
                )


if __name__ == "__main__":
    unittest.main()
