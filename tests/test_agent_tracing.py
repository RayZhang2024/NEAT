from __future__ import annotations

from contextlib import nullcontext
from unittest import TestCase
from unittest.mock import patch

from NEAT.services.agent_tracing import (
    TracedLLMCallResult,
    agent_langsmith_context,
    openai_chat_completion_usage_metadata,
    trace_llm_call,
)


class AgentTracingTests(TestCase):
    def test_openai_usage_is_normalized_for_langsmith(self):
        self.assertEqual(
            openai_chat_completion_usage_metadata(
                {
                    "usage": {
                        "prompt_tokens": 13,
                        "completion_tokens": 7,
                        "total_tokens": 20,
                    }
                }
            ),
            {"input_tokens": 13, "output_tokens": 7, "total_tokens": 20},
        )

    def test_llm_usage_metadata_is_attached_to_the_trace_run(self):
        run_tree = type("RunTree", (), {"metadata": {}})()
        output = {"status_code": 200, "body": "{}"}

        def traceable_factory(**_options):
            def decorate(function):
                def traced(inputs):
                    return function(inputs)

                return traced

            return decorate

        with (
            patch("langsmith.traceable", side_effect=traceable_factory),
            patch(
                "langsmith.run_helpers.get_tracing_context",
                return_value={"enabled": True},
            ),
            patch(
                "langsmith.run_helpers.get_current_run_tree",
                return_value=run_tree,
            ),
        ):
            result = trace_llm_call(
                name="Test model request",
                provider="openai",
                model_id="gpt-test",
                inputs={},
                operation=lambda: TracedLLMCallResult(
                    output=output,
                    usage_metadata={
                        "input_tokens": 13,
                        "output_tokens": 7,
                        "total_tokens": 20,
                    },
                ),
            )

        self.assertIs(result, output)
        self.assertEqual(
            run_tree.metadata["usage_metadata"],
            {"input_tokens": 13, "output_tokens": 7, "total_tokens": 20},
        )

    def test_llm_call_creates_langsmith_llm_run_with_inputs_and_outputs(self):
        observed = {}
        inputs = {
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "test"}],
        }
        outputs = {"status_code": 200, "body": '{"ok":true}'}
        operation_calls = []

        def traceable_factory(**options):
            observed["options"] = options

            def decorate(function):
                def traced(inputs):
                    observed["inputs"] = inputs
                    result = function(inputs)
                    observed["outputs"] = result
                    return result

                return traced

            return decorate

        def operation():
            operation_calls.append(True)
            return outputs

        with (
            patch("langsmith.traceable", side_effect=traceable_factory),
            patch(
                "langsmith.run_helpers.get_tracing_context",
                return_value={"enabled": True},
            ),
        ):
            result = trace_llm_call(
                name="Test model request",
                provider="openai",
                model_id="gpt-test",
                inputs=inputs,
                operation=operation,
            )

        self.assertIs(result, outputs)
        self.assertEqual(operation_calls, [True])
        self.assertEqual(observed["options"]["name"], "Test model request")
        self.assertEqual(observed["options"]["run_type"], "llm")
        self.assertEqual(
            observed["options"]["metadata"],
            {"ls_provider": "openai", "ls_model_name": "gpt-test"},
        )
        self.assertEqual(observed["inputs"], inputs)
        self.assertIs(observed["outputs"], outputs)

    def test_enabled_trace_uses_unfiltered_client_and_agent_address(self):
        with (
            patch("langsmith.Client") as client_factory,
            patch("langsmith.run_helpers.tracing_context", return_value=nullcontext()) as trace,
            agent_langsmith_context(
                enabled=True,
                api_key="ls-test-secret",
                agent_id="neat-test-agent",
                environment="development",
                region="eu",
                workspace_id="workspace-test-id",
                workflow="neat_location_selection",
            ) as active,
        ):
            self.assertTrue(active)

        client_factory.assert_called_once()
        options = client_factory.call_args.kwargs
        self.assertEqual(options["api_key"], "ls-test-secret")
        self.assertEqual(options["api_url"], "https://eu.api.smith.langchain.com")
        self.assertEqual(options["workspace_id"], "workspace-test-id")
        self.assertFalse(options["hide_inputs"])
        self.assertFalse(options["hide_outputs"])
        self.assertFalse(options["hide_metadata"])
        self.assertIsNone(options["anonymizer"])
        self.assertFalse(options["omit_traced_runtime_info"])
        trace.assert_called_once()
        self.assertEqual(
            trace.call_args.kwargs["address"],
            {
                "kind": "AGENT",
                "id": "neat-test-agent",
                "environment": "DEVELOPMENT",
            },
        )

    def test_disabled_trace_overrides_global_environment_setting(self):
        with (
            patch(
                "langsmith.run_helpers.tracing_context", return_value=nullcontext()
            ) as trace,
            agent_langsmith_context(
                enabled=False,
                api_key="unused",
                agent_id="neat-test-agent",
                environment="local",
                region="eu",
                workflow="neat_read_only_plan",
            ) as active,
        ):
            self.assertFalse(active)

        trace.assert_called_once_with(enabled=False)
