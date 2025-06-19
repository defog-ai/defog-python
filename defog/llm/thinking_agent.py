"""Agent with enhanced thinking and reasoning capabilities."""

import json
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime

from .orchestrator import Agent
from .shared_context import SharedContextStore, ArtifactType
from .enhanced_memory import EnhancedMemoryManager
from .utils import chat_async
from .providers.base import LLMResponse

logger = logging.getLogger(__name__)


class ThinkingAgent(Agent):
    """
    Agent with enhanced thinking capabilities and scratchpad functionality.

    This agent extends the base Agent with:
    - Extended thinking mode for complex reasoning
    - Scratchpad for intermediate thoughts and plans
    - Integration with shared context for storing reasoning artifacts
    - Alternative solution exploration
    """

    def __init__(
        self,
        agent_id: str,
        provider: Union[str, Any],
        model: str,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Callable]] = None,
        memory_config: Optional[Dict[str, Any]] = None,
        reasoning_effort: Optional[str] = "medium",
        shared_context_store: Optional[SharedContextStore] = None,
        enable_thinking_mode: bool = True,
        thinking_model: Optional[str] = None,
        thinking_provider: Optional[str] = None,
        max_thinking_tokens: int = 8000,
        **kwargs,
    ):
        """
        Initialize ThinkingAgent.

        Args:
            agent_id: Unique identifier for the agent
            provider: LLM provider
            model: Model to use for main processing
            system_prompt: System prompt for the agent
            tools: List of tools available to the agent
            memory_config: Configuration for memory management
            reasoning_effort: Default reasoning effort level
            shared_context_store: Shared context for storing artifacts
            enable_thinking_mode: Whether to enable thinking mode
            thinking_model: Optional different model for thinking (defaults to main model)
            thinking_provider: Optional different provider for thinking
            max_thinking_tokens: Maximum tokens for thinking phase
            **kwargs: Additional arguments for base Agent
        """
        super().__init__(
            agent_id=agent_id,
            provider=provider,
            model=model,
            system_prompt=system_prompt,
            tools=tools,
            memory_config=memory_config,
            reasoning_effort=reasoning_effort,
            **kwargs,
        )

        self.shared_context = shared_context_store
        self.enable_thinking_mode = enable_thinking_mode
        self.thinking_model = thinking_model or model
        self.thinking_provider = thinking_provider or provider
        self.max_thinking_tokens = max_thinking_tokens

        # Initialize enhanced memory if shared context is available
        if self.shared_context and self.memory_manager:
            self.memory_manager = EnhancedMemoryManager(
                token_threshold=self.memory_manager.token_threshold,
                preserve_last_n_messages=self.memory_manager.preserve_last_n_messages,
                summary_max_tokens=self.memory_manager.summary_max_tokens,
                enabled=self.memory_manager.enabled,
                shared_context_store=self.shared_context,
                agent_id=self.agent_id,
            )

        logger.info(
            f"ThinkingAgent {agent_id} initialized with thinking_mode={enable_thinking_mode}"
        )

    async def think(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        save_to_shared_context: bool = True,
    ) -> str:
        """
        Extended thinking mode for planning and reasoning.

        Args:
            prompt: The thinking prompt/question
            context: Optional context for thinking
            save_to_shared_context: Whether to save thinking results

        Returns:
            The thinking output
        """
        thinking_messages = [
            {
                "role": "system",
                "content": """You are in extended thinking mode. This is your scratchpad for:
- Breaking down complex problems
- Exploring different approaches
- Planning your strategy
- Considering edge cases and potential issues
- Reflecting on the best solution

Be thorough and consider multiple perspectives. This is your space to think deeply.""",
            },
            {"role": "user", "content": f"Think about this: {prompt}"},
        ]

        if context:
            thinking_messages[0]["content"] += (
                f"\n\nContext:\n{json.dumps(context, indent=2)}"
            )

        try:
            # Use reasoning_effort="medium" for thinking
            response = await chat_async(
                provider=self.thinking_provider,
                model=self.thinking_model,
                messages=thinking_messages,
                max_completion_tokens=self.max_thinking_tokens,
                temperature=0.3,
                reasoning_effort="medium",  # Always use medium for thinking
            )

            thinking_result = response.content

            # Save to shared context if enabled
            if save_to_shared_context and self.shared_context:
                thinking_key = f"thinking/{self.agent_id}/{datetime.now().isoformat()}"
                await self.shared_context.write_artifact(
                    agent_id=self.agent_id,
                    key=thinking_key,
                    content={
                        "prompt": prompt,
                        "thinking": thinking_result,
                        "context": context,
                    },
                    artifact_type=ArtifactType.PLAN,
                    metadata={
                        "tokens": response.output_tokens,
                        "model": self.thinking_model,
                    },
                )

            return thinking_result

        except Exception as e:
            logger.error(f"Thinking phase failed: {e}")
            return f"Thinking phase failed: {str(e)}"

    async def process_with_thinking(
        self,
        messages: List[Dict[str, str]],
        context: Optional[Dict[str, Any]] = None,
        think_first: bool = True,
        thinking_prompt: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Process messages with optional thinking phase.

        Args:
            messages: Input messages
            context: Optional context
            think_first: Whether to think before processing
            thinking_prompt: Custom thinking prompt (if None, derives from messages)
            **kwargs: Additional arguments for process()

        Returns:
            LLM response
        """
        # Prepare final context
        final_context = context or {}

        # Perform thinking phase if enabled
        if think_first and self.enable_thinking_mode:
            # Derive thinking prompt if not provided
            if not thinking_prompt:
                last_user_message = None
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        last_user_message = msg.get("content", "")
                        break

                thinking_prompt = last_user_message or "the current task"

            # Think about the problem
            thinking_result = await self.think(prompt=thinking_prompt, context=context)

            # Add thinking result to context
            final_context["thinking_result"] = thinking_result

            logger.debug(f"Agent {self.agent_id} completed thinking phase")

        # Process with the enhanced context
        return await self.process(messages=messages, context=final_context, **kwargs)

    async def explore_alternatives(
        self,
        task: str,
        current_approach: Optional[str] = None,
        num_alternatives: int = 3,
    ) -> List[Dict[str, str]]:
        """
        Explore alternative approaches to a task.

        Args:
            task: The task to explore alternatives for
            current_approach: Current approach being considered
            num_alternatives: Number of alternatives to generate

        Returns:
            List of alternative approaches with explanations
        """
        exploration_prompt = f"""Task: {task}
        
{"Current approach: " + current_approach if current_approach else ""}

Generate {num_alternatives} alternative approaches to accomplish this task.
For each alternative, provide:
1. A brief description of the approach
2. Key advantages
3. Potential challenges
4. When this approach would be most suitable"""

        exploration_messages = [
            {
                "role": "system",
                "content": "You are exploring alternative solutions. Be creative and consider different paradigms.",
            },
            {"role": "user", "content": exploration_prompt},
        ]

        try:
            response = await chat_async(
                provider=self.provider,
                model=self.model,
                messages=exploration_messages,
                temperature=0.7,  # Higher temperature for creativity
                reasoning_effort="medium",
            )

            # Parse alternatives (in a real implementation, you might want structured output)
            alternatives_text = response.content

            # Store exploration results
            if self.shared_context:
                exploration_key = (
                    f"exploration/{self.agent_id}/{datetime.now().isoformat()}"
                )
                await self.shared_context.write_artifact(
                    agent_id=self.agent_id,
                    key=exploration_key,
                    content={
                        "task": task,
                        "current_approach": current_approach,
                        "alternatives": alternatives_text,
                    },
                    artifact_type=ArtifactType.EXPLORATION,
                    metadata={"num_alternatives": num_alternatives},
                )

            # Return parsed alternatives (simplified for now)
            return [
                {"approach": f"Alternative {i + 1}", "description": alternatives_text}
                for i in range(num_alternatives)
            ]

        except Exception as e:
            logger.error(f"Alternative exploration failed: {e}")
            return []

    async def reflect_on_results(
        self, task: str, results: Any, save_reflection: bool = True
    ) -> Dict[str, Any]:
        """
        Reflect on task results to extract insights and improvements.

        Args:
            task: The completed task
            results: The results obtained
            save_reflection: Whether to save reflection to shared context

        Returns:
            Reflection insights
        """
        reflection_prompt = f"""Reflect on the following completed task and results:

Task: {task}

Results: {json.dumps(results, indent=2) if isinstance(results, (dict, list)) else str(results)}

Please provide:
1. Was the task completed successfully?
2. What worked well?
3. What could be improved?
4. Any insights for similar future tasks?
5. Rate the solution quality (1-10) with justification"""

        reflection_result = await self.think(
            prompt=reflection_prompt, save_to_shared_context=save_reflection
        )

        return {
            "task": task,
            "reflection": reflection_result,
            "timestamp": datetime.now().isoformat(),
        }

    async def collaborate_with_agent(
        self, other_agent_id: str, collaboration_context: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve and process relevant context from another agent.

        Args:
            other_agent_id: ID of the agent to collaborate with
            collaboration_context: Context about what to collaborate on

        Returns:
            Collaboration insights if available
        """
        if not self.shared_context or not isinstance(
            self.memory_manager, EnhancedMemoryManager
        ):
            logger.warning("Collaboration requires shared context and enhanced memory")
            return None

        # Get other agent's recent artifacts
        other_artifacts = await self.shared_context.list_artifacts(
            agent_id=other_agent_id,
            since=datetime.now().replace(
                hour=0, minute=0, second=0
            ),  # Today's artifacts
        )

        if not other_artifacts:
            return None

        # Get cross-agent memories
        cross_memories = await self.memory_manager.get_cross_agent_context(
            other_agent_ids=[other_agent_id], max_entries=5
        )

        # Prepare collaboration summary
        collaboration_data = {
            "other_agent_id": other_agent_id,
            "context": collaboration_context,
            "recent_artifacts": [
                {
                    "key": a.key,
                    "type": a.artifact_type.value,
                    "created": a.created_at.isoformat(),
                }
                for a in other_artifacts[:5]
            ],
            "shared_memories": len(cross_memories),
        }

        # Think about the collaboration
        collaboration_thinking = await self.think(
            prompt=f"How can I best collaborate with {other_agent_id} on: {collaboration_context}",
            context=collaboration_data,
        )

        collaboration_data["collaboration_strategy"] = collaboration_thinking

        return collaboration_data
