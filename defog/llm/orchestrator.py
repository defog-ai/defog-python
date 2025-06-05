"""Agent orchestration system for hierarchical task delegation."""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import inspect
import uuid
from pprint import pformat

logger = logging.getLogger(__name__)

from .utils import chat_async
from .utils_memory import chat_async_with_memory, create_memory_manager
from .memory.history_manager import MemoryManager
from .providers.base import BaseLLMProvider, LLMResponse
from .tools.handler import ToolHandler


class ExecutionMode(Enum):
    """Execution mode for subagent tasks."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


@dataclass
class SubAgentTask:
    """Task definition for a subagent."""
    agent_id: str
    task_description: str
    context: Optional[Dict[str, Any]] = None
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    dependencies: Optional[List[str]] = None  # IDs of tasks that must complete first


@dataclass
class SubAgentPlan:
    """Plan for creating a dynamic subagent."""
    agent_id: str
    system_prompt: str
    task_description: str
    tools: List[str]  # Tool names from available tools
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    dependencies: Optional[List[str]] = None


@dataclass
class SubAgentResult:
    """Result from a subagent execution."""
    agent_id: str
    task_id: str
    success: bool
    result: Any
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class Agent:
    """Base agent class with provider, model, tools, and memory support."""
    
    def __init__(
        self,
        agent_id: str,
        provider: Union[str, BaseLLMProvider],
        model: str,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Callable]] = None,
        memory_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.agent_id = agent_id
        self.provider = provider
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.kwargs = kwargs  # Additional params for chat_async
        
        # Initialize memory if configured
        self.memory_manager = None
        if memory_config:
            self.memory_manager = create_memory_manager(**memory_config)
    
    async def process(
        self,
        messages: List[Dict[str, str]],
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> LLMResponse:
        """Process messages with optional context injection."""
        # Prepare messages with system prompt and context
        final_messages = messages.copy()
        
        # Check if first message is already a system message
        has_system_message = len(final_messages) > 0 and final_messages[0].get("role") == "system"
        
        if self.system_prompt:
            system_content = self.system_prompt
            if context:
                system_content += f"\n\nContext:\n{json.dumps(context, indent=2)}"
            
            if has_system_message:
                # Update existing system message
                final_messages[0]["content"] = system_content + "\n\n" + final_messages[0]["content"]
            else:
                # Insert system message at the beginning
                final_messages.insert(0, {"role": "system", "content": system_content})
        
        # Merge kwargs
        call_kwargs = {**self.kwargs, **kwargs}
        
        # Use memory-enabled chat if memory manager exists
        if self.memory_manager:
            response = await chat_async_with_memory(
                provider=self.provider,
                model=self.model,
                messages=final_messages,
                tools=self.tools,
                memory_manager=self.memory_manager,
                **call_kwargs
            )
        else:
            response = await chat_async(
                provider=self.provider,
                model=self.model,
                messages=final_messages,
                tools=self.tools,
                **call_kwargs
            )
        
        return response
    
    def clear_memory(self):
        """Clear agent's memory if it exists."""
        if self.memory_manager:
            self.memory_manager.clear()


class AgentOrchestrator:
    """Orchestrator for managing main agent and subagents with task delegation."""
    
    def __init__(
        self,
        main_agent: Agent,
        max_parallel_tasks: int = 5,
        available_tools: Optional[List[Callable]] = None,
        subagent_provider: Optional[str] = None,
        subagent_model: Optional[str] = None,
        planning_provider: str = "anthropic",
        planning_model: str = "claude-opus-4-20250514"
    ):
        self.main_agent = main_agent
        self.subagents = {}
        self.max_parallel_tasks = max_parallel_tasks
        self.task_results: Dict[str, SubAgentResult] = {}
        
        # For dynamic subagent creation
        self.available_tools = available_tools or []
        self.tool_registry = {self._get_tool_name(tool): tool for tool in self.available_tools}
        self.subagent_provider = subagent_provider or main_agent.provider
        self.subagent_model = subagent_model or main_agent.model
        self.planning_provider = planning_provider
        self.planning_model = planning_model
        
        # Add delegation tools to main agent
        self._add_delegation_tool()
        if self.available_tools:
            self._add_dynamic_planning_tool()
    
    def _add_delegation_tool(self):
        """Add delegation tool to main agent's toolset."""
        from pydantic import BaseModel, Field
        from typing import List as ListType
        
        class DelegationRequest(BaseModel):
            """Request to delegate tasks to subagents."""
            tasks: ListType[Dict[str, Any]] = Field(
                description="List of tasks to delegate. Each task should have: agent_id, task_description, context (optional), execution_mode (optional), dependencies (optional)"
            )
        
        async def delegate_to_subagents(input: DelegationRequest) -> Dict[str, Any]:
            """Delegate tasks to subagents based on the request."""
            tasks = []
            for task_data in input.tasks:
                task = SubAgentTask(
                    agent_id=task_data["agent_id"],
                    task_description=task_data["task_description"],
                    context=task_data.get("context"),
                    execution_mode=ExecutionMode(task_data.get("execution_mode", "sequential")),
                    dependencies=task_data.get("dependencies")
                )
                tasks.append(task)
            
            results = await self._execute_subagent_tasks(tasks)
            
            # Format results for main agent
            formatted_results = {}
            for result in results:
                formatted_results[result.task_id] = {
                    "agent_id": result.agent_id,
                    "success": result.success,
                    "result": result.result,
                    "error": result.error,
                    "metadata": result.metadata
                }
            
            return formatted_results
        
        # Add to main agent's tools
        self.main_agent.tools.append(delegate_to_subagents)
    
    def register_subagent(self, agent: Agent):
        """Register a new subagent."""
        self.subagents[agent.agent_id] = agent
    
    def unregister_subagent(self, agent_id: str):
        """Unregister a subagent."""
        if agent_id in self.subagents:
            del self.subagents[agent_id]
    
    async def _execute_subagent_tasks(self, tasks: List[SubAgentTask]) -> List[SubAgentResult]:
        """Execute subagent tasks with dependency and parallelism management."""
        results = []
        completed_tasks = set()
        
        # Group tasks by execution order considering dependencies
        task_groups = self._group_tasks_by_dependencies(tasks)
        
        for group in task_groups:
            # Within each group, respect execution modes
            parallel_tasks = []
            sequential_tasks = []
            
            for task in group:
                if task.execution_mode == ExecutionMode.PARALLEL:
                    parallel_tasks.append(task)
                else:
                    sequential_tasks.append(task)
            
            # Execute parallel tasks
            if parallel_tasks:
                parallel_results = await self._execute_parallel_tasks(parallel_tasks)
                results.extend(parallel_results)
                for result in parallel_results:
                    completed_tasks.add(result.task_id)
            
            # Execute sequential tasks
            for task in sequential_tasks:
                result = await self._execute_single_task(task)
                results.append(result)
                completed_tasks.add(result.task_id)
        
        return results
    
    def _group_tasks_by_dependencies(self, tasks: List[SubAgentTask]) -> List[List[SubAgentTask]]:
        """Group tasks by dependency order."""
        # Simple topological sort
        task_map = {f"task_{i}": task for i, task in enumerate(tasks)}
        for task_id, task in task_map.items():
            task.task_id = task_id
        
        groups = []
        completed = set()
        
        while len(completed) < len(tasks):
            current_group = []
            for task_id, task in task_map.items():
                if task_id not in completed:
                    # Check if all dependencies are completed
                    deps = task.dependencies or []
                    if all(dep in completed for dep in deps):
                        current_group.append(task)
            
            if not current_group:
                # Circular dependency or invalid dependency
                raise ValueError("Invalid task dependencies detected")
            
            groups.append(current_group)
            for task in current_group:
                completed.add(task.task_id)
        
        return groups
    
    async def _execute_parallel_tasks(self, tasks: List[SubAgentTask]) -> List[SubAgentResult]:
        """Execute multiple tasks in parallel with concurrency limit."""
        semaphore = asyncio.Semaphore(self.max_parallel_tasks)
        
        async def execute_with_limit(task):
            async with semaphore:
                return await self._execute_single_task(task)
        
        results = await asyncio.gather(
            *[execute_with_limit(task) for task in tasks],
            return_exceptions=True
        )
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(SubAgentResult(
                    agent_id=tasks[i].agent_id,
                    task_id=tasks[i].task_id,
                    success=False,
                    result=None,
                    error=str(result)
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _execute_single_task(self, task: SubAgentTask) -> SubAgentResult:
        """Execute a single subagent task."""
        if task.agent_id not in self.subagents:
            return SubAgentResult(
                agent_id=task.agent_id,
                task_id=task.task_id,
                success=False,
                result=None,
                error=f"Subagent '{task.agent_id}' not found"
            )
        
        agent = self.subagents[task.agent_id]
        
        try:
            # Prepare messages for subagent
            messages = [{"role": "user", "content": task.task_description}]
            
            # Execute task
            response = await agent.process(messages, context=task.context)
            
            return SubAgentResult(
                agent_id=task.agent_id,
                task_id=task.task_id,
                success=True,
                result=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "total_tokens": response.input_tokens + response.output_tokens,
                    "cost_in_cents": response.cost_in_cents,
                    "tool_outputs": response.tool_outputs
                }
            )
        except Exception as e:
            return SubAgentResult(
                agent_id=task.agent_id,
                task_id=task.task_id,
                success=False,
                result=None,
                error=str(e)
            )
    
    async def process(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> LLMResponse:
        """Process messages through the main agent with subagent delegation capability."""
        return await self.main_agent.process(messages, **kwargs)
    
    def clear_all_memory(self):
        """Clear memory for all agents."""
        self.main_agent.clear_memory()
        for agent in self.subagents.values():
            agent.clear_memory()
    
    def _get_tool_name(self, tool: Callable) -> str:
        """Extract tool name from function."""
        return tool.__name__
    
    def _get_tool_description(self, tool: Callable) -> str:
        """Extract tool description from function docstring."""
        return inspect.getdoc(tool) or "No description available"
    
    def _add_dynamic_planning_tool(self):
        """Add tool for dynamically planning and creating subagents."""
        from pydantic import BaseModel, Field
        from typing import List as ListType
        
        class DynamicPlanningRequest(BaseModel):
            """Request to dynamically create subagents and plan tasks."""
            user_request: str = Field(
                description="The user's original request that needs to be broken down"
            )
            analysis: str = Field(
                description="Your analysis of what needs to be done and why certain subagents are needed"
            )
        
        # Define the response format for planning
        class SubAgentPlanResponse(BaseModel):
            agent_id: str
            system_prompt: str
            task_description: str
            tools: List[str]
            execution_mode: str = "sequential"
            dependencies: List[str] = []
        
        class PlanningResponse(BaseModel):
            subagent_plans: List[SubAgentPlanResponse]
            reasoning: str
        
        async def plan_and_create_subagents(input: DynamicPlanningRequest) -> Dict[str, Any]:
            """Dynamically create subagents based on the user's request."""
            logger.info(f"Planning request: {pformat(input)}")
            # Use the planning LLM to analyze the request and create subagent plans
            planning_messages = [
                {
                    "role": "system",
                    "content": f"""You are a planning agent that designs specialized subagents for complex tasks.
                    
Available tools that can be assigned to subagents:
{self._format_available_tools()}

Your task:
1. Analyze the user's request
2. Design specialized subagents with specific roles
3. Assign appropriate tools to each subagent
4. Create tasks for each subagent
5. Determine execution order (parallel or sequential)

Return a JSON object with this structure:
{{
    "subagent_plans": [
        {{
            "agent_id": "unique_id",
            "system_prompt": "Detailed prompt explaining the agent's role and expertise",
            "task_description": "Specific task for this agent",
            "tools": ["tool_name1", "tool_name2"],
            "execution_mode": "parallel" or "sequential",
            "dependencies": []  // List of agent_ids this depends on
        }}
    ],
    "reasoning": "Explanation of why these subagents and this structure"
}}"""
                },
                {
                    "role": "user",
                    "content": f"User request: {input.user_request}\n\nAnalysis: {input.analysis}"
                }
            ]
            
            logger.info(f"Calling planning LLM with provider={self.planning_provider}, model={self.planning_model}")
            try:
                planning_response = await chat_async(
                    provider=self.planning_provider,
                    model=self.planning_model,
                    messages=planning_messages,
                    temperature=0.2,
                    response_format=PlanningResponse
                )
                logger.info(f"Planning response: {pformat(planning_response)}")
                
                # The response is already a structured Pydantic object
                plan_data = planning_response.content
            except Exception as e:
                logger.error(f"Error in planning LLM call: {e}", exc_info=True)
                return {"error": f"Planning LLM call failed: {str(e)}"}
            
            # Create subagents based on the plan
            created_agents = []
            tasks = []
            
            for plan in plan_data.subagent_plans:
                agent_id = plan.agent_id
                
                # Get tools for this subagent
                agent_tools = []
                for tool_name in plan.tools:
                    if tool_name in self.tool_registry:
                        agent_tools.append(self.tool_registry[tool_name])
                
                # Create the subagent
                subagent = Agent(
                    agent_id=agent_id,
                    provider=self.subagent_provider,
                    model=self.subagent_model,
                    system_prompt=plan.system_prompt,
                    tools=agent_tools
                )
                
                # Register the subagent
                self.register_subagent(subagent)
                created_agents.append(agent_id)
                
                # Create task for this subagent
                task = SubAgentTask(
                    agent_id=agent_id,
                    task_description=plan.task_description,
                    execution_mode=ExecutionMode(plan.execution_mode),
                    dependencies=plan.dependencies
                )
                tasks.append(task)
            
            # Execute the tasks
            results = await self._execute_subagent_tasks(tasks)
            
            # Format results
            formatted_results = {
                "created_agents": created_agents,
                "reasoning": plan_data.reasoning,
                "task_results": {}
            }
            
            for result in results:
                formatted_results["task_results"][result.task_id] = {
                    "agent_id": result.agent_id,
                    "success": result.success,
                    "result": result.result,
                    "error": result.error,
                    "metadata": result.metadata
                }
            
            return formatted_results
        
        # Add to main agent's tools
        self.main_agent.tools.append(plan_and_create_subagents)
    
    def _format_available_tools(self) -> str:
        """Format available tools for the planning prompt."""
        tool_descriptions = []
        for name, tool in self.tool_registry.items():
            desc = self._get_tool_description(tool)
            tool_descriptions.append(f"- {name}: {desc}")
        return "\n".join(tool_descriptions)