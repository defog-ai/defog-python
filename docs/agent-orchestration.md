# Agent Orchestration

This document covers the advanced agent orchestration capabilities in defog-python, including multi-agent coordination, thinking agents, and shared context management.

## Enhanced Agent Orchestrator

The `EnhancedAgentOrchestrator` provides advanced multi-agent coordination:

```python
from defog.llm.enhanced_orchestrator import (
    EnhancedAgentOrchestrator,
    EnhancedOrchestratorConfig,
    ExecutionMode
)

# Create orchestrator with full configuration
config = EnhancedOrchestratorConfig(
    # Core settings
    model="claude-3-5-sonnet",
    provider="anthropic",
    temperature=0.7,
    max_tokens=4096,
    
    # Agent features
    enable_thinking_agents=True,
    enable_shared_context=True,
    enable_alternative_paths=True,
    
    # Memory management
    enable_cross_agent_memory=True,
    memory_config={
        "token_threshold": 50000,
        "preserve_last_n_messages": 10
    },
    
    # Execution settings
    max_parallel_agents=5,
    agent_timeout_seconds=300,
    retry_failed_agents=True,
    
    # Cost management
    max_cost_per_agent_cents=100,
    total_cost_limit_cents=1000
)

orchestrator = EnhancedAgentOrchestrator(config=config)
```

## Thinking Agents

Thinking agents provide enhanced reasoning capabilities:

```python
from defog.llm.thinking_agent import ThinkingAgent, ThinkingAgentConfig

# Create a thinking agent
thinking_agent = ThinkingAgent(
    agent_id="researcher",
    config=ThinkingAgentConfig(
        model="claude-3-5-sonnet",
        provider="anthropic",
        
        # Thinking configuration
        enable_extended_thinking=True,
        max_thinking_tokens=8000,
        thinking_style="analytical",  # analytical, creative, systematic
        
        # Reasoning features
        enable_self_reflection=True,
        enable_alternative_exploration=True,
        confidence_threshold=0.8,
        
        # Collaboration
        enable_cross_agent_queries=True,
        shared_context_access=True
    )
)

# Execute task with thinking
result = await thinking_agent.execute_task(
    task="Analyze the implications of quantum computing on cryptography",
    
    # Thinking directives
    thinking_directives=[
        "Consider both near-term and long-term impacts",
        "Evaluate current mitigation strategies",
        "Identify knowledge gaps"
    ],
    
    # Collaboration context
    available_agents=["security_expert", "quantum_physicist"],
    shared_artifacts=["research_papers", "threat_models"]
)

# Access thinking process
print(f"Thinking process: {result['thinking_process']}")
print(f"Confidence: {result['confidence']}")
print(f"Alternative approaches: {result['alternatives']}")
```

## Shared Context Store

Enable cross-agent information sharing:

```python
from defog.llm.shared_context import SharedContextStore

# Initialize shared context
context_store = SharedContextStore(
    workspace_path="/tmp/agent_workspace",
    
    # Versioning
    enable_versioning=True,
    max_versions_per_artifact=5,
    
    # Cleanup
    auto_cleanup=True,
    ttl_hours=24,
    
    # Access control
    agent_permissions={
        "researcher": ["read", "write"],
        "reviewer": ["read"],
        "supervisor": ["read", "write", "delete"]
    }
)

# Store artifact with metadata
await context_store.store_artifact(
    artifact_id="market_analysis",
    content={"data": analysis_data},
    metadata={
        "agent_id": "analyst",
        "timestamp": "2024-01-15T10:00:00Z",
        "confidence": 0.85,
        "tags": ["finance", "quarterly"]
    }
)

# Retrieve with lineage
artifact = await context_store.get_artifact(
    artifact_id="market_analysis",
    include_lineage=True
)
print(f"Created by: {artifact['lineage']['created_by']}")
print(f"Modified by: {artifact['lineage']['modifications']}")
```

## Complex Multi-Agent Workflows

### Hierarchical Task Delegation

```python
from defog.llm.enhanced_orchestrator import SubAgentTask, TaskDependency

# Define complex workflow
workflow_tasks = [
    # Phase 1: Research (parallel)
    SubAgentTask(
        agent_id="market_researcher",
        task_description="Research current market trends",
        execution_mode=ExecutionMode.PARALLEL,
        priority="high",
        estimated_duration_seconds=120
    ),
    SubAgentTask(
        agent_id="competitor_analyst",
        task_description="Analyze competitor strategies",
        execution_mode=ExecutionMode.PARALLEL,
        priority="high"
    ),
    
    # Phase 2: Synthesis (depends on Phase 1)
    SubAgentTask(
        agent_id="strategy_synthesizer",
        task_description="Synthesize research into strategic recommendations",
        dependencies=["market_researcher", "competitor_analyst"],
        execution_mode=ExecutionMode.SEQUENTIAL,
        
        # Advanced options
        require_all_dependencies=True,
        confidence_threshold=0.8,
        alternative_paths_allowed=True
    ),
    
    # Phase 3: Review and refinement
    SubAgentTask(
        agent_id="senior_reviewer",
        task_description="Review and refine strategy",
        dependencies=["strategy_synthesizer"],
        
        # Conditional execution
        condition=lambda results: results["strategy_synthesizer"]["confidence"] < 0.9,
        fallback_agent="expert_consultant"
    )
]

# Execute workflow
results = await orchestrator.execute_workflow(
    tasks=workflow_tasks,
    
    # Workflow options
    continue_on_failure=True,
    collect_thinking_traces=True,
    generate_summary=True
)

# Access comprehensive results
print(f"Workflow summary: {results['summary']}")
print(f"Total duration: {results['duration_seconds']}s")
print(f"Total cost: ${results['total_cost_cents'] / 100:.2f}")
```

### Dynamic Agent Creation

```python
# Define agent templates
agent_templates = {
    "researcher": {
        "model": "claude-3-5-sonnet",
        "temperature": 0.3,
        "system_prompt": "You are a thorough researcher..."
    },
    "analyst": {
        "model": "gpt-4o",
        "temperature": 0.5,
        "tools": ["web_search", "code_interpreter"]
    }
}

# Dynamic agent spawning
async def create_specialized_agent(specialty: str):
    if specialty in agent_templates:
        return ThinkingAgent(
            agent_id=f"{specialty}_{uuid.uuid4().hex[:8]}",
            config=agent_templates[specialty]
        )
    else:
        # Create generic agent with specialty prompt
        return ThinkingAgent(
            agent_id=f"generic_{specialty}",
            config={
                "model": "gpt-4o",
                "system_prompt": f"You are a specialist in {specialty}..."
            }
        )

# Spawn agents based on task requirements
task_requirements = analyze_task_requirements(user_query)
agents = []
for requirement in task_requirements:
    agent = await create_specialized_agent(requirement)
    agents.append(agent)
```

## Agent Communication Patterns

### Direct Agent-to-Agent Communication

```python
# Enable direct communication between agents
class CollaborativeAgent(ThinkingAgent):
    async def consult_peer(self, peer_id: str, question: str):
        """Consult another agent directly"""
        response = await self.orchestrator.route_message(
            from_agent=self.agent_id,
            to_agent=peer_id,
            message={
                "type": "consultation",
                "question": question,
                "context": self.current_context
            }
        )
        return response

# Usage in agent task
async def analyze_with_consultation(self, data):
    initial_analysis = await self.analyze(data)
    
    if initial_analysis["confidence"] < 0.7:
        # Consult specialist
        specialist_opinion = await self.consult_peer(
            "domain_specialist",
            f"Please review this analysis: {initial_analysis}"
        )
        
        # Incorporate feedback
        final_analysis = await self.refine_analysis(
            initial_analysis,
            specialist_opinion
        )
    
    return final_analysis
```

### Broadcasting and Subscriptions

```python
# Broadcast findings to interested agents
await orchestrator.broadcast(
    from_agent="researcher",
    event_type="discovery",
    data={
        "finding": "New market opportunity identified",
        "confidence": 0.9,
        "impact": "high"
    },
    
    # Subscription filters
    subscriber_filter=lambda agent: "analyst" in agent.role
)

# Subscribe to events
orchestrator.subscribe(
    agent_id="portfolio_manager",
    event_types=["discovery", "risk_alert"],
    callback=handle_market_event
)
```

## Advanced Orchestration Patterns

### Map-Reduce Pattern

```python
async def map_reduce_analysis(orchestrator, documents):
    # Map phase: Analyze documents in parallel
    map_tasks = []
    for i, doc in enumerate(documents):
        task = SubAgentTask(
            agent_id=f"analyzer_{i}",
            task_description=f"Analyze document: {doc['title']}",
            execution_mode=ExecutionMode.PARALLEL,
            input_data=doc
        )
        map_tasks.append(task)
    
    map_results = await orchestrator.execute_tasks(map_tasks)
    
    # Reduce phase: Combine analyses
    reduce_task = SubAgentTask(
        agent_id="synthesizer",
        task_description="Combine all document analyses",
        input_data=map_results,
        execution_mode=ExecutionMode.SEQUENTIAL
    )
    
    final_result = await orchestrator.execute_task(reduce_task)
    return final_result
```

### Pipeline Pattern

```python
# Create processing pipeline
pipeline = orchestrator.create_pipeline([
    ("data_cleaner", "Clean and validate input data"),
    ("feature_extractor", "Extract relevant features"),
    ("model_builder", "Build predictive model"),
    ("validator", "Validate model performance"),
    ("reporter", "Generate final report")
])

# Execute pipeline with data flow
result = await pipeline.execute(
    input_data=raw_data,
    
    # Pipeline options
    stop_on_error=False,
    collect_intermediates=True,
    parallel_stages=["feature_extractor", "model_builder"]
)

# Access intermediate results
for stage, output in result["intermediates"].items():
    print(f"{stage}: {output['summary']}")
```

### Consensus Pattern

```python
async def consensus_decision(orchestrator, question):
    # Get opinions from multiple agents
    agents = ["expert_1", "expert_2", "expert_3", "expert_4", "expert_5"]
    
    opinion_tasks = [
        SubAgentTask(
            agent_id=agent,
            task_description=f"Provide opinion on: {question}",
            execution_mode=ExecutionMode.PARALLEL
        )
        for agent in agents
    ]
    
    opinions = await orchestrator.execute_tasks(opinion_tasks)
    
    # Aggregate opinions
    consensus_task = SubAgentTask(
        agent_id="consensus_builder",
        task_description="Build consensus from expert opinions",
        input_data=opinions,
        
        # Consensus parameters
        consensus_config={
            "method": "weighted_voting",  # majority, weighted, deliberative
            "confidence_weights": True,
            "require_quorum": 0.8
        }
    )
    
    consensus = await orchestrator.execute_task(consensus_task)
    return consensus
```

## Monitoring and Debugging

### Agent Performance Monitoring

```python
# Enable comprehensive monitoring
orchestrator.enable_monitoring(
    metrics=[
        "execution_time",
        "token_usage",
        "cost",
        "error_rate",
        "confidence_scores"
    ],
    
    # Real-time dashboard
    dashboard_port=8080,
    
    # Alerts
    alert_thresholds={
        "error_rate": 0.1,
        "cost_per_task": 1.0,
        "execution_time": 300
    }
)

# Get performance metrics
metrics = orchestrator.get_metrics()
print(f"Average task duration: {metrics['avg_duration']}s")
print(f"Success rate: {metrics['success_rate']:.2%}")
print(f"Total cost: ${metrics['total_cost']:.2f}")
```

### Debugging Tools

```python
# Enable debug mode
orchestrator.set_debug_mode(
    enabled=True,
    
    # Debug options
    log_level="DEBUG",
    save_thinking_traces=True,
    save_agent_conversations=True,
    breakpoint_on_error=True
)

# Inspect agent state
agent_state = await orchestrator.inspect_agent("researcher")
print(f"Current task: {agent_state['current_task']}")
print(f"Memory usage: {agent_state['memory_tokens']}")
print(f"Thinking trace: {agent_state['thinking_trace']}")

# Replay workflow for debugging
replay_result = await orchestrator.replay_workflow(
    workflow_id="wf_123",
    modifications={
        "agent_model": "gpt-4o",  # Try different model
        "temperature": 0.3         # Adjust temperature
    }
)
```

## Best Practices

1. **Design clear agent roles** with specific responsibilities
2. **Use thinking agents** for complex reasoning tasks
3. **Implement proper error handling** and fallback strategies
4. **Monitor costs** and set appropriate limits
5. **Use shared context** judiciously to avoid information overload
6. **Test workflows** with smaller/cheaper models first
7. **Document agent interactions** for debugging
8. **Implement timeouts** to prevent hanging agents
9. **Use consensus patterns** for critical decisions
10. **Clean up resources** after workflow completion