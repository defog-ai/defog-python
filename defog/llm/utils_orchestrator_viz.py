"""Visualization utilities for Agent Orchestrator."""

from typing import Dict, List, Any


def generate_orchestrator_flowchart(tool_outputs: List[Dict[str, Any]]) -> str:
    """
    Generate an ASCII flowchart representation of orchestrator execution.

    Args:
        tool_outputs: List of tool outputs from orchestrator response

    Returns:
        ASCII flowchart string
    """
    if not tool_outputs:
        return "No orchestrator data available"

    # Find the plan_and_create_subagents tool output
    plan_output = None
    for output in tool_outputs:
        if output.get("name") == "plan_and_create_subagents":
            plan_output = output.get("result", {})
            break

    if not plan_output:
        return "No orchestrator planning data found"

    # Extract subagent information
    created_agents = plan_output.get("created_agents", [])
    task_results = plan_output.get("task_results", {})

    # Build the flowchart
    lines = []
    lines.append("┌─────────────────────────┐")
    lines.append("│   Agent Orchestrator    │")
    lines.append("└────────────┬────────────┘")
    lines.append("             │")
    lines.append("             ▼")
    lines.append("┌─────────────────────────┐")
    lines.append("│  plan_and_create_       │")
    lines.append("│      subagents          │")
    lines.append("└────────────┬────────────┘")

    # Show parallel execution if multiple agents
    if len(created_agents) > 1:
        lines.append("             │")
        lines.append("      ┌──────┴──────┐")

        # Create branches for each subagent
        branch_lines = []
        for i, agent_id in enumerate(created_agents):
            if i == 0:
                branch_lines.append("      │             │")
            elif i == len(created_agents) - 1:
                branch_lines.append("      │             │")
            else:
                branch_lines.append("      │      │      │")

        lines.extend(branch_lines)
        lines.append("      ▼      ▼      ▼")
    else:
        lines.append("             │")
        lines.append("             ▼")

    # Add subagent boxes with their tools
    agent_boxes = []
    max_width = 0

    for agent_id in created_agents:
        # Find the task result for this agent
        task_info = None
        for task_id, result in task_results.items():
            if result.get("agent_id") == agent_id:
                task_info = result
                break

        # Extract tool calls from metadata
        tools_used = []
        if task_info and task_info.get("metadata", {}).get("tool_outputs"):
            for tool_output in task_info["metadata"]["tool_outputs"]:
                tool_name = tool_output.get("name", "unknown_tool")
                tools_used.append(tool_name)

        # Create agent box
        agent_lines = []
        agent_name = agent_id.replace("_", " ").title()

        # Calculate box width
        width = max(len(agent_name) + 4, 25)
        for tool in tools_used:
            width = max(width, len(tool) + 6)
        max_width = max(max_width, width)

        # Build agent box
        agent_lines.append("┌" + "─" * (width - 2) + "┐")
        agent_lines.append("│" + agent_name.center(width - 2) + "│")
        agent_lines.append("├" + "─" * (width - 2) + "┤")

        if tools_used:
            agent_lines.append("│ Tools:".ljust(width - 1) + "│")
            for tool in tools_used:
                agent_lines.append("│  • " + tool.ljust(width - 5) + "│")
        else:
            agent_lines.append("│ No tools used".ljust(width - 1) + "│")

        agent_lines.append("└" + "─" * (width - 2) + "┘")

        agent_boxes.append(agent_lines)

    # Arrange agent boxes side by side
    if len(agent_boxes) > 1:
        # Find the maximum number of lines
        max_lines = max(len(box) for box in agent_boxes)

        # Pad shorter boxes
        for box in agent_boxes:
            while len(box) < max_lines:
                box.append(" " * len(box[0]))

        # Combine boxes horizontally
        spacing = "  "
        for line_idx in range(max_lines):
            line_parts = []
            for box in agent_boxes:
                line_parts.append(box[line_idx])
            lines.append(spacing.join(line_parts))
    else:
        # Single agent
        for box in agent_boxes:
            lines.extend(box)

    # Add summary statistics
    lines.append("")
    lines.append("Summary:")
    lines.append("─" * 40)

    total_tools = 0
    total_llm_cost = 0
    total_tool_cost = 0
    total_cost = 0

    for task_id, result in task_results.items():
        metadata = result.get("metadata", {})
        if metadata.get("tool_outputs"):
            total_tools += len(metadata["tool_outputs"])

        # Aggregate costs
        if metadata.get("total_cost_in_cents"):
            total_cost += metadata.get("total_cost_in_cents", 0)
            total_llm_cost += metadata.get("cost_in_cents", 0)
            total_tool_cost += metadata.get("tool_costs_in_cents", 0)
        elif metadata.get("cost_in_cents"):
            cost = metadata.get("cost_in_cents", 0)
            total_cost += cost
            total_llm_cost += cost

    lines.append(f"Total Subagents Created: {len(created_agents)}")
    lines.append(f"Total Tool Calls: {total_tools}")

    # Add execution mode info
    execution_modes = set()
    for output in tool_outputs:
        if output.get("args", {}).get("tasks"):
            for task in output["args"]["tasks"]:
                mode = task.get("execution_mode", "parallel")
                execution_modes.add(mode)

    if execution_modes:
        lines.append(f"Execution Mode(s): {', '.join(execution_modes)}")

    # Add cost breakdown
    lines.append("")
    lines.append("Cost Breakdown:")
    lines.append("─" * 40)

    # Check for overall cost breakdown in plan_output
    if "total_cost_breakdown" in plan_output:
        breakdown = plan_output["total_cost_breakdown"]
        lines.append(
            f"Planning Cost: ${breakdown.get('planning_cost_in_cents', 0) / 100:.4f}"
        )
        lines.append(
            f"Subagent Cost: ${breakdown.get('subagent_costs_in_cents', 0) / 100:.4f}"
        )
        lines.append(f"Tool Cost: ${breakdown.get('tool_costs_in_cents', 0) / 100:.4f}")
        lines.append(
            f"Total Cost: ${breakdown.get('total_cost_in_cents', 0) / 100:.4f}"
        )
    else:
        # Fallback to aggregated costs
        if total_tool_cost > 0:
            lines.append(f"LLM Cost: ${total_llm_cost / 100:.4f}")
            lines.append(f"Tool Cost: ${total_tool_cost / 100:.4f}")
        lines.append(f"Total Cost: ${total_cost / 100:.4f}")

    return "\n".join(lines)


def generate_detailed_tool_trace(tool_outputs: List[Dict[str, Any]]) -> str:
    """
    Generate a detailed trace of all tool calls in hierarchical format.

    Args:
        tool_outputs: List of tool outputs from orchestrator response

    Returns:
        Detailed trace string
    """
    lines = []
    lines.append("Orchestrator Execution Trace")
    lines.append("=" * 50)
    lines.append("")

    for i, output in enumerate(tool_outputs):
        tool_name = output.get("name", "unknown")
        lines.append(f"{i + 1}. {tool_name}")

        if tool_name == "plan_and_create_subagents":
            result = output.get("result", {})
            task_results = result.get("task_results", {})

            for task_id, task_result in task_results.items():
                agent_id = task_result.get("agent_id", "unknown")
                success = "✓" if task_result.get("success") else "✗"
                lines.append(f"   └─ {agent_id} [{success}]")

                # Show tool calls made by this subagent
                tool_outputs_sub = task_result.get("metadata", {}).get(
                    "tool_outputs", []
                )
                for j, tool_output in enumerate(tool_outputs_sub):
                    tool_name_sub = tool_output.get("name", "unknown")
                    tool_result = tool_output.get("result", {})
                    tool_success = (
                        "✓"
                        if isinstance(tool_result, dict)
                        and tool_result.get("success", True)
                        or not isinstance(tool_result, dict)
                        else "✗"
                    )

                    # Show tool arguments preview
                    args = tool_output.get("args", {})
                    args_preview = ""
                    if args:
                        # Get first key-value pair as preview
                        first_key = list(args.keys())[0] if args else ""
                        if first_key:
                            value = str(args[first_key])[:50]
                            if len(str(args[first_key])) > 50:
                                value += "..."
                            args_preview = f" ({first_key}: {value})"

                    # Check if tool has cost
                    tool_cost_str = ""
                    if isinstance(tool_result, dict) and "cost_in_cents" in tool_result:
                        tool_cost = tool_result["cost_in_cents"]
                        tool_cost_str = f" 💵${tool_cost / 100:.4f}"

                    is_last = j == len(tool_outputs_sub) - 1
                    prefix = "       └─" if is_last else "       ├─"
                    lines.append(
                        f"{prefix} {tool_name_sub}{args_preview} [{tool_success}]{tool_cost_str}"
                    )

                # Show token usage and cost breakdown
                metadata = task_result.get("metadata", {})
                if metadata.get("total_tokens") or metadata.get("total_cost_in_cents"):
                    cost_info = []
                    if metadata.get("total_tokens"):
                        cost_info.append(f"Tokens: {metadata['total_tokens']}")

                    # Show detailed cost breakdown if available
                    if metadata.get("total_cost_in_cents"):
                        llm_cost = metadata.get("cost_in_cents", 0)
                        tool_cost = metadata.get("tool_costs_in_cents", 0)
                        total_cost = metadata.get("total_cost_in_cents", 0)

                        if tool_cost > 0:
                            cost_info.append(f"LLM: ${llm_cost / 100:.4f}")
                            cost_info.append(f"Tools: ${tool_cost / 100:.4f}")
                            cost_info.append(f"Total: ${total_cost / 100:.4f}")
                        else:
                            cost_info.append(f"Cost: ${total_cost / 100:.4f}")
                    elif metadata.get("cost_in_cents"):
                        cost_info.append(
                            f"Cost: ${metadata['cost_in_cents'] / 100:.4f}"
                        )

                    if cost_info:
                        lines.append(f"       └─ {' | '.join(cost_info)}")

            # Add overall cost breakdown at the end if available
            if "total_cost_breakdown" in result:
                breakdown = result["total_cost_breakdown"]
                lines.append("")
                lines.append("   💰 Total Cost Breakdown:")
                lines.append(
                    f"      Planning: ${breakdown.get('planning_cost_in_cents', 0) / 100:.4f}"
                )
                lines.append(
                    f"      Subagents: ${breakdown.get('subagent_costs_in_cents', 0) / 100:.4f}"
                )
                lines.append(
                    f"      Tools: ${breakdown.get('tool_costs_in_cents', 0) / 100:.4f}"
                )
                lines.append("      ─────────────────")
                lines.append(
                    f"      TOTAL: ${breakdown.get('total_cost_in_cents', 0) / 100:.4f}"
                )

    return "\n".join(lines)
