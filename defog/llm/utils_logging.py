"""Beautiful logging utilities for the orchestrator system."""

import logging
import json
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.tree import Tree
from rich.text import Text
from rich import box
from rich.padding import Padding
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)

# Create a console instance for direct rich output
console = Console()


class OrchestrationLogger:
    """Beautiful logger for orchestration events."""

    def __init__(
        self, name: str = "defog.orchestrator", console: Optional[Console] = None
    ):
        self.logger = logging.getLogger(name)
        self.console = console or Console()
        self._progress = None
        self._current_tasks = {}

    def setup_rich_logging(self, level: int = logging.INFO):
        """Set up rich logging handler."""
        # Remove existing handlers
        self.logger.handlers = []

        # Create rich handler with custom formatting
        rich_handler = RichHandler(
            console=self.console,
            show_time=True,
            show_path=False,
            enable_link_path=False,
            markup=True,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
        )

        # Custom format without the logger name
        FORMAT = "%(message)s"
        rich_handler.setFormatter(logging.Formatter(FORMAT))

        self.logger.addHandler(rich_handler)
        self.logger.setLevel(level)

    def log_request_start(self, request: str):
        """Log the start of a request with a beautiful header."""
        self.console.print()
        self.console.rule("[bold cyan]🚀 New Request[/bold cyan]", style="cyan")
        self.console.print(
            Panel(
                f"[bold white]{request}[/bold white]",
                title="[bold yellow]User Request[/bold yellow]",
                border_style="yellow",
                padding=(1, 2),
            )
        )

    def log_planning_analysis(self, analysis: str):
        """Log the planning analysis in a structured way."""
        self.console.print()
        self.console.print("[bold magenta]📋 Planning Analysis:[/bold magenta]")

        # Split analysis into paragraphs and format
        paragraphs = analysis.strip().split("\n\n")
        for para in paragraphs:
            if para.strip():
                # Check if it's a numbered list
                if any(para.strip().startswith(f"{i}.") for i in range(1, 10)):
                    # Format as a list
                    items = para.strip().split("\n")
                    tree = Tree("[bold]Task Breakdown[/bold]")
                    for item in items:
                        if item.strip():
                            tree.add(f"[cyan]{item.strip()}[/cyan]")
                    self.console.print(Padding(tree, (1, 2)))
                else:
                    # Regular paragraph
                    self.console.print(
                        Padding(
                            Text(para.strip(), style="white", justify="left"),
                            (0, 2, 1, 2),
                        )
                    )

    def log_llm_call(
        self, provider: str, model: str, purpose: str = "Subagent Designer"
    ):
        """Log LLM API calls."""
        self.console.print()
        self.console.print(
            f"[dim]🤖 {purpose} LLM: [bold]{provider}[/bold] / [bold]{model}[/bold][/dim]"
        )

    def log_subagent_plans(self, plans: List[Dict[str, Any]], reasoning: str):
        """Log subagent plans in a beautiful table format."""
        self.console.print()
        self.console.rule("[bold green]📊 Execution Plan[/bold green]", style="green")

        # Display reasoning first
        if reasoning:
            self.console.print(
                Panel(
                    reasoning,
                    title="[bold]Strategy[/bold]",
                    border_style="green",
                    padding=(1, 2),
                )
            )

        # Create a table for subagent plans
        table = Table(
            title="[bold]Subagent Configuration[/bold]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
            padding=(0, 1),
        )

        table.add_column("Agent ID", style="yellow", no_wrap=True)
        table.add_column("Role", style="white", max_width=40)
        table.add_column("Tools", style="green")
        table.add_column("Mode", style="magenta")
        table.add_column("Dependencies", style="blue")

        for plan in plans:
            # Truncate system prompt for display
            role = (
                plan["system_prompt"][:80] + "..."
                if len(plan["system_prompt"]) > 80
                else plan["system_prompt"]
            )
            role = role.split(".")[0] + "."  # Just first sentence

            tools = "\n".join(plan["tools"]) if plan["tools"] else "None"
            deps = ", ".join(plan["dependencies"]) if plan["dependencies"] else "None"

            table.add_row(plan["agent_id"], role, tools, plan["execution_mode"], deps)

        self.console.print(table)

        # Show task details
        self.console.print("\n[bold]Task Assignments:[/bold]")
        for i, plan in enumerate(plans, 1):
            self.console.print(
                Panel(
                    plan["task_description"],
                    title=f"[bold yellow]{plan['agent_id']}[/bold yellow]",
                    border_style="yellow",
                    padding=(0, 2),
                )
            )

    def log_task_execution_start(self, agent_id: str, task_id: str, mode: str):
        """Log the start of task execution."""
        emoji = "⚡" if mode == "parallel" else "➤"
        self.console.print(
            f"\n{emoji} [bold cyan]Starting:[/bold cyan] {agent_id} "
            f"[dim]({mode} mode)[/dim]"
        )

    def log_retry_attempt(
        self, task_id: str, attempt: int, max_attempts: int, wait_time: float = None
    ):
        """Log retry attempts."""
        self.console.print(
            f"   [yellow]🔄 Retry {attempt}/{max_attempts}[/yellow] for {task_id}"
            + (f" [dim](waiting {wait_time:.1f}s)[/dim]" if wait_time else "")
        )

    def log_error_detail(self, error_type: str, error_msg: str):
        """Log error details in a compact format."""
        color_map = {
            "NETWORK_ERROR": "red",
            "RATE_LIMIT": "yellow",
            "AUTH_ERROR": "red",
            "MODEL_ERROR": "orange",
            "VALIDATION_ERROR": "magenta",
            "SERVER_ERROR": "red",
            "TIMEOUT": "yellow",
            "UNKNOWN_ERROR": "dim red",
        }

        color = color_map.get(error_type, "red")
        self.console.print(
            f"   [dim]❌ {error_type}:[/dim] [{color}]{error_msg[:100]}{'...' if len(error_msg) > 100 else ''}[/{color}]"
        )

    def log_task_execution_complete(
        self,
        agent_id: str,
        task_id: str,
        success: bool,
        result: Optional[Any] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log task completion with result summary."""
        status_emoji = "✅" if success else "❌"
        status_color = "green" if success else "red"

        self.console.print(
            f"{status_emoji} [bold {status_color}]Completed:[/bold {status_color}] {agent_id}"
        )

        if error:
            self.console.print(
                Panel(
                    f"[red]{error}[/red]",
                    title="[bold red]Error[/bold red]",
                    border_style="red",
                    padding=(0, 2),
                )
            )
        elif result and isinstance(result, str):
            # Show a preview of the result
            preview = result[:200] + "..." if len(result) > 200 else result
            self.console.print(Padding(Text(preview, style="dim white"), (0, 4)))

        if metadata:
            tokens = metadata.get("total_tokens", 0)
            cost = metadata.get("cost_in_cents", 0) or 0

            # Check for detailed cost breakdown
            if "cost_breakdown" in metadata:
                breakdown = metadata["cost_breakdown"]
                self.console.print(
                    f"    [dim]Tokens: {tokens} | "
                    f"LLM: ${breakdown['llm_cost'] / 100:.4f} | "
                    f"Tools: ${breakdown['tool_cost'] / 100:.4f} | "
                    f"Total: ${breakdown['total_cost'] / 100:.4f}[/dim]"
                )
            elif tokens or cost:
                self.console.print(
                    f"    [dim]Tokens: {tokens} | Cost: ${cost / 100:.4f}[/dim]"
                )

    def log_orchestration_complete(self, results: Dict[str, Any]):
        """Log the completion of orchestration."""
        self.console.print()
        self.console.rule(
            "[bold green]✨ Orchestration Complete[/bold green]", style="green"
        )

        # Summary statistics
        if "task_results" in results:
            task_results = results["task_results"]
            total_tasks = len(task_results)
            successful_tasks = sum(1 for r in task_results.values() if r["success"])

            stats_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
            stats_table.add_column("Metric", style="bold")
            stats_table.add_column("Value", style="cyan")

            stats_table.add_row("Total Tasks", str(total_tasks))
            stats_table.add_row("Successful", str(successful_tasks))
            stats_table.add_row("Failed", str(total_tasks - successful_tasks))

            # Calculate total cost and tokens
            total_cost = 0
            total_tokens = 0
            total_llm_cost = 0
            total_tool_cost = 0

            for result in task_results.values():
                if result.get("metadata"):
                    metadata = result["metadata"]
                    total_tokens += metadata.get("total_tokens", 0)

                    # Use detailed cost breakdown if available
                    if "total_cost_in_cents" in metadata:
                        total_cost += metadata.get("total_cost_in_cents", 0) or 0
                        total_llm_cost += metadata.get("cost_in_cents", 0) or 0
                        total_tool_cost += metadata.get("tool_costs_in_cents", 0) or 0
                    else:
                        # Fallback to simple cost
                        cost = metadata.get("cost_in_cents", 0) or 0
                        total_cost += cost
                        total_llm_cost += cost

            stats_table.add_row("Total Tokens", f"{total_tokens:,}")

            # Show cost breakdown if we have tool costs
            if total_tool_cost > 0:
                stats_table.add_row("LLM Cost", f"${total_llm_cost / 100:.4f}")
                stats_table.add_row("Tool Cost", f"${total_tool_cost / 100:.4f}")
                stats_table.add_row(
                    "[bold]Total Cost[/bold]", f"[bold]${total_cost / 100:.4f}[/bold]"
                )
            else:
                stats_table.add_row("Total Cost", f"${total_cost / 100:.4f}")

            # Add overall cost breakdown if available in results
            if "total_cost_breakdown" in results:
                breakdown = results["total_cost_breakdown"]
                stats_table.add_row("", "")  # Empty row for spacing
                stats_table.add_row(
                    "[dim]Planning Cost[/dim]",
                    f"[dim]${breakdown.get('planning_cost_in_cents', 0) / 100:.4f}[/dim]",
                )
                stats_table.add_row(
                    "[dim]Subagent Cost[/dim]",
                    f"[dim]${breakdown.get('subagent_costs_in_cents', 0) / 100:.4f}[/dim]",
                )
                stats_table.add_row(
                    "[dim]Tool Cost[/dim]",
                    f"[dim]${breakdown.get('tool_costs_in_cents', 0) / 100:.4f}[/dim]",
                )
                stats_table.add_row(
                    "[bold]Grand Total[/bold]",
                    f"[bold green]${breakdown.get('total_cost_in_cents', 0) / 100:.4f}[/bold green]",
                )

            self.console.print(
                Panel(
                    stats_table,
                    title="[bold]Summary[/bold]",
                    border_style="green",
                )
            )

    def log_raw_response(self, response: Any, title: str = "Response"):
        """Log raw response data in a formatted way."""
        if isinstance(response, (dict, list)):
            json_str = json.dumps(response, indent=2)
            syntax = Syntax(json_str, "json", theme="monokai", line_numbers=False)
            self.console.print(
                Panel(
                    syntax,
                    title=f"[bold]{title}[/bold]",
                    border_style="blue",
                )
            )
        else:
            self.console.print(
                Panel(
                    str(response),
                    title=f"[bold]{title}[/bold]",
                    border_style="blue",
                )
            )

    def log_delegation_tree(
        self, main_agent_id: str, delegations: List[Dict[str, Any]]
    ):
        """Display delegation hierarchy as a tree."""
        tree = Tree(f"[bold yellow]{main_agent_id}[/bold yellow] (Main Agent)")

        # Group by execution mode
        parallel_tasks = []
        sequential_tasks = []

        for task in delegations:
            if task.get("execution_mode") == "parallel":
                parallel_tasks.append(task)
            else:
                sequential_tasks.append(task)

        if parallel_tasks:
            parallel_branch = tree.add("[bold cyan]⚡ Parallel Tasks[/bold cyan]")
            for task in parallel_tasks:
                parallel_branch.add(f"[green]{task['agent_id']}[/green]")

        if sequential_tasks:
            seq_branch = tree.add("[bold magenta]➤ Sequential Tasks[/bold magenta]")
            for task in sequential_tasks:
                seq_branch.add(f"[yellow]{task['agent_id']}[/yellow]")

        self.console.print(tree)


# Create a global orchestration logger instance
orch_logger = OrchestrationLogger()
orch_logger.setup_rich_logging()


class ToolProgressTracker:
    """Context manager for tracking progress of LLM tool operations."""

    def __init__(
        self, tool_name: str, description: str, console: Optional[Console] = None
    ):
        self.tool_name = tool_name
        self.description = description
        self.console = console or Console()
        self.progress = None
        self.task_id = None
        self.start_time = None

    async def __aenter__(self):
        """Start progress tracking."""
        from datetime import datetime

        self.start_time = datetime.now()

        # Log tool start
        self.console.print()
        self.console.rule(f"[bold cyan]🔧 {self.tool_name}[/bold cyan]", style="cyan")
        self.console.print(
            Panel(
                f"[white]{self.description}[/white]",
                border_style="cyan",
                padding=(1, 2),
                box=box.ROUNDED,
            )
        )

        # Create progress bar
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console,
        )
        self.progress.start()
        self.task_id = self.progress.add_task("[cyan]Processing...", total=100)

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Stop progress tracking."""
        from datetime import datetime

        if self.progress:
            if exc_type is None:
                # Success - complete the progress
                self.progress.update(
                    self.task_id, completed=100, description="[green]✓ Complete"
                )
                self.progress.stop()

                # Calculate duration
                duration = (datetime.now() - self.start_time).total_seconds()
                self.console.print(f"[dim]Completed in {duration:.1f}s[/dim]")
            else:
                # Error - mark as failed
                self.progress.update(self.task_id, description="[red]✗ Failed")
                self.progress.stop()

                # Log error
                self.console.print(
                    Panel(
                        f"[red]{exc_val}[/red]",
                        title="[bold red]Error[/bold red]",
                        border_style="red",
                        padding=(1, 2),
                    )
                )

    def update(self, progress: float, description: Optional[str] = None):
        """Update progress percentage and optionally the description."""
        if self.progress and self.task_id is not None:
            update_kwargs = {"completed": progress}
            if description:
                update_kwargs["description"] = f"[cyan]{description}"
            self.progress.update(self.task_id, **update_kwargs)


class SubTaskLogger:
    """Logger for subtasks within a tool operation."""

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()

    def log_subtask(self, message: str, status: str = "info"):
        """Log a subtask with appropriate styling."""
        status_styles = {
            "info": ("ℹ️", "blue"),
            "success": ("✓", "green"),
            "warning": ("⚠️", "yellow"),
            "error": ("✗", "red"),
            "processing": ("⚡", "cyan"),
        }

        emoji, color = status_styles.get(status, ("•", "white"))
        self.console.print(f"  {emoji} [{color}]{message}[/{color}]")

    def log_provider_info(self, provider: str, model: str):
        """Log provider and model information."""
        self.console.print(
            f"  [dim]Provider: [bold]{provider}[/bold] | Model: [bold]{model}[/bold][/dim]"
        )

    def log_document_upload(self, doc_name: str, doc_index: int, total_docs: int):
        """Log document upload progress."""
        self.console.print(
            f"  📄 Uploading document [{doc_index}/{total_docs}]: [yellow]{doc_name}[/yellow]"
        )

    def log_vector_store_status(self, completed: int, total: int):
        """Log vector store indexing status."""
        percentage = (completed / total * 100) if total > 0 else 0
        self.console.print(
            f"  🔍 Vector store indexing: [cyan]{completed}/{total}[/cyan] files "
            f"([bold]{percentage:.0f}%[/bold])"
        )

    def log_search_status(self, query: str, max_results: Optional[int] = None):
        """Log web search status."""
        msg = f"  🌐 Searching web for: [yellow]{query[:50]}{'...' if len(query) > 50 else ''}[/yellow]"
        if max_results:
            msg += f" [dim](max {max_results} results)[/dim]"
        self.console.print(msg)

    def log_code_execution(self, language: str = "python"):
        """Log code execution start."""
        self.console.print(
            f"  💻 Executing {language} code in sandboxed environment..."
        )

    def log_result_summary(
        self, result_type: str, details: Optional[Dict[str, Any]] = None
    ):
        """Log a summary of results."""
        if details:
            # Create a simple table for results
            table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
            table.add_column("Key", style="bold cyan")
            table.add_column("Value", style="white")

            for key, value in details.items():
                if isinstance(value, (list, dict)):
                    value_str = (
                        f"{len(value)} items"
                        if isinstance(value, list)
                        else f"{len(value)} fields"
                    )
                else:
                    value_str = str(value)
                table.add_row(key.replace("_", " ").title(), value_str)

            self.console.print(
                Panel(
                    table,
                    title=f"[bold]{result_type} Summary[/bold]",
                    border_style="green",
                    padding=(1, 2),
                )
            )


class NoOpToolProgressTracker:
    """No-op version of ToolProgressTracker for when verbose=False."""

    def __init__(self, tool_name: str = "", description: str = ""):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    def update(self, progress: float, description: str = None):
        pass


class NoOpSubTaskLogger:
    """No-op version of SubTaskLogger for when verbose=False."""

    def __init__(self):
        pass

    def log_subtask(self, message: str, status: str = "info"):
        pass

    def log_provider_info(self, provider: str, model: str):
        pass

    def log_document_upload(self, doc_name: str, doc_index: int, total_docs: int):
        pass

    def log_vector_store_status(self, completed: int, total: int):
        pass

    def log_search_status(self, query: str, max_results: int = None):
        pass

    def log_code_execution(self, language: str = "python"):
        pass

    def log_result_summary(self, result_type: str, details: Dict[str, Any] = None):
        pass
