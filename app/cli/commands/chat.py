"""Interactive chat CLI for conversing with articles.

Commands:
- chat <article_id>: Start interactive chat about an article
"""

import asyncio
import uuid

import typer
from rich.panel import Panel
from rich.prompt import Prompt
from sqlalchemy import select

from app.cli.output import console
from app.cli.session import get_cli_session
from app.models.orm_models import ChatConversation, NewsArticle

chat_app = typer.Typer(
    name="chat",
    help="Interactive chat with news articles",
)


@chat_app.command()
def interactive(
    article_id: str = typer.Argument(
        ...,
        help="Article ID to chat about (UUID)",
    ),
    user_id: int = typer.Option(
        1,
        "--user",
        "-u",
        help="User ID for conversation",
    ),
) -> None:
    """Start interactive chat session about an article.

    Creates or resumes a conversation for the specified article
    and enters an interactive loop for user input.

    Example:
        refinery chat abc123
        refinery chat abc123 --user 42

    Exit commands: quit, exit, q
    """
    # Validate article ID
    try:
        article_uuid = uuid.UUID(article_id)
    except ValueError:
        console.print("[red]Error: Invalid article ID (must be UUID)[/red]")
        raise typer.Exit(1)

    async def _get_article_context() -> tuple[dict, str]:
        """Fetch article context and create/get conversation."""
        async with get_cli_session() as session:
            # Fetch article
            stmt = select(NewsArticle).where(NewsArticle.id == article_uuid)
            result = await session.execute(stmt)
            article = result.scalar_one_or_none()

            if not article:
                console.print("[red]Error: Article not found[/red]")
                raise typer.Exit(1)

            article_context = {
                "id": str(article.id),
                "title": article.chinese_title or article.original_title,
                "summary": article.chinese_summary,
                "content": article.full_content,
                "deepsearch": article.deepsearch_report,
                "source": article.source_name,
                "url": article.source_url,
            }

            # Check for existing active conversation
            conv_stmt = select(ChatConversation).where(
                ChatConversation.article_id == article_uuid,
                ChatConversation.user_id == user_id,
                ChatConversation.status == "active",
            )
            conv_result = await session.execute(conv_stmt)
            conversation = conv_result.scalar_one_or_none()

            if conversation:
                conversation_id = str(conversation.id)
                console.print(
                    f"[cyan]Resuming conversation: {conversation_id}[/cyan]"
                )
            else:
                # Create new conversation
                conversation = ChatConversation(
                    article_id=article_uuid,
                    user_id=user_id,
                    title=f"Chat about: {article_context['title'][:50]}",
                    status="active",
                )
                session.add(conversation)
                await session.flush()
                await session.refresh(conversation)
                conversation_id = str(conversation.id)
                console.print(
                    f"[cyan]Created new conversation: {conversation_id}[/cyan]"
                )

            return article_context, conversation_id

    async def _run_chat(
        conversation_id: str,
        article_id: str,
        user_message: str,
    ) -> str:
        """Run chat workflow and return response."""
        async with get_cli_session() as session:
            from app.chat.graph import run_chat

            result = await run_chat(
                session=session,
                conversation_id=conversation_id,
                article_id=article_id,
                user_id=user_id,
                user_message=user_message,
            )

            return result.get("final_response", "No response generated")

    try:
        # Get article context and conversation ID
        article_context, conversation_id = asyncio.run(_get_article_context())

        # Display article info
        console.print(
            Panel(
                f"[bold]Title:[/bold] {article_context['title']}\n"
                f"[bold]Source:[/bold] {article_context['source']}\n"
                f"[bold]URL:[/bold] {article_context['url']}\n"
                f"[bold]DeepSearch:[/bold] {'Available' if article_context['deepsearch'] else 'Not available'}",
                title="[bold green]Article Context[/bold green]",
                border_style="green",
            )
        )

        console.print(
            "\n[bold]Starting interactive chat...[/bold]\n"
            "[dim]Type 'quit', 'exit', or 'q' to end the session[/dim]\n"
        )

        # Interactive loop
        while True:
            # Get user input
            user_message = Prompt.ask("[bold blue]You[/bold blue]")

            # Check for exit commands
            if user_message.lower() in ("quit", "exit", "q"):
                console.print("[yellow]Ending chat session...[/yellow]")
                break

            if not user_message.strip():
                console.print("[dim]Please enter a message[/dim]")
                continue

            # Process message
            with console.status("[bold green]Processing..."):
                response = asyncio.run(
                    _run_chat(
                        conversation_id=conversation_id,
                        article_id=article_id,
                        user_message=user_message,
                    )
                )

            # Display response
            console.print(
                Panel(
                    response,
                    title="[bold magenta]AI Response[/bold magenta]",
                    border_style="magenta",
                    expand=False,
                )
            )

        console.print("[green]Chat session ended[/green]")

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)