"""Workflow API endpoints."""

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db
from app.config import get_settings
from app.models.orm_models import NewsArticle, RSSFeedSource, WorkflowRun
from app.models.schemas import (
    ArticleListResponse,
    ArticleResponse,
    RSSFeedCreate,
    RSSFeedListResponse,
    RSSFeedResponse,
    WorkflowRunListResponse,
    WorkflowRunResponse,
    WorkflowTriggerRequest,
)
from app.workflow.graph import run_workflow

router = APIRouter()
settings = get_settings()


@router.post("/run", response_model=WorkflowRunResponse)
async def trigger_workflow(
    request: WorkflowTriggerRequest = WorkflowTriggerRequest(),
    db: AsyncSession = Depends(get_db),
) -> WorkflowRunResponse:
    """Manually trigger the news aggregation workflow.

    Args:
        request: Workflow trigger options
        db: Database session

    Returns:
        Workflow run details
    """
    try:
        # Execute workflow
        state = await run_workflow(
            session=db,
            feed_urls=request.feed_urls,
            score_threshold=request.score_threshold,
            force_reprocess=request.force,
        )

        # Get the workflow run record
        stmt = select(WorkflowRun).where(WorkflowRun.id == uuid.UUID(state["run_id"]))
        result = await db.execute(stmt)
        workflow_run = result.scalar_one_or_none()

        if not workflow_run:
            raise HTTPException(status_code=500, detail="Workflow run not found")

        return WorkflowRunResponse.model_validate(workflow_run)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/runs", response_model=WorkflowRunListResponse)
async def list_workflow_runs(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> WorkflowRunListResponse:
    """List workflow run history.

    Args:
        page: Page number (1-indexed)
        page_size: Number of items per page
        db: Database session

    Returns:
        List of workflow runs
    """
    # Get total count
    count_stmt = select(func.count()).select_from(WorkflowRun)
    total = await db.scalar(count_stmt) or 0

    # Get paginated results
    offset = (page - 1) * page_size
    stmt = (
        select(WorkflowRun)
        .order_by(WorkflowRun.started_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    result = await db.execute(stmt)
    runs = result.scalars().all()

    return WorkflowRunListResponse(
        runs=[WorkflowRunResponse.model_validate(run) for run in runs],
        total=total,
    )


@router.get("/runs/{run_id}", response_model=WorkflowRunResponse)
async def get_workflow_run(
    run_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> WorkflowRunResponse:
    """Get details of a specific workflow run.

    Args:
        run_id: Workflow run ID
        db: Database session

    Returns:
        Workflow run details
    """
    stmt = select(WorkflowRun).where(WorkflowRun.id == run_id)
    result = await db.execute(stmt)
    workflow_run = result.scalar_one_or_none()

    if not workflow_run:
        raise HTTPException(status_code=404, detail="Workflow run not found")

    return WorkflowRunResponse.model_validate(workflow_run)


@router.get("/articles", response_model=ArticleListResponse)
async def list_articles(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    min_score: float | None = Query(None, ge=0, le=10),
    published_only: bool = Query(True),
    db: AsyncSession = Depends(get_db),
) -> ArticleListResponse:
    """List news articles.

    Args:
        page: Page number (1-indexed)
        page_size: Number of items per page
        min_score: Minimum total score filter
        published_only: Only show published articles
        db: Database session

    Returns:
        List of articles
    """
    # Build query
    query = select(NewsArticle)
    count_query = select(func.count()).select_from(NewsArticle)

    if min_score is not None:
        query = query.where(NewsArticle.total_score >= min_score)
        count_query = count_query.where(NewsArticle.total_score >= min_score)

    if published_only:
        query = query.where(NewsArticle.is_published == True)
        count_query = count_query.where(NewsArticle.is_published == True)

    # Get total count
    total = await db.scalar(count_query) or 0

    # Get paginated results
    offset = (page - 1) * page_size
    query = query.order_by(NewsArticle.published_at.desc()).offset(offset).limit(page_size)
    result = await db.execute(query)
    articles = result.scalars().all()

    return ArticleListResponse(
        articles=[ArticleResponse.model_validate(article) for article in articles],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/articles/{article_id}", response_model=ArticleResponse)
async def get_article(
    article_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> ArticleResponse:
    """Get article details.

    Args:
        article_id: Article ID
        db: Database session

    Returns:
        Article details
    """
    stmt = select(NewsArticle).where(NewsArticle.id == article_id)
    result = await db.execute(stmt)
    article = result.scalar_one_or_none()

    if not article:
        raise HTTPException(status_code=404, detail="Article not found")

    return ArticleResponse.model_validate(article)


@router.get("/feeds", response_model=RSSFeedListResponse)
async def list_feeds(
    active_only: bool = Query(True),
    db: AsyncSession = Depends(get_db),
) -> RSSFeedListResponse:
    """List RSS feed sources.

    Args:
        active_only: Only show active feeds
        db: Database session

    Returns:
        List of RSS feeds
    """
    query = select(RSSFeedSource)
    count_query = select(func.count()).select_from(RSSFeedSource)

    if active_only:
        query = query.where(RSSFeedSource.is_active == True)
        count_query = count_query.where(RSSFeedSource.is_active == True)

    # Get total count
    total = await db.scalar(count_query) or 0

    # Get results
    query = query.order_by(RSSFeedSource.name)
    result = await db.execute(query)
    feeds = result.scalars().all()

    return RSSFeedListResponse(
        feeds=[RSSFeedResponse.model_validate(feed) for feed in feeds],
        total=total,
    )


@router.post("/feeds", response_model=RSSFeedResponse)
async def create_feed(
    feed: RSSFeedCreate,
    db: AsyncSession = Depends(get_db),
) -> RSSFeedResponse:
    """Add a new RSS feed source.

    Args:
        feed: Feed creation data
        db: Database session

    Returns:
        Created feed
    """
    # Check if feed already exists
    stmt = select(RSSFeedSource).where(RSSFeedSource.url == feed.url)
    result = await db.execute(stmt)
    existing = result.scalar_one_or_none()

    if existing:
        raise HTTPException(status_code=400, detail="Feed URL already exists")

    # Create new feed
    db_feed = RSSFeedSource(
        name=feed.name,
        url=feed.url,
        is_active=feed.is_active,
    )
    db.add(db_feed)
    await db.flush()
    await db.refresh(db_feed)

    return RSSFeedResponse.model_validate(db_feed)


@router.delete("/feeds/{feed_id}")
async def delete_feed(
    feed_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Delete an RSS feed source.

    Args:
        feed_id: Feed ID
        db: Database session

    Returns:
        Success message
    """
    stmt = select(RSSFeedSource).where(RSSFeedSource.id == feed_id)
    result = await db.execute(stmt)
    feed = result.scalar_one_or_none()

    if not feed:
        raise HTTPException(status_code=404, detail="Feed not found")

    await db.delete(feed)
    await db.flush()

    return {"message": "Feed deleted successfully"}


@router.patch("/feeds/{feed_id}/toggle", response_model=RSSFeedResponse)
async def toggle_feed(
    feed_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> RSSFeedResponse:
    """Toggle RSS feed active status.

    Args:
        feed_id: Feed ID
        db: Database session

    Returns:
        Updated feed
    """
    stmt = select(RSSFeedSource).where(RSSFeedSource.id == feed_id)
    result = await db.execute(stmt)
    feed = result.scalar_one_or_none()

    if not feed:
        raise HTTPException(status_code=404, detail="Feed not found")

    feed.is_active = not feed.is_active
    await db.flush()
    await db.refresh(feed)

    return RSSFeedResponse.model_validate(feed)