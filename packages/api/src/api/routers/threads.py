"""Conversation thread listing and task history endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel

from ..dependencies import get_task_service
from ..schemas import (
    CreateTaskRequest,
    ResumeTaskRequest,
    TaskResponse,
    ThreadEventResponse,
    ThreadResponse,
)
from ..services.task_service import TaskService

router = APIRouter(prefix="/threads", tags=["threads"])


class CreateThreadRequest(BaseModel):
    title: str | None = None


@router.post("", response_model=ThreadResponse, status_code=status.HTTP_201_CREATED)
async def create_thread(
    body: CreateThreadRequest,
    service: TaskServiceDep,
) -> ThreadResponse:
    return ThreadResponse.from_record(await service.create_thread(body.title))


@router.delete("/{thread_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_thread(thread_id: str, service: TaskServiceDep):
    """Delete a thread and all of its tasks, runs, and events."""
    from fastapi import Response

    from ..errors import ThreadNotFound

    if not await service.delete_thread(thread_id):
        raise ThreadNotFound(thread_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)

TaskServiceDep = Annotated[TaskService, Depends(get_task_service)]
Limit = Annotated[int, Query(ge=1, le=500)]
Offset = Annotated[int, Query(ge=0)]


@router.post("/{thread_id}/tasks", response_model=TaskResponse, status_code=202)
async def create_thread_task(
    thread_id: str,
    body: CreateTaskRequest,
    service: TaskServiceDep,
) -> TaskResponse:
    """Append a new task to an existing conversation thread."""
    task = await service.create_thread_task(
        thread_id=thread_id,
        goal=body.goal,
        track=body.track,
        metadata=body.metadata,
        workspace=body.workspace,
    )
    return TaskResponse.from_task(task, status="pending")


@router.get("", response_model=list[ThreadResponse])
async def list_threads(
    service: TaskServiceDep,
    limit: Limit = 100,
    offset: Offset = 0,
) -> list[ThreadResponse]:
    """Return conversation threads ordered by most recent activity."""
    records = await service.list_threads(limit=limit, offset=offset)
    return [ThreadResponse.from_record(record) for record in records]


@router.get("/{thread_id}/tasks", response_model=list[TaskResponse])
async def list_thread_tasks(
    thread_id: str,
    service: TaskServiceDep,
    limit: Limit = 100,
    offset: Offset = 0,
) -> list[TaskResponse]:
    """Return one thread's tasks ordered newest-first."""
    tasks = await service.list_thread_task_details(
        thread_id,
        limit=limit,
        offset=offset,
    )
    return [
        TaskResponse.from_task(
            task,
            status=run.status.value if run is not None else None,
            run=run,
        )
        for task, run in tasks
    ]


@router.get("/{thread_id}/tasks/{task_id}", response_model=TaskResponse)
async def get_thread_task(
    thread_id: str,
    task_id: str,
    service: TaskServiceDep,
) -> TaskResponse:
    """Return a task only when it belongs to the requested conversation thread."""
    task, run = await service.get_task_in_thread(thread_id, task_id)
    return TaskResponse.from_task(
        task,
        status=run.status.value if run is not None else None,
        run=run,
    )


@router.post("/{thread_id}/tasks/{task_id}/resume", response_model=TaskResponse)
async def resume_thread_task(
    thread_id: str,
    task_id: str,
    body: ResumeTaskRequest,
    service: TaskServiceDep,
) -> TaskResponse:
    task = await service.resume_task_in_thread(
        thread_id,
        task_id,
        resume_value=body.resume_value,
        checkpoint_id=body.checkpoint_id,
    )
    return TaskResponse.from_task(task, status="pending")


@router.get("/{thread_id}/events", response_model=list[ThreadEventResponse])
async def list_thread_events(
    thread_id: str,
    service: TaskServiceDep,
) -> list[ThreadEventResponse]:
    """Return the durable event history for every task in a thread."""
    events = await service.list_thread_events(thread_id)
    return [
        ThreadEventResponse(task_id=task_id, type=event.type, payload=event.payload)
        for task_id, event in events
    ]
