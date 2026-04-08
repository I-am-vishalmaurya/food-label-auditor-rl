# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the FoodLabelAuditor Environment.

Exposes the FoodLabelAuditorEnvironment over HTTP and WebSocket endpoints,
compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - GET /tasks: List available tasks with grader info
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import AuditAction, AuditObservation
    from .food_label_auditor_environment import FoodLabelAuditorEnvironment
except ImportError:
    from models import AuditAction, AuditObservation
    from server.food_label_auditor_environment import FoodLabelAuditorEnvironment


app = create_app(
    FoodLabelAuditorEnvironment,
    AuditAction,
    AuditObservation,
    env_name="food_label_auditor",
    max_concurrent_envs=4,
)


TASK_DEFINITIONS = [
    {
        "id": "food_audit_task1",
        "name": "Single Product Audit",
        "difficulty": "easy",
        "description": (
            "Audit 1 high-risk product (NOVA-4, Nutri-Score D/E) for 1 health profile. "
            "Tests basic food safety understanding and risk classification."
        ),
        "max_steps": 1,
        "grader": "grade_task1",
        "score_range": [0.0, 1.0],
        "reset_params": {"task_id": 1},
    },
    {
        "id": "food_audit_task2",
        "name": "Multi-Product Multi-Profile Audit",
        "difficulty": "medium",
        "description": (
            "Audit 10 products across 2 health profiles. Agent must give different "
            "risk levels for the same product depending on user health conditions."
        ),
        "max_steps": 10,
        "grader": "grade_task2",
        "score_range": [0.0, 1.0],
        "reset_params": {"task_id": 2},
    },
    {
        "id": "food_audit_task3",
        "name": "Adversarial Label Auditing",
        "difficulty": "hard",
        "description": (
            "Audit 30 products including adversarial cases where marketing claims "
            "contradict actual ingredients. Requires FSSAI-specific regulatory reasoning."
        ),
        "max_steps": 30,
        "grader": "grade_task3",
        "score_range": [0.0, 1.0],
        "reset_params": {"task_id": 3},
    },
]


@app.get("/tasks", tags=["Tasks"])
async def list_tasks():
    """List all available tasks with their grader information."""
    return {"tasks": TASK_DEFINITIONS}


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution.

    Usage:
        uv run --project . server
        python -m food_label_auditor.server.app
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
