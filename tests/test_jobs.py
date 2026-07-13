"""Tests for `jobs` namespace."""

from __future__ import annotations

import httpx

from ._helpers import json_response


def _make_job(job_id: str, status: str = "succeeded") -> dict[str, object]:
    return {
        "id": job_id,
        "type": "batch_stt",
        "status": status,
        "progress_pct": 100 if status == "succeeded" else 50,
        "result_url": "https://example.com/out.json" if status == "succeeded" else None,
        "error_message": None,
        "error_code": None,
        "created_at": "1970-01-01T00:00:00Z",
        "finished_at": "1970-01-01T00:00:00Z" if status == "succeeded" else None,
    }


async def test_jobs_list_forwards_filters(build_mock_client):  # type: ignore[no-untyped-def]
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(dict(request.url.params))
        return json_response(
            200,
            {
                "object": "list",
                "data": [_make_job("j1")],
                "has_more": False,
                "next_cursor": None,
            },
        )

    async with build_mock_client(handler) as hakim:
        await hakim.jobs.list(
            status="succeeded", type="batch_stt", limit=10, cursor="j_prev"
        )

    assert captured == {
        "status": "succeeded",
        "type": "batch_stt",
        "limit": "10",
        "cursor": "j_prev",
    }


async def test_jobs_retrieve_url_encodes_id(build_mock_client):  # type: ignore[no-untyped-def]
    seen_path: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_path.append(request.url.raw_path.decode())
        return json_response(200, _make_job("j 1"))

    async with build_mock_client(handler) as hakim:
        job = await hakim.jobs.retrieve("j 1")

    assert seen_path == ["/v1/jobs/j%201"]
    assert job["id"] == "j 1"


async def test_jobs_iter_walks_every_page(build_mock_client):  # type: ignore[no-untyped-def]
    pages = [
        {
            "object": "list",
            "data": [_make_job("j1"), _make_job("j2")],
            "has_more": True,
            "next_cursor": "cur1",
        },
        {
            "object": "list",
            "data": [_make_job("j3")],
            "has_more": False,
            "next_cursor": None,
        },
    ]
    call = [0]

    def handler(_: httpx.Request) -> httpx.Response:
        body = pages[call[0]]
        call[0] += 1
        return json_response(200, body)

    async with build_mock_client(handler) as hakim:
        out: list[str] = []
        async for j in hakim.jobs.iter():
            out.append(j["id"])  # type: ignore[typeddict-item]

    assert out == ["j1", "j2", "j3"]
