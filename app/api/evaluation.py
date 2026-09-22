from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/evaluation/latest")
def latest(request: Request) -> dict:
    return request.app.state.service.latest_evaluation()