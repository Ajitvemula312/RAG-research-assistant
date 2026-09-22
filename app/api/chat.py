from fastapi import APIRouter, Request

from app.schemas.models import ResearchRequest, ResearchResponse

router = APIRouter()


@router.post("/chat", response_model=ResearchResponse)
@router.post("/research", response_model=ResearchResponse)
def research(request: ResearchRequest, app_request: Request) -> ResearchResponse:
    return app_request.app.state.service.answer(request)