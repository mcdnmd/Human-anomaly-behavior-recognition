from fastapi import APIRouter

router = APIRouter(
    prefix='/api'
)


@router.get('/get-info')
def get_info():
    return {"name": "New project"}
