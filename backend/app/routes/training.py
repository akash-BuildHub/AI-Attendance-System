from fastapi import APIRouter
from app.services.training_service import train_prototypes, load_prototypes

router = APIRouter(prefix="/train", tags=["training"])

@router.post("")
def train():
    return train_prototypes()

@router.get("/status")
def status():
    prot = load_prototypes()
    return {
        "success": True,
        "data": {
            "people": sorted(list(prot.keys())),
            "count": len(prot)
        }
    }
