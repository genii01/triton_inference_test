from fastapi import APIRouter, UploadFile, File
from services.inference import TritonInferenceService
from api.schemas import PredictionResponse
import io
from PIL import Image

router = APIRouter()
inference_service = TritonInferenceService()


@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # 이미지 읽기
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    # 알파 채널이 있는 경우 RGB로 변환
    if image.mode in ("RGBA", "LA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        image = image.convert("RGB")

    # 추론 실행
    prediction = await inference_service.predict(image)

    # 딕셔너리 키로 접근하도록 수정
    return PredictionResponse(
        class_id=prediction["class_id"],
        class_name=prediction["class_name"],
        confidence=prediction["confidence"],
    )
