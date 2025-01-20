import numpy as np
from PIL import Image
import tritonclient.http as httpclient
from tritonclient.utils import triton_to_np_dtype
from torchvision import transforms
import json
import os
import time


class TritonInferenceService:
    def __init__(self):
        # 1. Triton 서버 연결 시도
        max_retries = 5
        retry_delay = 5  # seconds

        for attempt in range(max_retries):
            try:
                # 2. 클라이언트 초기화
                triton_url = os.getenv("TRITON_URI", "triton:8000")
                self.client = httpclient.InferenceServerClient(
                    url=triton_url,
                    verbose=True,  # 디버깅을 위해 verbose 활성화
                    concurrency=1,
                )

                # 3. 서버 상태 확인
                if not self.client.is_server_live():
                    raise Exception("Triton server is not live")
                break
            except Exception as e:
                # 4. 실패시 재시도
                if attempt == max_retries - 1:
                    raise Exception(
                        f"Failed to connect to Triton server after {max_retries} attempts"
                    )
                time.sleep(retry_delay)

        self.preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # ImageNet 클래스 레이블 로드
        with open("imagenet_classes.json", "r") as f:
            self.classes = json.load(f)

    async def predict(self, image: Image.Image):
        # 이미지 전처리
        img_tensor = self.preprocess(image)
        # 배치 차원 추가 (3,224,224) -> (1,3,224,224)
        input_data = img_tensor.unsqueeze(0).numpy()

        # Triton 입력 설정
        inputs = []
        inputs.append(httpclient.InferInput("input", input_data.shape, "FP32"))
        inputs[0].set_data_from_numpy(input_data)

        # 추론 요청
        results = self.client.infer("mobilenet", inputs)
        output = results.as_numpy("output")

        # 결과 처리 (배치 차원 제거)
        output = output.squeeze(0)  # 배치 차원 제거
        class_id = np.argmax(output)
        confidence = float(output[class_id])
        class_name = self.classes[class_id]

        return {
            "class_id": int(class_id),
            "class_name": class_name,
            "confidence": confidence,
        }
