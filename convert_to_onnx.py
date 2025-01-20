import torch
from torchvision.models import mobilenet_v3_large
import onnx
import onnxruntime
import os


def convert_to_onnx():
    # PyTorch 모델 로드
    model = mobilenet_v3_large()
    model.load_state_dict(torch.load("./models/mobilenet_v3_large.pth"))
    model.eval()

    # models 폴더 생성
    os.makedirs("models", exist_ok=True)

    # 더미 입력 생성 (배치 크기 1, 채널 3, 높이 224, 너비 224)
    dummy_input = torch.randn(1, 3, 224, 224)

    # ONNX 포맷으로 변환
    onnx_path = os.path.join("models", "mobilenet_v3_large.onnx")
    torch.onnx.export(
        model,  # 실행될 모델
        dummy_input,  # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
        onnx_path,  # models 폴더에 저장
        export_params=True,  # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
        opset_version=11,  # ONNX 버전
        do_constant_folding=True,  # 최적화시 상수폴딩을 사용할지의 여부
        input_names=["input"],  # 모델의 입력값을 가리키는 이름
        output_names=["output"],  # 모델의 출력값을 가리키는 이름
        dynamic_axes={  # 가변적인 길이를 가진 차원
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    print(f"모델이 '{onnx_path}'로 변환되어 저장되었습니다.")

    # ONNX 모델 검증
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX 모델 검증이 완료되었습니다.")

    # ONNX Runtime을 사용한 추론 테스트
    ort_session = onnxruntime.InferenceSession(onnx_path)

    # 테스트 실행
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)
    print("ONNX Runtime 추론 테스트가 완료되었습니다.")


if __name__ == "__main__":
    convert_to_onnx()
