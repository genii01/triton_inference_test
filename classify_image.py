import torch
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from PIL import Image
import json
from torchvision import transforms
import os


def load_imagenet_labels():
    # ImageNet 클래스 레이블 로드
    labels_path = "imagenet_classes.json"
    try:
        with open(labels_path, "r") as f:
            categories = json.load(f)
    except FileNotFoundError:
        # ImageNet 클래스 이름이 없는 경우 다운로드
        from urllib.request import urlopen

        url = (
            "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        )
        with urlopen(url) as response:
            categories = [line.decode("utf-8").strip() for line in response.readlines()]
        # JSON 파일로 저장
        with open(labels_path, "w") as f:
            json.dump(categories, f, ensure_ascii=False, indent=2)
    return categories


def classify_image(image_path):
    # 모델과 가중치 로드
    weights = MobileNet_V3_Large_Weights.DEFAULT
    model = mobilenet_v3_large(weights=weights)
    model.eval()

    # models 폴더 생성 (없는 경우)
    os.makedirs("models", exist_ok=True)

    # 모델 저장
    model_path = os.path.join("models", "mobilenet_v3_large.pth")
    torch.save(model.state_dict(), model_path)
    print(f"모델이 '{model_path}'에 저장되었습니다.")

    # 이미지 전처리
    # preprocess = weights.transforms()  # 기존 코드 제거

    # 커스텀 전처리 함수 정의
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),  # 짧은 쪽을 256으로 리사이즈
            transforms.CenterCrop(224),  # 중앙 224x224 크롭
            transforms.ToTensor(),  # PIL 이미지를 텐서로 변환 (0-255 -> 0-1)
            transforms.Normalize(  # ImageNet 평균/표준편차로 정규화
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    # 이미지 로드 및 RGB로 변환
    img = Image.open(image_path).convert("RGB")
    img_processed = preprocess(img)

    # 배치 차원 추가
    batch = img_processed.unsqueeze(0)

    # 예측
    with torch.no_grad():
        prediction = model(batch)

    # 결과 처리
    probabilities = torch.nn.functional.softmax(prediction[0], dim=0)
    top_prob, top_catid = torch.topk(probabilities, 1)

    # 클래스 레이블 로드
    categories = load_imagenet_labels()

    # 결과 출력
    for i in range(top_prob.size(0)):
        print(f"클래스 인덱스: {top_catid[i].item()}")
        print(f"클래스 이름: {categories[top_catid[i]]}")
        print(f"확률: {top_prob[i].item():.4f}")


if __name__ == "__main__":
    # 분류할 이미지 경로 지정
    image_path = "./truck.png"  # 실제 이미지 경로로 변경해주세요
    classify_image(image_path)
