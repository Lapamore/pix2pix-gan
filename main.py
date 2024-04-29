import argparse
from inference import ImageColorizer


def main():
    parser = argparse.ArgumentParser(description="Раскраска черно-белого изображения.")
    parser.add_argument("--image_path", type=str, help="Путь к изображению.")
    parser.add_argument("--device", type=str, help="Устройство для обработки (cpu или cuda).")
    parser.add_argument('--save_image', action='store_true', help='Флаг для сохранения изображения.')

    args = parser.parse_args()
    inference = ImageColorizer(
        image_path = args.image_path,
        device = args.device,
        save_image = args.save_image
    )

    inference.load_model()
    inference.inference_image()

if __name__ == "__main__":
    main()
