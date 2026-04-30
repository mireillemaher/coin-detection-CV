from src.preprocessing import Preprocessor


def main():
    input_folder = "data/raw"
    preprocessor = Preprocessor()
    results = preprocessor.process_folder(input_folder)

    print("\nPreprocessing complete.")
    print("Processed images:\n")
    for name, path in results:
        print(f"  {name} -> {path}")
    print("\nThese images are ready for the edge detection stage.")


if __name__ == "__main__":
    main()
