from src.person1_preprocessing import Preprocessor

def main():
    input_folder = "data/raw"

    preprocessor = Preprocessor()

    results = preprocessor.process_folder(input_folder)

    print("\n✅ Preprocessing Completed!")
    print("Processed Images:\n")

    for name, path in results:
        print(f"{name} -> {path}")

    print("\n➡️ These images are ready for Person 2 (Edge Detection)")

if __name__ == "__main__":
    main()