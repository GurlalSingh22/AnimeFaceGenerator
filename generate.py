from inference import generate_faces


def main():
    print("\nAnime Face Generator")
    print("--------------------------")
    print("1  = Generate 1 Face")
    print("4  = Generate 4 Faces")
    print("16 = Generate 16 Faces")
    print("64 = Generate 64 Faces")

    num = int(input("\nEnter Number of Faces : "))
    if num not in {1, 4, 16, 64}:
        raise ValueError("Please choose one of: 1, 4, 16, 64")

    output_path = generate_faces(num)
    print(f"\n✅ Saved : {output_path}")


if __name__ == "__main__":
    main()
