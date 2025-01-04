from predict import get_coordinates
import os

def main():
    # Get list of image files in current directory
    image_files = [f for f in os.listdir('.') if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("No image files found in the current directory!")
        return

    # Display available images
    print("\nAvailable images:")
    for i, img in enumerate(image_files, 1):
        print(f"{i}. {img}")

    # Get image selection
    while True:
        try:
            selection = int(input("\nSelect image number: ")) - 1
            if 0 <= selection < len(image_files):
                image_path = image_files[selection]
                break
            print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

    # Get prompt
    prompt = input("\nEnter your prompt: ").strip()

    print("\nSending request...")
    coordinates = get_coordinates(image_path, prompt, stream=False)
    
    if coordinates is not None:
        print("\nRequest completed successfully!")
        print(f"Click coordinates: {coordinates}")
    else:
        print("\nFailed to get coordinates.")

if __name__ == "__main__":
    main() 