from predict import get_coordinates
import os

def main():
    # Get list of image files in current directory
    image_files = [f for f in os.listdir('.') if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("No image files found in the current directory!")
        return

    # Display available images and URL option
    print("\nSelect input type:")
    print("1. Local image")
    print("2. Image URL")
    
    while True:
        try:
            input_type = int(input("\nEnter your choice (1 or 2): "))
            if input_type in [1, 2]:
                break
            print("Invalid choice. Please enter 1 or 2.")
        except ValueError:
            print("Please enter a valid number.")

    image_input = ""
    if input_type == 1:
        # Handle local image selection
        print("\nAvailable local images:")
        for i, img in enumerate(image_files, 1):
            print(f"{i}. {img}")

        while True:
            try:
                selection = int(input("\nSelect image number: ")) - 1
                if 0 <= selection < len(image_files):
                    image_input = image_files[selection]
                    break
                print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
    else:
        # Handle URL input
        while True:
            image_input = input("\nEnter image URL: ").strip()
            if image_input:
                break
            print("URL cannot be empty. Please try again.")

    # Get prompt
    while True:
        prompt = input("\nEnter your prompt: ").strip()
        if prompt:
            break
        print("Prompt cannot be empty. Please try again.")

    print("\nSending request...")
    coordinates = get_coordinates(image_input, prompt)
    
    if coordinates is not None:
        print("\nRequest completed successfully!")
        print(f"Click coordinates: {coordinates}")
    else:
        print("\nFailed to get coordinates.")

if __name__ == "__main__":
    main() 