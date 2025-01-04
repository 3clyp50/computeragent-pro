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

    print("\nStarting streaming response...")
    print("-" * 50)
    
    response = get_coordinates(image_path, prompt, stream=True)
    if response:
        for line in response.iter_lines():
            if line:
                print(line.decode('utf-8'))
    
    print("-" * 50)
    print("Stream completed.")

if __name__ == "__main__":
    main() 