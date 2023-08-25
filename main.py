from ImageClassifier import ImageClassifier  # Import the ImageClassifier class
from PatchGAN_Discriminator import PatchGAN_Discriminator  # Import the PatchGAN_Discriminator class

def main():
    # Initialize Image Classifier
    classifier = ImageClassifier()
    
    # Initialize PatchGAN Discriminator with the same image size as the classifier
    patch_gan_discriminator = PatchGAN_Discriminator(image_size=ImageClassifier.IMAGE_HEIGHT)
    
    
    # Compile and fit the model
    classifier.compile_and_fit(patch_gan_discriminator)
    
    # Here you can add code to use the PatchGAN_Discriminator
    # For example, you might want to use it to evaluate the realism of generated images
    # ...

if __name__ == "__main__":
    main()
