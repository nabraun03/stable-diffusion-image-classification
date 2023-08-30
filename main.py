from ImageClassifier import ImageClassifier  # Import the ImageClassifier class
from PatchGAN_Discriminator import PatchGAN_Discriminator  # Import the PatchGAN_Discriminator class
from Simple_CNN import Simple_CNN
import tensorflow as tf

def generate_model_diagrams():
    p = PatchGAN_Discriminator()
    c = Simple_CNN()
    tf.keras.utils.plot_model(p, to_file='patchgan.png', show_shapes = True)
    tf.keras.utils.plot_model(c, to_file = 'cnn.png', show_shapes = True)


def main():
    # Initialize Image Classifier
    classifier = ImageClassifier()
    
    # Initialize PatchGAN Discriminator with the same image size as the classifier
    patch_gan_discriminator = PatchGAN_Discriminator(image_size=ImageClassifier.IMAGE_HEIGHT)
    
    # Compile and fit the model
    classifier.compile_and_fit(patch_gan_discriminator)

if __name__ == "__main__":
    main()
