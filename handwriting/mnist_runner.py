class MNISTModelRunner:
    # Load the MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalize the images to pixel values between 0 and 1
    train_images = train_images / 255.0
    test_images = test_images / 255.0
