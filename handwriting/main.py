import numpy as np
import matplotlib.pyplot as plt


def display_image_and_prediction(index):
    plt.imshow(test_images[index], cmap="gray")
    plt.title(f"Actual: {test_labels[index]}, Predicted: {predicted_classes[index]}")
    plt.axis("off")  # Turn off axis numbering
    plt.show()


def display_image_grid(images, actual_labels, predicted_labels, grid_size=(5, 10)):
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(12, 12))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)

    for i, ax in enumerate(axes.flat):
        # Display image
        ax.imshow(images[i], cmap="gray")

        # Title with actual and predicted labels
        ax.set_title(f"Actual: {actual_labels[i]}\nPredicted: {predicted_labels[i]}")

        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


def display_image_grid_random(
    images, actual_labels, predicted_labels, sample_size, grid_size=(5, 10)
):
    rng = np.random.default_rng(seed=123)
    indices = rng.choice(len(images), size=sample_size, replace=False)

    # Sample images and labels based on the indices
    sampled_images = images[indices]
    sampled_actual_labels = actual_labels[indices]
    sampled_predicted_labels = predicted_labels[indices]

    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(12, 12))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)

    for i, ax in enumerate(axes.flat):
        # Display image
        ax.imshow(sampled_images[i], cmap="gray")

        # Title with actual and predicted labels
        ax.set_title(
            f"Actual: {sampled_actual_labels[i]}\nPredicted: {sampled_predicted_labels[i]}"
        )

        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


def display_image_grid_misclassified_random(
    images, actual_labels, predicted_labels, sample_size, grid_size=(5, 2)
):
    rng = np.random.default_rng(seed=123)
    indices = rng.choice(len(images), size=sample_size, replace=False)

    misclassified_indices = np.where(predicted_labels != actual_labels)[0]
    num_misclassified = len(misclassified_indices)

    # Plot a selection of misclassified examples
    num_plot = min(num_misclassified, 10)  # Limit the number to plot for visibility
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(12, 12))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)

    for i, ax in enumerate(axes.flat):
        if i < num_plot:
            ax.imshow(images[misclassified_indices[i]], cmap="gray")
            ax.set_title(
                f"True: {actual_labels[misclassified_indices[i]]}, Pred: {predicted_labels[misclassified_indices[i]]}"
            )
            ax.axis("off")
        else:
            ax.remove()  # Remove any unused axes if there are less than 10 misclassified examples

    plt.show()


def execute_basic_neural_model(train_images, train_labels, test_images, test_labels):
    # Build the model
    model = models.Sequential(
        [
            layers.Flatten(input_shape=(28, 28)),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    model.fit(train_images, train_labels, epochs=5)

    _, test_acc = model.evaluate(test_images, test_labels)
    print("\nTest accuracy:", test_acc)

    return model


def execute_convolutional_neural_model(
    train_images, train_labels, test_images, test_labels
):
    # Build the model
    model = models.Sequential(
        [
            layers.Conv2D(
                filters=32,
                kernel_size=(3, 3),
                activation="relu",
                input_shape=(28, 28, 1),
            ),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(input_shape=(28, 28)),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    model.fit(train_images, train_labels, epochs=5)

    _, test_acc = model.evaluate(test_images, test_labels)
    print("\nTest accuracy:", test_acc)

    return model


def main():
    # Load the MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalize the images to pixel values between 0 and 1
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    bnn = execute_basic_neural_model(
        train_images, train_labels, test_images, test_labels
    )
    cnn = execute_convolutional_neural_model(
        train_images, train_labels, test_images, test_labels
    )

    predictions = cnn.predict(test_images)
    predicted_classes = np.argmax(predictions, axis=1)
    # num_images = 50
    #
    # display_image_grid(test_images[:num_images],test_labels[:num_images],predicted_classes[:num_images])
    display_image_grid_random(test_images, test_labels, predicted_classes, 50)
    display_image_grid_misclassified_random()


main()
