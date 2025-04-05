import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Define the class labels you want to collect
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']  # Example for the English alphabet
number_of_classes = len(class_labels)
dataset_size = 100  # Number of images to collect per class

cap = cv2.VideoCapture(0)  # Adjust camera index if needed

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

for i, label in enumerate(class_labels):
    class_path = os.path.join(DATA_DIR, label)
    if not os.path.exists(class_path):
        os.makedirs(class_path)

    print('Collecting data for class {}'.format(label))
    print('Press "Q" when ready to start collecting images for this sign.')

    done = False
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        cv2.putText(frame, f'Ready for {label}? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    print(f'Collecting {dataset_size} images for class {label}...')
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        image_path = os.path.join(class_path, '{:04d}.jpg'.format(counter))  # Use formatted counter for better sorting
        cv2.imwrite(image_path, frame)
        print(f'Saved image: {image_path}')
        counter += 1

    print(f'Finished collecting data for class {label}.')

cap.release()
cv2.destroyAllWindows()

print("Data collection complete. You can now run train_classifier.py")