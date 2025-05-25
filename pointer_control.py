
'''import pygame
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import random
import time
import pandas as pd
from scipy.signal import butter, filtfilt

# Initialize Pygame
pygame.init()

# Set up the display
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("EEG Pointer Control")

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)

# Load the model
model = load_model('eeg_model.h5')

# EEG Data Processing Functions
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def bandpass_filter_data(data, lowcut=8, highcut=30, fs=250):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return filtfilt(b, a, data, axis=0)

class EEGBuffer:
    def __init__(self, buffer_size=1000, num_channels=22):
        self.buffer_size = buffer_size
        self.num_channels = num_channels
        self.buffer = np.zeros((num_channels, buffer_size))
        self.current_index = 0
        self.is_full = False

    def add_data(self, new_data):
        # new_data should be shape (num_channels,)
        if self.current_index < self.buffer_size:
            self.buffer[:, self.current_index] = new_data
            self.current_index += 1
            if self.current_index == self.buffer_size:
                self.is_full = True
        else:
            # Shift buffer and add new data
            self.buffer = np.roll(self.buffer, -1, axis=1)
            self.buffer[:, -1] = new_data

    def get_processed_data(self):
        if not self.is_full:
            return None
        
        # Process the data similar to training
        data = self.buffer.copy()
        
        # Bandpass filter
        data = bandpass_filter_data(data.T).T
        
        # Z-score normalization per channel
        data = (data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-6)
        
        # Reshape for model input: (1, channels, time, 1)
        return data.reshape(1, self.num_channels, self.buffer_size, 1)

class Pointer:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = 10
        self.speed = 5

    def move(self, direction):
        if direction == 0:  # Up
            self.y -= self.speed
        elif direction == 1:  # Right
            self.x += self.speed
        elif direction == 2:  # Down
            self.y += self.speed
        elif direction == 3:  # Left
            self.x -= self.speed

        # Keep pointer within screen bounds
        self.x = max(self.radius, min(WINDOW_WIDTH - self.radius, self.x))
        self.y = max(self.radius, min(WINDOW_HEIGHT - self.radius, self.y))

    def draw(self, screen):
        pygame.draw.circle(screen, BLUE, (int(self.x), int(self.y)), self.radius)

class Target:
    def __init__(self):
        self.radius = 15
        self.respawn()

    def respawn(self):
        self.x = random.randint(self.radius, WINDOW_WIDTH - self.radius)
        self.y = random.randint(self.radius, WINDOW_HEIGHT - self.radius)

    def draw(self, screen):
        pygame.draw.circle(screen, RED, (int(self.x), int(self.y)), self.radius)

def main():
    clock = pygame.time.Clock()
    pointer = Pointer(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
    target = Target()
    running = True
    last_prediction_time = time.time()
    prediction_interval = 0.5  # Time between predictions in seconds
    
    # Initialize EEG buffer
    eeg_buffer = EEGBuffer()
    
    # For demonstration, we'll simulate EEG data
    # In a real application, this would come from your EEG device
    def simulate_eeg_data():
        return np.random.randn(22)  # 22 channels of random data

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Simulate getting new EEG data
        new_eeg_data = simulate_eeg_data()
        eeg_buffer.add_data(new_eeg_data)

        # Get model prediction at regular intervals
        current_time = time.time()
        if current_time - last_prediction_time >= prediction_interval:
            processed_data = eeg_buffer.get_processed_data()
            if processed_data is not None:
                prediction = model.predict(processed_data, verbose=0)
                direction = np.argmax(prediction[0])
                pointer.move(direction)
                last_prediction_time = current_time

        # Check if pointer reached target
        distance = np.sqrt((pointer.x - target.x)**2 + (pointer.y - target.y)**2)
        if distance < (pointer.radius + target.radius):
            target.respawn()

        # Draw everything
        screen.fill(WHITE)
        pointer.draw(screen)
        target.draw(screen)
        
        # Draw buffer status
        buffer_percent = (eeg_buffer.current_index / eeg_buffer.buffer_size) * 100
        pygame.draw.rect(screen, GREEN, (10, 10, buffer_percent * 2, 20))
        pygame.draw.rect(screen, BLACK, (10, 10, 200, 20), 2)
        
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main() '''


import pygame
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import random
import time
from scipy.signal import butter, filtfilt

# Initialize Pygame
pygame.init()
pygame.font.init()

# Set up the display
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("EEG Pointer Control")

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)

# Load the model
model = load_model(r"E:\sem8\Final\NEW_TRY\Final-BCI\eeg_model.h5")

# EEG Data Processing Functions
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def bandpass_filter_data(data, lowcut=8, highcut=30, fs=250):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return filtfilt(b, a, data, axis=0)

class EEGBuffer:
    def __init__(self, buffer_size=1000, num_channels=22):
        self.buffer_size = buffer_size
        self.num_channels = num_channels
        self.buffer = np.zeros((num_channels, buffer_size))
        self.current_index = 0
        self.is_full = False

    def add_data(self, new_data):
        if self.current_index < self.buffer_size:
            self.buffer[:, self.current_index] = new_data
            self.current_index += 1
            if self.current_index == self.buffer_size:
                self.is_full = True
        else:
            self.buffer = np.roll(self.buffer, -1, axis=1)
            self.buffer[:, -1] = new_data

    def get_processed_data(self):
        if not self.is_full:
            return None
        data = self.buffer.copy()
        data = bandpass_filter_data(data.T).T
        data = (data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-6)
        return data.reshape(1, self.num_channels, self.buffer_size, 1)

class Pointer:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = 10
        self.speed = 5

    def move(self, direction):
        if direction == 0:  # Up
            self.y -= self.speed
        elif direction == 1:  # Right
            self.x += self.speed
        elif direction == 2:  # Down
            self.y += self.speed
        elif direction == 3:  # Left
            self.x -= self.speed

        self.x = max(self.radius, min(WINDOW_WIDTH - self.radius, self.x))
        self.y = max(self.radius, min(WINDOW_HEIGHT - self.radius, self.y))

    def draw(self, screen):
        pygame.draw.circle(screen, BLUE, (int(self.x), int(self.y)), self.radius)

class Target:
    def __init__(self):
        self.radius = 15
        self.respawn()

    def respawn(self):
        self.x = random.randint(self.radius, WINDOW_WIDTH - self.radius)
        self.y = random.randint(self.radius, WINDOW_HEIGHT - self.radius)

    def draw(self, screen):
        pygame.draw.circle(screen, RED, (int(self.x), int(self.y)), self.radius)

def main():
    clock = pygame.time.Clock()
    pointer = Pointer(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
    target = Target()
    running = True
    last_prediction_time = time.time()
    prediction_interval = 0.5

    eeg_buffer = EEGBuffer()
    font = pygame.font.SysFont(None, 36)
    predicted_class = None
    direction_labels = ["Up", "Right", "Down", "Left"]

    def simulate_eeg_data():
        return np.random.randn(22)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        new_eeg_data = simulate_eeg_data()
        eeg_buffer.add_data(new_eeg_data)

        current_time = time.time()
        if current_time - last_prediction_time >= prediction_interval:
            processed_data = eeg_buffer.get_processed_data()
            if processed_data is not None:
                prediction = model.predict(processed_data, verbose=0)
                direction = np.argmax(prediction[0])
                predicted_class = direction
                pointer.move(direction)
                last_prediction_time = current_time

        distance = np.sqrt((pointer.x - target.x)**2 + (pointer.y - target.y)**2)
        if distance < (pointer.radius + target.radius):
            target.respawn()

        screen.fill(WHITE)
        pointer.draw(screen)
        target.draw(screen)

        # Draw buffer status
        buffer_percent = (eeg_buffer.current_index / eeg_buffer.buffer_size) * 100
        pygame.draw.rect(screen, GREEN, (10, 10, buffer_percent * 2, 20))
        pygame.draw.rect(screen, BLACK, (10, 10, 200, 20), 2)

        # Display predicted class
        if predicted_class is not None:
            label = direction_labels[predicted_class]
            text_surface = font.render(f"Predicted Class: {label}", True, BLACK)
            screen.blit(text_surface, (10, 40))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
