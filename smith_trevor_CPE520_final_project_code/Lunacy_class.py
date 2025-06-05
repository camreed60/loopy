
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import clear_output, display

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
import networkx as nx
from collections import Counter

from Loopy_sim import LoopySim



class AGMActivation(nn.Module):
    '''create custom activation function from otte's Artifical Group Mind paper'''
    def forward(self, x):
        return 1.7159 * torch.tanh(2 * x / 3)


class NeighborLayer(nn.Module):
    '''Custom network layer that only communicates with nearest neighbors'''
    def __init__(self, num_cells):
        super(NeighborLayer, self).__init__()
        self.num_cells = num_cells                              # Number of cells
        self.weights = nn.Parameter(torch.randn(num_cells, 3))  # Weights for left, center, right from previous layer
        self.bias = nn.Parameter(torch.zeros(num_cells))        # biases for each neuron in the layer
        self.activation = AGMActivation()                       # use custom activation from AGM paper

    def forward(self, x):
        ''' forward propagation throught the layer'''
        
        out = torch.zeros_like(x)

        # get activation from neighbors, with wrap around indexing due to circlular topology
        for i in range(self.num_cells):

            # Neighbor indices with wrapping
            left_idx = (i - 1) % self.num_cells
            center_idx = i
            right_idx = (i + 1) % self.num_cells

            left = x[:, left_idx] * self.weights[i, 0]
            center = x[:, center_idx] * self.weights[i, 1]
            right = x[:, right_idx] * self.weights[i, 2]
            
            out[:, i] = left + center + right + self.bias[i]
        
        # perform activation function
        return self.activation(out)


class MulticlassNeighborLayer(nn.Module):
    ''' Custom network layer for the output layer to give additional neurons to each cell for each class'''
    def __init__(self, num_cells, num_classes):
        super(MulticlassNeighborLayer, self).__init__()
        
        self.num_cells = num_cells       # number of cells in the layer
        self.num_classes = num_classes   # number of classes to pick from

        # Weights for left, center, right neighbors (per class)
        self.weights = nn.Parameter(torch.randn(num_cells, num_classes, 3))  # [cells, classes, 3] 
        
        # Bias per cell and per class
        self.bias = nn.Parameter(torch.zeros(num_cells, num_classes))  # [cells, classes]
        
        # artificial group mind activation function
        self.activation = AGMActivation()

    def forward(self, x):
        ''' forward propagation through the layer'''
        
        batch_size, num_cells, _ = x.size()
        out = torch.zeros(batch_size, num_cells, self.num_classes, device=x.device)

        for i in range(num_cells):
            # Neighbor indices with wrapping
            left_idx = (i - 1) % num_cells
            center_idx = i
            right_idx = (i + 1) % num_cells

            # get activation from neighbors, with wrap around indexing due to circlular topology
            left = x[:, left_idx, :] * self.weights[i, :, 0]  
            center = x[:, center_idx, :] * self.weights[i, :, 1] 
            right = x[:, right_idx, :] * self.weights[i, :, 2] 

            out[:, i, :] = left.squeeze(-1) + center.squeeze(-1) + right.squeeze(-1) + self.bias[i, :] 

        # Perform activation function
        return self.activation(out)


class LunacyMulticlassifier(nn.Module):
    def __init__(self, num_cells, num_classes, num_hidden_layers, num_output_layers=1):
        """Lunacy multi-classifier model that can only talk to its neighbors"""
        super(LunacyMulticlassifier, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.num_output_layers = num_output_layers

        # Hidden layers: NeighborLayer
        self.hidden_layers = nn.ModuleList([
            NeighborLayer(num_cells) for _ in range(num_hidden_layers)
        ])

        # Output layers: MulticlassNeighborLayer
        self.output_layers = nn.ModuleList([
            MulticlassNeighborLayer(num_cells, num_classes) for _ in range(num_output_layers)
        ])

    def forward(self, x):
        ''' forward propagation through the model'''
       
        # propagate through the hidden layers
        for layer in self.hidden_layers:
            x = layer(x)  # Shape: [batch, num_cells]
        
        # Reshape to add class dimension
        x = x.unsqueeze(-1)

        # propagate through the multi-class layers
        for output_layer in self.output_layers:
            x = output_layer(x)  # Output shape: [batch, cells, classes]

        return x 
    
class LunacyMultiTool:
    ''' class used as a toolbox of helper functions to work with the Lunacy model'''
    def __init__(self, datafile, model, sim):
        self.datafile = datafile  # data set file
        self.model = model        # Lunacy model
        self.sim = sim            # Loopy simulator

        # load and preprocess the data set
        raw_data, raw_labels, raw_metadata = self.load_and_label_multiclass_data(class_labels=['predator', 'prey', 'noise'])
        data, labels, metadata = self.shuffle_and_uniform_sample(raw_data, raw_labels, raw_metadata, 40000, class_labels=['predator', 'prey', 'noise'])
        self.train_data, self.train_labels, self.train_meta, self.val_data, self.val_labels, self.val_meta = self.split_data(data, labels, metadata, train_ratio=0.8)

             
    def load_weights(self, weights_path):
        self.model.load_state_dict(torch.load(weights_path))
        self.model.eval()
    
    def load_and_label_multiclass_data(self, input_size=36, class_labels=None):
        '''loads the dataset and assigns multi-class labels '''
        
        if class_labels is None:
            class_labels = ['predator', 'prey', 'noise']  # Default classes

        data = []
        labels = []
        metadata = []

        # Load data from CSV file
        with open(self.datafile, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:

                # Extract colors as data
                colors = eval(row['colors'])  # Convert string representation of list to list
                data.append(colors)

                # One-hot encode the label based on class_labels
                label_index = class_labels.index(row['label'])  # Find the class index
                one_hot_label = [0] * len(class_labels)
                one_hot_label[label_index] = 1                  # Set the corresponding index to 1

                # Create a per-cell label tensor for multi-class output
                per_cell_label = [one_hot_label] * input_size
                labels.append(per_cell_label)

                # Collect metadata fields
                metadata.append({
                    'x_shift': float(row['x_shift']),
                    'y_shift': float(row['y_shift']),
                    'rotation': float(row['rotation']),
                    'label': row['label']
                })

        # Convert data and labels to tensors
        data_tensor = torch.tensor(data, dtype=torch.float32)  
        label_tensor = torch.tensor(labels, dtype=torch.float32) 

        return data_tensor, label_tensor, metadata
    
    def shuffle_and_uniform_sample(self, data, labels, metadata, num_samples, class_labels):
        ''' Shuffle the dataset and uniformly sample the specified number of samples for each class'''
   
        # Shuffle the dataset
        indices = torch.randperm(len(data))
        shuffled_data = data[indices]
        shuffled_labels = labels[indices]
        shuffled_metadata = [metadata[i] for i in indices.tolist()]

        num_classes = len(class_labels)

        # Allocate samples evenly
        samples_per_class = num_samples // num_classes

        
        sampled_data = []
        sampled_labels = []
        sampled_metadata = []
        label_counts = {class_id: 0 for class_id in range(num_classes)}

        # collect samples per class from shuffled data
        for i in range(len(shuffled_data)):

            # Extract the class ID 
            label_class = shuffled_metadata[i]['label']
            class_index = list(class_labels).index(label_class)

            # add sample if we need more of that class
            if label_counts[class_index] < samples_per_class:
                sampled_data.append(shuffled_data[i])
                sampled_labels.append(shuffled_labels[i])
                sampled_metadata.append(shuffled_metadata[i])
                label_counts[class_index] += 1

            # stop if we have enough samples
            if sum(label_counts.values()) >= num_samples:
                break

        # Combine and convert data to tensors
        sampled_data = torch.stack(sampled_data)
        sampled_labels = torch.stack(sampled_labels)

        print("Sampled data shape:", sampled_data.shape)
        print("Sampled labels shape:", sampled_labels.shape)
        print("Number of sampled metadata entries:", len(sampled_metadata))
        print("Label distribution:", Counter(meta['label'] for meta in sampled_metadata))

        return sampled_data, sampled_labels, sampled_metadata

    def split_data(self, data, labels, metadata, train_ratio=0.8):
        ''' Split the data into training and validation sets '''
        
        # Compute split sizes
        train_size = int(train_ratio * len(data))
        val_size = len(data) - train_size

        # Split data and labels
        train_data, val_data = torch.split(data, [train_size, val_size])
        train_labels, val_labels = torch.split(labels, [train_size, val_size])

        # Split metadata
        train_meta = metadata[:train_size]
        val_meta = metadata[train_size:]

        print("Train data shape:", train_data.shape)
        print("Validation data shape:", val_data.shape)
        print("Train label shape:", train_labels.shape)
        print("Validation label shape:", val_labels.shape)

        return train_data, train_labels, train_meta, val_data, val_labels, val_meta
    
    def visualize_datasample(self,data_sample, metadata_sample, data_label):
        
        # Extract metadata
        x_shift = metadata_sample['x_shift']
        y_shift = metadata_sample['y_shift']
        rotation = metadata_sample['rotation']
        label = metadata_sample['label']
        
        # Generate Loopy coordinates
        x_coords, y_coords = self.sim.generate_loopy_coords()

        # generate class image
        if label == 'predator':
            background_image = self.sim.predator_image_creator()
        elif label == 'prey':
            background_image = self.sim.prey_image_creator()
        else:
            background_image = self.sim.nothing_image_creator()

        # plot loopy sample and get true color values
        true_colors, transformed_x_coords, transformed_y_coords = self.sim.overlay_loopy_on_image(background_image, x_coords, y_coords, x_pos=x_shift, y_pos=y_shift, rotation=rotation, plot=True)

        # validate recorded data
        print("true_colors:", true_colors)
        print("true label:", label)

        print("data_sample:", data_sample)
        print("data_label:", data_label)

        # print position and rotation
        print("x center:", x_shift)
        print("y center:", y_shift)
        print("rotation:", rotation)
    
    def train_model(self, num_epochs=1000, lr=0.01, directory='weights', class_labels=None):
        '''Train the LunacyMulticlassifier model'''
        
        # Provide default class labels if not specified
        if class_labels is None:
            class_labels = ["predator", "prey", "noise"]

        # Ensure weights directory exists
        os.makedirs(directory, exist_ok=True)

        # cross entroy loss with class weights and Adam optimizer
        class_weights = torch.tensor([2.5, 2.0, 1.0], dtype=torch.float32, device=self.train_data.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        # Initialize live plot
        plt.figure(figsize=(12, 8))
        num_classes = len(class_labels)

        for epoch in range(num_epochs):
            
            # Training step
            self.model.train()
            outputs = self.model(self.train_data)

            # Reshape from ones-hot to 0, 1, or 2 selected by max activation for CrossEntropyLoss
            outputs_flat = outputs.view(-1, num_classes) 
            train_labels_flat = self.train_labels.argmax(dim=2).view(-1)

            # Compute training loss and update weights
            train_loss = criterion(outputs_flat, train_labels_flat)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # Compute training accuracy
            train_preds = outputs.argmax(dim=2)
            train_accuracy = (train_preds == self.train_labels.argmax(dim=2)).float().mean().item()

            # Per-class training accuracy
            train_correct = (train_preds == self.train_labels.argmax(dim=2))
            train_per_class_acc = [train_correct[self.train_labels.argmax(dim=2) == c].float().mean().item() for c in range(num_classes)]

            # Validation step
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(self.val_data)
                val_outputs_flat = val_outputs.view(-1, num_classes) 
                val_labels_flat = self.val_labels.argmax(dim=2).view(-1)

                val_loss = criterion(val_outputs_flat, val_labels_flat)

                # Compute validation accuracy
                val_preds = val_outputs.argmax(dim=2) 
                val_accuracy = (val_preds == self.val_labels.argmax(dim=2)).float().mean().item()

                # Per-class validation accuracy
                val_correct = (val_preds == self.val_labels.argmax(dim=2))
                val_per_class_acc = [
                    val_correct[self.val_labels.argmax(dim=2) == c].float().mean().item()
                    for c in range(num_classes)
                ]

            # Store losses and accuracies
            train_losses.append(train_loss.item())
            val_losses.append(val_loss.item())
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)

            # Update live plot every 10 epochs
            if (epoch + 1) % 10 == 0:
                
                # clear the output and plot
                clear_output(wait=True)
                plt.clf()

                # Plot training and validation losses
                plt.subplot(2, 1, 1)  # Loss subplot
                plt.plot(range(1, epoch + 2), train_losses, label="Train Loss", color="blue")
                plt.plot(range(1, epoch + 2), val_losses, label="Validation Loss", color="orange")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title("Training and Validation Losses")
                plt.legend()
                plt.grid(True)

                # Plot training and validation accuracies
                plt.subplot(2, 1, 2)  # Accuracy subplot
                plt.plot(range(1, epoch + 2), train_accuracies, label="Train Accuracy", color="green")
                plt.plot(range(1, epoch + 2), val_accuracies, label="Validation Accuracy", color="red")
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                plt.title("Training and Validation Accuracies")
                plt.legend()
                plt.grid(True)

                # Display updated plot
                display(plt.gcf())

                # Print progress
                print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss.item():.4f}, Validation Loss: {val_loss.item():.4f}")
                print(f"Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}")

                # Print per-class accuracies
                for c, class_label in enumerate(class_labels):
                    print(f"Class '{class_label}': Train Accuracy: {train_per_class_acc[c]:.4f}, Validation Accuracy: {val_per_class_acc[c]:.4f}")

                # save weights
                weights_path = f"{directory}/lunacy_classifier_epoch_{epoch + 1}.pth"
                torch.save(self.model.state_dict(), weights_path)
                print(f"Weights saved to {weights_path}")

    def visualize_with_loopy(self, index, ax, print_output=False):
        ''' visualize validation sample with loopy plot'''

        metadata_sample = self.val_meta[index]
        data_sample = self.val_data[index]

        # Extract metadata
        x_shift = metadata_sample['x_shift']
        y_shift = metadata_sample['y_shift']
        rotation = metadata_sample['rotation']
        label = metadata_sample['label']

        # Generate Loopy coordinates
        x_coords, y_coords = self.sim.generate_loopy_coords()

        # Determine the background image based on true label
        if label == 'predator':
            background_image = self.sim.predator_image_creator()
        elif label == 'prey':
            background_image = self.sim.prey_image_creator()
        else:
            background_image = self.sim.nothing_image_creator()

        # Overlay Loopy on the background image
        true_colors, transformed_x_coords, transformed_y_coords = self.sim.overlay_loopy_on_image(
            background_image, x_coords, y_coords, x_pos=x_shift, y_pos=y_shift, rotation=rotation, plot=False
        )
        if print_output:
            print("true_colors:", true_colors)

        # Convert the data sample to a tensor and make a prediction
        input_tensor = torch.tensor(data_sample, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            output_vector = self.model(input_tensor).squeeze(0).numpy()

        if print_output:
            print("Predicted output:", output_vector)

        # Determine predicted classes
        predicted_classes = output_vector.argmax(axis=1) 

        # only add labels as needed to avoid duplicates
        labels_added = {'Predator': False, 'Prey': False, 'Noise': False}

        # Plot Loopy path with markers based on predicted class
        ax.imshow(background_image, cmap='gray', origin='upper', vmin=0, vmax=1)

        for i, (tx, ty) in enumerate(zip(transformed_x_coords, transformed_y_coords)):
            if predicted_classes[i] == 0:  # Predator
                label = 'Predator' if not labels_added['Predator'] else ""
                ax.scatter(tx, ty, color='red', s=100, marker='*', edgecolors='black', linewidth=1, label=label)
                labels_added['Predator'] = True
            elif predicted_classes[i] == 1:  # Prey
                label = 'Prey' if not labels_added['Prey'] else ""
                ax.scatter(tx, ty, color='green', s=100, marker='s', edgecolors='black', linewidth=1, label=label)
                labels_added['Prey'] = True
            elif predicted_classes[i] == 2:  # Noise
                label = 'Noise' if not labels_added['Noise'] else ""
                ax.scatter(tx, ty, color='blue', s=50, marker='o', edgecolors='black', linewidth=1, label=label)
                labels_added['Noise'] = True

        ax.set_xlim(0, background_image.shape[1])
        ax.set_ylim(background_image.shape[0], 0)  # Invert y-axis for correct orientation
        ax.set_title(f"Sample Prediction (Index: {index}, True Label: {label})")
        ax.legend(loc='upper right')
        plt.show()
    
    def animate_samples(self, sample_indices, interval=500, save_as=None):
        ''' creat a video of multiple samples for ease of viewing'''
        
        fig, ax = plt.subplots()

        def update(frame):
            ax.clear()
            sample_index = sample_indices[frame]
            self.visualize_with_loopy(sample_index, ax)
            ax.set_title(f"Sample {sample_index}")

        ani = animation.FuncAnimation(fig, update, frames=len(sample_indices), interval=interval, repeat=True)

        # Save the animation
        if save_as:
            ani.save(save_as, writer='ffmpeg' if save_as.endswith('.mp4') else 'pillow', fps=1000 // interval)

        plt.show()
        return ani

    def micro_validate_model(self, plot=True, class_labels=None):
        """
        Evaluate the model's performance at the cellular level with accuracy, precision, recall, F1-score, and specificity.
        """

        if class_labels is None:
            class_labels = ["predator", "prey", "noise"]

        num_classes = len(class_labels)

        # Collect predictions and true labels across all cells
        with torch.no_grad():
            self.model.eval()
            outputs = self.model(self.val_data)  
            predictions = outputs.argmax(dim=2).flatten() 
            true_labels = self.val_labels.argmax(dim=2).flatten()

        # Compute confusion matrix
        conf_matrix = MulticlassConfusionMatrix(num_classes=num_classes)
        confusion = conf_matrix(predictions, true_labels)

        # Compute metrics
        accuracy_metric = MulticlassAccuracy(num_classes=num_classes, average="macro")
        precision_metric = MulticlassPrecision(num_classes=num_classes, average="macro")
        recall_metric = MulticlassRecall(num_classes=num_classes, average="macro")
        f1_metric = MulticlassF1Score(num_classes=num_classes, average="macro")

        accuracy = accuracy_metric(predictions, true_labels).item()
        precision = precision_metric(predictions, true_labels).item()
        recall = recall_metric(predictions, true_labels).item()
        f1 = f1_metric(predictions, true_labels).item()

        # Calculate specificity
        specificity = []
        for i in range(num_classes):
            tn = confusion.sum() - (confusion[i, :].sum() + confusion[:, i].sum() - confusion[i, i])
            fp = confusion[:, i].sum() - confusion[i, i]
            specificity.append((tn / (tn + fp)).item())
        avg_specificity = sum(specificity) / num_classes

        if plot:
            # Print results
            print("\nAggregate Metrics Across All Cells:")
            print(f"Confusion Matrix:\n{confusion}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"Specificity: {avg_specificity:.4f}\n")

            # Plot metrics
            metrics = [accuracy, precision, recall, f1, avg_specificity]
            metric_names = ["Accuracy", "Precision", "Recall", "F1 Score", "Specificity"]
            plt.figure(figsize=(10, 6))
            plt.bar(metric_names, metrics, color=["blue", "green", "orange", "red", "purple"])
            plt.ylim(0, 1)
            plt.title("Aggregate Metrics Across All Cells")
            plt.ylabel("Metric Value")
            plt.grid(axis="y")
            plt.show()

        # Return metrics as a dictionary
        return {
            "confusion_matrix": confusion,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "specificity": avg_specificity,
        }

    def macro_validate_model(self, threshold, plot=True, class_labels=None):
        """
        Evaluate the model's performance on the validation set at the Loopy level with accuracy, precision, recall, F1-score, and specificity 
        for a given minimum concensus threshold
        """
        if class_labels is None:
            class_labels = ["predator", "prey", "noise"]

        num_classes = len(class_labels)
        num_samples = self.val_data.shape[0] 
       
        # Convert threshold to fraction
        threshold_fraction = threshold / 100

        # Collect predictions and true labels
        with torch.no_grad():
            self.model.eval()
            outputs = self.model(self.val_data)  # Shape: [Batch, cells, classes]
            predictions = outputs.argmax(dim=2)  # Shape: [batch, cells]
            true_labels = self.val_labels.argmax(dim=2)  # Shape: [batch, cells]

        # Determine if each cell's prediction matches the true label
        correct_per_cell = (predictions == true_labels).float() 

        # Calculate the percentage of correct predictions for each sample
        correct_percentage_per_sample = correct_per_cell.mean(dim=1) 

        # Assign macro-level predictions based on the threshold
        noise_class = class_labels.index("noise")
        macro_predictions = torch.full_like(predictions[:, 0], noise_class, dtype=torch.long)  # Default to noise if consensus not met
        for i in range(num_samples):
            if correct_percentage_per_sample[i] >= threshold_fraction:
                # Assign the majority-voted prediction as the sample's macro prediction
                macro_predictions[i] = predictions[i].mode().values

        # get true labels at the sample level
        majority_true_labels = true_labels.mode(dim=1).values

        # Compute confusion matrix
        conf_matrix = MulticlassConfusionMatrix(num_classes=num_classes)
        confusion = conf_matrix(macro_predictions, majority_true_labels)

        # Compute metrics
        accuracy_metric = MulticlassAccuracy(num_classes=num_classes, average="macro")
        precision_metric = MulticlassPrecision(num_classes=num_classes, average="macro")
        recall_metric = MulticlassRecall(num_classes=num_classes, average="macro")
        f1_metric = MulticlassF1Score(num_classes=num_classes, average="macro")

        accuracy = accuracy_metric(macro_predictions, majority_true_labels).item()
        precision = precision_metric(macro_predictions, majority_true_labels).item()
        recall = recall_metric(macro_predictions, majority_true_labels).item()
        f1 = f1_metric(macro_predictions, majority_true_labels).item()

        # Calculate specificity
        specificity = []
        for i in range(num_classes):
            tn = confusion.sum() - (confusion[i, :].sum() + confusion[:, i].sum() - confusion[i, i])
            fp = confusion[:, i].sum() - confusion[i, i]
            specificity.append((tn / (tn + fp)).item())
        avg_specificity = sum(specificity) / num_classes

        if plot:
            # Print results
            print("\nMacro Metrics with Threshold:", threshold)
            print(f"Confusion Matrix:\n{confusion}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"Specificity: {avg_specificity:.4f}\n")

            # Plot metrics
            metrics = [accuracy, precision, recall, f1, avg_specificity]
            metric_names = ["Accuracy", "Precision", "Recall", "F1 Score", "Specificity"]
            plt.figure(figsize=(10, 6))
            plt.bar(metric_names, metrics, color=["blue", "green", "orange", "red", "purple"])
            plt.ylim(0, 1)
            plt.title(f"Macro Metrics with Threshold {threshold}%")
            plt.ylabel("Metric Value")
            plt.grid(axis="y")
            plt.show()

        # Return metrics as a dictionary
        return {
            "confusion_matrix": confusion,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "specificity": avg_specificity,
        }