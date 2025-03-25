import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, 
                            confusion_matrix, classification_report, average_precision_score,
                            RocCurveDisplay, PrecisionRecallDisplay)
import logging

def set_plot_style():
    """Set plot style for attractive visualizations."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("deep")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, best_epoch, output_file='training_curves.png'):
    """Plot training and validation loss/accuracy curves."""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='Validation Loss')
    plt.axvline(x=best_epoch, color='g', linestyle='--', label='Best Model')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accs) + 1), train_accs, 'b-', label='Training Accuracy')
    plt.plot(range(1, len(val_accs) + 1), val_accs, 'r-', label='Validation Accuracy')
    plt.axvline(x=best_epoch, color='g', linestyle='--', label='Best Model')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Training curves saved to {output_file}")

    # Write data to CSV for further analysis
    training_data = {
        'Epoch': range(1, len(train_losses) + 1),
        'Train Loss': train_losses,
        'Validation Loss': val_losses,
        'Train Accuracy': train_accs,
        'Validation Accuracy': val_accs
    }
    training_df = pd.DataFrame(training_data)
    training_df.to_csv('training_curves_data.csv', index=False)


def plot_roc_curve(y_true, y_scores, output_file='roc_curve.png'):
    """Plot ROC curve."""
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    # Add some threshold annotations
    for i, threshold in enumerate(np.linspace(0, 1, 5)):
        closest_idx = np.argmin(np.abs(roc_thresholds - threshold))
        if closest_idx < len(fpr):  # Make sure we don't go out of bounds
            plt.annotate(f'Threshold: {threshold:.1f}', 
                         xy=(fpr[closest_idx], tpr[closest_idx]),
                         xytext=(fpr[closest_idx]+0.05, tpr[closest_idx]-0.05),
                         arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"ROC curve saved to {output_file}")

    # Write data to CSV for further analysis
    roc_data = pd.DataFrame({'False Positive Rate': fpr, 'True Positive Rate': tpr})
    roc_data['Threshold'] = roc_thresholds
    roc_data.to_csv('roc_curve_data.csv', index=False)
    logging.info("ROC curve data saved to roc_curve_data.csv")
    
    return roc_auc


def plot_precision_recall_curve(y_true, y_scores, output_file='precision_recall_curve.png'):
    """Plot Precision-Recall curve."""
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)

    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='green', lw=2, 
             label=f'Precision-Recall curve (AP = {avg_precision:.3f})')
    plt.axhline(y=sum(y_true)/len(y_true), color='navy', 
                linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve', fontweight='bold')
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)

    # Add some threshold annotations
    # Adjust to avoid index errors
    for i, threshold in enumerate(np.linspace(0.2, 0.8, 4)):
        if len(pr_thresholds) > 0:
            closest_idx = np.argmin(np.abs(pr_thresholds - threshold))
            if closest_idx < len(precision)-1:  # Precision is one element longer than thresholds
                plt.annotate(f'Threshold: {threshold:.1f}', 
                             xy=(recall[closest_idx], precision[closest_idx]),
                             xytext=(recall[closest_idx]-0.15, precision[closest_idx]-0.15),
                             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Precision-Recall curve saved to {output_file}")

    # Write data to CSV for further analysis
    pr_data = pd.DataFrame({'Recall': recall, 'Precision': precision})
    pr_data['Threshold'] = pr_thresholds
    pr_data.to_csv('precision_recall_data.csv', index=False)
    
    return avg_precision


def plot_confusion_matrix(y_true, y_pred, output_file='confusion_matrix.png'):
    """Plot confusion matrix."""
    conf_matrix = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'])

    # Add percentages
    group_counts = ["{0:0.0f}".format(value) for value in conf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in conf_matrix.flatten()/np.sum(conf_matrix)]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    for i in range(2):
        for j in range(2):
            plt.text(j+0.5, i+0.5, labels[i, j], 
                     ha="center", va="center", color="black" if conf_matrix[i, j] < conf_matrix.max()/2 else "white",
                     fontsize=12, fontweight='bold')

    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Confusion matrix saved to {output_file}")
    
    return conf_matrix


def plot_score_distribution(scores, labels=None, output_file='prediction_score_distribution.png'):
    """
    Plot distribution of prediction scores with different colors for positive and negative targets.
    
    Parameters:
    scores (array-like): Model prediction scores
    labels (array-like, optional): True labels corresponding to scores. If provided, scores will be colored by class.
    output_file (str): Path to save the output image
    """
    plt.figure(figsize=(10, 6))
    
    if labels is not None:
        # Convert to numpy arrays if they aren't already
        scores = np.array(scores)
        labels = np.array(labels)
        
        # Create a DataFrame for seaborn to use for stacked histograms
        import pandas as pd
        df = pd.DataFrame({
            'score': scores,
            'class': ['Positive Class' if label == 1 else 'Negative Class' for label in labels]
        })
        
        # Plot stacked distributions
        sns.histplot(data=df, x='score', hue='class', bins=50, 
                    multiple="stack", palette={'Negative Class': 'royalblue', 'Positive Class': 'crimson'})
    else:
        # If no labels are provided, just plot all scores in one color
        sns.histplot(scores, bins=50, color='royalblue', label='All Predictions')
    
    plt.axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold (0.5)')
    plt.xlabel('Prediction Score')
    plt.ylabel('Count')
    plt.title('Distribution of Model Prediction Scores', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Prediction score distribution saved to {output_file}")


def create_model_dashboard(val_scores, val_labels, train_losses, val_losses, best_epoch, 
                          output_file='model_performance_dashboard.png'):
    """Create a dashboard of model performance metrics."""
    val_preds = (val_scores > 0.5).astype(int)
    conf_matrix = confusion_matrix(val_labels, val_preds)
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(val_labels, val_scores)
    roc_auc = auc(fpr, tpr)
    
    # Calculate PR curve
    precision, recall, _ = precision_recall_curve(val_labels, val_scores)
    avg_precision = average_precision_score(val_labels, val_scores)
    
    plt.figure(figsize=(16, 12))

    # ROC Curve (top left)
    plt.subplot(2, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    # Precision-Recall Curve (top right)
    plt.subplot(2, 2, 2)
    plt.plot(recall, precision, color='green', lw=2, 
             label=f'PR curve (AP = {avg_precision:.3f})')
    plt.axhline(y=sum(val_labels)/len(val_labels), color='navy', 
                linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)

    # Confusion Matrix (bottom left)
    plt.subplot(2, 2, 3)
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')

    # Training/Validation Loss Curve (bottom right)
    plt.subplot(2, 2, 4)
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Training')
    plt.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='Validation')
    plt.axvline(x=best_epoch, color='g', linestyle='--', label='Best Model')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.suptitle('Model Performance Dashboard', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the super title
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Model performance dashboard saved to {output_file}")

    # Also write the raw data to a CSV file for further analysis
    dashboard_data = {
        'Epoch': range(1, len(train_losses) + 1),
        'Train Loss': train_losses,
        'Validation Loss': val_losses,
        'Validation Score': val_scores,
        'Validation Label': val_labels
    }
    dashboard_df = pd.DataFrame(dashboard_data)
    dashboard_df.to_csv('model_performance_data.csv', index=False)
    logging.info("Model performance data saved to model_performance_data.csv")




