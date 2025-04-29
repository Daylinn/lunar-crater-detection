import matplotlib.pyplot as plt
import numpy as np
import os

def generate_speed_comparison(output_dir='results/evaluation'):
    models = ['YOLOv5', 'YOLOv8']
    times = [0.155, 0.090]  # From evaluation results
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, times, color=['#FF6B6B', '#4ECDC4'])
    plt.title('Average Inference Time per Image')
    plt.ylabel('Seconds')
    plt.xlabel('Model')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}s',
                ha='center', va='bottom')
    
    plt.savefig(os.path.join(output_dir, 'speed_comparison.png'))
    plt.close()

def generate_performance_comparison(output_dir='results/evaluation'):
    models = ['YOLOv5', 'YOLOv8']
    metrics_to_compare = {
        'Detections per Image': [0.5, 0.65],
        'Average Confidence': [0.409, 0.341],
        'Average Diameter (px)': [102.11, 345.62]
    }
    
    x = np.arange(len(models))
    width = 0.25
    multiplier = 0
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for attribute, measurement in metrics_to_compare.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3, fmt='%.2f')
        multiplier += 1
    
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width, models)
    ax.legend(loc='upper right')
    ax.set_ylim(0, max([max(v) for v in metrics_to_compare.values()]) * 1.2)
    
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'))
    plt.close()

def main():
    # Create the visualization directory if it doesn't exist
    os.makedirs('results/evaluation', exist_ok=True)
    
    generate_speed_comparison()
    generate_performance_comparison()
    print("Visualizations generated in results/evaluation/")

if __name__ == "__main__":
    main() 