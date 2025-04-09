import pandas as pd

def calculate_class_wise_accuracy(df):
    """
    Calculate class-wise accuracy from a DataFrame containing 'ids', 'outputs', and 'labels'.
    
    Parameters:
    df (pd.DataFrame): DataFrame with columns 'ids', 'outputs', and 'labels'.
    
    Returns:
    dict: A dictionary with class labels as keys and their corresponding accuracy as values.
    """
    class_wise_accuracy = {}
    
    # Group by labels to calculate accuracy for each class
    for label in df['labels'].unique():
        class_data = df[df['labels'] == label]
        correct_predictions = (class_data['outputs'] == label).sum()
        total_predictions = class_data.shape[0]
        
        if total_predictions > 0:
            accuracy = correct_predictions / total_predictions
        else:
            accuracy = 0.0
        
        class_wise_accuracy[str(label)] = accuracy.item()
    
    return class_wise_accuracy

df = pd.read_csv('/home/jma/Documents/yu/cnn_lstm/validation_outputs_labels_mixup.csv')
results = calculate_class_wise_accuracy(df)
results_df = pd.DataFrame(list(results.items()), columns=['Class', 'Accuracy'])
results_df.to_csv('/home/jma/Documents/yu/cnn_lstm/class_wise_accuracy_mixup.csv', index=False)
