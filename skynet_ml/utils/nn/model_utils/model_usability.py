import plotly.graph_objects as go
import pandas as pd
import joblib
import kaleido


def load_model(filename: str):
    """
    Load a previously saved machine learning model from a file.

    Args:
        filename (str): Path to the file where the model is saved.

    Returns:
        model: The loaded machine learning model.
    """
    return joblib.load(filename)


def save_model(model, filename: str) -> None:
    """
    Save a machine learning model to a file.

    Args:
        model: Machine learning model to be saved.
        filename (str): Path to the file where the model will be saved.
    """
    joblib.dump(model, filename)
    
    
def plot_model(model, save_in=None):
    """
    Generate a text-based representation of the model architecture.

    Args:
        model: Machine learning model whose architecture is to be represented.
        save_in (str, optional): Path to the file where the representation will be saved. If None, prints to stdout.

    Returns:
        None
    """
    i = 1
    output_str = ''

    for layer in model.layers:
        layer_config = layer.get_config()
        
        layer_type = layer_config["name"]
        units = layer_config["n_units"]
        activation = layer_config["activation"]
        initializer = layer_config["initializer"]
        has_bias = layer_config["has_bias"]
        input_dim = layer_config["input_dim"]
        
        item = tuple([layer_type, units, activation, initializer, has_bias, input_dim])
        str1 = "Layer " + str(i) + "\n | Layer Type: {:<10}\n | Units: {:<10}\n | Activation: {:<10}\n | Initializer: {:<10}\n | Has Bias: {:<10}\n | Input Dim: {:<10}\n".format(*item)
        
        output_str += str1
        
        i += 1
        
        if layer == model.layers[-1]:
            break
        
        arrow_str = '\n       |\n       |\n       |\n       V\n\n'
        output_str += arrow_str


    if save_in:
        with open(save_in, 'w') as f:
            f.write(output_str)
        print(output_str)
    else:
        print(output_str)
        
        
def plot_training_history(file_name, save_in=None):
    """
    Visualize the training history of a model, plotting loss values across epochs.

    Args:
        file_name (str): Path to the CSV file containing training history data.
        save_in (str, optional): Path to the file where the plot will be saved in PNG format. If None, the plot will be displayed.

    Returns:
        None
    """
    
    df = pd.read_csv(file_name)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['epoch'], y=df['loss'], mode='lines', name='Training Loss', line=dict(color='DodgerBlue', width=2)))

    if 'val_loss' in df.columns:
        fig.add_trace(go.Scatter(x=df['epoch'], y=df['val_loss'], mode='lines', name='Validation Loss', line=dict(color='red', width=2, dash='dash')))
    
    max_loss = df[['loss', 'val_loss']].max().max()  
    min_loss = df[['loss', 'val_loss']].min().min()  
    third_loss = (max_loss - min_loss) / 3
    two_third_loss = 2 * third_loss + min_loss

    fig.update_layout(title='Loss Across Epochs', xaxis_title='Epoch', yaxis_title='Loss', plot_bgcolor='white', xaxis=dict(showgrid=True, gridcolor='lightgray'), yaxis=dict(showgrid=True, gridcolor='lightgray'))
    fig.add_shape(type="rect", xref="paper", yref="paper", x0=0, x1=1, y0=2/3, y1=1, fillcolor="Red", opacity=0.2, layer="below", line_width=0)
    fig.add_shape(type="rect", xref="paper", yref="paper", x0=0, x1=1, y0=1/3, y1=2/3, fillcolor="DarkOrange", opacity=0.2, layer="below", line_width=0)
    fig.add_shape(type="rect", xref="paper", yref="paper", x0=0, x1=1, y0=0, y1=1/3, fillcolor="LimeGreen", opacity=0.2, layer="below", line_width=0)
    
    if save_in:
        fig.write_image(save_in, format='png')
        fig.show()
    else:
        fig.show()
