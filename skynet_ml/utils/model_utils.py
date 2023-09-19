import joblib
from skynet_ml.utils._activation_factory import ActivationFactory
from skynet_ml.nn.activation_functions.activation import Activation
import pandas as pd
import plotly.graph_objects as go


def save_model(model, filename: str) -> None:
    """
    Saves the given neural network model to a specified filename using joblib.

    Args:
    - model: The neural network model to save. Expected to be of type Sequential.
    - filename (str): The path and name of the file where the model will be saved.

    Raises:
    - ValueError: If the provided model is not an instance of Sequential.
    """

    # Import Sequential here to avoid the circular import at module level
    from skynet_ml.nn.models.sequential import Sequential
    
    # Check if the passed model is an instance of Sequential
    if not isinstance(model, Sequential):
        raise ValueError("The model must be an instance of Sequential.")
    
    model_data = {
        "config": model.get_config(),  # Assumes a get_config() method in the Sequential class
        "weights": model.get_weights()
    }
    
    # Save model using joblib
    joblib.dump(model_data, filename)
    
    
def load_model(filename: str):
    """
    Loads a neural network model from a specified filename.

    Args:
    - filename (str): The path and name of the file from where the model will be loaded.

    Returns:
    - model: The loaded neural network model, reconstructed from the saved data.
    """

    # Import Sequential here to avoid the circular import at module level
    from skynet_ml.nn.models.sequential import Sequential
    
    # Load model data
    model_data = joblib.load(filename)
    
    # Reconstruct model and set weights
    model = Sequential.from_config(model_data["config"])  # Assumes a from_config() classmethod in Sequential
    model.set_weights(model_data["weights"])
    
    return model


def create_layer_from_config(layer_config):
    """
    Create a layer instance based on its configuration dictionary.

    Args:
    - layer_config (dict): Configuration dictionary of the layer.

    Returns:
    - BaseLayer: An instance of the appropriate layer class based on the configuration.

    Raises:
    - ValueError: If the layer name specified in the configuration is unsupported.
    """
    
    # Handle Dense layer
    if layer_config["name"] == "Dense":
        from skynet_ml.nn.layers.dense import Dense
        
        # Handle the activation
        activation_name_or_instance = layer_config["activation"]
        if isinstance(activation_name_or_instance, Activation):
            # Use the instance directly
            activation = activation_name_or_instance
        else:
            # Assume it's a string and use the ActivationFactory to get the instance
            activation_factory = ActivationFactory()
            activation = activation_factory.get_activation(activation_name_or_instance)

        return Dense(
            n_units=layer_config["units"],
            activation=activation,
            initializer=layer_config["initializer"],
            has_bias=layer_config["has_bias"],
            input_dim=layer_config.get("input_shape") 
        )
    # TODO: Add cases for other layer types as needed.
    
    raise ValueError(f"Unsupported layer name: {layer_config['name']}")


def plot_training_history(file_name, save_in=None):
    """
    Plots the training and validation loss across epochs from a CSV file.
    """
    
    # Load the CSV
    df = pd.read_csv(file_name)
    
    # Create a new figure
    fig = go.Figure()

    # Add the training loss line
    fig.add_trace(go.Scatter(x=df['epoch'], y=df['loss'], mode='lines',
                             name='Training Loss',
                             line=dict(color='DodgerBlue', width=2)))

    # Add the validation loss line if it exists in the CSV
    if 'val_loss' in df.columns:
        fig.add_trace(go.Scatter(x=df['epoch'], y=df['val_loss'], mode='lines',
                                 name='Validation Loss',
                                 line=dict(color='red', width=2, dash='dash')))
    
    # Background gradient colors
    max_loss = df[['loss', 'val_loss']].max().max()  # Consider the max from both losses
    min_loss = df[['loss', 'val_loss']].min().min()  # Consider the min from both losses
    third_loss = (max_loss - min_loss) / 3
    two_third_loss = 2 * third_loss + min_loss

    fig.update_layout(
        title='Loss Across Epochs',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        plot_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridcolor='lightgray')
    )

    # Red background for high loss
    fig.add_shape(
        type="rect",
        xref="paper", yref="paper",
        x0=0, x1=1, y0=2/3, y1=1,  # Adjust to span the top third of the graph
        fillcolor="Red",
        opacity=0.2,
        layer="below",
        line_width=0
    )
    
    # Yellow background for medium loss
    fig.add_shape(
        type="rect",
        xref="paper", yref="paper",
        x0=0, x1=1, y0=1/3, y1=2/3,  # Adjust to span the middle third of the graph
        fillcolor="DarkOrange",
        opacity=0.2,
        layer="below",
        line_width=0
    )
    
    # Green background for low loss
    fig.add_shape(
        type="rect",
        xref="paper", yref="paper",
        x0=0, x1=1, y0=0, y1=1/3,  # Adjust to span the bottom third of the graph
        fillcolor="LimeGreen",
        opacity=0.2,
        layer="below",
        line_width=0
)
    
    # Save the plot if the save_in parameter is provided
    if save_in:
        fig.write_image(save_in, format='png')
        fig.show()
    else:
        fig.show()