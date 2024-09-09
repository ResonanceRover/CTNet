from flask import Flask
from Test_model import Test
import os

# Initialize Flask application
app = Flask(__name__)


# Load configuration
def load_config(config_path):
    config = {}
    try:
        with open(config_path, encoding="utf-8") as f:
            contents = f.read().splitlines()
            for line in contents:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    config[key] = value
        return config
    except FileNotFoundError:
        print(f"Configuration file {config_path} not found.")
        return {}
    except Exception as e:
        print(f"Error reading configuration file: {e}")
        return {}

    # Initialize model


def init_model(data_path):
    try:
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        return Test(data_path)
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return None

    # Load configuration and initialize model


config = load_config("./config/config.ini")
data_path = config.get('data_path')
server_port = config.get('server_port', 5000)  # Default port 5000
pm = init_model(data_path)


# Define route
@app.route('/')
def get_label():
    if pm:
        pm.inference()
        return "The computation has been completed."
    else:
        return "Model initialization failed.", 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=int(server_port), debug=False)