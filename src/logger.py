# logger_config.py
import logging

logger = logging.getLogger('MLPInit')
logger.setLevel(logging.INFO)
logger.propagate = False  # Add this line to prevent duplicated logging

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create a formatter and add it to the console handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add the console handler to the logger
logger.addHandler(console_handler)