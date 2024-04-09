
from src.core.constants import ENV_CONSTANTS

def get_env_variables():
    """
    Retrieves environment variables from a dotenv file.

    Returns:
    - env_variables: A dictionary containing the retrieved environment variables.
    - error: A boolean indicating whether an error occurred during retrieval.
    - error_msg: A string containing an error message if an error occurred, else an empty string.
    """

    
    with open('.env', "r") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                key, value = line.strip().split("=", 1)
                

    # List of environment variable names expected to be present
    env_variable_names = [
        ENV_CONSTANTS.OPEN_AI_API_FIELD,
        ENV_CONSTANTS.MONGO_URI_FIELD,
        ENV_CONSTANTS.DB_RAW_COLLECTION_FIELD,
        ENV_CONSTANTS.DB_CLEAN_COLLECTION_FIELD
    ]

    env_variables = {}

    try:
        # Load environment variables from the .env file
        with open('.env', "r") as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    env_variables[key] = value

        # Check for missing environment variables
        missing_env_variables = [env_variable_name for env_variable_name in env_variable_names if env_variable_name not in env_variables]

        # If any required environment variables are missing, return an error
        if len(missing_env_variables) != 0:
            return {}, True, f'Missing the following environment variables: {",".join(missing_env_variables)}.'

        # Return the retrieved environment variables and indicate no error
        return env_variables, False, ''

    # If an exception occurs during retrieval, return an error
    except Exception as e:
        return {}, True, str(e)
