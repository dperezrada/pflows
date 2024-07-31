def run_function_path(dataset, file_path, function_name):
    import importlib.util
    import sys

    try:
        # Load the module from the given file path
        spec = importlib.util.spec_from_file_location("module.name", file_path)
        if spec is None:
            raise ImportError(f"Could not load module specification from {file_path}")
        if spec.loader is None:
            raise ImportError(f"Could not load module loader from {file_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules["module.name"] = module
        spec.loader.exec_module(module)

        # Get the function from the module
        function_to_run = getattr(module, function_name)

        # Execute the function with the provided dataset
        result = function_to_run(dataset)
        return result
    except ImportError as e:
        print(f"Error importing module: {e}")
    except AttributeError:
        print(f"Error: Function '{function_name}' not found in the module.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return None  # Return None if any error occurs


def run_function(dataset, function):
    return function(dataset)
