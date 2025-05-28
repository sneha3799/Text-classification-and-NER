from waitress import serve  # Import the Waitress WSGI server for serving the Flask app
from deploy import app  # Import the Flask app instance from the deploy module
import multiprocessing  # For getting the number of CPU cores

if __name__ == "__main__":
    """
    Main entry point for running the server with optimal configurations.
    
    This script configures the number of threads for the server based on the number of CPU cores
    available on the machine and then starts the server using Waitress.
    """
    # Get the number of CPUs available on the machine
    num_cpus = multiprocessing.cpu_count()

    # Set the number of threads per worker to be one less than the number of CPUs (or 1 if only 1 CPU)
    threads_per_worker = max(1, num_cpus - 1)

    # Print the number of threads being used for each worker for informational purposes
    print("Threads:", threads_per_worker)

    # Notify that the server is starting
    print('Server Started')

    # Start the Waitress server with the specified configurations
    serve(
        app,  # The Flask app to be served
        host='0.0.0.0',  # Bind to all available network interfaces
        port=8080,  # Run the server on port 8080
        threads=threads_per_worker  # Number of threads per worker
    )
