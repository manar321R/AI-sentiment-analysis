# Use the official Python image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /code

# Copy the requirements file and install dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of the project files (code, model, interface)
COPY . .

# Run the FastAPI server on port 7860 (the default port for Hugging Face Spaces)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
