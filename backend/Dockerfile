# Use an official Python runtime as a base image
FROM python:3.8

# Set the working directory in the container
WORKDIR /app

# Copy only the necessary files to the container
COPY chatbotapi.py /app
COPY requirements.txt /app
COPY chat_histories /app/chat_histories
COPY functions.json /app
COPY memory.py /app


# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable
ENV PYTHONUNBUFFERED 1

# Command to run the application
CMD ["uvicorn", "chatbotapi:app", "--host", "0.0.0.0", "--port", "8000"]
