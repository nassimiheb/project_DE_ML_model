# Using the base image with python 3.10
FROM python:3.10

RUN mkdir /app
# Set our working directory as app
WORKDIR /app

# Copy the utils directory and server.py files
COPY ./requirements.txt .

# Run pip install  
RUN pip install -r requirements.txt

COPY . .

# Exposing the port 3000 from the container to start the application
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "80"]