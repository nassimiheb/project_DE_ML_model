 # Using the base image with python 3.10
 FROM python:3.10
 
 # Set our working directory as app
 WORKDIR /app

  # Copy the utils directory and server.py files
 ADD ./utils ./utils
 ADD . .

 # Run pip install  
 RUN pip install -r requirements.txt
 

 
 # Exposing the port 3000 from the container to start the application
 EXPOSE 3000 
 CMD ["gunicorn", "--bind", "0.0.0.0:3000", "server:app"]